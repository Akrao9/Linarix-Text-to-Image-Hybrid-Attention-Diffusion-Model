"""Alternatives to fp32 AdamW: CAME8Bit, Muon, and diffusion-specific CMuon.

Three optimizers with different aims, all reachable from `--optimizer`:

  * `CAME8Bit` cuts optimizer STATE ~7.5x (measured 0.88 GiB vs 6.63 GiB on v8's 890M),
    hoping to be quality-neutral. Learning rate does NOT transfer from AdamW.
  * `MuonWithAuxAdamW` targets QUALITY -- orthogonalised momentum on 2D hidden weights --
    and halves state as a side effect, since Muon carries momentum only. Learning rate
    DOES transfer from AdamW via `adjust_lr_fn="match_rms_adamw"`.
  * `CMuonWithAuxAdamW` keeps Muon's fused momentum but independently orthogonalises
    the semantic rows of QKV, FFN gate+up, and adaLN projections. It uses Microsoft
    Dion's `split_sizes` engine and the CMuon paper's six-step Newton--Schulz recipe.

--- CAME8Bit -------------------------------------------------------------------

Gradient checkpointing took v8's ACTIVATION peak to 11.07 GB, which promotes optimizer
state to a comparable share of the budget: fused AdamW holds `exp_avg` + `exp_avg_sq` in
fp32, i.e. 8 bytes per parameter, ~7 GB at v8's scale, and checkpointing does nothing
about it.

CAME (`Confidence-guided Adaptive Memory Efficient optimization`) attacks that pool two
ways, and the sizes are worth separating because the name emphasises the smaller one:

  * FACTORING (the big win). For 2D parameters the second moment is kept as Adafactor
    row/col vectors instead of a full matrix: `out + in` floats replace `out * in`. On a
    5760x2304 GLUMBConv weight that is 8 KB in place of 53 MB.
  * 8-BIT (the smaller win). The first moment is block-quantised to uint8, 1 byte per
    element instead of 4.

Together those take the dominant 2D pool from ~8 bytes/param to ~1. CAME's own
contribution over Adafactor is the confidence-guided term: it tracks the residual
`(update - exp_avg)^2` in the same factored form and divides the step by it, so
directions where the instantaneous update disagrees with the momentum get damped.

Reference: SANA's `CAME8BitWrapper` (sibling repo `ImageModel/Sana`,
`diffusion/utils/optimizer.py`), which follows arXiv 2307.02047 / 2501.18427. This
implementation is algorithmically faithful to it but differs in two ways that matter at
our scale, both explained where they occur: the quantiser is vectorised rather than a
Python loop over chunks, and the state is stored as flat tensors so that a checkpoint
round-trips through `state_dict()`.

!! LEARNING RATE IS NOT TRANSFERABLE FROM AdamW !!
CAME's update is RMS-clipped to `clip_threshold` and carries no bias correction, so its
natural step size is far smaller. SANA's own configs run CAME8bit at 2e-5 where the
AdamW recipes sit near 1e-4. `--lr` is NOT reused implicitly; see `--came-lr`.
"""

from __future__ import annotations

import functools
import math
from collections import defaultdict

import torch

__all__ = [
    "CAME8Bit",
    "CMuonWithAuxAdamW",
    "MuonWithAuxAdamW",
    "CMUON_EXCLUDE_PREFIXES",
    "MUON_EXCLUDE_PREFIXES",
    "split_cmuon_parameter_groups",
    "split_muon_parameters",
    "came_state_bytes",
    "cmuon_state_bytes",
    "muon_state_bytes",
    "adamw_state_bytes",
]

# A block whose values are all equal has zero range. Dividing by the clamped scale sends
# every code to 0, and dequantisation then returns `bmin` exactly -- constant blocks
# (notably the all-zero state at step 0) round-trip losslessly instead of producing NaN.
_TINY = 1e-12


def _quantize(t: torch.Tensor, block_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Block-wise min/max quantisation to uint8, vectorised over blocks.

    The reference does `for chunk in tensor.split(block_size)` and appends a dict per
    chunk. At v8's scale that is ruinous: a 5760x2304 weight is 6,480 chunks, and the
    optimiser would issue millions of tiny CUDA launches per step across the model. The
    padded `view(-1, block_size)` below computes every block's min, max and codes in a
    handful of kernels instead, which is what makes the optimiser's step cost negligible
    next to the backward pass.
    """
    flat = t.reshape(-1)
    numel = flat.numel()
    pad = (-numel) % block_size
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    blocks = flat.view(-1, block_size)
    bmin = blocks.amin(dim=1, keepdim=True)
    bmax = blocks.amax(dim=1, keepdim=True)
    scale = (bmax - bmin) / 255.0
    codes = ((blocks - bmin) / scale.clamp(min=_TINY)).round_().clamp_(0, 255).to(torch.uint8)
    return codes.view(-1), scale.squeeze(1), bmin.squeeze(1)


def _dequantize(
    codes: torch.Tensor,
    scale: torch.Tensor,
    bmin: torch.Tensor,
    shape: torch.Size,
    block_size: int,
) -> torch.Tensor:
    numel = int(torch.Size(shape).numel())
    blocks = codes.view(-1, block_size).to(scale.dtype)
    out = blocks * scale[:, None] + bmin[:, None]
    return out.view(-1)[:numel].view(shape)


class CAME8Bit(torch.optim.Optimizer):
    """CAME with factored second moments and block-quantised first moments.

    Args:
        params: parameters or param groups.
        lr: learning rate. Read the module docstring -- this is NOT an AdamW lr.
        eps: (second-moment floor, confidence floor).
        clip_threshold: RMS ceiling applied to the update before momentum.
        betas: (momentum, second moment, confidence residual).
        weight_decay: decoupled, applied as ``p -= lr * wd * p``.
        block_size: quantisation block. Larger = fewer scale floats, coarser codes.
        min_8bit_size: parameters with fewer elements than this keep an fp32 first
            moment. Small tensors are a rounding error in the budget and quantising them
            only adds noise.
    """

    def __init__(
        self,
        params,
        lr: float,
        eps: tuple[float, float] = (1e-30, 1e-16),
        clip_threshold: float = 1.0,
        betas: tuple[float, float, float] = (0.9, 0.999, 0.9999),
        weight_decay: float = 0.0,
        block_size: int = 2048,
        min_8bit_size: int = 16384,
    ) -> None:
        if not lr > 0.0:
            raise ValueError(f"lr must be positive, got {lr}")
        if not all(0.0 <= b <= 1.0 for b in betas):
            raise ValueError(f"betas must lie in [0, 1], got {betas}")
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        super().__init__(
            params,
            dict(
                lr=lr,
                eps=eps,
                clip_threshold=clip_threshold,
                betas=betas,
                weight_decay=weight_decay,
                block_size=block_size,
                min_8bit_size=min_8bit_size,
            ),
        )

    # -- layout decisions -------------------------------------------------------
    @staticmethod
    def _factored(shape: torch.Size) -> bool:
        """Can the second moment be kept as row/col vectors?

        2D weights, and 1x1 convolutions (which are 2D matmuls wearing a 4D shape).
        Depthwise/spatial convs and 1D parameters (norm weights, scale_shift_table) keep
        a dense second moment -- they are small enough that it does not matter.
        """
        if len(shape) == 2:
            return True
        return len(shape) == 4 and shape[2] == 1 and shape[3] == 1

    @staticmethod
    def _rms(t: torch.Tensor) -> torch.Tensor:
        return t.norm(2) / (t.numel() ** 0.5)

    @staticmethod
    def _approx_sq_grad(row: torch.Tensor, col: torch.Tensor) -> torch.Tensor:
        r_factor = (row / row.mean(dim=-1, keepdim=True)).rsqrt_().unsqueeze(-1)
        c_factor = col.unsqueeze(-2).rsqrt()
        return torch.mul(r_factor, c_factor)

    # -- first-moment storage ---------------------------------------------------
    def _read_exp_avg(self, state: dict, shape: torch.Size, block_size: int) -> torch.Tensor:
        if "exp_avg" in state:
            return state["exp_avg"]
        return _dequantize(
            state["exp_avg_q"], state["exp_avg_scale"], state["exp_avg_min"], shape, block_size
        )

    def _write_exp_avg(self, state: dict, value: torch.Tensor, block_size: int) -> None:
        if "exp_avg" in state:
            state["exp_avg"] = value
            return
        codes, scale, bmin = _quantize(value, block_size)
        state["exp_avg_q"] = codes
        state["exp_avg_scale"] = scale
        state["exp_avg_min"] = bmin

    def load_state_dict(self, state_dict) -> None:
        """Restore state, keeping the quantised first moment quantised.

        `torch.optim.Optimizer.load_state_dict` casts every non-"step" state tensor to
        the PARAMETER's dtype when the parameter is floating point -- it does not consult
        the value's own dtype. Our parameters are fp32, so the uint8 codes come back as
        fp32. The dequantised values are unchanged (a code of 204.0 scales identically to
        204), which is exactly why this is dangerous: training resumes correctly while
        the first moment silently costs 4x what it should, undoing the feature on the
        first resume. Cast the codes back.
        """
        super().load_state_dict(state_dict)
        for state in self.state.values():
            codes = state.get("exp_avg_q")
            if isinstance(codes, torch.Tensor) and codes.dtype != torch.uint8:
                state["exp_avg_q"] = codes.round().clamp_(0, 255).to(torch.uint8)

    @torch.no_grad()
    def step(self, closure=None):  # noqa: C901 - the algorithm is one linear sequence
        # Lightning's automatic-optimization closure owns forward + backward.  The
        # optimizer update itself must stay under no_grad, but the closure must not:
        # running it under this method's decorator silently disables autograd.
        with torch.enable_grad():
            loss = closure() if closure is not None else None

        for group in self.param_groups:
            beta_m, beta_v, beta_r = group["betas"]
            eps_v, eps_r = group["eps"]
            block_size = group["block_size"]
            lr = group["lr"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("CAME8Bit does not support sparse gradients")
                if grad.dtype in (torch.float16, torch.bfloat16):
                    grad = grad.float()

                shape = grad.shape
                factored = self._factored(shape)
                state = self.state[p]

                if not state:
                    state["step"] = 0
                    # Quantise the first moment only where it pays: a large, factorable
                    # weight. Everything else keeps fp32 -- see min_8bit_size.
                    if factored and grad.numel() >= group["min_8bit_size"]:
                        codes, scale, bmin = _quantize(torch.zeros_like(grad), block_size)
                        state["exp_avg_q"] = codes
                        state["exp_avg_scale"] = scale
                        state["exp_avg_min"] = bmin
                    else:
                        state["exp_avg"] = torch.zeros_like(grad)

                    if factored:
                        rows, cols = shape[0], shape[1]
                        state["exp_avg_sq_row"] = grad.new_zeros(rows)
                        state["exp_avg_sq_col"] = grad.new_zeros(cols)
                        state["exp_avg_res_row"] = grad.new_zeros(rows)
                        state["exp_avg_res_col"] = grad.new_zeros(cols)
                    else:
                        state["exp_avg_sq"] = torch.zeros_like(grad)

                state["step"] += 1
                exp_avg = self._read_exp_avg(state, shape, block_size)

                # --- second moment -> normalised update
                update = grad.square() + eps_v
                if factored:
                    flat_update = update.view(shape[0], shape[1]) if len(shape) == 4 else update
                    state["exp_avg_sq_row"].mul_(beta_v).add_(
                        flat_update.mean(dim=1), alpha=1.0 - beta_v
                    )
                    state["exp_avg_sq_col"].mul_(beta_v).add_(
                        flat_update.mean(dim=0), alpha=1.0 - beta_v
                    )
                    update = self._approx_sq_grad(
                        state["exp_avg_sq_row"], state["exp_avg_sq_col"]
                    )
                    if len(shape) == 4:
                        update = update.view(shape)
                    update = update.mul(grad)
                else:
                    exp_avg_sq = state["exp_avg_sq"]
                    exp_avg_sq.mul_(beta_v).add_(update, alpha=1.0 - beta_v)
                    update = exp_avg_sq.rsqrt().mul_(grad)

                # CAME/Adafactor RMS clip: the update, not the gradient. This is why the
                # learning rate does not transfer from AdamW.
                update.div_((self._rms(update) / group["clip_threshold"]).clamp_(min=1.0))

                # --- momentum
                exp_avg = exp_avg.mul(beta_m).add_(update, alpha=1.0 - beta_m)

                # --- confidence-guided correction: damp directions where the current
                # update disagrees with accumulated momentum.
                res = (update - exp_avg).square_().add_(eps_r)
                if factored:
                    flat_res = res.view(shape[0], shape[1]) if len(shape) == 4 else res
                    state["exp_avg_res_row"].mul_(beta_r).add_(
                        flat_res.mean(dim=1), alpha=1.0 - beta_r
                    )
                    state["exp_avg_res_col"].mul_(beta_r).add_(
                        flat_res.mean(dim=0), alpha=1.0 - beta_r
                    )
                    res_approx = self._approx_sq_grad(
                        state["exp_avg_res_row"], state["exp_avg_res_col"]
                    )
                    if len(shape) == 4:
                        res_approx = res_approx.view(shape)
                    step = res_approx.mul_(exp_avg)
                else:
                    step = exp_avg.clone()

                self._write_exp_avg(state, exp_avg, block_size)

                if group["weight_decay"] != 0.0:
                    p.add_(p, alpha=-group["weight_decay"] * lr)
                p.add_(step, alpha=-lr)

        return loss


def came_state_bytes(model: torch.nn.Module, block_size: int = 2048,
                     min_8bit_size: int = 16384) -> int:
    """Bytes of persistent optimizer state CAME8Bit will hold for `model`.

    Pairs with `adamw_state_bytes` so a launch can print the saving without allocating
    anything. Counts steady-state storage only -- the step transiently materialises one
    dequantised fp32 buffer for the parameter it is updating.
    """
    total = 0
    for p in model.parameters():
        if not p.requires_grad:
            continue
        shape, numel = p.shape, p.numel()
        factored = CAME8Bit._factored(shape)
        if factored and numel >= min_8bit_size:
            blocks = -(-numel // block_size)
            total += numel + 8 * blocks  # uint8 codes + fp32 scale/min per block
        else:
            total += 4 * numel
        if factored:
            total += 4 * 2 * (shape[0] + shape[1])  # sq row/col + res row/col
        else:
            total += 4 * numel
    return total


def adamw_state_bytes(model: torch.nn.Module) -> int:
    """Bytes of persistent optimizer state fp32 AdamW holds: exp_avg + exp_avg_sq."""
    return sum(8 * p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Muon
# ---------------------------------------------------------------------------
# Muon orthogonalises the momentum of 2D weight matrices with a Newton-Schulz
# iteration, so each step moves every singular direction by a comparable amount
# instead of letting a few dominant ones set the scale. It went into production
# during 2025 (Kimi K2 at 1T, GLM-4.5 at 355B, INTELLECT-3 at 106B) and now ships
# as `torch.optim.Muon` in torch >= 2.9, so there is no third-party dependency.
#
# Two facts shape the wrapper below.
#
# 1. `torch.optim.Muon` REJECTS anything that is not 2D -- "Muon only supports 2D
#    parameters". Norm weights, biases and scale_shift_table have to go to a
#    second optimizer. That is not a workaround, it is the documented design:
#    "Other parameters, such as bias, and embedding, should be optimized by a
#    standard method such as AdamW."
# 2. `adjust_lr_fn="match_rms_adamw"` rescales the update so its RMS matches what
#    AdamW would produce (Moonshot's 0.2*lr*sqrt(max(A,B)), from "Muon is
#    Scalable for LLM Training"). With it, the AdamW learning rate transfers --
#    which is why --muon-lr defaults to --lr rather than demanding a new search,
#    the opposite of the CAME8Bit situation above.
#
# On v8's layout 99.3% of parameters live in 2D weights, so Muon covers nearly
# the whole model; the aux group is ~2% and exists mostly for correctness.

# Input embeddings and output heads stay on AdamW. This is the standard Muon
# split: those matrices map between a hidden space and a *semantically fixed*
# space (pixels, latents, the text-encoder readout, the DINO target), so their
# singular directions are not interchangeable the way a hidden weight's are, and
# orthogonalising them throws away scale information that matters. Everything
# else -- attention projections, GLUMBConv, adaLN modulation -- is a hidden
# transform and goes to Muon.
MUON_EXCLUDE_PREFIXES: tuple[str, ...] = (
    "x_embedder",          # latent -> hidden
    "decoder_x_embedder",  # DDT decoder's latent -> hidden
    "caption_embedder",    # Qwen readout -> hidden
    "out_proj",            # hidden -> latent (the velocity head)
    "repa_proj",           # hidden -> DINO space (auxiliary head, dropped at 50k)
    "pos_embed",
)

# CMuon's paper keeps embeddings and the final layer on AdamW.  Boomer's
# final_t_block is part of the velocity head even though it is physically a Linear,
# so CMuon excludes it in addition to the fixed-space projections above.  Keep this
# separate from MUON_EXCLUDE_PREFIXES so selecting --optimizer cmuon cannot silently
# alter the established --optimizer muon experiment.
CMUON_EXCLUDE_PREFIXES: tuple[str, ...] = MUON_EXCLUDE_PREFIXES + ("final_t_block",)

# Fused Linear projections whose output rows have distinct semantic roles.  CMuon
# updates momentum on the original fused matrix, then independently orthogonalises
# these row blocks.  This preserves the fast fused GEMM in the model while avoiding
# the cross-role coupling of vanilla Muon.
_CMUON_FIXED_CHUNKS: dict[str, int] = {
    "qkv": 3,
    "image_qkv": 3,
    "text_qkv": 3,
    "kv_linear": 2,
    "cond_mod": 6,
}
_CMUON_MODULATION_PARENTS: frozenset[str] = frozenset(
    {
        "mod",
        "image_mod",
        "text_mod",
        "shared_mod",
        "shared_image_attn_mod",
        "shared_dual_image_mod",
        "shared_dual_text_mod",
    }
)


def _cmuon_split_sizes(module_name: str, module: torch.nn.Linear) -> tuple[int, ...] | None:
    """Return the paper's semantic row chunks for one fused Boomer Linear."""
    leaf = module_name.rsplit(".", 1)[-1]
    chunks = _CMUON_FIXED_CHUNKS.get(leaf)

    # GLUMBConv's input projection is [gate; value], exactly the fused FFN
    # projection chunked in CMuon.  The trailing ``.0`` is the Linear inside its
    # Sequential(Linear, SiLU).
    if module_name.endswith("inverted_fc.0"):
        chunks = 2

    # adaLN projections occur both per-block and as v8's shared projections.  Their
    # number of semantic outputs is encoded by out_features / hidden_dim (3, 6, or
    # 9), rather than hard-coding one value for every block type.
    parts = module_name.split(".")
    parent = parts[-2] if len(parts) >= 2 else ""
    if leaf == "1" and parent in _CMUON_MODULATION_PARENTS:
        if module.out_features % module.in_features:
            raise ValueError(
                f"CMuon adaLN projection {module_name!r} has shape "
                f"({module.out_features}, {module.in_features}); output rows must be "
                "an integer multiple of the hidden width"
            )
        chunks = module.out_features // module.in_features

    if chunks is None or chunks <= 1:
        return None
    if module.out_features % chunks:
        raise ValueError(
            f"CMuon projection {module_name!r} has {module.out_features} output rows, "
            f"which cannot be divided into {chunks} semantic chunks"
        )
    rows = module.out_features // chunks
    return (rows,) * chunks


def split_cmuon_parameter_groups(
    model: torch.nn.Module,
    exclude_prefixes: tuple[str, ...] = CMUON_EXCLUDE_PREFIXES,
) -> tuple[list[dict], list[torch.nn.Parameter]]:
    """Partition a model into semantic CMuon groups and an AdamW auxiliary set.

    Groups are keyed by ``split_sizes`` so Dion can megabatch parameters sharing the
    same semantic layout.  A group without ``split_sizes`` is ordinary Muon.  Only
    actual ``nn.Linear.weight`` objects are eligible; learned 2D tables remain AdamW.
    """
    while hasattr(model, "_orig_mod"):
        model = model._orig_mod

    by_split: dict[tuple[int, ...] | None, list[torch.nn.Parameter]] = defaultdict(list)
    eligible_ids: set[int] = set()
    for module_name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear) or not module.weight.requires_grad:
            continue
        excluded = any(
            module_name == prefix or module_name.startswith(f"{prefix}.")
            for prefix in exclude_prefixes
        )
        if excluded:
            continue
        split_sizes = _cmuon_split_sizes(module_name, module)
        by_split[split_sizes].append(module.weight)
        eligible_ids.add(id(module.weight))

    aux = [
        param
        for param in model.parameters()
        if param.requires_grad and id(param) not in eligible_ids
    ]
    groups = [
        {
            "params": params,
            "split_sizes": split_sizes,
            "boomer_cmuon_chunks": len(split_sizes) if split_sizes else 1,
        }
        for split_sizes, params in by_split.items()
    ]
    return groups, aux


def split_muon_parameters(
    model: torch.nn.Module, exclude_prefixes: tuple[str, ...] = MUON_EXCLUDE_PREFIXES
) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    """Partition into (Muon-eligible hidden Linear weights, everything else).

    ``torch.compile`` wraps the real model in an ``OptimizedModule`` whose parameter
    names are prefixed with ``_orig_mod.``.  Unwrap it before matching exclusions so a
    compiled launch cannot silently send the latent/text embedders and output heads to
    Muon.

    Eligibility is intentionally based on module identity rather than merely
    ``param.ndim == 2``.  Boomer has several learned two-dimensional *tables* (notably
    adaLN ``scale_shift_table`` parameters); orthogonalising those as if they were
    hidden-to-hidden matrices changes their meaning.  Muon receives only
    ``nn.Linear.weight`` parameters outside the fixed-space embedder/head prefixes.
    """
    while hasattr(model, "_orig_mod"):
        model = model._orig_mod

    eligible_ids: set[int] = set()
    for module_name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear) or not module.weight.requires_grad:
            continue
        excluded = any(
            module_name == prefix or module_name.startswith(f"{prefix}.")
            for prefix in exclude_prefixes
        )
        if not excluded:
            eligible_ids.add(id(module.weight))

    muon: list[torch.nn.Parameter] = []
    aux: list[torch.nn.Parameter] = []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        if id(param) in eligible_ids:
            muon.append(param)
        else:
            aux.append(param)
    return muon, aux


class MuonWithAuxAdamW(torch.optim.Optimizer):
    """Muon on 2D hidden weights, AdamW on everything else, as one optimizer.

    This MUST subclass `torch.optim.Optimizer`, not merely satisfy Lightning's
    structural `Optimizable` protocol: `torch.optim.lr_scheduler.LambdaLR` does a hard
    `isinstance(optimizer, Optimizer)` and raises `TypeError` otherwise, so a duck-typed
    wrapper passes Lightning and then dies when the scheduler is attached.

    The two sub-optimizers ADOPT this optimizer's own param-group dicts rather than
    building their own: `Optimizer.add_param_group` appends the dict it is handed
    (verified in tests), so `self.param_groups[0] is self._muon.param_groups[0]`. That
    identity is what makes an LR scheduler work -- it mutates `group["lr"]` on our
    groups, and the sub-optimizer that will actually read it sees the same dict. Copying
    the groups instead would leave the scheduler adjusting a rate nobody reads.

    Each group carries a `boomer_role` of "muon" or "aux" so a resume can restore the
    two different learning rates to the right groups.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        lr: float,
        aux_lr: float,
        weight_decay: float = 0.1,
        aux_weight_decay: float = 0.0,
        aux_betas: tuple[float, float] = (0.9, 0.999),
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        adjust_lr_fn: str = "match_rms_adamw",
        exclude_prefixes: tuple[str, ...] = MUON_EXCLUDE_PREFIXES,
    ) -> None:
        try:
            from torch.optim import Muon
        except ImportError as exc:  # pragma: no cover - depends on torch version
            raise RuntimeError(
                "torch.optim.Muon requires torch >= 2.9; this build has "
                f"{torch.__version__}. Upgrade torch or use --optimizer adamw."
            ) from exc

        muon_params, aux_params = split_muon_parameters(model, exclude_prefixes)
        if not muon_params:
            raise ValueError(
                "no Muon-eligible parameters found (2D weights outside "
                f"{exclude_prefixes}); refusing to run Muon over nothing"
            )

        groups: list[dict] = [
            {"params": muon_params, "lr": lr, "boomer_role": "muon"}
        ]
        if aux_params:
            groups.append({"params": aux_params, "lr": aux_lr, "boomer_role": "aux"})
        super().__init__(groups, dict(lr=lr))

        # Hand the sub-optimizers OUR group dicts (see class docstring): they append the
        # same objects, so scheduler mutations on self.param_groups land where they are
        # read. Each sub-optimizer setdefaults its own hyperparameters into the group.
        self._muon = Muon(
            [self.param_groups[0]],
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adjust_lr_fn=adjust_lr_fn,
        )
        # AdamW here is deliberately plain: the aux group is norms, biases and the
        # embedding/head matrices, which is where AdamW is known-good and where Muon's
        # own authors keep it.
        self._aux = (
            torch.optim.AdamW(
                [self.param_groups[1]],
                lr=aux_lr,
                betas=aux_betas,
                weight_decay=aux_weight_decay,
            )
            if aux_params
            else None
        )

        self.defaults = dict(
            lr=lr, aux_lr=aux_lr, weight_decay=weight_decay, momentum=momentum,
            ns_steps=ns_steps, adjust_lr_fn=adjust_lr_fn,
        )
        self.muon_params = sum(p.numel() for p in muon_params)
        self.aux_params = sum(p.numel() for p in aux_params)

    @torch.no_grad()
    def step(self, closure=None):
        # Optimizer closures (including Lightning's automatic-optimization closure)
        # perform forward and backward, so explicitly re-enable autograd inside this
        # otherwise no-grad update method.
        with torch.enable_grad():
            loss = closure() if closure is not None else None
        self._muon.step()
        if self._aux is not None:
            self._aux.step()
        return loss

    def state_dict(self) -> dict:
        return {
            "muon": self._muon.state_dict(),
            "aux": self._aux.state_dict() if self._aux is not None else None,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self._muon.load_state_dict(state_dict["muon"])
        if self._aux is not None and state_dict.get("aux") is not None:
            self._aux.load_state_dict(state_dict["aux"])
        # Optimizer.load_state_dict replaces each sub-optimizer's param-group dicts.
        # Rebind the wrapper to those NEW live objects; otherwise LambdaLR and
        # BoomerLit.on_train_start mutate the stale pre-load groups while Muon/AdamW
        # continue forever at the checkpoint's restored rates.
        self.param_groups = list(self._muon.param_groups)
        if self._aux is not None:
            self.param_groups.extend(self._aux.param_groups)

    def __repr__(self) -> str:  # pragma: no cover - diagnostics
        return (
            f"MuonWithAuxAdamW(muon={self.muon_params / 1e6:.1f}M params, "
            f"aux={self.aux_params / 1e6:.1f}M params)"
        )


@torch.compile(fullgraph=True)
def _cmuon_newton_schulz(
    grad: torch.Tensor,
    epsilon: float | torch.Tensor = 1e-7,
    *,
    steps: int = 6,
) -> torch.Tensor:
    """CMuon paper Algorithm 3, supporting Dion's batched 2D matrices.

    Dion's default Polar Express kernel is a good general Muon implementation, but
    the CMuon DC-AE/DiT-XL result used six iterations of the fixed quintic map below.
    Supplying it as Dion's custom NS function makes the training recipe explicit and
    avoids silently benchmarking a different polar approximation.
    """
    if grad.ndim not in (2, 3):
        raise ValueError(f"CMuon Newton-Schulz expects a 2D/3D tensor, got {grad.ndim}D")
    if steps <= 0:
        raise ValueError(f"CMuon Newton-Schulz steps must be positive, got {steps}")

    x = grad.to(torch.bfloat16)
    transposed = x.shape[-2] > x.shape[-1]
    if transposed:
        x = x.mT
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + epsilon)
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        gram = x @ x.mT
        x = a * x + (b * gram + c * (gram @ gram)) @ x
    return x.mT if transposed else x


class CMuonWithAuxAdamW(torch.optim.Optimizer):
    """Paper-faithful chunked Muon using Dion, plus AdamW for scalar/fixed heads.

    Momentum is accumulated on each original fused parameter by Dion *before* the
    update is split.  QKV, GLUMBConv gate/value, and adaLN modulation rows are then
    independently orthogonalised and concatenated.  The optional sqrt(N) multiplier
    is represented as a per-group LR scale; weight decay is divided by the same scale
    so the resulting decoupled decay coefficient remains ``base_lr * weight_decay``.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        lr: float,
        aux_lr: float,
        weight_decay: float = 0.0,
        aux_weight_decay: float = 0.0,
        aux_betas: tuple[float, float] = (0.9, 0.95),
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 6,
        chunk_rescale: bool = True,
        exclude_prefixes: tuple[str, ...] = CMUON_EXCLUDE_PREFIXES,
    ) -> None:
        try:
            from dion import Muon as DionMuon
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "--optimizer cmuon requires Microsoft Dion with split_sizes support. "
                "Install the pinned dependency with `pip install -e '.[cmuon]'`, or "
                "for a sibling clone run `pip install -e ../dion`. Dion2 is not a "
                "substitute: split_sizes is currently implemented by dion.Muon."
            ) from exc

        specs, aux_params = split_cmuon_parameter_groups(model, exclude_prefixes)
        if not specs:
            raise ValueError(
                "no CMuon-eligible hidden Linear weights found; refusing to run CMuon "
                "over only an AdamW auxiliary group"
            )

        groups: list[dict] = []
        for spec in specs:
            chunks = int(spec["boomer_cmuon_chunks"])
            lr_scale = math.sqrt(chunks) if chunk_rescale and chunks > 1 else 1.0
            group = {
                **spec,
                "lr": lr * lr_scale,
                "weight_decay": weight_decay / lr_scale,
                "boomer_role": "muon",
                "boomer_lr_scale": lr_scale,
            }
            # None is accepted by our splitter but omitting it keeps Dion's ordinary
            # Muon group unambiguous in checkpoint diagnostics.
            if group["split_sizes"] is None:
                group.pop("split_sizes")
            groups.append(group)

        muon_group_count = len(groups)
        if aux_params:
            groups.append({"params": aux_params, "lr": aux_lr, "boomer_role": "aux"})
        super().__init__(groups, dict(lr=lr))

        ns_func = functools.partial(_cmuon_newton_schulz, steps=ns_steps)
        self._muon = DionMuon(
            self.param_groups[:muon_group_count],
            lr=lr,
            mu=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
            adjust_lr="rms_norm",
            use_polar_express=False,
            newton_schulz_func=ns_func,
        )
        self._aux = (
            torch.optim.AdamW(
                [self.param_groups[-1]],
                lr=aux_lr,
                betas=aux_betas,
                weight_decay=aux_weight_decay,
            )
            if aux_params
            else None
        )

        self.defaults = dict(
            lr=lr,
            aux_lr=aux_lr,
            weight_decay=weight_decay,
            momentum=momentum,
            ns_steps=ns_steps,
            chunk_rescale=chunk_rescale,
            adjust_lr_fn="match_rms_adamw",
        )
        self.muon_params = sum(p.numel() for spec in specs for p in spec["params"])
        self.aux_params = sum(p.numel() for p in aux_params)
        self.chunked_params = sum(
            p.numel()
            for spec in specs
            if int(spec["boomer_cmuon_chunks"]) > 1
            for p in spec["params"]
        )
        self.chunked_matrices = sum(
            len(spec["params"])
            for spec in specs
            if int(spec["boomer_cmuon_chunks"]) > 1
        )

    @torch.no_grad()
    def step(self, closure=None):
        with torch.enable_grad():
            loss = closure() if closure is not None else None
        self._muon.step()
        if self._aux is not None:
            self._aux.step()
        return loss

    def state_dict(self) -> dict:
        return {
            "muon": self._muon.state_dict(),
            "aux": self._aux.state_dict() if self._aux is not None else None,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self._muon.load_state_dict(state_dict["muon"])
        if self._aux is not None and state_dict.get("aux") is not None:
            self._aux.load_state_dict(state_dict["aux"])
        self.param_groups = list(self._muon.param_groups)
        if self._aux is not None:
            self.param_groups.extend(self._aux.param_groups)

    def __repr__(self) -> str:  # pragma: no cover - diagnostics
        return (
            f"CMuonWithAuxAdamW(cmuon={self.muon_params / 1e6:.1f}M params, "
            f"chunked={self.chunked_params / 1e6:.1f}M/{self.chunked_matrices} matrices, "
            f"aux={self.aux_params / 1e6:.1f}M params)"
        )


def muon_state_bytes(model: torch.nn.Module,
                     exclude_prefixes: tuple[str, ...] = MUON_EXCLUDE_PREFIXES) -> int:
    """Persistent optimizer state for the Muon/AdamW split.

    Muon keeps ONE momentum buffer per 2D weight (4 bytes/param) against AdamW's two
    (8), so the hybrid roughly halves optimizer state on top of whatever it does for
    quality. Compare with `adamw_state_bytes` / `came_state_bytes`.
    """
    muon_params, aux_params = split_muon_parameters(model, exclude_prefixes)
    return 4 * sum(p.numel() for p in muon_params) + 8 * sum(p.numel() for p in aux_params)


def cmuon_state_bytes(
    model: torch.nn.Module,
    exclude_prefixes: tuple[str, ...] = CMUON_EXCLUDE_PREFIXES,
) -> int:
    """Persistent state for the CMuon/Dion momentum + auxiliary AdamW split."""
    groups, aux_params = split_cmuon_parameter_groups(model, exclude_prefixes)
    cmuon_params = [param for group in groups for param in group["params"]]
    return 4 * sum(p.numel() for p in cmuon_params) + 8 * sum(p.numel() for p in aux_params)
