"""CPU regression coverage for boomer.optim: CAME8Bit and MuonWithAuxAdamW.

An alternative optimizer is only worth running if it round-trips through a checkpoint,
really is smaller than AdamW, and survives the harness it will run under. For CAME that
means the quantiser must not eat the signal; for Muon it means the hybrid must satisfy
BOTH Lightning's structural protocol and LambdaLR's isinstance check, and its two
learning rates must survive a resume. Everything here is CPU-only and fast.
"""

from __future__ import annotations

import copy
import importlib.util
import math

import pytest
import torch
import torch.nn as nn

from boomer.optim import (
    CAME8Bit,
    _dequantize,
    _quantize,
    adamw_state_bytes,
    came_state_bytes,
)


# --- quantiser -----------------------------------------------------------------


@pytest.mark.parametrize("numel", [2048, 4096, 5000, 1, 17])
def test_quantize_roundtrip_shapes(numel):
    """Round-trip preserves shape for both exact and ragged block counts."""
    t = torch.randn(numel)
    codes, scale, bmin = _quantize(t, 2048)
    out = _dequantize(codes, scale, bmin, t.shape, 2048)
    assert out.shape == t.shape


def test_quantize_constant_block_is_exact_not_nan():
    """A zero-range block must not divide by zero.

    This is the state at step 0: exp_avg is all zeros, so every block has
    max == min. Dividing by the raw scale gives 0/0 = NaN, which would poison the
    first momentum update. The clamped divisor sends every code to 0 and
    dequantisation returns bmin exactly.
    """
    for value in (0.0, -3.5, 7.25):
        t = torch.full((4096,), value)
        codes, scale, bmin = _quantize(t, 2048)
        out = _dequantize(codes, scale, bmin, t.shape, 2048)
        assert torch.isfinite(out).all()
        assert torch.equal(out, t)


def test_quantize_error_within_block_resolution():
    """Error is bounded by half a quantisation step of the block's own range."""
    torch.manual_seed(0)
    t = torch.randn(8192)
    codes, scale, bmin = _quantize(t, 2048)
    out = _dequantize(codes, scale, bmin, t.shape, 2048)
    per_block_range = t.view(-1, 2048).amax(dim=1) - t.view(-1, 2048).amin(dim=1)
    tol = (per_block_range / 255.0).max().item()
    assert (out - t).abs().max().item() <= tol


def test_quantize_is_vectorized_not_chunk_list():
    """State must be flat tensors, not a Python list of per-chunk dicts.

    The reference implementation returns one dict per 2048-element chunk, which at
    v8's scale is hundreds of thousands of dicts and millions of tiny kernel launches
    per step. It also does not survive `state_dict()`.
    """
    codes, scale, bmin = _quantize(torch.randn(8192), 2048)
    assert isinstance(codes, torch.Tensor) and codes.dtype == torch.uint8
    assert isinstance(scale, torch.Tensor) and scale.numel() == 4
    assert isinstance(bmin, torch.Tensor) and bmin.numel() == 4


# --- layout decisions ----------------------------------------------------------


def test_factored_layout_selection():
    assert CAME8Bit._factored(torch.Size([256, 128])) is True
    assert CAME8Bit._factored(torch.Size([64, 32, 1, 1])) is True
    # spatial convs and 1D params keep a dense second moment
    assert CAME8Bit._factored(torch.Size([64, 32, 3, 3])) is False
    assert CAME8Bit._factored(torch.Size([512])) is False


def test_large_2d_gets_quantized_first_moment_small_does_not():
    big = nn.Linear(256, 256, bias=False)     # 65536 elements >= min_8bit_size
    small = nn.Linear(16, 16, bias=False)     # 256 elements
    opt = CAME8Bit(list(big.parameters()) + list(small.parameters()), lr=1e-3)
    for m in (big, small):
        m.weight.grad = torch.randn_like(m.weight)
    opt.step()

    assert "exp_avg_q" in opt.state[big.weight]
    assert "exp_avg" not in opt.state[big.weight]
    assert opt.state[big.weight]["exp_avg_q"].dtype == torch.uint8
    # factored: no dense second moment anywhere
    assert "exp_avg_sq" not in opt.state[big.weight]
    assert opt.state[big.weight]["exp_avg_sq_row"].numel() == 256

    assert "exp_avg" in opt.state[small.weight]
    assert opt.state[small.weight]["exp_avg"].dtype == torch.float32


def test_1d_param_keeps_dense_second_moment():
    norm = nn.LayerNorm(64)
    opt = CAME8Bit(norm.parameters(), lr=1e-3)
    norm.weight.grad = torch.randn_like(norm.weight)
    norm.bias.grad = torch.randn_like(norm.bias)
    opt.step()
    assert opt.state[norm.weight]["exp_avg_sq"].shape == norm.weight.shape
    assert "exp_avg_sq_row" not in opt.state[norm.weight]


# --- the actual claim: less memory than AdamW ----------------------------------


def test_state_is_much_smaller_than_adamw():
    """The whole point. Factoring dominates; 8-bit adds to it."""
    model = nn.Sequential(nn.Linear(1024, 1024), nn.Linear(1024, 1024))
    came = came_state_bytes(model)
    adamw = adamw_state_bytes(model)
    assert came < adamw / 5, f"expected >5x saving, got {adamw / came:.1f}x"


def test_state_accounting_matches_reality():
    """The predicted byte count must match what stepping actually allocates."""
    model = nn.Sequential(nn.Linear(512, 512, bias=False), nn.LayerNorm(512))
    opt = CAME8Bit(model.parameters(), lr=1e-3)
    for p in model.parameters():
        p.grad = torch.randn_like(p)
    opt.step()

    actual = 0
    for state in opt.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                actual += value.numel() * value.element_size()
    assert actual == came_state_bytes(model)


# --- optimization behaviour ----------------------------------------------------


def test_converges_on_a_quadratic():
    torch.manual_seed(0)
    layer = nn.Linear(256, 256, bias=False)
    target = torch.randn(32, 256)
    inputs = torch.randn(32, 256)
    opt = CAME8Bit(layer.parameters(), lr=1e-2)

    losses = []
    for _ in range(60):
        opt.zero_grad()
        loss = ((layer(inputs) - target) ** 2).mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] * 0.5, f"no progress: {losses[0]:.4f} -> {losses[-1]:.4f}"
    assert all(torch.isfinite(p).all() for p in layer.parameters())


def test_step_is_finite_with_zero_and_tiny_gradients():
    """Degenerate gradients must not produce NaN through the rsqrt/factor path."""
    layer = nn.Linear(256, 256, bias=False)
    opt = CAME8Bit(layer.parameters(), lr=1e-3)
    for grad in (torch.zeros_like(layer.weight), torch.full_like(layer.weight, 1e-30)):
        layer.weight.grad = grad.clone()
        opt.step()
        assert torch.isfinite(layer.weight).all()


def test_came_closure_runs_with_autograd_enabled():
    """Lightning-style closures include backward and must escape step's no-grad scope."""
    layer = nn.Linear(16, 16)
    opt = CAME8Bit(layer.parameters(), lr=1e-3)
    inputs, target = torch.randn(8, 16), torch.randn(8, 16)
    before = layer.weight.detach().clone()

    def closure():
        opt.zero_grad()
        loss = ((layer(inputs) - target) ** 2).mean()
        loss.backward()
        return loss

    loss = opt.step(closure)
    assert loss.requires_grad
    assert not torch.equal(before, layer.weight)


def test_weight_decay_shrinks_a_zero_gradient_parameter():
    layer = nn.Linear(256, 256, bias=False)
    with torch.no_grad():
        layer.weight.fill_(1.0)
    opt = CAME8Bit(layer.parameters(), lr=0.1, weight_decay=0.1)
    layer.weight.grad = torch.zeros_like(layer.weight)
    opt.step()
    assert layer.weight.max().item() < 1.0


def test_rejects_adamw_style_misuse():
    layer = nn.Linear(8, 8)
    with pytest.raises(ValueError):
        CAME8Bit(layer.parameters(), lr=0.0)
    with pytest.raises(ValueError):
        CAME8Bit(layer.parameters(), lr=1e-3, betas=(0.9, 1.5, 0.99))


def test_sparse_gradients_rejected():
    p = nn.Parameter(torch.randn(16, 16))
    opt = CAME8Bit([p], lr=1e-3)
    p.grad = torch.sparse_coo_tensor(torch.tensor([[0], [0]]), torch.tensor([1.0]), (16, 16))
    with pytest.raises(RuntimeError, match="sparse"):
        opt.step()


def test_bf16_gradients_are_promoted():
    """Statistics stay fp32 even if a gradient arrives in bf16.

    v8 trains bf16-mixed with fp32 parameters, so grads are normally fp32 already; this
    guards the branch that would otherwise accumulate second moments in bf16.
    """
    layer = nn.Linear(256, 256, bias=False).to(torch.bfloat16)
    opt = CAME8Bit(layer.parameters(), lr=1e-3)
    layer.weight.grad = torch.randn_like(layer.weight)
    assert layer.weight.grad.dtype == torch.bfloat16
    opt.step()
    assert torch.isfinite(layer.weight).all()
    assert opt.state[layer.weight]["exp_avg_sq_row"].dtype == torch.float32


# --- checkpoint resume ---------------------------------------------------------


def test_state_dict_resume_is_bit_identical():
    """A resumed optimizer must continue on exactly the same trajectory.

    Quantised state is what makes this non-trivial: uint8 codes plus per-block scales
    have to survive `state_dict()` / `load_state_dict()` without being cast to the
    parameter dtype or flattened into the wrong shape.
    """
    torch.manual_seed(0)
    inputs = torch.randn(16, 256)
    target = torch.randn(16, 256)

    def run(resume_at: int | None):
        torch.manual_seed(1)
        layer = nn.Linear(256, 256, bias=False)
        opt = CAME8Bit(layer.parameters(), lr=1e-2)
        for step in range(10):
            if resume_at is not None and step == resume_at:
                snapshot = copy.deepcopy(opt.state_dict())
                weights = layer.weight.detach().clone()
                layer = nn.Linear(256, 256, bias=False)
                with torch.no_grad():
                    layer.weight.copy_(weights)
                opt = CAME8Bit(layer.parameters(), lr=1e-2)
                opt.load_state_dict(snapshot)
            opt.zero_grad()
            ((layer(inputs) - target) ** 2).mean().backward()
            opt.step()
        return layer.weight.detach().clone()

    assert torch.equal(run(None), run(5))


def test_resume_preserves_the_memory_saving():
    """Resuming must not silently inflate the state back to fp32.

    Numerics alone would not catch this: fp32-cast codes dequantise to the same
    values, so training would continue correctly while the feature quietly stopped
    saving anything.
    """
    layer = nn.Linear(256, 256, bias=False)
    opt = CAME8Bit(layer.parameters(), lr=1e-3)
    layer.weight.grad = torch.randn_like(layer.weight)
    opt.step()

    def state_bytes(o):
        return sum(
            v.numel() * v.element_size()
            for s in o.state.values()
            for v in s.values()
            if isinstance(v, torch.Tensor)
        )

    before = state_bytes(opt)
    restored = CAME8Bit(nn.Linear(256, 256, bias=False).parameters(), lr=1e-3)
    restored.load_state_dict(copy.deepcopy(opt.state_dict()))
    assert state_bytes(restored) == before


def test_state_dict_preserves_uint8_codes():
    layer = nn.Linear(256, 256, bias=False)
    opt = CAME8Bit(layer.parameters(), lr=1e-3)
    layer.weight.grad = torch.randn_like(layer.weight)
    opt.step()

    restored = CAME8Bit(nn.Linear(256, 256, bias=False).parameters(), lr=1e-3)
    restored.load_state_dict(copy.deepcopy(opt.state_dict()))
    state = next(iter(restored.state.values()))
    assert state["exp_avg_q"].dtype == torch.uint8, (
        "load_state_dict cast the quantised codes -- resume would corrupt momentum"
    )


# --- trainer wiring ------------------------------------------------------------


def _hparams(**overrides):
    from argparse import Namespace

    base = dict(
        optimizer="came8bit", lr=1e-4, came_lr=2e-5, weight_decay=0.0,
        beta1=0.9, beta2=0.95, fused_adamw=True,
        came_block_size=2048, came_min_8bit_size=16384,
        muon_lr=None, muon_aux_lr=None, muon_weight_decay=0.1,
        muon_momentum=0.95, muon_ns_steps=5, muon_adjust_lr_fn="match_rms_adamw",
        cmuon_lr=None, cmuon_aux_lr=None, cmuon_weight_decay=0.0,
        cmuon_momentum=0.95, cmuon_ns_steps=6, cmuon_chunk_rescale=True,
        warmup_steps=0, lr_scheduler="constant", min_lr_ratio=0.1, max_steps=100,
    )
    base.update(overrides)
    return Namespace(**base)


class _TinyLit:
    """Enough of BoomerLit for configure_optimizers, without building a DiT."""

    def __init__(self, hparams):
        import torch.nn as _nn

        self.hparams = hparams
        self.model = _nn.Sequential(_nn.Linear(64, 64), _nn.LayerNorm(64))

    def _cli_lr_for_group(self, group):
        # on_train_start calls this on self; borrow the real implementation.
        import train

        return train.BoomerLit._cli_lr_for_group(self, group)


def test_configure_optimizers_builds_came8bit():
    import train

    lit = _TinyLit(_hparams())
    out = train.BoomerLit.configure_optimizers(lit)
    optim = out["optimizer"]
    assert isinstance(optim, CAME8Bit)
    assert optim.param_groups[0]["lr"] == 2e-5, "must use --came-lr, not --lr"


def test_configure_optimizers_defaults_to_adamw():
    import train

    lit = _TinyLit(_hparams(optimizer="adamw"))
    optim = train.BoomerLit.configure_optimizers(lit)["optimizer"]
    assert isinstance(optim, torch.optim.AdamW)
    assert optim.param_groups[0]["lr"] == 1e-4


def test_came8bit_without_came_lr_is_rejected():
    """Inheriting the AdamW LR would be ~5x too large and would look like it trained."""
    import train

    lit = _TinyLit(_hparams(came_lr=None))
    with pytest.raises(ValueError, match="came-lr"):
        train.BoomerLit.configure_optimizers(lit)


def test_on_train_start_does_not_push_adamw_lr_into_came_groups():
    """The --lr resume override must target the ACTIVE optimizer's rate.

    on_train_start rewrites param_groups["initial_lr"] and the scheduler base_lrs from
    the CLI so a changed LR wins over restored state. Reading hparams.lr unconditionally
    would reset CAME to the AdamW rate on every resume.
    """
    import train
    from types import SimpleNamespace

    lit = _TinyLit(_hparams())
    optim = train.BoomerLit.configure_optimizers(lit)["optimizer"]
    for group in optim.param_groups:
        group["initial_lr"] = 999.0
    # mirrors LambdaLR, which carries .optimizer and a base_lrs aligned to its groups
    sched = SimpleNamespace(base_lrs=[999.0] * len(optim.param_groups), optimizer=optim)
    lit.trainer = SimpleNamespace(
        optimizers=[optim],
        lr_scheduler_configs=[SimpleNamespace(scheduler=sched)],
    )
    train.BoomerLit.on_train_start(lit)

    assert optim.param_groups[0]["initial_lr"] == 2e-5
    assert sched.base_lrs == [2e-5]


# --- Muon ----------------------------------------------------------------------


def _muon_available():
    return hasattr(torch.optim, "Muon")


muon_required = pytest.mark.skipif(not _muon_available(), reason="torch.optim.Muon needs torch>=2.9")


def _split_model():
    return nn.Sequential(
        nn.Linear(64, 64),          # 2D weight + 1D bias
        nn.LayerNorm(64),           # 1D weight + 1D bias
        nn.Conv2d(4, 4, 3),         # 4D
    )


def test_split_sends_only_2d_hidden_weights_to_muon():
    from boomer.optim import split_muon_parameters

    model = _split_model()
    muon, aux = split_muon_parameters(model)
    assert all(p.ndim == 2 for p in muon), "Muon rejects anything but 2D"
    assert {p.ndim for p in aux} <= {1, 3, 4}
    assert len(muon) == 1  # only the Linear weight


def test_split_excludes_embeddings_and_heads():
    """Input embeddings and output heads must stay on AdamW.

    They map hidden space to a semantically fixed space (latents, the text readout,
    the DINO target), where orthogonalising away scale is not the right prior.
    """
    from boomer.optim import MUON_EXCLUDE_PREFIXES, split_muon_parameters

    class Named(nn.Module):
        def __init__(self):
            super().__init__()
            self.x_embedder = nn.Linear(8, 16, bias=False)
            self.out_proj = nn.Linear(16, 8, bias=False)
            self.repa_proj = nn.Linear(16, 16, bias=False)
            self.blocks = nn.Linear(16, 16, bias=False)  # a hidden weight

    muon, aux = split_muon_parameters(Named())
    assert len(muon) == 1 and muon[0].shape == (16, 16)
    assert len(aux) == 3
    for prefix in ("x_embedder", "out_proj", "repa_proj"):
        assert prefix in MUON_EXCLUDE_PREFIXES


def test_split_is_identical_after_torch_compile():
    """OptimizedModule prefixes names with ``_orig_mod.``; exclusions must survive."""
    from boomer.optim import split_muon_parameters

    class Named(nn.Module):
        def __init__(self):
            super().__init__()
            self.x_embedder = nn.Linear(8, 16, bias=False)
            self.blocks = nn.Linear(16, 16, bias=False)
            self.out_proj = nn.Linear(16, 8, bias=False)

        def forward(self, x):
            return self.out_proj(self.blocks(self.x_embedder(x)))

    eager = Named()
    compiled = torch.compile(eager)
    eager_muon, eager_aux = split_muon_parameters(eager)
    compiled_muon, compiled_aux = split_muon_parameters(compiled)
    assert {id(p) for p in compiled_muon} == {id(p) for p in eager_muon}
    assert {id(p) for p in compiled_aux} == {id(p) for p in eager_aux}
    assert eager_muon == [eager.blocks.weight]
    assert set(eager_aux) == {eager.x_embedder.weight, eager.out_proj.weight}


def test_split_keeps_two_dimensional_parameter_tables_on_adamw():
    """Shape alone is insufficient: adaLN tables are not hidden weight matrices."""
    from boomer.optim import split_muon_parameters

    class WithTable(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale_shift_table = nn.Parameter(torch.zeros(6, 32))
            self.proj = nn.Linear(32, 32, bias=False)

    model = WithTable()
    muon, aux = split_muon_parameters(model)
    assert muon == [model.proj.weight]
    assert model.scale_shift_table in aux


@muon_required
def test_torch_muon_rejects_non_2d_which_is_why_the_split_exists():
    with pytest.raises(ValueError, match="2D"):
        opt = torch.optim.Muon([nn.Parameter(torch.randn(16))], lr=1e-3)
        opt.param_groups[0]["params"][0].grad = torch.randn(16)
        opt.step()


@muon_required
def test_hybrid_satisfies_lightning_optimizable_protocol():
    """Lightning structurally types optimizers; the wrapper is not an Optimizer subclass."""
    from lightning.fabric.utilities.types import Optimizable

    from boomer.optim import MuonWithAuxAdamW

    opt = MuonWithAuxAdamW(_split_model(), lr=1e-3, aux_lr=1e-4)
    assert isinstance(opt, Optimizable)


@muon_required
def test_hybrid_param_groups_are_the_live_dicts_not_copies():
    """An LR scheduler mutates group['lr']; it must reach the real optimizer."""
    from boomer.optim import MuonWithAuxAdamW

    opt = MuonWithAuxAdamW(_split_model(), lr=1e-3, aux_lr=1e-4)
    for group in opt.param_groups:
        group["lr"] = 0.5
    assert all(g["lr"] == 0.5 for g in opt._muon.param_groups)
    assert all(g["lr"] == 0.5 for g in opt._aux.param_groups)


@muon_required
def test_hybrid_tags_roles_and_keeps_two_rates():
    from boomer.optim import MuonWithAuxAdamW

    opt = MuonWithAuxAdamW(_split_model(), lr=2e-2, aux_lr=1e-4)
    roles = {g["boomer_role"] for g in opt.param_groups}
    assert roles == {"muon", "aux"}
    assert {g["lr"] for g in opt.param_groups if g["boomer_role"] == "muon"} == {2e-2}
    assert {g["lr"] for g in opt.param_groups if g["boomer_role"] == "aux"} == {1e-4}


@muon_required
def test_hybrid_steps_every_parameter():
    """Both halves must actually move — a silent no-op on one group would be invisible."""
    from boomer.optim import MuonWithAuxAdamW

    torch.manual_seed(0)
    model = _split_model()
    opt = MuonWithAuxAdamW(model, lr=1e-2, aux_lr=1e-3)
    before = [p.detach().clone() for p in model.parameters()]
    for p in model.parameters():
        p.grad = torch.randn_like(p)
    opt.step()
    for name, old, new in zip(
        [n for n, _ in model.named_parameters()], before, model.parameters()
    ):
        assert not torch.equal(old, new), f"{name} did not move"
        assert torch.isfinite(new).all()


@muon_required
def test_hybrid_closure_runs_with_autograd_enabled():
    """The wrapper must support the closure protocol used by Lightning."""
    from boomer.optim import MuonWithAuxAdamW

    model = nn.Sequential(nn.Linear(16, 16), nn.LayerNorm(16))
    opt = MuonWithAuxAdamW(model, lr=1e-2, aux_lr=1e-3)
    inputs, target = torch.randn(8, 16), torch.randn(8, 16)
    before = model[0].weight.detach().clone()

    def closure():
        opt.zero_grad()
        loss = ((model(inputs) - target) ** 2).mean()
        loss.backward()
        return loss

    loss = opt.step(closure)
    assert loss.requires_grad
    assert not torch.equal(before, model[0].weight)


@muon_required
def test_hybrid_converges_on_a_quadratic():
    from boomer.optim import MuonWithAuxAdamW

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(64, 64), nn.LayerNorm(64), nn.Linear(64, 64))
    inputs, target = torch.randn(32, 64), torch.randn(32, 64)
    opt = MuonWithAuxAdamW(model, lr=1e-2, aux_lr=1e-3)
    losses = []
    for _ in range(60):
        opt.zero_grad()
        loss = ((model(inputs) - target) ** 2).mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] * 0.5, f"{losses[0]:.4f} -> {losses[-1]:.4f}"


@muon_required
def test_hybrid_state_dict_round_trips():
    from boomer.optim import MuonWithAuxAdamW

    torch.manual_seed(0)
    inputs, target = torch.randn(8, 64), torch.randn(8, 64)

    def run(resume_at):
        torch.manual_seed(1)
        model = nn.Sequential(nn.Linear(64, 64), nn.LayerNorm(64))
        opt = MuonWithAuxAdamW(model, lr=1e-2, aux_lr=1e-3)
        for step in range(8):
            if resume_at is not None and step == resume_at:
                snapshot = copy.deepcopy(opt.state_dict())
                weights = [p.detach().clone() for p in model.parameters()]
                model = nn.Sequential(nn.Linear(64, 64), nn.LayerNorm(64))
                with torch.no_grad():
                    for p, w in zip(model.parameters(), weights):
                        p.copy_(w)
                opt = MuonWithAuxAdamW(model, lr=1e-2, aux_lr=1e-3)
                opt.load_state_dict(snapshot)
            opt.zero_grad()
            ((model(inputs) - target) ** 2).mean().backward()
            opt.step()
        return [p.detach().clone() for p in model.parameters()]

    for a, b in zip(run(None), run(4)):
        assert torch.equal(a, b)


@muon_required
def test_hybrid_resume_keeps_scheduler_groups_live():
    """Loading replaces suboptimizer group dicts; the wrapper must follow them."""
    from boomer.optim import MuonWithAuxAdamW

    model = _split_model()
    opt = MuonWithAuxAdamW(model, lr=1e-3, aux_lr=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: 1.0 / (step + 1))
    snapshot = copy.deepcopy(opt.state_dict())
    opt.load_state_dict(snapshot)

    assert opt.param_groups[0] is opt._muon.param_groups[0]
    assert opt.param_groups[1] is opt._aux.param_groups[0]
    for p in model.parameters():
        p.grad = torch.randn_like(p)
    opt.step()
    sched.step()
    assert opt._muon.param_groups[0]["lr"] == pytest.approx(5e-4)
    assert opt._aux.param_groups[0]["lr"] == pytest.approx(5e-5)


@muon_required
def test_hybrid_state_is_about_half_of_adamw():
    from boomer.optim import adamw_state_bytes, muon_state_bytes

    model = nn.Sequential(nn.Linear(1024, 1024, bias=False), nn.Linear(1024, 1024, bias=False))
    assert muon_state_bytes(model) == pytest.approx(adamw_state_bytes(model) / 2, rel=0.01)


@muon_required
def test_hybrid_refuses_a_model_with_no_2d_weights():
    from boomer.optim import MuonWithAuxAdamW

    with pytest.raises(ValueError, match="no Muon-eligible"):
        MuonWithAuxAdamW(nn.LayerNorm(8), lr=1e-3, aux_lr=1e-3)


@muon_required
def test_configure_optimizers_builds_muon_and_defaults_lr_to_cli_lr():
    import train

    lit = _TinyLit(_hparams(optimizer="muon", muon_lr=None, muon_aux_lr=None))
    optim = train.BoomerLit.configure_optimizers(lit)["optimizer"]
    from boomer.optim import MuonWithAuxAdamW

    assert isinstance(optim, MuonWithAuxAdamW)
    # match_rms_adamw makes the AdamW rate transfer, so --lr is the right default
    assert all(g["lr"] == 1e-4 for g in optim.param_groups)
    assert optim.defaults["weight_decay"] == 0.1, "Muon needs decoupled weight decay"


@muon_required
def test_on_train_start_restores_both_muon_rates_separately():
    """The resume-time LR override must not collapse Muon's two rates into one."""
    import train
    from types import SimpleNamespace

    lit = _TinyLit(_hparams(optimizer="muon", muon_lr=2e-2, muon_aux_lr=1e-4))
    optim = train.BoomerLit.configure_optimizers(lit)["optimizer"]
    for group in optim.param_groups:
        group["initial_lr"] = 999.0
    sched = SimpleNamespace(base_lrs=[999.0] * len(optim.param_groups), optimizer=optim)
    lit.trainer = SimpleNamespace(
        optimizers=[optim], lr_scheduler_configs=[SimpleNamespace(scheduler=sched)]
    )
    train.BoomerLit.on_train_start(lit)

    for group in optim.param_groups:
        expected = 2e-2 if group["boomer_role"] == "muon" else 1e-4
        assert group["initial_lr"] == expected
    assert set(sched.base_lrs) == {2e-2, 1e-4}


@muon_required
def test_lambdalr_accepts_the_hybrid_and_scaling_reaches_both_suboptimizers():
    """LambdaLR does isinstance(optimizer, Optimizer) — a duck-typed wrapper dies here.

    And once attached, its warmup multiplier has to reach the dicts the sub-optimizers
    actually read, or the schedule would be adjusting a rate nobody uses.
    """
    from boomer.optim import MuonWithAuxAdamW

    model = nn.Sequential(nn.Linear(64, 64), nn.LayerNorm(64))
    opt = MuonWithAuxAdamW(model, lr=2e-2, aux_lr=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: 0.5)

    assert opt.param_groups[0] is opt._muon.param_groups[0], "group identity is the mechanism"
    assert opt.param_groups[1] is opt._aux.param_groups[0]

    for p in model.parameters():
        p.grad = torch.randn_like(p)
    opt.step()
    sched.step()
    assert opt._muon.param_groups[0]["lr"] == pytest.approx(1e-2)
    assert opt._aux.param_groups[0]["lr"] == pytest.approx(5e-5)


@muon_required
def test_end_to_end_lightning_fit_with_muon():
    """The wrapper has to survive a real Trainer.fit, not just direct stepping."""
    import lightning as L
    from torch.utils.data import DataLoader, TensorDataset

    from boomer.optim import MuonWithAuxAdamW

    class Tiny(L.LightningModule):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(16, 16), nn.LayerNorm(16), nn.Linear(16, 16))

        def training_step(self, batch, _):
            x, y = batch
            return ((self.net(x) - y) ** 2).mean()

        def configure_optimizers(self):
            opt = MuonWithAuxAdamW(self.net, lr=1e-2, aux_lr=1e-3)
            sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
            return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}

    torch.manual_seed(0)
    data = TensorDataset(torch.randn(32, 16), torch.randn(32, 16))
    model = Tiny()
    before = model.net[0].weight.detach().clone()
    L.Trainer(max_steps=4, accelerator="cpu", logger=False, enable_checkpointing=False,
              enable_progress_bar=False, enable_model_summary=False).fit(
        model, DataLoader(data, batch_size=8)
    )
    assert not torch.equal(before, model.net[0].weight), "Muon group never moved"
    assert torch.isfinite(model.net[0].weight).all()


# --- CMuon ---------------------------------------------------------------------


def _cmuon_layout_model():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.x_embedder = nn.Linear(8, 8, bias=False)
            self.qkv = nn.Linear(8, 24, bias=False)
            self.cross_attn = nn.Module()
            self.cross_attn.kv_linear = nn.Linear(8, 16, bias=False)
            self.mlp = nn.Module()
            self.mlp.inverted_fc = nn.Sequential(nn.Linear(8, 16, bias=False), nn.SiLU())
            self.mod = nn.Sequential(nn.SiLU(), nn.Linear(8, 48, bias=False))
            self.proj = nn.Linear(8, 8, bias=False)
            self.final_t_block = nn.Sequential(nn.SiLU(), nn.Linear(8, 16, bias=False))
            self.scale_shift_table = nn.Parameter(torch.zeros(6, 8))

    return Model()


def test_cmuon_split_routes_semantic_fused_rows_and_keeps_final_head_aux():
    from boomer.optim import split_cmuon_parameter_groups

    model = _cmuon_layout_model()
    groups, aux = split_cmuon_parameter_groups(model)
    by_param = {
        id(param): group.get("split_sizes")
        for group in groups
        for param in group["params"]
    }

    assert by_param[id(model.qkv.weight)] == (8, 8, 8)
    assert by_param[id(model.cross_attn.kv_linear.weight)] == (8, 8)
    assert by_param[id(model.mlp.inverted_fc[0].weight)] == (8, 8)
    assert by_param[id(model.mod[1].weight)] == (8, 8, 8, 8, 8, 8)
    assert by_param[id(model.proj.weight)] is None
    aux_ids = {id(param) for param in aux}
    assert id(model.x_embedder.weight) in aux_ids
    assert id(model.final_t_block[1].weight) in aux_ids
    assert id(model.scale_shift_table) in aux_ids


def test_cmuon_split_unwraps_torch_compile_without_losing_exclusions():
    from boomer.optim import split_cmuon_parameter_groups

    eager = _cmuon_layout_model()
    compiled = torch.compile(eager)
    eager_groups, eager_aux = split_cmuon_parameter_groups(eager)
    compiled_groups, compiled_aux = split_cmuon_parameter_groups(compiled)
    assert {
        (id(param), group.get("split_sizes"))
        for group in eager_groups
        for param in group["params"]
    } == {
        (id(param), group.get("split_sizes"))
        for group in compiled_groups
        for param in group["params"]
    }
    assert {id(p) for p in eager_aux} == {id(p) for p in compiled_aux}


def test_cmuon_newton_schulz_is_batched_finite_and_shape_preserving():
    from boomer.optim import _cmuon_newton_schulz

    torch.manual_seed(0)
    for shape in ((24, 8), (3, 8, 16)):
        grad = torch.randn(*shape)
        out = _cmuon_newton_schulz(grad, steps=6)
        assert out.shape == grad.shape
        assert out.dtype == torch.bfloat16
        assert torch.isfinite(out).all()


def _dion_available():
    return importlib.util.find_spec("dion") is not None


dion_required = pytest.mark.skipif(not _dion_available(), reason="CMuon needs Microsoft Dion")


@dion_required
def test_cmuon_wrapper_applies_paper_chunk_scaling_without_changing_decay():
    from boomer.optim import CMuonWithAuxAdamW

    opt = CMuonWithAuxAdamW(
        _cmuon_layout_model(),
        lr=1e-3,
        aux_lr=2e-4,
        weight_decay=0.1,
        chunk_rescale=True,
    )
    for group in opt.param_groups:
        if group["boomer_role"] == "aux":
            assert float(group["lr"]) == pytest.approx(2e-4)
            continue
        chunks = group["boomer_cmuon_chunks"]
        scale = math.sqrt(chunks) if chunks > 1 else 1.0
        assert float(group["lr"]) == pytest.approx(1e-3 * scale)
        # lr*wd must remain the requested decoupled-decay coefficient.
        assert float(group["lr"]) * float(group["weight_decay"]) == pytest.approx(1e-4)


@dion_required
def test_cmuon_wrapper_steps_checkpoints_and_keeps_scheduler_groups_live():
    from boomer.optim import CMuonWithAuxAdamW

    torch.manual_seed(0)
    model = _cmuon_layout_model()
    opt = CMuonWithAuxAdamW(model, lr=1e-3, aux_lr=2e-4)
    before = [p.detach().clone() for p in model.parameters()]
    for p in model.parameters():
        p.grad = torch.randn_like(p)
    opt.step()
    assert all(not torch.equal(old, new) for old, new in zip(before, model.parameters()))

    snapshot = copy.deepcopy(opt.state_dict())
    opt.load_state_dict(snapshot)
    live_groups = opt._muon.param_groups + opt._aux.param_groups
    assert all(a is b for a, b in zip(opt.param_groups, live_groups))

    old_lrs = [float(g["lr"]) for g in opt.param_groups]
    torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 0.5)
    assert [float(g["lr"]) for g in live_groups] == pytest.approx(
        [lr * 0.5 for lr in old_lrs]
    )


@dion_required
def test_configure_optimizers_builds_cmuon_and_resume_preserves_scaled_rates():
    import train
    from types import SimpleNamespace

    lit = _TinyLit(_hparams(optimizer="cmuon", cmuon_lr=2e-4, cmuon_aux_lr=1e-4))
    optim = train.BoomerLit.configure_optimizers(lit)["optimizer"]
    from boomer.optim import CMuonWithAuxAdamW

    assert isinstance(optim, CMuonWithAuxAdamW)
    for group in optim.param_groups:
        group["initial_lr"] = 999.0
    sched = SimpleNamespace(base_lrs=[999.0] * len(optim.param_groups), optimizer=optim)
    lit.trainer = SimpleNamespace(
        optimizers=[optim], lr_scheduler_configs=[SimpleNamespace(scheduler=sched)]
    )
    train.BoomerLit.on_train_start(lit)
    expected = [train.BoomerLit._cli_lr_for_group(lit, g) for g in optim.param_groups]
    assert [g["initial_lr"] for g in optim.param_groups] == pytest.approx(expected)
    assert sched.base_lrs == pytest.approx(expected)


def test_cmuon_newton_schulz_is_not_torch_compiled():
    """Dynamo specialises on shape; v8 has more shapes than recompile_limit allows.

    A `@torch.compile(fullgraph=True)` Newton-Schulz does not degrade to eager once the
    limit is hit -- it raises FailOnRecompileLimitHit and kills the run at the first
    optimizer step. Observed in Colab against the real 890M model.
    """
    from boomer.optim import _cmuon_newton_schulz

    assert not hasattr(_cmuon_newton_schulz, "_torchdynamo_orig_callable")
    assert _cmuon_newton_schulz.__module__ == "boomer.optim"

    # Many distinct shapes in one process must all work.
    for rows, cols in ((8, 16), (16, 8), (32, 16), (12, 12), (64, 32), (7, 5),
                       (128, 64), (24, 48), (96, 96), (5, 11)):
        out = _cmuon_newton_schulz(torch.randn(rows, cols), steps=6)
        assert out.shape == (rows, cols)
        assert torch.isfinite(out).all()


@dion_required
def test_patch_removes_dion_compile_and_the_full_size_temporary():
    """Dion ships both Muon helpers @torch.compile(fullgraph=True) and copies U."""
    from dion import muon as dion_muon

    from boomer.optim import patch_dion_muon_kernels

    patch_dion_muon_kernels()
    for name in ("muon_update_pre_orthogonalize", "muon_update_post_orthogonalize"):
        fn = getattr(dion_muon, name)
        assert not hasattr(fn, "_torchdynamo_orig_callable"), f"{name} still compiled"
    assert patch_dion_muon_kernels() == {}, "patch must be idempotent"


@dion_required
def test_patched_post_orthogonalize_matches_upstream_numerically():
    """Guards against upstream drift: same result, one fewer allocation."""
    from boomer.optim import _muon_update_post_orthogonalize_inplace

    torch.manual_seed(0)
    for cautious in (False, True):
        args = lambda: (  # noqa: E731
            [torch.full((4, 6), 0.5), torch.full((3, 3), -0.25)],
            [torch.arange(24.0).reshape(4, 6) / 24, torch.ones(3, 3) * 0.1],
        )
        base_lr, adj_lr, wd = 1e-3, 2e-3, 0.05

        # Upstream body, transcribed out-of-place.
        X_ref, U_ref = args()
        if cautious:
            masks = torch._foreach_minimum(
                torch._foreach_add(torch._foreach_sign(
                    torch._foreach_mul(X_ref, U_ref)), 1), 1)
            terms = torch._foreach_mul(torch._foreach_mul(X_ref, masks), base_lr * wd)
            torch._foreach_sub_(X_ref, terms)
        else:
            torch._foreach_mul_(X_ref, 1 - base_lr * wd)
        torch._foreach_sub_(X_ref, torch._foreach_mul(U_ref, adj_lr))

        X_new, U_new = args()
        _muon_update_post_orthogonalize_inplace(
            X_new, U_new, base_lr, adj_lr, wd, cautious_wd=cautious
        )
        for ref, got in zip(X_ref, X_new):
            torch.testing.assert_close(got, ref)


@dion_required
def test_patched_step_runs_more_shape_buckets_than_the_default_limit():
    """The original crash: >8 distinct shapes through Dion, now uncompiled."""
    from boomer.optim import CMuonWithAuxAdamW

    class ManyShapes(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList(
                [nn.Linear(8 + 2 * i, 16 + 4 * i, bias=False) for i in range(12)]
            )

    model = ManyShapes()
    opt = CMuonWithAuxAdamW(model, lr=1e-3, aux_lr=1e-4)
    for param in model.parameters():
        param.grad = torch.randn_like(param)
    opt.step()  # must not raise FailOnRecompileLimitHit
    for param in model.parameters():
        assert torch.isfinite(param).all()
