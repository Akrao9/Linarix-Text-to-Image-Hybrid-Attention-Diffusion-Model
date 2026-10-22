"""Lightning module, frozen encoders, losses, and optimizer configuration."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

from boomer.models.qwen_connector import (
    QwenConnectorConfig,
    QwenReferenceEncoder,
    encode_qwen_text_fixed,
)
from boomer.models.boomer_fla_dit import (
    BoomerFLADiTConfig,
    cross_sdpa_backend,
    joint_sdpa_backend,
)
from boomer.models.factory import create_model
from boomer.sana_flow import (
    flow_min_snr_loss_weight,
    flow_training_loss_weight,
    noise_latent_at_sigma,
    sample_flow_sigmas,
)
from boomer.fill_masks import (
    parse_fill_mask_probs,
    sample_fill_masks,
)
from boomer.sana_latent_cache import (
    DEFAULT_SCALING_FACTOR,
    DEFAULT_VAE_ID,
    CacheConfig,
    extract_latent,
    load_vae,
    vae_scaling_factor,
)
from boomer.training._lightning import pl
from boomer.training.config import parse_int_tuple


def dispersive_loss(z: torch.Tensor, temperature: float, metric: str) -> torch.Tensor:
    """InfoNCE-style dispersive regularizer (no positives, no external target).

    Pushes the batch's per-sample features apart by minimizing the log-mean-exp of
    pairwise similarity. `metric="l2"` uses negative squared distance as the score
    (the paper's recommended default); `metric="cosine"` uses cosine similarity on
    L2-normalized features. Diagonal pairs are included, matching the paper.
    """
    z = z.float()
    batch = z.shape[0]
    if batch < 2:
        return z.new_zeros(())
    if metric == "cosine":
        z = F.normalize(z, dim=-1)
        scores = (z @ z.t()) / temperature              # [B, B] similarity / tau
    else:  # l2
        # Squared distance via the gram identity ||a-b||^2 = |a|^2 + |b|^2 - 2 a.b:
        # one matmul, no sqrt (cdist would sqrt then we'd square it back). Normalize
        # by feature dim so tau is scale-stable across hidden sizes; otherwise high-dim
        # distances underflow exp() and the gradient vanishes (regularizer does nothing).
        sq_norm = z.pow(2).sum(dim=-1)                          # [B]
        sq_dist = (sq_norm[:, None] + sq_norm[None, :] - 2.0 * (z @ z.t())).clamp_min(0.0)
        # Self-distance is exactly 0 by definition. The gram matmul runs under bf16
        # autocast, so its diagonal is imprecise and grows with |z|^2 -- left alone it
        # drags the loss below the true -log(B) floor once feature norms get large.
        sq_dist.fill_diagonal_(0.0)
        scores = -sq_dist / (z.shape[-1] * temperature)         # [B, B]
    # log( mean_{i,j} exp(scores) ) -- minimizing this disperses the features.
    return torch.logsumexp(scores.reshape(-1), dim=0) - math.log(batch * batch)


class BoomerLit(pl.LightningModule):
    def __init__(
        self,
        model_config: BoomerFLADiTConfig,
        hf_token: str | bool | None = None,
        **hp: Any,
    ) -> None:
        super().__init__()
        self.save_hyperparameters({**hp, "model_config": model_config.__dict__})
        self.config = model_config
        self.model = create_model(model_config)
        # Frozen on-the-fly text encoder (--monet-direct). Kept in a plain list so it is
        # NOT a registered submodule: it must stay out of state_dict (checkpoints would
        # grow by the full encoder), out of self.model.parameters() (optimizer + grad
        # clip), and out of Lightning's device management (moved lazily on first use).
        self._hf_token = hf_token
        self._text_encoder_box: list[QwenReferenceEncoder] = []
        self._text_encoder_compiled = False
        # Frozen on-the-fly DC-AE (--edit-direct pixel pairs). Same out-of-tree
        # residency rules as the text encoder box above.
        self._vae_box: list[Any] = []
        self._vae_compiled = False
        # Plain list, NOT a bare attribute: assigning an nn.Module to self registers it
        # as a submodule, which puts the whole frozen DC-AE encoder into state_dict and
        # then into every checkpoint -- and makes those checkpoints fail to load with
        # "Unexpected key(s) in state_dict: _vae_eager_encoder.*". Same out-of-tree
        # residency rule as _vae_box / _text_encoder_box above, for the same reason.
        self._vae_eager_box: list[Any] = []
        # --fill-training: mask-type mix, parsed once (fails fast on a bad spec).
        self._fill_mask_probs = (
            parse_fill_mask_probs(hp.get("fill_mask_probs")) if hp.get("fill_training") else None
        )
        # --sigma-buckets: running (sum, count) of the UNWEIGHTED per-sample flow MSE
        # per fixed sigma band, flushed every --sigma-bucket-log-every steps. Kept off
        # the state_dict (plain attrs, not buffers): they are a logging accumulator, so
        # restoring a mid-window partial sum into a resumed run would be meaningless.
        self._sigma_bucket_sum: torch.Tensor | None = None
        self._sigma_bucket_cnt: torch.Tensor | None = None
        # global_step of the last flush. Under --accumulate N, training_step runs N
        # times per global_step, so a bare `global_step % every == 0` test fires N
        # times: the first flush drains the window and the rest log a near-empty one.
        self._sigma_bucket_flushed_at = -1
        # Same accumulation trap for AttnRes telemetry: without this guard, a logging
        # step would collect weights and submit all 116 scalars once per microbatch.
        self._attnres_collected_at = -1
        self._joint_backend_logged = False
        self._cross_backend_logged = False
        if hp.get("compile"):
            self.model = torch.compile(self.model, mode=hp.get("compile_mode", "default"))

    def _resident_text_encoder(self) -> QwenReferenceEncoder:
        if not self._text_encoder_box:
            hp = self.hparams
            readout_layers = parse_int_tuple(hp.get("text_encoder_layers"))
            cfg = QwenConnectorConfig(
                qwen_model_id=hp.text_encoder_id,
                readout_layer=hp.text_encoder_layer,
                readout_layers=readout_layers,
                instruction=(
                    QwenConnectorConfig.instruction if hp.text_encoder_instruction else None
                ),
                dtype="bf16",
                max_text_tokens=hp.text_encoder_max_length,
            )
            encoder = QwenReferenceEncoder(cfg, hf_token=self._hf_token)
            encoder.load()
            # The DiT's text_dim is fixed at construction (main() multiplies it by
            # len(readout_layers)); the encoder's width is decided here. A disagreement
            # otherwise surfaces as an opaque broadcast error inside the first
            # apply_condition_dropout, one full data-pipeline startup later.
            if encoder.readout_dim != self.config.text_dim:
                raise SystemExit(
                    f"text encoder emits dim {encoder.readout_dim} but the DiT was built "
                    f"for text_dim={self.config.text_dim}. readout_layers={readout_layers}, "
                    f"readout_layer={hp.text_encoder_layer}, hidden={encoder.hidden_size}. "
                    "Check that --text-encoder-layers reached BoomerLit's hparams."
                )
            layer_desc = (
                f"layers={readout_layers} (fused, dim={encoder.readout_dim})"
                if readout_layers
                else f"layer={hp.text_encoder_layer}"
            )
            print(
                f"[text-encoder] {hp.text_encoder_id} resident (hidden={encoder.hidden_size}, "
                f"{layer_desc}, max_length={hp.text_encoder_max_length})",
                flush=True,
            )
            self._text_encoder_box.append(encoder)
        encoder = self._text_encoder_box[0]
        param = next(encoder.model.parameters())
        if param.device != self.device:
            encoder.to(self.device)
        # Compile AFTER truncation (in load()) and after the device move, so the graph is
        # traced on the layers/device actually used. Once per process; the encoder tracks
        # its own compiled state.
        if self.hparams.get("compile_text_encoder") and not self._text_encoder_compiled:
            encoder.compile_base_model(mode=self.hparams.get("compile_mode", "default"))
            self._text_encoder_compiled = True
        return encoder

    def encode_captions(self, captions: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode raw captions with the resident frozen Qwen encoder.

        Returns fixed-shape ``(text [B, L, D] bf16, mask [B, L])`` on self.device.
        The hidden states stay bf16 (the encoder's compute dtype): upcasting to fp32
        would double the [B, L, 2560] conditioning activations for nothing — the
        denoiser consumes them under bf16 autocast (or FSDP2's bf16 compute policy).
        ``encode_qwen_text_fixed`` runs under no_grad (not inference_mode), so the
        outputs enter the training graph directly with no per-step clone.
        """
        hp = self.hparams
        encoder = self._resident_text_encoder()
        hidden, mask = encode_qwen_text_fixed(encoder, captions, hp.text_encoder_max_length)
        if (
            not torch.is_autocast_enabled(hidden.device.type)
            and self.device_mesh is None
            and hidden.dtype != self.model.caption_embedder.y_proj[0].weight.dtype
        ):
            # 32-true precision without FSDP2: no autocast and no MixedPrecisionPolicy
            # will reconcile bf16 text with the fp32 caption embedder — upcast here
            # rather than crash on the dtype mismatch.
            hidden = hidden.to(self.model.caption_embedder.y_proj[0].weight.dtype)
        return hidden, mask

    def _resident_vae(self) -> Any:
        if not self._vae_box:
            hp = self.hparams
            cfg = CacheConfig(
                vae_id=getattr(hp, "edit_vae_id", DEFAULT_VAE_ID),
                dtype="bf16",
                device=str(self.device),
                scaling_factor=getattr(hp, "edit_scaling_factor", DEFAULT_SCALING_FACTOR),
            )
            vae = load_vae(cfg, hf_token=self._hf_token)
            vae.requires_grad_(False)
            vae.eval()
            print(f"[pixel-direct] DC-AE {cfg.vae_id} resident (frozen, bf16, "
                  f"scaling={cfg.scaling_factor})", flush=True)
            self._vae_box.append(vae)
        vae = self._vae_box[0]
        if next(vae.parameters()).device != self.device:
            vae = vae.to(self.device)
            self._vae_box[0] = vae
        # --compile-vae: the DC-AE lives outside self.model (so it stays out of
        # state_dict/optimizer/Lightning device management), which also means --compile
        # never reaches it -- it runs EAGER, paying full per-op Python dispatch on every
        # conv and norm while encoding --batch-size images at --t2i-resolution every
        # step. Profiled on the v7 1024 run: DC-AE 50% of step time vs 20% for the DiT
        # itself, with _conv_forward and normalization.forward dominating OwnTime, which
        # is the dispatch-overhead signature. Compile the encoder stack only: the
        # AutoencoderDC.encode wrapper returns a dataclass and is not worth tracing.
        if self.hparams.get("compile_vae") and not self._vae_compiled:
            encoder = getattr(vae, "encoder", None)
            if encoder is None:
                print("[compile-vae] no .encoder submodule on the DC-AE; skipping",
                      flush=True)
            else:
                mode = self.hparams.get("compile_mode", "default")
                # Keep the eager module so a compiled-graph failure can fall back
                # instead of killing the run, mirroring the text encoder's
                # _eager_base_model. diffusers has open reports of torch.compile
                # failing on this model family (huggingface/diffusers#10937,
                # #11864) -- both on the @apply_forward_hook-decorated .encode
                # rather than the raw .encoder compiled here, but a frozen
                # preprocessing step is not worth a hard crash either way.
                self._vae_eager_box.append(encoder)
                vae.encoder = torch.compile(encoder, mode=mode)
                print(f"[compile-vae] torch.compile(mode={mode}) on the DC-AE encoder",
                      flush=True)
            self._vae_compiled = True
        return vae

    def _encode_pixel_chunks(self, vae, pixels, *, chunk: int, dtype) -> torch.Tensor:
        parts = []
        for part in pixels.split(chunk):
            images = part.to(device=self.device, dtype=dtype, non_blocking=True)
            images = images / 127.5 - 1.0
            parts.append(extract_latent(vae.encode(images)))
        return torch.cat(parts, dim=0)

    @torch.no_grad()
    def encode_pixels(self, pixels: torch.Tensor) -> torch.Tensor:
        """Encode uint8 (B, 3, H, W) pixels to DC-AE latents in RAW cache space.

        Chunked (--edit-encode-chunk) so 2x batch images at 1024^2 never spike
        VRAM next to the DiT. no_grad (not inference_mode) for the same reason
        as encode_captions: the latents enter the training graph directly.
        The scaling factor matches the cache pipeline (verified round-trip
        corr 0.994 against boomer-finet2i-1024-parquet), so on-the-fly latents
        live in the exact distribution the t2i phases trained on.
        """
        hp = self.hparams
        vae = self._resident_vae()
        dtype = next(vae.parameters()).dtype
        chunk = max(1, int(getattr(hp, "edit_encode_chunk", 8)))
        scale = vae_scaling_factor(vae, getattr(hp, "edit_scaling_factor", DEFAULT_SCALING_FACTOR))
        try:
            return self._encode_pixel_chunks(vae, pixels, chunk=chunk, dtype=dtype) * scale
        except Exception as exc:  # noqa: BLE001 - --compile-vae fallback, see below
            if not self._vae_eager_box:
                raise
            # Drop the compiled encoder permanently and redo this batch eagerly. The
            # latents are numerically the same either way, so the run continues with
            # nothing worse than the speed it had before --compile-vae.
            print(f"[compile-vae] compiled DC-AE encoder failed ({type(exc).__name__}: "
                  f"{exc}); falling back to eager for the rest of the run", flush=True)
            vae.encoder = self._vae_eager_box.pop()
            return self._encode_pixel_chunks(vae, pixels, chunk=chunk, dtype=dtype) * scale

    def configure_model(self) -> None:
        """--fsdp2 (Lightning ModelParallelStrategy): shard every DiT block, then the
        root module, over the data-parallel mesh with FSDP2's ``fully_shard``.

        Sharded master parameters stay fp32; ``MixedPrecisionPolicy`` gathers/computes
        forward+backward in bf16 and all-reduces gradients in fp32, so Trainer
        precision must be 32-true (main() enforces it — Lightning-side autocast would
        be redundant and bf16-true would destroy the fp32 masters). The frozen Qwen
        encoder is untouched: it lives outside the module tree (``_text_encoder_box``)
        and every rank keeps its own full bf16 replica. This hook runs for every
        strategy, so it is a no-op when there is no device mesh (single GPU / DDP).
        """
        if self.device_mesh is None:
            return
        try:  # public path (torch >= 2.6), composable fallback (2.4/2.5)
            from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
            from torch.distributed.tensor import DTensor
        except ImportError:  # pragma: no cover - depends on the torch version
            from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard
            from torch.distributed._tensor import DTensor

        model = getattr(self.model, "_orig_mod", self.model)
        if any(isinstance(p, DTensor) for p in model.parameters()):
            return  # already sharded (the hook may be invoked more than once)
        mesh = self.device_mesh["data_parallel"]
        policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
        # Per-block sharding: each block's params all-gather just-in-time for its
        # forward/backward and reshard after, overlapping comms with compute. The
        # root shard picks up everything outside the blocks (embedders, final layer).
        for block in model.blocks:
            fully_shard(block, mesh=mesh, mp_policy=policy)
        fully_shard(model, mesh=mesh, mp_policy=policy)
        print(
            f"[fsdp2] sharded {len(model.blocks)} blocks + root over "
            f"{mesh.size()}-way data-parallel mesh (bf16 compute, fp32 reduce/master)",
            flush=True,
        )

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        # Keep inference self-describing even though Lightning's native state uses
        # `state_dict`/`hyper_parameters`. Older sampling tools consume these fields.
        checkpoint["config"] = dict(self.config.__dict__)
        checkpoint["train_args"] = {
            key: value for key, value in dict(self.hparams).items() if key != "model_config"
        }

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        # An early --compile-vae build stashed the eager DC-AE encoder as a bare
        # attribute, which registered it as a submodule and baked the whole frozen
        # encoder into state_dict. Those checkpoints are otherwise fine but refuse to
        # load ("Unexpected key(s) in state_dict: _vae_eager_encoder.*"), so drop the
        # keys here. Purely a compatibility shim for checkpoints already on disk;
        # current builds keep the encoder in _vae_eager_box and never write them.
        state = checkpoint.get("state_dict")
        if state:
            stale = [k for k in state if k.startswith("_vae_eager_encoder.")]
            for key in stale:
                del state[key]
            if stale:
                print(f"[resume] dropped {len(stale)} stale _vae_eager_encoder.* keys "
                      "(frozen DC-AE wrongly saved into an older checkpoint)", flush=True)

        # Adam moments saved by the legacy Conv2d GLUMBConv are 4D (O, I, 1, 1); the
        # migrated Linear params are 2D. Optimizer.load_state_dict casts dtype/device
        # but never reshapes, so realign same-numel moment tensors to the current
        # param shapes before Lightning restores the optimizer — otherwise the first
        # step after a full resume of a pre-refactor checkpoint broadcast-errors.
        params = list(self.model.parameters())
        realigned = 0
        for opt_state in checkpoint.get("optimizer_states") or []:
            flat = [i for g in opt_state.get("param_groups") or [] for i in g.get("params", [])]
            if len(flat) != len(params):
                continue  # param set changed; let Lightning surface the real error
            idx_to_param = dict(zip(flat, params))
            for idx, entry in (opt_state.get("state") or {}).items():
                param = idx_to_param.get(idx)
                if param is None:
                    continue
                for key, value in entry.items():
                    if (
                        torch.is_tensor(value)
                        and value.ndim > 0
                        and value.shape != param.shape
                        and value.numel() == param.numel()
                    ):
                        entry[key] = value.reshape(param.shape)
                        realigned += 1
        if realigned:
            print(f"[resume] reshaped {realigned} optimizer moment tensors to migrated "
                  "param shapes (legacy GLUMBConv checkpoint)", flush=True)

    def _log_sigma_buckets(self, t: torch.Tensor, raw_per_sample: torch.Tensor, bs: int) -> None:
        """Accumulate the UNWEIGHTED flow MSE into fixed sigma bands and flush the means
        every ``--sigma-bucket-log-every`` steps.

        Why this beats train/flow_loss as a progress signal: the aggregate loss is an
        average over whatever sigma distribution the sampler happens to draw, so its
        value moves whenever the WEIGHTING moves — changing --flow-shift, or crossing
        --t-sampler-final-step — with no change in model quality. A band's mean is far
        less sensitive to that, so these curves stay roughly comparable across the
        512->1024 shift change and the polish switch, and they say WHICH noise regime
        is still learning: high-sigma bands carry global composition, low-sigma bands
        carry texture.

        NOT fully invariant, and the residual matters. Reweighting also moves the
        distribution WITHIN a band, which leaks into its mean. Measured on a synthetic
        loss linear in t (the worst case) over --flow-shift 1.5 -> 3.0: the aggregate
        moved 0.158, the worst band 0.019 at 5 bands and 0.006 at 10. Since real
        progress at this stage is ~0.01-0.02 per 100k steps, at 5 bands the leak is
        the same size as the signal — so read across a weighting change with care, and
        raise --sigma-buckets if a finer read is needed. The tradeoff is population:
        at 64 samples/step and a 100-step window, 5 bands leaves >=44 samples in the
        thinnest band under shift 3.0, 10 bands leaves 5, and 20 bands leaves one band
        empty. Hence the default of 5.

        Bands are fixed equal-width slices of t in [0, 1], never quantiles, so a bin
        means the same thing at every step. train/sigma_frac_* reports how much of the
        batch landed in each band — the direct readout of the current weighting, and
        the check that a band is populated enough for its mean to mean anything.

        Accumulating instead of logging per-step is the point: at batch 16 a single
        step puts ~3 samples in a band, which is pure noise. Cost is one index_add_ per
        step and one device sync per flush.
        """
        nb = int(getattr(self.hparams, "sigma_buckets", 0) or 0)
        every = int(getattr(self.hparams, "sigma_bucket_log_every", 0) or 0)
        if nb <= 0 or every <= 0:
            return
        with torch.no_grad():
            flat = raw_per_sample.detach().float().flatten()
            idx = (t.detach().float().flatten() * nb).long().clamp_(0, nb - 1)
            if self._sigma_bucket_sum is None or self._sigma_bucket_sum.device != flat.device:
                self._sigma_bucket_sum = torch.zeros(nb, device=flat.device)
                self._sigma_bucket_cnt = torch.zeros(nb, device=flat.device)
            self._sigma_bucket_sum.index_add_(0, idx, flat)
            self._sigma_bucket_cnt.index_add_(0, idx, torch.ones_like(flat))

            step = self.global_step
            if step == 0 or step % every != 0 or step == self._sigma_bucket_flushed_at:
                return
            self._sigma_bucket_flushed_at = step
            # One transfer, then plain Python: a sync every `every` steps is free, a
            # sync per step (indexing the GPU tensors inline) would not be.
            sums = self._sigma_bucket_sum.cpu().tolist()
            cnts = self._sigma_bucket_cnt.cpu().tolist()
            total = sum(cnts)
            for i, (s, c) in enumerate(zip(sums, cnts)):
                lo, hi = i / nb, (i + 1) / nb
                band = f"{lo:.2f}_{hi:.2f}"
                if c > 0:
                    self.log(f"train/flow_loss_sigma_{band}", s / c,
                             on_step=True, on_epoch=False, batch_size=bs)
                if total > 0:
                    self.log(f"train/sigma_frac_{band}", c / total,
                             on_step=True, on_epoch=False, batch_size=bs)
            self._sigma_bucket_sum.zero_()
            self._sigma_bucket_cnt.zero_()

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        hp = self.hparams
        x0 = batch.get("image_latent")
        src_latent = None
        if x0 is None:
            pixels = batch.get("pixels")
            if pixels is not None:
                # --t2i-direct / --monet-image-direct: the batch carries a single
                # image; encode it to x0 with the frozen DC-AE (no cond_input).
                x0 = self.encode_pixels(pixels)
            else:
                # --edit-direct: the batch carries a pixel pair; encode both sides
                # with the resident frozen DC-AE (target -> x0, source -> conditioning).
                x0 = self.encode_pixels(batch["target_pixels"])
                src_latent = self.encode_pixels(batch["source_pixels"])
        if hp.latent_mean != 0.0 or hp.latent_std != 1.0:
            x0 = (x0 - hp.latent_mean) / hp.latent_std
            if src_latent is not None:
                src_latent = (src_latent - hp.latent_mean) / hp.latent_std

        text = batch.get("text_embedding")
        if text is None:
            # Direct pixel/MONET paths carry raw captions; encode them on-GPU with
            # the resident frozen Qwen encoder (no cached text in the stream).
            text, mask = self.encode_captions(batch["caption"])
        else:
            mask = batch["attention_mask"]
        text, mask = self.model.apply_condition_dropout(text, mask, hp.cond_dropout)

        noise = torch.randn_like(x0)
        # "Texture polish" phase: once global_step crosses t_sampler_final_step, switch the
        # timestep sampler (e.g. -> uniform), reallocating compute toward the low-noise
        # t~0 regime to sharpen textures/edges instead of refining global layout.
        sampler = hp.t_sampler
        polish = hp.t_sampler_final and self.global_step >= hp.t_sampler_final_step
        if polish:
            sampler = hp.t_sampler_final
        t = sample_flow_sigmas(
            x0.shape[0],
            device=x0.device,
            dtype=x0.dtype,
            sampler=sampler,
            bias=hp.t_bias,
            flow_shift=hp.flow_shift,
            logit_mean=hp.logit_mean,
            logit_std=hp.logit_std,
            plateau_low=hp.plateau_low,
            plateau_high=hp.plateau_high,
            stretch_q_low=hp.stretch_q_low,
            stretch_q_high=hp.stretch_q_high,
        )
        x_t, _ = noise_latent_at_sigma(x0, t, noise=noise)
        target = noise - x0

        # Reserved-channel conditioning (extra_input_channels = latent + 1):
        #   * --fill-training (E1): [x0 * (1 - m), m] from a synthesized mask mix —
        #     self-supervised inpainting over the plain t2i stream.
        #   * --edit-direct (E2): [source latent, all-ones mask] — full-frame
        #     instruction editing under the same fill interface.
        # Both in the NORMALIZED latent space the model sees. cond_input_dropout
        # zeroes the whole block per-sample (exactly the zero-fed t2i pretrain
        # state), enabling image-side CFG at inference.
        cond_input = None
        if self._fill_mask_probs is not None:
            fill_mask = sample_fill_masks(
                x0.shape[0], x0.shape[-2], x0.shape[-1],
                probs=self._fill_mask_probs, device=x0.device, dtype=x0.dtype,
            )
            cond_input = torch.cat([x0 * (1.0 - fill_mask), fill_mask], dim=1)
        elif src_latent is not None:
            ones = src_latent.new_ones(src_latent.shape[0], 1, *src_latent.shape[-2:])
            cond_input = torch.cat([src_latent, ones], dim=1)
        if cond_input is not None:
            drop_p = float(getattr(hp, "cond_input_dropout", 0.0))
            if drop_p > 0.0:
                drop = torch.rand(cond_input.shape[0], device=cond_input.device) < drop_p
                cond_input = torch.where(
                    drop.view(-1, 1, 1, 1), torch.zeros_like(cond_input), cond_input
                )

        # REPA: align a pooled mid-depth feature to monet's DINOv2 CLS when enabled
        # (repa_dim > 0, repa_weight > 0, and the cache carries dino_embedding).
        repa_target = batch.get("dino_embedding")
        want_repa = (
            hp.repa_weight > 0.0
            and getattr(self.config, "repa_dim", 0) > 0
            and repa_target is not None
        )
        # Dispersive loss needs no external target, so it only depends on its weight.
        want_disp = hp.dispersive_weight > 0.0
        # v8 Block AttnRes telemetry is opt-in per step: return_weights=True plus 23
        # per-site reductions every microbatch (and up to 116 self.log calls x accum)
        # is pure overhead off the logging cadence, so collect only when this step's
        # scalars will actually be written — and only on the FIRST microbatch of the
        # accumulation window (global_step is constant across all of them; same trap
        # as _sigma_bucket_flushed_at).
        attnres = getattr(self.model, "block_attnres", None)
        if attnres is not None:
            log_every = max(int(getattr(self.trainer, "log_every_n_steps", 50) or 1), 1)
            attnres.collect_weights = (
                self.global_step % log_every == 0
                and self.global_step != self._attnres_collected_at
            )
        out = self.model(
            x_t, t, text, mask,
            cond_input=cond_input,
            return_repa=want_repa,
            return_dispersive=want_disp,
        )
        if want_repa and want_disp:
            pred, repa_feat, disp_feat = out
        elif want_repa:
            pred, repa_feat = out
            disp_feat = None
        elif want_disp:
            pred, disp_feat = out
            repa_feat = None
        else:
            pred = out
            repa_feat = disp_feat = None

        per_sample = (pred.float() - target.float()).pow(2).mean(dim=(1, 2, 3))
        # Hold the pre-reweighting MSE for the per-sigma buckets. The reweighting below
        # rebinds `per_sample` to a new tensor, so this keeps pointing at the raw one.
        raw_per_sample = per_sample
        # Exactly one per-timestep reweighting (mutual exclusion enforced at startup in
        # main()); timestep_weight takes precedence if both ever slip through.
        if hp.timestep_weight:
            per_sample = per_sample * flow_training_loss_weight(t, target).mean(dim=(1, 2, 3))
        elif hp.min_snr_gamma > 0.0:
            per_sample = per_sample * flow_min_snr_loss_weight(t, target, hp.min_snr_gamma).mean(dim=(1, 2, 3))
        flow_loss = per_sample.mean()

        # Log the components separately. train/total_loss is what the optimizer actually
        # receives (flow + repa_w * repa); train/flow_loss is the pure denoising MSE.
        bs = x0.shape[0]
        loss = flow_loss
        self.log("train/flow_loss", flow_loss, prog_bar=True, on_step=True, on_epoch=False, batch_size=bs)
        # Training-health canaries: scheduled LR (warmup/decay visibility) and the mean
        # sampled sigma (confirms the timestep sampler + the polish-phase switch).
        sched = self.lr_schedulers()
        if sched is not None:
            self.log("train/lr", sched.get_last_lr()[0], on_step=True, on_epoch=False, batch_size=bs)
        self.log("train/sigma_mean", t.float().mean(), on_step=True, on_epoch=False, batch_size=bs)
        self._log_sigma_buckets(t, raw_per_sample, bs)
        if hp.t_sampler_final:
            self.log("train/t_polish", float(polish), on_step=True, on_epoch=False, batch_size=bs)
        if self._fill_mask_probs is not None:
            # Mean hole fraction across the batch (post mask-mix, pre cond dropout):
            # sanity signal that the mask distribution is what --fill-mask-probs asked for.
            self.log("train/fill_coverage", fill_mask.float().mean(),
                     on_step=True, on_epoch=False, batch_size=bs)
        # Report which SDPA kernel the 8 joint-attention sites resolved to, once. The
        # math backend materializes [B, heads, S, S] and keeps it for backward, so a
        # silent fall back to it is a large memory regression; joint_sdpa also warns.
        if not self._joint_backend_logged:
            backend = joint_sdpa_backend()
            if backend is not None:
                self._joint_backend_logged = True
                print(f"[attn] joint attention SDPA backend: {backend}", flush=True)
        if not self._cross_backend_logged:
            backend = cross_sdpa_backend()
            if backend is not None:
                self._cross_backend_logged = True
                print(f"[attn] cross attention SDPA backend: {backend}", flush=True)

        # v8 Block AttnRes telemetry: mean depth-attention weight per (site, slot). Sites
        # 0..encoder_depth-1 are per-block retrievals; the last site is the final z
        # aggregation. Slot 0 is the token embedding, middle slots are completed group
        # delta sums (shallow -> deep), the last slot is the current partial sum. Init is
        # the paper's uniform 1/N; weights staying pinned there = AttnRes is not
        # differentiating sources; structured drift = it is earning its parameters.
        # Collected only on collect_weights steps (set above at the logging cadence);
        # BlockAttnRes.forward is compiler-disabled (FLA's dispatch already graph-breaks
        # there), so last_weights refreshes on those steps even under --compile.
        if attnres is not None and attnres.collect_weights:
            self._attnres_collected_at = self.global_step
            for site, w in enumerate(attnres.last_weights):
                if w is None:
                    continue
                for slot in range(w.numel()):
                    self.log(f"attnres/site{site}_slot{slot}", w[slot],
                             on_step=True, on_epoch=False, batch_size=bs)

        if repa_feat is not None:
            cos = F.cosine_similarity(
                F.normalize(repa_feat.float(), dim=-1),
                F.normalize(repa_target.float(), dim=-1),
                dim=-1,
            )
            repa_loss = (1.0 - cos).mean()
            repa_w = hp.repa_weight
            if hp.repa_anneal_steps > 0:
                repa_w *= max(0.0, 1.0 - self.global_step / hp.repa_anneal_steps)
            loss = flow_loss + repa_w * repa_loss
            self.log("train/repa_loss", repa_loss, prog_bar=True, on_step=True, on_epoch=False, batch_size=bs)
            self.log("train/repa_w", repa_w, on_step=True, on_epoch=False, batch_size=bs)
            self.log("train/repa_cos", cos.mean(), on_step=True, on_epoch=False, batch_size=bs)
            self.log("train/repa_feat_norm", repa_feat.float().norm(dim=-1).mean(),
                     on_step=True, on_epoch=False, batch_size=bs)

        if disp_feat is not None:
            disp_loss = dispersive_loss(disp_feat, hp.dispersive_temp, hp.dispersive_metric)
            disp_w = hp.dispersive_weight
            if hp.dispersive_anneal_steps > 0:
                disp_w *= max(0.0, 1.0 - self.global_step / hp.dispersive_anneal_steps)
            disp_term = disp_w * disp_loss          # actual contribution to total loss
            loss = loss + disp_term
            # raw dispersive_loss is negative by construction; dispersive_term is the
            # signed amount actually added to total_loss (clearer to read on W&B).
            self.log("train/dispersive_loss", disp_loss, prog_bar=True, on_step=True, on_epoch=False, batch_size=bs)
            self.log("train/dispersive_w", disp_w, on_step=True, on_epoch=False, batch_size=bs)
            self.log("train/dispersive_term", disp_term, on_step=True, on_epoch=False, batch_size=bs)
            # Debug: the actual batch dim the loss sees and the l2 floor (-log B). If
            # dispersive_loss ever prints below dispersive_floor, B isn't what you think.
            disp_bs = disp_feat.shape[0]
            self.log("train/dispersive_bs", float(disp_bs), on_step=True, on_epoch=False, batch_size=bs)
            self.log("train/dispersive_floor", -math.log(disp_bs), on_step=True, on_epoch=False, batch_size=bs)
            # Mean pooled-feature norm: with the l2 metric the paper attributes its gains
            # partly to growing representation norms, so watch this trend upward.
            self.log("train/dispersive_feat_norm", disp_feat.float().norm(dim=-1).mean(),
                     on_step=True, on_epoch=False, batch_size=bs)

        self.log("train/total_loss", loss, prog_bar=True, on_step=True, on_epoch=False, batch_size=bs)
        return loss

    def on_train_batch_start(self, batch, batch_idx: int) -> None:
        """Open a fresh CUDA peak window for this complete dataloader batch."""
        del batch, batch_idx
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
        """Log after automatic backward/optimizer/callback work, not mid-forward.

        Lightning calls this hook after the batch's automatic optimization path.  The
        paired reset in ``on_train_batch_start`` makes these genuine per-batch peaks;
        ``max_memory_reserved`` is used rather than the current allocator reservation.
        On an accumulation boundary the window also includes optimizer and EMA work.
        """
        del outputs, batch_idx
        if not torch.cuda.is_available():
            return
        batch_size = 1
        if isinstance(batch, dict):
            batch_size = next(
                (int(value.shape[0]) for value in batch.values()
                 if isinstance(value, torch.Tensor) and value.ndim > 0),
                1,
            )
        self.log(
            "train/vram_peak_gb",
            torch.cuda.max_memory_allocated() / 1e9,
            on_step=True,
            on_epoch=False,
            batch_size=batch_size,
        )
        self.log(
            "train/vram_reserved_peak_gb",
            torch.cuda.max_memory_reserved() / 1e9,
            on_step=True,
            on_epoch=False,
            batch_size=batch_size,
        )

    def on_train_start(self) -> None:
        # Let the CLI --lr win over restored state, the way --flow-shift already does.
        # A full --resume restores BOTH optimizer param_groups["lr"] AND the
        # scheduler's base_lrs (they are in LambdaLR.state_dict()), so without this a
        # changed --lr is silently ignored and the run quietly continues at the old
        # value. base_lrs is what the warmup/decay lambda multiplies, so both must be
        # rewritten or the scheduler would stomp the param group on its next step.
        # Under --optimizer came8bit the live learning rate is --came-lr; --lr belongs to
        # AdamW and is ignored there. Taking self.hparams.lr unconditionally would push
        # the (much larger) AdamW rate into CAME's param groups on every resume.
        # Under --optimizer muon/cmuon there are TWO base rates by design (Muon on 2D hidden
        # weights, AdamW on norms/biases/embeddings), so the target is resolved PER
        # GROUP off its boomer_role tag. Collapsing both to one value would silently
        # retune half the model on every resume.
        changed = False
        for optim in self.trainer.optimizers:
            for group in optim.param_groups:
                target = self._cli_lr_for_group(group)
                if group.get("initial_lr") is not None and group["initial_lr"] != target:
                    group["initial_lr"] = target
                    changed = True
        for config in self.trainer.lr_scheduler_configs:
            sched = config.scheduler
            base = getattr(sched, "base_lrs", None)
            if not base:
                continue
            # base_lrs is positionally aligned with the optimizer's param_groups.
            desired = [self._cli_lr_for_group(g) for g in sched.optimizer.param_groups]
            if list(base) != desired:
                sched.base_lrs = desired
                changed = True
        if changed:
            mode = getattr(self.hparams, "optimizer", "adamw")
            flag = {
                "came8bit": "--came-lr",
                "muon": "--muon-lr/--muon-aux-lr",
                "cmuon": "--cmuon-lr/--cmuon-aux-lr",
            }.get(mode, "--lr")
            rates = sorted({g["initial_lr"] for o in self.trainer.optimizers
                            for g in o.param_groups if g.get("initial_lr") is not None})
            shown = ", ".join(f"{r:g}" for r in rates)
            print(f"[lr] CLI {flag} ({shown}) overrides the resumed optimizer/scheduler "
                  f"learning rate", flush=True)

    def _cli_lr_for_group(self, group: dict) -> float:
        """The learning rate the CLI asks for, for one param group.

        --lr belongs to AdamW. came8bit runs on --came-lr and must never inherit --lr
        (it is ~5x too large). muon runs two groups at once, distinguished by the
        boomer_role tag its optimizer stamps on each group.
        """
        hp = self.hparams
        mode = getattr(hp, "optimizer", "adamw")
        if mode == "came8bit":
            return float(hp.came_lr)
        if mode == "muon":
            if group.get("boomer_role") == "muon":
                return float(getattr(hp, "muon_lr", None) or hp.lr)
            return float(getattr(hp, "muon_aux_lr", None) or hp.lr)
        if mode == "cmuon":
            if group.get("boomer_role") == "muon":
                base = float(getattr(hp, "cmuon_lr", None) or hp.lr)
                # Paper-default chunk rescaling gives fused QKV/FFN/adaLN groups
                # sqrt(N) times the base rate. Preserve that relative scale when a
                # resumed scheduler is overwritten from the CLI.
                return base * float(group.get("boomer_lr_scale", 1.0))
            return float(getattr(hp, "cmuon_aux_lr", None) or hp.lr)
        return float(hp.lr)

    def configure_optimizers(self):
        hp = self.hparams
        if getattr(hp, "optimizer", "adamw") == "came8bit":
            from boomer.optim import CAME8Bit, adamw_state_bytes, came_state_bytes

            # CAME's step is RMS-clipped and bias-correction-free, so its natural step
            # size is roughly an order of magnitude under AdamW's. Silently reusing --lr
            # would look like it trained and quietly diverge, so require the LR to be
            # stated rather than inherited.
            came_lr = getattr(hp, "came_lr", None)
            if not came_lr:
                raise ValueError(
                    "--optimizer came8bit requires an explicit --came-lr. CAME's update is "
                    "RMS-clipped to clip_threshold and carries no bias correction, so an "
                    f"AdamW learning rate (--lr {hp.lr:g}) is far too large for it. SANA's "
                    "came8bit configs use 2e-5; start there."
                )
            optim = CAME8Bit(
                self.model.parameters(),
                lr=float(came_lr),
                weight_decay=hp.weight_decay,
                block_size=int(getattr(hp, "came_block_size", 2048)),
                min_8bit_size=int(getattr(hp, "came_min_8bit_size", 16384)),
            )
            gib = 1024 ** 3
            print(
                f"[optim] CAME8bit lr={float(came_lr):g} (AdamW --lr {hp.lr:g} ignored) | "
                f"optimizer state {came_state_bytes(self.model) / gib:.2f} GiB vs "
                f"{adamw_state_bytes(self.model) / gib:.2f} GiB for fp32 AdamW",
                flush=True,
            )
        elif getattr(hp, "optimizer", "adamw") == "muon":
            from boomer.optim import (
                MuonWithAuxAdamW,
                adamw_state_bytes,
                muon_state_bytes,
            )

            # match_rms_adamw makes the AdamW rate transfer, so defaulting to --lr is
            # correct here — the opposite of came8bit, which refuses to inherit it.
            muon_lr = float(getattr(hp, "muon_lr", None) or hp.lr)
            aux_lr = float(getattr(hp, "muon_aux_lr", None) or hp.lr)
            optim = MuonWithAuxAdamW(
                self.model,
                lr=muon_lr,
                aux_lr=aux_lr,
                weight_decay=float(getattr(hp, "muon_weight_decay", 0.1)),
                aux_weight_decay=hp.weight_decay,
                aux_betas=(hp.beta1, hp.beta2),
                momentum=float(getattr(hp, "muon_momentum", 0.95)),
                ns_steps=int(getattr(hp, "muon_ns_steps", 5)),
                adjust_lr_fn=str(getattr(hp, "muon_adjust_lr_fn", "match_rms_adamw")),
            )
            gib = 1024 ** 3
            print(
                f"[optim] Muon lr={muon_lr:g} wd={optim.defaults['weight_decay']:g} "
                f"adjust_lr_fn={optim.defaults['adjust_lr_fn']} on "
                f"{optim.muon_params / 1e6:.1f}M params | AdamW lr={aux_lr:g} on "
                f"{optim.aux_params / 1e6:.1f}M (norms/biases/embeddings/heads) | "
                f"optimizer state {muon_state_bytes(self.model) / gib:.2f} GiB vs "
                f"{adamw_state_bytes(self.model) / gib:.2f} GiB for fp32 AdamW",
                flush=True,
            )
        elif getattr(hp, "optimizer", "adamw") == "cmuon":
            from boomer.optim import (
                CMuonWithAuxAdamW,
                adamw_state_bytes,
                cmuon_state_bytes,
            )

            cmuon_lr = float(getattr(hp, "cmuon_lr", None) or hp.lr)
            aux_lr = float(getattr(hp, "cmuon_aux_lr", None) or hp.lr)
            optim = CMuonWithAuxAdamW(
                self.model,
                lr=cmuon_lr,
                aux_lr=aux_lr,
                weight_decay=float(getattr(hp, "cmuon_weight_decay", 0.0)),
                aux_weight_decay=hp.weight_decay,
                aux_betas=(hp.beta1, hp.beta2),
                momentum=float(getattr(hp, "cmuon_momentum", 0.95)),
                ns_steps=int(getattr(hp, "cmuon_ns_steps", 6)),
                chunk_rescale=bool(getattr(hp, "cmuon_chunk_rescale", True)),
            )
            gib = 1024 ** 3
            print(
                f"[optim] CMuon (Dion) base_lr={cmuon_lr:g} "
                f"wd={optim.defaults['weight_decay']:g} ns_steps="
                f"{optim.defaults['ns_steps']} chunk_rescale="
                f"{optim.defaults['chunk_rescale']} | "
                f"{optim.chunked_params / 1e6:.1f}M params in "
                f"{optim.chunked_matrices} semantic fused matrices, "
                f"{(optim.muon_params - optim.chunked_params) / 1e6:.1f}M ordinary "
                f"Muon params | AdamW lr={aux_lr:g} on "
                f"{optim.aux_params / 1e6:.1f}M | optimizer state "
                f"{cmuon_state_bytes(self.model) / gib:.2f} GiB vs "
                f"{adamw_state_bytes(self.model) / gib:.2f} GiB for fp32 AdamW",
                flush=True,
            )
            if optim.dion_patches:
                # Absence of this line on a fresh Colab means the patch did not apply
                # and the run will die at the first optimizer step, as it did before.
                shown = ", ".join(f"{k} ({v})" for k, v in sorted(optim.dion_patches.items()))
                print(f"[optim] patched Dion kernels: {shown}", flush=True)
        else:
            optim = torch.optim.AdamW(
                self.model.parameters(),
                lr=hp.lr,
                weight_decay=hp.weight_decay,
                betas=(hp.beta1, hp.beta2),
                fused=hp.fused_adamw,
            )
        warmup = max(0, int(hp.warmup_steps))
        sched_name = getattr(hp, "lr_scheduler", "linear-warmup-constant")
        min_lr_ratio = float(getattr(hp, "min_lr_ratio", 0.1))
        total_steps = max(int(getattr(hp, "max_steps", 0)), 1)

        def lr_lambda(step: int) -> float:
            # "constant": flat (no warmup), matching overfit_latent_dit.
            if sched_name == "constant":
                return 1.0
            # warmup ramp (shared by the warmup-bearing schedules)
            if warmup > 0 and step < warmup:
                return (step + 1) / warmup
            if sched_name == "linear-warmup-cosine":
                decay_steps = max(total_steps - warmup, 1)
                progress = min(max((step - warmup) / decay_steps, 0.0), 1.0)
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
            # default "linear-warmup-constant": warmup then hold (legacy train.py behavior)
            return 1.0

        scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)
        return {"optimizer": optim, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    def configure_gradient_clipping(self, optimizer, gradient_clip_val=None, gradient_clip_algorithm=None):
        # Lightning's AMP plugin refuses automatic clipping with a FUSED optimizer (it
        # unscales internally). Under bf16-mixed there is no grad scaler, so a manual
        # norm-clip on the raw grads is correct and lets us keep fused AdamW.
        if gradient_clip_val and gradient_clip_val > 0:
            # clip_grad_norm_ returns the total norm BEFORE clipping -- log it so you can
            # see grad-norm spikes and how often clipping is active (norm > clip_val).
            total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(gradient_clip_val))
            if hasattr(total_norm, "full_tensor"):
                # FSDP2: grads are DTensors, so the clip's total norm is a DTensor too
                # (the clip itself is correct — torch reduces the partial norms across
                # the mesh); materialize it for logging.
                total_norm = total_norm.full_tensor()
            self.log("train/grad_norm", total_norm, on_step=True, on_epoch=False, prog_bar=True)

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        # Explicit non_blocking H2D copies. The DataLoader pins memory (pin_memory=True),
        # so pinned-CPU -> CUDA transfers can overlap the previous step's compute on a
        # separate copy stream instead of stalling the main stream. Lightning only applies
        # non_blocking opportunistically, so we do it explicitly for every tensor.
        if isinstance(batch, dict):
            return {
                k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
        return super().transfer_batch_to_device(batch, device, dataloader_idx)
