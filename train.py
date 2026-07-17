"""PyTorch Lightning trainer for Boomer FLA (from-scratch, Qwen-conditioned).

A thin Lightning wrapper over the same building blocks as tools/overfit_latent_dit.py:
  - model:   boomer.models.factory.create_model(BoomerFLADiTConfig)
  - data:    boomer.datasets.wds_latents.RollingStreamingWdsLatentCacheDataset
  - flow:    boomer.sana_flow (sigma sampling, min-SNR / Sana loss weights)
  - cond:    Qwen hidden states fed directly into the DiT's caption_embedder (Sana /
             Qwen-Image style); CFG via the model's learned null + --cond-dropout.

text_dim / text_seq_len / latent shape are auto-detected from one cached record.
Reserved inpainting channels are supported via --extra-input-channels (zero-fed here).

Example (from scratch, 701M GDN-2 on the PD12M Qwen cache):

    python3 train.py \
      --cache-wds "hf://datasets/akrao9/pd12m-boomer-qwen-cache/chunk_*/*.tar" \
      --wds-local-dir /content/_wds_cache --wds-rolling-shards 2 --wds-shuffle-buffer 2000 \
      --hidden-dim 896 --depth 24 --num-heads 14 --mlp-ratio 2.5 \
      --mixer-type fla_gdn2 --fla-bidirectional --use-short-conv \
      --image-attn-every 6 --image-attn-qk-rmsnorm --cross-attn-qk-rmsnorm --extra-input-channels 33 \
      --batch-size 64 --lr 1e-4 --cond-dropout 0.1 \
      --t-sampler plateau-logit-normal --flow-shift 1.5 --min-snr-gamma 5 \
      --ema-decay 0.9999 --max-steps 100000 \
      --checkpoint-dir /content/ckpt_stage0 --hf-token true

Direct-monet mode (--monet-direct): no cache build. --cache-wds points at raw
jasperai/monet parquet globs; the dataset's precomputed SANA f32c32 latents are
used as-is and captions are encoded on-GPU every step by a resident frozen Qwen
encoder (default Qwen/Qwen3.5-4B, hidden_states layer -2):

    python3 train.py --monet-direct \
      --cache-wds "hf://datasets/jasperai/monet/v1.2.0/cc12m/*/*.parquet,hf://datasets/jasperai/monet/v1.2.0/laion/*/*.parquet" \
      --text-encoder-id Qwen/Qwen3.5-4B --text-encoder-layer -2 --text-encoder-max-length 300 \
      --monet-min-aesthetic 4.5 --sample-prompts-file eval_prompts.txt \
      ... (model/optim flags as usual; text dims are auto-set from the encoder)

Multi-GPU (8x H200, single node) with FSDP2 sharding, subset mixing and per-
subset caption sampling (nano_mix.yaml template: synthetic 0.5 with its true
generating prompt caption_original@1.0, five real subsets 0.1 each sampling the
4 captioners uniformly and skipping the noisy alt-text caption_original):

    python3 train.py --monet-direct --fsdp2 --devices 8 \
      --cache-wds "hf://datasets/jasperai/monet/v1.2.0/synthetic/*/*/*.parquet,hf://datasets/jasperai/monet/v1.2.0/cc12m/*/*.parquet,hf://datasets/jasperai/monet/v1.2.0/megalith10m/*/*.parquet,hf://datasets/jasperai/monet/v1.2.0/laion/*/*.parquet,hf://datasets/jasperai/monet/v1.2.0/coyo/*/*.parquet,hf://datasets/jasperai/monet/v1.2.0/commoncatalog-cc-by/*/*.parquet" \
      --monet-weights "0.5,0.1,0.1,0.1,0.1,0.1" \
      --monet-caption-probs "0,0,0,0,1;.25,.25,.25,.25,0;.25,.25,.25,.25,0;.25,.25,.25,.25,0;.25,.25,.25,.25,0;.25,.25,.25,.25,0" \
      --checkpoint-dir /ckpt_local \
      --checkpoint-upload-remote hf://buckets/akrao9/Boomerv6-ckpt \
      ... (per-GPU --batch-size; global batch = batch_size * devices)

Checkpoints save to --checkpoint-dir (local NVMe) synchronously, then a rank-0
background thread uploads them to --checkpoint-upload-remote — an HF storage
bucket via the hf CLI (`hf sync`), an rclone remote, or a mounted directory —
and prunes local disk to the newest --checkpoint-local-retain once uploaded.
"""

from __future__ import annotations

import argparse
import math
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:  # Lightning >= 2.0 namespace, with legacy fallback.
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import (
        Callback,
        LearningRateMonitor,
        ModelCheckpoint,
        TQDMProgressBar,
    )
    from lightning.pytorch.loggers import WandbLogger
except ImportError:  # pragma: no cover
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import (
        Callback,
        LearningRateMonitor,
        ModelCheckpoint,
        TQDMProgressBar,
    )
    from pytorch_lightning.loggers import WandbLogger

from boomer.datasets.latent_shards import collate_latent_records  # noqa: E402
from boomer.datasets.monet_direct import (  # noqa: E402
    MONET_CAPTION_KEYS,
    MONET_DINO_KEY,
    MONET_LATENT_KEY,
    MonetDirectStreamingDataset,
    collate_monet_records,
)
from boomer.datasets.wds_latents import (  # noqa: E402
    ParquetStreamingLatentCacheDataset,
    PrefetchingWdsLatentCacheDataset,
    StreamingWdsLatentCacheDataset,
)
from boomer.models.qwen_connector import (  # noqa: E402
    QwenConnectorConfig,
    QwenReferenceEncoder,
    encode_qwen_text_fixed,
)
from boomer.models.boomer_fla_dit import (  # noqa: E402
    BoomerFLADiTConfig,
    migrate_glumbconv_state_dict,
)
from boomer.models.factory import create_model  # noqa: E402
from boomer.sana_flow import (  # noqa: E402
    flow_min_snr_loss_weight,
    flow_training_loss_weight,
    noise_latent_at_sigma,
    sample_flow_sigmas,
)
from boomer.datasets.edit_pairs import (  # noqa: E402
    EditPairsStreamingDataset,
    collate_edit_records,
)
from boomer.fill_masks import parse_fill_mask_probs, sample_fill_masks  # noqa: E402
from boomer.latent_norm import denormalize_latent  # noqa: E402
from boomer.sampling import sample_latent  # noqa: E402
from boomer.sana_latent_cache import (  # noqa: E402
    DEFAULT_SCALING_FACTOR,
    DEFAULT_VAE_ID,
    CacheConfig,
    decode_image_latent,
    extract_latent,
    hf_token_value,
    load_vae,
    tensor_to_image,
    vae_scaling_factor,
)


# --------------------------------------------------------------------------- #
# Dispersive loss (https://arxiv.org/abs/2506.09027)
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# LightningModule
# --------------------------------------------------------------------------- #
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
        # Frozen on-the-fly DC-AE (--edit-direct pixel pairs). Same out-of-tree
        # residency rules as the text encoder box above.
        self._vae_box: list[Any] = []
        # --fill-training: mask-type mix, parsed once (fails fast on a bad spec).
        self._fill_mask_probs = (
            parse_fill_mask_probs(hp.get("fill_mask_probs")) if hp.get("fill_training") else None
        )
        if hp.get("compile"):
            self.model = torch.compile(self.model, mode=hp.get("compile_mode", "default"))

    def _resident_text_encoder(self) -> QwenReferenceEncoder:
        if not self._text_encoder_box:
            hp = self.hparams
            cfg = QwenConnectorConfig(
                qwen_model_id=hp.text_encoder_id,
                readout_layer=hp.text_encoder_layer,
                instruction=(
                    QwenConnectorConfig.instruction if hp.text_encoder_instruction else None
                ),
                dtype="bf16",
                max_text_tokens=hp.text_encoder_max_length,
            )
            encoder = QwenReferenceEncoder(cfg, hf_token=self._hf_token)
            encoder.load()
            print(
                f"[text-encoder] {hp.text_encoder_id} resident (hidden={encoder.hidden_size}, "
                f"layer={hp.text_encoder_layer}, max_length={hp.text_encoder_max_length})",
                flush=True,
            )
            self._text_encoder_box.append(encoder)
        encoder = self._text_encoder_box[0]
        param = next(encoder.model.parameters())
        if param.device != self.device:
            encoder.to(self.device)
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
            print(f"[edit-direct] DC-AE {cfg.vae_id} resident (frozen, bf16, "
                  f"scaling={cfg.scaling_factor})", flush=True)
            self._vae_box.append(vae)
        vae = self._vae_box[0]
        if next(vae.parameters()).device != self.device:
            vae = vae.to(self.device)
            self._vae_box[0] = vae
        return vae

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
        parts = []
        for part in pixels.split(chunk):
            images = part.to(device=self.device, dtype=dtype, non_blocking=True)
            images = images / 127.5 - 1.0
            parts.append(extract_latent(vae.encode(images)))
        return torch.cat(parts, dim=0) * scale

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

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        hp = self.hparams
        x0 = batch.get("image_latent")
        src_latent = None
        if x0 is None:
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
            # --monet-direct: the batch carries raw captions; encode them on-GPU with
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
        if hp.t_sampler_final:
            self.log("train/t_polish", float(polish), on_step=True, on_epoch=False, batch_size=bs)
        if self._fill_mask_probs is not None:
            # Mean hole fraction across the batch (post mask-mix, pre cond dropout):
            # sanity signal that the mask distribution is what --fill-mask-probs asked for.
            self.log("train/fill_coverage", fill_mask.float().mean(),
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

    def configure_optimizers(self):
        hp = self.hparams
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


# --------------------------------------------------------------------------- #
# DataModule
# --------------------------------------------------------------------------- #
def is_remote_cache(cache_wds: str) -> bool:
    return "hf://" in cache_wds


def is_parquet_cache(cache_wds: str) -> bool:
    """Parquet cache (range-streamed via `datasets`), local or hf://. Takes precedence
    over the tar-shard paths in make_wds_dataset / the worker-count logic."""
    return ".parquet" in cache_wds


def make_wds_dataset(
    *,
    cache_wds: str,
    local_dir: str,
    prefetch_shards: int,
    download_workers: int,
    shuffle_buffer: int,
    hf_token: str | bool | None,
    seed: int,
    decode_queue_size: int = 512,
    shard_shuffle: int = 0,
    download_max_retries: int = 2,
    max_consecutive_skips: int = 8,
    parquet_read_batch_size: int = 32,
    parquet_memory_trim_every: int = 4096,
    parquet_prefetch_ranges: bool = True,
    parquet_range_size_mib: int = 64,
    captions_only: bool = False,
):
    """Parquet caches range-stream via `datasets` (faster + stall-resilient); remote
    hf:// tar caches use the parallel download-ahead prefetch window (thread-pool into
    local_dir, delete after use); local tar globs read directly with plain streaming."""
    if is_parquet_cache(cache_wds):
        return ParquetStreamingLatentCacheDataset(
            paths=cache_wds,
            hf_token=hf_token,
            shuffle_buffer=shuffle_buffer,
            seed=seed,
            download_max_retries=download_max_retries,
            read_batch_size=parquet_read_batch_size,
            memory_trim_every=parquet_memory_trim_every,
            prefetch_ranges=parquet_prefetch_ranges,
            range_size_mib=parquet_range_size_mib,
            captions_only=captions_only,
        )
    if is_remote_cache(cache_wds):
        return PrefetchingWdsLatentCacheDataset(
            paths=cache_wds,
            local_dir=local_dir,
            prefetch_shards=prefetch_shards,
            download_workers=download_workers,
            hf_token=hf_token,
            shuffle_buffer=shuffle_buffer,
            seed=seed,
            decode_queue_size=decode_queue_size,
            download_max_retries=download_max_retries,
            max_consecutive_skips=max_consecutive_skips,
            captions_only=captions_only,
        )
    return StreamingWdsLatentCacheDataset(
        paths=cache_wds,
        hf_token=hf_token,
        shardshuffle=shard_shuffle if shard_shuffle > 0 else True,
        shuffle_buffer=shuffle_buffer,
        captions_only=captions_only,
    )


class BoomerData(pl.LightningDataModule):
    def __init__(
        self,
        *,
        cache_wds: str,
        local_dir: str,
        prefetch_shards: int,
        download_workers: int,
        shuffle_buffer: int,
        hf_token: str | bool | None,
        batch_size: int,
        seed: int,
        num_workers: int,
        decode_queue_size: int = 512,
        shard_shuffle: int = 0,
        pin_memory: bool = True,
        prefetch_factor: int = 2,
        persistent_workers: bool | None = None,
        download_max_retries: int = 2,
        max_consecutive_skips: int = 8,
        parquet_read_batch_size: int = 32,
        parquet_memory_trim_every: int = 4096,
        parquet_prefetch_ranges: bool = True,
        parquet_range_size_mib: int = 64,
        captions_only: bool = False,
        monet_direct: bool = False,
        monet_latent_key: str = MONET_LATENT_KEY,
        monet_caption_keys: tuple[str, ...] = MONET_CAPTION_KEYS,
        monet_caption_probs: Any = None,  # CSV / ';'-grouped str, tuple, or None
        monet_weights: tuple[float, ...] | None = None,
        monet_dino_key: str | None = MONET_DINO_KEY,
        monet_min_aesthetic: float | None = None,
        monet_max_nsfw: float | None = None,
        monet_max_watermark: float | None = None,
        edit_direct: bool = False,
        edit_resolution: int = 1024,
        edit_min_o_score: float | None = None,
        edit_tasks: tuple[str, ...] | None = None,
        edit_prompt_pick: str = "random",
        edit_min_short_side: int = 0,
    ) -> None:
        super().__init__()
        self.captions_only = captions_only
        self.monet_direct = monet_direct
        self.edit_direct = edit_direct
        self.edit_resolution = edit_resolution
        self.edit_min_o_score = edit_min_o_score
        self.edit_tasks = edit_tasks
        self.edit_prompt_pick = edit_prompt_pick
        self.edit_min_short_side = edit_min_short_side
        self.monet_latent_key = monet_latent_key
        self.monet_caption_keys = monet_caption_keys
        self.monet_caption_probs = monet_caption_probs
        self.monet_weights = monet_weights
        self.monet_dino_key = monet_dino_key
        self.monet_min_aesthetic = monet_min_aesthetic
        self.monet_max_nsfw = monet_max_nsfw
        self.monet_max_watermark = monet_max_watermark
        self.cache_wds = cache_wds
        self.local_dir = local_dir
        self.prefetch_shards = prefetch_shards
        self.download_workers = download_workers
        self.shuffle_buffer = shuffle_buffer
        self.hf_token = hf_token
        self.batch_size = batch_size
        self.seed = seed
        self.num_workers = num_workers
        self.decode_queue_size = decode_queue_size
        self.shard_shuffle = shard_shuffle
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers
        self.download_max_retries = download_max_retries
        self.max_consecutive_skips = max_consecutive_skips
        self.parquet_read_batch_size = parquet_read_batch_size
        self.parquet_memory_trim_every = parquet_memory_trim_every
        self.parquet_prefetch_ranges = parquet_prefetch_ranges
        self.parquet_range_size_mib = parquet_range_size_mib

    def _resume_seed_salt(self) -> int:
        """Salt the stream seed with the resumed global step so a restarted run does not
        replay the identical shard order + shuffle from the head of the stream (over many
        Colab preemptions that systematically oversamples early rows). Same salt for every
        worker, preserving the disjoint file sharding; 0 for a fresh run, deterministic
        given the checkpoint."""
        try:
            if self.trainer is not None:
                return int(self.trainer.global_step)
        except RuntimeError:
            pass  # not attached to a trainer (direct/offline use)
        return 0

    def _trainer_rank_world(self) -> tuple[int | None, int | None]:
        """The Trainer's DDP rank/world, resolved HERE (main process, trainer attached)
        and passed to the dataset explicitly — spawned DataLoader workers have no
        initialized process group to ask."""
        try:
            if self.trainer is not None:
                return int(self.trainer.global_rank), int(self.trainer.world_size)
        except RuntimeError:
            pass  # not attached to a trainer (direct/offline use)
        return None, None

    def _dataset(self, *, seed_salt: int = 0):
        if self.edit_direct:
            rank, world = self._trainer_rank_world()
            return EditPairsStreamingDataset(
                paths=self.cache_wds,
                hf_token=self.hf_token,
                shuffle_buffer=self.shuffle_buffer,
                seed=self.seed + seed_salt,
                download_max_retries=self.download_max_retries,
                read_batch_size=self.parquet_read_batch_size,
                memory_trim_every=self.parquet_memory_trim_every,
                prefetch_ranges=self.parquet_prefetch_ranges,
                range_size_mib=self.parquet_range_size_mib,
                resolution=self.edit_resolution,
                min_o_score=self.edit_min_o_score,
                tasks=self.edit_tasks,
                prompt_pick=self.edit_prompt_pick,
                min_short_side=self.edit_min_short_side,
                glob_weights=self.monet_weights,
                rank=rank,
                world_size=world,
            )
        if self.monet_direct:
            rank, world = self._trainer_rank_world()
            return MonetDirectStreamingDataset(
                paths=self.cache_wds,
                hf_token=self.hf_token,
                shuffle_buffer=self.shuffle_buffer,
                seed=self.seed + seed_salt,
                download_max_retries=self.download_max_retries,
                read_batch_size=self.parquet_read_batch_size,
                memory_trim_every=self.parquet_memory_trim_every,
                prefetch_ranges=self.parquet_prefetch_ranges,
                range_size_mib=self.parquet_range_size_mib,
                latent_key=self.monet_latent_key,
                caption_keys=self.monet_caption_keys,
                caption_probs=self.monet_caption_probs,
                glob_weights=self.monet_weights,
                dino_key=self.monet_dino_key,
                min_aesthetic=self.monet_min_aesthetic,
                max_nsfw=self.monet_max_nsfw,
                max_watermark=self.monet_max_watermark,
                rank=rank,
                world_size=world,
            )
        return make_wds_dataset(
            cache_wds=self.cache_wds,
            local_dir=self.local_dir,
            prefetch_shards=self.prefetch_shards,
            download_workers=self.download_workers,
            shuffle_buffer=self.shuffle_buffer,
            hf_token=self.hf_token,
            seed=self.seed + seed_salt,
            decode_queue_size=self.decode_queue_size,
            shard_shuffle=self.shard_shuffle,
            download_max_retries=self.download_max_retries,
            max_consecutive_skips=self.max_consecutive_skips,
            parquet_read_batch_size=self.parquet_read_batch_size,
            parquet_memory_trim_every=self.parquet_memory_trim_every,
            parquet_prefetch_ranges=self.parquet_prefetch_ranges,
            parquet_range_size_mib=self.parquet_range_size_mib,
            captions_only=self.captions_only,
        )

    def train_dataloader(self) -> DataLoader:
        # The rolling remote TAR cache REQUIRES num_workers=0 (single-process shard
        # download). Parquet range-streaming and local globs use multiple workers
        # (datasets shards parquet files across workers automatically).
        parquet = self.monet_direct or self.edit_direct or is_parquet_cache(self.cache_wds)
        if parquet:
            workers = self.num_workers
        elif is_remote_cache(self.cache_wds):
            workers = 0
        else:
            workers = self.num_workers

        # DataLoader spawn workers inherit the environment present when iteration
        # begins. Parquet wants cancellable HTTP Range reads. Xet's reconstruction
        # path has an 8 GiB default buffer limit per process (64 GiB in HP mode), so
        # six workers can consume tens of GiB and wait minutes before yielding row 1.
        # Set both flags before spawn/import so Hugging Face actually selects HTTP.
        if parquet:
            if os.environ.get("HF_HUB_DISABLE_XET") != "1":
                print(
                    "[parquet-wds] disabling Xet reconstruction; using HTTP Range reads",
                    flush=True,
                )
            os.environ["HF_HUB_DISABLE_XET"] = "1"
            os.environ["HF_XET_HIGH_PERFORMANCE"] = "0"
            # Applied when spawn starts fresh worker interpreters. This bounds glibc's
            # per-thread arenas, which otherwise retain large Arrow/Python allocations.
            os.environ.setdefault("MALLOC_ARENA_MAX", "2")
        seed_salt = self._resume_seed_salt()
        if seed_salt:
            print(
                f"[data] resume detected (global_step={seed_salt}); salting stream seed "
                f"{self.seed} -> {self.seed + seed_salt} so the restarted stream does not "
                "replay the same head-of-stream rows",
                flush=True,
            )
        persistent = self.persistent_workers if self.persistent_workers is not None else workers > 0
        kwargs = {}
        if workers > 0:
            # Each worker stages prefetch_factor batches ahead so the pinned-memory +
            # non_blocking H2D copy always has something ready, overlapping decode with compute.
            kwargs["prefetch_factor"] = self.prefetch_factor
            if parquet:
                # FORK-SAFETY: the parquet loader touches HF (HfFileSystem glob / datasets
                # streaming) which leaves live fsspec/`requests` session + connection-pool
                # locks in this process. Default fork() workers inherit that half-initialized
                # state and DEADLOCK on their first range read (silent hang). `spawn` starts
                # each worker as a fresh interpreter that builds its own HF session, so there
                # is nothing poisoned to inherit. The dataset pickles cleanly (manifest dict +
                # str globs + token), and env (HF_TOKEN) propagates to spawned children.
                import multiprocessing as _mp

                kwargs["multiprocessing_context"] = _mp.get_context("spawn")
        return DataLoader(
            self._dataset(seed_salt=seed_salt),
            batch_size=self.batch_size,
            num_workers=workers,
            collate_fn=(
                collate_edit_records if self.edit_direct
                else collate_monet_records if self.monet_direct
                else collate_latent_records
            ),
            pin_memory=self.pin_memory,
            drop_last=True,
            persistent_workers=persistent and workers > 0,
            **kwargs,
        )


# --------------------------------------------------------------------------- #
# EMA callback (no native Lightning EMA)
# --------------------------------------------------------------------------- #
class EMACallback(Callback):
    def __init__(self, decay: float = 0.9999, update_every: int = 1) -> None:
        self.decay = decay
        self.update_every = max(1, update_every)
        self.ema: dict[str, torch.Tensor] = {}

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not self.ema:  # fresh run (not resumed)
            self.ema = {
                k: v.detach().clone().float()
                for k, v in pl_module.model.state_dict().items()
                if torch.is_floating_point(v)
            }

    @torch.no_grad()
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if trainer.global_step % self.update_every != 0:
            return
        sd = pl_module.model.state_dict()
        for key, ema_v in self.ema.items():
            target = sd[key].detach().float()
            if ema_v.device != target.device:
                # On resume the EMA shadow is restored on CPU while the live model is
                # already on GPU; realign once (later steps are a cheap device compare).
                ema_v = ema_v.to(target.device)
                self.ema[key] = ema_v
            ema_v.mul_(self.decay).add_(target, alpha=1.0 - self.decay)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint) -> None:
        checkpoint["ema"] = self.ema

    def on_load_checkpoint(self, trainer, pl_module, checkpoint) -> None:
        if "ema" in checkpoint:
            ema = {k: v.float() for k, v in checkpoint["ema"].items()}
            # The module-level GLUMBConv migration only covers Module.load_state_dict;
            # the EMA shadow is a raw dict and must be migrated explicitly or the first
            # update after resuming a pre-refactor checkpoint KeyErrors.
            n = migrate_glumbconv_state_dict(ema)
            if n:
                print(f"[ema] migrated {n} legacy GLUMBConv entr{'y' if n == 1 else 'ies'}",
                      flush=True)
            self.ema = ema


class StepUnitProgressBar(TQDMProgressBar):
    """Progress bar in OPTIMIZER-STEP units: global_step / max_steps.

    Lightning's default bar counts micro-batches, and for our infinite streaming
    caches it derives the total from max_steps -- which is in optimizer steps, a
    unit mismatch under --accumulate > 1 (upstream fix: Lightning PR #21802). Rather
    than reconciling both sides to batch units, display the run in the units every
    other knob here already uses (warmup, anneals, save cadence, max-steps): the
    numerator is global_step, the denominator max_steps, and it/s means optimizer
    steps/s. global_step is restored across ckpt resume, so a resumed session shows
    absolute progress (e.g. 40000/100000); `initial` is anchored at session start so
    tqdm's rate/ETA reflect only this session's throughput, not the pre-resume jump.
    """

    @property
    def total_train_batches(self) -> int | float:
        if self.trainer.max_steps is not None and self.trainer.max_steps > 0:
            return self.trainer.max_steps
        return super().total_train_batches

    def on_train_epoch_start(self, trainer, *args) -> None:
        super().on_train_epoch_start(trainer, *args)
        self.train_progress_bar.initial = trainer.global_step
        self.train_progress_bar.n = trainer.global_step

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        n = trainer.global_step
        if self._should_update(n, self.train_progress_bar.total):
            self.train_progress_bar.n = n
            self.train_progress_bar.refresh()
            self.train_progress_bar.set_postfix(self.get_metrics(trainer, pl_module))


# --------------------------------------------------------------------------- #
# In-training sample previews -> W&B
# --------------------------------------------------------------------------- #
class SamplePreviewCallback(Callback):
    """Every N optimizer steps, sample a fixed set of prompts and log the images to W&B.

    Designed to add ~zero steady-state VRAM and never disturb training:
      * Reuses the resident DiT — no second model copy.
      * Samples on the UNCOMPILED module (``model._orig_mod``), so torch.compile's
        training graph is never re-traced for the inference shapes.
      * Text is PRE-encoded (see tools/encode_prompts.py), so Qwen is never loaded
        here; only the small DC-AE decoder is touched, and it's parked on CPU between
        previews and moved to GPU just for the decode.
      * Optionally swaps in EMA weights for the preview, then restores the live weights
        (backup kept on CPU to avoid a GPU spike).
      * The whole preview is wrapped in try/except: a preview failure (e.g. a transient
        OOM) logs a warning and training continues — a preview must never kill a run.
    """

    def __init__(
        self,
        *,
        every_n_steps: int,
        prompts_cache: str,
        ema_callback: "EMACallback | None",
        use_ema: bool,
        steps: int,
        sampler: str,
        stork_substeps: int,
        cfg_scale: float,
        cfg_rescale: float,
        cfg_interval: tuple[float, float],
        flow_shift: float,
        latent_mean: float,
        latent_std: float,
        seed: int,
        dtype: str,
        vae_id: str,
        scaling_factor: float,
        max_length: int,
        hf_token: str | bool | None,
        encode_on_the_fly: bool = False,
        prompts_file: str | None = None,
    ) -> None:
        self.every_n_steps = every_n_steps
        self.prompts_cache = prompts_cache
        # --monet-direct: pre-encoded prompt caches were built with the OLD encoder
        # (wrong dim/layer), so only prompt STRINGS are loaded and the resident
        # training encoder embeds them lazily at the first preview.
        self.encode_on_the_fly = encode_on_the_fly
        self.prompts_file = prompts_file
        self.ema_callback = ema_callback
        self.use_ema = use_ema
        self.steps = steps
        self.sampler = sampler
        self.stork_substeps = stork_substeps
        self.cfg_scale = cfg_scale
        self.cfg_rescale = cfg_rescale
        self.cfg_interval = cfg_interval
        self.flow_shift = flow_shift
        self.latent_mean = latent_mean
        self.latent_std = latent_std
        self.seed = seed
        self.dtype = dtype
        self.hf_token = hf_token
        self._enabled = False
        self._last_preview_step = -1
        self._prompts: list[str] = []
        self._text_embeddings: torch.Tensor | None = None
        self._attention_masks: torch.Tensor | None = None
        self._vae = None
        self._cache_config = CacheConfig(
            vae_id=vae_id,
            max_length=max_length,
            dtype=dtype,
            device="cuda",
            scaling_factor=scaling_factor,
        )

    def on_fit_start(self, trainer, pl_module) -> None:
        if trainer.global_rank != 0:
            return
        if not isinstance(trainer.logger, WandbLogger):
            print("[sample-preview] no WandbLogger; previews disabled.", flush=True)
            return
        if self.encode_on_the_fly:
            if self.prompts_file and Path(self.prompts_file).is_file():
                lines = Path(self.prompts_file).read_text(encoding="utf-8").splitlines()
                self._prompts = [line.strip() for line in lines if line.strip()]
            elif Path(self.prompts_cache).is_file():
                # Reuse only the prompt strings; the cached embeddings belong to the
                # cache-era encoder and would be the wrong dim/representation.
                payload = torch.load(self.prompts_cache, map_location="cpu", weights_only=False)
                self._prompts = list(payload["prompts"])
            if not self._prompts:
                print(f"[sample-preview] no prompts found ({self.prompts_file or self.prompts_cache}); "
                      "previews disabled. Pass --sample-prompts-file with one prompt per line.",
                      flush=True)
                return
            self._enabled = True
            print(f"[sample-preview] {len(self._prompts)} prompts (encoded on the fly by the "
                  f"resident text encoder), every {self.every_n_steps} steps "
                  f"({'EMA' if self.use_ema else 'live'} weights, {self.sampler}, {self.steps} steps).",
                  flush=True)
            return
        path = Path(self.prompts_cache)
        if not path.is_file():
            print(f"[sample-preview] prompts cache not found ({path}); previews disabled. "
                  f"Run tools/encode_prompts.py --out {path} first.", flush=True)
            return
        payload = torch.load(path, map_location="cpu", weights_only=False)
        self._prompts = list(payload["prompts"])
        self._text_embeddings = payload["text_embeddings"]
        self._attention_masks = payload["attention_masks"]
        self._enabled = True
        print(f"[sample-preview] {len(self._prompts)} prompts, every {self.every_n_steps} steps "
              f"({'EMA' if self.use_ema else 'live'} weights, {self.sampler}, {self.steps} steps).",
              flush=True)

    @staticmethod
    def _strip_compile_prefix(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        prefix = "_orig_mod."
        return {(k[len(prefix):] if k.startswith(prefix) else k): v for k, v in state.items()}

    @torch.no_grad()
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if not self._enabled:
            return
        step = trainer.global_step
        # on_train_batch_end fires once per MICRO-batch, but global_step advances
        # only every accumulate_grad_batches steps -- so at an interval boundary the
        # step-modulo is true for all A micro-batches of the window. Dedup so the
        # preview runs exactly once per optimizer step.
        if step == 0 or step % self.every_n_steps != 0 or step == self._last_preview_step:
            return
        self._last_preview_step = step
        try:
            self._run_preview(trainer, pl_module, step)
        except Exception as exc:  # noqa: BLE001 - a preview must never kill training
            print(f"[sample-preview] step {step} failed ({type(exc).__name__}: {exc}); "
                  "skipping and continuing training.", flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _run_preview(self, trainer, pl_module, step: int) -> None:
        import wandb

        if self.encode_on_the_fly and self._text_embeddings is None:
            # One-time: the encoder is frozen and the prompts fixed, so encode once
            # and keep the embeddings on CPU (sliced back to GPU per prompt below).
            emb, mask = pl_module.encode_captions(self._prompts)
            self._text_embeddings = emb.cpu()
            self._attention_masks = mask.cpu()

        device = str(pl_module.device)
        device_type = "cuda" if "cuda" in device else "cpu"
        amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[self.dtype]
        # Sample on the eager module so torch.compile's training graph isn't re-traced.
        eager = getattr(pl_module.model, "_orig_mod", pl_module.model)
        was_training = eager.training
        eager.eval()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Optionally swap in EMA weights (backup the live weights on CPU to avoid a spike).
        backup = None
        if self.use_ema and self.ema_callback is not None and self.ema_callback.ema:
            backup = {k: v.detach().to("cpu", copy=True) for k, v in eager.state_dict().items()}
            eager.load_state_dict(self._strip_compile_prefix(self.ema_callback.ema), strict=False)

        try:
            self._ensure_vae(device)
            images, captions = [], []
            for i, prompt in enumerate(self._prompts):
                text_embedding = self._text_embeddings[i : i + 1]
                attention_mask = self._attention_masks[i : i + 1]
                with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=self.dtype != "fp32"):
                    latent = sample_latent(
                        eager,
                        text_embedding,
                        attention_mask,
                        steps=self.steps,
                        seed=self.seed,
                        dtype=torch.float32,
                        device=device,
                        cfg_scale=self.cfg_scale,
                        cfg_rescale=self.cfg_rescale,
                        sampler=self.sampler,
                        flow_shift=self.flow_shift,
                        stork_substeps=self.stork_substeps,
                        cfg_interval=self.cfg_interval,
                    )
                latent = denormalize_latent(latent.float(), self.latent_mean, self.latent_std)
                decoded = decode_image_latent(self._vae, latent, self._cache_config)
                images.append(wandb.Image(tensor_to_image(decoded), caption=prompt))
                captions.append(prompt)
            trainer.logger.experiment.log({"samples": images, "trainer/global_step": step})
            print(f"[sample-preview] logged {len(images)} images at step {step}.", flush=True)
        finally:
            if backup is not None:
                eager.load_state_dict(backup, strict=False)
                del backup
            if was_training:
                eager.train()
            self._park_vae()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _ensure_vae(self, device: str) -> None:
        if self._vae is None:
            self._vae = load_vae(self._cache_config, hf_token=self.hf_token)
        self._vae = self._vae.to(device)

    def _park_vae(self) -> None:
        # Keep the decoder on CPU between previews so it costs ~0 steady-state GPU memory.
        if self._vae is not None:
            self._vae = self._vae.to("cpu")


# --------------------------------------------------------------------------- #
# Fill/edit conditioning previews -> W&B
# --------------------------------------------------------------------------- #
class CondSamplePreviewCallback(Callback):
    """Every N optimizer steps, regenerate a FIXED set of CONDITIONED samples.

    Conditioning material is captured once from the FIRST training batch (zero
    startup cost — no extra stream, no eval-set plumbing):
      * fill mode: the first K clean latents + captions, with K deterministic
        masks synthesized once (seeded) and reused all run — so previews are
        comparable across steps: watch the kept region lock in early and the
        hole sharpen over training.
      * edit mode: the first K source images + instructions; mask is all-ones
        and the source is DC-AE-encoded at preview time.
    Logs [conditioning | generated] pairs under "cond_samples". Same
    never-kill-training / EMA-swap / parked-VAE discipline as SamplePreviewCallback.
    """

    def __init__(
        self,
        *,
        mode: str,  # "fill" | "edit"
        every_n_steps: int,
        count: int,
        ema_callback: "EMACallback | None",
        use_ema: bool,
        steps: int,
        sampler: str,
        stork_substeps: int,
        cfg_scale: float,
        cfg_rescale: float,
        cfg_interval: tuple[float, float],
        flow_shift: float,
        latent_mean: float,
        latent_std: float,
        seed: int,
        dtype: str,
        vae_id: str,
        scaling_factor: float,
        hf_token: str | bool | None,
        fill_mask_probs: dict[str, float] | None = None,
        encode_chunk: int = 8,
        image_cfg_scale: float = 0.0,
    ) -> None:
        self.mode = mode
        self.image_cfg_scale = image_cfg_scale
        self.every_n_steps = every_n_steps
        self.count = max(1, count)
        self.ema_callback = ema_callback
        self.use_ema = use_ema
        self.steps = steps
        self.sampler = sampler
        self.stork_substeps = stork_substeps
        self.cfg_scale = cfg_scale
        self.cfg_rescale = cfg_rescale
        self.cfg_interval = cfg_interval
        self.flow_shift = flow_shift
        self.latent_mean = latent_mean
        self.latent_std = latent_std
        self.seed = seed
        self.dtype = dtype
        self.hf_token = hf_token
        self.fill_mask_probs = fill_mask_probs
        self.encode_chunk = max(1, encode_chunk)
        self._enabled = False
        self._last_preview_step = -1
        self._prompts: list[str] = []
        self._latents: torch.Tensor | None = None  # fill: RAW cache-space (K,C,H,W)
        self._masks: torch.Tensor | None = None  # fill: fixed (K,1,H,W)
        self._pixels: torch.Tensor | None = None  # edit: uint8 (K,3,R,R)
        self._text: torch.Tensor | None = None
        self._text_mask: torch.Tensor | None = None
        self._vae = None
        self._cache_config = CacheConfig(
            vae_id=vae_id, dtype=dtype, device="cuda", scaling_factor=scaling_factor
        )

    def on_fit_start(self, trainer, pl_module) -> None:
        if trainer.global_rank != 0:
            return
        if not isinstance(trainer.logger, WandbLogger):
            print("[cond-preview] no WandbLogger; conditioning previews disabled.", flush=True)
            return
        self._enabled = True
        print(f"[cond-preview] {self.mode} mode: {self.count} fixed conditions from the "
              f"first batch, every {self.every_n_steps} steps "
              f"({'EMA' if self.use_ema else 'live'} weights).", flush=True)

    @torch.no_grad()
    def _capture(self, batch: dict[str, Any]) -> None:
        self._prompts = [str(c) for c in batch["caption"][: self.count]]
        if self.mode == "fill":
            lat = batch["image_latent"][: self.count].detach().float().cpu()
            self._latents = lat
            gen = torch.Generator().manual_seed(self.seed)
            self._masks = sample_fill_masks(
                lat.shape[0], lat.shape[-2], lat.shape[-1],
                probs=self.fill_mask_probs, generator=gen,
            )
            # Cached-text runs carry usable embeddings; captions-only/monet runs
            # leave them None and the resident encoder fills in at first preview.
            text = batch.get("text_embedding")
            if text is not None:
                self._text = text[: self.count].detach().cpu()
                self._text_mask = batch["attention_mask"][: self.count].detach().cpu()
        else:
            self._pixels = batch["source_pixels"][: self.count].detach().cpu()
        coverage = ""
        if self._masks is not None:
            coverage = f", mask coverage {[round(float(m.mean()), 2) for m in self._masks]}"
        print(f"[cond-preview] captured {len(self._prompts)} conditions from the first "
              f"batch{coverage}.", flush=True)

    @torch.no_grad()
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if not self._enabled:
            return
        if not self._prompts:
            try:
                self._capture(batch)
            except Exception as exc:  # noqa: BLE001 - never kill training
                print(f"[cond-preview] capture failed ({type(exc).__name__}: {exc}); "
                      "conditioning previews disabled.", flush=True)
                self._enabled = False
                return
        step = trainer.global_step
        if step == 0 or step % self.every_n_steps != 0 or step == self._last_preview_step:
            return
        self._last_preview_step = step
        try:
            self._run_preview(trainer, pl_module, step)
        except Exception as exc:  # noqa: BLE001 - a preview must never kill training
            print(f"[cond-preview] step {step} failed ({type(exc).__name__}: {exc}); "
                  "skipping and continuing training.", flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _encode_source_latents(self, device: str) -> torch.Tensor:
        """edit mode: source pixels -> RAW cache-space latents via the preview VAE."""
        dtype = next(self._vae.parameters()).dtype
        scale = vae_scaling_factor(self._vae, self._cache_config.scaling_factor)
        parts = []
        for part in self._pixels.split(self.encode_chunk):
            images = part.to(device=device, dtype=dtype) / 127.5 - 1.0
            parts.append(extract_latent(self._vae.encode(images)))
        return torch.cat(parts, dim=0).float() * scale

    def _run_preview(self, trainer, pl_module, step: int) -> None:
        import wandb

        if self._text is None:
            emb, mask = pl_module.encode_captions(self._prompts)
            self._text = emb.cpu()
            self._text_mask = mask.cpu()

        device = str(pl_module.device)
        device_type = "cuda" if "cuda" in device else "cpu"
        amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[self.dtype]
        eager = getattr(pl_module.model, "_orig_mod", pl_module.model)
        was_training = eager.training
        eager.eval()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        backup = None
        if self.use_ema and self.ema_callback is not None and self.ema_callback.ema:
            backup = {k: v.detach().to("cpu", copy=True) for k, v in eager.state_dict().items()}
            eager.load_state_dict(
                SamplePreviewCallback._strip_compile_prefix(self.ema_callback.ema), strict=False
            )

        try:
            self._ensure_vae(device)
            # Build the conditioning batch in the NORMALIZED space the model sees,
            # and a raw-space visual of what the model was shown.
            if self.mode == "fill":
                raw = self._latents.to(device)
                masks = self._masks.to(device=device, dtype=raw.dtype)
                norm = (raw - self.latent_mean) / self.latent_std
                cond_all = torch.cat([norm * (1.0 - masks), masks], dim=1)
                visual_latents = raw * (1.0 - masks)  # raw-space holes decode ~gray
            else:
                raw = self._encode_source_latents(device)
                norm = (raw - self.latent_mean) / self.latent_std
                ones = norm.new_ones(norm.shape[0], 1, *norm.shape[-2:])
                cond_all = torch.cat([norm, ones], dim=1)
                visual_latents = None

            images = []
            for i, prompt in enumerate(self._prompts):
                if visual_latents is not None:
                    shown = decode_image_latent(
                        self._vae, visual_latents[i : i + 1], self._cache_config
                    )
                    cond_pil = tensor_to_image(shown)
                else:
                    from PIL import Image as _PILImage

                    cond_pil = _PILImage.fromarray(self._pixels[i].permute(1, 2, 0).numpy())
                with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=self.dtype != "fp32"):
                    latent = sample_latent(
                        eager,
                        self._text[i : i + 1],
                        self._text_mask[i : i + 1],
                        steps=self.steps,
                        seed=self.seed,
                        dtype=torch.float32,
                        device=device,
                        cfg_scale=self.cfg_scale,
                        cfg_rescale=self.cfg_rescale,
                        sampler=self.sampler,
                        flow_shift=self.flow_shift,
                        stork_substeps=self.stork_substeps,
                        cfg_interval=self.cfg_interval,
                        cond_input=cond_all[i : i + 1],
                        image_cfg_scale=self.image_cfg_scale,
                    )
                latent = denormalize_latent(latent.float(), self.latent_mean, self.latent_std)
                decoded = decode_image_latent(self._vae, latent, self._cache_config)
                images.append(wandb.Image(cond_pil, caption=f"[{self.mode} input] {prompt}"))
                images.append(wandb.Image(tensor_to_image(decoded), caption=prompt))
            trainer.logger.experiment.log({"cond_samples": images, "trainer/global_step": step})
            print(f"[cond-preview] logged {len(self._prompts)} conditioned pairs at step {step}.",
                  flush=True)
        finally:
            if backup is not None:
                eager.load_state_dict(backup, strict=False)
                del backup
            if was_training:
                eager.train()
            self._park_vae()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _ensure_vae(self, device: str) -> None:
        if self._vae is None:
            self._vae = load_vae(self._cache_config, hf_token=self.hf_token)
        self._vae = self._vae.to(device)

    def _park_vae(self) -> None:
        if self._vae is not None:
            self._vae = self._vae.to("cpu")


# --------------------------------------------------------------------------- #
# Async checkpoint upload (local -> Google Drive / rclone, in the background)
# --------------------------------------------------------------------------- #
class CheckpointUploadCallback(Callback):
    """Copy finished local checkpoints to a remote in a BACKGROUND thread, then keep
    only the newest ``local_retain`` checkpoints on local disk.

    Lightning's ModelCheckpoint writes locally (fast, reliable). This callback must be
    placed AFTER ModelCheckpoint in the callbacks list, so by the time its hook runs the
    .ckpt file exists on disk; it then enqueues the file for an async copy. Training is
    never blocked on the upload, and a Drive-mount stall can't corrupt the local save or
    pause the step. Local pruning only deletes checkpoints that have already uploaded, so
    a slow upload never loses a checkpoint.

    ``remote`` forms:
      * ``hf://buckets/<owner>/<bucket>`` — an HF storage bucket, uploaded with the
        ``hf`` CLI (``hf sync``). Each checkpoint is staged alone (hardlink) into a
        temp dir and synced, so one call never re-uploads or deletes earlier ones.
      * an rclone remote (``gdrive:Boomer/ckpts``).
      * a directory path such as a mounted Drive folder or an hf-mount'ed bucket
        (``hf-mount start bucket akrao9/Boomerv6-ckpt ./local`` then ``./local``);
        copied via tmp-file + rename so a reader never sees a partial file (with a
        direct-copy fallback for FUSE mounts that reject rename).
    """

    def __init__(
        self,
        *,
        checkpoint_dir: str,
        remote: str,
        local_retain: int = 2,
        keep_uploaded: int = 0,
        rclone_bin: str = "rclone",
        chunk_size: str = "128M",
        hf_bin: str = "hf",
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.remote = remote
        self.local_retain = max(1, int(local_retain))
        self.keep_uploaded = int(keep_uploaded)  # 0 = keep all uploaded
        self.rclone_bin = rclone_bin
        self.chunk_size = chunk_size
        self.hf_bin = hf_bin
        # hf:// = HF bucket via the hf CLI; rclone remotes look like "name:path";
        # anything else is a plain directory (mount).
        self.use_hf = remote.startswith("hf://")
        self.use_rclone = not self.use_hf and ":" in remote and not os.path.isabs(remote)
        if not self.use_hf and not self.use_rclone:
            remote_dir = Path(remote).resolve()
            local_dir = self.checkpoint_dir.resolve()
            if remote_dir == local_dir or local_dir.is_relative_to(remote_dir):
                raise ValueError(
                    "--checkpoint-upload-remote must be a DIFFERENT directory from "
                    "--checkpoint-dir: checkpoints save to fast local disk first, then "
                    "the background thread copies them to the remote/mount. Saving "
                    "straight into a network mount would block every training step "
                    "on network I/O."
                )
        self._q: queue.Queue = queue.Queue()
        self._uploaded: set[str] = set()       # filenames confirmed on the remote
        self._enqueued: set[str] = set()        # filenames already handed to the worker
        self._lock = threading.Lock()
        self._worker: threading.Thread | None = None
        self._stop = threading.Event()

    # ---- worker plumbing -------------------------------------------------- #
    def on_fit_start(self, trainer, pl_module) -> None:
        if trainer.global_rank != 0 or self._worker is not None:
            return  # rank 0 only; single worker
        if self.use_hf:
            if shutil.which(self.hf_bin) is None:
                # Fail at fit start, not at the first save 2000 steps in.
                raise RuntimeError(
                    f"[ckpt-upload] '{self.hf_bin}' CLI not found for {self.remote}; "
                    "install it first: curl -LsSf https://hf.co/cli/install.sh | bash"
                )
            print(f"[ckpt-upload] hf sync -> {self.remote}", flush=True)
        elif self.use_rclone:
            print(f"[ckpt-upload] rclone -> {self.remote}", flush=True)
        else:
            os.makedirs(self.remote, exist_ok=True)
            print(f"[ckpt-upload] copy -> {self.remote}", flush=True)
        self._worker = threading.Thread(target=self._run, name="ckpt-upload", daemon=True)
        self._worker.start()

    def _run(self) -> None:
        while not (self._stop.is_set() and self._q.empty()):
            try:
                path = self._q.get(timeout=1.0)
            except queue.Empty:
                continue
            try:
                self._upload_one(Path(path))
                with self._lock:
                    self._uploaded.add(Path(path).name)
                self._prune_local()
                if self.keep_uploaded > 0:
                    self._prune_remote()
            except Exception as exc:  # never let an upload error kill the worker
                # Drop it from _enqueued so the next _scan_and_enqueue re-queues the
                # (still-local) checkpoint instead of leaving it stranded until fit end.
                with self._lock:
                    self._enqueued.discard(Path(path).name)
                print(f"[ckpt-upload] FAILED {Path(path).name} (will retry): {exc}", flush=True)
            finally:
                self._q.task_done()

    def _upload_one(self, local: Path) -> None:
        if not local.exists():
            return
        if self.use_hf:
            import tempfile

            # Stage the ONE finished checkpoint alone (hardlink: same fs, no copy
            # cost) and sync that dir, so `hf sync` never walks in-progress files in
            # checkpoint_dir and one upload can't touch previously uploaded ckpts.
            with tempfile.TemporaryDirectory(dir=str(local.parent)) as stage:
                link = Path(stage) / local.name
                try:
                    os.link(str(local), str(link))
                except OSError:  # cross-device / fs without hardlinks
                    shutil.copyfile(str(local), str(link))
                proc = subprocess.run(
                    [self.hf_bin, "sync", stage, self.remote],
                    capture_output=True, text=True,
                )
                if proc.returncode != 0:
                    raise RuntimeError(
                        f"hf sync exited {proc.returncode}: {proc.stderr.strip()[-500:]}"
                    )
        elif self.use_rclone:
            dest = f"{self.remote.rstrip('/')}/{local.name}"
            cmd = [self.rclone_bin, "copyto", str(local), dest,
                   "--drive-chunk-size", self.chunk_size, "--no-traverse"]
            subprocess.run(cmd, check=True, capture_output=True)
        else:
            dest = Path(self.remote) / local.name
            tmp = Path(self.remote) / (local.name + ".uploading")
            try:
                # copyfile, not copy2: FUSE-backed mounts (hf-mount buckets, some
                # Drive mounts) often reject the chmod/utime metadata step.
                shutil.copyfile(str(local), str(tmp))  # copy to tmp...
                os.replace(str(tmp), str(dest))        # ...then atomic swap into place
            except OSError as exc:
                # Mounts without rename support: copy straight to the final name. This
                # is non-atomic, so if the copy is interrupted, delete the partial file
                # rather than leave a truncated boomer-step=N.ckpt that the resume
                # scanner would pick and fail to load; the retry re-uploads it.
                print(f"[ckpt-upload] tmp+rename failed on {self.remote} ({exc}); "
                      f"copying {local.name} directly", flush=True)
                tmp.unlink(missing_ok=True)
                try:
                    shutil.copyfile(str(local), str(dest))
                except BaseException:
                    Path(dest).unlink(missing_ok=True)
                    raise
        print(f"[ckpt-upload] uploaded {local.name}", flush=True)

    # ---- pruning ---------------------------------------------------------- #
    @staticmethod
    def _step_key(name: str) -> tuple[int, str]:
        """Retention order = training-step order parsed from 'boomer-step=N.ckpt'.

        mtime is unreliable (uploads finish out of order; copies reset it) and a
        lexicographic name sort misorders unpadded steps (10000 < 5000), either of
        which makes retention delete the NEWEST checkpoint. Unparseable names sort
        first (pruned before any step checkpoint)."""
        m = re.search(r"step=(\d+)", name)
        return (int(m.group(1)) if m else -1, name)

    def _local_step_ckpts(self) -> list[Path]:
        # All step checkpoints except the rolling 'last.ckpt' (which is overwritten).
        return sorted(
            (p for p in self.checkpoint_dir.glob("*.ckpt") if p.name != "last.ckpt"),
            key=lambda p: self._step_key(p.name),
        )

    def _prune_local(self) -> None:
        ckpts = self._local_step_ckpts()
        # keep the newest local_retain; delete older ones ONLY if already uploaded.
        for p in ckpts[: max(0, len(ckpts) - self.local_retain)]:
            with self._lock:
                done = p.name in self._uploaded
            if done:
                try:
                    p.unlink()
                    print(f"[ckpt-upload] pruned local {p.name}", flush=True)
                except FileNotFoundError:
                    pass

    def _prune_remote(self) -> None:
        if self.use_hf:
            # No known bucket-side delete verb in the hf CLI yet; keep every
            # uploaded checkpoint in the bucket (announce once).
            if not getattr(self, "_hf_prune_warned", False):
                self._hf_prune_warned = True
                print(
                    f"[ckpt-upload] --checkpoint-keep-uploaded is not supported for "
                    f"hf:// buckets; keeping all uploads in {self.remote}",
                    flush=True,
                )
            return
        if self.use_rclone:
            out = subprocess.run([self.rclone_bin, "lsf", self.remote],
                                 capture_output=True, text=True)
            names = sorted(
                (n for n in out.stdout.split() if n.endswith(".ckpt") and n != "last.ckpt"),
                key=self._step_key,
            )
            for n in names[: max(0, len(names) - self.keep_uploaded)]:
                subprocess.run([self.rclone_bin, "deletefile", f"{self.remote.rstrip('/')}/{n}"],
                               capture_output=True)
        else:
            files = sorted(
                (p for p in Path(self.remote).glob("*.ckpt") if p.name != "last.ckpt"),
                key=lambda p: self._step_key(p.name),
            )
            for p in files[: max(0, len(files) - self.keep_uploaded)]:
                try:
                    p.unlink()
                except FileNotFoundError:
                    pass

    # ---- hook: enqueue new checkpoints ------------------------------------ #
    def _scan_and_enqueue(self, trainer) -> None:
        if trainer.global_rank != 0:
            return
        for p in self.checkpoint_dir.glob("*.ckpt"):
            if p.name == "last.ckpt":
                continue
            with self._lock:
                seen = p.name in self._enqueued
            if not seen:
                with self._lock:
                    self._enqueued.add(p.name)
                self._q.put(str(p))

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        # Runs after ModelCheckpoint's hook (this callback is appended later), so any
        # checkpoint written this step already exists on disk.
        self._scan_and_enqueue(trainer)

    def on_fit_end(self, trainer, pl_module) -> None:
        # Make sure the final checkpoint(s) are enqueued, then drain before exit.
        self._scan_and_enqueue(trainer)
        if self._worker is None:
            return
        print("[ckpt-upload] draining pending uploads...", flush=True)
        self._q.join()
        self._stop.set()
        self._worker.join(timeout=30.0)

    def finalize_uploads(self, trainer) -> None:
        """Synchronously ensure every local step checkpoint is on the remote.

        Safe to call from the main thread after fit(). Lightning's on_fit_end async drain
        can be pre-empted by the DataLoader-teardown crash (the final checkpoint is saved
        locally but its Drive copy is killed mid-flight), and the os._exit() fast-path would
        kill the worker thread too — so this is the reliable last flush before exit.
        """
        if getattr(trainer, "global_rank", 0) != 0:
            return
        # Stop the async worker, letting it finish any in-flight copy first.
        if self._worker is not None:
            self._stop.set()
            self._worker.join(timeout=120.0)
            self._worker = None
        # Synchronously copy anything not yet confirmed on the remote (worker has exited,
        # so no concurrent copy of the same file).
        for ckpt in sorted(self._local_step_ckpts(), key=lambda p: p.stat().st_mtime):
            if ckpt.name in self._uploaded:
                continue
            try:
                self._upload_one(ckpt)
                with self._lock:
                    self._uploaded.add(ckpt.name)
            except Exception as exc:  # noqa: BLE001 - a flush failure must not abort exit
                print(f"[ckpt-upload] final flush FAILED {ckpt.name}: {exc}", flush=True)


# --------------------------------------------------------------------------- #
# dim auto-detection
# --------------------------------------------------------------------------- #
def peek_record(args: argparse.Namespace, hf_token: str | bool | None) -> Any:
    """Pull one cached record to read latent/text shapes (downloads one shard if remote)."""
    ds = make_wds_dataset(
        cache_wds=args.cache_wds,
        local_dir=args.wds_local_dir,
        prefetch_shards=1,
        download_workers=1,
        shuffle_buffer=1,
        hf_token=hf_token,
        seed=args.seed,
        decode_queue_size=0,  # one record: decode inline, no producer thread
        # Peek must mirror training decode: with --cache-captions-only the record's
        # text_embedding is None and build_config takes text dims from the encoder.
        captions_only=args.cache_captions_only,
    )
    return next(iter(ds))


def text_encoder_hidden_size(model_id: str, hf_token: str | bool | None) -> int:
    """Read the encoder's hidden size from its HF config (one small config.json fetch),
    so --monet-direct can size the DiT's text_dim before the encoder itself loads."""
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model_id, token=hf_token or None)
    hidden = getattr(cfg, "hidden_size", None)
    if not hidden:
        hidden = getattr(getattr(cfg, "text_config", None), "hidden_size", None)
    if not hidden:
        raise RuntimeError(f"Could not read hidden_size from {model_id}'s config")
    return int(hidden)


def build_config(args: argparse.Namespace, record: Any) -> BoomerFLADiTConfig:
    # With --skip-peek, record is None and dims come from the explicit flags (no shard
    # download at startup). Otherwise dims are read from one peeked cache record.
    if record is not None:
        latent_channels = int(record.image_latent.shape[0])
        latent_size = int(record.image_latent.shape[-1])
        if record.text_embedding is not None:
            text_dim = int(record.text_embedding.shape[-1])
            text_seq_len = int(record.text_embedding.shape[-2])
        else:
            # --cache-captions-only: latent dims from the peeked record, text dims from
            # the on-the-fly encoder (main() set these from its config.json).
            text_dim = args.text_dim
            text_seq_len = args.text_seq_len
    else:
        latent_channels = args.latent_channels
        latent_size = args.latent_size
        text_dim = args.text_dim
        text_seq_len = args.text_seq_len
    return BoomerFLADiTConfig(
        latent_channels=latent_channels,
        latent_size=latent_size,
        text_dim=text_dim,
        text_seq_len=text_seq_len,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        y_norm=args.y_norm,
        y_norm_scale_factor=args.y_norm_scale_factor,
        mixer_type=args.mixer_type,
        fla_mode=args.fla_mode,
        fla_feature_map=args.fla_feature_map,
        fla_bidirectional=args.fla_bidirectional,
        use_short_conv=args.use_short_conv,
        conv_size=args.conv_size,
        image_attention_every=args.image_attn_every,
        image_attention_backend=args.image_attn_backend,
        image_attention_qk_rmsnorm=args.image_attn_qk_rmsnorm,
        image_attention_rope=args.image_rope,
        image_rope_theta=args.image_rope_theta,
        cross_attention_backend=args.cross_attn_backend,
        cross_attention_qk_norm=args.cross_attn_qk_norm,
        cross_attention_every=args.cross_attn_every,
        shared_adaln=args.shared_adaln,
        fourier_timestep_embed=args.fourier_timestep_embed,
        hilbert_mixer_order=args.hilbert_mixer_order,
        parallel_block=args.parallel_block,
        dual_stream_depth=args.dual_stream_depth,
        multimodal_coord_ids=args.multimodal_coord_ids,
        joint_refresh_attention=args.joint_refresh_attention,
        sandwich_rmsnorm=args.sandwich_rmsnorm,
        rope_ntk_factor=args.rope_ntk_factor,
        use_abs_pos_embed=args.abs_pos_embed,
        patch_size=args.patch_size,
        extra_input_channels=args.extra_input_channels,
        gradient_checkpointing=args.gradient_checkpointing,
        repa_dim=args.repa_dim,
        repa_layer=args.repa_layer,
        dispersive_layer=args.dispersive_layer,
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Lightning trainer for Boomer FLA (Qwen-conditioned, from scratch).")
    bo = argparse.BooleanOptionalAction

    # data
    p.add_argument("--cache-wds", required=True, help="WDS glob, e.g. hf://datasets/<repo>/chunk_*/*.tar")
    p.add_argument("--wds-local-dir", default="/content/_wds_cache")
    p.add_argument("--wds-prefetch-shards", type=int, default=8,
                   help="hf:// caches: tars to download in parallel ahead of the GPU "
                        "(disk window = this x shard size). Same prefetch as the caching tool.")
    p.add_argument("--wds-download-workers", type=int, default=4,
                   help="hf:// caches: parallel shard-download threads.")
    p.add_argument("--wds-decode-queue", type=int, default=512,
                   help="hf:// caches: bounded record queue for the background decode "
                        "thread (overlaps sample reconstruction with the GPU step at "
                        "num_workers=0). Set 0 to decode inline on the consumer thread.")
    p.add_argument("--wds-download-max-retries", type=int, default=2,
                   help="hf:// caches: re-download a stalled shard up to N times (the stuck "
                        "worker is replaced, not killed) before skipping it.")
    p.add_argument("--wds-max-consecutive-skips", type=int, default=8,
                   help="hf:// caches: abort the run after this many back-to-back shard "
                        "skips, so a dead network fails loudly instead of silently training "
                        "on a shrinking data slice.")
    p.add_argument("--wds-rolling-shards", type=int, default=2,
                   help="(deprecated alias for --wds-prefetch-shards; ignored)")
    p.add_argument("--wds-shuffle-buffer", type=int, default=2000)
    p.add_argument("--wds-shard-shuffle", type=int, default=0,
                   help="LOCAL streaming only: shard shuffle buffer/count. 0 disables.")
    p.add_argument("--num-workers", type=int, default=4,
                   help="DataLoader workers for LOCAL shard globs (forced to 0 for hf:// prefetch cache).")
    p.add_argument("--pin-memory", action=bo, default=True,
                   help="Pin CPU batches for faster async GPU transfer (CUDA only).")
    p.add_argument("--prefetch-factor", type=int, default=2,
                   help="Batches prefetched per worker when --num-workers > 0.")
    p.add_argument("--persistent-workers", action=bo, default=None,
                   help="Keep DataLoader workers alive between epochs. Default: on when --num-workers > 0.")
    p.add_argument("--parquet-read-batch-size", type=int, default=32,
                   help="Rows materialized by Arrow at once per Parquet worker. The cache has "
                        "large 512-row groups; 32 bounds transient host RAM without changing "
                        "the DataLoader batch size.")
    p.add_argument("--parquet-memory-trim-every", type=int, default=4096,
                   help="Parquet rows per worker between best-effort Arrow/libc memory "
                        "release calls. 0 disables; default prevents long-run RSS growth.")
    p.add_argument("--parquet-prefetch", dest="parquet_prefetch_ranges", action=bo, default=True,
                   help="Background-prefetch the next Parquet byte range while the current one "
                        "decodes (pyarrow CacheOptions). Skipped automatically on datasets/"
                        "pyarrow versions without fragment_scan_options support.")
    p.add_argument("--parquet-range-mib", dest="parquet_range_size_mib", type=int, default=64,
                   help="Per-request Parquet range size in MiB (pyarrow default 32). Larger = "
                        "fewer requests but ~2x this buffered per worker with prefetch on; "
                        "0 keeps the pyarrow default.")
    p.add_argument("--hf-token", default=None, help="HF token, or 'true' for the logged-in token.")

    # Direct jasperai/monet streaming (no cache build): --cache-wds points at raw monet
    # parquet globs (e.g. hf://datasets/jasperai/monet/v1.2.0/cc12m/*/*.parquet), the
    # precomputed SANA f32c32 latents are used as-is, and captions are encoded on-GPU
    # every step by a resident frozen Qwen encoder (see --text-encoder-*).
    p.add_argument("--monet-direct", action=bo, default=False,
                   help="Stream jasperai/monet parquet shards directly: latents from the "
                        "dataset, text encoded on the fly (skips the caching pipeline).")
    p.add_argument("--cache-captions-only", action=bo, default=False,
                   help="Boomer WDS/parquet caches: IGNORE the cached text/text_mask "
                        "columns (parquet never even downloads them) and encode the raw "
                        "caption column on the fly with --text-encoder-*. For caches whose "
                        "text was baked with a different encoder than the current model, "
                        "e.g. boomer-finet2i-1024-parquet's 2048-d text vs v6's 1536-d.")
    p.add_argument("--monet-latent-key", default=MONET_LATENT_KEY)
    p.add_argument("--monet-caption-keys", default=",".join(MONET_CAPTION_KEYS),
                   help="comma-ordered caption-column priority; first non-empty per row wins "
                        "(unless --monet-caption-probs samples among them instead).")
    p.add_argument("--monet-caption-probs", default=None,
                   help="Caption sampling weights aligned 1:1 with --monet-caption-keys: each "
                        "row picks ONE caption at random among its non-empty columns with "
                        "these relative probabilities (nano-t2i captioners_probabilities); "
                        "zero-weight keys are never used. With --monet-weights, ';'-separated "
                        "groups give each subset glob its own policy (nano_mix style; an "
                        "empty group = priority order). Unset = strict priority order.")
    p.add_argument("--monet-weights", default=None,
                   help="CSV weights aligned 1:1 with the comma-separated globs in --cache-wds: "
                        "each batch row is drawn from glob i with probability w_i/sum(w) "
                        "(row-level stream interleave; nano_mix.yaml-style subset proportions). "
                        "Unset = all globs merged into one stream.")
    p.add_argument("--monet-dino", action=bo, default=True,
                   help="carry monet's DINOv2 embedding into batches (REPA target).")
    p.add_argument("--monet-min-aesthetic", type=float, default=None)
    p.add_argument("--monet-max-nsfw", type=float, default=None)
    p.add_argument("--monet-max-watermark", type=float, default=None)

    # Editing phases: E1 self-supervised inpaint/fill over any latent cache, and
    # E2 instruction editing over pixel-pair corpora (OmniEdit). Both condition the
    # model's reserved input channels (--extra-input-channels = latent_channels + 1).
    p.add_argument("--fill-training", action=bo, default=False,
                   help="E1: synthesize inpainting masks over the latent stream and train "
                        "with cond_input=[x0*(1-m), m] (FLUX-Fill style). Works with any "
                        "cache/monet stream; requires --extra-input-channels = "
                        "latent_channels + 1.")
    p.add_argument("--fill-mask-probs", default=None,
                   help="mask-type mix, e.g. 'blob=0.3,rect=0.25,outpaint=0.15,full=0.1,"
                        "tiny=0.2' (normalized; that mix is the default). 'full' = pure-t2i "
                        "retention samples under the fill interface.")
    p.add_argument("--cond-input-dropout", type=float, default=0.1,
                   help="per-sample probability of zeroing the WHOLE cond_input block "
                        "(exactly the zero-fed t2i pretrain state) -> retains pure t2i and "
                        "enables image-side CFG (--sample-image-cfg / IP2P dual guidance). "
                        "Independent of --cond-dropout on the text; only applies when a "
                        "cond_input exists (fill/edit phases). Set 0 to disable.")
    p.add_argument("--edit-direct", action=bo, default=False,
                   help="E2: stream OmniEdit-style parquet pixel pairs directly; both sides "
                        "are DC-AE-encoded on the fly, source latent + all-ones mask become "
                        "cond_input, the instruction is the caption (encoded by "
                        "--text-encoder-*). No cache build.")
    p.add_argument("--edit-resolution", type=int, default=1024,
                   help="square training resolution for the edit pair (edit-aware shared "
                        "crop, then resize).")
    p.add_argument("--edit-min-o-score", type=float, default=None,
                   help="drop rows below this o_score (OmniEdit-Filtered is already >= 9).")
    p.add_argument("--edit-tasks", default=None,
                   help="CSV whitelist of OmniEdit task names; unset = all tasks.")
    p.add_argument("--edit-prompt-pick", choices=["random", "first", "last"], default="random",
                   help="which entry of edited_prompt_list to train on; 'random' samples "
                        "per pass (short vs detailed phrasing augmentation).")
    p.add_argument("--edit-min-short-side", type=int, default=0,
                   help="drop pairs whose short side is below this many pixels (0 = keep "
                        "all; OmniEdit's minimum is 768).")
    p.add_argument("--edit-encode-chunk", type=int, default=8,
                   help="images per DC-AE encode chunk in training_step (bounds encode VRAM).")
    p.add_argument("--edit-vae-id", default=DEFAULT_VAE_ID,
                   help="DC-AE used for on-the-fly pixel encoding (must match the VAE the "
                        "t2i latents were cached with).")
    p.add_argument("--edit-scaling-factor", type=float, default=DEFAULT_SCALING_FACTOR,
                   help="latent scaling factor for on-the-fly encoding (cache parity).")
    p.add_argument("--sample-cond-count", type=int, default=4,
                   help="fill/edit runs: number of FIXED conditioned previews captured from "
                        "the first training batch and logged with --sample-every-steps "
                        "(0 disables the conditioning preview).")
    p.add_argument("--sample-image-cfg", type=float, default=0.0,
                   help="conditioning previews: IP2P dual-guidance image scale (three-branch "
                        "CFG: uncond / source-only / source+instruction). 0 = classic "
                        "two-branch text CFG with the condition pinned in both branches. "
                        "Typical: 1.2-1.8 once --cond-input-dropout trained the zero branch.")
    p.add_argument("--expect-pristine-cond-input", action=bo, default=False,
                   help="fail launch if the x_embedder reserved slice is NON-zero — guard "
                        "for a fresh E1 start from a pure-t2i seed (leave off for E1 "
                        "resumes / E2 init from an E1 checkpoint).")

    # On-the-fly text encoder (used with --monet-direct / --cache-captions-only / --edit-direct).
    p.add_argument("--text-encoder-id", default="Qwen/Qwen3.5-4B")
    p.add_argument("--text-encoder-layer", type=int, default=-2,
                   help="hidden_states readout layer (-2 = second-to-last, raw/un-normed; "
                        "layers above it are dropped so they are never computed).")
    p.add_argument("--text-encoder-max-length", type=int, default=300,
                   help="fixed token length for on-the-fly encoding; becomes text_seq_len.")
    p.add_argument("--text-encoder-instruction", action=bo, default=True,
                   help="prepend the standard conditioning instruction before each caption "
                        "(matches the cache pipeline's default).")

    p.add_argument("--latent-mean", type=float, default=0.0)
    p.add_argument("--latent-std", type=float, default=1.0)
    # Skip the startup dim-peek (which downloads one full shard) and use explicit dims.
    # Defaults match the monet DC-AE/Qwen cache: latent (32,16,16), text (224,2048).
    p.add_argument("--skip-peek", action="store_true",
                   help="Skip the dim-peek shard download; use --latent-*/--text-* dims directly.")
    p.add_argument("--latent-channels", type=int, default=32)
    p.add_argument("--latent-size", type=int, default=16)
    p.add_argument("--text-dim", type=int, default=2048)
    p.add_argument("--text-seq-len", type=int, default=224)

    # model
    p.add_argument("--hidden-dim", type=int, default=896)
    p.add_argument("--depth", type=int, default=24)
    p.add_argument("--num-heads", type=int, default=14)
    p.add_argument("--mlp-ratio", type=float, default=2.5)
    p.add_argument("--mixer-type", default="fla_gdn2",
                   choices=["mamba3", "fla_gated_deltanet", "fla_gdn2", "fla_linear", "fla_gla",
                            "fla_delta_net", "fla_hgrn2", "fla_based", "mlp", "fallback"])
    p.add_argument("--fla-mode", default="chunk", choices=["chunk", "fused_chunk"])
    p.add_argument("--fla-feature-map", default="relu")
    p.add_argument("--fla-bidirectional", action=bo, default=True)
    p.add_argument("--use-short-conv", action="store_true")
    p.add_argument("--conv-size", type=int, default=4)
    p.add_argument("--image-attn-every", type=int, default=6)
    p.add_argument("--image-attn-backend", default="sdpa",
                   choices=["sdpa", "flash2", "flash3", "flash4", "auto"])
    p.add_argument("--image-attn-qk-rmsnorm", "--image-attn-qk-norm", dest="image_attn_qk_rmsnorm",
                   action=bo, default=True,
                   help="Sana-style per-head Q/K RMSNorm inside full image self-attention.")
    p.add_argument("--image-rope", action=bo, default=False,
                   help="Apply 2D RoPE to full image self-attention anchor blocks.")
    p.add_argument("--image-rope-theta", type=float, default=10000.0,
                   help="RoPE frequency base for --image-rope.")
    p.add_argument("--cross-attn-backend", default="sdpa", choices=["sdpa", "xformers", "auto", "mha"])
    p.add_argument("--cross-attn-qk-rmsnorm", "--cross-attn-qk-norm", dest="cross_attn_qk_norm",
                   action=bo, default=True,
                   help="Sana-style Q/K RMSNorm inside text cross-attention.")
    p.add_argument("--cross-attn-every", type=int, default=1,
                   help="Cross-attend to text every Nth single-stream block (1 = every block). "
                        "PixArt-Sigma-style thinning; pilot-ablate before a long run.")
    p.add_argument("--shared-adaln", action=bo, default=False,
                   help="PixArt-alpha adaLN-single: one shared timestep-modulation projection for "
                        "all blocks (per-block scale_shift_table keeps layer variation). Saves ~27%% "
                        "params at the deployed size.")
    p.add_argument("--fourier-timestep", dest="fourier_timestep_embed", action=bo, default=False,
                   help="DiT-style sinusoidal timestep features before the timestep MLP instead of "
                        "feeding the raw sigma scalar to Linear(1, hidden).")
    p.add_argument("--hilbert-order", dest="hilbert_mixer_order", action=bo, default=False,
                   help="Feed the FLA mixers Hilbert-curve token order (restored afterward) so "
                        "sequence adjacency tracks 2D adjacency. Needs a square power-of-two grid.")
    p.add_argument("--parallel-block", action=bo, default=False,
                   help="FLUX/SD3-style parallel residual branches inside blocks.")
    p.add_argument("--dual-stream-depth", type=int, default=0)
    p.add_argument("--multimodal-coord-ids", action=bo, default=False)
    p.add_argument("--joint-refresh-attn", dest="joint_refresh_attention", action=bo, default=False,
                   help="v6: every --image-attn-every-th post-prefix block becomes a joint refresh "
                        "block — one MMDiT-style attention over [text; image] replaces the "
                        "image-only full attention + that block's cross-attn, and text is updated "
                        "through its own out-proj + FFN (SD3-style renegotiation).")
    p.add_argument("--sandwich-rmsnorm", action=bo, default=False,
                   help="v6: Lumina-Next sandwich norm — RMSNorm each sublayer output before the "
                        "residual add (the residual stream itself is never normalized).")
    p.add_argument("--rope-ntk-factor", type=float, default=1.0,
                   help="v6: NTK-aware RoPE rescale for image-spatial axes only (text positions "
                        "keep base frequencies). 1.0 for 512px pretrain; ~2.0 for the 1024px "
                        "fine-tune where the token grid doubles.")
    p.add_argument("--abs-pos-embed", action=bo, default=True)
    p.add_argument("--y-norm", action=bo, default=True)
    p.add_argument("--y-norm-scale-factor", type=float, default=0.01, help="Sana uses 0.01.")
    p.add_argument("--patch-size", type=int, default=1)
    p.add_argument("--extra-input-channels", type=int, default=0,
                   help="FLUX-Fill-style reserved channels (33 = masked latent + mask) for future inpainting.")
    p.add_argument("--gradient-checkpointing", "--grad-checkpoint", dest="gradient_checkpointing",
                   action="store_true")

    # REPA representation alignment (needs a cache with dino.npy, e.g. monet).
    p.add_argument("--repa-dim", type=int, default=0, help="SSL target dim; 1536 for DINOv2-ViT-G. 0=off")
    p.add_argument("--repa-layer", type=int, default=-1, help="DiT block to align; -1 => depth//2")
    p.add_argument("--repa-weight", type=float, default=0.0, help="REPA loss weight (e.g. 0.5)")
    p.add_argument("--repa-anneal-steps", type=int, default=0, help="decay REPA weight to 0 by this step")

    # Dispersive loss (arxiv 2506.09027): target-free batch-dispersion regularizer.
    # Works on the existing cache (no dino needed). 0 weight = off.
    p.add_argument("--dispersive-weight", type=float, default=0.0,
                   help="Dispersive loss weight (e.g. 0.25–0.5). 0=off.")
    p.add_argument("--dispersive-layer", type=int, default=-1,
                   help="DiT block whose pooled hidden state to disperse; -1 => depth//2.")
    p.add_argument("--dispersive-temp", type=float, default=0.5,
                   help="Temperature tau in the InfoNCE-style dispersive loss.")
    p.add_argument("--dispersive-metric", default="l2", choices=["l2", "cosine"],
                   help="Pairwise score: 'l2' (neg squared distance, paper default) or 'cosine'.")
    p.add_argument("--dispersive-anneal-steps", type=int, default=0,
                   help="decay dispersive weight to 0 by this step (0 = constant).")

    # optim / flow
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--fused-adamw", action=bo, default=True)
    p.add_argument("--warmup-steps", type=int, default=1000,
                   help="Linear warmup steps for the warmup-bearing schedulers.")
    p.add_argument("--lr-scheduler", default="linear-warmup-constant",
                   choices=["constant", "linear-warmup-constant", "linear-warmup-cosine"],
                   help="constant=flat (no warmup); linear-warmup-constant=warmup then hold "
                        "(legacy default); linear-warmup-cosine=warmup then cosine decay to "
                        "--min-lr-ratio. Computed from global step, so resume is deterministic.")
    p.add_argument("--min-lr-ratio", type=float, default=0.1,
                   help="Final LR as a fraction of --lr for linear-warmup-cosine decay.")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--accumulate", type=int, default=1)
    p.add_argument("--cond-dropout", type=float, default=0.1)
    p.add_argument("--t-sampler", default="plateau-logit-normal",
                   choices=["uniform", "high", "low", "logit-normal", "bounded-logit-normal",
                            "stretched-logit-normal", "plateau-logit-normal"])
    # "Texture polish" final phase: from --t-sampler-final-step onward, switch the timestep
    # sampler (e.g. to uniform), shifting compute to the low-noise t~0 regime to sharpen
    # textures/edges. Off unless --t-sampler-final is set.
    p.add_argument("--t-sampler-final", default=None,
                   choices=["uniform", "high", "low", "logit-normal", "bounded-logit-normal",
                            "stretched-logit-normal", "plateau-logit-normal"],
                   help="Sampler to switch to for the final polish phase (e.g. uniform).")
    p.add_argument("--t-sampler-final-step", type=int, default=80000,
                   help="global_step at which to switch to --t-sampler-final.")
    p.add_argument("--t-bias", type=float, default=2.0)
    p.add_argument("--flow-shift", type=float, default=1.5)
    p.add_argument("--logit-mean", type=float, default=0.0)
    p.add_argument("--logit-std", type=float, default=1.0)
    p.add_argument("--plateau-low", type=float, default=0.0)
    p.add_argument("--plateau-high", type=float, default=1.0)
    # SANA stretched-logit-normal quantile window (normal CDF space). Truncates the
    # logit-normal to [q-low, q-high] quantiles and stretches the central band back to
    # [0,1] — flattens the peaked logit-normal toward uniform (wider sigma coverage).
    # Narrower window => flatter; 0/1 == plain logit-normal (no truncation).
    p.add_argument("--stretch-q-low", type=float, default=0.0)
    p.add_argument("--stretch-q-high", type=float, default=1.0)
    p.add_argument("--timestep-weight", action="store_true")
    p.add_argument("--min-snr-gamma", type=float, default=5.0)

    # ema / trainer
    p.add_argument("--ema-decay", type=float, default=0.9999)
    p.add_argument("--ema-update-every", type=int, default=1)
    p.add_argument("--max-steps", "--steps", dest="max_steps", type=int, default=100000)
    p.add_argument("--precision", default="bf16-mixed")
    p.add_argument("--dtype", default=None, choices=["bf16", "fp16", "fp32"],
                   help="If set, overrides --precision (bf16->bf16-mixed, fp16->16-mixed, fp32->32-true).")
    p.add_argument("--device", default="cuda",
                   help="'cuda' (gpu accelerator) or 'cpu'. Lightning manages device placement.")
    p.add_argument("--devices", type=int, default=1, help="GPUs per node.")
    p.add_argument("--num-nodes", type=int, default=1)
    p.add_argument("--fsdp2", action=bo, default=False,
                   help="Shard the DiT with FSDP2 (fully_shard per block) via Lightning's "
                        "ModelParallelStrategy — pure data parallel (tensor_parallel_size=1, "
                        "data_parallel_size=devices*num_nodes; never 'auto', which would go "
                        "tensor-parallel on a single node). Forces precision 32-true: bf16 "
                        "compute + fp32 grad reduce come from the FSDP2 MixedPrecisionPolicy "
                        "with fp32 sharded master weights. Disables EMA and in-training "
                        "sample previews (neither is FSDP2-safe yet); --compile unsupported.")
    p.add_argument("--fsdp2-distributed-checkpoint", action=bo, default=False,
                   help="Save sharded (DCP directory) checkpoints instead of consolidating "
                        "full weights into a single rank-0 .ckpt file. Keep OFF so resume/"
                        "export/upload tooling keeps working; the consolidated save is cheap "
                        "at Boomer's scale.")
    p.add_argument("--checkpoint-dir", default="/content/ckpt_stage0",
                   help="Local directory for checkpoints (always written here first).")
    p.add_argument("--save-every-steps", "--save-every", dest="save_every_steps", type=int, default=2000)
    # Async checkpoint upload: save locally, copy to Drive/rclone in the background,
    # keep only the newest N local checkpoints (older ones deleted after they upload).
    p.add_argument("--checkpoint-upload-remote", default=None,
                   help="Background upload destination: an HF bucket "
                        "('hf://buckets/akrao9/Boomerv6-ckpt', uploaded via the hf CLI), "
                        "an rclone remote ('gdrive:Boomer/ckpts'), OR a directory such as "
                        "a mounted Drive folder / hf-mount'ed bucket. Checkpoints always "
                        "save to --checkpoint-dir (fast local disk) first and copy over "
                        "asynchronously. Off if unset.")
    p.add_argument("--checkpoint-hf-bin", default="hf",
                   help="hf CLI executable for hf://buckets/... upload remotes "
                        "(install: curl -LsSf https://hf.co/cli/install.sh | bash).")
    p.add_argument("--checkpoint-local-retain", type=int, default=2,
                   help="Keep this many newest checkpoints on local disk; older uploaded ones are deleted.")
    p.add_argument("--checkpoint-keep-uploaded", type=int, default=0,
                   help="Prune the remote to this many newest checkpoints. 0 = keep all uploaded.")
    p.add_argument("--checkpoint-rclone-bin", default="rclone",
                   help="rclone executable for --checkpoint-upload-remote rclone remotes.")
    p.add_argument("--checkpoint-upload-chunk-size", default="128M",
                   help="rclone --drive-chunk-size for Google Drive uploads.")
    p.add_argument("--log-every-steps", "--log-every", dest="log_every_steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", default=None, help="Path to a .ckpt to resume from.")

    # W&B logging (off unless --wandb). train/loss, train/repa_loss, lr, etc. log automatically.
    p.add_argument("--wandb", action="store_true", help="Log training metrics to Weights & Biases.")
    p.add_argument("--wandb-project", default="boomer", help="W&B project name.")
    p.add_argument("--wandb-run-name", default=None, help="Optional W&B run name.")
    p.add_argument("--wandb-entity", default=None, help="Optional W&B entity/team.")
    p.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online",
                   help="W&B mode. Use offline on air-gapped machines and sync later.")
    p.add_argument("--wandb-run-id", default=None,
                   help="Stable W&B run id for resuming the SAME run across sessions. "
                        "Defaults to the WANDB_RUN_ID env var. The run name is just a "
                        "display label and is not a stable key -- set this to resume.")
    # In-training sample previews -> W&B. Pre-encode prompts with tools/encode_prompts.py.
    p.add_argument("--sample-every-steps", type=int, default=0,
                   help="Sample the eval prompts every N optimizer steps and log to W&B. 0 disables.")
    p.add_argument("--sample-prompts-cache", default="prompts_cond.pt",
                   help="Path to the .pt produced by tools/encode_prompts.py (embeddings + masks).")
    p.add_argument("--sample-prompts-file", default=None,
                   help="--monet-direct: plain-text prompt list (one per line) for previews, "
                        "encoded by the resident text encoder. Falls back to the prompt "
                        "STRINGS inside --sample-prompts-cache when unset.")
    p.add_argument("--sample-weights", choices=["ema", "live"], default="ema",
                   help="Which weights the preview samples: EMA shadow (default) or live training weights.")
    p.add_argument("--sample-steps", type=int, default=32, help="Inference steps per preview image.")
    p.add_argument("--sample-sampler", choices=["stork2", "stork4", "euler", "heun"], default="stork2")
    p.add_argument("--sample-stork-substeps", type=int, default=5)
    p.add_argument("--sample-cfg-scale", type=float, default=4.0)
    p.add_argument("--sample-cfg-rescale", type=float, default=0.5)
    p.add_argument("--sample-cfg-interval-low", type=float, default=0.1)
    p.add_argument("--sample-cfg-interval-high", type=float, default=0.9)
    p.add_argument("--sample-seed", type=int, default=0, help="Fixed seed so previews are comparable across steps.")
    p.add_argument("--sample-vae-id", default=CacheConfig.vae_id, help="DC-AE decoder id for previews.")
    p.add_argument("--sample-scaling-factor", type=float, default=CacheConfig.scaling_factor)
    p.add_argument("--resume-weights-only", action="store_true",
                   help="With --resume, load model weights only; reset optimizer/scheduler/step.")
    p.add_argument("--compile", action="store_true",
                   help="Wrap the training model in torch.compile. EMA is left uncompiled.")
    p.add_argument("--compile-mode",
                   choices=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"],
                   default="default",
                   help="torch.compile mode. 'default' avoids aggressive Triton autotune candidates "
                        "that can exceed Blackwell's per-SM resource limits.")
    return p


def main() -> None:
    args = build_parser().parse_args()
    # `--timestep-weight` (Sana sqrt(1+sigma^2)/sigma, upweights detail) and
    # `--min-snr-gamma` (downweights detail) are *alternative* per-timestep loss
    # reweightings, not composable — stacking them multiplies into a product no one
    # designed. Refuse both at once rather than silently compounding.
    if args.timestep_weight and args.min_snr_gamma > 0.0:
        raise SystemExit(
            "Use only one per-timestep loss weighting: --timestep-weight "
            "(Sana sqrt(1+sigma^2)/sigma) OR --min-snr-gamma, not both."
        )
    # Editing-phase invariants (fail at launch, not at first step).
    if args.fill_training and args.edit_direct:
        raise SystemExit("--fill-training and --edit-direct are separate phases (E1/E2); "
                         "pick one per run.")
    if args.edit_direct and (args.monet_direct or args.cache_captions_only):
        raise SystemExit("--edit-direct streams pixel pairs; it cannot combine with "
                         "--monet-direct or --cache-captions-only.")
    if args.fill_training or args.edit_direct:
        expected_extra = args.latent_channels + 1
        if args.extra_input_channels != expected_extra:
            raise SystemExit(
                f"--fill-training/--edit-direct condition on [latent ({args.latent_channels}ch) "
                f"+ mask (1ch)]; set --extra-input-channels {expected_extra} "
                f"(got {args.extra_input_channels}). This must also match the seeded checkpoint."
            )
        parse_fill_mask_probs(args.fill_mask_probs)  # fail fast on a bad spec
    if args.edit_direct and args.latent_size * 32 != args.edit_resolution:
        # DC-AE f32: latent grid = resolution / 32. A mismatch would build a DiT whose
        # RoPE/token grid disagrees with what encode_pixels produces every step.
        raise SystemExit(
            f"--edit-resolution {args.edit_resolution} needs --latent-size "
            f"{args.edit_resolution // 32} (DC-AE f32), got --latent-size {args.latent_size}."
        )
    if args.edit_direct and args.wds_shuffle_buffer > 512:
        # A pixel-pair record is ~6 MB (2 x 3 x 1024^2 uint8) vs ~64 KB for a latent
        # record; a latent-sized reservoir would be tens of GB of host RAM per worker
        # group. Shard-order shuffling still decorrelates the stream.
        print(f"[edit-direct] clamping --wds-shuffle-buffer {args.wds_shuffle_buffer} -> 512 "
              "(pixel records are ~6 MB each)", flush=True)
        args.wds_shuffle_buffer = 512
    # Fail CSV-alignment errors at launch, not minutes later inside a spawned
    # DataLoader worker (the dataset re-validates, but this is the friendly path).
    def _csv_len(csv: str) -> int:
        return len([v for v in csv.split(",") if v.strip()])

    if args.monet_caption_probs:
        groups = args.monet_caption_probs.split(";")
        if len(groups) > 1:  # per-glob ';' groups pair with the mix globs
            if not args.monet_weights:
                raise SystemExit(
                    "per-glob --monet-caption-probs (';' groups) require --monet-weights."
                )
            if len(groups) != _csv_len(args.cache_wds):
                raise SystemExit(
                    f"--monet-caption-probs has {len(groups)} ';' groups for "
                    f"{_csv_len(args.cache_wds)} globs in --cache-wds; they must align 1:1."
                )
        for group in groups:
            if group.strip() and _csv_len(group) != _csv_len(args.monet_caption_keys):
                raise SystemExit(
                    f"--monet-caption-probs group '{group}' has {_csv_len(group)} weights "
                    f"for {_csv_len(args.monet_caption_keys)} --monet-caption-keys; "
                    "they must align 1:1."
                )
    if args.monet_weights and _csv_len(args.monet_weights) != _csv_len(args.cache_wds):
        raise SystemExit(
            f"--monet-weights has {_csv_len(args.monet_weights)} weights for "
            f"{_csv_len(args.cache_wds)} globs in --cache-wds; they must align 1:1."
        )

    # Multi-GPU strategy. --fsdp2 = ModelParallelStrategy (FSDP2 via configure_model's
    # fully_shard); otherwise plain DDP when more than one device is requested.
    total_devices = args.devices * args.num_nodes
    strategy: Any = "auto"
    if args.fsdp2:
        if total_devices < 2:
            raise SystemExit("--fsdp2 needs at least 2 GPUs (set --devices / --num-nodes).")
        if args.compile:
            raise SystemExit(
                "--compile with --fsdp2 is not wired up (compile must be applied per block "
                "AFTER fully_shard); drop one of the two."
            )
        if args.fsdp2_distributed_checkpoint and args.checkpoint_upload_remote:
            raise SystemExit(
                "--checkpoint-upload-remote copies single .ckpt FILES; distributed (DCP "
                "directory) checkpoints can't be uploaded by it. Drop one of the two."
            )
        from lightning.pytorch.strategies import ModelParallelStrategy

        strategy = ModelParallelStrategy(
            data_parallel_size=total_devices,  # explicit: 'auto' on one node means
            tensor_parallel_size=1,            # TENSOR parallel across local GPUs
            save_distributed_checkpoint=args.fsdp2_distributed_checkpoint,
        )
        if (args.dtype or args.precision) not in ("fp32", "32-true"):
            print(
                "[fsdp2] precision -> 32-true (bf16 compute + fp32 grad reduce come from "
                "the FSDP2 MixedPrecisionPolicy; fp32 sharded master weights)",
                flush=True,
            )
        args.precision, args.dtype = "32-true", None
        if args.ema_decay > 0.0:
            print("[fsdp2] EMA disabled: the EMA callback shadows full state_dicts and "
                  "is not DTensor-safe.", flush=True)
            args.ema_decay = 0.0
        if args.sample_every_steps > 0:
            print("[fsdp2] in-training sample previews disabled: rank-0-only sampling "
                  "would deadlock FSDP2's collective all-gathers.", flush=True)
            args.sample_every_steps = 0
    elif total_devices > 1:
        from lightning.pytorch.strategies import DDPStrategy

        # static_graph=True silences AccumulateGrad stream-mismatch warnings that show up
        # with gradient checkpointing + DDP (the graph topology is fixed per step).
        strategy = DDPStrategy(static_graph=True)

    pl.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision("high")

    hf_token = hf_token_value(args.hf_token)

    if args.monet_direct or args.cache_captions_only or args.edit_direct:
        # On-the-fly text: text_seq_len is the fixed encode length and text_dim comes
        # from the encoder's config.json. --monet-direct also skips the shard peek
        # entirely (monet's 512px f32c32 latents are the --latent-* defaults);
        # --cache-captions-only keeps the peek for LATENT dims (e.g. 1024px caches);
        # --edit-direct has no latents to peek at all (pixels; dims from flags).
        args.text_seq_len = args.text_encoder_max_length
        args.text_dim = text_encoder_hidden_size(args.text_encoder_id, hf_token)
        tag = (
            "monet-direct" if args.monet_direct
            else "edit-direct" if args.edit_direct
            else "captions-only"
        )
        print(
            f"[{tag}] on-the-fly text: {args.text_encoder_id} "
            f"(layer {args.text_encoder_layer}) -> text=({args.text_seq_len},{args.text_dim})",
            flush=True,
        )
    if args.monet_direct or args.edit_direct:
        args.skip_peek = True

    if args.skip_peek:
        print(f"Skipping dim peek; using explicit dims "
              f"(latent {args.latent_channels}x{args.latent_size}, "
              f"text {args.text_seq_len}x{args.text_dim}).", flush=True)
        record = None
    else:
        print("Peeking one cached record for dims (downloads one shard)...", flush=True)
        record = peek_record(args, hf_token)
    config = build_config(args, record)
    print(
        f"config: latent={config.latent_channels}x{config.latent_size} "
        f"text=({config.text_seq_len},{config.text_dim}) hidden={config.hidden_dim} "
        f"depth={config.depth} mixer={config.mixer_type} extra_in={config.extra_input_channels}",
        flush=True,
    )

    module = BoomerLit(
        config,
        hf_token=hf_token,
        monet_direct=args.monet_direct,
        fill_training=args.fill_training,
        fill_mask_probs=args.fill_mask_probs,
        cond_input_dropout=args.cond_input_dropout,
        edit_direct=args.edit_direct,
        edit_vae_id=args.edit_vae_id,
        edit_scaling_factor=args.edit_scaling_factor,
        edit_encode_chunk=args.edit_encode_chunk,
        fsdp2=args.fsdp2,
        text_encoder_id=args.text_encoder_id,
        text_encoder_layer=args.text_encoder_layer,
        text_encoder_max_length=args.text_encoder_max_length,
        text_encoder_instruction=args.text_encoder_instruction,
        lr=args.lr,
        weight_decay=args.weight_decay,
        beta1=args.beta1,
        beta2=args.beta2,
        fused_adamw=args.fused_adamw,
        warmup_steps=args.warmup_steps,
        lr_scheduler=args.lr_scheduler,
        min_lr_ratio=args.min_lr_ratio,
        max_steps=args.max_steps,
        cond_dropout=args.cond_dropout,
        t_sampler=args.t_sampler,
        t_sampler_final=args.t_sampler_final,
        t_sampler_final_step=args.t_sampler_final_step,
        t_bias=args.t_bias,
        flow_shift=args.flow_shift,
        logit_mean=args.logit_mean,
        logit_std=args.logit_std,
        plateau_low=args.plateau_low,
        plateau_high=args.plateau_high,
        stretch_q_low=args.stretch_q_low,
        stretch_q_high=args.stretch_q_high,
        timestep_weight=args.timestep_weight,
        min_snr_gamma=args.min_snr_gamma,
        latent_mean=args.latent_mean,
        latent_std=args.latent_std,
        repa_weight=args.repa_weight,
        repa_anneal_steps=args.repa_anneal_steps,
        dispersive_weight=args.dispersive_weight,
        dispersive_temp=args.dispersive_temp,
        dispersive_metric=args.dispersive_metric,
        dispersive_anneal_steps=args.dispersive_anneal_steps,
        compile=args.compile,
        compile_mode=args.compile_mode,
    )

    # Optionally seed the model from a checkpoint's weights without restoring the
    # optimizer/scheduler/step (dataset continuation / fine-tuning).
    if args.resume and args.resume_weights_only:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        state = ckpt.get("state_dict", ckpt)
        missing, unexpected = module.load_state_dict(state, strict=False)
        print(f"resume-weights-only: loaded weights "
              f"(missing={len(missing)}, unexpected={len(unexpected)})", flush=True)

    if args.fill_training or args.edit_direct:
        # The reserved-channel embedder slice was zero-init'ed and receives zero
        # gradient while fed zeros, so a pure-t2i seed must still be exactly 0 here
        # (verified on the v6 512 checkpoint). Non-zero = already fill/edit-trained.
        eager = getattr(module.model, "_orig_mod", module.model)
        extra_absmax = float(eager.x_embedder.weight[:, config.latent_channels:].abs().max())
        state = (
            "pristine zero (t2i seed; conditioning pathway lights up from scratch)"
            if extra_absmax == 0.0
            else "non-zero (seed already fill/edit-trained)"
        )
        print(f"[cond-input] x_embedder reserved-slice abs-max={extra_absmax:.3e}: {state}",
              flush=True)
        if args.expect_pristine_cond_input and extra_absmax != 0.0:
            raise SystemExit(
                "--expect-pristine-cond-input: reserved slice is non-zero — this is not a "
                "pure-t2i seed. Drop the flag if resuming E1 / seeding E2 from an E1 checkpoint."
            )

    datamodule = BoomerData(
        cache_wds=args.cache_wds,
        local_dir=args.wds_local_dir,
        prefetch_shards=args.wds_prefetch_shards,
        download_workers=args.wds_download_workers,
        shuffle_buffer=args.wds_shuffle_buffer,
        hf_token=hf_token,
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
        decode_queue_size=args.wds_decode_queue,
        shard_shuffle=args.wds_shard_shuffle,
        pin_memory=args.pin_memory,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers,
        download_max_retries=args.wds_download_max_retries,
        max_consecutive_skips=args.wds_max_consecutive_skips,
        parquet_read_batch_size=args.parquet_read_batch_size,
        parquet_memory_trim_every=args.parquet_memory_trim_every,
        parquet_prefetch_ranges=args.parquet_prefetch_ranges,
        parquet_range_size_mib=args.parquet_range_size_mib,
        captions_only=args.cache_captions_only,
        monet_direct=args.monet_direct,
        monet_latent_key=args.monet_latent_key,
        monet_caption_keys=tuple(
            key.strip() for key in args.monet_caption_keys.split(",") if key.strip()
        ),
        monet_caption_probs=args.monet_caption_probs,  # dataset parses CSV/';' groups
        monet_weights=(
            tuple(float(v) for v in args.monet_weights.split(",") if v.strip())
            if args.monet_weights else None
        ),
        # Only fetch the 1536-d DINO column when REPA actually consumes it: otherwise
        # it is streamed, decoded, collated, and H2D-copied every step just to be
        # discarded by training_step.
        monet_dino_key=(
            MONET_DINO_KEY
            if args.monet_dino and args.repa_dim > 0 and args.repa_weight > 0
            else None
        ),
        monet_min_aesthetic=args.monet_min_aesthetic,
        monet_max_nsfw=args.monet_max_nsfw,
        monet_max_watermark=args.monet_max_watermark,
        edit_direct=args.edit_direct,
        edit_resolution=args.edit_resolution,
        edit_min_o_score=args.edit_min_o_score,
        edit_tasks=(
            tuple(t.strip() for t in args.edit_tasks.split(",") if t.strip())
            if args.edit_tasks else None
        ),
        edit_prompt_pick=args.edit_prompt_pick,
        edit_min_short_side=args.edit_min_short_side,
    )

    callbacks: list[Callback] = [
        ModelCheckpoint(
            dirpath=args.checkpoint_dir,
            every_n_train_steps=args.save_every_steps,
            save_top_k=-1,
            save_last=True,
            filename="boomer-{step}",
        ),
        LearningRateMonitor(logging_interval="step"),
        # In Colab/non-TTY output tqdm reprints a line every refresh, so throttle the
        # progress bar to the same cadence as metric logging instead of every step.
        StepUnitProgressBar(refresh_rate=max(1, args.log_every_steps)),
    ]
    ema_cb = None
    if args.ema_decay > 0.0:
        ema_cb = EMACallback(decay=args.ema_decay, update_every=args.ema_update_every)
        callbacks.append(ema_cb)
    if args.sample_every_steps > 0:
        callbacks.append(SamplePreviewCallback(
            every_n_steps=args.sample_every_steps,
            prompts_cache=args.sample_prompts_cache,
            ema_callback=ema_cb,
            use_ema=args.sample_weights == "ema",
            steps=args.sample_steps,
            sampler=args.sample_sampler,
            stork_substeps=args.sample_stork_substeps,
            cfg_scale=args.sample_cfg_scale,
            cfg_rescale=args.sample_cfg_rescale,
            cfg_interval=(args.sample_cfg_interval_low, args.sample_cfg_interval_high),
            flow_shift=args.flow_shift,
            latent_mean=args.latent_mean,
            latent_std=args.latent_std,
            seed=args.sample_seed,
            dtype=args.dtype or "bf16",
            vae_id=args.sample_vae_id,
            scaling_factor=args.sample_scaling_factor,
            max_length=args.text_seq_len,
            hf_token=hf_token,
            encode_on_the_fly=args.monet_direct or args.cache_captions_only or args.edit_direct,
            prompts_file=args.sample_prompts_file,
        ))
    if (args.fill_training or args.edit_direct) and args.sample_every_steps > 0 \
            and args.sample_cond_count > 0:
        callbacks.append(CondSamplePreviewCallback(
            mode="fill" if args.fill_training else "edit",
            every_n_steps=args.sample_every_steps,
            count=args.sample_cond_count,
            ema_callback=ema_cb,
            use_ema=args.sample_weights == "ema",
            steps=args.sample_steps,
            sampler=args.sample_sampler,
            stork_substeps=args.sample_stork_substeps,
            cfg_scale=args.sample_cfg_scale,
            cfg_rescale=args.sample_cfg_rescale,
            cfg_interval=(args.sample_cfg_interval_low, args.sample_cfg_interval_high),
            flow_shift=args.flow_shift,
            latent_mean=args.latent_mean,
            latent_std=args.latent_std,
            seed=args.sample_seed,
            dtype=args.dtype or "bf16",
            vae_id=args.sample_vae_id,
            scaling_factor=args.sample_scaling_factor,
            hf_token=hf_token,
            fill_mask_probs=parse_fill_mask_probs(args.fill_mask_probs),
            encode_chunk=args.edit_encode_chunk,
            image_cfg_scale=args.sample_image_cfg,
        ))
    # Appended AFTER ModelCheckpoint so a checkpoint written this step already exists on
    # disk when this callback's hook runs and enqueues it for async upload.
    upload_cb: CheckpointUploadCallback | None = None
    if args.checkpoint_upload_remote:
        upload_cb = CheckpointUploadCallback(
            checkpoint_dir=args.checkpoint_dir,
            remote=args.checkpoint_upload_remote,
            local_retain=args.checkpoint_local_retain,
            keep_uploaded=args.checkpoint_keep_uploaded,
            rclone_bin=args.checkpoint_rclone_bin,
            chunk_size=args.checkpoint_upload_chunk_size,
            hf_bin=args.checkpoint_hf_bin,
        )
        callbacks.append(upload_cb)

    precision = args.precision
    if args.dtype is not None:
        precision = {"bf16": "bf16-mixed", "fp16": "16-mixed", "fp32": "32-true"}[args.dtype]

    logger: Any = True
    if args.wandb:
        # A stable `id` (not the display `name`) is what lets W&B resume the same run
        # across Colab sessions. Without it, Lightning's WandbLogger mints a fresh random
        # id every launch, so each relaunch becomes a NEW run even with the same name.
        run_id = args.wandb_run_id or os.environ.get("WANDB_RUN_ID")
        logger = WandbLogger(
            project=args.wandb_project,
            name=args.wandb_run_name,
            id=run_id,
            resume="allow" if run_id else None,
            entity=args.wandb_entity,
            mode=args.wandb_mode,
            save_dir=args.checkpoint_dir,
        )
        # Record the full run config (model + hparams) in the W&B run.
        logger.log_hyperparams({**vars(args), **{f"model/{k}": v for k, v in vars(config).items()}})

    trainer = pl.Trainer(
        accelerator="cpu" if args.device == "cpu" else "gpu",
        devices=args.devices,
        num_nodes=args.num_nodes,
        strategy=strategy,
        precision=precision,
        max_steps=args.max_steps,
        gradient_clip_val=args.grad_clip,
        accumulate_grad_batches=args.accumulate,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=args.log_every_steps,
        enable_checkpointing=True,
        default_root_dir=args.checkpoint_dir,
    )
    # resume-weights-only already loaded the weights above; don't restore trainer state.
    resume_ckpt = None if args.resume_weights_only else args.resume
    fit_error: BaseException | None = None
    try:
        trainer.fit(module, datamodule=datamodule, ckpt_path=resume_ckpt)
    except BaseException as exc:  # noqa: BLE001 - see the teardown note below
        fit_error = exc

    # Flush pending checkpoint uploads synchronously before any exit path. Lightning's
    # on_fit_end async drain can be pre-empted by the teardown crash (final checkpoint saved
    # locally but its Drive copy killed mid-flight), and the os._exit() fast-path below would
    # kill the upload worker — so the newest checkpoint reaches the remote only if we drain
    # here, regardless of how fit() ended.
    if upload_cb is not None:
        try:
            upload_cb.finalize_uploads(trainer)
        except Exception as exc:  # noqa: BLE001
            print(f"[ckpt-upload] finalize failed: {exc}", flush=True)

    completed = trainer.global_step >= args.max_steps and not trainer.interrupted

    # Lightning tears down the persistent DataLoader workers at fit-end. Our streaming
    # IterableDataset loops forever, so a worker is typically blocked in an in-flight HF
    # range read when the shutdown signal lands; it can't unwind the C-level httpx/Arrow
    # call, the interpreter finalizes under it ("PyGILState_Release ... finalizing"), and
    # it abort()s -> "DataLoader worker ... killed by signal: Aborted", which propagates
    # out of fit() as a teardown RuntimeError. When this happens AFTER max_steps was
    # reached, training and checkpoints are already complete, so treat it as success.
    # Re-raise anything that happened BEFORE completion (a real OOM/bug/^C).
    if fit_error is not None and not completed:
        raise fit_error
    if fit_error is not None:
        print(f"[teardown] reached max_steps={args.max_steps}; ignoring DataLoader "
              f"shutdown error: {type(fit_error).__name__}: {fit_error}", flush=True)

    # Close the W&B run cleanly so it reads "finished" instead of lingering as "crashed".
    # Lightning's WandbLogger does NOT reliably call finish() on fit-end. An early-stopped
    # / ^C / OOM session never sets completed, so it stays crashed-but-resumable under
    # WANDB_RESUME=allow.
    if args.wandb and completed:
        import wandb

        if wandb.run is not None:
            wandb.finish()
            print(f"[wandb] run finished cleanly at step {trainer.global_step}", flush=True)

    if completed:
        # Durable state is flushed above: checkpoints uploaded via finalize_uploads(), W&B
        # finished. Skip Python's interpreter finalization, which otherwise re-triggers the
        # same httpx/Arrow worker-thread GIL fatal on the way out and gives a non-zero exit.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
