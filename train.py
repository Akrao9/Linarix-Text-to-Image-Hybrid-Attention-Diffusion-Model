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
      --image-attn-every 6 --cross-attn-qk-norm --extra-input-channels 33 \
      --batch-size 64 --lr 1e-4 --cond-dropout 0.1 \
      --t-sampler plateau-logit-normal --flow-shift 1.5 --min-snr-gamma 5 \
      --ema-decay 0.9999 --max-steps 100000 \
      --checkpoint-dir /content/ckpt_stage0 --hf-token true
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:  # Lightning >= 2.0 namespace, with legacy fallback.
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
    from lightning.pytorch.loggers import WandbLogger
except ImportError:  # pragma: no cover
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
    from pytorch_lightning.loggers import WandbLogger

from boomer.datasets.latent_shards import collate_latent_records  # noqa: E402
from boomer.datasets.wds_latents import (  # noqa: E402
    PrefetchingWdsLatentCacheDataset,
    StreamingWdsLatentCacheDataset,
)
from boomer.models.boomer_fla_dit import BoomerFLADiTConfig  # noqa: E402
from boomer.models.factory import create_model  # noqa: E402
from boomer.sana_flow import (  # noqa: E402
    flow_min_snr_loss_weight,
    flow_training_loss_weight,
    noise_latent_at_sigma,
    sample_flow_sigmas,
)
from boomer.sana_latent_cache import hf_token_value  # noqa: E402


# --------------------------------------------------------------------------- #
# LightningModule
# --------------------------------------------------------------------------- #
class BoomerLit(pl.LightningModule):
    def __init__(self, model_config: BoomerFLADiTConfig, **hp: Any) -> None:
        super().__init__()
        self.save_hyperparameters(hp)
        self.config = model_config
        self.model = create_model(model_config)
        if hp.get("compile"):
            self.model = torch.compile(self.model, mode=hp.get("compile_mode", "default"))

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        hp = self.hparams
        x0 = batch["image_latent"]
        if hp.latent_mean != 0.0 or hp.latent_std != 1.0:
            x0 = (x0 - hp.latent_mean) / hp.latent_std

        text = batch["text_embedding"]
        mask = batch["attention_mask"]
        text, mask = self.model.apply_condition_dropout(text, mask, hp.cond_dropout)

        noise = torch.randn_like(x0)
        t = sample_flow_sigmas(
            x0.shape[0],
            device=x0.device,
            dtype=x0.dtype,
            sampler=hp.t_sampler,
            bias=hp.t_bias,
            flow_shift=hp.flow_shift,
            logit_mean=hp.logit_mean,
            logit_std=hp.logit_std,
            plateau_low=hp.plateau_low,
            plateau_high=hp.plateau_high,
        )
        x_t, _ = noise_latent_at_sigma(x0, t, noise=noise)
        target = noise - x0

        # REPA: align a pooled mid-depth feature to monet's DINOv2 CLS when enabled
        # (repa_dim > 0, repa_weight > 0, and the cache carries dino_embedding).
        repa_target = batch.get("dino_embedding")
        want_repa = (
            hp.repa_weight > 0.0
            and getattr(self.config, "repa_dim", 0) > 0
            and repa_target is not None
        )
        if want_repa:
            pred, repa_feat = self.model(x_t, t, text, mask, return_repa=True)
        else:
            pred = self.model(x_t, t, text, mask)
            repa_feat = None

        per_sample = (pred.float() - target.float()).pow(2).mean(dim=(1, 2, 3))
        if hp.timestep_weight:
            per_sample = per_sample * flow_training_loss_weight(t, target).mean(dim=(1, 2, 3))
        if hp.min_snr_gamma > 0.0:
            per_sample = per_sample * flow_min_snr_loss_weight(t, target, hp.min_snr_gamma).mean(dim=(1, 2, 3))
        loss = per_sample.mean()

        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=False, batch_size=x0.shape[0])

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
            loss = loss + repa_w * repa_loss
            self.log("train/repa_loss", repa_loss, prog_bar=True, on_step=True, on_epoch=False, batch_size=x0.shape[0])
            self.log("train/repa_w", repa_w, on_step=True, on_epoch=False, batch_size=x0.shape[0])

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
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(gradient_clip_val))

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
):
    """Remote hf:// caches use the SAME parallel download-ahead streaming as the
    caching tool (a thread-pool prefetch window into local_dir, delete after use);
    local shard globs read directly with plain streaming."""
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
        )
    return StreamingWdsLatentCacheDataset(
        paths=cache_wds,
        hf_token=hf_token,
        shardshuffle=shard_shuffle if shard_shuffle > 0 else True,
        shuffle_buffer=shuffle_buffer,
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
    ) -> None:
        super().__init__()
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

    def _dataset(self):
        return make_wds_dataset(
            cache_wds=self.cache_wds,
            local_dir=self.local_dir,
            prefetch_shards=self.prefetch_shards,
            download_workers=self.download_workers,
            shuffle_buffer=self.shuffle_buffer,
            hf_token=self.hf_token,
            seed=self.seed,
            decode_queue_size=self.decode_queue_size,
            shard_shuffle=self.shard_shuffle,
        )

    def train_dataloader(self) -> DataLoader:
        # The rolling remote cache REQUIRES num_workers=0 (single-process shard
        # download); local streaming can use multiple workers.
        workers = 0 if is_remote_cache(self.cache_wds) else self.num_workers
        persistent = self.persistent_workers if self.persistent_workers is not None else workers > 0
        kwargs = {}
        if workers > 0:
            # Each worker stages prefetch_factor batches ahead so the pinned-memory +
            # non_blocking H2D copy always has something ready, overlapping decode with compute.
            kwargs["prefetch_factor"] = self.prefetch_factor
        return DataLoader(
            self._dataset(),
            batch_size=self.batch_size,
            num_workers=workers,
            collate_fn=collate_latent_records,
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
            ema_v.mul_(self.decay).add_(sd[key].detach().float(), alpha=1.0 - self.decay)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint) -> None:
        checkpoint["ema"] = self.ema

    def on_load_checkpoint(self, trainer, pl_module, checkpoint) -> None:
        if "ema" in checkpoint:
            self.ema = {k: v.float() for k, v in checkpoint["ema"].items()}


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
    )
    return next(iter(ds))


def build_config(args: argparse.Namespace, record: Any) -> BoomerFLADiTConfig:
    return BoomerFLADiTConfig(
        latent_channels=int(record.image_latent.shape[0]),
        latent_size=int(record.image_latent.shape[-1]),
        text_dim=int(record.text_embedding.shape[-1]),
        text_seq_len=int(record.text_embedding.shape[-2]),
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
        image_attention_rope=args.image_rope,
        image_rope_theta=args.image_rope_theta,
        cross_attention_backend=args.cross_attn_backend,
        cross_attention_qk_norm=args.cross_attn_qk_norm,
        parallel_block=args.parallel_block,
        dual_stream_depth=args.dual_stream_depth,
        multimodal_coord_ids=args.multimodal_coord_ids,
        use_abs_pos_embed=args.abs_pos_embed,
        patch_size=args.patch_size,
        extra_input_channels=args.extra_input_channels,
        gradient_checkpointing=args.gradient_checkpointing,
        repa_dim=args.repa_dim,
        repa_layer=args.repa_layer,
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
    p.add_argument("--hf-token", default=None, help="HF token, or 'true' for the logged-in token.")
    p.add_argument("--latent-mean", type=float, default=0.0)
    p.add_argument("--latent-std", type=float, default=1.0)

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
    p.add_argument("--image-attn-backend", default="sdpa", choices=["sdpa", "flash3", "flash4", "auto"])
    p.add_argument("--image-rope", action=bo, default=False,
                   help="Apply 2D RoPE to full image self-attention anchor blocks.")
    p.add_argument("--image-rope-theta", type=float, default=10000.0,
                   help="RoPE frequency base for --image-rope.")
    p.add_argument("--cross-attn-backend", default="sdpa", choices=["sdpa", "xformers", "auto", "mha"])
    p.add_argument("--cross-attn-qk-norm", action=bo, default=True,
                   help="Sana-style q/k RMSNorm inside text cross-attention.")
    p.add_argument("--parallel-block", action=bo, default=False,
                   help="FLUX/SD3-style parallel residual branches inside blocks.")
    p.add_argument("--dual-stream-depth", type=int, default=0)
    p.add_argument("--multimodal-coord-ids", action=bo, default=False)
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
                   choices=["uniform", "high", "low", "logit-normal", "bounded-logit-normal", "plateau-logit-normal"])
    p.add_argument("--t-bias", type=float, default=2.0)
    p.add_argument("--flow-shift", type=float, default=1.5)
    p.add_argument("--logit-mean", type=float, default=0.0)
    p.add_argument("--logit-std", type=float, default=1.0)
    p.add_argument("--plateau-low", type=float, default=0.0)
    p.add_argument("--plateau-high", type=float, default=1.0)
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
    p.add_argument("--checkpoint-dir", default="/content/ckpt_stage0")
    p.add_argument("--save-every-steps", "--save-every", dest="save_every_steps", type=int, default=2000)
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
    p.add_argument("--resume-weights-only", action="store_true",
                   help="With --resume, load model weights only; reset optimizer/scheduler/step.")
    p.add_argument("--compile", action="store_true",
                   help="Wrap the training model in torch.compile. EMA is left uncompiled.")
    p.add_argument("--compile-mode",
                   choices=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"],
                   default="max-autotune-no-cudagraphs",
                   help="torch.compile mode. max-autotune-no-cudagraphs is the safe Blackwell default.")
    return p


def main() -> None:
    args = build_parser().parse_args()
    pl.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision("high")

    hf_token = hf_token_value(args.hf_token)

    print("Peeking one cached record for dims...", flush=True)
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
        t_bias=args.t_bias,
        flow_shift=args.flow_shift,
        logit_mean=args.logit_mean,
        logit_std=args.logit_std,
        plateau_low=args.plateau_low,
        plateau_high=args.plateau_high,
        timestep_weight=args.timestep_weight,
        min_snr_gamma=args.min_snr_gamma,
        latent_mean=args.latent_mean,
        latent_std=args.latent_std,
        repa_weight=args.repa_weight,
        repa_anneal_steps=args.repa_anneal_steps,
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
    ]
    if args.ema_decay > 0.0:
        callbacks.append(EMACallback(decay=args.ema_decay, update_every=args.ema_update_every))

    precision = args.precision
    if args.dtype is not None:
        precision = {"bf16": "bf16-mixed", "fp16": "16-mixed", "fp32": "32-true"}[args.dtype]

    logger: Any = True
    if args.wandb:
        logger = WandbLogger(
            project=args.wandb_project,
            name=args.wandb_run_name,
            entity=args.wandb_entity,
            mode=args.wandb_mode,
            save_dir=args.checkpoint_dir,
        )
        # Record the full run config (model + hparams) in the W&B run.
        logger.log_hyperparams({**vars(args), **{f"model/{k}": v for k, v in vars(config).items()}})

    trainer = pl.Trainer(
        accelerator="cpu" if args.device == "cpu" else "gpu",
        devices=1,
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
    trainer.fit(module, datamodule=datamodule, ckpt_path=resume_ckpt)


if __name__ == "__main__":
    main()
