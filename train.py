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
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:  # Lightning >= 2.0 namespace, with legacy fallback.
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
except ImportError:  # pragma: no cover
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import Callback, LearningRateMonitor, ModelCheckpoint

from boomer.datasets.latent_shards import collate_latent_records  # noqa: E402
from boomer.datasets.wds_latents import RollingStreamingWdsLatentCacheDataset  # noqa: E402
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

        pred = self.model(x_t, t, text, mask)
        per_sample = (pred.float() - target.float()).pow(2).mean(dim=(1, 2, 3))
        if hp.timestep_weight:
            per_sample = per_sample * flow_training_loss_weight(t, target).mean(dim=(1, 2, 3))
        if hp.min_snr_gamma > 0.0:
            per_sample = per_sample * flow_min_snr_loss_weight(t, target, hp.min_snr_gamma).mean(dim=(1, 2, 3))
        loss = per_sample.mean()

        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=False, batch_size=x0.shape[0])
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

        def lr_lambda(step: int) -> float:
            if warmup > 0 and step < warmup:
                return (step + 1) / warmup
            return 1.0

        scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)
        return {"optimizer": optim, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}


# --------------------------------------------------------------------------- #
# DataModule
# --------------------------------------------------------------------------- #
class BoomerData(pl.LightningDataModule):
    def __init__(
        self,
        *,
        cache_wds: str,
        local_dir: str,
        rolling_shards: int,
        shuffle_buffer: int,
        hf_token: str | bool | None,
        batch_size: int,
        seed: int,
    ) -> None:
        super().__init__()
        self.cache_wds = cache_wds
        self.local_dir = local_dir
        self.rolling_shards = rolling_shards
        self.shuffle_buffer = shuffle_buffer
        self.hf_token = hf_token
        self.batch_size = batch_size
        self.seed = seed

    def _dataset(self) -> RollingStreamingWdsLatentCacheDataset:
        return RollingStreamingWdsLatentCacheDataset(
            paths=self.cache_wds,
            local_dir=self.local_dir,
            rolling_shards=self.rolling_shards,
            hf_token=self.hf_token,
            shuffle_buffer=self.shuffle_buffer,
            seed=self.seed,
        )

    def train_dataloader(self) -> DataLoader:
        # Rolling cache REQUIRES num_workers=0 (single-process shard download).
        return DataLoader(
            self._dataset(),
            batch_size=self.batch_size,
            num_workers=0,
            collate_fn=collate_latent_records,
            pin_memory=True,
            drop_last=True,
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
    """Pull one cached record to read latent/text shapes (downloads one shard)."""
    ds = RollingStreamingWdsLatentCacheDataset(
        paths=args.cache_wds,
        local_dir=args.wds_local_dir,
        rolling_shards=1,
        hf_token=hf_token,
        shuffle_buffer=1,
        seed=args.seed,
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
        mixer_type=args.mixer_type,
        fla_mode=args.fla_mode,
        fla_bidirectional=args.fla_bidirectional,
        use_short_conv=args.use_short_conv,
        conv_size=args.conv_size,
        image_attention_every=args.image_attn_every,
        image_attention_backend=args.image_attn_backend,
        cross_attention_backend=args.cross_attn_backend,
        cross_attention_qk_norm=args.cross_attn_qk_norm,
        dual_stream_depth=args.dual_stream_depth,
        multimodal_coord_ids=args.multimodal_coord_ids,
        use_abs_pos_embed=args.abs_pos_embed,
        patch_size=args.patch_size,
        extra_input_channels=args.extra_input_channels,
        gradient_checkpointing=args.gradient_checkpointing,
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
    p.add_argument("--wds-rolling-shards", type=int, default=2)
    p.add_argument("--wds-shuffle-buffer", type=int, default=2000)
    p.add_argument("--hf-token", default=None, help="HF token, or 'true' for the logged-in token.")
    p.add_argument("--latent-mean", type=float, default=0.0)
    p.add_argument("--latent-std", type=float, default=1.0)

    # model
    p.add_argument("--hidden-dim", type=int, default=896)
    p.add_argument("--depth", type=int, default=24)
    p.add_argument("--num-heads", type=int, default=14)
    p.add_argument("--mlp-ratio", type=float, default=2.5)
    p.add_argument("--mixer-type", default="fla_gdn2")
    p.add_argument("--fla-mode", default="chunk", choices=["chunk", "fused_chunk"])
    p.add_argument("--fla-bidirectional", action=bo, default=True)
    p.add_argument("--use-short-conv", action="store_true")
    p.add_argument("--conv-size", type=int, default=4)
    p.add_argument("--image-attn-every", type=int, default=6)
    p.add_argument("--image-attn-backend", default="sdpa")
    p.add_argument("--cross-attn-backend", default="sdpa")
    p.add_argument("--cross-attn-qk-norm", action="store_true")
    p.add_argument("--dual-stream-depth", type=int, default=0)
    p.add_argument("--multimodal-coord-ids", action="store_true")
    p.add_argument("--abs-pos-embed", action=bo, default=True)
    p.add_argument("--y-norm", action=bo, default=True)
    p.add_argument("--patch-size", type=int, default=1)
    p.add_argument("--extra-input-channels", type=int, default=0,
                   help="FLUX-Fill-style reserved channels (33 = masked latent + mask) for future inpainting.")
    p.add_argument("--gradient-checkpointing", action="store_true")

    # optim / flow
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--fused-adamw", action=bo, default=True)
    p.add_argument("--warmup-steps", type=int, default=1000)
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
    p.add_argument("--max-steps", type=int, default=100000)
    p.add_argument("--precision", default="bf16-mixed")
    p.add_argument("--checkpoint-dir", default="/content/ckpt_stage0")
    p.add_argument("--save-every-steps", type=int, default=2000)
    p.add_argument("--log-every-steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", default=None, help="Path to a .ckpt to resume from.")
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
    )

    datamodule = BoomerData(
        cache_wds=args.cache_wds,
        local_dir=args.wds_local_dir,
        rolling_shards=args.wds_rolling_shards,
        shuffle_buffer=args.wds_shuffle_buffer,
        hf_token=hf_token,
        batch_size=args.batch_size,
        seed=args.seed,
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

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        precision=args.precision,
        max_steps=args.max_steps,
        gradient_clip_val=args.grad_clip,
        accumulate_grad_batches=args.accumulate,
        callbacks=callbacks,
        log_every_n_steps=args.log_every_steps,
        enable_checkpointing=True,
        default_root_dir=args.checkpoint_dir,
    )
    trainer.fit(module, datamodule=datamodule, ckpt_path=args.resume)


if __name__ == "__main__":
    main()
