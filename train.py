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
"""

from __future__ import annotations

import argparse
import math
import os
import queue
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
from boomer.datasets.wds_latents import (  # noqa: E402
    ParquetStreamingLatentCacheDataset,
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
from boomer.latent_norm import denormalize_latent  # noqa: E402
from boomer.sampling import sample_latent  # noqa: E402
from boomer.sana_latent_cache import (  # noqa: E402
    CacheConfig,
    decode_image_latent,
    hf_token_value,
    load_vae,
    tensor_to_image,
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
    def __init__(self, model_config: BoomerFLADiTConfig, **hp: Any) -> None:
        super().__init__()
        self.save_hyperparameters({**hp, "model_config": model_config.__dict__})
        self.config = model_config
        self.model = create_model(model_config)
        if hp.get("compile"):
            self.model = torch.compile(self.model, mode=hp.get("compile_mode", "default"))

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        # Keep inference self-describing even though Lightning's native state uses
        # `state_dict`/`hyper_parameters`. Older sampling tools consume these fields.
        checkpoint["config"] = dict(self.config.__dict__)
        checkpoint["train_args"] = {
            key: value for key, value in dict(self.hparams).items() if key != "model_config"
        }

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        hp = self.hparams
        x0 = batch["image_latent"]
        if hp.latent_mean != 0.0 or hp.latent_std != 1.0:
            x0 = (x0 - hp.latent_mean) / hp.latent_std

        text = batch["text_embedding"]
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
        download_max_retries: int = 2,
        max_consecutive_skips: int = 8,
        parquet_read_batch_size: int = 32,
        parquet_memory_trim_every: int = 4096,
        parquet_prefetch_ranges: bool = True,
        parquet_range_size_mib: int = 64,
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

    def _dataset(self, *, seed_salt: int = 0):
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
        )

    def train_dataloader(self) -> DataLoader:
        # The rolling remote TAR cache REQUIRES num_workers=0 (single-process shard
        # download). Parquet range-streaming and local globs use multiple workers
        # (datasets shards parquet files across workers automatically).
        parquet = is_parquet_cache(self.cache_wds)
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
            self.ema = {k: v.float() for k, v in checkpoint["ema"].items()}


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
    ) -> None:
        self.every_n_steps = every_n_steps
        self.prompts_cache = prompts_cache
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

    ``remote`` may be either an rclone remote (e.g. ``gdrive:Boomer/ckpts``) or an
    absolute directory path such as a mounted Drive folder
    (``/content/drive/MyDrive/Boomer/ckpts``); the latter is copied via tmp-file +
    atomic rename so a reader never sees a partial file.
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
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.remote = remote
        self.local_retain = max(1, int(local_retain))
        self.keep_uploaded = int(keep_uploaded)  # 0 = keep all uploaded
        self.rclone_bin = rclone_bin
        self.chunk_size = chunk_size
        # rclone remotes look like "name:path"; an absolute/relative dir does not.
        self.use_rclone = ":" in remote and not os.path.isabs(remote)
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
        if self.use_rclone:
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
                print(f"[ckpt-upload] FAILED {Path(path).name}: {exc}", flush=True)
            finally:
                self._q.task_done()

    def _upload_one(self, local: Path) -> None:
        if not local.exists():
            return
        if self.use_rclone:
            dest = f"{self.remote.rstrip('/')}/{local.name}"
            cmd = [self.rclone_bin, "copyto", str(local), dest,
                   "--drive-chunk-size", self.chunk_size, "--no-traverse"]
            subprocess.run(cmd, check=True, capture_output=True)
        else:
            dest = Path(self.remote) / local.name
            tmp = Path(self.remote) / (local.name + ".uploading")
            shutil.copy2(str(local), str(tmp))  # copy to tmp...
            os.replace(str(tmp), str(dest))     # ...then atomic swap into place
        print(f"[ckpt-upload] uploaded {local.name}", flush=True)

    # ---- pruning ---------------------------------------------------------- #
    def _local_step_ckpts(self) -> list[Path]:
        # All step checkpoints except the rolling 'last.ckpt' (which is overwritten).
        return sorted(
            (p for p in self.checkpoint_dir.glob("*.ckpt") if p.name != "last.ckpt"),
            key=lambda p: p.stat().st_mtime,
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
        if self.use_rclone:
            out = subprocess.run([self.rclone_bin, "lsf", self.remote],
                                 capture_output=True, text=True)
            names = sorted(n for n in out.stdout.split() if n.endswith(".ckpt") and n != "last.ckpt")
            for n in names[: max(0, len(names) - self.keep_uploaded)]:
                subprocess.run([self.rclone_bin, "deletefile", f"{self.remote.rstrip('/')}/{n}"],
                               capture_output=True)
        else:
            files = sorted(
                (p for p in Path(self.remote).glob("*.ckpt") if p.name != "last.ckpt"),
                key=lambda p: p.stat().st_mtime,
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
    )
    return next(iter(ds))


def build_config(args: argparse.Namespace, record: Any) -> BoomerFLADiTConfig:
    # With --skip-peek, record is None and dims come from the explicit flags (no shard
    # download at startup). Otherwise dims are read from one peeked cache record.
    if record is not None:
        latent_channels = int(record.image_latent.shape[0])
        latent_size = int(record.image_latent.shape[-1])
        text_dim = int(record.text_embedding.shape[-1])
        text_seq_len = int(record.text_embedding.shape[-2])
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
                        "all blocks (per-block scale_shift_table keeps layer variation). Saves ~27% "
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
    p.add_argument("--checkpoint-dir", default="/content/ckpt_stage0",
                   help="Local directory for checkpoints (always written here first).")
    p.add_argument("--save-every-steps", "--save-every", dest="save_every_steps", type=int, default=2000)
    # Async checkpoint upload: save locally, copy to Drive/rclone in the background,
    # keep only the newest N local checkpoints (older ones deleted after they upload).
    p.add_argument("--checkpoint-upload-remote", default=None,
                   help="Background upload destination: an rclone remote "
                        "(e.g. 'gdrive:Boomer/ckpts') OR an absolute dir such as a mounted "
                        "Drive folder ('/content/drive/MyDrive/Boomer/ckpts'). Off if unset.")
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
    pl.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision("high")

    hf_token = hf_token_value(args.hf_token)

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
