"""Overfit a tiny latent DiT on cached Boomer records."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import sys
import time
from collections import deque
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from boomer.checkpoint_utils import (  # noqa: E402
    collect_train_args,
    config_from_checkpoint,
    load_checkpoint,
    load_compatible_state_dict,
    save_training_checkpoints,
)
from boomer.datasets.latent_shards import LatentCacheDataset, collate_latent_records, load_latent_record  # noqa: E402
from boomer.datasets.wds_latents import RollingStreamingWdsLatentCacheDataset, StreamingWdsLatentCacheDataset  # noqa: E402
from boomer.latent_norm import LatentStat, denormalize_latent, latent_stat_summary, normalize_latent  # noqa: E402
from boomer.models.factory import Config, create_model  # noqa: E402
from boomer.models.latent_dit import LatentDiTConfig  # noqa: E402
from boomer.models.mamba_anchor_dit import MambaAnchorDiTConfig  # noqa: E402
from boomer.models.boomer_fla_dit import BoomerFLADiTConfig  # noqa: E402
from boomer.denoise import denoise_latent  # noqa: E402
from boomer.sana_flow import (  # noqa: E402
    flow_min_snr_loss_weight,
    flow_training_loss_weight,
    noise_latent_at_sigma,
    sample_flow_sigmas,
)
from boomer.sana_latent_cache import CacheConfig, decode_image_latent, hf_token_value, load_vae, tensor_to_image  # noqa: E402

def format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    if seconds < 3600.0:
        return f"{seconds / 60.0:.1f}m"
    return f"{seconds / 3600.0:.1f}h"


RESUME_TRAIN_ARG_KEYS = (
    "cond_dropout",
    "ema_decay",
    "t_sampler",
    "t_bias",
    "flow_shift",
    "logit_mean",
    "logit_std",
    "plateau_low",
    "plateau_high",
    "y_norm",
    "y_norm_scale_factor",
    "grad_clip",
    "timestep_weight",
    "min_snr_gamma",
    "lr_scheduler",
    "warmup_steps",
    "min_lr_ratio",
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def apply_resume_train_args(args: argparse.Namespace, parser: argparse.ArgumentParser, train_args: dict[str, Any]) -> None:
    for key in RESUME_TRAIN_ARG_KEYS:
        if key in train_args and getattr(args, key) == parser.get_default(key):
            setattr(args, key, train_args[key])


def sample_flow_batch(
    x0: torch.Tensor,
    *,
    t_sampler: str,
    t_bias: float,
    flow_shift: float,
    logit_mean: float,
    logit_std: float,
    plateau_low: float,
    plateau_high: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    noise = torch.randn_like(x0)
    t = sample_flow_sigmas(
        x0.shape[0],
        device=x0.device,
        dtype=x0.dtype,
        sampler=t_sampler,
        bias=t_bias,
        flow_shift=flow_shift,
        logit_mean=logit_mean,
        logit_std=logit_std,
        plateau_low=plateau_low,
        plateau_high=plateau_high,
    )
    x_t, _ = noise_latent_at_sigma(x0, t, noise=noise)
    velocity = noise - x0
    return x_t, t, velocity


def sigma_bin_means(per_sample_loss: torch.Tensor, sigma: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return low/mid/high sigma loss means without syncing to host."""
    sigma_f = sigma.float()
    loss_f = per_sample_loss.float()
    masks = {
        "low": sigma_f < 0.33,
        "mid": (sigma_f >= 0.33) & (sigma_f < 0.66),
        "high": sigma_f >= 0.66,
    }
    means: dict[str, torch.Tensor] = {}
    for name, mask in masks.items():
        weight = mask.to(dtype=loss_f.dtype)
        # Empty bins are reported as 0.0 to avoid a Python-side mask.any()
        # CUDA sync in the hot path.
        means[name] = (loss_f * weight).sum() / weight.sum().clamp_min(1.0)
    return means


def lr_multiplier(
    step: int,
    total_steps: int,
    *,
    scheduler: str,
    warmup_steps: int,
    min_lr_ratio: float,
) -> float:
    if scheduler == "constant":
        return 1.0
    if scheduler != "linear-warmup-cosine":
        raise ValueError(f"Unsupported lr scheduler: {scheduler}")
    if warmup_steps > 0 and step <= warmup_steps:
        return max(step, 1) / warmup_steps

    decay_steps = max(total_steps - warmup_steps, 1)
    progress = min(max((step - warmup_steps) / decay_steps, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def move_batch_tensor(
    tensor: torch.Tensor,
    *,
    device: str,
    dtype: torch.dtype | None = None,
    non_blocking: bool = False,
) -> torch.Tensor:
    moved = tensor.to(device=device, non_blocking=non_blocking)
    if dtype is not None and moved.dtype != dtype:
        moved = moved.to(dtype=dtype)
    return moved


def load_latent_normalizer(args: argparse.Namespace, resume_train_args: dict[str, Any]) -> tuple[LatentStat, LatentStat]:
    stats: dict[str, Any] = {}
    if args.latent_stats_json:
        stats_path = Path(args.latent_stats_json)
        stats = json.loads(stats_path.read_text(encoding="utf-8"))

    if args.latent_mean is not None:
        latent_mean: LatentStat = float(args.latent_mean)
    elif stats and args.latent_stats_mode == "channel" and "channel_mean" in stats:
        latent_mean = [float(value) for value in stats["channel_mean"]]
    elif stats:
        latent_mean = float(stats["global_mean"])
    else:
        latent_mean = resume_train_args.get("latent_mean", 0.0)

    if args.latent_std is not None:
        latent_std: LatentStat = float(args.latent_std)
    elif stats and args.latent_stats_mode == "channel" and "channel_std" in stats:
        latent_std = [float(value) for value in stats["channel_std"]]
    elif stats:
        latent_std = float(stats["global_std"])
    else:
        latent_std = resume_train_args.get("latent_std", 1.0)

    return latent_mean, latent_std


def build_model_config(args: argparse.Namespace, sample_record: Any, resume_checkpoint: dict[str, Any] | None) -> Config:
    if resume_checkpoint is not None:
        config = config_from_checkpoint(resume_checkpoint)
        train_args = resume_checkpoint.get("train_args") or {}
        updates: dict[str, Any] = {}
        if (
            getattr(args, "resume_weights_only", False)
            and isinstance(config, BoomerFLADiTConfig)
        ):
            if getattr(args, "fla_bidirectional", False):
                updates["fla_bidirectional"] = True
            if getattr(args, "parallel_block", False):
                updates["parallel_block"] = True
            if getattr(args, "multimodal_coord_ids", False):
                updates["multimodal_coord_ids"] = True
            if getattr(args, "dual_stream_depth", 0) > 0:
                updates["dual_stream_depth"] = int(args.dual_stream_depth)
            if not getattr(args, "abs_pos_embed", True):
                updates["use_abs_pos_embed"] = False
            if getattr(args, "patch_size", config.patch_size) != config.patch_size:
                updates["patch_size"] = int(args.patch_size)
            # Resolution continuation: let latent geometry update from the new data.
            # RoPE2D/coord-RoPE rebuild their grid tables from latent_size; persistent
            # RoPE buffers that changed shape are silently skipped by load_compatible_state_dict,
            # so the model gets freshly-initialised tables for the new grid.
            new_latent_size = int(sample_record.image_latent.shape[-1])
            if new_latent_size != getattr(config, "latent_size", new_latent_size):
                updates["latent_size"] = new_latent_size
                print(
                    f"[resume-weights-only] latent_size override: "
                    f"{config.latent_size} -> {new_latent_size} "
                    f"(RoPE tables will be re-initialised for the new grid)",
                    flush=True,
                )
            new_latent_channels = int(sample_record.image_latent.shape[0])
            if new_latent_channels != getattr(config, "latent_channels", new_latent_channels):
                updates["latent_channels"] = new_latent_channels
            new_text_seq_len = int(sample_record.text_embedding.shape[-2])
            if new_text_seq_len != getattr(config, "text_seq_len", new_text_seq_len):
                updates["text_seq_len"] = new_text_seq_len
        if "y_norm" in train_args and not getattr(config, "y_norm", False):
            updates["y_norm"] = bool(train_args["y_norm"])
        if "y_norm_scale_factor" in train_args:
            updates["y_norm_scale_factor"] = float(train_args["y_norm_scale_factor"])
        if updates:
            config = replace(config, **updates)
        return config

    text_dim = int(sample_record.text_embedding.shape[-1])
    text_seq_len = int(sample_record.text_embedding.shape[-2])
    latent_channels = int(sample_record.image_latent.shape[0])
    latent_size = int(sample_record.image_latent.shape[-1])
    if args.model_type == "mamba_anchor":
        return MambaAnchorDiTConfig(
            latent_channels=latent_channels,
            latent_size=latent_size,
            text_dim=text_dim,
            text_seq_len=text_seq_len,
            hidden_dim=args.hidden_dim,
            depth=args.depth,
            num_heads=args.num_heads,
            mlp_ratio=args.mlp_ratio,
            y_norm=bool(args.y_norm),
            y_norm_scale_factor=args.y_norm_scale_factor,
            num_anchors=args.num_anchors,
            text_attention_every=args.text_attn_every,
            anchor_attention_every=args.anchor_attn_every,
            mixer_type=args.mixer_type,
            mamba_d_state=args.mamba_d_state,
            mamba_headdim=args.mamba_headdim,
            mamba_chunk_size=args.mamba_chunk_size,
            mamba_is_mimo=args.mamba_mimo,
        )
    if args.model_type == "boomer_fla":
        return BoomerFLADiTConfig(
            latent_channels=latent_channels,
            latent_size=latent_size,
            text_dim=text_dim,
            text_seq_len=text_seq_len,
            hidden_dim=args.hidden_dim,
            depth=args.depth,
            num_heads=args.num_heads,
            mlp_ratio=args.mlp_ratio,
            y_norm=bool(args.y_norm),
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
        )

    return LatentDiTConfig(
        latent_channels=latent_channels,
        latent_size=latent_size,
        text_dim=text_dim,
        text_seq_len=text_seq_len,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        cross_attention=args.cross_attn,
        y_norm=bool(args.y_norm),
        y_norm_scale_factor=args.y_norm_scale_factor,
    )


def create_ema_model(model: torch.nn.Module, *, dtype: torch.dtype) -> torch.nn.Module:
    ema_model = copy.deepcopy(model).to(dtype=dtype).eval()
    ema_model.requires_grad_(False)
    # Caches are built lazily on first update against the live source model.
    ema_model._boomer_ema_param_cache = None  # type: ignore[attr-defined]
    ema_model._boomer_ema_buffer_cache = None  # type: ignore[attr-defined]
    return ema_model


def _build_ema_caches(ema_model: torch.nn.Module, source_model: torch.nn.Module) -> None:
    src_params_by_name = dict(source_model.named_parameters())
    ema_float_params: list[torch.Tensor] = []
    src_float_params: list[torch.Tensor] = []
    for name, ema_p in ema_model.named_parameters():
        src_p = src_params_by_name.get(name)
        if src_p is None or not torch.is_floating_point(ema_p):
            continue
        ema_float_params.append(ema_p)
        src_float_params.append(src_p)

    src_bufs_by_name = dict(source_model.named_buffers())
    ema_bufs: list[torch.Tensor] = []
    src_bufs: list[torch.Tensor] = []
    for name, ema_b in ema_model.named_buffers():
        src_b = src_bufs_by_name.get(name)
        if src_b is None:
            continue
        ema_bufs.append(ema_b)
        src_bufs.append(src_b)

    ema_model._boomer_ema_param_cache = (ema_float_params, src_float_params)  # type: ignore[attr-defined]
    ema_model._boomer_ema_buffer_cache = (ema_bufs, src_bufs)  # type: ignore[attr-defined]


@torch.no_grad()
def update_ema_model(ema_model: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    if getattr(ema_model, "_boomer_ema_param_cache", None) is None:
        _build_ema_caches(ema_model, model)
    ema_params, src_params = ema_model._boomer_ema_param_cache  # type: ignore[attr-defined]
    if ema_params:
        # When dtypes match (fp32 master + fp32 EMA, the common path) .to() is a no-op
        # and returns the same tensor, so no extra allocations are made.
        src_cast = [
            p.detach().to(dtype=e.dtype, device=e.device, non_blocking=True)
            for p, e in zip(src_params, ema_params)
        ]
        torch._foreach_mul_(ema_params, decay)
        torch._foreach_add_(ema_params, src_cast, alpha=1.0 - decay)
    ema_bufs, src_bufs = ema_model._boomer_ema_buffer_cache  # type: ignore[attr-defined]
    for ema_b, src_b in zip(ema_bufs, src_bufs):
        ema_b.copy_(src_b.to(dtype=ema_b.dtype, device=ema_b.device, non_blocking=True))


def build_train_args(
    args: argparse.Namespace,
    config: Config,
    *,
    latent_mean: LatentStat,
    latent_std: LatentStat,
) -> dict[str, Any]:
    return collect_train_args(
        ema_decay=args.ema_decay,
        ema_update_every=args.ema_update_every,
        cond_dropout=args.cond_dropout,
        latent_mean=latent_mean,
        latent_std=latent_std,
        t_sampler=args.t_sampler,
        t_bias=args.t_bias,
        flow_shift=args.flow_shift,
        logit_mean=args.logit_mean,
        logit_std=args.logit_std,
        plateau_low=args.plateau_low,
        plateau_high=args.plateau_high,
        y_norm=config.y_norm,
        y_norm_scale_factor=config.y_norm_scale_factor,
        grad_clip=args.grad_clip,
        timestep_weight=args.timestep_weight,
        min_snr_gamma=args.min_snr_gamma,
        lr_scheduler=args.lr_scheduler,
        warmup_steps=args.warmup_steps,
        min_lr_ratio=args.min_lr_ratio,
    )


def create_optimizer(model: torch.nn.Module, args: argparse.Namespace, device: str) -> torch.optim.AdamW:
    adam_kwargs: dict[str, Any] = {"lr": args.lr, "weight_decay": 0.0}
    if args.fused_adamw:
        if not str(device).startswith("cuda"):
            print("warning: --fused-adamw is ignored when not training on CUDA")
        else:
            try:
                return torch.optim.AdamW(model.parameters(), fused=True, **adam_kwargs)
            except TypeError:
                print("warning: this PyTorch build does not support fused AdamW; using the standard kernel")
    return torch.optim.AdamW(model.parameters(), **adam_kwargs)


@torch.inference_mode()
def save_denoise_preview(
    *,
    model: torch.nn.Module,
    vae: Any,
    record_path: str | Path,
    out_dir: str | Path,
    cache_config: CacheConfig,
    step: int,
    preview_t: float,
    preview_steps: int,
    preview_sampler: str,
    stork_derivative_order: int,
    stork_substeps: int,
    seed: int,
    device: str,
    dtype: torch.dtype,
    latent_mean: LatentStat,
    latent_std: LatentStat,
    flow_shift: float,
    sample_index: int = 0,
) -> tuple[float, float]:
    record = load_latent_record(record_path)
    raw_x0 = record.image_latent.unsqueeze(0).to(device=device, dtype=dtype)
    x0 = normalize_latent(raw_x0, latent_mean, latent_std)
    text_embedding = record.text_embedding.unsqueeze(0).to(device=device, dtype=dtype)
    attention_mask = record.attention_mask.unsqueeze(0).to(device=device)

    generator = torch.Generator(device=device).manual_seed(seed + sample_index)
    x_t, _ = noise_latent_at_sigma(x0, preview_t, generator=generator)
    denoised = denoise_latent(
        model,
        x_t,
        preview_t,
        text_embedding,
        attention_mask,
        preview_steps,
        flow_shift,
        sampler=preview_sampler,
        stork_derivative_order=stork_derivative_order,
        stork_substeps=stork_substeps,
    )

    out_dir = Path(out_dir)
    step_dir = out_dir / f"step_{step:06d}"
    sample_dir = step_dir / f"sample_{sample_index:02d}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    tensor_to_image(decode_image_latent(vae, raw_x0, cache_config)).save(sample_dir / "original.png")
    tensor_to_image(decode_image_latent(vae, denormalize_latent(x_t, latent_mean, latent_std), cache_config)).save(
        sample_dir / f"noised_t{preview_t:.2f}.png"
    )
    tensor_to_image(decode_image_latent(vae, denormalize_latent(denoised, latent_mean, latent_std), cache_config)).save(
        sample_dir / f"denoised_t{preview_t:.2f}_{preview_sampler}_steps{preview_steps}.png"
    )

    mse_noised = torch.mean((x_t.float() - x0.float()) ** 2).item()
    mse_denoised = torch.mean((denoised.float() - x0.float()) ** 2).item()
    return mse_noised, mse_denoised


@torch.inference_mode()
def save_denoise_previews(
    *,
    model: torch.nn.Module,
    vae: Any,
    record_paths: list[str | Path],
    out_dir: str | Path,
    cache_config: CacheConfig,
    step: int,
    preview_t: float,
    preview_steps: int,
    preview_sampler: str,
    stork_derivative_order: int,
    stork_substeps: int,
    seed: int,
    device: str,
    dtype: torch.dtype,
    latent_mean: LatentStat,
    latent_std: LatentStat,
    flow_shift: float,
) -> tuple[float, float]:
    if not record_paths:
        raise ValueError("record_paths must contain at least one preview cache record.")
    mse_noised_total = 0.0
    mse_denoised_total = 0.0
    for sample_index, record_path in enumerate(record_paths):
        mse_noised, mse_denoised = save_denoise_preview(
            model=model,
            vae=vae,
            record_path=record_path,
            out_dir=out_dir,
            cache_config=cache_config,
            step=step,
            preview_t=preview_t,
            preview_steps=preview_steps,
            preview_sampler=preview_sampler,
            stork_derivative_order=stork_derivative_order,
            stork_substeps=stork_substeps,
            seed=seed,
            device=device,
            dtype=dtype,
            latent_mean=latent_mean,
            latent_std=latent_std,
            flow_shift=flow_shift,
            sample_index=sample_index,
        )
        mse_noised_total += mse_noised
        mse_denoised_total += mse_denoised
    count = len(record_paths)
    return mse_noised_total / count, mse_denoised_total / count


def normalize_preview_records(raw: list[str] | None) -> list[str]:
    if not raw:
        return []
    records: list[str] = []
    for item in raw:
        records.extend(part.strip() for part in item.split(",") if part.strip())
    return records


def open_loss_csv(
    path: str | Path,
    *,
    resume: bool,
) -> tuple[Any, csv.DictWriter, bool]:
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    append = resume and csv_path.exists() and csv_path.stat().st_size > 0
    csv_file = csv_path.open("a" if append else "w", newline="", encoding="utf-8")
    csv_writer = csv.DictWriter(csv_file, fieldnames=["step", "loss", "loss_ma", "lr"])
    if not append:
        csv_writer.writeheader()
    return csv_file, csv_writer, append


def init_wandb_run(
    args: argparse.Namespace,
    *,
    config: Config,
    start_step: int,
    resume: bool,
) -> Any:
    if not args.wandb:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise SystemExit("wandb is required when --wandb is set. Install with: pip install wandb") from exc

    init_kwargs: dict[str, Any] = {
        "project": args.wandb_project,
        "config": {
            **{key: value for key, value in vars(args).items() if not key.startswith("_")},
            "model_type": config.model_type,
        },
        "resume": "allow" if resume else None,
        "settings": wandb.Settings(save_code=False),
    }
    if args.wandb_run_name:
        init_kwargs["name"] = args.wandb_run_name
    if args.wandb_entity:
        init_kwargs["entity"] = args.wandb_entity
    if args.wandb_mode in {"online", "offline", "disabled"}:
        init_kwargs["mode"] = args.wandb_mode

    run = wandb.init(**init_kwargs)
    if start_step > 0:
        wandb.log({"resume/start_step": start_step}, step=start_step)
    return run


def log_wandb_metrics(
    *,
    step: int,
    loss: float,
    loss_ma: float,
    lr: float,
    step_s: float | None = None,
    elapsed_s: float | None = None,
    eta_s: float | None = None,
    grad_norm: float | None = None,
    raw_loss: float | None = None,
    sigma_low_loss: float | None = None,
    sigma_mid_loss: float | None = None,
    sigma_high_loss: float | None = None,
    sigma_mean: float | None = None,
) -> None:
    import wandb

    metrics: dict[str, float] = {
        "train/loss": loss,
        "train/loss_ma": loss_ma,
        "train/lr": lr,
    }
    if step_s is not None:
        metrics["train/step_s"] = step_s
    if elapsed_s is not None:
        metrics["train/elapsed_s"] = elapsed_s
    if eta_s is not None:
        metrics["train/eta_s"] = eta_s
    if grad_norm is not None:
        metrics["train/grad_norm"] = grad_norm
    if raw_loss is not None:
        metrics["train/raw_mse"] = raw_loss
    if sigma_low_loss is not None:
        metrics["train/raw_mse_sigma_low"] = sigma_low_loss
    if sigma_mid_loss is not None:
        metrics["train/raw_mse_sigma_mid"] = sigma_mid_loss
    if sigma_high_loss is not None:
        metrics["train/raw_mse_sigma_high"] = sigma_high_loss
    if sigma_mean is not None:
        metrics["train/sigma_mean"] = sigma_mean
    wandb.log(metrics, step=step)


def log_wandb_previews(
    *,
    out_dir: str | Path,
    step: int,
    preview_t: float,
    preview_sampler: str,
    preview_steps: int,
    num_samples: int,
    mse_noised: float,
    mse_denoised: float,
) -> None:
    import wandb

    preview_dir = Path(out_dir) / f"step_{step:06d}"
    denoised_images: list[Any] = []
    for sample_index in range(num_samples):
        sample_dir = preview_dir / f"sample_{sample_index:02d}"
        denoised = sample_dir / f"denoised_t{preview_t:.2f}_{preview_sampler}_steps{preview_steps}.png"
        if denoised.is_file():
            denoised_images.append(
                wandb.Image(str(denoised), caption=f"sample_{sample_index:02d}")
            )

    payload: dict[str, Any] = {
        "preview/latent_mse_noised": mse_noised,
        "preview/latent_mse_denoised": mse_denoised,
    }
    if denoised_images:
        payload["previews"] = denoised_images
    wandb.log(payload, step=step)


def finish_wandb_run(run: Any) -> None:
    if run is None:
        return
    import wandb

    wandb.finish()


def release_preview_vae(vae: Any, device: str) -> None:
    """Drop VAE weights after a preview so training can reclaim GPU memory."""
    del vae
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Boomer latent flow denoiser on cached latents.")
    parser.add_argument("--cache", help="Cache directory or .pt file.")
    parser.add_argument("--cache-wds", help="WebDataset latent cache shards, local glob or hf://datasets/.../train-*.tar.")
    parser.add_argument(
        "--wds-local-dir",
        help="Local directory for a rolling window of HF WDS shards. Use with --wds-rolling-shards.",
    )
    parser.add_argument(
        "--wds-rolling-shards",
        type=int,
        default=0,
        help="Keep this many tar shards on disk and roll through --cache-wds as training reads them.",
    )
    parser.add_argument("--wds-shuffle-buffer", type=int, default=0, help="Streaming WDS sample shuffle buffer. Rolling mode requires >= 1000.")
    parser.add_argument("--wds-shard-shuffle", type=int, default=0, help="Streaming WDS shard shuffle buffer/count. 0 disables.")
    parser.add_argument("--model-type", choices=["latent_dit", "mamba_anchor", "boomer_fla"], default="latent_dit")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker processes for parallel cache reads. Try 4 on Colab for large batches.",
    )
    parser.add_argument(
        "--pin-memory",
        action="store_true",
        help="Pin CPU batches for faster async GPU transfer (CUDA only).",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
        help="Batches prefetched per worker when --num-workers > 0.",
    )
    parser.add_argument(
        "--persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Keep DataLoader workers alive between epochs. Default: on when --num-workers > 0.",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=384)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=6)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--cross-attn", action="store_true", help="Use cross-attention over full Gemma token states.")
    parser.add_argument("--num-anchors", type=int, default=64, help="MambaAnchor-DiT learned anchor tokens.")
    parser.add_argument("--text-attn-every", type=int, default=2, help="MambaAnchor-DiT text attention frequency.")
    parser.add_argument("--anchor-attn-every", type=int, default=3, help="MambaAnchor-DiT anchor attention frequency.")
    parser.add_argument(
        "--mixer-type",
        choices=[
            "mamba3",
            "fla_gated_deltanet",
            "fla_linear",
            "fla_gla",
            "fla_delta_net",
            "fla_hgrn2",
            "fla_based",
            "mlp",
            "fallback",
        ],
        default="mamba3",
    )
    parser.add_argument("--mamba-d-state", type=int, default=128)
    parser.add_argument("--mamba-headdim", type=int, default=64)
    parser.add_argument("--mamba-chunk-size", type=int, default=64)
    parser.add_argument("--mamba-mimo", action="store_true", help="Use Mamba3 MIMO mode. Start without this.")
    parser.add_argument("--fla-mode", choices=["chunk", "fused_chunk"], default="chunk")
    parser.add_argument("--fla-feature-map", default="relu")
    parser.add_argument(
        "--fla-bidirectional",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run FLA image mixers forward and backward over the token sequence, sharing weights.",
    )
    parser.add_argument("--use-short-conv", action="store_true")
    parser.add_argument("--conv-size", type=int, default=4)
    parser.add_argument(
        "--image-attn-every",
        type=int,
        default=0,
        help="BoomerFLA full image self-attention frequency. Use 4 for blocks 4, 8, 12...",
    )
    parser.add_argument(
        "--image-attn-backend",
        choices=["sdpa", "flash3", "flash4", "auto"],
        default="sdpa",
        help="Backend for BoomerFLA periodic full image self-attention.",
    )
    parser.add_argument(
        "--image-rope",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply 2D RoPE to BoomerFLA full image self-attention anchor blocks.",
    )
    parser.add_argument(
        "--image-rope-theta",
        type=float,
        default=10000.0,
        help="RoPE frequency base for --image-rope. Default matches Wan/HunyuanVideo.",
    )
    parser.add_argument(
        "--cross-attn-backend",
        choices=["sdpa", "xformers", "auto", "mha"],
        default="sdpa",
        help="BoomerFLA text cross-attention backend. Use sdpa by default; xformers is optional.",
    )
    parser.add_argument(
        "--cross-attn-qk-norm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use Sana-style q/k RMSNorm inside BoomerFLA text cross-attention.",
    )
    parser.add_argument(
        "--parallel-block",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use FLUX/SD3-style parallel residual branches inside BoomerFLA blocks.",
    )
    parser.add_argument(
        "--dual-stream-depth",
        type=int,
        default=0,
        help="Number of early BoomerFLA blocks that update both image and text streams.",
    )
    parser.add_argument(
        "--multimodal-coord-ids",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Add explicit image/text token-type and coordinate embeddings.",
    )
    parser.add_argument(
        "--abs-pos-embed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use learned absolute latent-token positional embeddings. Disable with --no-abs-pos-embed for RoPE-only variable-size runs.",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=1,
        help="Latent patch size for BoomerFLA ViT patchification. Use 2 for 64x64 f16 latents to reduce sequence length to 32x32.",
    )
    parser.add_argument(
        "--cond-dropout",
        type=float,
        default=0.0,
        help="Per-sample probability of replacing text conditioning with learned null embed for CFG training.",
    )
    parser.add_argument(
        "--ema-decay",
        type=float,
        default=0.0,
        help=(
            "EMA decay for a float32 shadow model. Set to 0 to disable. "
            "Large runs often use 0.96 with --ema-update-every 8."
        ),
    )
    parser.add_argument(
        "--ema-update-every",
        type=int,
        default=8,
        help=(
            "Update EMA once every N optimizer steps. Default 8. "
            "Pair with --ema-decay 0.96 for smoothing similar to 0.995 every 4 steps."
        ),
    )
    parser.add_argument(
        "--fused-adamw",
        action="store_true",
        help="Use the fused CUDA AdamW kernel when supported by this PyTorch build.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Wrap the training model in torch.compile. EMA is left uncompiled.",
    )
    parser.add_argument(
        "--compile-mode",
        choices=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"],
        default="max-autotune-no-cudagraphs",
        help="torch.compile mode. max-autotune-no-cudagraphs is the safe Blackwell default.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--resume", help="Optional checkpoint to resume from. --steps is the total target step.")
    parser.add_argument(
        "--resume-weights-only",
        action="store_true",
        help=(
            "With --resume, load model/EMA/config weights but reset optimizer, scheduler, "
            "and start_step to 0. Use this for dataset continuation/fine-tuning."
        ),
    )
    parser.add_argument("--save", help="Optional checkpoint path.")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--loss-window", type=int, default=50)
    parser.add_argument("--log-csv", help="Optional CSV loss log path.")
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Log training metrics (and preview images) to Weights & Biases.",
    )
    parser.add_argument("--wandb-project", default="boomer", help="W&B project name.")
    parser.add_argument("--wandb-run-name", default=None, help="Optional W&B run name.")
    parser.add_argument("--wandb-entity", default=None, help="Optional W&B entity/team.")
    parser.add_argument(
        "--wandb-mode",
        choices=["online", "offline", "disabled"],
        default="online",
        help="W&B mode. Use offline on air-gapped machines and sync later.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=0,
        help="Save a checkpoint every N steps. Large runs often use 4000.",
    )
    parser.add_argument("--checkpoint-dir", help="Directory for periodic checkpoints.")
    parser.add_argument(
        "--preview-every",
        type=int,
        default=0,
        help="Run a VAE decode preview every N steps. Large runs often use 4000.",
    )
    parser.add_argument(
        "--preview-cache-record",
        action="append",
        default=None,
        help="Cached .pt record(s) for periodic denoise previews. Pass multiple times for multi-image previews.",
    )
    parser.add_argument("--preview-out-dir", help="Directory for periodic denoise previews.")
    parser.add_argument("--preview-t", type=float, default=0.3)
    parser.add_argument(
        "--preview-steps",
        type=int,
        default=16,
        help="Reverse denoise steps for periodic previews. Large runs often use 16.",
    )
    parser.add_argument("--preview-seed", type=int, default=0)
    parser.add_argument(
        "--preview-sampler",
        choices=["stork2", "stork4", "euler", "heun"],
        default="stork2",
        help="Reverse denoise solver for periodic previews. stork2 matches sample_latent_dit.py.",
    )
    parser.add_argument("--preview-stork-derivative-order", type=int, default=1)
    parser.add_argument(
        "--preview-stork-substeps",
        type=int,
        default=5,
        help=(
            "STORK intra-step RK micro-steps (maps to STORKScheduler s). "
            "This is NOT extra model forwards — only internal integration per preview step. "
            "STORK defaults to 50 for max quality; 5 matches STORK PNDM quick demos and is enough for training previews."
        ),
    )
    parser.add_argument("--vae-id", default=CacheConfig.vae_id)
    parser.add_argument("--text-encoder-id", default=CacheConfig.text_encoder_id)
    parser.add_argument("--max-length", type=int, default=CacheConfig.max_length)
    parser.add_argument("--scaling-factor", type=float, default=CacheConfig.scaling_factor)
    parser.add_argument("--latent-mean", type=float, help="Optional scalar mean used to normalize cached image latents.")
    parser.add_argument("--latent-std", type=float, help="Optional scalar std used to normalize cached image latents.")
    parser.add_argument("--latent-stats-json", help="Stats JSON from tools/inspect_latent_stats.py.")
    parser.add_argument(
        "--latent-stats-mode",
        choices=["channel", "global"],
        default="channel",
        help="Use channel-wise or global stats when --latent-stats-json is provided.",
    )
    parser.add_argument(
        "--t-sampler",
        choices=["uniform", "high", "low", "logit-normal", "bounded-logit-normal", "plateau-logit-normal"],
        default="uniform",
        help="Timestep sampling distribution. logit-normal matches Sana's training choice.",
    )
    parser.add_argument("--t-bias", type=float, default=2.0, help="Bias strength for --t-sampler high/low.")
    parser.add_argument("--flow-shift", type=float, default=1.0, help="Sana/FlowMatch sigma shift. Sana uses 3.0.")
    parser.add_argument("--logit-mean", type=float, default=0.0, help="Mean for --t-sampler logit-normal.")
    parser.add_argument("--logit-std", type=float, default=1.0, help="Std for --t-sampler logit-normal.")
    parser.add_argument(
        "--plateau-low",
        type=float,
        default=0.1,
        help="Lower unshifted sigma bound for --t-sampler bounded-logit-normal.",
    )
    parser.add_argument(
        "--plateau-high",
        type=float,
        default=0.9,
        help="Upper unshifted sigma bound for --t-sampler bounded-logit-normal.",
    )
    parser.add_argument(
        "--y-norm",
        dest="y_norm",
        action="store_true",
        default=None,
        help="Use learnable Sana-style attention_y_norm inside LatentDiT.",
    )
    parser.add_argument("--no-y-norm", dest="y_norm", action="store_false")
    parser.add_argument("--y-norm-scale-factor", type=float, default=0.01, help="Sana uses 0.01.")
    parser.add_argument(
        "--timestep-weight",
        action="store_true",
        help="Apply Sana per-timestep loss weight sqrt(1+sigma^2)/sigma.",
    )
    parser.add_argument(
        "--min-snr-gamma",
        type=float,
        default=0.0,
        help=(
            "Apply Min-SNR loss weighting with this gamma. 0 disables. "
            "For flow x_t=(1-sigma)x0+sigma*noise, SNR=((1-sigma)/sigma)^2; "
            "the weight is min(SNR,gamma)/SNR, downweighting low-noise/high-SNR samples."
        ),
    )
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping norm. Sana configs use 0.1.")
    parser.add_argument(
        "--lr-scheduler",
        choices=["constant", "linear-warmup-cosine"],
        default="constant",
        help="Learning-rate schedule. linear-warmup-cosine computes LR from global step, so resume is deterministic.",
    )
    parser.add_argument("--warmup-steps", type=int, default=0, help="Linear warmup steps for --lr-scheduler.")
    parser.add_argument(
        "--min-lr-ratio",
        type=float,
        default=0.1,
        help="Final LR as a fraction of --lr for cosine decay.",
    )
    parser.add_argument("--hf-token", help="Hugging Face token, or 'true' to use the logged-in token.")
    args = parser.parse_args()
    if bool(args.cache) == bool(args.cache_wds):
        parser.error("provide exactly one of --cache or --cache-wds.")
    if not 0.0 <= args.cond_dropout <= 1.0:
        parser.error("--cond-dropout must be between 0 and 1.")
    if not 0.0 <= args.ema_decay < 1.0:
        parser.error("--ema-decay must be in [0, 1).")
    if args.ema_update_every < 1:
        parser.error("--ema-update-every must be at least 1.")

    device = args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    if device == "cpu":
        dtype = torch.float32

    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    set_seed(args.seed)

    if args.resume_weights_only and not args.resume:
        parser.error("--resume-weights-only requires --resume.")
    if args.resume and Path(args.resume).suffix == ".safetensors" and not args.resume_weights_only:
        parser.error("--resume requires a full .pt checkpoint with optimizer state, not a .safetensors file.")
    resume_map_location = "cpu" if args.resume_weights_only else device
    resume_checkpoint = load_checkpoint(args.resume, map_location=resume_map_location) if args.resume else None
    if resume_checkpoint is not None and args.resume_weights_only and "optimizer" in resume_checkpoint:
        # Dataset continuation only needs model/EMA/config/train_args. Keeping
        # the old Adam moments around can waste many GB, especially when the
        # checkpoint was saved from a large fp32-master run.
        del resume_checkpoint["optimizer"]
    start_step = 0 if args.resume_weights_only else int(resume_checkpoint.get("steps", 0)) if resume_checkpoint else 0
    resume_train_args = (resume_checkpoint.get("train_args") or {}) if resume_checkpoint is not None else {}
    if resume_checkpoint is not None and not args.resume_weights_only:
        apply_resume_train_args(args, parser, resume_train_args)
    elif args.y_norm is None:
        args.y_norm = False

    latent_mean, latent_std = load_latent_normalizer(args, resume_train_args)
    if resume_checkpoint is not None and args.resume_weights_only and resume_train_args:
        old_mean = resume_train_args.get("latent_mean")
        old_std = resume_train_args.get("latent_std")
        if old_mean != latent_mean or old_std != latent_std:
            print(
                "warning: weights-only continuation is using a new latent normalizer "
                f"old_mean=({latent_stat_summary(old_mean)}) old_std=({latent_stat_summary(old_std)}) "
                f"new_mean=({latent_stat_summary(latent_mean)}) new_std=({latent_stat_summary(latent_std)})"
            )
    if args.t_bias <= 0.0:
        parser.error("--t-bias must be positive.")
    if args.flow_shift <= 0.0:
        parser.error("--flow-shift must be positive.")
    if args.logit_std <= 0.0:
        parser.error("--logit-std must be positive.")
    if not 0.0 <= args.plateau_low < args.plateau_high <= 1.0:
        parser.error("--plateau-low and --plateau-high must satisfy 0 <= low < high <= 1.")
    if args.min_snr_gamma < 0.0:
        parser.error("--min-snr-gamma must be non-negative.")
    if args.y_norm_scale_factor <= 0.0:
        parser.error("--y-norm-scale-factor must be positive.")
    if args.grad_clip <= 0.0:
        parser.error("--grad-clip must be positive.")
    if args.mlp_ratio <= 0.0:
        parser.error("--mlp-ratio must be positive.")
    if args.num_anchors <= 0:
        parser.error("--num-anchors must be positive.")
    if args.text_attn_every < 0 or args.anchor_attn_every < 0:
        parser.error("--text-attn-every and --anchor-attn-every must be non-negative.")
    if args.image_attn_every < 0:
        parser.error("--image-attn-every must be non-negative.")
    if args.dual_stream_depth < 0:
        parser.error("--dual-stream-depth must be non-negative.")
    if args.dual_stream_depth > args.depth:
        parser.error("--dual-stream-depth must be <= --depth.")
    if args.patch_size <= 0:
        parser.error("--patch-size must be positive.")
    if args.image_rope_theta <= 0.0:
        parser.error("--image-rope-theta must be positive.")
    if args.mamba_d_state <= 0 or args.mamba_headdim <= 0 or args.mamba_chunk_size <= 0:
        parser.error("Mamba dimensions must be positive.")
    if args.warmup_steps < 0:
        parser.error("--warmup-steps must be non-negative.")
    if not 0.0 <= args.min_lr_ratio <= 1.0:
        parser.error("--min-lr-ratio must be in [0, 1].")
    if args.num_workers < 0:
        parser.error("--num-workers must be non-negative.")
    if args.prefetch_factor < 1:
        parser.error("--prefetch-factor must be at least 1.")
    if args.wds_shuffle_buffer < 0 or args.wds_shard_shuffle < 0:
        parser.error("--wds-shard-shuffle must be non-negative.")
    if args.wds_rolling_shards < 0:
        parser.error("--wds-rolling-shards must be non-negative.")
    if args.wds_rolling_shards and not args.wds_local_dir:
        parser.error("--wds-rolling-shards requires --wds-local-dir.")
    if args.wds_local_dir and not args.wds_rolling_shards:
        parser.error("--wds-local-dir requires --wds-rolling-shards.")
    if args.wds_rolling_shards and not str(args.cache_wds or "").startswith("hf://"):
        parser.error("rolling local WDS requires an hf://datasets/... --cache-wds glob.")
    if args.wds_rolling_shards and args.num_workers > 0:
        parser.error("rolling local WDS requires --num-workers 0.")
    if args.wds_rolling_shards:
        if args.wds_shuffle_buffer < 1000:
            parser.error("rolling local WDS requires --wds-shuffle-buffer >= 1000.")
    if args.preview_steps < 1:
        parser.error("--preview-steps must be at least 1.")
    preview_records = normalize_preview_records(args.preview_cache_record)
    if args.preview_every and not preview_records and not args.cache_wds:
        pass  # filled from local cache below
    if args.preview_every and args.cache_wds and not preview_records:
        parser.error("--preview-cache-record is required when --preview-every is set with --cache-wds.")
    if args.preview_stork_derivative_order not in {1, 2, 3}:
        parser.error("--preview-stork-derivative-order must be 1, 2, or 3.")
    if args.preview_stork_substeps < 2:
        parser.error("--preview-stork-substeps must be at least 2.")

    if args.cache_wds:
        if args.wds_rolling_shards:
            dataset = RollingStreamingWdsLatentCacheDataset(
                args.cache_wds,
                local_dir=args.wds_local_dir,
                rolling_shards=args.wds_rolling_shards,
                hf_token=hf_token_value(args.hf_token),
                shuffle_buffer=args.wds_shuffle_buffer,
                seed=args.seed,
            )
        else:
            dataset = StreamingWdsLatentCacheDataset(
                args.cache_wds,
                hf_token=hf_token_value(args.hf_token),
                shardshuffle=args.wds_shard_shuffle or False,
                shuffle_buffer=args.wds_shuffle_buffer,
            )
        probe_record = next(iter(dataset))
        records_label = "streaming"
    else:
        dataset = LatentCacheDataset(args.cache)
        probe_record = dataset[0]
        records_label = str(len(dataset))
    drop_last = False if args.cache_wds else len(dataset) >= args.batch_size
    pin_memory = args.pin_memory and str(device).startswith("cuda")
    if args.pin_memory and not pin_memory:
        print("warning: --pin-memory is ignored when not training on CUDA")
    persistent_workers = args.persistent_workers
    if persistent_workers is None:
        persistent_workers = args.num_workers > 0
    if persistent_workers and args.num_workers == 0:
        print("warning: --persistent-workers is ignored when --num-workers=0")
        persistent_workers = False
    loader_kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": args.batch_size,
        "drop_last": drop_last,
        "collate_fn": collate_latent_records,
        "num_workers": args.num_workers,
        "pin_memory": pin_memory,
    }
    if args.cache_wds:
        loader_kwargs["shuffle"] = False
    else:
        loader_kwargs["shuffle"] = True
        loader_kwargs["generator"] = torch.Generator().manual_seed(args.seed)
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
        loader_kwargs["persistent_workers"] = persistent_workers
    loader = DataLoader(**loader_kwargs)
    iterator = iter(loader)

    config = build_model_config(args, probe_record, resume_checkpoint)
    # fp32 master weights; bf16/fp16 compute happens via torch.autocast in the loop.
    # Keeps AdamW second-moments numerically sound for long runs and lets EMA stay fp32
    # without an extra cast in the inner loop.
    model = create_model(config).to(device=device)
    # Keep EMA in fp32 when kernels allow it. BF16 EMA with high decay (0.999)
    # can quantize away tiny updates and make early previews look much worse
    # than the live model. Mamba3 is the exception: its Colab kernel path has
    # been validated in BF16 first, so keep that EMA in training dtype.
    ema_dtype = dtype if config.model_type == "mamba_anchor" else torch.float32
    ema_model = create_ema_model(model, dtype=ema_dtype) if args.ema_decay > 0.0 else None
    optimizer = create_optimizer(model, args, device)
    if resume_checkpoint is not None:
        skipped = load_compatible_state_dict(model, resume_checkpoint["model"])
        if skipped:
            print(f"warning: skipped {len(skipped)} incompatible model tensors from resume checkpoint")
        if "optimizer" in resume_checkpoint and not args.resume_weights_only:
            try:
                optimizer.load_state_dict(resume_checkpoint["optimizer"])
            except ValueError as exc:
                print(f"warning: skipped incompatible optimizer state from resume checkpoint: {exc}")
        elif args.resume_weights_only:
            print("resume-weights-only: optimizer and LR schedule reset to step 0")
        if ema_model is not None and "ema_model" in resume_checkpoint:
            skipped = load_compatible_state_dict(ema_model, resume_checkpoint["ema_model"])
            if skipped:
                print(f"warning: skipped {len(skipped)} incompatible EMA tensors from resume checkpoint")

    # Keep an unwrapped reference for state_dict, EMA updates, and any
    # method calls (apply_condition_dropout) that we don't want to push
    # through the compiled graph. EMA is intentionally never compiled.
    raw_model = model
    compile_active = bool(args.compile) and str(device).startswith("cuda")
    if args.compile and not compile_active:
        print("warning: --compile is ignored when not training on CUDA")
    if compile_active:
        model = torch.compile(
            model,
            mode=args.compile_mode,
            fullgraph=False,
            dynamic=False,
        )
        print(f"torch.compile enabled (mode={args.compile_mode})")

    csv_file = None
    csv_writer = None
    if args.log_csv:
        csv_file, csv_writer, _ = open_loss_csv(
            args.log_csv,
            resume=resume_checkpoint is not None and not args.resume_weights_only,
        )

    if not preview_records and args.preview_every and not args.cache_wds:
        preview_records = [str(dataset.files[0])]
    preview_cache_config = None
    if args.preview_every:
        if not args.preview_out_dir:
            raise ValueError("--preview-out-dir is required when --preview-every is set.")
        preview_cache_config = CacheConfig(
            vae_id=args.vae_id,
            text_encoder_id=args.text_encoder_id,
            max_length=args.max_length,
            dtype=args.dtype,
            device=device,
            scaling_factor=args.scaling_factor,
        )

    print(
        f"records={records_label} model_type={config.model_type} params={sum(p.numel() for p in model.parameters()):,} "
        f"device={device} dtype={dtype} batch_size={args.batch_size} num_workers={args.num_workers} "
        f"pin_memory={pin_memory} prefetch_factor={loader_kwargs.get('prefetch_factor', 'n/a')} "
        f"persistent_workers={loader_kwargs.get('persistent_workers', False)} "
        f"wds_shuffle_buffer={args.wds_shuffle_buffer if args.cache_wds else 'n/a'} "
        f"wds_shard_shuffle={args.wds_shard_shuffle if args.cache_wds else 'n/a'} "
        f"wds_rolling_shards={args.wds_rolling_shards if args.cache_wds else 'n/a'} "
        f"wds_local_dir={args.wds_local_dir if args.cache_wds else 'n/a'} "
        f"seed={args.seed} cond_dropout={args.cond_dropout} fused_adamw={args.fused_adamw} "
        f"parallel_block={getattr(config, 'parallel_block', False)} "
        f"dual_stream_depth={getattr(config, 'dual_stream_depth', 0)} "
        f"multimodal_coord_ids={getattr(config, 'multimodal_coord_ids', False)} "
        f"abs_pos_embed={getattr(config, 'use_abs_pos_embed', True)} "
        f"patch_size={getattr(config, 'patch_size', 1)} "
        f"ema_decay={args.ema_decay} ema_update_every={args.ema_update_every} "
        f"t_sampler={args.t_sampler} t_bias={args.t_bias} flow_shift={args.flow_shift} "
        f"logit_mean={args.logit_mean} logit_std={args.logit_std} "
        f"plateau_low={args.plateau_low} plateau_high={args.plateau_high} "
        f"y_norm={config.y_norm} y_norm_scale_factor={config.y_norm_scale_factor} "
        f"timestep_weight={args.timestep_weight} min_snr_gamma={args.min_snr_gamma} grad_clip={args.grad_clip} "
        f"lr_scheduler={args.lr_scheduler} warmup_steps={args.warmup_steps} min_lr_ratio={args.min_lr_ratio} "
        f"latent_mean=({latent_stat_summary(latent_mean)}) latent_std=({latent_stat_summary(latent_std)})"
    )
    wandb_run: Any | None = None
    if resume_checkpoint is not None:
        mode = "weights-only" if args.resume_weights_only else "full"
        print(f"resumed checkpoint ({mode}): {args.resume} start_step={start_step} target_step={args.steps}")
    wandb_run = init_wandb_run(
        args,
        config=config,
        start_step=start_step,
        resume=resume_checkpoint is not None and not args.resume_weights_only,
    )
    losses: deque[float] = deque(maxlen=args.loss_window)
    # Per-step values are accumulated as GPU tensors and only synced to host on
    # log_every boundaries. This removes ~1 cudaStreamSynchronize per step.
    pending_steps: list[int] = []
    pending_losses: list[torch.Tensor] = []
    pending_raw_losses: list[torch.Tensor] = []
    pending_sigma_means: list[torch.Tensor] = []
    pending_sigma_low_losses: list[torch.Tensor] = []
    pending_sigma_mid_losses: list[torch.Tensor] = []
    pending_sigma_high_losses: list[torch.Tensor] = []
    pending_grad_norms: list[torch.Tensor] = []
    pending_lrs: list[float] = []
    last_step = start_step
    train_started = time.perf_counter()

    autocast_enabled = str(device).startswith("cuda") and dtype != torch.float32
    autocast_dtype = dtype if dtype != torch.float32 else torch.bfloat16

    # Constant schedule: set lr once and skip the per-step write below.
    if args.lr_scheduler == "constant":
        set_optimizer_lr(optimizer, args.lr)

    try:
        for step in range(start_step + 1, args.steps + 1):
            step_started = time.perf_counter()
            last_step = step
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)

            model.train()
            if args.lr_scheduler != "constant":
                lr_mult = lr_multiplier(
                    step,
                    args.steps,
                    scheduler=args.lr_scheduler,
                    warmup_steps=args.warmup_steps,
                    min_lr_ratio=args.min_lr_ratio,
                )
                set_optimizer_lr(optimizer, args.lr * lr_mult)
            non_blocking = pin_memory
            x0 = move_batch_tensor(
                batch["image_latent"],
                device=device,
                dtype=dtype,
                non_blocking=non_blocking,
            )
            x0 = normalize_latent(x0, latent_mean, latent_std)
            text_embedding = move_batch_tensor(
                batch["text_embedding"],
                device=device,
                dtype=dtype,
                non_blocking=non_blocking,
            )
            attention_mask = move_batch_tensor(
                batch["attention_mask"],
                device=device,
                non_blocking=non_blocking,
            )
            text_embedding, attention_mask = raw_model.apply_condition_dropout(
                text_embedding, attention_mask, args.cond_dropout
            )
            x_t, t, target_velocity = sample_flow_batch(
                x0,
                t_sampler=args.t_sampler,
                t_bias=args.t_bias,
                flow_shift=args.flow_shift,
                logit_mean=args.logit_mean,
                logit_std=args.logit_std,
                plateau_low=args.plateau_low,
                plateau_high=args.plateau_high,
            )

            with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=autocast_enabled):
                pred_velocity = model(x_t, t, text_embedding, attention_mask)
            # Loss math stays in fp32 outside autocast for numerical stability.
            raw_per_sample_loss = (pred_velocity.float() - target_velocity.float()).pow(2).mean(dim=(1, 2, 3))
            bin_losses = sigma_bin_means(raw_per_sample_loss, t)
            per_sample_loss = raw_per_sample_loss
            if args.timestep_weight:
                per_sample_loss = per_sample_loss * flow_training_loss_weight(t, target_velocity).mean(dim=(1, 2, 3))
            if args.min_snr_gamma > 0.0:
                per_sample_loss = per_sample_loss * flow_min_snr_loss_weight(
                    t,
                    target_velocity,
                    args.min_snr_gamma,
                ).mean(dim=(1, 2, 3))
            loss = per_sample_loss.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # Keep grad_norm as a tensor; only sync at log boundaries below.
            grad_norm_tensor = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            if ema_model is not None and step % args.ema_update_every == 0:
                update_ema_model(ema_model, raw_model, args.ema_decay)

            pending_steps.append(step)
            pending_losses.append(loss.detach())
            pending_raw_losses.append(raw_per_sample_loss.mean().detach())
            pending_sigma_means.append(t.float().mean().detach())
            pending_sigma_low_losses.append(bin_losses["low"].detach())
            pending_sigma_mid_losses.append(bin_losses["mid"].detach())
            pending_sigma_high_losses.append(bin_losses["high"].detach())
            pending_grad_norms.append(grad_norm_tensor.detach())
            pending_lrs.append(optimizer.param_groups[0]["lr"])

            should_log = step == 1 or (args.log_every and step % args.log_every == 0)
            if should_log:
                # One device->host transfer for all buffered steps.
                loss_vals = torch.stack(pending_losses).to("cpu", non_blocking=False).tolist()
                raw_loss_vals = torch.stack(pending_raw_losses).to("cpu", non_blocking=False).tolist()
                sigma_mean_vals = torch.stack(pending_sigma_means).to("cpu", non_blocking=False).tolist()
                sigma_low_vals = torch.stack(pending_sigma_low_losses).to("cpu", non_blocking=False).tolist()
                sigma_mid_vals = torch.stack(pending_sigma_mid_losses).to("cpu", non_blocking=False).tolist()
                sigma_high_vals = torch.stack(pending_sigma_high_losses).to("cpu", non_blocking=False).tolist()
                grad_vals = torch.stack(pending_grad_norms).to("cpu", non_blocking=False).tolist()
                if csv_writer is not None:
                    for s, lv, lr_v in zip(pending_steps, loss_vals, pending_lrs):
                        losses.append(lv)
                        loss_ma_row = sum(losses) / len(losses)
                        csv_writer.writerow({"step": s, "loss": lv, "loss_ma": loss_ma_row, "lr": lr_v})
                    csv_file.flush()
                else:
                    for lv in loss_vals:
                        losses.append(lv)
                loss_value = loss_vals[-1]
                grad_norm = grad_vals[-1]
                loss_ma = sum(losses) / len(losses)
                lr = pending_lrs[-1]
                pending_steps.clear()
                pending_losses.clear()
                pending_raw_losses.clear()
                pending_sigma_means.clear()
                pending_sigma_low_losses.clear()
                pending_sigma_mid_losses.clear()
                pending_sigma_high_losses.clear()
                pending_grad_norms.clear()
                pending_lrs.clear()

                elapsed = time.perf_counter() - train_started
                step_s = time.perf_counter() - step_started
                done = step - start_step
                eta = (elapsed / done) * max(args.steps - step, 0) if done else 0.0
                print(
                    f"step={step:05d} loss={loss_value:.6f} loss_ma={loss_ma:.6f} "
                    f"step_s={step_s:.2f}s elapsed={format_duration(elapsed)} eta={format_duration(eta)}"
                )
                if wandb_run is not None:
                    log_wandb_metrics(
                        step=step,
                        loss=loss_value,
                        loss_ma=loss_ma,
                        lr=lr,
                        step_s=step_s,
                        elapsed_s=elapsed,
                        eta_s=eta,
                        grad_norm=grad_norm,
                        raw_loss=raw_loss_vals[-1],
                        sigma_low_loss=sigma_low_vals[-1],
                        sigma_mid_loss=sigma_mid_vals[-1],
                        sigma_high_loss=sigma_high_vals[-1],
                        sigma_mean=sigma_mean_vals[-1],
                    )

            if args.save_every and args.checkpoint_dir and step % args.save_every == 0:
                try:
                    ckpt_path = Path(args.checkpoint_dir) / f"latent_dit_step_{step:06d}.pt"
                    train_args = build_train_args(args, config, latent_mean=latent_mean, latent_std=latent_std)
                    resume_path, inference_path = save_training_checkpoints(
                        ckpt_path,
                        raw_model,
                        config,
                        optimizer,
                        step,
                        ema_model=ema_model,
                        train_args=train_args,
                    )
                    print(f"saved resume checkpoint: {resume_path}")
                    print(f"saved inference checkpoint: {inference_path}")
                except Exception as exc:
                    print(f"warning: checkpoint save failed at step={step}: {exc}")

            if args.preview_every and step % args.preview_every == 0:
                assert preview_cache_config is not None and preview_records
                # Always preview through the uncompiled module so we don't
                # trigger an eval-mode recompile or wrap the EMA in dynamo.
                preview_model = ema_model if ema_model is not None else raw_model
                preview_dtype = ema_dtype if ema_model is not None else torch.float32
                preview_model.eval()
                try:
                    preview_vae = load_vae(preview_cache_config, hf_token=hf_token_value(args.hf_token))
                    try:
                        mse_noised, mse_denoised = save_denoise_previews(
                            model=preview_model,
                            vae=preview_vae,
                            record_paths=preview_records,
                            out_dir=args.preview_out_dir,
                            cache_config=preview_cache_config,
                            step=step,
                            preview_t=args.preview_t,
                            preview_steps=args.preview_steps,
                            preview_sampler=args.preview_sampler,
                            stork_derivative_order=args.preview_stork_derivative_order,
                            stork_substeps=args.preview_stork_substeps,
                            seed=args.preview_seed,
                            device=device,
                            dtype=preview_dtype,
                            latent_mean=latent_mean,
                            latent_std=latent_std,
                            flow_shift=args.flow_shift,
                        )
                    finally:
                        release_preview_vae(preview_vae, device)
                    print(
                        f"preview step={step:05d} "
                        f"weights={'ema' if ema_model is not None else 'model'} "
                        f"samples={len(preview_records)} "
                        f"latent_mse_noised={mse_noised:.6f} latent_mse_denoised={mse_denoised:.6f}"
                    )
                    if wandb_run is not None and args.preview_out_dir:
                        log_wandb_previews(
                            out_dir=args.preview_out_dir,
                            step=step,
                            preview_t=args.preview_t,
                            preview_sampler=args.preview_sampler,
                            preview_steps=args.preview_steps,
                            num_samples=len(preview_records),
                            mse_noised=mse_noised,
                            mse_denoised=mse_denoised,
                        )
                except Exception as exc:
                    print(f"warning: preview failed at step={step}: {exc}")
                finally:
                    if preview_model is raw_model:
                        raw_model.train()
    finally:
        if csv_file is not None:
            csv_file.flush()
            csv_file.close()
        finish_wandb_run(wandb_run)

    if args.save:
        try:
            train_args = build_train_args(args, config, latent_mean=latent_mean, latent_std=latent_std)
            resume_path, inference_path = save_training_checkpoints(
                args.save,
                raw_model,
                config,
                optimizer,
                last_step,
                ema_model=ema_model,
                train_args=train_args,
            )
            print(f"saved resume checkpoint: {resume_path}")
            print(f"saved inference checkpoint: {inference_path}")
        except Exception as exc:
            print(f"warning: final checkpoint save failed: {exc}")


if __name__ == "__main__":
    main()
