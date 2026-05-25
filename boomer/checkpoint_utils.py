"""Shared checkpoint load/save helpers for Boomer training tools."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from boomer.latent_norm import LatentStat
from boomer.models.factory import Config, config_from_dict


def load_torch_file(path: str | Path, map_location: str = "cpu") -> dict[str, Any]:
    """Load a checkpoint or cache record, preferring safe tensor-only loads."""
    path = Path(path)
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        pass
    except Exception:
        pass
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def flatten_state_dict(state_dict: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    return {f"{prefix}.{name}": tensor for name, tensor in state_dict.items()}


def unflatten_state_dict(
    tensors: dict[str, torch.Tensor],
    prefix: str,
) -> dict[str, torch.Tensor]:
    prefix_dot = f"{prefix}."
    return {
        name[len(prefix_dot) :]: tensor
        for name, tensor in tensors.items()
        if name.startswith(prefix_dot)
    }


def collect_train_args(
    *,
    ema_decay: float = 0.0,
    ema_update_every: int = 1,
    cond_dropout: float = 0.0,
    latent_mean: LatentStat = 0.0,
    latent_std: LatentStat = 1.0,
    t_sampler: str = "uniform",
    t_bias: float = 2.0,
    flow_shift: float = 1.0,
    logit_mean: float = 0.0,
    logit_std: float = 1.0,
    plateau_low: float = 0.1,
    plateau_high: float = 0.9,
    y_norm: bool = False,
    y_norm_scale_factor: float = 0.01,
    grad_clip: float = 1.0,
    timestep_weight: bool = False,
    min_snr_gamma: float = 0.0,
    lr_scheduler: str = "constant",
    warmup_steps: int = 0,
    min_lr_ratio: float = 0.1,
) -> dict[str, Any]:
    return {
        "ema_decay": ema_decay,
        "ema_update_every": ema_update_every,
        "cond_dropout": cond_dropout,
        "latent_mean": latent_mean,
        "latent_std": latent_std,
        "t_sampler": t_sampler,
        "t_bias": t_bias,
        "flow_shift": flow_shift,
        "logit_mean": logit_mean,
        "logit_std": logit_std,
        "plateau_low": plateau_low,
        "plateau_high": plateau_high,
        "y_norm": y_norm,
        "y_norm_scale_factor": y_norm_scale_factor,
        "grad_clip": grad_clip,
        "timestep_weight": timestep_weight,
        "min_snr_gamma": min_snr_gamma,
        "lr_scheduler": lr_scheduler,
        "warmup_steps": warmup_steps,
        "min_lr_ratio": min_lr_ratio,
    }


def inference_checkpoint_path(path: str | Path) -> Path:
    return Path(path).with_suffix(".safetensors")


def save_resume_checkpoint(
    path: str | Path,
    model: nn.Module,
    config: Config,
    optimizer: torch.optim.Optimizer,
    step: int,
    *,
    ema_model: nn.Module | None = None,
    train_args: dict[str, Any] | None = None,
) -> Path:
    """Save a full PyTorch checkpoint for exact training resume."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint: dict[str, Any] = {
        "model": model.state_dict(),
        "config": config.__dict__,
        "optimizer": optimizer.state_dict(),
        "steps": step,
        "train_args": train_args or {},
    }
    if ema_model is not None:
        checkpoint["ema_model"] = ema_model.state_dict()
    torch.save(checkpoint, path)
    return path


def save_inference_checkpoint(
    path: str | Path,
    model: nn.Module,
    config: Config,
    step: int,
    *,
    ema_model: nn.Module | None = None,
    train_args: dict[str, Any] | None = None,
) -> Path:
    """Save model/EMA weights in safetensors for faster inference-oriented loads."""
    from safetensors.torch import save_file

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tensors = flatten_state_dict(model.state_dict(), "model")
    if ema_model is not None:
        tensors.update(flatten_state_dict(ema_model.state_dict(), "ema_model"))
    metadata = {
        "steps": str(step),
        "config": json.dumps(config.__dict__),
        "train_args": json.dumps(train_args or {}),
    }
    save_file(tensors, path, metadata=metadata)
    return path


def save_training_checkpoints(
    resume_path: str | Path,
    model: nn.Module,
    config: Config,
    optimizer: torch.optim.Optimizer,
    step: int,
    *,
    ema_model: nn.Module | None = None,
    train_args: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Write a full .pt resume checkpoint and a model-only .safetensors inference checkpoint."""
    resume_path = save_resume_checkpoint(
        resume_path,
        model,
        config,
        optimizer,
        step,
        ema_model=ema_model,
        train_args=train_args,
    )
    inference_path = save_inference_checkpoint(
        inference_checkpoint_path(resume_path),
        model,
        config,
        step,
        ema_model=ema_model,
        train_args=train_args,
    )
    return resume_path, inference_path


def load_inference_checkpoint(path: str | Path, map_location: str = "cpu") -> dict[str, Any]:
    from safetensors import safe_open

    path = Path(path)
    tensors: dict[str, torch.Tensor] = {}
    metadata: dict[str, str] = {}
    with safe_open(path, framework="pt", device=map_location) as opened:
        metadata = dict(opened.metadata() or {})
        for key in opened.keys():
            tensors[key] = opened.get_tensor(key)

    model_state = unflatten_state_dict(tensors, "model")
    if not model_state:
        raise ValueError(f"{path} does not contain model.* tensors")

    checkpoint: dict[str, Any] = {
        "model": model_state,
        "config": json.loads(metadata.get("config", "{}")),
        "train_args": json.loads(metadata.get("train_args", "{}")),
        "steps": int(metadata.get("steps", "0")),
    }
    ema_state = unflatten_state_dict(tensors, "ema_model")
    if ema_state:
        checkpoint["ema_model"] = ema_state
    return checkpoint


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> dict[str, Any]:
    path = Path(path)
    if path.suffix == ".safetensors":
        return load_inference_checkpoint(path, map_location=map_location)
    return load_torch_file(path, map_location=map_location)


def config_from_checkpoint(checkpoint: dict[str, Any]) -> Config:
    raw = checkpoint.get("config") or {}
    return config_from_dict(raw)


def select_model_state(checkpoint: dict[str, Any], weights: str) -> tuple[dict[str, torch.Tensor], str]:
    if weights == "ema":
        if "ema_model" not in checkpoint:
            raise ValueError("Checkpoint does not contain EMA weights.")
        return checkpoint["ema_model"], "ema"
    if weights == "auto" and "ema_model" in checkpoint:
        return checkpoint["ema_model"], "ema"
    return checkpoint["model"], "model"


def load_compatible_state_dict(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
) -> list[str]:
    """Load matching checkpoint tensors and skip renamed or reshaped tensors."""
    model_state = model.state_dict()
    compatible_state: dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    for name, value in state_dict.items():
        if name not in model_state:
            skipped.append(name)
            continue
        if tuple(model_state[name].shape) != tuple(value.shape):
            skipped.append(name)
            continue
        compatible_state[name] = value

    model.load_state_dict(compatible_state, strict=False)
    return skipped
