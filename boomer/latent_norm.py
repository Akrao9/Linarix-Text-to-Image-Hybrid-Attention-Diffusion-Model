"""Latent normalization helpers used by training and sampling tools."""

from __future__ import annotations

from typing import Any

import torch


LatentStat = float | int | list[float] | tuple[float, ...] | torch.Tensor


def _stat_tensor(value: LatentStat, latent: torch.Tensor, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value, device=latent.device, dtype=latent.dtype)
    if tensor.ndim == 0:
        return tensor
    if tensor.ndim == 1:
        if latent.ndim < 2:
            raise ValueError(f"{name} is channel-wise but latent has shape {tuple(latent.shape)}")
        if tensor.numel() != latent.shape[1]:
            raise ValueError(f"{name} has {tensor.numel()} channels, expected {latent.shape[1]}")
        return tensor.view(1, tensor.numel(), *([1] * (latent.ndim - 2)))
    raise ValueError(f"{name} must be a scalar or 1D channel list, got shape {tuple(tensor.shape)}")


def _check_std(std: LatentStat, latent: torch.Tensor) -> torch.Tensor:
    tensor = _stat_tensor(std, latent, "latent std")
    if bool((tensor <= 0).any()):
        raise ValueError(f"latent std must be positive, got {std}")
    return tensor


def normalize_latent(latent: torch.Tensor, mean: LatentStat = 0.0, std: LatentStat = 1.0) -> torch.Tensor:
    mean_tensor = _stat_tensor(mean, latent, "latent mean")
    std_tensor = _check_std(std, latent)
    return (latent - mean_tensor) / std_tensor


def denormalize_latent(latent: torch.Tensor, mean: LatentStat = 0.0, std: LatentStat = 1.0) -> torch.Tensor:
    mean_tensor = _stat_tensor(mean, latent, "latent mean")
    std_tensor = _check_std(std, latent)
    return latent * std_tensor + mean_tensor


def latent_stat_summary(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        tensor = torch.as_tensor(value, dtype=torch.float32)
        return (
            f"channels={tensor.numel()} mean={tensor.mean().item():.6f} "
            f"min={tensor.min().item():.6f} max={tensor.max().item():.6f}"
        )
    return str(value)
