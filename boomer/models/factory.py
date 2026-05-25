"""Model factory helpers for Boomer denoisers."""

from __future__ import annotations

from typing import Any

from boomer.models.latent_dit import LatentDiT, LatentDiTConfig
from boomer.models.mamba_anchor_dit import MambaAnchorDiT, MambaAnchorDiTConfig
from boomer.models.boomer_fla_dit import BoomerFLADiT, BoomerFLADiTConfig


Config = LatentDiTConfig | MambaAnchorDiTConfig | BoomerFLADiTConfig


def config_from_dict(raw: dict[str, Any]) -> Config:
    model_type = raw.get("model_type", "latent_dit")
    if model_type == "sana_fla":
        model_type = "boomer_fla"
        raw = {**raw, "model_type": model_type}
    if model_type == "latent_dit":
        allowed = LatentDiTConfig.__dataclass_fields__.keys()
        return LatentDiTConfig(**{key: value for key, value in raw.items() if key in allowed})
    if model_type == "mamba_anchor":
        allowed = MambaAnchorDiTConfig.__dataclass_fields__.keys()
        return MambaAnchorDiTConfig(**{key: value for key, value in raw.items() if key in allowed})
    if model_type == "boomer_fla":
        allowed = BoomerFLADiTConfig.__dataclass_fields__.keys()
        return BoomerFLADiTConfig(**{key: value for key, value in raw.items() if key in allowed})
    raise ValueError(f"Unsupported model_type: {model_type}")


def create_model(config: Config) -> LatentDiT | MambaAnchorDiT | BoomerFLADiT:
    if isinstance(config, LatentDiTConfig) or config.model_type == "latent_dit":
        return LatentDiT(config)  # type: ignore[arg-type]
    if isinstance(config, MambaAnchorDiTConfig) or config.model_type == "mamba_anchor":
        return MambaAnchorDiT(config)  # type: ignore[arg-type]
    if isinstance(config, BoomerFLADiTConfig) or config.model_type in {"boomer_fla", "sana_fla"}:
        return BoomerFLADiT(config)  # type: ignore[arg-type]
    raise ValueError(f"Unsupported model config: {config}")
