"""MambaAnchor-DiT denoiser for Boomer latent flow matching."""

from __future__ import annotations

import math
import inspect
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from boomer.models.latent_dit import AttentionRMSNorm, CaptionEmbedder, TimestepEmbedder, masked_mean


@dataclass(frozen=True)
class MambaAnchorDiTConfig:
    model_type: str = "mamba_anchor"
    latent_channels: int = 32
    latent_size: int = 16
    text_dim: int = 1536
    text_seq_len: int = 300
    hidden_dim: int = 1280
    depth: int = 24
    num_heads: int = 16
    mlp_ratio: float = 4.0
    y_norm: bool = True
    y_norm_scale_factor: float = 0.01
    num_anchors: int = 64
    text_attention_every: int = 2
    anchor_attention_every: int = 3
    mixer_type: str = "mamba3"
    mamba_d_state: int = 128
    mamba_headdim: int = 64
    mamba_chunk_size: int = 64
    mamba_is_mimo: bool = False


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (normed * self.weight).type_as(x)


class SwiGLU(nn.Module):
    def __init__(self, hidden_dim: int, ratio: float) -> None:
        super().__init__()
        inner_dim = int(hidden_dim * ratio)
        self.in_proj = nn.Linear(hidden_dim, inner_dim * 2)
        self.out_proj = nn.Linear(inner_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, value = self.in_proj(x).chunk(2, dim=-1)
        return self.out_proj(torch.nn.functional.silu(gate) * value)


class AdaLNModulation(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 3))
        nn.init.zeros_(self.net[1].weight)
        nn.init.zeros_(self.net[1].bias)
        self.net[1].bias.data[hidden_dim * 2 :].fill_(1.0)

    def forward(self, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shift, scale, gate = self.net(cond).chunk(3, dim=-1)
        return shift, scale, gate


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + scale[:, None, :]) + shift[:, None, :]


def maybe_add_sibling_mamba_repo() -> None:
    candidates = [
        Path(__file__).resolve().parents[3] / "mamba",
        Path("/content/mamba"),
    ]
    for path in candidates:
        if (path / "mamba_ssm").is_dir() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


def maybe_add_sibling_flame_repo() -> None:
    candidates = [
        Path(__file__).resolve().parents[3] / "flash-linear-attention",
        Path(__file__).resolve().parents[3] / "flame",
        Path("/content/flash-linear-attention"),
        Path("/content/flame"),
    ]
    for path in candidates:
        if (path / "flame").is_dir() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


class Mamba3Mixer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        *,
        d_state: int,
        headdim: int,
        chunk_size: int,
        is_mimo: bool,
    ) -> None:
        super().__init__()
        try:
            from mamba_ssm.modules.mamba3 import Mamba3
        except Exception:
            maybe_add_sibling_mamba_repo()
            from mamba_ssm.modules.mamba3 import Mamba3

        self.mixer = Mamba3(
            d_model=hidden_dim,
            d_state=d_state,
            headdim=headdim,
            is_mimo=is_mimo,
            chunk_size=chunk_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mixer(x)


class FlaMixer(nn.Module):
    """Thin adapter around flash-linear-attention sequence mixers.

    FLA layer constructor signatures move faster than this project, so this
    adapter filters a broad set of common kwargs against the installed class.
    """

    CLASS_BY_MIXER = {
        "fla_gated_deltanet": "GatedDeltaNet",
        "fla_gla": "GatedLinearAttention",
        "fla_delta_net": "DeltaNet",
        "fla_hgrn2": "HGRN2",
        "fla_based": "BasedLinearAttention",
    }

    def __init__(self, mixer_type: str, hidden_dim: int, *, num_heads: int, chunk_size: int) -> None:
        super().__init__()
        try:
            import fla.layers as fla_layers
        except Exception:
            maybe_add_sibling_flame_repo()
            import fla.layers as fla_layers

        class_name = self.CLASS_BY_MIXER[mixer_type]
        mixer_cls = getattr(fla_layers, class_name)
        signature = inspect.signature(mixer_cls)
        common_kwargs = {
            "hidden_size": hidden_dim,
            "hidden_dim": hidden_dim,
            "d_model": hidden_dim,
            "num_heads": num_heads,
            "num_attention_heads": num_heads,
            "head_dim": hidden_dim // num_heads,
            "mode": "chunk",
            "attn_mode": "chunk",
            "chunk_size": chunk_size,
            "use_short_conv": True,
            "conv_size": 4,
            "use_gate": True,
        }
        kwargs = {name: value for name, value in common_kwargs.items() if name in signature.parameters}
        try:
            self.mixer = mixer_cls(**kwargs)
        except TypeError as exc:
            raise TypeError(
                f"Could not instantiate fla.layers.{class_name} with filtered kwargs {kwargs}. "
                "Run tools/probe_sequence_mixers.py after installing flash-linear-attention to inspect the backend."
            ) from exc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.mixer(x)
        if isinstance(y, tuple):
            y = y[0]
        return y


class MlpMixer(nn.Module):
    """Import-free fallback mixer used for CPU tests and dry runs."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def scan_indices(latent_size: int, scan_mode: int, device: torch.device) -> torch.Tensor:
    grid = torch.arange(latent_size * latent_size, device=device).reshape(latent_size, latent_size)
    if scan_mode == 0:
        ordered = grid
    elif scan_mode == 1:
        ordered = torch.flip(grid, dims=(0, 1))
    elif scan_mode == 2:
        ordered = grid.t()
    elif scan_mode == 3:
        ordered = torch.flip(grid.t(), dims=(0, 1))
    else:
        raise ValueError(f"Unsupported scan mode: {scan_mode}")
    return ordered.reshape(-1)


class MambaAnchorBlock(nn.Module):
    def __init__(
        self,
        config: MambaAnchorDiTConfig,
        *,
        block_index: int,
    ) -> None:
        super().__init__()
        hidden_dim = config.hidden_dim
        self.scan_mode = block_index % 4
        self.use_text_attention = config.text_attention_every > 0 and block_index % config.text_attention_every == 0
        self.use_anchor_attention = config.anchor_attention_every > 0 and block_index % config.anchor_attention_every == 0

        self.mixer_norm = RMSNorm(hidden_dim)
        self.mixer_mod = AdaLNModulation(hidden_dim)
        if config.mixer_type == "mamba3":
            self.mixer = Mamba3Mixer(
                hidden_dim,
                d_state=config.mamba_d_state,
                headdim=config.mamba_headdim,
                chunk_size=config.mamba_chunk_size,
                is_mimo=config.mamba_is_mimo,
            )
        elif config.mixer_type.startswith("fla_"):
            self.mixer = FlaMixer(
                config.mixer_type,
                hidden_dim,
                num_heads=config.num_heads,
                chunk_size=config.mamba_chunk_size,
            )
        elif config.mixer_type in {"mlp", "fallback"}:
            self.mixer = MlpMixer(hidden_dim)
        else:
            raise ValueError(f"Unsupported mixer_type: {config.mixer_type}")

        if self.use_text_attention:
            self.text_norm = RMSNorm(hidden_dim)
            self.text_mod = AdaLNModulation(hidden_dim)
            self.text_attn = nn.MultiheadAttention(hidden_dim, config.num_heads, batch_first=True)

        if self.use_anchor_attention:
            self.anchor_text_norm = RMSNorm(hidden_dim)
            self.anchor_image_norm = RMSNorm(hidden_dim)
            self.anchor_mod = AdaLNModulation(hidden_dim)
            self.anchor_text_attn = nn.MultiheadAttention(hidden_dim, config.num_heads, batch_first=True)
            self.image_anchor_attn = nn.MultiheadAttention(hidden_dim, config.num_heads, batch_first=True)

        self.mlp_norm = RMSNorm(hidden_dim)
        self.mlp_mod = AdaLNModulation(hidden_dim)
        self.mlp = SwiGLU(hidden_dim, config.mlp_ratio)

    def _run_mixer(self, x: torch.Tensor, latent_size: int) -> torch.Tensor:
        order = scan_indices(latent_size, self.scan_mode, x.device)
        inverse = torch.empty_like(order)
        inverse[order] = torch.arange(order.numel(), device=x.device)
        mixed = self.mixer(x[:, order, :])
        return mixed[:, inverse, :]

    def forward(
        self,
        x: torch.Tensor,
        anchors: torch.Tensor,
        cond: torch.Tensor,
        text_tokens: torch.Tensor,
        text_key_padding_mask: torch.Tensor,
        *,
        latent_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shift, scale, gate = self.mixer_mod(cond)
        mixed = self._run_mixer(modulate(self.mixer_norm(x), shift, scale), latent_size)
        x = x + gate[:, None, :] * mixed

        if self.use_text_attention:
            shift, scale, gate = self.text_mod(cond)
            text_out = self.text_attn(
                modulate(self.text_norm(x), shift, scale),
                text_tokens,
                text_tokens,
                key_padding_mask=text_key_padding_mask,
                need_weights=False,
            )[0]
            x = x + gate[:, None, :] * text_out

        if self.use_anchor_attention:
            shift, scale, gate = self.anchor_mod(cond)
            normed_anchors = modulate(self.anchor_text_norm(anchors), shift, scale)
            anchor_delta = self.anchor_text_attn(
                normed_anchors,
                text_tokens,
                text_tokens,
                key_padding_mask=text_key_padding_mask,
                need_weights=False,
            )[0]
            anchors = anchors + gate[:, None, :] * anchor_delta
            anchor_out = self.image_anchor_attn(
                modulate(self.anchor_image_norm(x), shift, scale),
                anchors,
                anchors,
                need_weights=False,
            )[0]
            x = x + gate[:, None, :] * anchor_out

        shift, scale, gate = self.mlp_mod(cond)
        x = x + gate[:, None, :] * self.mlp(modulate(self.mlp_norm(x), shift, scale))
        return x, anchors


class MambaAnchorDiT(nn.Module):
    """Mamba3 + anchor-attention denoiser for f32c32/f64c128 latent grids."""

    def __init__(self, config: MambaAnchorDiTConfig = MambaAnchorDiTConfig()) -> None:
        super().__init__()
        self.config = config
        hidden_dim = config.hidden_dim
        self.latent_proj = nn.Linear(config.latent_channels, hidden_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, config.latent_size * config.latent_size, hidden_dim))
        self.time_embed = TimestepEmbedder(hidden_dim)
        self.caption_embedder = CaptionEmbedder(config.text_dim, hidden_dim, config.text_seq_len)
        self.attention_y_norm = (
            AttentionRMSNorm(hidden_dim, scale_factor=config.y_norm_scale_factor) if config.y_norm else None
        )
        self.anchors = nn.Parameter(torch.randn(1, config.num_anchors, hidden_dim) / math.sqrt(hidden_dim))
        self.blocks = nn.ModuleList([MambaAnchorBlock(config, block_index=i) for i in range(config.depth)])
        self.final_norm = RMSNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, config.latent_channels)
        nn.init.normal_(self.pos_embed, std=0.02)

    def apply_y_norm(self, caption_tokens: torch.Tensor) -> torch.Tensor:
        if self.attention_y_norm is None:
            return caption_tokens
        return self.attention_y_norm(caption_tokens)

    def null_condition(
        self,
        batch_size: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
        mask_dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.caption_embedder.null_condition(batch_size, device=device, dtype=dtype, mask_dtype=mask_dtype)

    def apply_condition_dropout(
        self,
        text_embedding: torch.Tensor,
        attention_mask: torch.Tensor,
        probability: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if probability <= 0.0:
            return text_embedding, attention_mask
        batch_size = text_embedding.shape[0]
        drop = torch.rand(batch_size, device=text_embedding.device) < probability
        if not bool(drop.any()):
            return text_embedding, attention_mask
        null_text, null_mask = self.null_condition(
            batch_size,
            device=text_embedding.device,
            dtype=text_embedding.dtype,
            mask_dtype=attention_mask.dtype,
        )
        text_embedding = text_embedding.clone()
        attention_mask = attention_mask.clone()
        text_embedding[drop] = null_text[drop]
        attention_mask[drop] = null_mask[drop]
        return text_embedding, attention_mask

    def forward(
        self,
        noisy_latent: torch.Tensor,
        timesteps: torch.Tensor,
        text_embedding: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, channels, height, width = noisy_latent.shape
        expected = self.config.latent_size
        if channels != self.config.latent_channels or height != expected or width != expected:
            raise ValueError(
                "Expected latent shape "
                f"(B, {self.config.latent_channels}, {expected}, {expected}), got {tuple(noisy_latent.shape)}"
            )
        if text_embedding.shape[-1] != self.config.text_dim:
            raise ValueError(f"text_embedding last dim must be {self.config.text_dim}, got {text_embedding.shape[-1]}")

        text_tokens = self.caption_embedder(text_embedding)
        text_tokens = self.apply_y_norm(text_tokens)
        text_key_padding_mask = attention_mask == 0
        cond = self.time_embed(timesteps) + masked_mean(text_tokens, attention_mask)

        tokens = noisy_latent.flatten(2).transpose(1, 2)
        x = self.latent_proj(tokens) + self.pos_embed
        anchors = self.anchors.expand(batch, -1, -1)
        for block in self.blocks:
            x, anchors = block(
                x,
                anchors,
                cond,
                text_tokens,
                text_key_padding_mask,
                latent_size=self.config.latent_size,
            )
        x = self.out_proj(self.final_norm(x))
        return x.transpose(1, 2).reshape(batch, channels, height, width)
