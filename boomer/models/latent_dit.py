"""Minimal latent DiT baseline for proving Boomer training."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class LatentDiTConfig:
    model_type: str = "latent_dit"
    latent_channels: int = 32
    latent_size: int = 16
    text_dim: int = 1536
    text_seq_len: int = 300
    hidden_dim: int = 384
    depth: int = 6
    num_heads: int = 6
    mlp_ratio: float = 4.0
    cross_attention: bool = False
    y_norm: bool = False
    y_norm_scale_factor: float = 0.01


class AttentionRMSNorm(nn.Module):
    """Sana-style learnable RMS norm on caption hidden states."""

    def __init__(self, dim: int, scale_factor: float = 0.01, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim) * scale_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = x.float() * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        weight = self.weight.view(*([1] * (x.ndim - 2)), -1)
        return (weight * normed).type_as(x)


class CaptionEmbedder(nn.Module):
    """Sana-style caption projection plus CFG null caption storage."""

    def __init__(self, in_channels: int, hidden_size: int, token_num: int) -> None:
        super().__init__()
        self.y_proj = nn.Sequential(
            nn.Linear(in_channels, hidden_size),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_size, hidden_size),
        )
        null_init = torch.randn(token_num, in_channels) / math.sqrt(in_channels)
        self.null_text_embedding = nn.Parameter(null_init.unsqueeze(0))

    def forward(self, caption: torch.Tensor) -> torch.Tensor:
        return self.y_proj(caption)

    def null_condition(
        self,
        batch_size: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
        mask_dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        text = self.null_text_embedding.expand(batch_size, -1, -1).to(device=device, dtype=dtype)
        mask = torch.ones(
            batch_size,
            text.shape[1],
            device=device,
            dtype=mask_dtype or torch.long,
        )
        return text, mask


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        dtype = self.net[0].weight.dtype
        return self.net(timesteps.to(dtype=dtype).view(-1, 1))


class LatentDiT(nn.Module):
    """A tiny transformer baseline over f32c32 latent tokens.

    Input/output shape is `(batch, 32, 16, 16)` for 512px Sana f32c32 latents.
    Text conditioning uses masked mean pooled Gemma states plus optional cross-attn.
    """

    def __init__(self, config: LatentDiTConfig = LatentDiTConfig()) -> None:
        super().__init__()
        self.config = config
        hidden_dim = config.hidden_dim
        self.token_proj = nn.Linear(config.latent_channels, hidden_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, config.latent_size * config.latent_size, hidden_dim))
        self.time_embed = TimestepEmbedder(hidden_dim)
        self.caption_embedder = CaptionEmbedder(
            in_channels=config.text_dim,
            hidden_size=hidden_dim,
            token_num=config.text_seq_len,
        )
        self.attention_y_norm = (
            AttentionRMSNorm(hidden_dim, scale_factor=config.y_norm_scale_factor) if config.y_norm else None
        )
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_dim=hidden_dim,
                    num_heads=config.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    cross_attention=config.cross_attention,
                )
                for _ in range(config.depth)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dim)
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
        """Learned raw null caption with all-ones mask for CFG cross-attn."""
        return self.caption_embedder.null_condition(
            batch_size,
            device=device,
            dtype=dtype,
            mask_dtype=mask_dtype,
        )

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
        dropped_embedding = text_embedding.clone()
        dropped_mask = attention_mask.clone()
        dropped_embedding[drop] = null_text[drop]
        # Keep null tokens unmasked so cross-attention never receives an all-masked sequence.
        dropped_mask[drop] = null_mask[drop]
        return dropped_embedding, dropped_mask

    def _validate_text_embedding(self, text_embedding: torch.Tensor) -> None:
        if text_embedding.shape[-1] != self.config.text_dim:
            raise ValueError(
                f"text_embedding last dim must be {self.config.text_dim}, got {text_embedding.shape[-1]}"
            )

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
        self._validate_text_embedding(text_embedding)

        caption_tokens = self.caption_embedder(text_embedding)
        caption_tokens = self.apply_y_norm(caption_tokens)
        tokens = noisy_latent.flatten(2).transpose(1, 2)
        x = self.token_proj(tokens) + self.pos_embed
        cond = self.time_embed(timesteps) + masked_mean(caption_tokens, attention_mask)
        x = x + cond[:, None, :]
        text_tokens = caption_tokens if self.config.cross_attention else None
        text_key_padding_mask = attention_mask == 0 if self.config.cross_attention else None
        for block in self.blocks:
            x = block(x, text_tokens=text_tokens, text_key_padding_mask=text_key_padding_mask)
        x = self.out_proj(self.norm(x))
        return x.transpose(1, 2).reshape(batch, channels, height, width)


class DiTBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: float, cross_attention: bool) -> None:
        super().__init__()
        self.cross_attention = cross_attention
        self.self_norm = nn.LayerNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=0.0, batch_first=True)
        if cross_attention:
            self.cross_norm = nn.LayerNorm(hidden_dim)
            self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=0.0, batch_first=True)
        self.mlp_norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, int(hidden_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(hidden_dim * mlp_ratio), hidden_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        text_tokens: torch.Tensor | None = None,
        text_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        normed = self.self_norm(x)
        self_out = self.self_attn(normed, normed, normed, need_weights=False)[0]
        x = x + self_out
        if self.cross_attention:
            if text_tokens is None:
                raise ValueError("text_tokens are required when cross_attention=True")
            cross_out = self.cross_attn(
                self.cross_norm(x),
                text_tokens,
                text_tokens,
                key_padding_mask=text_key_padding_mask,
                need_weights=False,
            )[0]
            x = x + cross_out
        x = x + self.mlp(self.mlp_norm(x))
        return x


def masked_mean(text_embedding: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.to(device=text_embedding.device, dtype=text_embedding.dtype)
    while mask.ndim < text_embedding.ndim:
        mask = mask.unsqueeze(-1)
    summed = (text_embedding * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return summed / denom
