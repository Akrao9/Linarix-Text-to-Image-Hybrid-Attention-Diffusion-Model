"""Boomer FLA DiT denoiser."""

from __future__ import annotations

import math
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from boomer.models.latent_dit import AttentionRMSNorm, CaptionEmbedder, TimestepEmbedder


@dataclass(frozen=True)
class BoomerFLADiTConfig:
    model_type: str = "boomer_fla"
    latent_channels: int = 32
    latent_size: int = 16
    text_dim: int = 1536
    text_seq_len: int = 300
    hidden_dim: int = 1152
    depth: int = 28
    num_heads: int = 16
    mlp_ratio: float = 2.5
    y_norm: bool = True
    y_norm_scale_factor: float = 0.01
    mixer_type: str = "fla_linear"
    fla_mode: str = "chunk"
    fla_feature_map: str = "relu"
    fla_bidirectional: bool = False
    use_short_conv: bool = False
    conv_size: int = 4
    image_attention_every: int = 0
    image_attention_backend: str = "sdpa"
    image_attention_rope: bool = False
    image_rope_theta: float = 10000.0
    cross_attention_backend: str = "sdpa"
    cross_attention_qk_norm: bool = True
    patch_size: int = 1


def maybe_add_sibling_fla_repo() -> None:
    candidates = [
        Path(__file__).resolve().parents[3] / "flash-linear-attention",
        Path("/content/flash-linear-attention"),
        Path("/content/flame"),
    ]
    for path in candidates:
        if (path / "fla").is_dir() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


def maybe_add_sibling_flash_attention_repo() -> None:
    candidates = [
        Path(__file__).resolve().parents[3] / "flash-attention" / "hopper",
        Path(__file__).resolve().parents[3] / "flash-attention",
        Path("/content/flash-attention/hopper"),
        Path("/content/flash-attention"),
    ]
    for path in candidates:
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + scale) + shift


class ConvLayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        kernel_size: int,
        *,
        groups: int = 1,
        bias: bool = False,
        act: str | None = None,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_dim,
            out_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=groups,
            bias=bias,
        )
        self.act = nn.SiLU() if act == "silu" else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))


class GLUMBConv(nn.Module):
    """Sana GLUMBConv FFN: 1x1 expand, depthwise spatial conv, GLU, 1x1 project."""

    def __init__(self, hidden_dim: int, mlp_ratio: float) -> None:
        super().__init__()
        inner_dim = int(hidden_dim * mlp_ratio)
        self.inner_dim = inner_dim
        self.inverted_conv = ConvLayer(hidden_dim, inner_dim * 2, 1, bias=True, act="silu")
        self.depth_conv = ConvLayer(inner_dim * 2, inner_dim * 2, 3, groups=inner_dim * 2, bias=True)
        self.point_conv = ConvLayer(inner_dim, hidden_dim, 1, bias=False)
        self.glu_act = nn.SiLU()

    def forward(self, x: torch.Tensor, *, height: int, width: int) -> torch.Tensor:
        batch, tokens, channels = x.shape
        if tokens != height * width:
            raise ValueError(f"Expected {height * width} image tokens, got {tokens}")
        x = x.reshape(batch, height, width, channels).permute(0, 3, 1, 2).contiguous()
        x = self.inverted_conv(x)
        x = self.depth_conv(x)
        x, gate = x.chunk(2, dim=1)
        x = x * self.glu_act(gate)
        x = self.point_conv(x)
        return x.reshape(batch, channels, tokens).transpose(1, 2).contiguous()


class TorchSelfAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.attn(x, x, x, need_weights=False)[0]


class RoPE2D(nn.Module):
    """2D RoPE for image tokens on a fixed H×W grid (row-major flattening).

    Splits head_dim in half: the first half encodes height, the second width.
    Each half uses standard 1D RoPE with shared cos/sin tables per axis.
    """

    def __init__(self, head_dim: int, grid_size: int, *, theta: float = 10000.0) -> None:
        super().__init__()
        if head_dim % 4 != 0:
            raise ValueError(
                f"head_dim={head_dim} must be divisible by 4 for 2D RoPE "
                f"(half for H, half for W, each needing pairs)"
            )
        if grid_size <= 0:
            raise ValueError(f"grid_size must be positive, got {grid_size}")
        if theta <= 0.0:
            raise ValueError(f"theta must be positive, got {theta}")
        self.head_dim = head_dim
        self.grid_size = grid_size
        self.half_dim = head_dim // 2

        freqs = 1.0 / (theta ** (torch.arange(0, self.half_dim, 2).float() / self.half_dim))
        token_idx = torch.arange(grid_size * grid_size)
        h_idx = token_idx // grid_size
        w_idx = token_idx % grid_size

        def axis_tables(pos_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            angles = torch.outer(pos_idx.float(), freqs)
            cos = torch.cat([angles.cos(), angles.cos()], dim=-1)[None, :, None, :]
            sin = torch.cat([angles.sin(), angles.sin()], dim=-1)[None, :, None, :]
            return cos, sin

        cos_h, sin_h = axis_tables(h_idx)
        cos_w, sin_w = axis_tables(w_idx)
        self.register_buffer("cos_h", cos_h, persistent=False)
        self.register_buffer("sin_h", sin_h, persistent=False)
        self.register_buffer("cos_w", cos_w, persistent=False)
        self.register_buffer("sin_w", sin_w, persistent=False)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def _apply_axis_rope(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        return x * cos.to(dtype=x.dtype) + self._rotate_half(x) * sin.to(dtype=x.dtype)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, tokens, num_heads, head_dim = q.shape
        if head_dim != self.head_dim:
            raise ValueError(f"expected head_dim={self.head_dim}, got {head_dim}")
        expected_tokens = self.grid_size * self.grid_size
        if tokens != expected_tokens:
            raise ValueError(f"expected {expected_tokens} image tokens, got {tokens}")

        q_h, q_w = q.chunk(2, dim=-1)
        k_h, k_w = k.chunk(2, dim=-1)
        q_h = self._apply_axis_rope(q_h, self.cos_h, self.sin_h)
        q_w = self._apply_axis_rope(q_w, self.cos_w, self.sin_w)
        k_h = self._apply_axis_rope(k_h, self.cos_h, self.sin_h)
        k_w = self._apply_axis_rope(k_w, self.cos_w, self.sin_w)
        return torch.cat([q_h, q_w], dim=-1), torch.cat([k_h, k_w], dim=-1)


class FullImageSelfAttention(nn.Module):
    """Full image-token attention for the small DC-AE latent grid."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        *,
        backend: str = "sdpa",
        grid_size: int | None = None,
        rope: bool = False,
        rope_theta: float = 10000.0,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}")
        if backend not in {"sdpa", "flash3", "flash4", "auto"}:
            raise ValueError(f"Unsupported image_attention_backend: {backend}")
        if rope and grid_size is None:
            raise ValueError("grid_size is required when rope=True")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.backend = backend
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        self.rope = (
            RoPE2D(self.head_dim, grid_size, theta=rope_theta)
            if rope and grid_size is not None
            else None
        )
        self._flash3_attn_func = None
        self._flash3_import_attempted = False
        self._flash4_attn_func = None
        self._flash4_import_attempted = False

    def _get_flash3_attn_func(self):
        if self._flash3_import_attempted:
            return self._flash3_attn_func
        self._flash3_import_attempted = True
        maybe_add_sibling_flash_attention_repo()
        try:
            from flash_attn_interface import flash_attn_func
        except Exception:
            try:
                from flash_attn.flash_attn_interface import flash_attn_func
            except Exception:
                flash_attn_func = None
        self._flash3_attn_func = flash_attn_func
        return self._flash3_attn_func

    def _get_flash4_attn_func(self):
        if self._flash4_import_attempted:
            return self._flash4_attn_func
        self._flash4_import_attempted = True
        maybe_add_sibling_flash_attention_repo()
        try:
            from flash_attn.cute.interface import flash_attn_func
        except Exception:
            flash4_paths = [
                Path(__file__).resolve().parents[3] / "flash-attention" / "flash_attn",
                Path("/content/flash-attention/flash_attn"),
            ]
            existing_paths = [str(path) for path in flash4_paths if (path / "cute").is_dir()]
            if existing_paths:
                for name in list(sys.modules):
                    if name == "flash_attn" or name.startswith("flash_attn."):
                        del sys.modules[name]
                flash_attn_pkg = types.ModuleType("flash_attn")
                flash_attn_pkg.__path__ = existing_paths
                sys.modules["flash_attn"] = flash_attn_pkg
                try:
                    from flash_attn.cute.interface import flash_attn_func
                except Exception:
                    flash_attn_func = None
            else:
                flash_attn_func = None
        self._flash4_attn_func = flash_attn_func
        return self._flash4_attn_func

    def _flash3_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        flash_attn_func = self._get_flash3_attn_func()
        if flash_attn_func is None:
            raise ImportError(
                "image_attention_backend='flash3' requires FlashAttention-3. "
                "Install it or use --image-attn-backend sdpa."
            )
        out = flash_attn_func(q, k, v, causal=False)
        if isinstance(out, tuple):
            out = out[0]
        return out

    def _flash4_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        flash_attn_func = self._get_flash4_attn_func()
        if flash_attn_func is None:
            raise ImportError(
                "image_attention_backend='flash4' requires FlashAttention-4/CuTe. "
                "Install flash-attn-4 or use --image-attn-backend sdpa."
            )
        out = flash_attn_func(q, k, v, causal=False)
        if isinstance(out, tuple):
            out = out[0]
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        if self.rope is not None:
            q, k = self.rope(q, k)

        use_flash = self.backend in {"flash3", "flash4", "auto"} and x.is_cuda and x.dtype in {
            torch.float16,
            torch.bfloat16,
        }
        if self.backend == "flash4" and use_flash:
            out = self._flash4_attention(q, k, v)
        elif self.backend == "flash3" and use_flash:
            out = self._flash3_attention(q, k, v)
        elif self.backend == "auto" and use_flash:
            try:
                out = self._flash4_attention(q, k, v)
            except Exception:
                try:
                    out = self._flash3_attention(q, k, v)
                except Exception:
                    use_flash = False
        if self.backend in {"flash3", "flash4"} and not use_flash:
            raise RuntimeError(
                f"image_attention_backend='{self.backend}' requires CUDA fp16/bf16 tensors; got {x.device} {x.dtype}"
            )
        if not use_flash:
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
            out = out.transpose(1, 2)

        out = out.reshape(batch, tokens, channels)
        return self.out_proj(out)


class SanaMultiHeadCrossAttention(nn.Module):
    """Sana-style cross-attention with optional q/k norm and SDPA/xformers kernels."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        *,
        backend: str = "sdpa",
        qk_norm: bool = True,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}")
        if backend not in {"sdpa", "xformers", "auto"}:
            raise ValueError(f"Unsupported cross_attention_backend: {backend}")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.backend = backend
        self.q_linear = nn.Linear(hidden_dim, hidden_dim)
        self.kv_linear = nn.Linear(hidden_dim, hidden_dim * 2)
        self.q_norm = AttentionRMSNorm(hidden_dim, scale_factor=1.0, eps=1e-6) if qk_norm else nn.Identity()
        self.k_norm = AttentionRMSNorm(hidden_dim, scale_factor=1.0, eps=1e-6) if qk_norm else nn.Identity()
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        # adaLN-Zero style: cross-attn starts as a no-op so Gemma text cannot spike GDN states early.
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self._xformers_ops = None
        self._xformers_import_attempted = False

    def _get_xformers_ops(self):
        if self._xformers_import_attempted:
            return self._xformers_ops
        self._xformers_import_attempted = True
        try:
            import xformers.ops as xops
        except Exception:
            xops = None
        self._xformers_ops = xops
        return self._xformers_ops

    def _xformers_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        xops = self._get_xformers_ops()
        if xops is None:
            raise ImportError(
                "cross_attention_backend='xformers' requires xformers. "
                "Install it or use --cross-attn-backend sdpa."
            )

        batch, image_tokens = q.shape[:2]
        text_tokens = k.shape[1]
        q_lens = [image_tokens] * batch
        q_compact = q.reshape(1, batch * image_tokens, self.num_heads, self.head_dim)
        if key_padding_mask is None:
            kv_lens = [text_tokens] * batch
            k_compact = k.reshape(1, batch * text_tokens, self.num_heads, self.head_dim)
            v_compact = v.reshape(1, batch * text_tokens, self.num_heads, self.head_dim)
        else:
            valid_mask = ~key_padding_mask.bool()
            kv_lens = valid_mask.sum(dim=1).tolist()
            if any(length <= 0 for length in kv_lens):
                raise ValueError("xformers cross-attention received a sample with zero valid text tokens")
            k_compact = torch.cat([k[index, valid_mask[index]] for index in range(batch)], dim=0).unsqueeze(0)
            v_compact = torch.cat([v[index, valid_mask[index]] for index in range(batch)], dim=0).unsqueeze(0)

        attn_bias = xops.fmha.BlockDiagonalMask.from_seqlens(q_lens, kv_lens)
        out = xops.memory_efficient_attention(q_compact, k_compact, v_compact, attn_bias=attn_bias, p=0.0)
        return out.reshape(batch, image_tokens, self.num_heads, self.head_dim)

    def _sdpa_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
        attn_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn_mask = attn_bias
        if attn_mask is None and key_padding_mask is not None:
            attn_mask = key_padding_mask[:, None, None, :].to(dtype=q.dtype)
            attn_mask = attn_mask.masked_fill(attn_mask > 0, -10000.0)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
        return out.transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        attn_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, image_tokens, channels = x.shape
        # Sana order: linear projection first, then per-token q/k RMSNorm before head split.
        # This caps dot-product growth when cond carries high-magnitude Gemma caption states.
        q = self.q_linear(x)
        q = self.q_norm(q).reshape(batch, image_tokens, self.num_heads, self.head_dim)
        k, v = self.kv_linear(cond).chunk(2, dim=-1)
        k = self.k_norm(k).reshape(batch, cond.shape[1], self.num_heads, self.head_dim)
        v = v.reshape(batch, cond.shape[1], self.num_heads, self.head_dim)

        use_xformers = self.backend in {"xformers", "auto"} and x.is_cuda and x.dtype in {
            torch.float16,
            torch.bfloat16,
        }
        if use_xformers:
            try:
                out = self._xformers_attention(q, k, v, key_padding_mask)
            except Exception:
                if self.backend == "xformers":
                    raise
                use_xformers = False
        if self.backend == "xformers" and not use_xformers:
            raise RuntimeError(
                f"cross_attention_backend='xformers' requires CUDA fp16/bf16 tensors; got {x.device} {x.dtype}"
            )
        if not use_xformers:
            out = self._sdpa_attention(q, k, v, key_padding_mask, attn_bias)

        return self.proj(out.reshape(batch, image_tokens, channels))


class FLASelfMixer(nn.Module):
    def __init__(self, config: BoomerFLADiTConfig, *, layer_idx: int) -> None:
        super().__init__()
        try:
            import fla.layers as fla_layers
        except Exception:
            maybe_add_sibling_fla_repo()
            import fla.layers as fla_layers

        hidden_dim = config.hidden_dim
        self.bidirectional = config.fla_bidirectional
        if config.mixer_type == "fla_linear":
            self.mixer = fla_layers.LinearAttention(
                hidden_size=hidden_dim,
                num_heads=config.num_heads,
                mode=config.fla_mode,
                feature_map=config.fla_feature_map,
                output_norm="rmsnorm",
                layer_idx=layer_idx,
            )
        elif config.mixer_type == "fla_gated_deltanet":
            self.mixer = fla_layers.GatedDeltaNet(
                hidden_size=hidden_dim,
                num_heads=config.num_heads,
                head_dim=hidden_dim // config.num_heads,
                expand_v=1,
                mode=config.fla_mode,
                use_short_conv=config.use_short_conv,
                conv_size=config.conv_size,
                layer_idx=layer_idx,
            )
        elif config.mixer_type == "fla_gla":
            self.mixer = fla_layers.GatedLinearAttention(
                hidden_size=hidden_dim,
                num_heads=config.num_heads,
                mode=config.fla_mode,
                feature_map=config.fla_feature_map,
                use_short_conv=config.use_short_conv,
                conv_size=config.conv_size,
                layer_idx=layer_idx,
            )
        else:
            raise ValueError(f"Unsupported FLA mixer_type: {config.mixer_type}")

    @staticmethod
    def _run_mixer(mixer: nn.Module, x: torch.Tensor) -> torch.Tensor:
        y = mixer(x)
        if isinstance(y, tuple):
            y = y[0]
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self._run_mixer(self.mixer, x)
        if not self.bidirectional:
            return y
        y_rev = self._run_mixer(self.mixer, x.flip(1)).flip(1)
        return 0.5 * (y + y_rev)


class BoomerFLABlock(nn.Module):
    def __init__(self, config: BoomerFLADiTConfig, *, layer_idx: int) -> None:
        super().__init__()
        hidden_dim = config.hidden_dim
        self.use_image_attention = (
            config.image_attention_every > 0 and (layer_idx + 1) % config.image_attention_every == 0
        )
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        if config.mixer_type in {"torch", "fallback"}:
            self.self_attn = TorchSelfAttention(hidden_dim, config.num_heads)
        else:
            self.self_attn = FLASelfMixer(config, layer_idx=layer_idx)
        if self.use_image_attention:
            self.image_attn_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
            self.image_attn = FullImageSelfAttention(
                hidden_dim,
                config.num_heads,
                backend=config.image_attention_backend,
                grid_size=config.latent_size,
                rope=config.image_attention_rope,
                rope_theta=config.image_rope_theta,
            )
            self.image_attn_scale_shift_table = nn.Parameter(torch.zeros(3, hidden_dim))
        cross_backend = config.cross_attention_backend
        if config.cross_attention_qk_norm and cross_backend == "mha":
            raise ValueError(
                "cross_attention_qk_norm requires SanaMultiHeadCrossAttention "
                "(cross_attention_backend sdpa/xformers/auto), not mha"
            )
        if cross_backend == "mha":
            self.cross_attn = nn.MultiheadAttention(hidden_dim, config.num_heads, batch_first=True)
        else:
            self.cross_attn = SanaMultiHeadCrossAttention(
                hidden_dim,
                config.num_heads,
                backend=cross_backend,
                qk_norm=config.cross_attention_qk_norm,
            )
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.mlp = GLUMBConv(hidden_dim, config.mlp_ratio)
        self.scale_shift_table = nn.Parameter(torch.zeros(6, hidden_dim))

    def forward(
        self,
        x: torch.Tensor,
        text_tokens: torch.Tensor,
        timestep_mod: torch.Tensor,
        text_key_padding_mask: torch.Tensor,
        text_attn_bias: torch.Tensor | None,
        *,
        height: int,
        width: int,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + timestep_mod.reshape(x.shape[0], 6, -1)
        ).chunk(6, dim=1)
        x = x + gate_msa * self.self_attn(modulate(self.norm1(x), shift_msa, scale_msa))
        if self.use_image_attention:
            shift_img, scale_img, gate_img = (
                self.image_attn_scale_shift_table[None] + timestep_mod.reshape(x.shape[0], 6, -1)[:, :3]
            ).chunk(3, dim=1)
            x = x + gate_img * self.image_attn(modulate(self.image_attn_norm(x), shift_img, scale_img))
        if isinstance(self.cross_attn, nn.MultiheadAttention):
            x = x + self.cross_attn(
                x,
                text_tokens,
                text_tokens,
                key_padding_mask=text_key_padding_mask,
                need_weights=False,
            )[0]
        else:
            x = x + self.cross_attn(x, text_tokens, text_key_padding_mask, text_attn_bias)
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp), height=height, width=width)
        return x


class BoomerFLADiT(nn.Module):
    """Boomer DiT with FLA mixers, optional full image attention, and GLUMBConv FFNs."""

    def __init__(self, config: BoomerFLADiTConfig = BoomerFLADiTConfig()) -> None:
        super().__init__()
        if config.patch_size != 1:
            raise ValueError("BoomerFLADiT currently supports patch_size=1 for DC-AE f32c32 latents.")
        self.config = config
        hidden_dim = config.hidden_dim
        token_count = config.latent_size * config.latent_size
        self.x_embedder = nn.Linear(config.latent_channels, hidden_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, token_count, hidden_dim))
        self.t_embedder = TimestepEmbedder(hidden_dim)
        self.t_block = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 6))
        self.caption_embedder = CaptionEmbedder(config.text_dim, hidden_dim, config.text_seq_len)
        self.attention_y_norm = (
            AttentionRMSNorm(hidden_dim, scale_factor=config.y_norm_scale_factor) if config.y_norm else None
        )
        self.blocks = nn.ModuleList([BoomerFLABlock(config, layer_idx=i) for i in range(config.depth)])
        self.final_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.final_t_block = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 2))
        self.out_proj = nn.Linear(hidden_dim, config.latent_channels)
        self.initialize_weights()

    def initialize_weights(self) -> None:
        nn.init.normal_(self.pos_embed, std=0.02)

        nn.init.normal_(self.t_block[1].weight, std=0.02)
        nn.init.zeros_(self.t_block[1].bias)

        for block in self.blocks:
            nn.init.normal_(block.scale_shift_table, std=0.02)
            if block.use_image_attention:
                nn.init.normal_(block.image_attn_scale_shift_table, std=0.02)

        nn.init.zeros_(self.final_t_block[1].weight)
        nn.init.zeros_(self.final_t_block[1].bias)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

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
        token_num: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.caption_embedder.null_condition(
            batch_size,
            device=device,
            dtype=dtype,
            mask_dtype=mask_dtype,
            token_num=token_num,
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
        null_text, null_mask = self.null_condition(
            batch_size,
            device=text_embedding.device,
            dtype=text_embedding.dtype,
            mask_dtype=attention_mask.dtype,
            token_num=text_embedding.shape[-2],
        )
        # torch.where over a per-sample bool. Avoids the bool(drop.any()) CUDA
        # sync (which would defeat the training-loop sync removal) and skips
        # the full-tensor .clone() that the previous in-place path required.
        drop = torch.rand(batch_size, device=text_embedding.device) < probability
        drop_text = drop.view(batch_size, *([1] * (text_embedding.dim() - 1)))
        drop_mask = drop.view(batch_size, *([1] * (attention_mask.dim() - 1)))
        text_embedding = torch.where(drop_text, null_text, text_embedding)
        attention_mask = torch.where(drop_mask, null_mask, attention_mask)
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

        x = noisy_latent.flatten(2).transpose(1, 2)
        x = self.x_embedder(x) + self.pos_embed
        text_attn_bias = text_key_padding_mask[:, None, None, :].to(dtype=x.dtype)
        text_attn_bias = text_attn_bias.masked_fill(text_attn_bias > 0, -10000.0)
        t_emb = self.t_embedder(timesteps)
        timestep_mod = self.t_block(t_emb)
        for block in self.blocks:
            x = block(
                x,
                text_tokens,
                timestep_mod,
                text_key_padding_mask,
                text_attn_bias,
                height=height,
                width=width,
            )
        final_mod = self.final_t_block(t_emb)
        shift, scale = final_mod.reshape(batch, 2, -1).chunk(2, dim=1)
        x = modulate(self.final_norm(x), shift, scale)
        x = self.out_proj(x)
        return x.transpose(1, 2).reshape(batch, channels, height, width)
