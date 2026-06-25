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
from torch.utils.checkpoint import checkpoint as _ckpt

from boomer.models.latent_dit import AttentionRMSNorm, CaptionEmbedder, TimestepEmbedder


@dataclass(frozen=True)
class BoomerFLADiTConfig:
    model_type: str = "boomer_fla"
    latent_channels: int = 32
    # FLUX-Fill-style reserved input channels (e.g. 33 = masked latent[32] + mask[1]).
    # Zero-initialized and zero-fed when no conditioning is supplied, so the base model
    # is behaviorally identical; lets you fine-tune into inpainting without a from-scratch
    # retrain. The output projection still predicts only `latent_channels`.
    extra_input_channels: int = 0
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
    parallel_block: bool = False
    dual_stream_depth: int = 0
    multimodal_coord_ids: bool = False
    use_abs_pos_embed: bool = True
    patch_size: int = 1
    gradient_checkpointing: bool = False
    # REPA representation alignment (https://arxiv.org/abs/2410.06940). When repa_dim > 0
    # the DiT exposes a pooled projection of a mid-depth block's hidden state so training
    # can align it to an external SSL target (monet's DINOv2 ViT-G/14 CLS = 1536). 0 = off.
    repa_dim: int = 0
    repa_layer: int = -1  # block whose output to align; -1 => depth // 2
    # Dispersive loss (https://arxiv.org/abs/2506.09027): a parameter-free regularizer
    # that pushes a mid-depth block's pooled hidden state apart across the batch. Needs
    # no external target (unlike REPA), so it works on the existing cache. The loss weight
    # is a training hyperparameter; this only selects which block's features to expose.
    dispersive_layer: int = -1  # block whose hidden state to disperse; -1 => depth // 2


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
        Path("/work/flash-attention/hopper"),
        Path("/work/flash-attention"),
        Path("/home/jovyan/work/flash-attention"),
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
        nn.init.zeros_(self.point_conv.conv.weight)
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


class TokenMLP(nn.Module):
    def __init__(self, hidden_dim: int, mlp_ratio: float) -> None:
        super().__init__()
        inner_dim = int(hidden_dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, inner_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(inner_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultimodalCoordinateRoPE(nn.Module):
    """FLUX-style coordinate-ID RoPE for joint text/image attention."""

    def __init__(self, head_dim: int, *, image_size: int, text_seq_len: int, theta: float = 10000.0) -> None:
        super().__init__()
        if head_dim < 6 or head_dim % 2 != 0:
            raise ValueError(f"head_dim={head_dim} must be even and at least 6 for multimodal RoPE")
        if theta <= 0.0:
            raise ValueError(f"theta must be positive, got {theta}")
        type_dim = max(2, (head_dim // 4) // 2 * 2)
        while type_dim > 2 and (head_dim - type_dim) % 4 != 0:
            type_dim -= 2
        remaining = head_dim - type_dim
        row_dim = max(2, (remaining // 2) // 2 * 2)
        col_dim = remaining - row_dim
        if col_dim < 2 or col_dim % 2 != 0:
            raise ValueError(f"could not split head_dim={head_dim} into even multimodal RoPE axes")
        self.axes_dim = (type_dim, row_dim, col_dim)
        self.head_dim = head_dim
        self.image_size = image_size
        self.text_seq_len = text_seq_len

        for index, dim in enumerate(self.axes_dim):
            inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
            self.register_buffer(f"inv_freq_{index}", inv_freq, persistent=False)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def image_ids(self, batch_size: int, *, height: int, width: int, device: torch.device | str) -> torch.Tensor:
        token_idx = torch.arange(height * width, device=device)
        rows = token_idx // width
        cols = token_idx % width
        token_type = torch.ones_like(rows)
        ids = torch.stack([token_type, rows, cols], dim=-1)
        return ids.unsqueeze(0).expand(batch_size, -1, -1)

    def text_ids(self, batch_size: int, token_count: int, *, device: torch.device | str) -> torch.Tensor:
        positions = torch.arange(token_count, device=device)
        token_type = torch.zeros_like(positions)
        zeros = torch.zeros_like(positions)
        ids = torch.stack([token_type, positions, zeros], dim=-1)
        return ids.unsqueeze(0).expand(batch_size, -1, -1)

    def _axis_apply(self, x: torch.Tensor, axis_ids: torch.Tensor, axis_index: int) -> torch.Tensor:
        inv_freq = getattr(self, f"inv_freq_{axis_index}")
        angles = axis_ids.float().unsqueeze(-1) * inv_freq.to(device=x.device).view(1, 1, -1)
        cos = torch.cat([angles.cos(), angles.cos()], dim=-1).unsqueeze(2).to(dtype=x.dtype)
        sin = torch.cat([angles.sin(), angles.sin()], dim=-1).unsqueeze(2).to(dtype=x.dtype)
        return x * cos + self._rotate_half(x) * sin

    def apply(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if q.shape[-1] != self.head_dim or k.shape[-1] != self.head_dim:
            raise ValueError(f"expected head_dim={self.head_dim}, got q={q.shape[-1]} k={k.shape[-1]}")
        if ids.shape[:2] != q.shape[:2] or ids.shape[-1] != len(self.axes_dim):
            raise ValueError(f"expected ids shape (B, T, {len(self.axes_dim)}), got {tuple(ids.shape)}")
        q_chunks = q.split(self.axes_dim, dim=-1)
        k_chunks = k.split(self.axes_dim, dim=-1)
        q_out = []
        k_out = []
        for index, (q_axis, k_axis) in enumerate(zip(q_chunks, k_chunks, strict=True)):
            q_out.append(self._axis_apply(q_axis, ids[..., index], index))
            k_out.append(self._axis_apply(k_axis, ids[..., index], index))
        return torch.cat(q_out, dim=-1), torch.cat(k_out, dim=-1)


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
        if backend not in {"sdpa", "flash2", "flash3", "flash4", "auto"}:
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
        self._flash2_attn_func = None
        self._flash2_import_attempted = False
        self._flash3_attn_func = None
        self._flash3_import_attempted = False
        self._flash4_attn_func = None
        self._flash4_import_attempted = False

    def _get_flash2_attn_func(self):
        if self._flash2_import_attempted:
            return self._flash2_attn_func
        self._flash2_import_attempted = True
        try:  # FlashAttention-2 public API.
            from flash_attn import flash_attn_func
        except Exception:
            flash_attn_func = None
        self._flash2_attn_func = flash_attn_func
        return self._flash2_attn_func

    def _flash2_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        flash_attn_func = self._get_flash2_attn_func()
        if flash_attn_func is None:
            raise ImportError(
                "image_attention_backend='flash2' requires FlashAttention-2 (the `flash_attn` "
                "package). Install it or use --image-attn-backend sdpa."
            )
        out = flash_attn_func(q, k, v, causal=False)
        if isinstance(out, tuple):
            out = out[0]
        return out

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
                Path("/work/flash-attention/flash_attn"),
                Path("/home/jovyan/work/flash-attention/flash_attn"),
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

    @staticmethod
    def _flash_compute_dtype(x: torch.Tensor) -> torch.dtype | None:
        """FA kernels need fp16/bf16; fp32 master weights + compile may still pass fp32 activations."""
        if not x.is_cuda:
            return None
        if x.dtype in {torch.float16, torch.bfloat16}:
            return x.dtype
        if torch.is_autocast_enabled():
            return torch.get_autocast_dtype("cuda")
        return torch.bfloat16

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        if self.rope is not None:
            q, k = self.rope(q, k)

        flash_dtype = self._flash_compute_dtype(x)
        use_flash = self.backend in {"flash2", "flash3", "flash4", "auto"} and flash_dtype is not None
        if use_flash and (q.dtype != flash_dtype or k.dtype != flash_dtype or v.dtype != flash_dtype):
            q, k, v = q.to(flash_dtype), k.to(flash_dtype), v.to(flash_dtype)

        if self.backend == "flash4" and use_flash:
            out = self._flash4_attention(q, k, v)
        elif self.backend == "flash3" and use_flash:
            out = self._flash3_attention(q, k, v)
        elif self.backend == "flash2" and use_flash:
            out = self._flash2_attention(q, k, v)
        elif self.backend == "auto" and use_flash:
            try:
                out = self._flash4_attention(q, k, v)
            except Exception:
                try:
                    out = self._flash3_attention(q, k, v)
                except Exception:
                    try:
                        out = self._flash2_attention(q, k, v)
                    except Exception:
                        use_flash = False
        if self.backend in {"flash2", "flash3", "flash4"} and not use_flash:
            raise RuntimeError(
                f"image_attention_backend='{self.backend}' requires CUDA fp16/bf16 compute; got {x.device} {x.dtype}"
            )
        if use_flash and out.dtype != x.dtype:
            out = out.to(dtype=x.dtype)
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

        def make_mixer() -> nn.Module:
            if config.mixer_type == "fla_linear":
                return fla_layers.LinearAttention(
                    hidden_size=hidden_dim,
                    num_heads=config.num_heads,
                    mode=config.fla_mode,
                    feature_map=config.fla_feature_map,
                    output_norm="rmsnorm",
                    layer_idx=layer_idx,
                )
            if config.mixer_type == "fla_gated_deltanet":
                return fla_layers.GatedDeltaNet(
                    hidden_size=hidden_dim,
                    num_heads=config.num_heads,
                    head_dim=hidden_dim // config.num_heads,
                    expand_v=1,
                    mode=config.fla_mode,
                    use_short_conv=config.use_short_conv,
                    conv_size=config.conv_size,
                    layer_idx=layer_idx,
                )
            if config.mixer_type == "fla_gdn2":
                # GatedDeltaNet-2: channel-wise erase (b, key axis) + write (w, value axis)
                # gates + channel-wise log-decay (extends KDA). Causal-only; bidirectionality is
                # handled by FLASelfMixer's fwd/flip-bwd wrapper, same as GatedDeltaNet. Heavier
                # than GDN (extra b/w/decay projections); chunk kernel needs fp32 initial_state,
                # so smoke-test under torch.compile + --fla-bidirectional before a long run.
                # Boomer never supplies an autoregressive state cache. Leaving layer_idx unset
                # avoids Dynamo specializing this shared forward on every block's integer index
                # (24 redundant graphs followed by recompile-limit eager fallback).
                return fla_layers.GatedDeltaNet2(
                    hidden_size=hidden_dim,
                    num_heads=config.num_heads,
                    head_dim=hidden_dim // config.num_heads,
                    expand_v=1.0,
                    use_short_conv=config.use_short_conv,
                    conv_size=config.conv_size,
                    layer_idx=None,
                )
            if config.mixer_type == "fla_gla":
                return fla_layers.GatedLinearAttention(
                    hidden_size=hidden_dim,
                    num_heads=config.num_heads,
                    mode=config.fla_mode,
                    feature_map=config.fla_feature_map,
                    use_short_conv=config.use_short_conv,
                    conv_size=config.conv_size,
                    layer_idx=layer_idx,
                )
            raise ValueError(f"Unsupported FLA mixer_type: {config.mixer_type}")

        self.mixer_fwd = make_mixer()
        self.mixer_bwd = make_mixer() if self.bidirectional else None
        if self.bidirectional:
            self.out_proj = nn.Linear(hidden_dim * 2, hidden_dim, bias=False)
            nn.init.zeros_(self.out_proj.weight)

    @staticmethod
    def _run_mixer(mixer: nn.Module, x: torch.Tensor) -> torch.Tensor:
        y = mixer(x)
        if isinstance(y, tuple):
            y = y[0]
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self._run_mixer(self.mixer_fwd, x)
        if not self.bidirectional:
            return y
        if self.mixer_bwd is None:
            raise RuntimeError("bidirectional FLASelfMixer is missing the backward mixer")
        y_rev = self._run_mixer(self.mixer_bwd, x.flip(1)).flip(1)
        return self.out_proj(torch.cat([y, y_rev], dim=-1))


class BoomerFLABlock(nn.Module):
    def __init__(self, config: BoomerFLADiTConfig, *, layer_idx: int) -> None:
        super().__init__()
        hidden_dim = config.hidden_dim
        self.parallel_block = config.parallel_block
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
            self.image_attn_mod = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 3))
            self.image_attn = FullImageSelfAttention(
                hidden_dim,
                config.num_heads,
                backend=config.image_attention_backend,
                grid_size=config.latent_size // config.patch_size,
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
        self.mod = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 9))
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.norm3 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.mlp = GLUMBConv(hidden_dim, config.mlp_ratio)
        self.scale_shift_table = nn.Parameter(torch.zeros(9, hidden_dim))

    def _cross_attention(
        self,
        x: torch.Tensor,
        text_tokens: torch.Tensor,
        text_key_padding_mask: torch.Tensor,
        text_attn_bias: torch.Tensor | None,
    ) -> torch.Tensor:
        if isinstance(self.cross_attn, nn.MultiheadAttention):
            return self.cross_attn(
                x,
                text_tokens,
                text_tokens,
                key_padding_mask=text_key_padding_mask,
                need_weights=False,
            )[0]
        return self.cross_attn(x, text_tokens, text_key_padding_mask, text_attn_bias)

    def forward(
        self,
        x: torch.Tensor,
        text_tokens: torch.Tensor,
        t_emb: torch.Tensor,
        text_key_padding_mask: torch.Tensor,
        text_attn_bias: torch.Tensor | None,
        *,
        height: int,
        width: int,
    ) -> torch.Tensor:
        timestep_mod = self.mod(t_emb)
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_cross,
            scale_cross,
            gate_cross,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = (self.scale_shift_table[None] + timestep_mod.reshape(x.shape[0], 9, -1)).chunk(9, dim=1)
        if self.parallel_block:
            base = x
            branches = [
                gate_msa * self.self_attn(modulate(self.norm1(base), shift_msa, scale_msa)),
                gate_cross
                * self._cross_attention(
                    modulate(self.norm3(base), shift_cross, scale_cross),
                    text_tokens,
                    text_key_padding_mask,
                    text_attn_bias,
                ),
                gate_mlp * self.mlp(modulate(self.norm2(base), shift_mlp, scale_mlp), height=height, width=width),
            ]
            if self.use_image_attention:
                image_attn_mod = self.image_attn_mod(t_emb)
                shift_img, scale_img, gate_img = (
                    self.image_attn_scale_shift_table[None] + image_attn_mod.reshape(x.shape[0], 3, -1)
                ).chunk(3, dim=1)
                branches.append(
                    gate_img * self.image_attn(modulate(self.image_attn_norm(base), shift_img, scale_img))
                )
            return base + sum(branches)

        x = x + gate_msa * self.self_attn(modulate(self.norm1(x), shift_msa, scale_msa))
        if self.use_image_attention:
            image_attn_mod = self.image_attn_mod(t_emb)
            shift_img, scale_img, gate_img = (
                self.image_attn_scale_shift_table[None] + image_attn_mod.reshape(x.shape[0], 3, -1)
            ).chunk(3, dim=1)
            x = x + gate_img * self.image_attn(modulate(self.image_attn_norm(x), shift_img, scale_img))
        x = x + gate_cross * self._cross_attention(
            modulate(self.norm3(x), shift_cross, scale_cross),
            text_tokens,
            text_key_padding_mask,
            text_attn_bias,
        )
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp), height=height, width=width)
        return x


class BoomerFLADualStreamBlock(nn.Module):
    """FLUX-style early block with one joint text+image attention operation."""

    updates_text = True

    def __init__(self, config: BoomerFLADiTConfig, *, layer_idx: int) -> None:
        super().__init__()
        hidden_dim = config.hidden_dim
        if hidden_dim % config.num_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by num_heads={config.num_heads}")
        self.num_heads = config.num_heads
        self.head_dim = hidden_dim // config.num_heads
        self.hidden_dim = hidden_dim
        self.parallel_block = config.parallel_block

        self.image_mod = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 6))
        self.image_norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.image_qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.image_q_norm = AttentionRMSNorm(self.head_dim, scale_factor=1.0, eps=1e-6)
        self.image_k_norm = AttentionRMSNorm(self.head_dim, scale_factor=1.0, eps=1e-6)
        self.image_out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.image_norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.image_mlp = GLUMBConv(hidden_dim, config.mlp_ratio)
        self.image_scale_shift_table = nn.Parameter(torch.zeros(6, hidden_dim))

        self.text_mod = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 6))
        self.text_norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.text_qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.text_q_norm = AttentionRMSNorm(self.head_dim, scale_factor=1.0, eps=1e-6)
        self.text_k_norm = AttentionRMSNorm(self.head_dim, scale_factor=1.0, eps=1e-6)
        self.text_out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.text_norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.text_mlp = TokenMLP(hidden_dim, config.mlp_ratio)
        self.text_scale_shift_table = nn.Parameter(torch.zeros(6, hidden_dim))

    def _qkv(
        self,
        x: torch.Tensor,
        qkv: nn.Linear,
        q_norm: AttentionRMSNorm,
        k_norm: AttentionRMSNorm,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, tokens, _ = x.shape
        q, k, v = qkv(x).reshape(batch, tokens, 3, self.num_heads, self.head_dim).unbind(dim=2)
        q = q_norm(q)
        k = k_norm(k)
        return q, k, v

    def _joint_attention(
        self,
        image_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        text_key_padding_mask: torch.Tensor,
        coord_rope: MultimodalCoordinateRoPE | None,
        image_coord_ids: torch.Tensor | None,
        text_coord_ids: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image_q, image_k, image_v = self._qkv(
            image_tokens,
            self.image_qkv,
            self.image_q_norm,
            self.image_k_norm,
        )
        text_q, text_k, text_v = self._qkv(
            text_tokens,
            self.text_qkv,
            self.text_q_norm,
            self.text_k_norm,
        )
        q = torch.cat([text_q, image_q], dim=1)
        k = torch.cat([text_k, image_k], dim=1)
        v = torch.cat([text_v, image_v], dim=1)
        if coord_rope is not None:
            if image_coord_ids is None or text_coord_ids is None:
                raise ValueError("coordinate ids are required when multimodal coord RoPE is enabled")
            coord_ids = torch.cat([text_coord_ids, image_coord_ids], dim=1)
            q, k = coord_rope.apply(q, k, coord_ids)

        image_mask = torch.zeros(
            image_tokens.shape[0],
            image_tokens.shape[1],
            device=image_tokens.device,
            dtype=text_key_padding_mask.dtype,
        )
        key_padding_mask = torch.cat([text_key_padding_mask, image_mask], dim=1)
        attn_bias = key_padding_mask[:, None, None, :].to(dtype=q.dtype)
        attn_bias = attn_bias.masked_fill(attn_bias > 0, -10000.0)
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            attn_mask=attn_bias,
            dropout_p=0.0,
            is_causal=False,
        )
        out = out.transpose(1, 2).reshape(image_tokens.shape[0], text_tokens.shape[1] + image_tokens.shape[1], -1)
        text_out, image_out = out.split([text_tokens.shape[1], image_tokens.shape[1]], dim=1)
        return self.image_out_proj(image_out), self.text_out_proj(text_out)

    def forward(
        self,
        x: torch.Tensor,
        text_tokens: torch.Tensor,
        t_emb: torch.Tensor,
        text_key_padding_mask: torch.Tensor,
        text_attn_bias: torch.Tensor | None,
        *,
        height: int,
        width: int,
        coord_rope: MultimodalCoordinateRoPE | None = None,
        image_coord_ids: torch.Tensor | None = None,
        text_coord_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del text_attn_bias
        image_timestep_mod = self.image_mod(t_emb)
        text_timestep_mod = self.text_mod(t_emb)
        image_shift_attn, image_scale_attn, image_gate_attn, image_shift_mlp, image_scale_mlp, image_gate_mlp = (
            self.image_scale_shift_table[None] + image_timestep_mod.reshape(x.shape[0], 6, -1)
        ).chunk(6, dim=1)
        text_shift_attn, text_scale_attn, text_gate_attn, text_shift_mlp, text_scale_mlp, text_gate_mlp = (
            self.text_scale_shift_table[None] + text_timestep_mod.reshape(text_tokens.shape[0], 6, -1)
        ).chunk(6, dim=1)

        image_base = x
        text_base = text_tokens
        image_attn_in = modulate(self.image_norm1(image_base), image_shift_attn, image_scale_attn)
        text_attn_in = modulate(self.text_norm1(text_base), text_shift_attn, text_scale_attn)
        image_attn, text_attn = self._joint_attention(
            image_attn_in,
            text_attn_in,
            text_key_padding_mask,
            coord_rope,
            image_coord_ids,
            text_coord_ids,
        )
        if self.parallel_block:
            x = image_base + image_gate_attn * image_attn + image_gate_mlp * self.image_mlp(
                modulate(self.image_norm2(image_base), image_shift_mlp, image_scale_mlp),
                height=height,
                width=width,
            )
            text_tokens = text_base + text_gate_attn * text_attn + text_gate_mlp * self.text_mlp(
                modulate(self.text_norm2(text_base), text_shift_mlp, text_scale_mlp)
            )
            return x, text_tokens

        x = image_base + image_gate_attn * image_attn
        text_tokens = text_base + text_gate_attn * text_attn
        x = x + image_gate_mlp * self.image_mlp(
            modulate(self.image_norm2(x), image_shift_mlp, image_scale_mlp),
            height=height,
            width=width,
        )
        text_tokens = text_tokens + text_gate_mlp * self.text_mlp(
            modulate(self.text_norm2(text_tokens), text_shift_mlp, text_scale_mlp)
        )
        return x, text_tokens


class BoomerFLADiT(nn.Module):
    """Boomer DiT with FLA mixers, optional full image attention, and GLUMBConv FFNs."""

    def __init__(self, config: BoomerFLADiTConfig = BoomerFLADiTConfig()) -> None:
        super().__init__()
        if config.patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {config.patch_size}")
        if config.latent_size % config.patch_size != 0:
            raise ValueError(
                f"latent_size={config.latent_size} must be divisible by patch_size={config.patch_size}"
            )
        if config.dual_stream_depth < 0:
            raise ValueError(f"dual_stream_depth must be non-negative, got {config.dual_stream_depth}")
        if config.dual_stream_depth > config.depth:
            raise ValueError(f"dual_stream_depth={config.dual_stream_depth} exceeds depth={config.depth}")
        self.config = config
        hidden_dim = config.hidden_dim
        self.patch_size = config.patch_size
        self.token_grid_size = config.latent_size // config.patch_size
        token_count = self.token_grid_size * self.token_grid_size
        in_channels = config.latent_channels + config.extra_input_channels
        self.x_embedder = (
            nn.Linear(in_channels, hidden_dim)
            if config.patch_size == 1
            else nn.Conv2d(
                in_channels,
                hidden_dim,
                kernel_size=config.patch_size,
                stride=config.patch_size,
            )
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, token_count, hidden_dim)) if config.use_abs_pos_embed else None
        self.t_embedder = TimestepEmbedder(hidden_dim)
        self.caption_embedder = CaptionEmbedder(config.text_dim, hidden_dim, config.text_seq_len)
        self.attention_y_norm = (
            AttentionRMSNorm(hidden_dim, scale_factor=config.y_norm_scale_factor) if config.y_norm else None
        )
        self.coord_embedder = (
            MultimodalCoordinateRoPE(
                hidden_dim // config.num_heads,
                image_size=self.token_grid_size,
                text_seq_len=config.text_seq_len,
                theta=config.image_rope_theta,
            )
            if config.multimodal_coord_ids
            else None
        )
        self.blocks = nn.ModuleList(
            [
                (
                    BoomerFLADualStreamBlock(config, layer_idx=i)
                    if i < config.dual_stream_depth
                    else BoomerFLABlock(config, layer_idx=i)
                )
                for i in range(config.depth)
            ]
        )
        self.final_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.final_t_block = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 2))
        self.out_proj = nn.Linear(hidden_dim, config.latent_channels * config.patch_size * config.patch_size)
        # REPA: pooled mid-depth feature -> SSL embedding space. 3-layer MLP head per the
        # REPA paper; only built (and only run) when repa_dim > 0 and return_repa is set.
        self.repa_layer = config.repa_layer if config.repa_layer >= 0 else config.depth // 2
        # Dispersive loss: no projector (parameter-free); only the capture layer is needed.
        self.dispersive_layer = (
            config.dispersive_layer if config.dispersive_layer >= 0 else config.depth // 2
        )
        self.repa_proj = (
            nn.Sequential(
                nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, config.repa_dim),
            )
            if config.repa_dim > 0
            else None
        )
        self.initialize_weights()

    def initialize_weights(self) -> None:
        if self.pos_embed is not None:
            nn.init.normal_(self.pos_embed, std=0.02)

        for block in self.blocks:
            if isinstance(block, BoomerFLADualStreamBlock):
                nn.init.zeros_(block.image_mod[1].weight)
                nn.init.zeros_(block.image_mod[1].bias)
                nn.init.zeros_(block.text_mod[1].weight)
                nn.init.zeros_(block.text_mod[1].bias)
                nn.init.normal_(block.image_scale_shift_table, std=0.02)
                nn.init.normal_(block.text_scale_shift_table, std=0.02)
            else:
                nn.init.zeros_(block.mod[1].weight)
                nn.init.zeros_(block.mod[1].bias)
                nn.init.normal_(block.scale_shift_table, std=0.02)
                if block.use_image_attention:
                    nn.init.zeros_(block.image_attn_mod[1].weight)
                    nn.init.zeros_(block.image_attn_mod[1].bias)
                    nn.init.normal_(block.image_attn_scale_shift_table, std=0.02)

        nn.init.zeros_(self.final_t_block[1].weight)
        nn.init.zeros_(self.final_t_block[1].bias)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        # Zero the reserved-input-channel slice so they contribute nothing until a future
        # inpainting fine-tune learns them (base channels keep their default init).
        if self.config.extra_input_channels > 0:
            with torch.no_grad():
                self.x_embedder.weight[:, self.config.latent_channels :].zero_()

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
        cond_input: torch.Tensor | None = None,
        return_repa: bool = False,
        return_dispersive: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        batch, channels, height, width = noisy_latent.shape
        if channels != self.config.latent_channels:
            raise ValueError(
                f"Expected latent_channels={self.config.latent_channels}, got shape {tuple(noisy_latent.shape)}"
            )
        if height % self.patch_size != 0 or width % self.patch_size != 0:
            raise ValueError(
                f"latent height/width must be divisible by patch_size={self.patch_size}, got {(height, width)}"
            )
        token_height = height // self.patch_size
        token_width = width // self.patch_size
        token_count = token_height * token_width
        if self.pos_embed is not None and token_count != self.pos_embed.shape[1]:
            raise ValueError(
                f"absolute pos_embed expects {self.pos_embed.shape[1]} latent tokens, got {token_count}. "
                "Disable it with --no-abs-pos-embed for variable latent sizes."
            )
        if text_embedding.shape[-1] != self.config.text_dim:
            raise ValueError(f"text_embedding last dim must be {self.config.text_dim}, got {text_embedding.shape[-1]}")

        text_tokens = self.caption_embedder(text_embedding)
        text_tokens = self.apply_y_norm(text_tokens)
        text_key_padding_mask = attention_mask == 0

        # FLUX-Fill-style reserved channels: concat conditioning (or zeros) onto the
        # noisy latent before embedding. Zeros => behaviorally identical to no conditioning.
        if self.config.extra_input_channels > 0:
            if cond_input is None:
                cond_input = noisy_latent.new_zeros(
                    batch, self.config.extra_input_channels, height, width
                )
            elif cond_input.shape[1] != self.config.extra_input_channels:
                raise ValueError(
                    f"cond_input must have {self.config.extra_input_channels} channels, "
                    f"got {tuple(cond_input.shape)}"
                )
            model_input = torch.cat([noisy_latent, cond_input], dim=1)
        elif cond_input is not None:
            raise ValueError("cond_input given but extra_input_channels=0")
        else:
            model_input = noisy_latent

        if self.patch_size == 1:
            x = model_input.flatten(2).transpose(1, 2)
            x = self.x_embedder(x)
        else:
            x = self.x_embedder(model_input).flatten(2).transpose(1, 2)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        image_coord_ids = None
        text_coord_ids = None
        if self.coord_embedder is not None:
            image_coord_ids = self.coord_embedder.image_ids(
                batch,
                height=token_height,
                width=token_width,
                device=x.device,
            )
            text_coord_ids = self.coord_embedder.text_ids(batch, text_tokens.shape[1], device=text_tokens.device)
        text_attn_bias = text_key_padding_mask[:, None, None, :].to(dtype=x.dtype)
        text_attn_bias = text_attn_bias.masked_fill(text_attn_bias > 0, -10000.0)
        t_emb = self.t_embedder(timesteps)
        use_ckpt = self.config.gradient_checkpointing and self.training
        want_repa = return_repa and self.repa_proj is not None
        want_dispersive = return_dispersive
        repa_feat: torch.Tensor | None = None
        dispersive_feat: torch.Tensor | None = None
        for layer_idx, block in enumerate(self.blocks):
            if getattr(block, "updates_text", False):
                # Dual-stream block: returns (x, text_tokens).
                # Non-tensor args (height, width, coord_rope, coord IDs) captured via closure.
                _h, _w = token_height, token_width
                _cr, _ii, _ti = self.coord_embedder, image_coord_ids, text_coord_ids
                if use_ckpt:
                    def _dual_fn(x, tt, te, mk, bi,
                                 _blk=block, h=_h, w=_w, cr=_cr, ii=_ii, ti=_ti):
                        return _blk(x, tt, te, mk, bi,
                                    height=h, width=w, coord_rope=cr,
                                    image_coord_ids=ii, text_coord_ids=ti)
                    x, text_tokens = _ckpt(_dual_fn, x, text_tokens, t_emb,
                                           text_key_padding_mask, text_attn_bias,
                                           use_reentrant=False,
                                           preserve_rng_state=False)
                else:
                    x, text_tokens = block(
                        x, text_tokens, t_emb, text_key_padding_mask, text_attn_bias,
                        height=token_height, width=token_width,
                        coord_rope=self.coord_embedder,
                        image_coord_ids=image_coord_ids, text_coord_ids=text_coord_ids,
                    )
            else:
                # Single-stream block: returns x only.
                _h, _w = token_height, token_width
                if use_ckpt:
                    def _single_fn(x, tt, te, mk, bi,
                                   _blk=block, h=_h, w=_w):
                        return _blk(x, tt, te, mk, bi, height=h, width=w)
                    x = _ckpt(_single_fn, x, text_tokens, t_emb,
                               text_key_padding_mask, text_attn_bias,
                               use_reentrant=False,
                               preserve_rng_state=False)
                else:
                    x = block(
                        x, text_tokens, t_emb, text_key_padding_mask, text_attn_bias,
                        height=token_height, width=token_width,
                    )
            if want_repa and layer_idx == self.repa_layer:
                # Pool image tokens -> project to the SSL space for alignment.
                repa_feat = self.repa_proj(x.mean(dim=1))
            if want_dispersive and layer_idx == self.dispersive_layer:
                # Pool image tokens -> per-sample feature; dispersion is computed in the
                # loss. No projector: the regularizer acts on the raw hidden state.
                dispersive_feat = x.mean(dim=1)
        final_mod = self.final_t_block(t_emb)
        shift, scale = final_mod.reshape(batch, 2, -1).chunk(2, dim=1)
        x = modulate(self.final_norm(x), shift, scale)
        x = self.out_proj(x)
        if self.patch_size == 1:
            out = x.transpose(1, 2).reshape(batch, channels, height, width)
        else:
            patch = self.patch_size
            x = x.reshape(batch, token_height, token_width, channels, patch, patch)
            x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
            out = x.reshape(batch, channels, height, width)
        # Return contract (kept backward-compatible for existing callers):
        #   return_repa only            -> (out, repa_feat)
        #   return_dispersive only      -> (out, dispersive_feat)
        #   both                        -> (out, repa_feat, dispersive_feat)
        #   neither                     -> out
        if return_repa and return_dispersive:
            return out, repa_feat, dispersive_feat
        if return_repa:
            return out, repa_feat
        if return_dispersive:
            return out, dispersive_feat
        return out
