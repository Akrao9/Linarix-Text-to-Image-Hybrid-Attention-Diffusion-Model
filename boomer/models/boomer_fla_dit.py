"""Boomer FLA DiT denoiser."""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _ckpt

from boomer.models.latent_dit import (
    AttentionRMSNorm,
    CaptionEmbedder,
    FourierTimestepEmbedder,
    TimestepEmbedder,
)


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
    image_attention_qk_rmsnorm: bool = True
    image_attention_rope: bool = False
    image_rope_theta: float = 10000.0
    cross_attention_backend: str = "sdpa"
    # Kept as "qk_norm" for backward compatibility with older checkpoints/configs.
    # The implementation is Q/K RMSNorm, matching Sana-style attention stabilization.
    cross_attention_qk_norm: bool = True
    # Cross-attend to text every Nth single-stream block (1 = every block, the historical
    # behavior). PixArt-Sigma-style thinning: text tokens are static after the dual-stream
    # prefix, so per-block cross-attention is partly redundant. Blocks without cross-attn
    # keep the 9-way modulation layout; their cross rows are simply unused.
    cross_attention_every: int = 1
    # PixArt-alpha adaLN-single: one shared SiLU+Linear timestep projection for all blocks
    # instead of a ~7M-param projection per block; the per-block scale_shift_table (which
    # already exists in both layouts) provides the layer-to-layer variation. Saves ~27% of
    # parameters at the deployed size with published evidence of parity.
    shared_adaln: bool = False
    # DiT-style sinusoidal timestep features (256 freqs, then MLP) instead of feeding the
    # raw sigma scalar through Linear(1, hidden). Multi-scale resolution over t matters
    # most near the endpoints where flow dynamics change fastest.
    fourier_timestep_embed: bool = False
    # Feed the FLA mixers tokens in Hilbert-curve order (undone right after the mixer) so
    # sequence adjacency tracks 2D adjacency; raster order puts vertical neighbors a full
    # row apart, which starves the recurrent mixers' implicit positional signal. Full/cross
    # attention and GLUMBConv still see raster order. Needs a square power-of-two grid.
    hilbert_mixer_order: bool = False
    parallel_block: bool = False
    dual_stream_depth: int = 0
    multimodal_coord_ids: bool = False
    # v6: every image_attention_every-th post-prefix block becomes a joint refresh block —
    # the image-only full attention and the separate text cross-attention are replaced by
    # one MMDiT-style attention over [text; image] whose text half is written back through
    # a text FFN, so text re-negotiates against the evolving image state (SD3-style)
    # instead of staying a static memory bank after the dual-stream prefix. A candidate
    # slot whose text update no later block would read (typically the final block) keeps
    # the classic image-attention block instead of carrying dead text-update weights.
    joint_refresh_attention: bool = False
    # v6: Lumina-Next sandwich norm — RMS-normalize each sublayer's output BEFORE the
    # residual add (x = x + gate * norm(sublayer(pre_norm(x)))). The residual stream is
    # never normalized after addition.
    sandwich_rmsnorm: bool = False
    # v6: NTK-aware RoPE rescaling for the image-spatial axes only (never text positions).
    # 1.0 for 512px pretrain; ~2.0 for the 1024px fine-tune where the token grid doubles.
    rope_ntk_factor: float = 1.0
    # v7: DDT-style encoder/decoder split (arXiv 2504.05741). The last N blocks become a
    # velocity decoder: full softmax attention + GLUMBConv, conditioned TOKEN-WISE on the
    # condition encoder's output z (the "self-condition") via adaLN, with the noisy latent
    # re-embedded through a separate decoder embedder. The first depth-N blocks (dual-stream
    # prefix + GDN2 spine + joint refresh) become the condition encoder. DDT-XL/2 at this
    # exact shape (hidden 1152 / depth 28) used 22en6de. 0 = off (v6 layout, identical
    # parameters and state-dict keys).
    ddt_decoder_depth: int = 0
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


def ntk_scaled_theta(theta: float, ntk_factor: float, rotary_dim: int) -> float:
    """NTK-aware RoPE base rescale: theta' = theta * s^(d/(d-2)) for scale factor s.

    Preserves high-frequency (local-offset) behavior learned at the pretrain grid while
    stretching the low-frequency axes to cover the larger fine-tune grid.
    """
    if ntk_factor == 1.0:
        return theta
    if ntk_factor <= 0.0:
        raise ValueError(f"rope_ntk_factor must be positive, got {ntk_factor}")
    if rotary_dim <= 2:
        return theta * ntk_factor
    return theta * ntk_factor ** (rotary_dim / (rotary_dim - 2))


def make_sandwich_norm(config: BoomerFLADiTConfig) -> nn.Module:
    """Sandwich RMSNorm for one sublayer output; Identity when disabled.

    Applied INSIDE the adaLN gate (x = x + gate * sandwich(sublayer_out)): RMSNorm is
    scale-invariant, so norming after the gate would cancel the near-zero gate magnitude
    and defeat the adaLN-Zero init.
    """
    if not config.sandwich_rmsnorm:
        return nn.Identity()
    return AttentionRMSNorm(config.hidden_dim, scale_factor=1.0, eps=1e-6)


def _hilbert_d2xy(n: int, d: int) -> tuple[int, int]:
    """Distance along an n x n Hilbert curve -> (x, y) grid coordinate."""
    x = y = 0
    t = d
    s = 1
    while s < n:
        rx = 1 & (t // 2)
        ry = 1 & (t ^ rx)
        if ry == 0:
            if rx == 1:
                x = s - 1 - x
                y = s - 1 - y
            x, y = y, x
        x += s * rx
        y += s * ry
        t //= 4
        s *= 2
    return x, y


def hilbert_permutation(height: int, width: int) -> torch.Tensor:
    """Map sequence position -> raster token index along a Hilbert curve.

    `x[:, perm]` reorders raster-flattened tokens into Hilbert order, where every
    sequential step is a 2D-adjacent grid move (raster order jumps a full row at
    each wrap, so the recurrent mixers' recency signal mistracks vertical
    adjacency). Invert with `argsort(perm)`.
    """
    if height != width:
        raise ValueError(f"hilbert_mixer_order needs a square token grid, got {height}x{width}")
    n = height
    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError(f"hilbert_mixer_order needs a power-of-two grid, got {n}")
    perm = torch.empty(n * n, dtype=torch.long)
    for d in range(n * n):
        x, y = _hilbert_d2xy(n, d)
        perm[d] = y * n + x
    return perm


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


_GLUMB_LEGACY_KEY_MAP = (
    ("inverted_conv.conv.weight", "inverted_fc.0.weight"),
    ("inverted_conv.conv.bias", "inverted_fc.0.bias"),
    ("point_conv.conv.weight", "point_fc.weight"),
)


def migrate_glumbconv_state_dict(state_dict: dict, prefix: str = "") -> int:
    """Rename legacy Conv2d-1x1 GLUMBConv keys to the Linear layout, in place.

    Handles raw state dicts that never pass through ``Module.load_state_dict``
    (EMA shadows, export tooling, optimizer-adjacent dumps). Weights saved as
    ``(O, I, 1, 1)`` are reshaped to ``(O, I)``. Returns the number of migrated
    entries; 0 for post-refactor dicts (idempotent).
    """
    migrated = 0
    for key in list(state_dict.keys()):
        if not key.startswith(prefix):
            continue
        for old_suffix, new_suffix in _GLUMB_LEGACY_KEY_MAP:
            if not key.endswith(old_suffix):
                continue
            tensor = state_dict.pop(key)
            if old_suffix.endswith(".weight") and tensor.ndim == 4:
                tensor = tensor.reshape(tensor.shape[0], tensor.shape[1])
            new_key = key[: -len(old_suffix)] + new_suffix
            if new_key not in state_dict:
                state_dict[new_key] = tensor
            migrated += 1
            break
    return migrated


class GLUMBConv(nn.Module):
    """Sana GLUMBConv FFN: 1x1 expand, depthwise spatial conv, GLU, 1x1 project."""

    def __init__(self, hidden_dim: int, mlp_ratio: float) -> None:
        super().__init__()
        inner_dim = int(hidden_dim * mlp_ratio)
        self.inner_dim = inner_dim
        self.inner_channels = inner_dim * 2
        # 1x1 pointwise ops are Linear on tokens: equivalent to Conv2d(1x1) but avoids
        # NCHW reshape grads with non-standard strides that trip DDP bucket layout.
        self.inverted_fc = nn.Sequential(nn.Linear(hidden_dim, inner_dim * 2), nn.SiLU())
        self.depth_conv = ConvLayer(inner_dim * 2, inner_dim * 2, 3, groups=inner_dim * 2, bias=True)
        self.point_fc = nn.Linear(inner_dim, hidden_dim, bias=False)
        nn.init.zeros_(self.point_fc.weight)
        self.glu_act = nn.SiLU()

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        # Migrate checkpoints saved with the legacy Conv2d 1x1 layers.
        migrate_glumbconv_state_dict(state_dict, prefix=prefix)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x: torch.Tensor, *, height: int, width: int) -> torch.Tensor:
        batch, tokens, channels = x.shape
        if tokens != height * width:
            raise ValueError(f"Expected {height * width} image tokens, got {tokens}")
        x = self.inverted_fc(x)
        x = (
            x.reshape(batch, height, width, self.inner_channels)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        x = self.depth_conv(x)
        x, gate = x.chunk(2, dim=1)
        x = x * self.glu_act(gate)
        x = x.permute(0, 2, 3, 1).reshape(batch, tokens, self.inner_dim).contiguous()
        return self.point_fc(x)


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

    def __init__(
        self,
        head_dim: int,
        *,
        image_size: int,
        text_seq_len: int,
        theta: float = 10000.0,
        ntk_factor: float = 1.0,
    ) -> None:
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
        self.ntk_factor = ntk_factor

        for index, dim in enumerate(self.axes_dim):
            inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
            self.register_buffer(f"inv_freq_{index}", inv_freq, persistent=False)
            if index > 0:
                # NTK-rescaled tables for the row/col (image-spatial) axes. Text tokens
                # share these axes for their 1D positions, so the scaled tables are
                # selected per token by type id in _axis_apply — text keeps base freqs.
                ntk_theta = ntk_scaled_theta(theta, ntk_factor, dim)
                inv_freq_ntk = 1.0 / (ntk_theta ** (torch.arange(0, dim, 2).float() / dim))
                self.register_buffer(f"inv_freq_ntk_{index}", inv_freq_ntk, persistent=False)

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

    def _axis_cos_sin(
        self,
        axis_ids: torch.Tensor,
        axis_index: int,
        image_token_mask: torch.Tensor | None,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """cos/sin tables for one axis — independent of q/k, so built once and reused
        for both. NTK selection is on the small inv_freq vector (not two full angle
        tensors + a where), one pass per axis rather than per q and per k."""
        inv_freq = getattr(self, f"inv_freq_{axis_index}").to(device=device).view(1, 1, -1)
        if image_token_mask is not None and axis_index > 0:
            inv_freq_ntk = getattr(self, f"inv_freq_ntk_{axis_index}").to(device=device).view(1, 1, -1)
            inv_freq = torch.where(image_token_mask.unsqueeze(-1), inv_freq_ntk, inv_freq)
        angles = axis_ids.float().unsqueeze(-1) * inv_freq
        cos = torch.cat([angles.cos(), angles.cos()], dim=-1).unsqueeze(2).to(dtype=dtype)
        sin = torch.cat([angles.sin(), angles.sin()], dim=-1).unsqueeze(2).to(dtype=dtype)
        return cos, sin

    def _apply_cos_sin(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
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
        # token_type 1 = image (see image_ids/text_ids): only image tokens get NTK freqs.
        image_token_mask = (ids[..., 0] == 1) if self.ntk_factor != 1.0 else None
        q_chunks = q.split(self.axes_dim, dim=-1)
        k_chunks = k.split(self.axes_dim, dim=-1)
        q_out = []
        k_out = []
        for index, (q_axis, k_axis) in enumerate(zip(q_chunks, k_chunks, strict=True)):
            cos, sin = self._axis_cos_sin(
                ids[..., index], index, image_token_mask, dtype=q.dtype, device=q.device
            )
            q_out.append(self._apply_cos_sin(q_axis, cos, sin))
            k_out.append(self._apply_cos_sin(k_axis, cos, sin))
        return torch.cat(q_out, dim=-1), torch.cat(k_out, dim=-1)


class RoPE2D(nn.Module):
    """2D RoPE for image tokens on a fixed H×W grid (row-major flattening).

    Splits head_dim in half: the first half encodes height, the second width.
    Each half uses standard 1D RoPE with shared cos/sin tables per axis.
    """

    def __init__(
        self, head_dim: int, grid_size: int, *, theta: float = 10000.0, ntk_factor: float = 1.0
    ) -> None:
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

        # Both axes are image-spatial, so the NTK rescale applies to the whole table.
        theta = ntk_scaled_theta(theta, ntk_factor, self.half_dim)
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
        qk_rmsnorm: bool = True,
        grid_size: int | None = None,
        rope: bool = False,
        rope_theta: float = 10000.0,
        rope_ntk_factor: float = 1.0,
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
        self.q_norm = AttentionRMSNorm(self.head_dim, scale_factor=1.0, eps=1e-6) if qk_rmsnorm else nn.Identity()
        self.k_norm = AttentionRMSNorm(self.head_dim, scale_factor=1.0, eps=1e-6) if qk_rmsnorm else nn.Identity()
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        self.rope = (
            RoPE2D(self.head_dim, grid_size, theta=rope_theta, ntk_factor=rope_ntk_factor)
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
        # Sana-style Q/K RMSNorm caps attention-logit growth in full image attention.
        q = self.q_norm(q)
        k = self.k_norm(k)
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
        qk_rmsnorm: bool = True,
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
        self.q_norm = AttentionRMSNorm(hidden_dim, scale_factor=1.0, eps=1e-6) if qk_rmsnorm else nn.Identity()
        self.k_norm = AttentionRMSNorm(hidden_dim, scale_factor=1.0, eps=1e-6) if qk_rmsnorm else nn.Identity()
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
            # Bool SDPA mask (True = attend): dispatches to fused kernels more readily
            # than a dense additive float bias and skips the masked_fill materialization.
            attn_mask = (~key_padding_mask.bool())[:, None, None, :]
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
            # Identity for the forward mixer, zero for the reverse. The residual branch is
            # already gated by the near-zero adaLN gate; an all-zero fusion on top of that
            # double-gates the mixer at init (product of two ~0 factors) and starves early
            # training of the layers doing most of the token mixing.
            with torch.no_grad():
                self.out_proj.weight.zero_()
                self.out_proj.weight[:, :hidden_dim].copy_(torch.eye(hidden_dim))

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
            self.image_attn_mod = (
                None if config.shared_adaln else nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 3))
            )
            self.image_attn = FullImageSelfAttention(
                hidden_dim,
                config.num_heads,
                backend=config.image_attention_backend,
                qk_rmsnorm=config.image_attention_qk_rmsnorm,
                grid_size=config.latent_size // config.patch_size,
                rope=config.image_attention_rope,
                rope_theta=config.image_rope_theta,
                rope_ntk_factor=config.rope_ntk_factor,
            )
            self.image_attn_scale_shift_table = nn.Parameter(torch.zeros(3, hidden_dim))
            self.sandwich_image_attn = make_sandwich_norm(config)
        self.use_cross_attention = (layer_idx + 1) % config.cross_attention_every == 0
        if not self.use_cross_attention:
            self.cross_attn = None
        else:
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
                    qk_rmsnorm=config.cross_attention_qk_norm,
                )
        self.mod = None if config.shared_adaln else nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 9))
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.norm3 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.mlp = GLUMBConv(hidden_dim, config.mlp_ratio)
        self.scale_shift_table = nn.Parameter(torch.zeros(9, hidden_dim))
        self.sandwich_mixer = make_sandwich_norm(config)
        self.sandwich_mlp = make_sandwich_norm(config)
        self.sandwich_cross = make_sandwich_norm(config) if self.use_cross_attention else nn.Identity()

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

    def _mixer(
        self,
        x: torch.Tensor,
        mixer_perm: torch.Tensor | None,
        mixer_perm_inv: torch.Tensor | None,
    ) -> torch.Tensor:
        if mixer_perm is None:
            return self.self_attn(x)
        # Hilbert order in, raster order out: only the sequence-order-sensitive FLA mixer
        # sees the permutation; everything else in the block stays raster.
        return self.self_attn(x.index_select(1, mixer_perm)).index_select(1, mixer_perm_inv)

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
        t_mod: torch.Tensor | None = None,
        image_attn_t_mod: torch.Tensor | None = None,
        mixer_perm: torch.Tensor | None = None,
        mixer_perm_inv: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.mod is not None:
            timestep_mod = self.mod(t_emb)
        elif t_mod is None:
            raise ValueError("shared_adaln blocks need the precomputed t_mod from BoomerFLADiT.forward")
        else:
            timestep_mod = t_mod
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
        if self.use_image_attention:
            if self.image_attn_mod is not None:
                img_mod = self.image_attn_mod(t_emb)
            elif image_attn_t_mod is None:
                raise ValueError("shared_adaln blocks need the precomputed image_attn_t_mod")
            else:
                img_mod = image_attn_t_mod
            shift_img, scale_img, gate_img = (
                self.image_attn_scale_shift_table[None] + img_mod.reshape(x.shape[0], 3, -1)
            ).chunk(3, dim=1)
        if self.parallel_block:
            base = x
            branches = [
                gate_msa
                * self.sandwich_mixer(
                    self._mixer(modulate(self.norm1(base), shift_msa, scale_msa), mixer_perm, mixer_perm_inv)
                ),
                gate_mlp
                * self.sandwich_mlp(
                    self.mlp(modulate(self.norm2(base), shift_mlp, scale_mlp), height=height, width=width)
                ),
            ]
            if self.use_cross_attention:
                branches.append(
                    gate_cross
                    * self.sandwich_cross(
                        self._cross_attention(
                            modulate(self.norm3(base), shift_cross, scale_cross),
                            text_tokens,
                            text_key_padding_mask,
                            text_attn_bias,
                        )
                    )
                )
            if self.use_image_attention:
                branches.append(
                    gate_img
                    * self.sandwich_image_attn(
                        self.image_attn(modulate(self.image_attn_norm(base), shift_img, scale_img))
                    )
                )
            return base + sum(branches)

        x = x + gate_msa * self.sandwich_mixer(
            self._mixer(modulate(self.norm1(x), shift_msa, scale_msa), mixer_perm, mixer_perm_inv)
        )
        if self.use_image_attention:
            x = x + gate_img * self.sandwich_image_attn(
                self.image_attn(modulate(self.image_attn_norm(x), shift_img, scale_img))
            )
        if self.use_cross_attention:
            x = x + gate_cross * self.sandwich_cross(
                self._cross_attention(
                    modulate(self.norm3(x), shift_cross, scale_cross),
                    text_tokens,
                    text_key_padding_mask,
                    text_attn_bias,
                )
            )
        x = x + gate_mlp * self.sandwich_mlp(
            self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp), height=height, width=width)
        )
        return x


class JointTextImageAttentionMixin:
    """MMDiT-style joint attention over [text; image] with per-modality QKV/projections.

    Host classes provide: num_heads, head_dim, image_qkv/text_qkv, image_q_norm/
    image_k_norm/text_q_norm/text_k_norm, image_out_proj/text_out_proj. Padded text
    tokens are masked out as keys, so they can never contribute to real tokens (their
    own query outputs are garbage but land only in padded slots, which every
    downstream consumer masks again).
    """

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

        # Bool SDPA mask (True = attend): image tokens always attend; padded text is masked.
        image_attend = torch.ones(
            image_tokens.shape[0],
            image_tokens.shape[1],
            device=image_tokens.device,
            dtype=torch.bool,
        )
        attend_mask = torch.cat([~text_key_padding_mask.bool(), image_attend], dim=1)
        attn_bias = attend_mask[:, None, None, :]
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


class BoomerFLADualStreamBlock(JointTextImageAttentionMixin, nn.Module):
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

        self.image_mod = (
            None if config.shared_adaln else nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 6))
        )
        self.image_norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.image_qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.image_q_norm = AttentionRMSNorm(self.head_dim, scale_factor=1.0, eps=1e-6)
        self.image_k_norm = AttentionRMSNorm(self.head_dim, scale_factor=1.0, eps=1e-6)
        self.image_out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.image_norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.image_mlp = GLUMBConv(hidden_dim, config.mlp_ratio)
        self.image_scale_shift_table = nn.Parameter(torch.zeros(6, hidden_dim))

        self.text_mod = (
            None if config.shared_adaln else nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 6))
        )
        self.text_norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.text_qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.text_q_norm = AttentionRMSNorm(self.head_dim, scale_factor=1.0, eps=1e-6)
        self.text_k_norm = AttentionRMSNorm(self.head_dim, scale_factor=1.0, eps=1e-6)
        self.text_out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.text_norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.text_mlp = TokenMLP(hidden_dim, config.mlp_ratio)
        self.text_scale_shift_table = nn.Parameter(torch.zeros(6, hidden_dim))

        self.image_sandwich_attn = make_sandwich_norm(config)
        self.image_sandwich_mlp = make_sandwich_norm(config)
        self.text_sandwich_attn = make_sandwich_norm(config)
        self.text_sandwich_mlp = make_sandwich_norm(config)

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
        image_t_mod: torch.Tensor | None = None,
        text_t_mod: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del text_attn_bias
        if self.image_mod is not None:
            image_timestep_mod = self.image_mod(t_emb)
            text_timestep_mod = self.text_mod(t_emb)
        elif image_t_mod is None or text_t_mod is None:
            raise ValueError("shared_adaln dual-stream blocks need precomputed image_t_mod/text_t_mod")
        else:
            image_timestep_mod = image_t_mod
            text_timestep_mod = text_t_mod
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
        image_attn = self.image_sandwich_attn(image_attn)
        text_attn = self.text_sandwich_attn(text_attn)
        if self.parallel_block:
            x = image_base + image_gate_attn * image_attn + image_gate_mlp * self.image_sandwich_mlp(
                self.image_mlp(
                    modulate(self.image_norm2(image_base), image_shift_mlp, image_scale_mlp),
                    height=height,
                    width=width,
                )
            )
            text_tokens = text_base + text_gate_attn * text_attn + text_gate_mlp * self.text_sandwich_mlp(
                self.text_mlp(modulate(self.text_norm2(text_base), text_shift_mlp, text_scale_mlp))
            )
            return x, text_tokens

        x = image_base + image_gate_attn * image_attn
        text_tokens = text_base + text_gate_attn * text_attn
        x = x + image_gate_mlp * self.image_sandwich_mlp(
            self.image_mlp(
                modulate(self.image_norm2(x), image_shift_mlp, image_scale_mlp),
                height=height,
                width=width,
            )
        )
        text_tokens = text_tokens + text_gate_mlp * self.text_sandwich_mlp(
            self.text_mlp(modulate(self.text_norm2(text_tokens), text_shift_mlp, text_scale_mlp))
        )
        return x, text_tokens


class BoomerFLAJointRefreshBlock(JointTextImageAttentionMixin, nn.Module):
    """v6 joint refresh block: GDN2 spine + one MMDiT-style [text; image] attention.

    Takes the slots the full-image-attention BoomerFLABlocks occupied (every
    image_attention_every-th post-prefix block). The FLA mixer and GLUMBConv stay the
    image-token spine, but the image-only attention and the separate text cross-attn
    are replaced by joint attention whose text half is written back through its own
    out-projection and FFN — so text re-negotiates against the evolving image state
    at every refresh instead of staying frozen after the dual-stream prefix.

    Modulation reuses the single-stream 9-way layout (the cross rows now modulate the
    joint attention), so shared-adaLN blocks consume the same shared_mod projection as
    BoomerFLABlock; the text side reuses the dual-stream 6-way text layout.
    """

    updates_text = True
    joint_refresh = True

    def __init__(self, config: BoomerFLADiTConfig, *, layer_idx: int) -> None:
        super().__init__()
        hidden_dim = config.hidden_dim
        if hidden_dim % config.num_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by num_heads={config.num_heads}")
        self.num_heads = config.num_heads
        self.head_dim = hidden_dim // config.num_heads
        self.hidden_dim = hidden_dim
        self.parallel_block = config.parallel_block

        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        if config.mixer_type in {"torch", "fallback"}:
            self.self_attn = TorchSelfAttention(hidden_dim, config.num_heads)
        else:
            self.self_attn = FLASelfMixer(config, layer_idx=layer_idx)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.mlp = GLUMBConv(hidden_dim, config.mlp_ratio)

        self.joint_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.image_qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.image_q_norm = AttentionRMSNorm(self.head_dim, scale_factor=1.0, eps=1e-6)
        self.image_k_norm = AttentionRMSNorm(self.head_dim, scale_factor=1.0, eps=1e-6)
        self.image_out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.text_norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.text_qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.text_q_norm = AttentionRMSNorm(self.head_dim, scale_factor=1.0, eps=1e-6)
        self.text_k_norm = AttentionRMSNorm(self.head_dim, scale_factor=1.0, eps=1e-6)
        self.text_out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.text_norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.text_mlp = TokenMLP(hidden_dim, config.mlp_ratio)
        # adaLN-Zero style out projections (same convention as FullImageSelfAttention /
        # SanaMultiHeadCrossAttention): the joint attention starts as a no-op on both
        # streams, which also makes a warm-start from a pre-v6 checkpoint behaviorally
        # identical at load.
        nn.init.zeros_(self.image_out_proj.weight)
        nn.init.zeros_(self.image_out_proj.bias)
        nn.init.zeros_(self.text_out_proj.weight)
        nn.init.zeros_(self.text_out_proj.bias)

        self.mod = None if config.shared_adaln else nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 9))
        self.text_mod = (
            None if config.shared_adaln else nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 6))
        )
        self.scale_shift_table = nn.Parameter(torch.zeros(9, hidden_dim))
        self.text_scale_shift_table = nn.Parameter(torch.zeros(6, hidden_dim))

        self.sandwich_mixer = make_sandwich_norm(config)
        self.sandwich_joint = make_sandwich_norm(config)
        self.sandwich_mlp = make_sandwich_norm(config)
        self.text_sandwich_attn = make_sandwich_norm(config)
        self.text_sandwich_mlp = make_sandwich_norm(config)

    def _mixer(
        self,
        x: torch.Tensor,
        mixer_perm: torch.Tensor | None,
        mixer_perm_inv: torch.Tensor | None,
    ) -> torch.Tensor:
        if mixer_perm is None:
            return self.self_attn(x)
        # Hilbert order in, raster order out, same as BoomerFLABlock.
        return self.self_attn(x.index_select(1, mixer_perm)).index_select(1, mixer_perm_inv)

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
        t_mod: torch.Tensor | None = None,
        text_t_mod: torch.Tensor | None = None,
        mixer_perm: torch.Tensor | None = None,
        mixer_perm_inv: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del text_attn_bias
        if self.mod is not None:
            timestep_mod = self.mod(t_emb)
            text_timestep_mod = self.text_mod(t_emb)
        elif t_mod is None or text_t_mod is None:
            raise ValueError("shared_adaln joint refresh blocks need precomputed t_mod/text_t_mod")
        else:
            timestep_mod = t_mod
            text_timestep_mod = text_t_mod
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_joint,
            scale_joint,
            gate_joint,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = (self.scale_shift_table[None] + timestep_mod.reshape(x.shape[0], 9, -1)).chunk(9, dim=1)
        (
            text_shift_attn,
            text_scale_attn,
            text_gate_attn,
            text_shift_mlp,
            text_scale_mlp,
            text_gate_mlp,
        ) = (self.text_scale_shift_table[None] + text_timestep_mod.reshape(text_tokens.shape[0], 6, -1)).chunk(
            6, dim=1
        )

        if self.parallel_block:
            image_base = x
            text_base = text_tokens
            image_attn, text_attn = self._joint_attention(
                modulate(self.joint_norm(image_base), shift_joint, scale_joint),
                modulate(self.text_norm1(text_base), text_shift_attn, text_scale_attn),
                text_key_padding_mask,
                coord_rope,
                image_coord_ids,
                text_coord_ids,
            )
            x = image_base + (
                gate_msa
                * self.sandwich_mixer(
                    self._mixer(modulate(self.norm1(image_base), shift_msa, scale_msa), mixer_perm, mixer_perm_inv)
                )
                + gate_joint * self.sandwich_joint(image_attn)
                + gate_mlp
                * self.sandwich_mlp(
                    self.mlp(modulate(self.norm2(image_base), shift_mlp, scale_mlp), height=height, width=width)
                )
            )
            text_tokens = text_base + (
                text_gate_attn * self.text_sandwich_attn(text_attn)
                + text_gate_mlp
                * self.text_sandwich_mlp(
                    self.text_mlp(modulate(self.text_norm2(text_base), text_shift_mlp, text_scale_mlp))
                )
            )
            return x, text_tokens

        x = x + gate_msa * self.sandwich_mixer(
            self._mixer(modulate(self.norm1(x), shift_msa, scale_msa), mixer_perm, mixer_perm_inv)
        )
        image_attn, text_attn = self._joint_attention(
            modulate(self.joint_norm(x), shift_joint, scale_joint),
            modulate(self.text_norm1(text_tokens), text_shift_attn, text_scale_attn),
            text_key_padding_mask,
            coord_rope,
            image_coord_ids,
            text_coord_ids,
        )
        x = x + gate_joint * self.sandwich_joint(image_attn)
        text_tokens = text_tokens + text_gate_attn * self.text_sandwich_attn(text_attn)
        x = x + gate_mlp * self.sandwich_mlp(
            self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp), height=height, width=width)
        )
        text_tokens = text_tokens + text_gate_mlp * self.text_sandwich_mlp(
            self.text_mlp(modulate(self.text_norm2(text_tokens), text_shift_mlp, text_scale_mlp))
        )
        return x, text_tokens


class BoomerFLADDTDecoderBlock(nn.Module):
    """v7 DDT velocity-decoder block: full softmax attention + GLUMBConv.

    Conditioned token-wise on the condition encoder's output z via adaLN — each image
    token gets its own shift/scale/gate computed from z[token] + t_emb, so the decoder
    re-reads the encoder's semantic plan at every layer (DDT's layerwise conditioning)
    instead of receiving it once. All six decoder sublayers are full softmax attention:
    the decoder is where final part-to-part structure gets committed, and 6 full-attn
    layers over the <=1024-token DC-AE grid are cheap.

    No text inputs: text conditioning is entirely the encoder's job; the decoder sees
    text only through z. No FLA mixer: global attention every layer supersedes it here.
    """

    def __init__(self, config: BoomerFLADiTConfig, *, layer_idx: int) -> None:
        super().__init__()
        del layer_idx  # uniform blocks; kept for signature parity with make_block
        hidden_dim = config.hidden_dim
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.attn = FullImageSelfAttention(
            hidden_dim,
            config.num_heads,
            backend=config.image_attention_backend,
            qk_rmsnorm=config.image_attention_qk_rmsnorm,
            grid_size=config.latent_size // config.patch_size,
            rope=config.image_attention_rope,
            rope_theta=config.image_rope_theta,
            rope_ntk_factor=config.rope_ntk_factor,
        )
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.mlp = GLUMBConv(hidden_dim, config.mlp_ratio)
        # Token-wise conditioning projection over z + t (zero-init in initialize_weights,
        # adaLN-Zero convention); the per-block scale_shift_table carries layer variation.
        self.cond_mod = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 6))
        self.scale_shift_table = nn.Parameter(torch.zeros(6, hidden_dim))
        self.sandwich_attn = make_sandwich_norm(config)
        self.sandwich_mlp = make_sandwich_norm(config)

    def forward(
        self,
        y: torch.Tensor,
        z: torch.Tensor,
        t_emb: torch.Tensor,
        *,
        height: int,
        width: int,
    ) -> torch.Tensor:
        batch, tokens, _ = y.shape
        if z.shape[:2] != (batch, tokens):
            raise ValueError(f"z shape {tuple(z.shape)} must match decoder tokens {(batch, tokens)}")
        # (B, T, 6, D): per-token modulation from the self-condition + timestep.
        mod = self.cond_mod(z + t_emb.unsqueeze(1)).reshape(batch, tokens, 6, -1)
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None, None] + mod
        ).unbind(dim=2)
        y = y + gate_attn * self.sandwich_attn(
            self.attn(modulate(self.norm1(y), shift_attn, scale_attn))
        )
        y = y + gate_mlp * self.sandwich_mlp(
            self.mlp(modulate(self.norm2(y), shift_mlp, scale_mlp), height=height, width=width)
        )
        return y


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
        if config.ddt_decoder_depth < 0:
            raise ValueError(f"ddt_decoder_depth must be non-negative, got {config.ddt_decoder_depth}")
        if config.ddt_decoder_depth >= config.depth:
            raise ValueError(
                f"ddt_decoder_depth={config.ddt_decoder_depth} must leave at least one encoder "
                f"block (depth={config.depth})"
            )
        # depth counts ALL blocks; the last ddt_decoder_depth of them are decoder blocks.
        encoder_depth = config.depth - config.ddt_decoder_depth
        if config.dual_stream_depth > encoder_depth:
            raise ValueError(
                f"dual_stream_depth={config.dual_stream_depth} exceeds the encoder depth "
                f"{encoder_depth} (depth={config.depth} - ddt_decoder_depth={config.ddt_decoder_depth})"
            )
        if config.cross_attention_every < 1:
            raise ValueError(f"cross_attention_every must be >= 1, got {config.cross_attention_every}")
        if config.joint_refresh_attention and config.image_attention_every <= 0:
            raise ValueError("joint_refresh_attention reuses the image_attention_every cadence; set it > 0")
        if config.rope_ntk_factor <= 0.0:
            raise ValueError(f"rope_ntk_factor must be positive, got {config.rope_ntk_factor}")
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
        self.t_embedder = (
            FourierTimestepEmbedder(hidden_dim) if config.fourier_timestep_embed else TimestepEmbedder(hidden_dim)
        )
        # adaLN-single (PixArt-alpha): shared timestep modulation projections, computed once
        # per forward and consumed by every block alongside its per-block scale_shift_table.
        # Each is only built if a block of that shape exists, so no parameter is ever unused.
        self.shared_mod = None
        self.shared_image_attn_mod = None
        self.shared_dual_image_mod = None
        self.shared_dual_text_mod = None
        # Joint refresh slot selection: a refresh block's text write-back is useful only
        # if some LATER block reads text again (another refresh block, or a cross-attn
        # block). The final candidate usually has no such reader, so its slot keeps the
        # classic image-attention block instead of carrying dead text-update weights.
        refresh_indices: set[int] = set()
        if config.joint_refresh_attention:
            reader_after = False
            for i in reversed(range(config.dual_stream_depth, encoder_depth)):
                if (i + 1) % config.image_attention_every == 0 and reader_after:
                    refresh_indices.add(i)
                    reads_text = True
                else:
                    # A plain (or demoted) block reads text iff it cross-attends.
                    reads_text = (i + 1) % config.cross_attention_every == 0
                reader_after = reader_after or reads_text
        self._joint_refresh_indices = refresh_indices

        if config.shared_adaln:
            # Joint refresh blocks consume the same 9-way shared_mod as BoomerFLABlock
            # (their cross rows modulate the joint attention) and the same 6-way text
            # projection as the dual-stream prefix. DDT decoder blocks never use shared
            # adaLN — their modulation is token-wise from z, inherently per-block.
            if config.dual_stream_depth < encoder_depth:
                self.shared_mod = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 9))
            has_image_attn = config.image_attention_every > 0 and any(
                (i + 1) % config.image_attention_every == 0 and i not in refresh_indices
                for i in range(config.dual_stream_depth, encoder_depth)
            )
            if has_image_attn:
                self.shared_image_attn_mod = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 3))
            if config.dual_stream_depth > 0:
                self.shared_dual_image_mod = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 6))
            if config.dual_stream_depth > 0 or refresh_indices:
                self.shared_dual_text_mod = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 6))
        # Hilbert mixer-order permutations, cached per (height, width, device); index
        # tensors only, never serialized.
        self._hilbert_perms: dict[tuple[int, int, str], tuple[torch.Tensor, torch.Tensor]] = {}
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
                ntk_factor=config.rope_ntk_factor,
            )
            if config.multimodal_coord_ids
            else None
        )

        def make_block(i: int) -> nn.Module:
            if i < config.dual_stream_depth:
                return BoomerFLADualStreamBlock(config, layer_idx=i)
            if i in refresh_indices:
                return BoomerFLAJointRefreshBlock(config, layer_idx=i)
            return BoomerFLABlock(config, layer_idx=i)

        # self.blocks is the condition encoder (== the whole model when DDT is off).
        self.blocks = nn.ModuleList([make_block(i) for i in range(encoder_depth)])
        self.encoder_depth = encoder_depth
        # v7 DDT velocity decoder: empty ModuleList (no state-dict keys) when off, so a
        # ddt_decoder_depth=0 model is key-for-key identical to pre-v7 checkpoints.
        self.decoder_blocks = nn.ModuleList(
            [
                BoomerFLADDTDecoderBlock(config, layer_idx=encoder_depth + i)
                for i in range(config.ddt_decoder_depth)
            ]
        )
        # DDT decoder re-embeds the noisy latent itself (D(x_t, z, t), not D(enc_out)):
        # z is forced to carry pure semantics, and the encoder can be reused across
        # adjacent sampling steps at inference.
        self.decoder_x_embedder = None
        if config.ddt_decoder_depth > 0:
            self.decoder_x_embedder = (
                nn.Linear(in_channels, hidden_dim)
                if config.patch_size == 1
                else nn.Conv2d(
                    in_channels,
                    hidden_dim,
                    kernel_size=config.patch_size,
                    stride=config.patch_size,
                )
            )
        self.final_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.final_t_block = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 2))
        self.out_proj = nn.Linear(hidden_dim, config.latent_channels * config.patch_size * config.patch_size)
        # REPA: pooled feature -> SSL embedding space. 3-layer MLP head per the REPA
        # paper; only built (and only run) when repa_dim > 0 and return_repa is set.
        # -1 => encoder mid-depth. The official DDT-XL 22en6de recipe keeps vanilla
        # REPA placement (align_layer 8, a mid-shallow ENCODER block) — it does NOT
        # align the encoder output z, so the default stays mid-depth under DDT too.
        self.repa_layer = config.repa_layer if config.repa_layer >= 0 else encoder_depth // 2
        # Dispersive loss: no projector (parameter-free); only the capture layer is needed.
        self.dispersive_layer = (
            config.dispersive_layer if config.dispersive_layer >= 0 else encoder_depth // 2
        )
        if self.repa_layer >= encoder_depth:
            raise ValueError(
                f"repa_layer={self.repa_layer} must index an encoder block "
                f"(< {encoder_depth}); DDT decoder blocks are not alignment targets"
            )
        if self.dispersive_layer >= encoder_depth:
            raise ValueError(
                f"dispersive_layer={self.dispersive_layer} must index an encoder block (< {encoder_depth})"
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
                if block.image_mod is not None:
                    nn.init.zeros_(block.image_mod[1].weight)
                    nn.init.zeros_(block.image_mod[1].bias)
                    nn.init.zeros_(block.text_mod[1].weight)
                    nn.init.zeros_(block.text_mod[1].bias)
                nn.init.normal_(block.image_scale_shift_table, std=0.02)
                nn.init.normal_(block.text_scale_shift_table, std=0.02)
            elif isinstance(block, BoomerFLAJointRefreshBlock):
                if block.mod is not None:
                    nn.init.zeros_(block.mod[1].weight)
                    nn.init.zeros_(block.mod[1].bias)
                    nn.init.zeros_(block.text_mod[1].weight)
                    nn.init.zeros_(block.text_mod[1].bias)
                nn.init.normal_(block.scale_shift_table, std=0.02)
                nn.init.normal_(block.text_scale_shift_table, std=0.02)
            else:
                if block.mod is not None:
                    nn.init.zeros_(block.mod[1].weight)
                    nn.init.zeros_(block.mod[1].bias)
                nn.init.normal_(block.scale_shift_table, std=0.02)
                if block.use_image_attention:
                    if block.image_attn_mod is not None:
                        nn.init.zeros_(block.image_attn_mod[1].weight)
                        nn.init.zeros_(block.image_attn_mod[1].bias)
                    nn.init.normal_(block.image_attn_scale_shift_table, std=0.02)

        for block in self.decoder_blocks:
            # adaLN-Zero over z: token-wise modulation starts at the (small-normal) table
            # values, same convention as every other block family here.
            nn.init.zeros_(block.cond_mod[1].weight)
            nn.init.zeros_(block.cond_mod[1].bias)
            nn.init.normal_(block.scale_shift_table, std=0.02)

        for shared in (
            self.shared_mod,
            self.shared_image_attn_mod,
            self.shared_dual_image_mod,
            self.shared_dual_text_mod,
        ):
            if shared is not None:
                nn.init.zeros_(shared[1].weight)
                nn.init.zeros_(shared[1].bias)

        nn.init.zeros_(self.final_t_block[1].weight)
        nn.init.zeros_(self.final_t_block[1].bias)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        # Sandwich RMSNorm is degenerate around an exactly-zero sublayer output: the
        # norm's Jacobian at 0 is rsqrt(eps) (~1000x gradient amplification into the
        # zero-init projection) and the first optimizer step snaps the normalized
        # branch from 0 straight to gate-scale unit RMS instead of ramping. With the
        # sandwich on, the no-op start is carried by the near-zero adaLN gates alone,
        # and the sandwich-wrapped output projections get a small normal init. The
        # model's zero-output-at-init property is unaffected (the head stays zero).
        if self.config.sandwich_rmsnorm:
            for block in [*self.blocks, *self.decoder_blocks]:
                for module in block.modules():
                    if isinstance(module, GLUMBConv):
                        nn.init.normal_(module.point_fc.weight, std=0.02)
                    elif isinstance(module, FullImageSelfAttention):
                        nn.init.normal_(module.out_proj.weight, std=0.02)
                    elif isinstance(module, SanaMultiHeadCrossAttention):
                        nn.init.normal_(module.proj.weight, std=0.02)
                if isinstance(block, BoomerFLAJointRefreshBlock):
                    nn.init.normal_(block.image_out_proj.weight, std=0.02)
                    nn.init.normal_(block.text_out_proj.weight, std=0.02)

        # Zero the reserved-input-channel slice so they contribute nothing until a future
        # inpainting fine-tune learns them (base channels keep their default init).
        if self.config.extra_input_channels > 0:
            with torch.no_grad():
                self.x_embedder.weight[:, self.config.latent_channels :].zero_()
                if self.decoder_x_embedder is not None:
                    self.decoder_x_embedder.weight[:, self.config.latent_channels :].zero_()

    def _mixer_perm(
        self, height: int, width: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = (height, width, str(device))
        cached = self._hilbert_perms.get(key)
        if cached is None:
            perm = hilbert_permutation(height, width).to(device)
            cached = (perm, torch.argsort(perm))
            self._hilbert_perms[key] = cached
        return cached

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
        # Bool SDPA mask (True = attend); built once and reused by every cross-attn block.
        text_attn_bias = ~text_key_padding_mask[:, None, None, :]
        t_emb = self.t_embedder(timesteps)
        # adaLN-single: one shared projection per modulation shape, computed once per forward.
        t_mod = self.shared_mod(t_emb) if self.shared_mod is not None else None
        image_attn_t_mod = (
            self.shared_image_attn_mod(t_emb) if self.shared_image_attn_mod is not None else None
        )
        dual_image_t_mod = (
            self.shared_dual_image_mod(t_emb) if self.shared_dual_image_mod is not None else None
        )
        dual_text_t_mod = (
            self.shared_dual_text_mod(t_emb) if self.shared_dual_text_mod is not None else None
        )
        mixer_perm = mixer_perm_inv = None
        if self.config.hilbert_mixer_order:
            mixer_perm, mixer_perm_inv = self._mixer_perm(token_height, token_width, x.device)
        use_ckpt = self.config.gradient_checkpointing and self.training
        want_repa = return_repa and self.repa_proj is not None
        want_dispersive = return_dispersive
        repa_feat: torch.Tensor | None = None
        dispersive_feat: torch.Tensor | None = None
        for layer_idx, block in enumerate(self.blocks):
            if getattr(block, "joint_refresh", False):
                # Joint refresh block: returns (x, text_tokens); needs both the mixer
                # permutation (GDN2 spine) and the joint-attention coord/text inputs.
                _h, _w = token_height, token_width
                _cr, _ii, _ti = self.coord_embedder, image_coord_ids, text_coord_ids
                if use_ckpt:
                    def _refresh_fn(x, tt, te, mk, bi, tm, ttm,
                                    _blk=block, h=_h, w=_w, cr=_cr, ii=_ii, ti=_ti,
                                    mp=mixer_perm, mpi=mixer_perm_inv):
                        return _blk(x, tt, te, mk, bi,
                                    height=h, width=w, coord_rope=cr,
                                    image_coord_ids=ii, text_coord_ids=ti,
                                    t_mod=tm, text_t_mod=ttm,
                                    mixer_perm=mp, mixer_perm_inv=mpi)
                    x, text_tokens = _ckpt(_refresh_fn, x, text_tokens, t_emb,
                                           text_key_padding_mask, text_attn_bias,
                                           t_mod, dual_text_t_mod,
                                           use_reentrant=False,
                                           preserve_rng_state=False)
                else:
                    x, text_tokens = block(
                        x, text_tokens, t_emb, text_key_padding_mask, text_attn_bias,
                        height=token_height, width=token_width,
                        coord_rope=self.coord_embedder,
                        image_coord_ids=image_coord_ids, text_coord_ids=text_coord_ids,
                        t_mod=t_mod, text_t_mod=dual_text_t_mod,
                        mixer_perm=mixer_perm, mixer_perm_inv=mixer_perm_inv,
                    )
            elif getattr(block, "updates_text", False):
                # Dual-stream block: returns (x, text_tokens).
                # Non-tensor args (height, width, coord_rope, coord IDs) captured via closure;
                # shared-adaLN mods are passed as tensor args so checkpointing tracks them.
                _h, _w = token_height, token_width
                _cr, _ii, _ti = self.coord_embedder, image_coord_ids, text_coord_ids
                if use_ckpt:
                    def _dual_fn(x, tt, te, mk, bi, im, tm,
                                 _blk=block, h=_h, w=_w, cr=_cr, ii=_ii, ti=_ti):
                        return _blk(x, tt, te, mk, bi,
                                    height=h, width=w, coord_rope=cr,
                                    image_coord_ids=ii, text_coord_ids=ti,
                                    image_t_mod=im, text_t_mod=tm)
                    x, text_tokens = _ckpt(_dual_fn, x, text_tokens, t_emb,
                                           text_key_padding_mask, text_attn_bias,
                                           dual_image_t_mod, dual_text_t_mod,
                                           use_reentrant=False,
                                           preserve_rng_state=False)
                else:
                    x, text_tokens = block(
                        x, text_tokens, t_emb, text_key_padding_mask, text_attn_bias,
                        height=token_height, width=token_width,
                        coord_rope=self.coord_embedder,
                        image_coord_ids=image_coord_ids, text_coord_ids=text_coord_ids,
                        image_t_mod=dual_image_t_mod, text_t_mod=dual_text_t_mod,
                    )
            else:
                # Single-stream block: returns x only.
                _h, _w = token_height, token_width
                if use_ckpt:
                    def _single_fn(x, tt, te, mk, bi, tm, iam,
                                   _blk=block, h=_h, w=_w,
                                   mp=mixer_perm, mpi=mixer_perm_inv):
                        return _blk(x, tt, te, mk, bi, height=h, width=w,
                                    t_mod=tm, image_attn_t_mod=iam,
                                    mixer_perm=mp, mixer_perm_inv=mpi)
                    x = _ckpt(_single_fn, x, text_tokens, t_emb,
                               text_key_padding_mask, text_attn_bias,
                               t_mod, image_attn_t_mod,
                               use_reentrant=False,
                               preserve_rng_state=False)
                else:
                    x = block(
                        x, text_tokens, t_emb, text_key_padding_mask, text_attn_bias,
                        height=token_height, width=token_width,
                        t_mod=t_mod, image_attn_t_mod=image_attn_t_mod,
                        mixer_perm=mixer_perm, mixer_perm_inv=mixer_perm_inv,
                    )
            if want_repa and layer_idx == self.repa_layer:
                # Pool image tokens -> project to the SSL space for alignment.
                repa_feat = self.repa_proj(x.mean(dim=1))
            if want_dispersive and layer_idx == self.dispersive_layer:
                # Pool image tokens -> per-sample feature; dispersion is computed in the
                # loss. No projector: the regularizer acts on the raw hidden state.
                dispersive_feat = x.mean(dim=1)
        z: torch.Tensor | None = None
        if self.decoder_x_embedder is not None:
            # v7 DDT: the encoder output is the self-condition z; the velocity decoder
            # restarts from a fresh embedding of the (conditioned) noisy latent and reads
            # z token-wise through each decoder block's adaLN.
            z = x
            if self.patch_size == 1:
                x = self.decoder_x_embedder(model_input.flatten(2).transpose(1, 2))
            else:
                x = self.decoder_x_embedder(model_input).flatten(2).transpose(1, 2)
            if self.pos_embed is not None:
                x = x + self.pos_embed
            for block in self.decoder_blocks:
                if use_ckpt:
                    def _dec_fn(y, zz, te, _blk=block, h=token_height, w=token_width):
                        return _blk(y, zz, te, height=h, width=w)
                    x = _ckpt(_dec_fn, x, z, t_emb,
                              use_reentrant=False, preserve_rng_state=False)
                else:
                    x = block(x, z, t_emb, height=token_height, width=token_width)
        if z is not None:
            # DDT final layer: token-wise modulation from silu(t + z) (final_t_block is
            # SiLU->Linear, matching the official FinalLayer(x, s) exactly), so the head
            # also reads the self-condition, not just the timestep.
            final_mod = self.final_t_block(z + t_emb.unsqueeze(1))
            shift, scale = final_mod.reshape(batch, token_count, 2, -1).unbind(dim=2)
        else:
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
