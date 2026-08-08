"""Attribute v8's training peak memory to specific features, at production shapes.

Runs forward+backward on the real model (no data pipeline, no Qwen) under bf16
autocast and reports torch.cuda.max_memory_allocated() for a series of configs that
differ by ONE feature each, so the delta is the feature's cost:

    v8 full -> v8 without Block AttnRes -> v8 with additive refresh (v7 form)
    -> v7-equivalent layout -> v7-equivalent with single-layer text

Usage (on the training GPU):

    python tools/mem_profile.py                 # 512 phase, batch 40
    python tools/mem_profile.py --batch 56
    python tools/mem_profile.py --latent-size 32 --batch 14   # 1024 phase

Also prints which SDPA backend the joint-attention blocks select: a masked
scaled_dot_product_attention that falls back to the math backend materializes
[B, heads, S, S] and retains it for backward, which at S = text_seq_len + tokens
dwarfs everything else.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from boomer.models.boomer_fla_dit import (  # noqa: E402
    BoomerFLADiT,
    BoomerFLADiTConfig,
    maybe_add_sibling_fla_repo,
)

maybe_add_sibling_fla_repo()


def base_kwargs(latent_size: int, text_dim: int) -> dict:
    """Mirrors the v8 512 launch flags exactly."""
    return dict(
        latent_channels=32,
        latent_size=latent_size,
        extra_input_channels=33,
        text_dim=text_dim,
        text_seq_len=320,
        hidden_dim=1152,
        depth=28,
        ddt_decoder_depth=6,
        num_heads=18,
        mlp_ratio=2.5,
        mixer_type="fla_gdn2",
        fla_mode="chunk",
        fla_bidirectional=True,
        use_short_conv=True,
        conv_size=4,
        image_attention_every=0,
        image_attention_backend="flash2",
        image_attention_rope=True,
        image_rope_theta=10000.0,
        cross_attention_backend="sdpa",
        image_attention_qk_rmsnorm=True,
        cross_attention_qk_norm=True,
        ddt_norm_impl="native",
        shared_adaln=True,
        fourier_timestep_embed=True,
        hilbert_mixer_order=True,
        parallel_block=True,
        dual_stream_depth=4,
        multimodal_coord_ids=True,
        joint_refresh_attention=True,
        refresh_layers=(7, 11, 15, 21),
        refresh_no_mixer=True,
        block_attnres=True,
        attnres_group_ends=(2, 5, 8, 11, 14, 17, 19, 21),
        sandwich_rmsnorm=True,
        rope_ntk_factor=1.0,
        use_abs_pos_embed=False,
        y_norm=True,
        y_norm_scale_factor=0.01,
        patch_size=1,
        repa_dim=1536,
        repa_layer=7,
    )


def measure(label: str, cfg: BoomerFLADiTConfig, batch: int) -> float:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = BoomerFLADiT(cfg).cuda().train()
    params_gb = sum(p.numel() for p in model.parameters()) * 4 / 1e9
    after_model = torch.cuda.memory_allocated() / 1e9

    x = torch.randn(batch, cfg.latent_channels + cfg.extra_input_channels,
                    cfg.latent_size, cfg.latent_size, device="cuda")
    t = torch.rand(batch, device="cuda")
    text = torch.randn(batch, cfg.text_seq_len, cfg.text_dim,
                       device="cuda", dtype=torch.bfloat16)
    mask = torch.ones(batch, cfg.text_seq_len, dtype=torch.long, device="cuda")

    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(x[:, :cfg.latent_channels], t, text, mask,
                    cond_input=x[:, cfg.latent_channels:])
    out.float().square().mean().backward()
    torch.cuda.synchronize()

    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"{label:<44} peak {peak:7.2f} GB   "
          f"(params {params_gb:.2f} GB fp32, resident after build {after_model:.2f} GB, "
          f"activations ~{peak - after_model:.2f} GB)")
    del model, x, t, text, mask, out
    torch.cuda.empty_cache()
    return peak


def report_sdpa_backend(batch: int, tokens: int, text_len: int) -> None:
    """Which SDPA kernel serves the masked joint attention? The math backend
    materializes [B, H, S, S] AND keeps it for backward."""
    seq = text_len + tokens
    heads, head_dim = 18, 64
    q, k, v = (torch.randn(batch, heads, seq, head_dim, device="cuda",
                           dtype=torch.bfloat16) for _ in range(3))
    bias = torch.ones(batch, 1, 1, seq, dtype=torch.bool, device="cuda")
    materialized = batch * heads * seq * seq * 2 / 1e9
    print(f"\njoint attention: S={seq} (text {text_len} + image {tokens}), "
          f"a materialized [B,H,S,S] bf16 score matrix is {materialized:.2f} GB per call")
    for name, flag in (
        ("flash", torch.nn.attention.SDPBackend.FLASH_ATTENTION),
        ("mem_efficient", torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION),
        ("math", torch.nn.attention.SDPBackend.MATH),
    ):
        try:
            with torch.nn.attention.sdpa_kernel(flag):
                torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=bias)
            print(f"  {name:<14} usable with a bool attn_mask: YES")
        except Exception as exc:
            print(f"  {name:<14} usable with a bool attn_mask: no ({type(exc).__name__})")
    del q, k, v, bias
    torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=40)
    ap.add_argument("--latent-size", type=int, default=16)
    args = ap.parse_args()

    print(f"device: {torch.cuda.get_device_name()}; torch {torch.__version__}")
    print(f"batch={args.batch} latent={args.latent_size} "
          f"tokens={args.latent_size ** 2}\n")

    tokens = args.latent_size ** 2
    report_sdpa_backend(args.batch, tokens, 320)
    print()

    full = base_kwargs(args.latent_size, 7680)
    measure("v8 full", BoomerFLADiTConfig(**full), args.batch)
    measure("v8 - Block AttnRes",
            BoomerFLADiTConfig(**{**full, "block_attnres": False,
                                  "attnres_group_ends": ()}), args.batch)
    measure("v8 - AttnRes - substitutive refresh",
            BoomerFLADiTConfig(**{**full, "block_attnres": False,
                                  "attnres_group_ends": (),
                                  "refresh_no_mixer": False}), args.batch)
    measure("v7 layout (cadence refresh, no AttnRes)",
            BoomerFLADiTConfig(**{**full, "block_attnres": False,
                                  "attnres_group_ends": (),
                                  "refresh_no_mixer": False,
                                  "refresh_layers": None,
                                  "image_attention_every": 4}), args.batch)
    measure("v7 layout + single-layer text (dim 2560)",
            BoomerFLADiTConfig(**{**base_kwargs(args.latent_size, 2560),
                                  "block_attnres": False,
                                  "attnres_group_ends": (),
                                  "refresh_no_mixer": False,
                                  "refresh_layers": None,
                                  "image_attention_every": 4}), args.batch)

    # Where do the baseline activations actually go? Each row disables ONE component
    # of v8 full; a large drop names the term. These configs are diagnostic only —
    # they change the architecture and are not trainable substitutes.
    print("\n--- baseline attribution (each row cripples one component) ---")
    measure("v8 + gradient checkpointing",
            BoomerFLADiTConfig(**{**full, "gradient_checkpointing": True}), args.batch)
    measure("v8, cross-attn every 4 (not every block)",
            BoomerFLADiTConfig(**{**full, "cross_attention_every": 4}), args.batch)
    measure("v8, mlp_ratio 1.0 (GLUMBConv width)",
            BoomerFLADiTConfig(**{**full, "mlp_ratio": 1.0}), args.batch)
    measure("v8, no dual-stream prefix",
            BoomerFLADiTConfig(**{**full, "dual_stream_depth": 0}), args.batch)
    measure("v8, torch attention instead of GDN2",
            BoomerFLADiTConfig(**{**full, "mixer_type": "torch"}), args.batch)
    measure("v8, GDN2 unidirectional",
            BoomerFLADiTConfig(**{**full, "fla_bidirectional": False}), args.batch)
    measure("v8, no short conv",
            BoomerFLADiTConfig(**{**full, "use_short_conv": False}), args.batch)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("needs the training GPU")
    main()
