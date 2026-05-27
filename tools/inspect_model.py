"""Instantiate a Boomer denoiser config and report parameter counts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from boomer.models.factory import create_model  # noqa: E402
from boomer.models.latent_dit import LatentDiTConfig  # noqa: E402
from boomer.models.mamba_anchor_dit import MambaAnchorDiTConfig  # noqa: E402
from boomer.models.boomer_fla_dit import BoomerFLADiTConfig  # noqa: E402


def parameter_count(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable


def build_config(args: argparse.Namespace):
    common = {
        "latent_channels": args.latent_channels,
        "latent_size": args.latent_size,
        "text_dim": args.text_dim,
        "text_seq_len": args.text_seq_len,
        "hidden_dim": args.hidden_dim,
        "depth": args.depth,
        "num_heads": args.num_heads,
        "mlp_ratio": args.mlp_ratio,
        "y_norm": args.y_norm,
        "y_norm_scale_factor": args.y_norm_scale_factor,
    }
    if args.model_type == "latent_dit":
        return LatentDiTConfig(**common, cross_attention=args.cross_attn)
    if args.model_type == "mamba_anchor":
        return MambaAnchorDiTConfig(
            **common,
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
            **common,
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
    raise ValueError(f"Unsupported model type: {args.model_type}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect Boomer denoiser model size.")
    parser.add_argument("--model-type", choices=["latent_dit", "mamba_anchor", "boomer_fla"], default="mamba_anchor")
    parser.add_argument("--latent-channels", type=int, default=32)
    parser.add_argument("--latent-size", type=int, default=16)
    parser.add_argument("--text-dim", type=int, default=1536)
    parser.add_argument("--text-seq-len", type=int, default=300)
    parser.add_argument("--hidden-dim", type=int, default=1280)
    parser.add_argument("--depth", type=int, default=24)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--cross-attn", action="store_true")
    parser.add_argument("--y-norm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--y-norm-scale-factor", type=float, default=0.01)
    parser.add_argument("--num-anchors", type=int, default=64)
    parser.add_argument("--text-attn-every", type=int, default=2)
    parser.add_argument("--anchor-attn-every", type=int, default=3)
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
    parser.add_argument("--mamba-mimo", action="store_true")
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
    parser.add_argument("--image-attn-every", type=int, default=0)
    parser.add_argument("--image-attn-backend", choices=["sdpa", "flash3", "flash4", "auto"], default="sdpa")
    parser.add_argument("--image-rope", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--image-rope-theta", type=float, default=10000.0)
    parser.add_argument("--cross-attn-backend", choices=["sdpa", "xformers", "auto", "mha"], default="sdpa")
    parser.add_argument("--cross-attn-qk-norm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--parallel-block", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dual-stream-depth", type=int, default=0)
    parser.add_argument("--multimodal-coord-ids", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--abs-pos-embed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--patch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--forward", action="store_true", help="Run one tiny random forward pass.")
    args = parser.parse_args()

    config = build_config(args)
    device = args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    if device == "cpu":
        dtype = torch.float32

    model = create_model(config).to(device=device, dtype=dtype)
    total, trainable = parameter_count(model)
    print(f"model_type={config.model_type}")
    print(f"config={config}")
    print(f"params_total={total:,}")
    print(f"params_trainable={trainable:,}")
    print(f"device={device} dtype={dtype}")

    if args.forward:
        model.eval()
        latent = torch.randn(
            1,
            config.latent_channels,
            config.latent_size,
            config.latent_size,
            device=device,
            dtype=dtype,
        )
        timesteps = torch.rand(1, device=device, dtype=dtype)
        text = torch.randn(1, config.text_seq_len, config.text_dim, device=device, dtype=dtype)
        mask = torch.ones(1, config.text_seq_len, device=device, dtype=torch.long)
        with torch.inference_mode():
            out = model(latent, timesteps, text, mask)
        print(f"forward_output_shape={tuple(out.shape)} forward_dtype={out.dtype}")


if __name__ == "__main__":
    main()
