"""Denoise a known cached latent to debug LatentDiT learning."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from boomer.checkpoint_utils import (  # noqa: E402
    config_from_checkpoint,
    load_checkpoint,
    load_compatible_state_dict,
    select_model_state,
)
from boomer.datasets.latent_shards import load_latent_record  # noqa: E402
from boomer.denoise import denoise_latent  # noqa: E402
from boomer.latent_norm import LatentStat, denormalize_latent, latent_stat_summary, normalize_latent  # noqa: E402
from boomer.models.factory import create_model  # noqa: E402
from boomer.sana_flow import noise_latent_at_sigma  # noqa: E402
from boomer.sana_latent_cache import CacheConfig, decode_image_latent, hf_token_value, load_vae, tensor_to_image  # noqa: E402


def split_float_csv(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def resolve_dtype(dtype: str, device: str) -> torch.dtype:
    if device == "cpu":
        return torch.float32
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]


def latent_normalizer_from_args(args: argparse.Namespace, checkpoint: dict) -> tuple[LatentStat, LatentStat]:
    train_args = checkpoint.get("train_args") or {}
    latent_mean: LatentStat = float(args.latent_mean) if args.latent_mean is not None else train_args.get("latent_mean", 0.0)
    latent_std: LatentStat = float(args.latent_std) if args.latent_std is not None else train_args.get("latent_std", 1.0)
    return latent_mean, latent_std


def condition_settings_from_args(args: argparse.Namespace, checkpoint: dict) -> float:
    train_args = checkpoint.get("train_args") or {}
    return float(args.flow_shift) if args.flow_shift is not None else float(train_args.get("flow_shift", 1.0))


@torch.inference_mode()
def decode_to_file(vae: Any, latent: torch.Tensor, cache_config: CacheConfig, path: Path) -> None:
    image = tensor_to_image(decode_image_latent(vae, latent, cache_config))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Denoise one cached latent with a Boomer checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache-record", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--t", type=float, default=0.3, help="Forward noising timestep in [0, 1] (flow-shifted sigma).")
    parser.add_argument("--t-values", type=split_float_csv, help="Comma-separated t grid, e.g. 0.3,0.6,0.8,0.95.")
    parser.add_argument("--steps", type=int, default=8, help="Reverse denoise steps from t to 0.")
    parser.add_argument("--sampler", choices=["stork2", "stork4", "euler", "heun"], default="stork2")
    parser.add_argument("--stork-derivative-order", type=int, default=1)
    parser.add_argument(
        "--stork-substeps",
        type=int,
        default=50,
        help=(
            "STORK intra-step RK micro-steps (STORKScheduler s). "
            "Not extra model forwards; internal integration per inference step. Default 50 matches STORKScheduler."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--weights",
        choices=["auto", "model", "ema"],
        default="auto",
        help="Which checkpoint weights to use. auto uses EMA when available.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--vae-id", default=CacheConfig.vae_id)
    parser.add_argument("--text-encoder-id", default=CacheConfig.text_encoder_id)
    parser.add_argument("--max-length", type=int, default=300)
    parser.add_argument("--scaling-factor", type=float, default=CacheConfig.scaling_factor)
    parser.add_argument("--latent-mean", type=float, help="Override checkpoint latent normalization mean.")
    parser.add_argument("--latent-std", type=float, help="Override checkpoint latent normalization std.")
    parser.add_argument("--flow-shift", type=float, help="Override checkpoint Sana/FlowMatch sigma shift.")
    parser.add_argument("--hf-token", help="Hugging Face token, or 'true' to use the logged-in token.")
    args = parser.parse_args()
    if args.flow_shift is not None and args.flow_shift <= 0.0:
        parser.error("--flow-shift must be positive.")
    if args.stork_derivative_order not in {1, 2, 3}:
        parser.error("--stork-derivative-order must be 1, 2, or 3.")
    if args.stork_substeps < 2:
        parser.error("--stork-substeps must be at least 2.")

    device = args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    dtype = resolve_dtype(args.dtype, device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = load_checkpoint(args.checkpoint)
    model_config = config_from_checkpoint(checkpoint)
    model = create_model(model_config).to(device=device, dtype=dtype).eval()
    state_dict, weights_name = select_model_state(checkpoint, args.weights)
    skipped = load_compatible_state_dict(model, state_dict)
    if skipped:
        print(f"warning: skipped {len(skipped)} incompatible checkpoint tensors")
    latent_mean, latent_std = latent_normalizer_from_args(args, checkpoint)
    flow_shift = condition_settings_from_args(args, checkpoint)

    record = load_latent_record(args.cache_record)
    raw_x0 = record.image_latent.unsqueeze(0).to(device=device, dtype=dtype)
    x0 = normalize_latent(raw_x0, latent_mean, latent_std)
    text_embedding = record.text_embedding.unsqueeze(0).to(device=device, dtype=dtype)
    attention_mask = record.attention_mask.unsqueeze(0).to(device=device)

    generator = torch.Generator(device=device).manual_seed(args.seed)

    cache_config = CacheConfig(
        vae_id=args.vae_id,
        text_encoder_id=args.text_encoder_id,
        max_length=args.max_length,
        dtype=args.dtype,
        device=device,
        scaling_factor=args.scaling_factor,
    )
    vae = load_vae(cache_config, hf_token=hf_token_value(args.hf_token))

    decode_to_file(vae, raw_x0, cache_config, out_dir / "original.png")
    print(f"caption: {record.caption}")
    print(f"weights: {weights_name} sampler={args.sampler} steps={args.steps}")
    print(f"model_type={model.config.model_type} flow_shift={flow_shift} y_norm={model.config.y_norm}")
    print(f"latent_normalizer: mean=({latent_stat_summary(latent_mean)}) std=({latent_stat_summary(latent_std)})")
    print(f"saved: {out_dir}")

    t_values = args.t_values if args.t_values is not None else [args.t]
    for t in t_values:
        if not 0.0 <= t <= 1.0:
            raise ValueError(f"t must be in [0, 1], got {t}")
        x_t, _ = noise_latent_at_sigma(x0, t, generator=generator)
        denoised = denoise_latent(
            model,
            x_t,
            t,
            text_embedding,
            attention_mask,
            steps=args.steps,
            flow_shift=flow_shift,
            sampler=args.sampler,
            stork_derivative_order=args.stork_derivative_order,
            stork_substeps=args.stork_substeps,
        )
        decode_to_file(vae, denormalize_latent(x_t, latent_mean, latent_std), cache_config, out_dir / f"noised_t{t:.2f}.png")
        decode_to_file(
            vae,
            denormalize_latent(denoised, latent_mean, latent_std),
            cache_config,
            out_dir / f"denoised_t{t:.2f}_steps{args.steps}.png",
        )
        mse_noised = torch.mean((x_t.float() - x0.float()) ** 2).item()
        mse_denoised = torch.mean((denoised.float() - x0.float()) ** 2).item()
        print(f"t={t:.2f} latent_mse_noised={mse_noised:.6f} latent_mse_denoised={mse_denoised:.6f}")


if __name__ == "__main__":
    main()
