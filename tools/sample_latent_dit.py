"""Sample and decode from a tiny LatentDiT checkpoint."""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from boomer.checkpoint_utils import (  # noqa: E402
    config_from_checkpoint,
    load_checkpoint,
    load_compatible_state_dict,
    select_model_state,
)
from boomer.datasets.latent_shards import load_latent_record  # noqa: E402
from boomer.latent_norm import LatentStat, denormalize_latent, latent_stat_summary  # noqa: E402
from boomer.models.factory import create_model  # noqa: E402
from boomer.sana_flow import make_inference_sigmas  # noqa: E402
from boomer.sana_latent_cache import (  # noqa: E402
    CacheConfig,
    decode_image_latent,
    encode_caption,
    hf_token_value,
    load_text_encoder,
    load_vae,
    tensor_to_image,
)
from boomer.stork_scheduler import make_stork_scheduler  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_dtype(dtype: str, device: str) -> torch.dtype:
    if device == "cpu":
        return torch.float32
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]


def latent_normalizer_from_args(args: argparse.Namespace, checkpoint: dict) -> tuple[LatentStat, LatentStat]:
    train_args = checkpoint.get("train_args") or {}
    latent_mean: LatentStat = float(args.latent_mean) if args.latent_mean is not None else train_args.get("latent_mean", 0.0)
    latent_std: LatentStat = float(args.latent_std) if args.latent_std is not None else train_args.get("latent_std", 1.0)
    return latent_mean, latent_std


def condition_settings_from_args(args: argparse.Namespace, checkpoint: dict) -> tuple[float, bool, float]:
    train_args = checkpoint.get("train_args") or {}
    config = checkpoint.get("config") or {}
    flow_shift = float(args.flow_shift) if args.flow_shift is not None else float(train_args.get("flow_shift", 1.0))
    y_norm = bool(config.get("y_norm", train_args.get("y_norm", False)))
    if args.y_norm is not None:
        y_norm = bool(args.y_norm)
    y_norm_scale = (
        float(args.y_norm_scale_factor)
        if args.y_norm_scale_factor is not None
        else float(config.get("y_norm_scale_factor", train_args.get("y_norm_scale_factor", 0.01)))
    )
    return flow_shift, y_norm, y_norm_scale


@torch.inference_mode()
def predict_velocity(
    model: torch.nn.Module,
    latent: torch.Tensor,
    timestep: torch.Tensor,
    text_embedding: torch.Tensor,
    attention_mask: torch.Tensor,
    uncond_text_embedding: torch.Tensor,
    uncond_attention_mask: torch.Tensor,
    cfg_scale: float,
    cfg_rescale: float = 0.0,
) -> torch.Tensor:
    if cfg_scale == 1.0:
        return model(latent, timestep, text_embedding, attention_mask)

    latent_batch = latent.repeat(2, 1, 1, 1)
    timestep_batch = timestep.repeat(2)
    text_batch = torch.cat([uncond_text_embedding, text_embedding], dim=0)
    mask_batch = torch.cat([uncond_attention_mask, attention_mask], dim=0)
    uncond_velocity, cond_velocity = model(latent_batch, timestep_batch, text_batch, mask_batch).chunk(2)
    guided = uncond_velocity + cfg_scale * (cond_velocity - uncond_velocity)
    if cfg_rescale <= 0.0:
        return guided

    std_cond = cond_velocity.std(dim=(1, 2, 3), keepdim=True)
    std_guided = guided.std(dim=(1, 2, 3), keepdim=True)
    rescaled = guided * (std_cond / std_guided.clamp_min(1e-5))
    return cfg_rescale * rescaled + (1.0 - cfg_rescale) * guided


@torch.inference_mode()
def sample_latent(
    model: torch.nn.Module,
    text_embedding: torch.Tensor,
    attention_mask: torch.Tensor,
    steps: int,
    seed: int,
    dtype: torch.dtype,
    device: str,
    cfg_scale: float,
    cfg_rescale: float = 0.0,
    sampler: str = "stork2",
    flow_shift: float = 1.0,
    stork_derivative_order: int = 1,
    stork_substeps: int = 50,
) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(seed)
    config = model.config
    latent = torch.randn(
        1,
        config.latent_channels,
        config.latent_size,
        config.latent_size,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    text_embedding = text_embedding.to(device=device, dtype=dtype)
    attention_mask = attention_mask.to(device=device)
    uncond_text_embedding, uncond_attention_mask = model.null_condition(
        1,
        device=device,
        dtype=dtype,
        mask_dtype=attention_mask.dtype,
        token_num=text_embedding.shape[-2],
    )

    if sampler.startswith("stork"):
        solver_order = int(sampler.removeprefix("stork"))
        scheduler = make_stork_scheduler(
            steps=steps,
            device=device,
            flow_shift=flow_shift,
            solver_order=solver_order,
            derivative_order=stork_derivative_order,
            substeps=stork_substeps,
        )
        for index in range(steps):
            sigma = scheduler.sigmas[index].to(device=device, dtype=dtype)
            velocity = predict_velocity(
                model,
                latent,
                sigma.expand(1),
                text_embedding,
                attention_mask,
                uncond_text_embedding,
                uncond_attention_mask,
                cfg_scale,
                cfg_rescale,
            )
            latent = scheduler.step(velocity, sigma, latent, return_dict=True).prev_sample
        return latent

    sigmas = make_inference_sigmas(steps, device=device, dtype=dtype, flow_shift=flow_shift)
    for index in range(steps):
        sigma = sigmas[index]
        next_sigma = sigmas[index + 1]
        dt = sigma - next_sigma
        timestep = sigma.expand(1)
        velocity = predict_velocity(
            model,
            latent,
            timestep,
            text_embedding,
            attention_mask,
            uncond_text_embedding,
            uncond_attention_mask,
            cfg_scale,
            cfg_rescale,
        )
        if sampler == "euler":
            latent = latent - velocity * dt
        elif sampler == "heun":
            predictor = latent - velocity * dt
            next_timestep = next_sigma.expand(1)
            next_velocity = predict_velocity(
                model,
                predictor,
                next_timestep,
                text_embedding,
                attention_mask,
                uncond_text_embedding,
                uncond_attention_mask,
                cfg_scale,
                cfg_rescale,
            )
            latent = latent - 0.5 * dt * (velocity + next_velocity)
        else:
            raise ValueError(f"Unsupported sampler: {sampler}")
    return latent


def load_condition_from_cache(cache_path: str | Path) -> tuple[torch.Tensor, torch.Tensor, str]:
    record = load_latent_record(cache_path)
    return record.text_embedding.unsqueeze(0), record.attention_mask.unsqueeze(0), record.caption


def load_condition_from_prompt(args: argparse.Namespace, cache_config: CacheConfig) -> tuple[torch.Tensor, torch.Tensor, str]:
    if not args.prompt:
        raise ValueError("Provide either --condition-cache or --prompt.")
    hf_token = hf_token_value(args.hf_token)
    text_processor, text_encoder = load_text_encoder(cache_config, hf_token=hf_token)
    text_embedding, attention_mask = encode_caption(text_processor, text_encoder, args.prompt, cache_config)
    return text_embedding.cpu(), attention_mask.cpu(), args.prompt


def validate_condition_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    args.condition_cache = args.condition_cache.strip() if args.condition_cache else None
    args.prompt = args.prompt.strip() if args.prompt else None

    if bool(args.condition_cache) == bool(args.prompt):
        parser.error(
            "provide exactly one of --condition-cache or --prompt. "
            "Colab note: shell variables created in one ! command do not persist; "
            "put FIRST_CACHE=... and python tools/sample_latent_dit.py in the same %%bash cell."
        )

    if args.condition_cache and not Path(args.condition_cache).is_file():
        parser.error(f"--condition-cache does not exist or is not a file: {args.condition_cache}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample a Boomer latent denoiser checkpoint and decode with AutoencoderDC.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--condition-cache", help="Cached .pt record to reuse text conditioning from.")
    parser.add_argument("--prompt", help="Fresh prompt to encode with Gemma 4.")
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
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
    parser.add_argument("--cfg-scale", type=float, default=1.0, help="Classifier-free guidance scale. 1 disables CFG.")
    parser.add_argument(
        "--cfg-rescale",
        type=float,
        default=0.0,
        help="CFG variance rescale mix in [0, 1]. 0 disables. Try 0.5-0.7 when --cfg-scale > 1.",
    )
    parser.add_argument(
        "--weights",
        choices=["auto", "model", "ema"],
        default="auto",
        help="Which checkpoint weights to sample. auto uses EMA when available.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--vae-id", default=CacheConfig.vae_id)
    parser.add_argument("--text-encoder-id", default=CacheConfig.text_encoder_id)
    parser.add_argument("--max-length", type=int, default=CacheConfig.max_length)
    parser.add_argument("--scaling-factor", type=float, default=CacheConfig.scaling_factor)
    parser.add_argument("--latent-mean", type=float, help="Override checkpoint latent normalization mean.")
    parser.add_argument("--latent-std", type=float, help="Override checkpoint latent normalization std.")
    parser.add_argument("--flow-shift", type=float, help="Override checkpoint Sana/FlowMatch sigma shift.")
    parser.add_argument("--y-norm", dest="y_norm", action="store_true", default=None, help="Override checkpoint y-norm on.")
    parser.add_argument("--no-y-norm", dest="y_norm", action="store_false", help="Override checkpoint y-norm off.")
    parser.add_argument("--y-norm-scale-factor", type=float, help="Override checkpoint y-norm scale.")
    parser.add_argument("--hf-token", help="Hugging Face token, or 'true' to use the logged-in token.")
    args = parser.parse_args()
    validate_condition_args(parser, args)
    if args.cfg_scale < 0.0:
        parser.error("--cfg-scale must be non-negative.")
    if not 0.0 <= args.cfg_rescale <= 1.0:
        parser.error("--cfg-rescale must be in [0, 1].")
    if args.flow_shift is not None and args.flow_shift <= 0.0:
        parser.error("--flow-shift must be positive.")
    if args.stork_derivative_order not in {1, 2, 3}:
        parser.error("--stork-derivative-order must be 1, 2, or 3.")
    if args.stork_substeps < 2:
        parser.error("--stork-substeps must be at least 2.")
    if args.y_norm_scale_factor is not None and args.y_norm_scale_factor <= 0.0:
        parser.error("--y-norm-scale-factor must be positive.")

    set_seed(args.seed)

    device = args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    dtype = resolve_dtype(args.dtype, device)
    cache_config = CacheConfig(
        vae_id=args.vae_id,
        text_encoder_id=args.text_encoder_id,
        max_length=args.max_length,
        dtype=args.dtype,
        device=device,
        scaling_factor=args.scaling_factor,
    )

    checkpoint = load_checkpoint(args.checkpoint)
    model_config = config_from_checkpoint(checkpoint)
    flow_shift, y_norm, y_norm_scale_factor = condition_settings_from_args(args, checkpoint)
    if args.y_norm is not None or args.y_norm_scale_factor is not None:
        model_config = replace(
            model_config,
            y_norm=y_norm,
            y_norm_scale_factor=y_norm_scale_factor,
        )
    model = create_model(model_config).to(device=device, dtype=dtype).eval()
    state_dict, weights_name = select_model_state(checkpoint, args.weights)
    skipped = load_compatible_state_dict(model, state_dict)
    if skipped:
        print(f"warning: skipped {len(skipped)} incompatible checkpoint tensors")
    latent_mean, latent_std = latent_normalizer_from_args(args, checkpoint)
    if (args.y_norm is not None or args.y_norm_scale_factor is not None) and (
        model.config.y_norm != bool((checkpoint.get("config") or {}).get("y_norm", False))
    ):
        print("warning: y_norm override changes the model architecture from the checkpoint config.")

    if args.condition_cache:
        text_embedding, attention_mask, caption = load_condition_from_cache(args.condition_cache)
    else:
        text_embedding, attention_mask, caption = load_condition_from_prompt(args, cache_config)

    latent = sample_latent(
        model=model,
        text_embedding=text_embedding,
        attention_mask=attention_mask,
        steps=args.steps,
        seed=args.seed,
        dtype=dtype,
        device=device,
        cfg_scale=args.cfg_scale,
        cfg_rescale=args.cfg_rescale,
        sampler=args.sampler,
        flow_shift=flow_shift,
        stork_derivative_order=args.stork_derivative_order,
        stork_substeps=args.stork_substeps,
    )
    latent = denormalize_latent(latent, latent_mean, latent_std)

    vae = load_vae(cache_config, hf_token=hf_token_value(args.hf_token))
    decoded = decode_image_latent(vae, latent, cache_config)
    image = tensor_to_image(decoded)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
    print(f"caption: {caption}")
    print(
        f"weights: {weights_name} sampler={args.sampler} steps={args.steps} cfg_scale={args.cfg_scale} "
        f"cfg_rescale={args.cfg_rescale} "
        f"model_type={model.config.model_type} flow_shift={flow_shift} "
        f"y_norm={model.config.y_norm} y_norm_scale_factor={model.config.y_norm_scale_factor}"
    )
    print(f"latent_normalizer: mean=({latent_stat_summary(latent_mean)}) std=({latent_stat_summary(latent_std)})")
    print(f"saved sample: {out_path}")


if __name__ == "__main__":
    main()
