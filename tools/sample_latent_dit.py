"""Sample and decode from a tiny LatentDiT checkpoint."""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

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
from boomer.models.qwen_connector import (  # noqa: E402
    QwenConnectorConfig,
    QwenReferenceEncoder,
    encode_qwen_text_fixed,
)
from boomer.sampling import predict_velocity, sample_latent  # noqa: E402,F401
from boomer.sana_latent_cache import (  # noqa: E402
    CacheConfig,
    decode_image_latent,
    encode_caption,
    hf_token_value,
    load_text_encoder,
    load_vae,
    tensor_to_image,
)
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


def load_condition_from_cache(cache_path: str | Path) -> tuple[torch.Tensor, torch.Tensor, str]:
    record = load_latent_record(cache_path)
    return record.text_embedding.unsqueeze(0), record.attention_mask.unsqueeze(0), record.caption


def load_condition_from_prompt(args: argparse.Namespace, cache_config: CacheConfig) -> tuple[torch.Tensor, torch.Tensor, str]:
    if not args.prompt:
        raise ValueError("Provide either --condition-cache or --prompt.")
    hf_token = hf_token_value(args.hf_token)
    if "qwen" in cache_config.text_encoder_id.lower():
        qwen_config = QwenConnectorConfig(
            qwen_model_id=cache_config.text_encoder_id,
            readout_layer=-1,
            dtype=cache_config.dtype,
            instruction=QwenConnectorConfig.instruction,
            max_text_tokens=cache_config.max_length,
            pad_to_max_length=True,
        )
        encoder = QwenReferenceEncoder(qwen_config, hf_token=hf_token)
        encoder.load()
        encoder.to(cache_config.device)
        text_embedding, attention_mask = encode_qwen_text_fixed(
            encoder, [args.prompt], cache_config.max_length
        )
        del encoder
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        text_processor, text_encoder = load_text_encoder(cache_config, hf_token=hf_token)
        text_embedding, attention_mask = encode_caption(text_processor, text_encoder, args.prompt, cache_config)
    return text_embedding.cpu(), attention_mask.cpu(), args.prompt


def model_config_override(value: str | None) -> dict[str, Any] | None:
    if not value:
        return None
    if value.startswith("@"):
        return json.loads(Path(value[1:]).read_text(encoding="utf-8"))
    return json.loads(value)


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
        "--cfg-interval-low",
        type=float,
        default=0.0,
        help="Apply CFG only while sigma >= this (guidance interval, Kynkaanniemi 2024). "
             "Skips CFG at very low noise to reduce over-saturation. 0 = no lower bound.",
    )
    parser.add_argument(
        "--cfg-interval-high",
        type=float,
        default=1.0,
        help="Apply CFG only while sigma <= this. Skips CFG at very high noise (cheaper, cond/uncond "
             "barely differ there). 1 = no upper bound. Try --cfg-interval-low 0.1 --cfg-interval-high 0.9.",
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
    parser.add_argument(
        "--model-config-json",
        help="Model config JSON, or @path/to/config.json. Required for legacy Lightning "
             "checkpoints that predate embedded model_config metadata.",
    )
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
    if not 0.0 <= args.cfg_interval_low <= args.cfg_interval_high <= 1.0:
        parser.error("--cfg-interval-low/--cfg-interval-high must satisfy 0 <= low <= high <= 1 (sigma units).")
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
    if override := model_config_override(args.model_config_json):
        checkpoint["config"] = override
    if not checkpoint.get("config"):
        parser.error(
            "checkpoint has no embedded model config; pass --model-config-json @config.json"
        )
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
        cfg_interval=(args.cfg_interval_low, args.cfg_interval_high),
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
        f"cfg_rescale={args.cfg_rescale} cfg_interval=[{args.cfg_interval_low},{args.cfg_interval_high}] "
        f"model_type={model.config.model_type} flow_shift={flow_shift} "
        f"y_norm={model.config.y_norm} y_norm_scale_factor={model.config.y_norm_scale_factor}"
    )
    print(f"latent_normalizer: mean=({latent_stat_summary(latent_mean)}) std=({latent_stat_summary(latent_std)})")
    print(f"saved sample: {out_path}")


if __name__ == "__main__":
    main()
