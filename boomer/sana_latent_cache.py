"""Sana-compatible latent cache utilities using Hugging Face encoders.

Adapted from NVlabs/Sana (https://github.com/NVlabs/Sana): resize-to-fill crop,
BOS + trailing-token text selection, and latent record layout.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


DEFAULT_VAE_ID = "mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers"
DEFAULT_TEXT_ENCODER_ID = "google/gemma-4-E2B-it"
DEFAULT_SCALING_FACTOR = 0.41407


@dataclass(frozen=True)
class CacheConfig:
    resolution: int = 512
    vae_id: str = DEFAULT_VAE_ID
    text_encoder_id: str = DEFAULT_TEXT_ENCODER_ID
    text_attn_implementation: str | None = None
    max_length: int = 340
    dtype: str = "bf16"
    device: str = "cuda"
    scaling_factor: float = DEFAULT_SCALING_FACTOR


def resolve_dtype(dtype: str, device: str) -> torch.dtype:
    normalized = dtype.lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16 if device != "cpu" else torch.float32
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16 if device != "cpu" else torch.float32
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def hf_token_value(token: str | None) -> str | bool | None:
    if token is None or token == "":
        return None
    if token.lower() in {"1", "true", "yes"}:
        return True
    return token


def resize_crop_to_fill_image(image: Image.Image, image_size: int) -> Image.Image:
    """Sana-style resize-to-fill followed by center crop."""
    width, height = image.size
    target_h = target_w = image_size
    ratio_h = target_h / height
    ratio_w = target_w / width

    if ratio_h > ratio_w:
        resized_h = target_h
        resized_w = round(width * ratio_h)
        crop_i = 0
        crop_j = int(round((resized_w - target_w) / 2.0))
    else:
        resized_h = round(height * ratio_w)
        resized_w = target_w
        crop_i = int(round((resized_h - target_h) / 2.0))
        crop_j = 0

    image = image.resize((resized_w, resized_h), Image.BICUBIC)
    return image.crop((crop_j, crop_i, crop_j + target_w, crop_i + target_h))


def image_to_tensor(image: Image.Image, resolution: int) -> torch.Tensor:
    image = resize_crop_to_fill_image(image.convert("RGB"), resolution)
    array = np.asarray(image, dtype=np.uint8).copy()
    tensor = torch.from_numpy(array).permute(2, 0, 1).float()
    tensor = tensor / 127.5 - 1.0
    return tensor.unsqueeze(0).contiguous()


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    tensor = tensor.detach().float().cpu().clamp(-1, 1)
    tensor = ((tensor + 1.0) * 127.5).round().to(torch.uint8)
    if tensor.ndim == 4:
        tensor = tensor[0]
    tensor = tensor.permute(1, 2, 0).contiguous()
    return Image.fromarray(tensor.numpy(), mode="RGB")


def load_vae(config: CacheConfig, hf_token: str | bool | None = None):
    from diffusers import AutoencoderDC

    dtype = resolve_dtype(config.dtype, config.device)
    vae = AutoencoderDC.from_pretrained(config.vae_id, torch_dtype=dtype, token=hf_token)
    vae = vae.to(config.device).eval()
    vae.requires_grad_(False)
    return vae


def _uses_sana_gemma_encoding(text_encoder_id: str) -> bool:
    lowered = text_encoder_id.lower()
    return "gemma" in lowered or "qwen" in lowered


def gemma_token_select_index(max_length: int) -> list[int]:
    """Sana-style BOS + trailing caption tokens."""
    return [0] + list(range(-max_length + 1, 0))


def load_text_encoder(config: CacheConfig, hf_token: str | bool | None = None):
    from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

    dtype = resolve_dtype(config.dtype, config.device)
    try:
        text_processor = AutoProcessor.from_pretrained(config.text_encoder_id, token=hf_token)
    except Exception:
        text_processor = AutoTokenizer.from_pretrained(config.text_encoder_id, token=hf_token)
    if hasattr(text_processor, "padding_side"):
        text_processor.padding_side = "right"
    elif hasattr(text_processor, "tokenizer"):
        text_processor.tokenizer.padding_side = "right"

    model_kwargs: dict[str, Any] = {"token": hf_token}
    if config.text_attn_implementation:
        model_kwargs["attn_implementation"] = config.text_attn_implementation
    try:
        model = AutoModelForCausalLM.from_pretrained(
            config.text_encoder_id,
            torch_dtype=dtype,
            **model_kwargs,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            config.text_encoder_id,
            dtype=dtype,
            **model_kwargs,
        )
    if _uses_sana_gemma_encoding(config.text_encoder_id) and hasattr(model, "get_decoder"):
        model = model.get_decoder()
    model = model.to(config.device).eval()
    model.requires_grad_(False)
    return text_processor, model


def vae_scaling_factor(vae: Any, fallback: float = DEFAULT_SCALING_FACTOR) -> float:
    config = getattr(vae, "config", None)
    value = getattr(config, "scaling_factor", None)
    return float(value) if value else fallback


def extract_latent(encoded: Any) -> torch.Tensor:
    if isinstance(encoded, tuple):
        return encoded[0]
    if hasattr(encoded, "latent"):
        return encoded.latent
    if hasattr(encoded, "latents"):
        return encoded.latents
    if hasattr(encoded, "latent_dist"):
        return encoded.latent_dist.mode()
    raise TypeError(f"Unsupported VAE encode output: {type(encoded)!r}")


@torch.inference_mode()
def encode_image_latent(vae: Any, image_tensor: torch.Tensor, config: CacheConfig) -> torch.Tensor:
    dtype = resolve_dtype(config.dtype, config.device)
    image_tensor = image_tensor.to(device=config.device, dtype=dtype, non_blocking=True)
    encoded = vae.encode(image_tensor)
    latent = extract_latent(encoded)
    return latent * vae_scaling_factor(vae, config.scaling_factor)


@torch.inference_mode()
def decode_image_latent(vae: Any, latent: torch.Tensor, config: CacheConfig) -> torch.Tensor:
    dtype = resolve_dtype(config.dtype, config.device)
    latent = latent.to(device=config.device, dtype=dtype)
    latent = latent / vae_scaling_factor(vae, config.scaling_factor)
    decoded = vae.decode(latent, return_dict=False)
    image = decoded[0] if isinstance(decoded, tuple) else decoded.sample
    return image


@torch.inference_mode()
def encode_captions(
    text_processor: Any,
    text_encoder: Any,
    captions: list[str],
    config: CacheConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not captions:
        raise ValueError("At least one caption is required.")
    token_kwargs = {
        "max_length": config.max_length,
        "padding": "max_length",
        "truncation": True,
        "return_tensors": "pt",
    }
    try:
        tokens = text_processor(text=captions, **token_kwargs)
    except TypeError:
        tokens = text_processor(captions, **token_kwargs)

    tokens = {key: value.to(config.device) for key, value in tokens.items() if torch.is_tensor(value)}
    attention_mask = tokens["attention_mask"]

    if _uses_sana_gemma_encoding(config.text_encoder_id):
        select_index = gemma_token_select_index(config.max_length)
        output = text_encoder(tokens["input_ids"], attention_mask=attention_mask)
        if isinstance(output, tuple):
            hidden = output[0]
        elif hasattr(output, "last_hidden_state"):
            hidden = output.last_hidden_state
        else:
            hidden = output
        hidden = hidden[:, select_index, :]
        attention_mask = attention_mask[:, select_index]
        return hidden, attention_mask

    model_inputs = {
        key: value
        for key, value in tokens.items()
        if key in {"input_ids", "attention_mask", "position_ids", "token_type_ids"}
    }
    output = text_encoder(**model_inputs, output_hidden_states=True, use_cache=False)
    if hasattr(output, "hidden_states") and output.hidden_states is not None:
        hidden = output.hidden_states[-1]
    elif hasattr(output, "last_hidden_state"):
        hidden = output.last_hidden_state
    elif isinstance(output, tuple):
        hidden = output[0]
    else:
        raise TypeError(f"Unsupported text encoder output: {type(output)!r}")
    return hidden, attention_mask


@torch.inference_mode()
def encode_caption(text_processor: Any, text_encoder: Any, caption: str, config: CacheConfig) -> tuple[torch.Tensor, torch.Tensor]:
    return encode_captions(text_processor, text_encoder, [caption], config)


def make_cache_record(
    image_latent: torch.Tensor,
    text_embedding: torch.Tensor,
    attention_mask: torch.Tensor,
    caption: str,
    config: CacheConfig,
    source_image: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "image_latent": image_latent.detach().cpu(),
        "text_embedding": text_embedding.detach().cpu(),
        "attention_mask": attention_mask.detach().cpu().to(torch.int16),
        "caption": caption,
        "metadata": {
            **asdict(config),
            "source_image": source_image,
            "image_latent_shape": tuple(image_latent.shape),
            "text_embedding_shape": tuple(text_embedding.shape),
        },
    }


def save_cache_record(record: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(record, path)
    metadata_path = path.with_suffix(path.suffix + ".json")
    metadata_path.write_text(json.dumps(record["metadata"], indent=2), encoding="utf-8")


def load_cache_record(path: str | Path, map_location: str = "cpu") -> dict[str, Any]:
    from boomer.checkpoint_utils import load_torch_file

    return load_torch_file(path, map_location=map_location)


def encode_one(image_path: str | Path, caption: str, config: CacheConfig, hf_token: str | bool | None = None) -> dict[str, Any]:
    image_path = Path(image_path)
    vae = load_vae(config, hf_token=hf_token)
    text_processor, text_encoder = load_text_encoder(config, hf_token=hf_token)

    with Image.open(image_path) as image:
        image_tensor = image_to_tensor(image, config.resolution)
    image_latent = encode_image_latent(vae, image_tensor, config)
    text_embedding, attention_mask = encode_caption(text_processor, text_encoder, caption, config)

    return make_cache_record(
        image_latent=image_latent,
        text_embedding=text_embedding,
        attention_mask=attention_mask,
        caption=caption,
        config=config,
        source_image=str(image_path),
    )


def run_smoke_test(args: argparse.Namespace) -> None:
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    config = CacheConfig(
        resolution=args.resolution,
        vae_id=args.vae_id,
        text_encoder_id=args.text_encoder_id,
        max_length=args.max_length,
        dtype=args.dtype,
        device=device,
        scaling_factor=args.scaling_factor,
    )
    hf_token = hf_token_value(args.hf_token)

    vae = load_vae(config, hf_token=hf_token)
    text_processor, text_encoder = load_text_encoder(config, hf_token=hf_token)

    with Image.open(args.image) as image:
        image_tensor = image_to_tensor(image, config.resolution)
    image_latent = encode_image_latent(vae, image_tensor, config)
    text_embedding, attention_mask = encode_caption(text_processor, text_encoder, args.caption, config)
    record = make_cache_record(
        image_latent=image_latent,
        text_embedding=text_embedding,
        attention_mask=attention_mask,
        caption=args.caption,
        config=config,
        source_image=args.image,
    )
    save_cache_record(record, args.out)

    if args.decoded_out:
        reloaded = load_cache_record(args.out)
        decoded = decode_image_latent(vae, reloaded["image_latent"], config)
        image = tensor_to_image(decoded)
        decoded_path = Path(args.decoded_out)
        decoded_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(decoded_path)

    print(f"saved cache record: {args.out}")
    if args.decoded_out:
        print(f"saved reconstruction: {args.decoded_out}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create one Sana-compatible Boomer latent cache record.")
    parser.add_argument("--image", required=True, help="Path to an input image.")
    parser.add_argument("--caption", required=True, help="Caption for the input image.")
    parser.add_argument("--out", required=True, help="Output .pt cache record path.")
    parser.add_argument("--decoded-out", help="Optional decoded reconstruction image path.")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--vae-id", default=DEFAULT_VAE_ID)
    parser.add_argument("--text-encoder-id", default=DEFAULT_TEXT_ENCODER_ID)
    parser.add_argument("--max-length", type=int, default=CacheConfig.max_length)
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--scaling-factor", type=float, default=DEFAULT_SCALING_FACTOR)
    parser.add_argument("--hf-token", help="Hugging Face token, or 'true' to use the logged-in token.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    run_smoke_test(parser.parse_args())


if __name__ == "__main__":
    main()
