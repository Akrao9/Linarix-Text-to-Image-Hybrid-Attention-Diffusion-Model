"""Cache a small Hugging Face image-text dataset slice into Boomer records."""

from __future__ import annotations

import argparse
import glob
import hashlib
import sys
from io import BytesIO
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from boomer.sana_latent_cache import (  # noqa: E402
    CacheConfig,
    encode_caption,
    encode_image_latent,
    hf_token_value,
    image_to_tensor,
    load_text_encoder,
    load_vae,
    make_cache_record,
    save_cache_record,
)

try:
    from boomer.sana_latent_cache import encode_captions  # type: ignore[attr-defined]  # noqa: E402
except ImportError:

    def encode_captions(text_processor: Any, text_encoder: Any, captions: list[str], config: CacheConfig):
        embeddings = []
        masks = []
        for caption in captions:
            text_embedding, attention_mask = encode_caption(text_processor, text_encoder, caption, config)
            embeddings.append(text_embedding)
            masks.append(attention_mask)
        return torch.cat(embeddings, dim=0), torch.cat(masks, dim=0)


def first_present(example: dict[str, Any], candidates: list[str]) -> Any:
    for key in candidates:
        if key in example and example[key] is not None:
            return example[key]
    return None


def normalize_caption(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        caption = value.strip()
        return caption or None
    if isinstance(value, list) and value:
        return normalize_caption(value[0])
    if isinstance(value, dict):
        for key in ("caption", "text", "prompt"):
            if key in value:
                return normalize_caption(value[key])
    return str(value).strip() or None


def normalize_image(value: Any) -> Image.Image | None:
    if value is None:
        return None
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, bytes):
        with Image.open(BytesIO(value)) as image:
            return image.convert("RGB")
    if isinstance(value, dict):
        for key in ("image", "jpg", "png", "bytes", "path"):
            if key in value:
                return normalize_image(value[key])
    if isinstance(value, (str, Path)):
        path = Path(value)
        if path.exists():
            with Image.open(path) as image:
                return image.convert("RGB")
    return None


def disable_hf_image_decoding(dataset: Any, image_columns: list[str]) -> Any:
    """Return dataset with HF Image columns left as bytes/path dictionaries."""
    try:
        from datasets import Image as DatasetImage
    except Exception:
        return dataset

    features = getattr(dataset, "features", None)
    if not features:
        return dataset

    for column in image_columns:
        if column not in features:
            continue
        try:
            dataset = dataset.cast_column(column, DatasetImage(decode=False))
            print(f"disabled HF image decoding for column: {column}")
        except Exception as exc:
            print(f"[warn] could not disable HF image decoding for {column}: {type(exc).__name__}: {exc}")
    return dataset


def stable_id(text: str, index: int) -> str:
    digest = hashlib.sha1(f"{index}:{text}".encode("utf-8")).hexdigest()[:12]
    return f"{index:08d}_{digest}"


def cache_dataset(args: argparse.Namespace) -> None:
    from datasets import load_dataset

    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")

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
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    load_kwargs = {
        "split": args.split,
        "streaming": args.streaming,
        "token": hf_token,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.cache_dir:
        load_kwargs["cache_dir"] = args.cache_dir
    if args.data_files:
        data_files = expand_data_files(args.data_files)
        if not data_files:
            raise FileNotFoundError(f"No files matched --data-files: {args.data_files}")
        load_kwargs["data_files"] = data_files
    if args.num_proc and not args.streaming:
        load_kwargs["num_proc"] = args.num_proc

    dataset = load_dataset(args.dataset, args.subset, **load_kwargs)
    dataset = disable_hf_image_decoding(dataset, args.image_columns)

    vae = load_vae(config, hf_token=hf_token)
    text_processor, text_encoder = load_text_encoder(config, hf_token=hf_token)

    written = 0
    skipped = 0
    existing_skipped = 0
    pending: list[dict[str, Any]] = []

    def flush_pending() -> None:
        nonlocal written, skipped, pending
        if not pending:
            return

        current = pending
        pending = []
        try:
            image_tensor = torch.cat([item["image_tensor"] for item in current], dim=0)
            captions = [item["caption"] for item in current]
            image_latents = encode_image_latent(vae, image_tensor, config)
            text_embeddings, attention_masks = encode_captions(text_processor, text_encoder, captions, config)
        except Exception as exc:
            print(f"[skip batch size={len(current)}] {type(exc).__name__}: {exc}")
            if len(current) > 1:
                print(f"[retry singly] batch size={len(current)}")
                for item in current:
                    pending = [item]
                    flush_pending()
                return
            skipped += 1
            return

        for item_index, item in enumerate(current):
            try:
                record = make_cache_record(
                    image_latent=image_latents[item_index : item_index + 1],
                    text_embedding=text_embeddings[item_index : item_index + 1],
                    attention_mask=attention_masks[item_index : item_index + 1],
                    caption=item["caption"],
                    config=config,
                    source_image=f"{args.dataset}:{args.split}:{item['index']}",
                )
                record["metadata"].update(
                    {
                        "dataset": args.dataset,
                        "subset": args.subset,
                        "split": args.split,
                        "index": item["index"],
                        "cache_id": item["cache_id"],
                    }
                )
                save_cache_record(record, item["out_path"])
                written += 1
                print(f"[{written}/{args.limit}] wrote {item['out_path']}")
            except Exception as exc:
                skipped += 1
                print(f"[skip index={item['index']}] {type(exc).__name__}: {exc}")

    index = -1
    dataset_iter = iter(dataset)
    while written < args.limit:
        index += 1
        try:
            example = next(dataset_iter)
        except StopIteration:
            break
        except Exception as exc:
            skipped += 1
            print(f"[skip stream index={index}] {type(exc).__name__}: {exc}")
            continue

        if args.offset and index < args.offset:
            continue
        if written >= args.limit:
            flush_pending()
            break

        try:
            image_value = first_present(example, args.image_columns)
            caption_value = first_present(example, args.caption_columns)
            image = normalize_image(image_value)
            caption = normalize_caption(caption_value)
        except Exception as exc:
            skipped += 1
            print(f"[skip index={index}] {type(exc).__name__}: {exc}")
            continue
        if image is None or caption is None:
            skipped += 1
            continue

        cache_id = stable_id(caption, index)
        out_path = out_dir / f"{cache_id}.pt"
        if out_path.exists() and not args.overwrite:
            existing_skipped += 1
            continue

        try:
            image_tensor = image_to_tensor(image, config.resolution)
            pending.append(
                {
                    "image_tensor": image_tensor,
                    "caption": caption,
                    "index": index,
                    "cache_id": cache_id,
                    "out_path": out_path,
                }
            )
            if len(pending) >= args.batch_size or written + len(pending) >= args.limit:
                flush_pending()
        except Exception as exc:
            skipped += 1
            print(f"[skip index={index}] {type(exc).__name__}: {exc}")

    flush_pending()
    print(f"done: wrote={written} skipped={skipped} existing_skipped={existing_skipped} out_dir={out_dir}")


def split_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def expand_data_files(patterns: list[str]) -> list[str]:
    files: list[str] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        files.extend(matches if matches else [pattern])
    return files


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cache a HF image-text dataset slice into Boomer .pt records.")
    parser.add_argument("--dataset", default="BLIP3o/BLIP3o-Pretrain-JourneyDB")
    parser.add_argument("--subset", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--data-files", nargs="+", help="Local/remote data files or glob patterns, e.g. '/data/*.tar'.")
    parser.add_argument("--cache-dir", help="Hugging Face datasets cache directory.")
    parser.add_argument("--num-proc", type=int, help="HF Datasets preprocessing workers for non-streaming loads.")
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1, help="Encoder batch size for VAE/Gemma caching.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--image-columns", type=split_csv, default=["jpg", "image", "png"])
    parser.add_argument(
        "--caption-columns",
        type=split_csv,
        default=["txt", "text", "prompt", "caption", "original_prompt", "generated_text"],
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--vae-id", default=CacheConfig.vae_id)
    parser.add_argument("--text-encoder-id", default=CacheConfig.text_encoder_id)
    parser.add_argument("--max-length", type=int, default=CacheConfig.max_length)
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--scaling-factor", type=float, default=CacheConfig.scaling_factor)
    parser.add_argument("--hf-token", help="Hugging Face token, or 'true' to use the logged-in token.")
    return parser


def main() -> None:
    cache_dataset(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
