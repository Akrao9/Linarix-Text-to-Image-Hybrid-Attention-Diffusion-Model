"""Inspect a Hugging Face dataset example before caching."""

from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import Any

from PIL import Image


def describe_value(value: Any) -> str:
    if isinstance(value, Image.Image):
        return f"PIL.Image size={value.size} mode={value.mode}"
    if isinstance(value, bytes):
        return f"bytes len={len(value)}"
    text = str(value)
    if len(text) > 300:
        text = text[:300] + "..."
    return f"{type(value).__name__}: {text}"


def expand_data_files(patterns: list[str]) -> list[str]:
    files: list[str] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        files.extend(matches if matches else [pattern])
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect one example from a Hugging Face dataset.")
    parser.add_argument("--dataset", default="BLIP3o/BLIP3o-Pretrain-JourneyDB")
    parser.add_argument("--subset", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--data-files", nargs="+", help="Local/remote data files or glob patterns, e.g. '/data/*.tar'.")
    parser.add_argument("--cache-dir", help="Hugging Face datasets cache directory.")
    parser.add_argument("--num-proc", type=int, help="HF Datasets preprocessing workers for non-streaming loads.")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--hf-token", help="Hugging Face token, or 'true' to use the logged-in token.")
    parser.add_argument("--save-image", help="Optional path to save the first PIL image column.")
    args = parser.parse_args()

    from datasets import load_dataset

    token = True if args.hf_token and args.hf_token.lower() in {"1", "true", "yes"} else args.hf_token
    load_kwargs = {
        "split": args.split,
        "streaming": args.streaming,
        "token": token,
    }
    if args.cache_dir:
        load_kwargs["cache_dir"] = args.cache_dir
    if args.data_files:
        load_kwargs["data_files"] = expand_data_files(args.data_files)
    if args.num_proc and not args.streaming:
        load_kwargs["num_proc"] = args.num_proc
    dataset = load_dataset(args.dataset, args.subset, **load_kwargs)

    if args.streaming:
        iterator = iter(dataset)
        example = None
        for _ in range(args.index + 1):
            example = next(iterator)
    else:
        example = dataset[args.index]

    assert example is not None
    print(f"dataset={args.dataset} split={args.split} streaming={args.streaming}")
    print(f"keys={list(example.keys())}")
    for key, value in example.items():
        print(f"{key}: {describe_value(value)}")
        if args.save_image and isinstance(value, Image.Image):
            path = Path(args.save_image)
            path.parent.mkdir(parents=True, exist_ok=True)
            value.save(path)
            print(f"saved image column '{key}' to {path}")
            args.save_image = None


if __name__ == "__main__":
    main()
