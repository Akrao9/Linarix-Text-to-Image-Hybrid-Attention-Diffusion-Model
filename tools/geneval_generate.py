"""Generate the GenEval image set from an exported Boomer/Linarix HF repo.

GenEval (Ghosh et al. 2023, arXiv:2310.11513) scores compositional prompt following --
object presence, counting, colors, position, attribute binding -- by running an object
detector over generated images. This script produces only the images, in the exact
layout ``evaluation/evaluate_images.py`` expects:

    <outdir>/
      00000/
        metadata.jsonl        (line N of evaluation_metadata.jsonl, verbatim)
        samples/0000.png ... 0003.png
      00001/
        ...

Scoring runs separately and needs mmdet 2.x, which does not coexist with a modern
torch -- see the module docstring note at the bottom of this file.

One pipeline call per prompt, with ``seed = --seed + prompt_index``. Batching several
prompts into one call would be faster, but then a prompt's samples would depend on how
the batch happened to be grouped, so a rerun with a different ``--prompt-limit`` or a
resumed run would not reproduce earlier images. A benchmark should be reproducible
row-by-row, so the batch is exactly one prompt's ``--n-samples`` copies.

Sampler settings are deliberately NOT passed: the exported checkpoint carries its own
defaults, so the benchmark measures what someone downloading the model actually gets.
Override them only to answer a specific question.

Example:
    python tools/geneval_generate.py \
      --metadata /content/geneval/prompts/evaluation_metadata.jsonl \
      --model Akrao9/Linarix-v2 \
      --outdir /content/geneval_images
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch


def _load_pipeline(model: str, dtype: torch.dtype, token: str | None):
    from diffusers import DiffusionPipeline

    pipe = DiffusionPipeline.from_pretrained(
        model,
        custom_pipeline="pipeline_boomer",
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    # The pipeline lazy-loads the VAE and text encoder and needs the token for a gated
    # repo; DiffusionPipeline.from_pretrained does not route through the override that
    # would otherwise set it.
    pipe._hf_token = token
    return pipe.to("cuda" if torch.cuda.is_available() else "cpu")


def _sampler_overrides(args: argparse.Namespace) -> dict:
    """Only the flags the caller actually set; everything else stays at export defaults."""
    names = (
        "steps", "cfg_scale", "cfg_rescale", "sampler",
        "derivative_order", "substeps", "cfg_interval_low", "cfg_interval_high",
    )
    return {name: getattr(args, name) for name in names if getattr(args, name) is not None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--metadata", required=True,
                        help="GenEval evaluation_metadata.jsonl (553 prompts).")
    parser.add_argument("--model", default="Akrao9/Linarix-v2",
                        help="HF repo id or a local exported directory.")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--n-samples", type=int, default=4,
                        help="Images per prompt. GenEval's published numbers use 4.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Base seed; prompt N uses --seed + N.")
    parser.add_argument("--prompt-limit", type=int, default=None,
                        help="Only the first N prompts, for a smoke test.")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--overwrite", action="store_true",
                        help="Regenerate prompts that already have all their samples.")
    # All default to None so the exported checkpoint's own settings apply.
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--cfg-scale", type=float, default=None)
    parser.add_argument("--cfg-rescale", type=float, default=None)
    parser.add_argument("--sampler", default=None, choices=["stork2", "stork4"])
    parser.add_argument("--derivative-order", type=int, default=None)
    parser.add_argument("--substeps", type=int, default=None)
    parser.add_argument("--cfg-interval-low", type=float, default=None)
    parser.add_argument("--cfg-interval-high", type=float, default=None)
    args = parser.parse_args()

    metadata_path = Path(args.metadata)
    if not metadata_path.is_file():
        raise SystemExit(f"metadata file not found: {metadata_path}")
    lines = [line for line in metadata_path.read_text().splitlines() if line.strip()]
    if args.prompt_limit is not None:
        lines = lines[: args.prompt_limit]
    if not lines:
        raise SystemExit("metadata file has no prompts")

    out_root = Path(args.outdir)
    out_root.mkdir(parents=True, exist_ok=True)

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    overrides = _sampler_overrides(args)

    print(f"[load] {args.model}")
    pipe = _load_pipeline(args.model, dtype, args.hf_token)
    print(
        f"[defaults] {pipe.default_sampler} steps={pipe.default_steps} "
        f"substeps={pipe.default_stork_substeps} cfg={pipe.default_cfg_scale}/"
        f"{pipe.default_cfg_rescale} interval={pipe.default_cfg_interval} "
        f"flow_shift={pipe.flow_shift}"
    )
    if overrides:
        print(f"[override] {overrides}")

    started = time.perf_counter()
    generated = 0
    skipped = 0
    for index, line in enumerate(lines):
        prompt_dir = out_root / f"{index:05d}"
        samples_dir = prompt_dir / "samples"
        expected = [samples_dir / f"{k:04d}.png" for k in range(args.n_samples)]
        if not args.overwrite and all(path.is_file() for path in expected):
            skipped += 1
            continue

        samples_dir.mkdir(parents=True, exist_ok=True)
        # Verbatim line, as evaluate_images.py re-parses it for the scoring rules.
        (prompt_dir / "metadata.jsonl").write_text(line + "\n")
        prompt = json.loads(line)["prompt"]

        images = pipe(
            [prompt] * args.n_samples,
            seed=args.seed + index,
            **overrides,
        )
        for path, image in zip(expected, images):
            image.save(path)
        generated += args.n_samples

        done = index + 1
        elapsed = time.perf_counter() - started
        rate = generated / elapsed if generated else 0.0
        remaining = (len(lines) - done) * args.n_samples / rate if rate else 0.0
        print(
            f"[{done:4d}/{len(lines)}] {rate:5.2f} img/s | "
            f"eta {remaining / 60:5.1f} min | {prompt[:58]}",
            flush=True,
        )

    print(
        f"[done] {generated} images written, {skipped} prompts skipped (already complete) "
        f"-> {out_root}"
    )
    print(
        "\nNext: score with GenEval's own environment (mmdet 2.x, separate from this one):\n"
        f"  python evaluation/evaluate_images.py {out_root} \\\n"
        "      --outfile results.jsonl --model-path <OBJECT_DETECTOR_FOLDER>\n"
        "  python evaluation/summary_scores.py results.jsonl"
    )


if __name__ == "__main__":
    main()
