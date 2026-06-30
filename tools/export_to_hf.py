"""Export a Boomer Lightning checkpoint to a HuggingFace diffusers-custom-pipeline repo.

Produces the exact layout ``BoomerPipeline`` expects:

    <out>/
      transformer/config.json
      transformer/diffusion_pytorch_model.safetensors
      pipeline_boomer.py
      modeling_boomer_fla.py
      STORKScheduler.py        (+ STORK_constants/)
      model_index.json
      README.md

What it does to the weights, and why:
  * EMA weights by default (``select_model_state(..., "ema")``).
  * Strips the ``_orig_mod.`` (torch.compile) and Lightning ``model.`` prefixes.
  * Drops the training-only REPA head (``repa_proj.*``) — not in the denoise path.
  * Folds the FLUX-Fill-style reserved input channels: for t2i those extra channels are
    always zero AND ``x_embedder.weight[:, latent_channels:]`` was zero-initialised and
    never received gradient, so slicing to the first ``latent_channels`` columns is exact.
    The published model is therefore a clean ``latent_channels``-in t2i transformer.
  * Writes the Qwen readout / DC-AE / sampler metadata the pipeline reads at run time
    (``_te_repo``, ``_instruction``, ``_flow_shift``, ``_scaling_factor``, ...).

The round-trip check (loads the written transformer back through ``BoomerFLADiT`` and
asserts ``missing == unexpected == 0``) needs ``fla`` installed, so run this in the same
environment you train in (Colab). Upload is OFF unless you pass ``--push``.

Example (Colab):
    python tools/export_to_hf.py \
      --checkpoint /content/ckpt_boomer-v4-1024-repa-5mix-b24run3/boomer-step=100000.ckpt \
      --out /content/Linarix-v1 --weights ema \
      --repo-id Akrao9/Linarix-v1 --stork-dir /content/STORK
      # add --push to upload (public repo) once the round-trip + smoke gen look good
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from boomer.checkpoint_utils import load_checkpoint, select_model_state  # noqa: E402
from boomer.models.qwen_connector import DEFAULT_INSTRUCTION  # noqa: E402
from modeling_boomer_fla import BoomerFLADiT, BoomerFLADiTConfig  # noqa: E402


def _strip_prefixes(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Drop torch.compile / Lightning wrappers so keys match the bare BoomerFLADiT."""
    out: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        clean = key.replace("_orig_mod.", "")
        if clean.startswith("model."):
            clean = clean[len("model.") :]
        out[clean] = value
    return out


def _fold_extra_input_channels(
    state: dict[str, torch.Tensor], latent_channels: int, extra: int
) -> dict[str, torch.Tensor]:
    """Slice the inpaint-reserved (always-zero) extra channels off x_embedder."""
    if extra <= 0:
        return state
    weight = state.get("x_embedder.weight")
    if weight is None:
        return state
    in_ch = weight.shape[1]
    if in_ch == latent_channels:
        return state  # already folded
    if in_ch != latent_channels + extra:
        raise SystemExit(
            f"x_embedder in_channels={in_ch} != latent_channels+extra="
            f"{latent_channels + extra}; refusing to guess the fold."
        )
    folded = dict(state)
    # Linear: [hidden, in_ch]; Conv2d: [hidden, in_ch, k, k]. Slice dim=1 either way.
    folded["x_embedder.weight"] = weight[:, :latent_channels].contiguous().clone()
    nonzero = float(weight[:, latent_channels:].abs().max())
    print(f"[fold] x_embedder {in_ch}->{latent_channels} input channels "
          f"(dropped columns abs-max={nonzero:.3e}; expected ~0)")
    return folded


def _build_config(train_cfg: dict) -> dict:
    """Map the training config onto BoomerFLADiTConfig fields (only the known ones)."""
    config: dict = {}
    for field in dataclasses.fields(BoomerFLADiTConfig):
        config[field.name] = train_cfg.get(field.name, field.default)
    # The standalone has no FA2 path; SDPA is numerically identical and dependency-free.
    config["image_attention_backend"] = "sdpa"
    return config


def _copy_stork(out_dir: Path, stork_dir: Path | None) -> None:
    candidates = [stork_dir] if stork_dir else []
    candidates += [Path("/content/STORK"), REPO_ROOT.parent / "STORK", REPO_ROOT / "STORK"]
    for cand in candidates:
        if cand and (cand / "STORKScheduler.py").exists():
            shutil.copy2(cand / "STORKScheduler.py", out_dir / "STORKScheduler.py")
            if (cand / "STORK_constants").is_dir():
                shutil.copytree(cand / "STORK_constants", out_dir / "STORK_constants",
                                dirs_exist_ok=True)
            print(f"[stork] bundled STORKScheduler.py + STORK_constants from {cand}")
            return
    print("[stork] WARNING: STORKScheduler.py / STORK_constants not found — the STORK "
          "sampler will not load from this repo. Pass --stork-dir explicitly.")


def _write_model_card(out_dir: Path, model_name: str, repo_id: str, config: dict) -> None:
    card = f"""---
license: apache-2.0
library_name: diffusers
pipeline_tag: text-to-image
tags:
- text-to-image
- diffusion
- flow-matching
- linear-attention
- boomer
---

# {model_name}

A {config['hidden_dim']}-dim / {config['depth']}-layer flow-matching text-to-image DiT
(GDN-2 linear-attention mixer with periodic full image attention), DC-AE f32c32 latents,
Qwen text conditioning. Renders at {config['latent_size'] * 32}px (latent {config['latent_size']}x{config['latent_size']}).

```python
import torch
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    "{repo_id}",
    custom_pipeline="pipeline_boomer",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
).to("cuda")

image = pipe("a lighthouse on a rocky cliff above crashing waves", cfg_scale=3.5)[0]
image.save("out.png")
```

Requires: `torch diffusers transformers accelerate safetensors` and
`flash-linear-attention` (`pip install git+https://github.com/fla-org/flash-linear-attention.git`).
The STORK scheduler is bundled (Tan et al. 2025, arXiv:2505.24210).
"""
    (out_dir / "README.md").write_text(card)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a Boomer checkpoint to a HF pipeline repo.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True, help="Local output dir for the repo.")
    parser.add_argument("--weights", default="ema", choices=["ema", "auto", "model"])
    parser.add_argument("--repo-id", default="Akrao9/Linarix-v1")
    parser.add_argument("--model-name", default="Linarix-v1")
    parser.add_argument("--te-repo", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--vae-repo", default="mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers")
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--max-text-length", type=int, default=320)
    parser.add_argument("--flow-shift", type=float, default=3.0)
    parser.add_argument("--scaling-factor", type=float, default=0.41407)
    parser.add_argument("--latent-mean", type=float, default=0.0)
    parser.add_argument("--latent-std", type=float, default=1.0)
    parser.add_argument("--stork-dir", default=None, help="Dir holding STORKScheduler.py + STORK_constants/.")
    parser.add_argument("--save-dtype", default="keep", choices=["keep", "bf16", "fp32"])
    parser.add_argument("--push", action="store_true", help="Upload to --repo-id (off by default).")
    parser.add_argument("--private", action="store_true", help="Create the repo private (with --push).")
    parser.add_argument("--hf-token", default=None)
    args = parser.parse_args()

    out_dir = Path(args.out)
    tf_dir = out_dir / "transformer"
    tf_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] {args.checkpoint}")
    ckpt = load_checkpoint(args.checkpoint)
    train_cfg = ckpt.get("config") or {}
    if not train_cfg:
        raise SystemExit("Checkpoint has no embedded 'config'; cannot build the model config.")

    state, tag = select_model_state(ckpt, args.weights)
    print(f"[weights] using '{tag}' weights ({len(state)} tensors)")

    state = _strip_prefixes(state)
    state = {k: v for k, v in state.items() if not k.startswith("repa_proj.")}

    latent_channels = int(train_cfg.get("latent_channels", 32))
    extra = int(train_cfg.get("extra_input_channels", 0))
    state = _fold_extra_input_channels(state, latent_channels, extra)

    if args.save_dtype != "keep":
        target = torch.bfloat16 if args.save_dtype == "bf16" else torch.float32
        state = {k: v.to(target) if v.is_floating_point() else v for k, v in state.items()}
    state = {k: v.cpu().contiguous() for k, v in state.items()}

    config = _build_config(train_cfg)

    # ── round-trip verify: the written architecture must accept the weights exactly ──
    model = BoomerFLADiT(BoomerFLADiTConfig(**config))
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [m for m in missing if "rope" not in m and "inv_freq" not in m]  # non-persistent buffers
    print(f"[verify] round-trip load: missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("  missing[:10]   :", missing[:10])
    if unexpected:
        print("  unexpected[:10]:", unexpected[:10])
    if missing or unexpected:
        raise SystemExit("Round-trip mismatch — refusing to export a broken model. See lists above.")
    print("[verify] OK — all denoise-path weights map 1:1.")

    # ── write transformer/ ──
    config_json = dict(config)
    config_json.update({
        "_latent_mean": args.latent_mean,
        "_latent_std": args.latent_std,
        "_scaling_factor": args.scaling_factor,
        "_flow_shift": args.flow_shift,
        "_max_text_length": args.max_text_length,
        "_vae_repo": args.vae_repo,
        "_te_repo": args.te_repo,
        "_instruction": args.instruction,
    })
    (tf_dir / "config.json").write_text(json.dumps(config_json, indent=2))

    from safetensors.torch import save_file
    save_file(state, str(tf_dir / "diffusion_pytorch_model.safetensors"),
              metadata={"format": "pt"})
    print(f"[write] transformer/ ({len(state)} tensors, {args.save_dtype} dtype)")

    # ── copy code + scheduler + card ──
    for fname in ("pipeline_boomer.py", "modeling_boomer_fla.py"):
        shutil.copy2(REPO_ROOT / fname, out_dir / fname)
    _copy_stork(out_dir, Path(args.stork_dir) if args.stork_dir else None)
    (out_dir / "model_index.json").write_text(json.dumps({
        "_class_name": "BoomerPipeline",
        "custom_pipeline": "pipeline_boomer",
    }, indent=2))
    _write_model_card(out_dir, args.model_name, args.repo_id, config)
    print(f"[done] repo assembled at {out_dir}")

    if not args.push:
        print("\n[push] skipped (no --push). Verify locally, then re-run with --push to upload.")
        return

    from huggingface_hub import HfApi
    api = HfApi(token=args.hf_token)
    api.create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    api.upload_folder(folder_path=str(out_dir), repo_id=args.repo_id, repo_type="model")
    print(f"[push] uploaded -> https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
