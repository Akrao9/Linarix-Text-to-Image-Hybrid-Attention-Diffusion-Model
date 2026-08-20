"""Export a Boomer Lightning checkpoint to a HuggingFace diffusers-custom-pipeline repo.

Produces the exact layout ``BoomerPipeline`` expects:

    <out>/
      transformer/config.json
      transformer/model.safetensors
      pipeline_boomer.py
      modeling_boomer_fla.py
      qwen_connector.py        (same implementation used by training)
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from boomer.checkpoint_utils import load_checkpoint, select_model_state  # noqa: E402
from boomer.models.boomer_fla_dit import migrate_glumbconv_state_dict  # noqa: E402
from boomer.models.qwen_connector import DEFAULT_INSTRUCTION  # noqa: E402
from boomer.showcase_prompts import SHOWCASE_PROMPTS  # noqa: E402
from modeling_boomer_fla import BoomerFLADiT, BoomerFLADiTConfig  # noqa: E402

FLA_PIN = "3c4c54ae7397d37130d7101edd0f4eb596af896d"


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
    """Slice the inpaint-reserved (always-zero) extra channels off every patch embedder.

    A DDT split (``ddt_decoder_depth > 0``) gives the velocity decoder its own
    ``decoder_x_embedder``, and the model zero-inits the reserved columns on both
    (boomer_fla_dit.py:2557-2559). Folding only the encoder's leaves the decoder at
    latent_channels+extra and the strict round-trip fails on shape.
    """
    if extra <= 0:
        return state
    folded = dict(state)
    for name in ("x_embedder.weight", "decoder_x_embedder.weight"):
        weight = state.get(name)
        if weight is None:
            continue  # dense (non-DDT) checkpoints have no decoder embedder
        in_ch = weight.shape[1]
        if in_ch == latent_channels:
            continue  # already folded
        if in_ch != latent_channels + extra:
            raise SystemExit(
                f"{name} in_channels={in_ch} != latent_channels+extra="
                f"{latent_channels + extra}; refusing to guess the fold."
            )
        # Linear: [hidden, in_ch]; Conv2d: [hidden, in_ch, k, k]. Slice dim=1 either way.
        folded[name] = weight[:, :latent_channels].contiguous().clone()
        nonzero = float(weight[:, latent_channels:].abs().max())
        print(f"[fold] {name.removesuffix('.weight')} {in_ch}->{latent_channels} input "
              f"channels (dropped columns abs-max={nonzero:.3e}; expected ~0)")
    return folded


def _build_config(train_cfg: dict) -> dict:
    """Map the training config onto BoomerFLADiTConfig fields (only the known ones)."""
    config: dict = {}
    for field in dataclasses.fields(BoomerFLADiTConfig):
        config[field.name] = train_cfg.get(field.name, field.default)
    # The standalone has no FA2 path; SDPA is numerically identical and dependency-free.
    config["image_attention_backend"] = "sdpa"
    # Inference-only overrides: the exported weights have the inpaint-reserved
    # channels folded off x_embedder and repa_proj stripped, and checkpointing is a
    # training concern — the modeling config now carries these fields, so they must
    # not leak through from the training config.
    config["extra_input_channels"] = 0
    config["repa_dim"] = 0
    config["gradient_checkpointing"] = False
    return config


def _copy_stork(out_dir: Path, stork_dir: Path | None) -> None:
    candidates = [stork_dir] if stork_dir else []
    candidates += [Path("/content/STORK"), REPO_ROOT.parent / "STORK", REPO_ROOT / "STORK"]
    for cand in candidates:
        if (
            cand
            and (cand / "STORKScheduler.py").is_file()
            and (cand / "STORK_constants").is_dir()
        ):
            shutil.copy2(cand / "STORKScheduler.py", out_dir / "STORKScheduler.py")
            if (cand / "STORK_constants").is_dir():
                shutil.copytree(cand / "STORK_constants", out_dir / "STORK_constants",
                                dirs_exist_ok=True)
            print(f"[stork] bundled STORKScheduler.py + STORK_constants from {cand}")
            return
    raise SystemExit(
        "STORKScheduler.py / STORK_constants not found; refusing to create an export "
        "whose default sampler cannot load. Pass --stork-dir explicitly."
    )


def _parse_layers(value: Any) -> tuple[int, ...] | None:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    else:
        parsed = tuple(int(item) for item in value)
    return parsed or None


@dataclass(frozen=True)
class ExportMetadata:
    te_repo: str
    text_encoder_layer: int
    text_encoder_layers: tuple[int, ...] | None
    instruction: str | None
    max_text_length: int
    vae_repo: str
    flow_shift: float
    scaling_factor: float
    latent_mean: float
    latent_std: float
    sampler: str
    steps: int
    stork_derivative_order: int
    stork_substeps: int
    cfg_scale: float
    cfg_rescale: float
    cfg_interval_low: float
    cfg_interval_high: float


def _resolve_metadata(ckpt: dict, args: argparse.Namespace) -> ExportMetadata:
    """Recover the inference contract from the checkpoint, with explicit CLI wins.

    Model architecture lives in ``checkpoint['config']`` but text/sampler choices are
    trainer arguments. Silently replacing Qwen3.5-4B layers 9/18/27 with the old 2B
    last-layer default produces valid shapes only for old checkpoints and garbage (or a
    width error) for v8, so this resolution is intentionally centralized and exported.
    """
    train_args = ckpt.get("train_args") or {}
    model_config = ckpt.get("config") or {}

    def choose(name: str, default: Any, *checkpoint_names: str) -> Any:
        explicit = getattr(args, name, None)
        if explicit is not None:
            return explicit
        for checkpoint_name in checkpoint_names or (name,):
            value = train_args.get(checkpoint_name)
            if value is not None:
                return value
            value = model_config.get(checkpoint_name)
            if value is not None:
                return value
        return default

    layers = _parse_layers(choose("text_encoder_layers", None, "text_encoder_layers"))
    instruction = getattr(args, "instruction", None)
    if instruction is None:
        instruction = (
            DEFAULT_INSTRUCTION if bool(train_args.get("text_encoder_instruction", True)) else None
        )
    flow_shift = choose("flow_shift", 3.0, "sample_flow_shift", "flow_shift")
    # argparse stores optional sample_flow_shift as None in older checkpoints.
    if flow_shift is None:
        flow_shift = train_args.get("flow_shift", 3.0)

    # `config['text_seq_len']` is the authority, NOT train_args: it sizes the learned
    # `null_text_embedding` parameter, and on the cached-text path train.py auto-detects
    # it from a record (train.py:2421) and leaves --text-encoder-max-length and
    # --text-seq-len at their unrelated defaults. Exporting a shorter value makes the
    # pipeline slice the null caption (`token_num=text_emb.shape[-2]`), so CFG's uncond
    # branch silently loses trained null tokens.
    max_text_length = getattr(args, "max_text_length", None)
    if max_text_length is None:
        max_text_length = model_config.get("text_seq_len")
        trained = train_args.get("text_encoder_max_length")
        if max_text_length is None:
            raise SystemExit(
                "Checkpoint config has no 'text_seq_len'; pass --max-text-length explicitly."
            )
        if trained is not None and int(trained) != int(max_text_length):
            print(
                f"[metadata] note: train_args text_encoder_max_length={trained} disagrees "
                f"with config text_seq_len={max_text_length}; using text_seq_len, which is "
                "the length the null caption parameter was trained at."
            )

    meta = ExportMetadata(
        te_repo=str(choose("te_repo", "Qwen/Qwen3.5-4B", "text_encoder_id")),
        text_encoder_layer=int(choose("text_encoder_layer", -2, "text_encoder_layer")),
        text_encoder_layers=layers,
        instruction=instruction,
        max_text_length=int(max_text_length),
        vae_repo=str(choose("vae_repo", "mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers")),
        flow_shift=float(flow_shift),
        scaling_factor=float(
            choose("scaling_factor", 0.41407, "sample_scaling_factor", "edit_scaling_factor")
        ),
        latent_mean=float(choose("latent_mean", 0.0, "latent_mean")),
        latent_std=float(choose("latent_std", 1.0, "latent_std")),
        sampler=str(choose("sampler", "stork4")),
        steps=int(choose("steps", 20)),
        stork_derivative_order=int(choose("stork_derivative_order", 1)),
        stork_substeps=int(choose("stork_substeps", 14)),
        cfg_scale=float(choose("cfg_scale", 4.0)),
        cfg_rescale=float(choose("cfg_rescale", 0.5)),
        cfg_interval_low=float(choose("cfg_interval_low", 0.1)),
        cfg_interval_high=float(choose("cfg_interval_high", 0.9)),
    )
    if not 0.0 <= meta.cfg_interval_low <= meta.cfg_interval_high <= 1.0:
        raise SystemExit("CFG interval must satisfy 0 <= low <= high <= 1")
    if meta.sampler not in {"stork2", "stork4"}:
        raise SystemExit("HF export currently supports --sampler stork2 or stork4")
    return meta


def _write_model_card(
    out_dir: Path, model_name: str, repo_id: str, config: dict, meta: ExportMetadata
) -> None:
    readout = (
        f"layers {list(meta.text_encoder_layers)} (per-layer RMSNorm + concatenation)"
        if meta.text_encoder_layers
        else f"layer {meta.text_encoder_layer}"
    )
    prompt_list = "\n".join(
        f"{index}. {prompt}" for index, prompt in enumerate(SHOWCASE_PROMPTS, 1)
    )
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
with a {config['depth'] - config['ddt_decoder_depth']}-layer semantic encoder and
{config['ddt_decoder_depth']}-layer full-attention DDT velocity decoder. The encoder uses
bidirectional GDN-2 mixers, substitutive joint-attention refresh layers
{config.get('refresh_layers')}, and Block AttnRes. Text conditioning is
`{meta.te_repo}` read from {readout}. Images use frozen DC-AE f32c32 latents and render at
{config['latent_size'] * 32}px (latent {config['latent_size']}x{config['latent_size']}).

```python
import torch
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    "{repo_id}",
    custom_pipeline="pipeline_boomer",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
).to("cuda")

image = pipe("a lighthouse on a rocky cliff above crashing waves")[0]
image.save("out.png")
```

Requires: `torch diffusers transformers accelerate safetensors` and the AttnRes-capable
FLA build used for export:
`pip install "flash-linear-attention @ git+https://github.com/fla-org/flash-linear-attention.git@{FLA_PIN}"`.
The STORK scheduler is bundled (Tan et al. 2025, arXiv:2505.24210).

## Samples

![{model_name} samples](showcase.png)

All {config['latent_size'] * 32}px, EMA weights, {meta.sampler.upper()} / {meta.steps} steps,
`derivative_order={meta.stork_derivative_order}`, `substeps={meta.stork_substeps}`,
`cfg_scale={meta.cfg_scale}`, `cfg_rescale={meta.cfg_rescale}`,
`cfg_interval=[{meta.cfg_interval_low}, {meta.cfg_interval_high}]`, and
`flow_shift={meta.flow_shift}`. Prompts (left→right, top→bottom):

{prompt_list}
"""
    (out_dir / "README.md").write_text(card)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a Boomer checkpoint to a HF pipeline repo.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True, help="Local output dir for the repo.")
    parser.add_argument("--weights", default="ema", choices=["ema", "auto", "model"])
    parser.add_argument("--repo-id", default="Akrao9/Linarix-v1")
    parser.add_argument("--model-name", default="Linarix-v1")
    # None means recover from checkpoint train_args. Explicit values always win.
    parser.add_argument("--te-repo", default=None)
    parser.add_argument("--text-encoder-layer", type=int, default=None)
    parser.add_argument("--text-encoder-layers", default=None, help="Comma-separated readout layers.")
    parser.add_argument("--vae-repo", default=None)
    parser.add_argument("--instruction", default=None)
    parser.add_argument("--max-text-length", type=int, default=None)
    parser.add_argument("--flow-shift", type=float, default=None)
    parser.add_argument("--scaling-factor", type=float, default=None)
    parser.add_argument("--latent-mean", type=float, default=None)
    parser.add_argument("--latent-std", type=float, default=None)
    parser.add_argument("--sampler", choices=["stork2", "stork4"], default="stork4")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--stork-derivative-order", type=int, default=1)
    parser.add_argument("--stork-substeps", type=int, default=14)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--cfg-rescale", type=float, default=0.5)
    parser.add_argument("--cfg-interval-low", type=float, default=0.1)
    parser.add_argument("--cfg-interval-high", type=float, default=0.9)
    parser.add_argument("--stork-dir", default=None, help="Dir holding STORKScheduler.py + STORK_constants/.")
    # bf16 by default: it's the inference dtype (the pipeline loads with
    # torch_dtype=bfloat16), so fp32 storage only doubles the download for no
    # precision the runtime ever uses. Pass --save-dtype keep/fp32 to override.
    parser.add_argument("--save-dtype", default="bf16", choices=["keep", "bf16", "fp32"])
    parser.add_argument("--push", action="store_true", help="Upload to --repo-id (off by default).")
    parser.add_argument("--private", action="store_true", help="Create the repo private (with --push).")
    parser.add_argument("--hf-token", default=None)
    args = parser.parse_args()

    out_dir = Path(args.out)
    tf_dir = out_dir / "transformer"
    tf_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] {args.checkpoint}")
    ckpt = load_checkpoint(args.checkpoint)
    metadata = _resolve_metadata(ckpt, args)
    print(
        "[metadata] "
        f"text={metadata.te_repo} layers={metadata.text_encoder_layers or metadata.text_encoder_layer} "
        f"max_length={metadata.max_text_length}; {metadata.sampler} steps={metadata.steps} "
        f"d={metadata.stork_derivative_order} s={metadata.stork_substeps} "
        f"cfg={metadata.cfg_scale}/{metadata.cfg_rescale} "
        f"interval=[{metadata.cfg_interval_low},{metadata.cfg_interval_high}] "
        f"flow_shift={metadata.flow_shift}"
    )
    train_cfg = ckpt.get("config") or {}
    if not train_cfg:
        raise SystemExit("Checkpoint has no embedded 'config'; cannot build the model config.")

    state, tag = select_model_state(ckpt, args.weights)
    print(f"[weights] using '{tag}' weights ({len(state)} tensors)")

    state = _strip_prefixes(state)
    state = {k: v for k, v in state.items() if not k.startswith("repa_proj.")}
    # Migrate legacy GLUMBConv Conv2d-1x1 keys BEFORE save: load_state_dict works on
    # an internal copy, so relying on the module hook would publish safetensors with
    # the old key names (loadable only through the migration shim).
    n_migrated = migrate_glumbconv_state_dict(state)
    if n_migrated:
        print(f"[migrate] renamed {n_migrated} legacy GLUMBConv tensors to the Linear layout")

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
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise SystemExit(f"Round-trip mismatch — refusing to export a broken model:\n{exc}") from exc
    print("[verify] OK — all denoise-path weights map 1:1.")

    # ── write transformer/ ──
    config_json = dict(config)
    config_json.update({
        "_latent_mean": metadata.latent_mean,
        "_latent_std": metadata.latent_std,
        "_scaling_factor": metadata.scaling_factor,
        "_flow_shift": metadata.flow_shift,
        "_max_text_length": metadata.max_text_length,
        "_vae_repo": metadata.vae_repo,
        "_te_repo": metadata.te_repo,
        "_text_encoder_layer": metadata.text_encoder_layer,
        "_text_encoder_layers": metadata.text_encoder_layers,
        "_instruction": metadata.instruction,
        "_sampler": metadata.sampler,
        "_steps": metadata.steps,
        "_stork_derivative_order": metadata.stork_derivative_order,
        "_stork_substeps": metadata.stork_substeps,
        "_cfg_scale": metadata.cfg_scale,
        "_cfg_rescale": metadata.cfg_rescale,
        "_cfg_interval_low": metadata.cfg_interval_low,
        "_cfg_interval_high": metadata.cfg_interval_high,
    })
    (tf_dir / "config.json").write_text(json.dumps(config_json, indent=2))

    from safetensors.torch import save_file
    save_file(state, str(tf_dir / "model.safetensors"), metadata={"format": "pt"})
    print(f"[write] transformer/ ({len(state)} tensors, {args.save_dtype} dtype)")

    # ── copy code + scheduler + card ──
    for fname in ("pipeline_boomer.py", "modeling_boomer_fla.py"):
        shutil.copy2(REPO_ROOT / fname, out_dir / fname)
    shutil.copy2(REPO_ROOT / "boomer" / "models" / "qwen_connector.py", out_dir / "qwen_connector.py")
    # diffusers loads the `transformer` component (declared in model_index.json) by
    # resolving its class relative to the transformer/ subfolder under trust_remote_code,
    # so the modeling file MUST also live there — matching the working akrao9/Boomer-T2I
    # layout. The root copy is for pipeline_boomer.py's sibling import in its override path.
    shutil.copy2(REPO_ROOT / "modeling_boomer_fla.py", tf_dir / "modeling_boomer_fla.py")
    _copy_stork(out_dir, Path(args.stork_dir) if args.stork_dir else None)
    try:
        import diffusers
        diffusers_version = diffusers.__version__
    except Exception:
        diffusers_version = "0.33.0"
    # Declaring the transformer component is what lets the documented one-liner
    # DiffusionPipeline.from_pretrained(custom_pipeline="pipeline_boomer", trust_remote_code=True)
    # work — without it diffusers loads 0 components and raises "expected ['transformer']".
    (out_dir / "model_index.json").write_text(json.dumps({
        "_class_name": "BoomerPipeline",
        "_diffusers_version": diffusers_version,
        "transformer": ["modeling_boomer_fla", "BoomerFLADiT"],
    }, indent=2))
    _write_model_card(out_dir, args.model_name, args.repo_id, config, metadata)
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
