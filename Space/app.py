"""Linarix-v2 — Hugging Face Space demo (ZeroGPU / RTX Pro 6000 Blackwell + Gradio)."""

from __future__ import annotations

import os
from pathlib import Path

# ZeroGPU Spaces: ~/.cache is often read-only — use /tmp for HF + diffusers caches.
_hf_root = Path(os.environ.get("HF_HOME", "/tmp/huggingface"))
for _sub in ("hub", "modules", "transformers", "diffusers"):
    (_hf_root / _sub).mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(_hf_root))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(_hf_root / "hub"))
os.environ.setdefault("HF_MODULES_CACHE", str(_hf_root / "modules"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(_hf_root / "transformers"))
os.environ.setdefault("DIFFUSERS_CACHE", str(_hf_root / "diffusers"))

import gc
import secrets
import sys
import time
import warnings

# FLA / hub noise at import on Spaces builder (Triton probe, py3.10 compile stub, deprecated kwargs).
warnings.filterwarnings("ignore", message="Triton is not supported on current platform")
warnings.filterwarnings("ignore", message="torch.compile is not available in Python 3.10")
warnings.filterwarnings("ignore", message="The `local_dir_use_symlinks` argument is deprecated")

import gradio as gr
import spaces
import torch
from huggingface_hub import snapshot_download

MODEL_ID = os.environ.get("BOOMER_MODEL_ID", "Akrao9/Linarix-v2")

DEFAULT_PROMPT = (
    "a lighthouse on a rocky cliff above crashing waves at golden hour, "
    "dramatic stormy sky, cinematic photography"
)

# Drawn from boomer/showcase_prompts.py (the set numbered under the model card's
# sample grid), expanded with the descriptive phrasing this model responds best to.
EXAMPLE_PROMPTS: list[tuple[str, str]] = [
    (
        "Mountain Village",
        "a snow-covered mountain village at blue hour, warm light glowing from every window, smoke rising from chimneys, deep alpine dusk.",
    ),
    (
        "Desert Canyon",
        "a desert canyon at sunset, towering layered red rock walls carved by wind, long shadows across the canyon floor, warm directional light.",
    ),
    (
        "Night Motorcycle",
        "a vintage motorcycle parked on a rain-slicked city street at night, neon signs reflecting in the wet asphalt, chrome catching the light, cinematic.",
    ),
    (
        "Venetian Canal",
        "a narrow canal in Venice in late afternoon, weathered pastel facades leaning over still green water, laundry strung between shutters, golden side light.",
    ),
]


def _hf_token() -> str | None:
    """Return HF token from Space secrets / env if set; None otherwise.

    Linarix-v2 itself is gated, so Spaces must set an ``HF_TOKEN`` secret whose
    owner can read ``Akrao9/Linarix-v2``. Qwen3.5 / DC-AE are public.
    """
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGING_HUB_TOKEN"):
        value = os.environ.get(key, "").strip()
        if value:
            return value
    return None


_hf = _hf_token()
if not _hf:
    raise RuntimeError(
        f"{MODEL_ID} is gated. Add an HF_TOKEN Space secret "
        "(Settings → Secrets) with read access to that repo, then restart."
    )

# Make the token available to every huggingface_hub / transformers / diffusers call.
os.environ.setdefault("HF_TOKEN", _hf)
os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", _hf)
try:
    from huggingface_hub import login as _hf_login

    _hf_login(token=_hf, add_to_git_credential=False)
except Exception as exc:  # noqa: BLE001 — non-fatal; explicit token= still used below
    print(f"Warning: huggingface_hub.login failed ({exc!r}); continuing with token= kwarg.", flush=True)

print(f"Loading Linarix pipeline from {MODEL_ID} ...", flush=True)
_model_dir = Path(
    snapshot_download(
        MODEL_ID,
        token=_hf,
        ignore_patterns=["*.png", "*.jpg", "*.jpeg"],
    )
)
if str(_model_dir) not in sys.path:
    sys.path.insert(0, str(_model_dir))

from pipeline_boomer import BoomerPipeline  # noqa: E402

pipe = BoomerPipeline.from_pretrained(str(_model_dir), torch_dtype=torch.bfloat16, token=_hf)
pipe.to("cuda")
pipe._hf_token = _hf

print("Pre-loading VAE on cuda ...", flush=True)
pipe._ensure_vae()
print(f"Pre-loading text encoder ({pipe._te_repo}) on cuda ...", flush=True)
pipe._ensure_text_encoder()
print("Model ready.", flush=True)

MODEL_RESOLUTION = int(pipe.transformer.config.latent_size) * 32
MODEL_PARAMS_M = sum(parameter.numel() for parameter in pipe.transformer.parameters()) / 1e6
MODEL_SAMPLER = str(pipe.default_sampler).upper()
DEFAULT_STEPS = int(pipe.default_steps)
DEFAULT_CFG = float(pipe.default_cfg_scale)
DEFAULT_CFG_RESCALE = float(pipe.default_cfg_rescale)


def _resolve_seed(seed: float | int | None) -> int:
    if seed is None or int(seed) < 0:
        # OS entropy, NOT torch.randint: ZeroGPU forks a worker per @spaces.GPU call
        # with an identical torch RNG snapshot (and the pipeline re-seeds the global
        # generator via torch.manual_seed), so torch draws here repeat the same value.
        return secrets.randbelow(2**32)
    return int(seed)


def _module_vram_gb(module: object | None) -> float:
    """CUDA memory for module parameters and buffers (weights only, not activations)."""
    if module is None or not torch.cuda.is_available() or not hasattr(module, "parameters"):
        return 0.0
    bytes_used = 0
    for tensor in (*module.parameters(), *module.buffers()):
        if tensor.is_cuda:
            bytes_used += tensor.numel() * tensor.element_size()
    return bytes_used / (1024**3)


def _cuda_peak_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024**3)


def _format_gen_stats(
    *,
    step_count: int,
    elapsed_s: float,
    peak_vram_gb: float,
    dit_weights_gb: float,
    denoise_vram_gb: float | None = None,
    decode_vram_gb: float | None = None,
) -> str:
    it_per_s = step_count / elapsed_s if elapsed_s > 0 else 0.0
    parts = [
        f"Generation Speed: {it_per_s:.2f} it/s",
        f"Peak VRAM: {peak_vram_gb:.2f} GB",
    ]
    if denoise_vram_gb is not None:
        parts.append(f"Denoise VRAM: {denoise_vram_gb:.2f} GB")
    if decode_vram_gb is not None:
        parts.append(f"Decode VRAM: {decode_vram_gb:.2f} GB")
    parts.append(f"DiT weights: {dit_weights_gb:.2f} GB")
    return " | ".join(parts)


# ZeroGPU "large" = half NVIDIA RTX Pro 6000 Blackwell MIG slice (48 GB VRAM).
@spaces.GPU(size="large", duration=150)
def generate_image(
    prompt: str,
    seed: float,
    steps: float,
    cfg_scale: float,
) -> tuple[object, str]:
    prompt = (prompt or "").strip()
    if not prompt:
        raise gr.Error("Please enter a prompt before generating.")

    resolved_seed = _resolve_seed(seed)
    step_count = max(1, int(steps))
    cfg = float(cfg_scale)

    dit_weights_gb = _module_vram_gb(pipe.transformer)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    started = time.perf_counter()
    result = pipe(
        prompt,
        steps=step_count,
        seed=resolved_seed,
        cfg_scale=cfg,
        cfg_rescale=DEFAULT_CFG_RESCALE,
        offload_text_encoder=True,
    )
    elapsed_s = time.perf_counter() - started
    image = result[0]
    peak_vram_gb = _cuda_peak_gb()
    run_stats = getattr(pipe, "last_run_stats", None)
    denoise_vram_gb = getattr(run_stats, "peak_denoise_gb", None) if run_stats else None
    decode_vram_gb = getattr(run_stats, "peak_decode_gb", None) if run_stats else None
    stats_line = _format_gen_stats(
        step_count=step_count,
        elapsed_s=elapsed_s,
        peak_vram_gb=peak_vram_gb,
        dit_weights_gb=dit_weights_gb,
        denoise_vram_gb=denoise_vram_gb,
        decode_vram_gb=decode_vram_gb,
    )
    print(stats_line, flush=True)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    status = (
        f"Done — seed {resolved_seed}, steps {step_count}, CFG {cfg:.1f}. "
        f"{MODEL_RESOLUTION}×{MODEL_RESOLUTION}px via {MODEL_SAMPLER}.\n{stats_line}"
    )
    return image, status


def reset_form() -> tuple[str, None, str, float]:
    return DEFAULT_PROMPT, None, "Reset.", -1.0


def build_ui() -> gr.Blocks:
    example_rows = [[text, -1, DEFAULT_STEPS, DEFAULT_CFG] for _, text in EXAMPLE_PROMPTS]

    with gr.Blocks(title="Linarix-v2") as demo:
        gr.Markdown(
            """
            # Linarix-v2 — Text to Image
            """
            f"""Generate **{MODEL_RESOLUTION}×{MODEL_RESOLUTION}** images from text with a
            **~{MODEL_PARAMS_M:.0f}M-parameter** GatedDeltaNet-2/DDT model using Block
            AttnRes and substitutive joint-attention refreshes. **Strong on landscapes, scenic and
            man-made structures, and composed portraits** — use descriptive prompts;
            **CFG {DEFAULT_CFG:.1f}** is the default. For exact object counts, try a few seeds.
            Runs on **ZeroGPU** (NVIDIA RTX Pro 6000 Blackwell, 48 GB). First run may
            take a minute while a GPU is allocated.
            """
        )

        with gr.Row(equal_height=False):
            with gr.Column(scale=3):
                prompt = gr.Textbox(
                    label="Prompt",
                    value=DEFAULT_PROMPT,
                    lines=10,
                )
                with gr.Accordion("Advanced settings", open=False):
                    seed = gr.Number(
                        label="Seed (-1 = random)",
                        value=-1,
                        precision=0,
                    )
                    steps = gr.Slider(
                        label="Denoising steps",
                        minimum=16,
                        maximum=64,
                        step=1,
                        value=DEFAULT_STEPS,
                    )
                    cfg_scale = gr.Slider(
                        label="CFG scale",
                        minimum=1.0,
                        maximum=8.0,
                        step=0.1,
                        value=DEFAULT_CFG,
                    )
                with gr.Row():
                    generate_btn = gr.Button("Generate", variant="primary")
                    reset_btn = gr.Button("Reset")
                output = gr.Image(label="Generated image", type="pil", height=640)
                status = gr.Textbox(label="Status", interactive=False)

            with gr.Column(scale=2):
                gr.Markdown("### Example prompts")
                gr.Markdown("Click an example to **generate immediately**.")
                gr.Examples(
                    examples=example_rows,
                    inputs=[prompt, seed, steps, cfg_scale],
                    outputs=[output, status],
                    fn=generate_image,
                    run_on_click=True,
                    cache_examples=False,
                    label="",
                    examples_per_page=4,
                )

        generate_btn.click(
            fn=generate_image,
            inputs=[prompt, seed, steps, cfg_scale],
            outputs=[output, status],
        )
        reset_btn.click(
            fn=reset_form,
            outputs=[prompt, output, status, seed],
        )

        prompt.submit(
            fn=generate_image,
            inputs=[prompt, seed, steps, cfg_scale],
            outputs=[output, status],
        )

    return demo


demo = build_ui()

if __name__ == "__main__":
    demo.launch()
