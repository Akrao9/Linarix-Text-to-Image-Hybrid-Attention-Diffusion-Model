"""Boomer FLA — Hugging Face Space demo (ZeroGPU / RTX Pro 6000 Blackwell + Gradio)."""

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

MODEL_ID = os.environ.get("BOOMER_MODEL_ID", "akrao9/BoomerV2-Text-to-Image")
DEFAULT_STEPS = 32
DEFAULT_CFG = 4.0
DEFAULT_CFG_RESCALE = 0.5

DEFAULT_PROMPT = (
    "A hyper-detailed, cinematic landscape photography shot of a pristine, mirror-like alpine lake "
    "nestled deeply between towering, jagged snow-capped mountain peaks. The scene is captured during "
    "the perfect golden hour, with the low-angled warm sun casting deep amber and violet hues across "
    "the rugged granite rock faces. In the foreground, vibrant clusters of purple lupines and orange "
    "poppies dot a lush emerald meadow that meets the crystal-clear turquoise edge of the water. "
    "Wisps of soft, low-hanging morning mist drift lazily across the lake's surface, breaking the "
    "perfect reflection of the monumental peaks above. Shot on 35mm lens, ultra-sharp focus, dramatic "
    "depth of field, 8k resolution, path-traced lighting textures."
)

EXAMPLE_PROMPTS: list[tuple[str, str]] = [
    (
        "Cyberpunk Highland",
        "A sweeping cinematic view of a futuristic Scottish highland at twilight, glowing neon purple and blue moss creeping over ancient rock formations, distant futuristic spires reflecting off a dark loch, dramatic low-angle photography.",
    ),
    (
        "Volcanic Caldera",
        "A high-contrast shot looking down into an active volcanic caldera, vibrant flowing rivers of orange lava cutting through pitch-black obsidian stone, thick plumes of dark smoke catching the crimson glow, hyper-detailed geological textures.",
    ),
    (
        "Nordic Winter Fjords",
        "A breathtaking winter view of deep Norwegian fjords during the blue hour, snow-laden mountain slopes plunging into dark mirror-like ocean water, vibrant emerald green Northern Lights stretching across the sky, crisp and ultra-clear atmospheric rendering.",
    ),
    (
        "Ancient Sunken Ruins",
        "A wide landscape shot of ancient stone ruins submerged in a shallow, crystal-clear tropical ocean, vibrant coral reefs and schools of exotic fish visible beneath the surface, sun rays slicing through the water, warm cinematic lighting.",
    ),
]


def _hf_token() -> str | None:
    """Return HF token from Space secrets; None if unset (do not pass token=True)."""
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGING_HUB_TOKEN"):
        value = os.environ.get(key, "").strip()
        if value:
            return value
    return None


_hf = _hf_token()
if _hf is None:
    print(
        "WARNING: No HF_TOKEN secret found. Add one in Space Settings → Secrets "
        "(required for gated Gemma text encoder).",
        flush=True,
    )

print(f"Loading Boomer pipeline from {MODEL_ID} ...", flush=True)
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
if _hf is not None:
    print("Pre-loading text encoder on cuda ...", flush=True)
    pipe._ensure_text_encoder()
else:
    print("Skipping text encoder preload (HF_TOKEN not set).", flush=True)
print("Model ready.", flush=True)


def _resolve_seed(seed: float | int | None) -> int:
    if seed is None or int(seed) < 0:
        return torch.randint(0, 2**32 - 1, (1,)).item()
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
    if pipe._hf_token is None:
        raise gr.Error(
            "HF_TOKEN Space secret is required. In Space Settings → Secrets, add "
            "HF_TOKEN with a Hugging Face token that has accepted the "
            "Gemma Terms of Use (https://ai.google.dev/gemma/terms)."
        )

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
        f"1024×1024px via STORK-2.\n{stats_line}"
    )
    return image, status


def reset_form() -> tuple[str, None, str, float]:
    return DEFAULT_PROMPT, None, "Reset.", -1.0


def build_ui() -> gr.Blocks:
    example_rows = [[text, -1, DEFAULT_STEPS, DEFAULT_CFG] for _, text in EXAMPLE_PROMPTS]

    with gr.Blocks(title="BoomerV2") as demo:
        gr.Markdown(
            """
            # BoomerV2 — Text to Image
            Generate **1024×1024** images from text. **Strong on landscapes, scenic
            environments, and people** (use composed prompts; CFG 4.0–4.5). Less reliable
            for domestic animals, hands, and legible text.
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
