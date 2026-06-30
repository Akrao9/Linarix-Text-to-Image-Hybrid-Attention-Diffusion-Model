---
title: Linarix V1 T2I
emoji: 🎨
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 6.15.2
app_file: app.py
python_version: 3.11
pinned: false
license: other
short_description: 1024px text-to-image with Linarix-v1 (GDN-2) on ZeroGPU
models:
  - Akrao9/Linarix-v1
  - Qwen/Qwen3.5-2B
  - mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers
preload_from_hub:
  - Akrao9/Linarix-v1
---

# Linarix-v1 — Text to Image Demo

Interactive demo for [Akrao9/Linarix-v1](https://huggingface.co/Akrao9/Linarix-v1): a ~712M-param 1024×1024 text-to-image flow-matching model with GatedDeltaNet-2 (Flash Linear Attention) and Q/K-RMSNorm full-attention layers every 4th block.

**Capabilities:** strong on landscapes, scenic/architectural scenes, **rigid man-made structures** (lighthouses, locomotives, bridges), and composed portraits (use descriptive prompts; **CFG 4.0**). Eyes/iris are the remaining frontier and exact object counts are seed-driven. Text conditioning via **Qwen3.5-2B**; decode via **DC-AE f32c32**; sampling via **STORK-2**.

## Space setup

1. Create a new **Gradio** Space on Hugging Face.
2. Upload the contents of this `Space/` folder (or point the Space repo here).
3. In **Settings → Hardware**, select **ZeroGPU** (requires HF PRO for hosting). This allocates a shared **NVIDIA RTX Pro 6000 Blackwell** GPU (48 GB via `@spaces.GPU(size="large")`).
4. **No secret is required** — Qwen3.5-2B is not gated. Optionally add an `HF_TOKEN` secret (Settings → Secrets) to lift anonymous download rate limits.
5. Restart the Space after the first build (model + VAE + text encoder download can take a few minutes).

## Local test

```bash
cd Space
pip install -r requirements.txt
python app.py
```

The `@spaces.GPU` decorator is a no-op locally; use a CUDA machine for full inference.

Pinned deps for ZeroGPU: do **not** pin `torch` in `requirements.txt` — the Space builder injects `torch<=2.11.0`. Include **`torchvision`** (unpinned) for the Qwen3.5 `AutoProcessor`. Other pins: diffusers 0.38.0, transformers 5.9.0, flash-linear-attention 0.5.0.

`app.py` redirects Hugging Face caches to `/tmp/huggingface` (ZeroGPU `~/.cache` is often read-only).

Set **`python_version: 3.11`** in this README (Spaces default is 3.10). After changing it, **Factory rebuild** the Space so the runtime upgrades.

## Startup warnings (usually harmless)

| Message | Cause | Action |
|---|---|---|
| Python 3.10 below recommended 3.11 | Spaces default until `python_version: 3.11` is applied | Factory rebuild after README change |
| `Triton is not supported… roll back to CPU` | FLA probes Triton at import; on some builders/GPUs it falls back to non-Triton CUDA kernels | Safe if generation works; rebuild on 3.11 |
| `torch.compile is not available in Python 3.10` | FLA training helper stub; **inference does not use torch.compile** | Fixed by Python 3.11 rebuild |
| `local_dir_use_symlinks` deprecated | Older `huggingface_hub` call inside a dependency | Cosmetic; ignore or upgrade hub later |

## Notes

- Uses `@spaces.GPU(size="large", duration=150)` — RTX Pro 6000 Blackwell 48 GB tier — for 32-step STORK-2 at 1024px ([Tan et al., 2025](https://arxiv.org/abs/2505.24210)).
- Example prompts run generation on click (`run_on_click=True`).
- **CFG 4.0** is the recommended default; higher (5+) over-saturates and hardens fine detail. For exact object counts, sweep a few seeds.
- VAE and Qwen3.5-2B are fetched from their upstream repos on first startup.
