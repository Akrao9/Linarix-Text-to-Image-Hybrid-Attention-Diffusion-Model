---
license: other
tags:
  - text-to-image
  - diffusion
  - linear-attention
  - gated-deltanet
  - pytorch
  - safetensors
language:
  - en
pipeline_tag: text-to-image
library_name: diffusers
---

# BoomerV2 — Text-to-Image

BoomerV2 is a **701M parameter** text-to-image diffusion model that generates **1024×1024px** images from text prompts.

Instead of standard quadratic self-attention, it uses **GatedDeltaNet-2 (GDN-2)** — a bidirectional Flash Linear Attention mixer with decoupled channel-wise erase/write gates — as the backbone of its transformer blocks. This keeps memory roughly flat with sequence length. Every 6th block adds a full SDPA layer with 2D RoPE for global spatial coherence.

Text conditioning uses **Gemma 4 E2B** (1536-dim embeddings, up to 384 tokens). Decoding uses the **DC-AE f32c32** VAE with 32× spatial compression, producing 32×32 latents from 1024px images. Inference uses **STORK-2** flow-matching sampling ([Tan et al., 2025](https://arxiv.org/abs/2505.24210)).

Pre-trained on ~3.8M JourneyDB 512px latents, then fine-tuned on ~600k FineT2I 1024px latents.

---

## Sample Outputs

![BoomerV2 Portfolio Grid](https://cdn-uploads.huggingface.co/production/uploads/68a39870573ae211ac576be6/fN3TCPf5lGBFCq_thgYkw.png)

Generated at 1024×1024px, STORK-2, 32 steps, CFG 4.0. Prompts are composed (scene + subject), which is how the model performs best.

## Architecture

![architecture](https://cdn-uploads.huggingface.co/production/uploads/68a39870573ae211ac576be6/9svsLkAt5aV9iWYD25xS_.png)

| Property | Value |
|---|---|
| Parameters | 701M |
| Backbone | Bidirectional GatedDeltaNet-2 (Flash Linear Attention) |
| Depth | 24 layers |
| Hidden dim | 896 |
| Heads | 14 |
| Image attention | Every 6th layer (full SDPA + 2D RoPE) |
| Patch size | 1 — one token per latent pixel (256 tokens @ 512px, 1024 tokens @ 1024px) |
| Text encoder | Gemma 4 E2B (`google/gemma-4-E2B-it`), up to 384 tokens |
| VAE | DC-AE f32c32 (`mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers`) |
| Sampler | STORK-2, 32 steps |
| Dtype | bfloat16 |

---

## Training details

| Setting | Value |
|---|---|
| Pre-train dataset | JourneyDB (~3.8M images, 512px, patch size 1) |
| Fine-tune dataset | FineT2I (~600k images, 1024px, patch size 1) |
| Optimizer | Fused AdamW |
| Objective | Flow matching (velocity), min-SNR γ=5, flow shift 1.5 |
| Timestep sampler | Plateau logit-normal (μ=0, σ=1) |
| Hardware | RTX PRO 6000 Blackwell Server Edition |
| Precision | bfloat16 |

---

## Performance

Measured at 1024×1024px, bfloat16, STORK-2 (32 steps), on an RTX PRO 6000 Blackwell.

**Memory** (the denoiser is tiny; footprint is dominated by the text encoder):

| Component | VRAM |
|---|---|
| DiT weights (EMA, bf16) | ~1.5 GB |
| Gemma 4 E2B text encoder | ~9.3 GB |
| DC-AE VAE | ~0.6 GB |
| **All loaded (resident)** | **~10 GB** |
| **Peak during generation** | **~13 GB allocated / ~15 GB reserved** |

Memory scales nearly flat with batch size (linear-attention backbone): batch 1→8 grows the DiT's allocated memory only ~20% (1.5→1.8 GB).

**Latency** (end-to-end, prompt → 1024px image, eager):

| Stage | Time | Share |
|---|---|---|
| Text encode (Gemma) | ~25 ms | 1% |
| DiT denoise (32 steps, CFG) | ~1.95 s | 95% |
| VAE decode | ~80 ms | 4% |
| **Total** | **~2.05 s/image** | |

For batched serving, the DiT forward reaches **~121 img/s at batch 8** (compiled). For single-image latency, eager is fastest (torch.compile does not speed up the launch-bound batch-1 case).

| Mode | Peak VRAM | Minimum GPU |
|---|---|---|
| **Pre-encoded embeddings** — no text encoder resident | **~5 GB** | RTX 3060 8GB, T4 |
| **Fresh-prompt** — text encoder + DiT + VAE together | **~13–15 GB** | RTX 3090, A100 |

---

## Usage

> **Requires `diffusers >= 0.38.0`** — earlier versions have a `trust_remote_code` RCE
> ([advisory](https://github.com/huggingface/diffusers/security/advisories/GHSA-98h9-4798-4q5v)).
> For production, pin a commit hash with `revision=` so the remote code can't change under you.

### Install

```bash
pip install -U "diffusers>=0.38.0" transformers accelerate safetensors torchvision scipy
pip install git+https://github.com/fla-org/flash-linear-attention.git
```

### Generate

```python
import torch
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    "akrao9/BoomerV2-Text-to-Image",
    custom_pipeline="pipeline_boomer",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
).to("cuda")

image = pipe("A hyper-detailed, cinematic landscape photography shot of a pristine, mirror-like alpine lake nestled deeply between towering, jagged snow-capped mountain peaks. The scene is captured during the perfect golden hour, with the low-angled warm sun casting deep amber and violet hues across the rugged granite rock faces. In the foreground, vibrant clusters of purple lupines and orange poppies dot a lush emerald meadow that meets the crystal-clear turquoise edge of the water. Wisps of soft, low-hanging morning mist drift lazily across the lake's surface, breaking the perfect reflection of the monumental peaks above. Shot on 35mm lens, ultra-sharp focus, dramatic depth of field, 8k resolution, path-traced lighting textures.")[0]
image.save("output.png")
```

Optional generation parameters:

```python
image = pipe(
    "a rocky coastline at sunset with crashing waves",
    steps=32,        # STORK-2 denoising steps
    cfg_scale=4.0,   # classifier-free guidance (4.0–4.5 recommended)
    cfg_rescale=0.5, # reduces over-saturation / dark crush at higher CFG
    seed=42,
)[0]
```

The transformer weights (~1.4 GB) download from this repo. The VAE and Gemma 4 E2B text encoder are fetched from their upstream HuggingFace repos on first use (~10 GB total). Accept the [Gemma Terms of Use](https://ai.google.dev/gemma/terms) and run `hf auth login` before first use.

### Prompting tips

- **Use composed, descriptive prompts** (scene + subject). Bare one-word prompts (e.g. `"a woman"`) can produce duplicated subjects; adding composition (`"a portrait of a single woman, ..."`) resolves it.
- **CFG 4.0–4.5** is the sweet spot. Too high crushes darks (e.g. black-void eyes on frontal faces).
- **For people**, prefer **three-quarter or profile poses** ("looking out at the sea", "profile") over direct frontal close-ups.

---

## Capabilities and limitations

Pre-trained on **JourneyDB** (512px) and fine-tuned on **FineT2I** (1024px). The training data is human- and scene-heavy (≈56% of captions mention people, ≈70% mention scenes), which shapes what the model does well.

**Strong:**
- Landscapes, natural environments, architectural and scenic scenes
- **Humans and portraits** — coherent faces and anatomy (young and elderly), especially in three-quarter / profile poses
- Subjects placed within a scene (subject-in-scene composition)

**Works, with care:**
- Everyday objects embedded in a scene (quality varies)
- Frontal close-up faces — good, but can show eye artifacts at high CFG; keep CFG ≤ 4.5
- **Dense, multi-attribute prompts** — handles spatially-localized attributes well (e.g. "purple lupines and orange poppies in the foreground, turquoise lake, golden-hour light on the peaks" renders each correctly); occasional minor blemishes on small repeated elements

**Less reliable:**
- **Domestic animals** (e.g. dogs) — the training set is animal-sparse and skewed toward dramatic/wild animals, so pets in wild settings can drift toward bears / large mammals
- **Hands** — the classic diffusion failure; not reliably correct
- Legible text in images, very fine small details

**Other notes:**
- Landscapes can show a painterly / HDR bias from heavily post-processed training images
- Not safety filtered — outputs may reflect biases in the training data
- Maximum tested resolution: **1024×1024px**

---

## Acknowledgements

- **GatedDeltaNet-2** — linear-attention backbone (NVIDIA, [arXiv:2605.22791](https://arxiv.org/abs/2605.22791)), via [flash-linear-attention](https://github.com/fla-org/flash-linear-attention)
- **STORK-2** — inference sampling ([Tan et al., 2025](https://arxiv.org/abs/2505.24210))
- **DC-AE** — latent autoencoder (`mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers`)
- **Plateau logit-normal** — training timestep distribution from [FLUX.2 representation comparison](https://bfl.ai/research/representation-comparison) (Black Forest Labs, 2025). BoomerV2 uses μ=0, σ=1 with flow shift 1.5.
