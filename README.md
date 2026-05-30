---
license: other
tags:
  - text-to-image
  - diffusion
  - linear-attention
  - pytorch
  - safetensors
language:
  - en
pipeline_tag: text-to-image
---

# Boomer FLA

Boomer FLA is a 657M parameter text-to-image diffusion model that generates **1024×1024px** images from text prompts.

Instead of standard quadratic self-attention, it uses **GatedDeltaNet** — a bidirectional Flash Linear Attention mixer — as the backbone of its transformer blocks. This keeps memory flat regardless of sequence length. Every 6th block adds a full SDPA layer for global spatial coherence.

Text conditioning uses **Gemma 4 2B** (1536-dim embeddings, up to 300 tokens). Decoding uses the **DC-AE f32c32** VAE with 32× spatial compression, producing 32×32 latents from 1024px images. Sampling uses **STORK-2**, a high-order Runge–Kutta flow matching solver that converges in 32 steps.

Fine-tuned from a JourneyDB-pretrained base on 600k high-resolution images at 1024px.

---

## Sample outputs

![Model grid — portraits and landscapes](boomer_model_grid.png)

*Portraits (top) and landscapes (bottom) generated at 1024×1024px, 32 STORK-2 steps, CFG 4.5.*

---

## Architecture

| Property | Value |
|---|---|
| Parameters | 657M |
| Backbone | Bidirectional GatedDeltaNet (Flash Linear Attention) |
| Depth | 24 layers |
| Hidden dim | 896 |
| Heads | 14 |
| Image attention | Every 6th layer (full SDPA + 2D RoPE) |
| Patch size | 1 — one token per latent pixel (256 tokens at 512px, 1024 tokens at 1024px) |
| Text encoder | Gemma 4 2B (`google/gemma-4-E2B-it`) |
| VAE | DC-AE f32c32 (`mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers`) |
| Sampler | STORK-2, 32 steps |
| Dtype | bfloat16 |

---

## Training details

| Setting | Value |
|---|---|
| Pre-train dataset | JourneyDB (~3.8M images, 512px, patch size 1) |
| Fine-tune dataset | FineT2I (~600k images, 1024px, patch size 1) |
| Fine-tune steps | 55,000 |
| Batch size | 24 |
| Learning rate | 1e-4, linear warmup (1000 steps) → cosine decay |
| Flow shift | 1.5 |
| Timestep sampler | Plateau logit-normal (μ=0, σ=1) |
| Min-SNR γ | 5.0 |
| CFG dropout | 0.1 |
| EMA decay | 0.999 |
| Gradient clip | 0.3 |
| Optimizer | Fused AdamW |
| Hardware | NVIDIA A100 (Google Colab) |
| Precision | bfloat16 |

---

## VRAM and RAM requirements

Measured at 1024×1024px, bfloat16, STORK-2, CFG batch=2.

| Component | VRAM |
|---|---|
| DiT weights (EMA, bf16) | 1.25 GB |
| Gemma 4 2B text encoder | 8.62 GB |
| Denoising peak (CFG on) | 1.36 GB |
| VAE decode peak | 3.51 GB |

| Mode | Peak VRAM | Minimum GPU |
|---|---|---|
| **Condition-cache** — pre-encoded embeddings, no text encoder in VRAM | **4.76 GB** | RTX 3060 8GB, T4 |
| **Fresh-prompt** — text encoder + DiT + VAE together | **13.38 GB** | RTX 3090, A100 |

**System RAM**: loading the text encoder (Gemma 4 2B) requires ~9 GB of system RAM even when using GPU. For condition-cache mode, encoding can be done on CPU with ~9 GB RAM — the generation step then needs only 5 GB VRAM.

---

## Usage

### Install

```bash
pip install torch diffusers transformers accelerate safetensors
pip install git+https://github.com/fla-org/flash-linear-attention.git
# STORK is bundled with the model — no separate install needed
```

### Generate

```python
import torch
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    "akrao9/Boomer-T2I",
    custom_pipeline="pipeline_boomer",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
).to("cuda")

image = pipe("a photorealistic portrait of a woman with dark hair")[0]
image.save("output.png")
```

> **Note:** The transformer weights (~1.25 GB) are downloaded from this repo. The VAE and Gemma 4 2B text encoder are fetched automatically from their upstream HuggingFace repos on first use (~10 GB total). Make sure you have accepted the [Gemma Terms of Use](https://ai.google.dev/gemma/terms) and are logged in with `huggingface-cli login`.

### Parameters

```python
image = pipe(
    "a rocky coastline at sunset with crashing waves",
    steps=32,        # denoising steps — 32 is recommended with STORK-2
    cfg_scale=4.5,   # classifier-free guidance scale (4.0–5.0)
    cfg_rescale=0.5, # reduces over-saturation at high CFG
    # seed=42,       # fix a seed for reproducibility; omit for a random image each run
)[0]
```

### Low VRAM — condition cache mode

Encode prompts once on any machine (including CPU), save the embedding, then generate with only the 1.25 GB DiT loaded. Peak VRAM drops from 13.38 GB → 4.76 GB.

```python
# Step 1 — encode on any machine (even CPU with 9GB RAM)
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

TE_REPO      = "google/gemma-4-E2B-it"
tokenizer    = AutoProcessor.from_pretrained(TE_REPO)
text_encoder = AutoModelForCausalLM.from_pretrained(
    TE_REPO, torch_dtype=torch.bfloat16
).get_decoder()

tokens = tokenizer(
    "a mountain lake surrounded by alpine peaks",
    max_length=300, padding="max_length",
    truncation=True, return_tensors="pt",
)
with torch.inference_mode():
    hidden = text_encoder(
        tokens["input_ids"], attention_mask=tokens["attention_mask"]
    )[0]
    idx  = [0] + list(range(-299, 0))
    emb  = hidden[:, idx]
    mask = tokens["attention_mask"][:, idx]

torch.save({"emb": emb.cpu(), "mask": mask.cpu()}, "condition.pt")
```

```python
# Step 2 — generate on low-VRAM GPU
# Load only the DiT weights; skip text encoder by providing embeddings directly
import torch
from huggingface_hub import snapshot_download
import sys

local = snapshot_download("akrao9/Boomer-T2I")
sys.path.insert(0, local)
from pipeline_boomer import BoomerPipeline
from modeling_boomer_fla import BoomerFLADiT

transformer = BoomerFLADiT.from_pretrained(local, subfolder="transformer").to("cuda", dtype=torch.bfloat16)
pipe = BoomerPipeline(transformer=transformer).to("cuda")
# pipe.text_encoder and pipe.vae remain None until __call__ — override with pre-encoded emb manually
```

---

## Limitations

- **Strong at** — photorealistic human portraits, dramatic landscapes, architectural scenes
- **Weak at** — animals, text rendering, small detailed objects (limited training data coverage)
- Landscapes have a painterly/HDR bias inherited from heavily post-processed stock images in the training set
- Not safety filtered — outputs may reflect biases in the training data
- Maximum tested resolution: **1024×1024px**

---

## License

The Boomer FLA model weights are released for research and personal use. Commercial use is not permitted without explicit permission.

Upstream component licenses:
- DC-AE VAE: [mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers](https://huggingface.co/mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers)
- Gemma 4 text encoder: [Gemma Terms of Use](https://ai.google.dev/gemma/terms)
