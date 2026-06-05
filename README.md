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

Boomer FLA is a **657M-parameter** text-to-image **flow-matching** model that generates **1024×1024px** images from natural-language prompts.

**Research project** — Boomer FLA is released for **research and educational purposes** only. It is not a production product; outputs are provided as-is with no guarantees of safety, quality, or fitness for any particular use.

**Pipeline:** prompt → **Gemma 4 2B** (text) → **BoomerFLADiT** (32 STORK-2 denoising steps in latent space) → **DC-AE f32c32** decode → pixels. Only the DiT is trained; the VAE and text encoder stay frozen.

The DiT never sees pixels directly. DC-AE compresses 1024×1024 RGB to **32×32×32** latents (32× spatial compression). Instead of quadratic self-attention everywhere, blocks use **bidirectional GatedDeltaNet** (Flash Linear Attention) for **O(N)** memory; every 6th block adds full **SDPA + 2D RoPE** for global spatial coherence.

**Best for** dramatic landscapes and scenic environments — not reliable for humans or portraits. See [Capabilities and limitations](#capabilities-and-limitations).

Trained on **JourneyDB** (~3.8M images, 512px pretrain) and **FineT2I** (~600k images, 1024px finetune). Inference uses **STORK-2** ([Tan et al., 2025](https://arxiv.org/abs/2505.24210)).

---

## Sample Outputs

![Boomer-T2I Portfolio Landscape Grid](https://cdn-uploads.huggingface.co/production/uploads/68a39870573ae211ac576be6/fN3TCPf5lGBFCq_thgYkw.png)

1. **Misty Pine Forest:** A cinematic, wide-angle shot of a misty pine forest at sunrise, deep green valleys, soft morning light piercing through fog, photorealistic, 8k resolution.
2. **Black Sand Beach:** A dramatic black sand beach in Iceland, towering basalt columns, massive white waves crashing on the shore, moody overcast sky, high detail landscape architecture.
3. **Alpine Lake:** A serene alpine lake reflecting jagged snow-capped mountain peaks, crystal clear turquoise water, vibrant wildflower meadows in the foreground, golden hour lighting.
4. **Tuscan Hills:** A sweeping view of rolling terracotta hills in Tuscany, isolated cypress trees lining a dirt road, warm late afternoon sun casting long shadows, classic landscape photography.
5. **Desert Oasis:** A futuristic desert oasis at dusk, neon-infused crystalline structures contrasting with deep orange sand dunes, hyper-detailed, sprawling cinematic atmosphere.
6. **Hidden Lagoon:** A majestic waterfall cascading down a sheer mossy cliff into a hidden tropical lagoon, lush emerald foliage, sunbeams cutting through the canopy, long exposure water effect.

## Architecture

| Property | Value |
|---|---|
| Parameters | 657M |
| Objective | Flow matching (velocity prediction in latent space) |
| Backbone | Bidirectional GatedDeltaNet (Flash Linear Attention) |
| Depth | 24 transformer blocks |
| Hidden dim | 896 (14 heads × 64 head dim) |
| FFN | GLUMBConv — 1×1 expand → depthwise 3×3 → GLU → 1×1 project |
| Image self-attention | GatedDeltaNet every block; full SDPA + 2D RoPE every 6th block (5, 11, 17, 23) |
| Text conditioning | Sana-style cross-attention with QK RMSNorm; learned null embedding for CFG |
| Patch size | 1 — one token per latent pixel (256 tokens at 512px, 1024 at 1024px) |
| Text encoder | Gemma 4 2B (`google/gemma-4-E2B-it`) — 300 tokens × 1536-dim |
| VAE | DC-AE f32c32 (`mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers`) — frozen |
| Timestep conditioning | adaLN-Zero from flow-matching sigma |
| Inference sampler | STORK-2 — 32 steps, solver order 2, 5 RK substeps per step |
| Dtype | bfloat16 |

Each block applies adaLN-Zero modulation → FLA self-mixer → text cross-attention → GLUMBConv FFN, with optional image SDPA on every 6th layer.

## Training details

Two-stage training on frozen DC-AE latents with per-channel mean/std normalization.

| Setting | Value |
|---|---|
| Pre-train | JourneyDB (~3.8M images, 512px → 16×16 = 256 tokens) — stopped at 52k steps, batch 96, lr 1.2e-4 |
| Fine-tune | FineT2I (~600k images, 1024px → 32×32 = 1024 tokens) — 75k scheduled, stopped at 55k, batch 24, lr 1e-4 (final published checkpoint) |
| LR schedule | Linear warmup (1000 steps) → cosine decay to 10% of peak (min_lr_ratio 0.1) |
| Fine-tune init | `--resume {pretrain step-52k ckpt} --resume-weights-only` (fresh optimizer/step/EMA) |
| Optimizer | Fused AdamW |
| Flow shift | 1.5 ([Sana](https://github.com/NVlabs/Sana)-style sigma shift) |
| Timestep sampler | Plateau logit-normal (μ=0, σ=1) — [FLUX.2](https://bfl.ai/research/representation-comparison) |
| Loss weighting | Min-SNR γ = 5.0 |
| CFG dropout | 0.1 (learned null text embedding) |
| EMA decay | 0.999 (inference weights, updated every 8 steps) |
| Gradient clip | 0.3 |
| Hardware | RTX Pro 6000 Blackwell Server Edition |
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
pip install -U torch diffusers transformers accelerate safetensors torchvision scipy
pip install git+https://github.com/fla-org/flash-linear-attention.git
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

image = pipe(
    "a cinematic alpine lake at golden hour, snow-capped peaks reflected in mirror-still water",
)[0]
image.save("output.png")
```

### Note

The transformer weights (1.25 GB) are downloaded from this repo. The VAE and Gemma 4 2B text encoder are fetched automatically from their upstream HuggingFace repos on first use (~10 GB total). Accept the [Gemma Terms of Use](https://ai.google.dev/gemma/terms) and run `huggingface-cli login` before first use.

### Parameters

```python
image = pipe(
    "a rocky coastline at sunset with crashing waves",
    steps=32,        # denoising steps — 32 recommended with STORK-2
    cfg_scale=4.5,   # classifier-free guidance scale (4.0–5.0)
    cfg_rescale=0.5, # reduces over-saturation at high CFG
    substeps=5,      # STORK-2 RK micro-steps per denoising step
    seed=42,
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

## Capabilities and limitations

Trained on **JourneyDB** (pretrain, 512px) and **FineT2I** (finetune, 1024px).

- **Strongest at** — dramatic landscapes, natural environments, and scenic/architectural scenes
- **Also works for** — many everyday objects when embedded in a scene (quality varies)
- **Not reliable** — humans, portraits, and people (faces and anatomy are inconsistent)
- **Less reliable** — animals, legible text in images, very fine small details
- Landscapes can show a painterly/HDR bias from heavily post-processed stock images in the training set
- Not safety filtered — outputs may reflect biases in the training data
- Maximum tested resolution: **1024×1024px**

---

## Acknowledgements

- **STORK-2** — inference sampling ([Tan et al., 2025](https://arxiv.org/abs/2505.24210))
- **Plateau logit-normal** — training timestep distribution from [FLUX.2 representation comparison](https://bfl.ai/research/representation-comparison) (Black Forest Labs, 2025). Boomer uses μ=0, σ=1 with flow shift 1.5.
- **Sana** — flow shift, logit-normal sampling, forward noising, and related helpers adapted from [NVlabs/Sana](https://github.com/NVlabs/Sana) (`boomer/sana_flow.py`, `boomer/sana_latent_cache.py`); several DiT blocks follow Sana conventions ([Xie et al., 2024](https://arxiv.org/abs/2410.10629))

---

## License

This is a **research-purpose project**. Model weights and code are intended for non-commercial research and personal experimentation only.

**Model weights** — research and personal use only. Commercial use requires explicit permission.

**Source code in this repository** — [MIT License](LICENSE).

**Third-party components** (also see [Acknowledgements](#acknowledgements)):

| Component | License / terms |
|---|---|
| STORK scheduler (`STORKScheduler.py`) | Apache 2.0 — [ZT220501/STORK](https://github.com/ZT220501/STORK) |
| Sana-derived helpers (`boomer/sana_flow.py`, `boomer/sana_latent_cache.py`) | Apache 2.0 — [NVlabs/Sana](https://github.com/NVlabs/Sana) |
| Flash Linear Attention (runtime dependency) | MIT — [fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention) |
| DC-AE VAE (`mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers`) | See [model card](https://huggingface.co/mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers) |
| Gemma 4 text encoder (`google/gemma-4-E2B-it`) | [Gemma Terms of Use](https://ai.google.dev/gemma/terms) |

Using the full inference pipeline requires complying with all upstream terms above.
