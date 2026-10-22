---
license: other
task_categories:
  - text-to-image
language:
  - en
tags:
  - text-to-image
  - diffusion
  - flow-matching
  - latent-cache
  - webdataset
  - dc-ae
  - qwen
  - dinov2
  - repa
size_categories:
  - 1M<n<10M
pretty_name: Boomer Monet Mix (precomputed latent/text/DINO cache)
---

# boomer-monet-mix5m

A **precomputed WebDataset cache** for training [Boomer](https://github.com/Akrao9/Boomer-Text-to-image-Hybdrid-Linear-Diffusion-Model)
V3 — a 704M-parameter flow-matching text-to-image DiT. Each sample bundles a
**DC-AE image latent**, a **Qwen3.5-2B text embedding**, and a **DINOv2 image
embedding** (for REPA), so training never has to run the VAE or the text encoder
on the fly.

Derived from [`jasperai/monet`](https://huggingface.co/datasets/jasperai/monet)
v1.1.0 by proportionally mixing **all seven subsets**, applying quality filters,
and re-encoding the captions through Qwen3.5-2B.

> **Heads-up:** this build was targeted at 5M samples but stopped at
> **~3.07M** (a recoverable upload stall during caching). The cache is complete
> and internally consistent — the proportional mix is preserved across the
> shards that landed — it's just smaller than the original 5M target.

## Contents

| | |
|---|---|
| Samples | **3,067,446** |
| Shards | 2,074 × `monet-mix-r00-NNNNNN.tar` (~1 GB each) |
| Format | [WebDataset](https://github.com/webdataset/webdataset) |
| Source | `jasperai/monet` v1.1.0 (all 7 subsets) |

### Per-sample members

| key | shape | dtype | notes |
|---|---|---|---|
| `latents.npy` | `[32, 16, 16]` | `bfloat16` | DC-AE f32c32 latent, **pre-scaled** (scaling_factor 0.41407, global std ≈ 0.93) — no extra normalization needed |
| `text.npy` | `[320, 2048]` | `float8_e4m3fn` | Qwen3.5-2B hidden states, final post-norm readout (`readout_layer=-1`), padded/truncated to 320 tokens. Stored as `uint8` bit-pattern; upcast to bf16 at load |
| `text_mask.npy` | `[320]` | `uint8` | 1 = real token, 0 = padding |
| `dino.npy` | `[1536]` | `float16` | DINOv2-ViT-G/14 embedding, the REPA alignment target |
| `caption.txt` | — | utf-8 | the caption text that was encoded |
| `meta.json` | — | json | `source`, `key`, `caption_key`, `aesthetic`, `nsfw`, `sscd_cluster_id` |

Repo root also contains `manifest.json` (config + fingerprint) and
`null_text.npy` / `null_mask.npy` (the empty-caption embedding for
classifier-free guidance dropout).

## Mix composition

Target weights (the realized proportions track these, since the scheduler pulls
from whichever subset is furthest below its target ratio):

| subset | weight | caption source (typical) |
|---|---|---|
| synthetic | 0.315 | `caption_original` |
| commoncatalog-cc-by | 0.285 | `caption_internvl-3-8b` |
| laion | 0.153 | `caption_gemini-2.5-flash-lite` |
| coyo | 0.132 | `caption_gemini-2.5-flash-lite` |
| megalith10m | 0.061 | `caption_gemini-2.5-flash-lite` |
| cc12m | 0.051 | `caption_gemini-2.5-flash-lite` |
| diffusion-aesthetic-4k | 0.003 | `caption_gemini-2.5-flash-lite` |

## How it was built

- **Captions:** preferred `caption_gemini-2.5-flash-lite`, falling back per row to
  `internvl-3-8b → sharegpt4v-7b → florence-2-large → caption_original`
  (subsets differ in which captioners ran; commoncatalog has no Gemini caption,
  so it lands on InternVL).
- **Filters:** aesthetic gate `aesthetic_laion-improved >= 4.5`, plus near-duplicate
  removal by `sscd_cluster_id` (one image per cluster).
- **Text encoding:** Qwen3.5-2B, instruction-prefixed chat template, `max_length=320`.
  At 320, ~3% of captions are truncated (the Gemini captions are verbose; 320 is
  the chosen elbow between truncation and storage).
- **fp8 fidelity:** the final post-norm readout has abs-max ≈ 77.5 (well within
  fp8 e4m3fn's ±448), and a round-trip through fp8 preserves cosine similarity
  ≈ 0.9996 (mean) with 0 out-of-range NaNs. Verified at build time
  (transformers 5.12.1) and recorded in `manifest.json`.

## Intended use

Drop-in training cache for Boomer V3. The model's text conditioning is built for
`[320, 2048]` — a model trained on a 224-length cache is **not** interchangeable
with this one.

### Loading (WebDataset)

```python
import webdataset as wds, numpy as np, torch

url = "https://huggingface.co/datasets/Akrao9/boomer-monet-mix5m/resolve/main/monet-mix-r00-{000000..002073}.tar"

def decode(sample):
    latent = torch.from_numpy(np.load(io.BytesIO(sample["latents.npy"])))  # bf16 [32,16,16]
    text   = torch.from_numpy(np.load(io.BytesIO(sample["text.npy"])).view(np.uint8)) \
                  .view(torch.float8_e4m3fn).to(torch.bfloat16)            # [320,2048]
    mask   = torch.from_numpy(np.load(io.BytesIO(sample["text_mask.npy"])))# [320]
    dino   = torch.from_numpy(np.load(io.BytesIO(sample["dino.npy"])))     # fp16 [1536]
    return latent, text, mask, dino

ds = wds.WebDataset(url).map(decode)
```

(Boomer's own loader handles the fp8 upcast and prefetch/streaming; this snippet
is illustrative.)

## Provenance & licensing

All images, latents, and DINO embeddings originate from
[`jasperai/monet`](https://huggingface.co/datasets/jasperai/monet), which
aggregates multiple source datasets (CommonCatalog CC-BY, LAION, COYO, CC12M,
Megalith-10M, a diffusion-aesthetic set, and synthetic generations). Licensing
and usage terms are inherited from those upstream sources and the monet dataset
itself — **consult the source dataset cards before any redistribution or
commercial use.** This cache is a derived, re-encoded artifact for model
training and adds no new content rights.

## Citation

If you use the underlying data, please cite the
[`jasperai/monet`](https://huggingface.co/datasets/jasperai/monet) dataset and
its upstream sources.
