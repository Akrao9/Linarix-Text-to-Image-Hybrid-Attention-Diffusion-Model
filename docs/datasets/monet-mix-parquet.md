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
  - parquet
  - dc-ae
  - qwen
  - dinov2
  - repa
size_categories:
  - 1M<n<10M
pretty_name: Boomer Monet Mix (precomputed latent/text/DINO cache, Parquet)
---

# boomer-monet-mix5m-parquet

A **precomputed Parquet cache** for training [Boomer](https://github.com/Akrao9/Boomer-Text-to-image-Hybdrid-Linear-Diffusion-Model)
V3 — a 704M-parameter flow-matching text-to-image DiT. Each row bundles a
**DC-AE image latent**, a **Qwen3.5-2B text embedding**, and a **DINOv2 image
embedding** (for REPA), so training never has to run the VAE or the text encoder
on the fly.

This is the **range-streamable Parquet edition** of
[`boomer-monet-mix5m`](https://huggingface.co/datasets/Akrao9/boomer-monet-mix5m).
Same source, same mix, same filters, same dtypes, same `max_length=320` — only
the container differs. Parquet is range-read through HF `datasets` (`streaming=True`),
which honors `HF_HUB_DOWNLOAD_TIMEOUT` and streams in ~128–330 MB row groups, so
it's both faster and stall-free compared to whole-shard WebDataset tar downloads.

Derived from [`jasperai/monet`](https://huggingface.co/datasets/jasperai/monet)
v1.1.0 by proportionally mixing **all seven subsets**, applying quality filters,
and re-encoding the captions through Qwen3.5-2B.

## Contents

| | |
|---|---|
| Samples | **4,997,788** (full re-encode from source; 5M target) |
| Shards | 1,688 × `monet-mix-r00-NNNNNN.parquet` (2 GB uncompressed budget → ~1.6 GB on disk, zstd) |
| Row group | 512 rows (~330 MB) — the range-read unit |
| Format | [Parquet](https://parquet.apache.org/), one row per sample |
| Source | `jasperai/monet` v1.1.0 (all 7 subsets) |

### Per-sample columns

Arrays are stored as their **`.npy` bytes in binary columns** (self-describing
dtype/shape — `np.load(io.BytesIO(col))` round-trips byte-identically to the tar
members); `caption` is a utf-8 string column and `meta` a JSON-string column
(`column_encoding=npy_bytes` in the manifest).

| column | shape | dtype | notes |
|---|---|---|---|
| `latents` | `[32, 16, 16]` | `bfloat16` | DC-AE f32c32 latent, **pre-scaled** (scaling_factor 0.41407, global std ≈ 0.93) — no extra normalization needed |
| `text` | `[320, 2048]` | `float8_e4m3fn` | Qwen3.5-2B hidden states, final post-norm readout (`readout_layer=-1`), padded/truncated to 320 tokens. Stored as `uint8` bit-pattern; upcast to bf16 at load |
| `text_mask` | `[320]` | `uint8` | 1 = real token, 0 = padding |
| `dino` | `[1536]` | `float16` | DINOv2-ViT-G/14 embedding, the REPA alignment target |
| `caption` | — | utf-8 string | the caption text that was encoded |
| `meta` | — | JSON string | `source`, `key`, `caption_key`, `aesthetic`, `nsfw`, `sscd_cluster_id` |

Repo root also contains `manifest.json` (config + fingerprint) and
`null_text.npy` / `null_mask.npy` (the empty-caption embedding for
classifier-free guidance dropout).

## Mix composition

Target weights and the realized sample counts from this build (the scheduler
pulls from whichever subset is furthest below its target ratio, so realized
proportions track the targets closely):

| subset | weight | realized | caption source (typical) |
|---|---|---|---|
| synthetic | 0.315 | 1,575,000 | `caption_original` |
| commoncatalog-cc-by | 0.285 | 1,425,000 | `caption_internvl-3-8b` |
| laion | 0.153 | 765,000 | `caption_gemini-2.5-flash-lite` |
| coyo | 0.132 | 660,000 | `caption_gemini-2.5-flash-lite` |
| megalith10m | 0.061 | 305,000 | `caption_gemini-2.5-flash-lite` |
| cc12m | 0.051 | 255,000 | `caption_gemini-2.5-flash-lite` |
| diffusion-aesthetic-4k | 0.003 | 12,788 | `caption_gemini-2.5-flash-lite` |

Realized caption sources across all 5M rows:
`gemini-2.5-flash-lite` 1,827,124 · `original` 1,575,000 ·
`internvl-3-8b` 1,559,210 · `florence-2-large` 34,087 · `sharegpt4v-7b` 2,367.
Build stats: `seen=5,529,660`, `filtered=531,872` (aesthetic + dedup),
`no_latent=0`, `no_caption=0`.

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

### Loading (HF `datasets`, range-streamed)

```python
import io, numpy as np, torch
from datasets import load_dataset

repo = "Akrao9/boomer-monet-mix5m-parquet"
ds = load_dataset(
    "parquet",
    data_files={"train": f"hf://datasets/{repo}/*.parquet"},
    split="train",
    streaming=True,                 # range reads, honors HF_HUB_DOWNLOAD_TIMEOUT
).shuffle(seed=0, buffer_size=2000)

def decode(row):
    latent = torch.from_numpy(np.load(io.BytesIO(row["latents"])))            # bf16 [32,16,16]
    text   = torch.from_numpy(np.load(io.BytesIO(row["text"])).view(np.uint8)) \
                  .view(torch.float8_e4m3fn).to(torch.bfloat16)               # [320,2048]
    mask   = torch.from_numpy(np.load(io.BytesIO(row["text_mask"])))          # [320]
    dino   = torch.from_numpy(np.load(io.BytesIO(row["dino"])))               # fp16 [1536]
    return latent, text, mask, dino

for row in ds:
    latent, text, mask, dino = decode(row)
    break
```

Boomer's own loader (`ParquetStreamingLatentCacheDataset`) handles the fp8 upcast,
shuffle buffer, per-epoch reshuffle, and `DataLoader`-worker sharding. Point
training at the glob and the loader auto-routes the Parquet path:

```bash
python train.py --cache-wds "hf://datasets/Akrao9/boomer-monet-mix5m-parquet/*.parquet" \
                --num-workers 8 --text-seq-len 320 ...
```

(The tar-only flags `--wds-prefetch-shards / --wds-download-workers /
--wds-decode-queue / --wds-local-dir` don't apply here; `datasets` shards the
Parquet files across workers itself.)

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
</content>
</invoke>
