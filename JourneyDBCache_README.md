---
license: other
task_categories:
  - text-to-image
  - image-to-image
language:
  - en
tags:
  - journeydb
  - webdataset
  - latent-cache
  - text-to-image
  - dc-ae
  - gemma
  - boomer
size_categories:
  - 1M<n<10M
---

# JourneyDBCache

Precomputed WebDataset cache for Boomer text-to-image training.

This dataset contains cached DC-AE image latents and cached Gemma text hidden states generated from JourneyDB-style image-caption WebDataset shards. It is intended to remove the frozen encoder cost from Boomer training.

## Status

This is a partial cache created before a Colab Pro+ session limit stopped the full run.

| Field | Value |
|---|---:|
| Written samples | 3,955,000 |
| Skipped source samples | 849 |
| Tar shard files | 1,728 |
| Resolution | 512 x 512 |
| Image latent shape | `[32, 16, 16]` |
| Text embedding shape | `[300, 1536]` |
| Latent dtype | `float16` |
| Text dtype | `float16` |
| Text mask dtype | `uint8` |
| Format | WebDataset `.tar` shards |

The cache is large enough for serious Boomer training, but it is not the complete JourneyDB training split.

## Source

Source stream:

```text
hf://datasets/BLIP3o/BLIP3o-Pretrain-JourneyDB/**/*.tar
```

The original source dataset and captions are not owned by this repository. This repository only stores derived training cache artifacts.

## Encoders

Image encoder:

```text
mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers
```

Text encoder:

```text
google/gemma-4-E2B-it
```

Text hidden states are cached directly so training does not need to run Gemma online.

## Preprocessing

Images were decoded, resized, cropped, normalized, and encoded as follows:

```text
raw image
-> RGB
-> resize-to-fill target resolution
-> center crop to 512 x 512
-> normalize pixels from [0, 255] to [-1, 1]
-> DC-AE encode
-> multiply by DC-AE scaling_factor 0.41407
-> store latent as float16
```

The cache run used the optimized GPU decode path:

```text
--gpu-decode
--channels-last
```

The GPU path preserves the same resize-to-fill plus center-crop geometry as the CPU/Sana-style path, but interpolation is not bit-identical to PIL.

Text captions were encoded with Gemma 4 E2B using:

```text
max_length = 300
Sana-style BOS + trailing-token selection
hidden state shape = [300, 1536]
```

## Shard Schema

Each sample in each tar shard contains:

```text
latents.npy       float16 [32, 16, 16]
text.npy          float16 [300, 1536]
text_mask.npy     uint8   [300]
caption.txt       UTF-8 caption
meta.json         per-sample metadata
```

Top-level auxiliary files:

```text
manifest.json
null_text.npy
null_mask.npy
```

`null_text.npy` and `null_mask.npy` contain the unconditional embedding for an empty prompt and can be used for classifier-free guidance dropout.

## Cache Command

The cache was produced with the Boomer script:

```bash
python tools/cache_hf_wds_to_boomer_wds.py \
  --source-glob "hf://datasets/BLIP3o/BLIP3o-Pretrain-JourneyDB/**/*.tar" \
  --output-dir "/content/boomer_wds_cache/train_512_gemma4_full" \
  --max-samples 4270000 \
  --skip-samples 0 \
  --batch-size 512 \
  --image-encode-batch 96 \
  --text-encode-batch 192 \
  --num-workers 16 \
  --prefetch-factor 4 \
  --source-chunk-size 16 \
  --pin-memory \
  --async-flush \
  --async-flush-depth 2 \
  --gpu-decode \
  --channels-last \
  --text-attn-implementation sdpa \
  --shard-size-gb 2.0 \
  --hf-repo-id "akrao9/JourneyDBCache" \
  --hf-token true \
  --create-repo \
  --delete-uploaded \
  --hf-upload-workers 6 \
  --log-every 5000
```

The run stopped at:

```text
written_samples = 3,955,000
skipped_samples = 849
complete = false
```

## Loading

Example WebDataset inspection with Boomer's URL resolver:

```python
import webdataset as wds

from boomer.datasets.wds_latents import resolve_wds_urls, sample_to_record

urls = resolve_wds_urls(
    "hf://datasets/akrao9/JourneyDBCache/train/*.tar",
    hf_token=True,
)
sample = next(iter(wds.WebDataset(urls[:1], shardshuffle=False).decode()))
record = sample_to_record(sample)

print(record.image_latent.shape, record.image_latent.dtype)
print(record.text_embedding.shape, record.text_embedding.dtype)
print(record.attention_mask.shape, record.attention_mask.dtype)
print(record.caption)
print(record.metadata)
```

## Boomer Training

Use Boomer's streaming WDS loader:

```bash
python tools/overfit_latent_dit.py \
  --cache-wds "hf://datasets/akrao9/JourneyDBCache/train/*.tar" \
  --model-type boomer_fla \
  --hidden-dim 1152 \
  --depth 28 \
  --num-heads 16 \
  --mlp-ratio 2.5 \
  --mixer-type fla_gated_deltanet \
  --fla-mode chunk \
  --use-short-conv \
  --conv-size 4 \
  --image-attn-every 4 \
  --image-attn-backend sdpa \
  --cross-attn-backend sdpa \
  --cross-attn-qk-norm \
  --batch-size 64 \
  --num-workers 4 \
  --prefetch-factor 2 \
  --wds-shard-shuffle 128 \
  --wds-shuffle-buffer 4096 \
  --latent-stats-json /path/to/stats.json \
  --latent-stats-mode channel \
  --dtype bf16 \
  --flow-shift 3.0 \
  --y-norm \
  --hf-token true
```

`text_dim` is inferred from the cached Gemma hidden states (`1536` for this dataset); `--hidden-dim 1152` is the Boomer model width.

Streaming WDS shards are split across DataLoader workers, so `--num-workers > 0` is safe. Start with `--num-workers 4` on Colab and tune `--prefetch-factor` if the GPU waits on I/O.

## Latent Normalization

Compute latent statistics from this cache before training:

```bash
python tools/inspect_latent_stats.py \
  --cache-wds "hf://datasets/akrao9/JourneyDBCache/train/*.tar" \
  --max-records 100000 \
  --json-out /path/to/stats.json \
  --hf-token true
```

Use the JSON during training:

```text
--latent-stats-json /path/to/stats.json
--latent-stats-mode channel
```

Recommended sample size: `100,000` cached records.

This cache used the GPU decode path, so compute stats from this cache rather than reusing stats from an older CPU/PIL cache.

## Limitations

- This cache is partial: `complete=false`.
- The source was web-scale data; a small number of corrupt or missing records were skipped.
- The cache stores derived latent/text features, not original images.
- `latents.npy` and `text.npy` are intended for Boomer-compatible training code.
- Do not assume compatibility with Sana checkpoints; the text encoder is Gemma 4 E2B and the Boomer model path differs from upstream Sana.

## Intended Use

This dataset is intended for research and development of the Boomer latent flow text-to-image model. Users are responsible for complying with the terms and licenses of the upstream source dataset and encoder models.
