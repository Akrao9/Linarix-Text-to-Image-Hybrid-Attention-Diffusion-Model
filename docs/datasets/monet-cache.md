---
license: apache-2.0
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
  - repa
size_categories:
  - 1M<n<10M
pretty_name: Boomer MONET Latent/Text Cache
---

# Boomer MONET Latent/Text Cache

A **precomputed training cache** for the [Boomer](https://huggingface.co/) text-to-image
flow-matching model. Each sample bundles everything the denoiser needs so training is a
pure GPU loop with no image decoding, VAE, or text-encoder forward passes at train time:

- a **DC-AE f32c32 image latent** (reused directly from [`jasperai/monet`](https://huggingface.co/datasets/jasperai/monet)'s precomputed SANA-VAE latents),
- the **Qwen3.5-2B text hidden states** for the caption (computed here), and
- the **DINOv2 ViT-G/14 embedding** for an optional [REPA](https://arxiv.org/abs/2410.06940) representation-alignment loss.

Stored as **WebDataset** tar shards, consumed directly by Boomer's
`boomer/datasets/wds_latents.py` loader.

## Provenance

Derived from **[jasperai/monet](https://huggingface.co/datasets/jasperai/monet)** (Apache-2.0),
subset **`commoncatalog-cc-by`** (real Flickr / YFCC100M images, CC-BY-4.0). MONET's latents were
verified drop-in compatible with Boomer's DC-AE f32c32 decoder, so they are reused unchanged; only
the text and the layout are produced here.

| Stage | Encoder | Notes |
|---|---|---|
| Image latent | DC-AE f32c32 (SANA-VAE) | reused from MONET; **not** re-encoded |
| Text | **Qwen3.5-2B** | instruction-prefixed hidden states, readout layer −1, `max_length=224` |
| REPA target | **DINOv2 ViT-G/14** | pooled CLS (1536-d), reused from MONET |
| Caption | Gemini-2.5-flash-lite → InternVL-3-8B → ShareGPT4V-7B → Florence-2 → original | first non-empty per image; recorded in `meta.json` |

Images are **512×512** (latent grid 16×16 × 32 channels).

## Shard layout

WebDataset tar shards (~4 GB each), one sample = these members:

| Member | Contents | dtype on disk | Shape |
|---|---|---|---|
| `latents.npy` | DC-AE f32c32 latent | bfloat16 (uint16 bits) | `(32, 16, 16)` |
| `text.npy` | Qwen3.5-2B hidden states | float8_e4m3fn (uint8 bits) | `(224, 2048)` |
| `text_mask.npy` | valid-token mask | uint8 | `(224,)` |
| `dino.npy` | DINOv2 ViT-G/14 CLS (REPA) | float16 | `(1536,)` |
| `caption.txt` | the caption string | utf-8 | — |
| `meta.json` | source, key, caption source, aesthetic/nsfw/cluster | json | — |

Alongside the tars, each prefix folder contains:

- **`manifest.json`** — dtypes/shapes the loader reads to decode the raw tensors.
- **`null_text.npy` / `null_mask.npy`** — the empty-caption embedding for classifier-free guidance.

> Tensors are stored as raw bit-patterns (bf16→uint16, fp8→uint8); the loader restores the
> logical dtype from `manifest.json` and upcasts fp8 text to bf16 on read.

## Usage

With the Boomer loader (handles manifest, dtype upcast, and DINO):

```python
from boomer.datasets.wds_latents import StreamingWdsLatentCacheDataset
from boomer.datasets.latent_shards import collate_latent_records
from torch.utils.data import DataLoader

ds = StreamingWdsLatentCacheDataset(
    paths="hf://datasets/<your-user>/<this-repo>/commoncatalog-cc-by/*.tar",
    hf_token=True, shardshuffle=True, shuffle_buffer=2000,
)
dl = DataLoader(ds, batch_size=64, num_workers=8, collate_fn=collate_latent_records)
batch = next(iter(dl))
# batch["image_latent"]  -> (B, 32, 16, 16) bf16
# batch["text_embedding"]-> (B, 224, 2048)  bf16
# batch["attention_mask"]-> (B, 224)
# batch["dino_embedding"]-> (B, 1536)        (REPA target; None if absent)
```

Or train directly:

```bash
python train.py \
  --cache-wds "hf://datasets/<your-user>/<this-repo>/commoncatalog-cc-by/*.tar" \
  --repa-dim 1536 --repa-weight 0.5 --repa-anneal-steps 50000 \
  --hidden-dim 896 --depth 24 --mixer-type fla_gdn2 --hf-token true
```

Plain WebDataset access also works (`latents.npy`, `text.npy`, etc. are standard `.npy`).

## Intended use

Pretraining / prototyping Boomer-family latent flow-matching DiTs at 512×512, with optional
DINOv2 **REPA** warm-up (pooled CLS alignment) to accelerate early convergence.

## Limitations

- **REPA is pooled, not patch-level** — MONET ships the DINOv2 CLS vector only, so this supports
  pooled-REPA (weaker than canonical patch-REPA).
- **Text is fp8** for storage; cosine-level fidelity is preserved but not bit-exact vs bf16.
- **Captions truncated to 224 tokens** (incl. the Qwen instruction prefix); a small fraction of
  long captions lose their tail.
- Inherits MONET's biases (Western skew, web provenance) and the `commoncatalog-cc-by` distribution.
- The latent space is tied to **DC-AE f32c32 / SANA-VAE** — decode only with that VAE.

## Licensing

Apache-2.0. Source images are CC-BY-4.0 (MONET `commoncatalog-cc-by`); MONET is Apache-2.0; the
Qwen and DINOv2 features derive from Apache-2.0 models; DC-AE/SANA is MIT. Users remain responsible
for their own due diligence on upstream terms.

## Citation

Built on MONET:

```bibtex
@article{aubin2026monet,
  title   = {MONET: A Massive, Open, Non-redundant and Enriched Text-to-image Dataset},
  author  = {Aubin, Benjamin and Quintana, Gonzalo I{\~n}aki and Tasar, Onur and Sreetharan, Sanjeev and Czerwinska, Urszula and Henry, Damien and Chadebec, Cl{\'e}ment},
  journal = {arXiv preprint arXiv:2605.21272},
  year    = {2026},
  note    = {Jasper Research}
}
```
