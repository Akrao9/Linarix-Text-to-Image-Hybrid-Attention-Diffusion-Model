# Boomer

Boomer is a Flame-backed text-to-image research project. The first milestone is a
Sana-compatible latent cache pipeline using Hugging Face models:

- image encoder/decoder: `mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers`
- text encoder: `google/gemma-4-E2B-it`

## First Smoke Test

From this folder:

```bash
python -m boomer.sana_latent_cache \
  --image /path/to/image.jpg \
  --caption "a detailed caption for the image" \
  --out cache/debug/sample.pt \
  --decoded-out cache/debug/reconstruction.png
```

This loads the Hugging Face encoders, creates one cached training record, reloads
it, and decodes the image latent back to a reconstruction.

Gemma 4 may require a recent Transformers install:

```bash
pip install -U transformers torch accelerate
```

Inspect a cache file before training:

```bash
python tools/inspect_latent_cache.py cache/debug/sample.pt
```

Inspect the Hugging Face JourneyDB mirror before caching:

```bash
python tools/inspect_hf_dataset.py \
  --dataset BLIP3o/BLIP3o-Pretrain-JourneyDB \
  --split train \
  --streaming \
  --save-image /content/drive/MyDrive/Boomer/cache/debug/hf_sample.png
```

Cache a tiny Hugging Face dataset slice:

```bash
python tools/cache_hf_dataset.py \
  --dataset BLIP3o/BLIP3o-Pretrain-JourneyDB \
  --split train \
  --streaming \
  --limit 32 \
  --out-dir /content/drive/MyDrive/Boomer/cache/journeydb_512_debug \
  --image-columns jpg,image,png \
  --caption-columns txt,text,prompt,caption,original_prompt,generated_text
```

Or cache local WebDataset tar shards:

```bash
python tools/cache_hf_dataset.py \
  --dataset webdataset \
  --data-files "/content/drive/MyDrive/Boomer/raw_journeydb/*.tar" \
  --split train \
  --no-streaming \
  --num-proc 8 \
  --cache-dir /content/hf_cache \
  --limit 32 \
  --out-dir /content/drive/MyDrive/Boomer/cache/journeydb_512_debug \
  --image-columns jpg,image,png \
  --caption-columns txt,text,prompt,caption
```

Overfit the first LatentDiT baseline on a tiny cache:

```bash
python tools/overfit_latent_dit.py \
  --cache /content/boomer_debug_cache \
  --steps 200 \
  --batch-size 4 \
  --save /content/boomer_debug_latent_dit.pt
```

Longer debug training with logs, periodic checkpoints, and denoise previews:

```bash
python tools/overfit_latent_dit.py \
  --cache /content/boomer_debug_cache \
  --steps 2000 \
  --batch-size 4 \
  --lr 3e-4 \
  --log-every 20 \
  --loss-window 100 \
  --log-csv /content/boomer_train_debug/loss.csv \
  --save-every 500 \
  --checkpoint-dir /content/boomer_train_debug/checkpoints \
  --preview-every 500 \
  --preview-cache-record /content/boomer_debug_cache/00000000_d8dfa99fb3d6.pt \
  --preview-out-dir /content/boomer_train_debug/previews \
  --preview-t 0.3 \
  --preview-steps 8 \
  --hf-token true \
  --save /content/boomer_train_debug/latent_dit_final.pt
```

Cross-attention variant over full Gemma tokens:

```bash
python tools/overfit_latent_dit.py \
  --cache /content/boomer_cache_512 \
  --steps 10000 \
  --batch-size 16 \
  --lr 3e-4 \
  --cross-attn \
  --cond-dropout 0.1 \
  --ema-decay 0.999 \
  --log-every 50 \
  --loss-window 200 \
  --log-csv /content/boomer_train_512_cross/loss.csv \
  --save-every 1000 \
  --checkpoint-dir /content/boomer_train_512_cross/checkpoints \
  --preview-every 1000 \
  --preview-cache-record /content/boomer_cache_512/PASTE_FIRST_CACHE_RECORD.pt \
  --preview-out-dir /content/boomer_train_512_cross/previews \
  --preview-t 0.3 \
  --preview-steps 8 \
  --hf-token true \
  --save /content/boomer_train_512_cross/latent_dit_final.pt
```

Recommended Colab notebook-style next run after smudged random samples:

```python
from pathlib import Path

CACHE_DIR = "/content/boomer_cache_2k"
TRAIN_DIR = "/content/boomer_train_2k_cross_cfg_ema"
DRIVE_CKPT_DIR = "/content/drive/MyDrive/Boomer/checkpoints/boomer_train_2k_cross_cfg_ema"
```

```python
!python tools/cache_hf_dataset.py \
  --dataset BLIP3o/BLIP3o-Pretrain-JourneyDB \
  --split train \
  --streaming \
  --limit 2048 \
  --batch-size 8 \
  --out-dir "{CACHE_DIR}" \
  --hf-token true
```

```python
FIRST_CACHE = str(next(Path(CACHE_DIR).glob("*.pt")))
FIRST_CACHE
```

```python
!python tools/overfit_latent_dit.py \
  --cache "{CACHE_DIR}" \
  --steps 20000 \
  --batch-size 32 \
  --lr 2e-4 \
  --cross-attn \
  --cond-dropout 0.1 \
  --ema-decay 0.999 \
  --log-every 50 \
  --loss-window 200 \
  --log-csv "{TRAIN_DIR}/loss.csv" \
  --save-every 1000 \
  --checkpoint-dir "{DRIVE_CKPT_DIR}/checkpoints" \
  --preview-every 1000 \
  --preview-cache-record "{FIRST_CACHE}" \
  --preview-out-dir "{TRAIN_DIR}/previews" \
  --preview-t 0.3 \
  --preview-steps 8 \
  --hf-token true \
  --save "{DRIVE_CKPT_DIR}/latent_dit_final.pt"
```

```python
!python tools/sample_latent_dit.py \
  --checkpoint "{DRIVE_CKPT_DIR}/latent_dit_final.pt" \
  --condition-cache "{FIRST_CACHE}" \
  --out "{TRAIN_DIR}/sample_cfg2.png" \
  --steps 96 \
  --sampler euler \
  --cfg-scale 2.0 \
  --weights auto \
  --seed 0 \
  --hf-token true
```

Compare Heun against Euler at the same seed. Heun uses two model calls per step, so 48 Heun steps is similar cost to 96 Euler steps:

```python
!python tools/sample_latent_dit.py \
  --checkpoint "{DRIVE_CKPT_DIR}/latent_dit_final.pt" \
  --condition-cache "{FIRST_CACHE}" \
  --out "{TRAIN_DIR}/sample_heun_cfg2.png" \
  --steps 48 \
  --sampler heun \
  --cfg-scale 2.0 \
  --weights auto \
  --seed 0 \
  --hf-token true
```

Sana-style flow run after blurry high-noise samples:

```python
from pathlib import Path

CACHE_DIR = "/content/boomer_cache_10k"
STATS_JSON = "/content/boomer_cache_10k_stats.json"
TRAIN_DIR = "/content/boomer_train_10k_sana_flow"
DRIVE_CKPT_DIR = "/content/drive/MyDrive/Boomer/checkpoints/boomer_train_10k_sana_flow"
```

```python
!python tools/cache_hf_dataset.py \
  --dataset BLIP3o/BLIP3o-Pretrain-JourneyDB \
  --split train \
  --streaming \
  --limit 10000 \
  --batch-size 8 \
  --out-dir "{CACHE_DIR}" \
  --hf-token true
```

```python
!python tools/inspect_latent_stats.py "{CACHE_DIR}" \
  --json-out "{STATS_JSON}"
```

```python
FIRST_CACHE = str(next(Path(CACHE_DIR).glob("*.pt")))
FIRST_CACHE
```

```python
!python tools/overfit_latent_dit.py \
  --cache "{CACHE_DIR}" \
  --steps 50000 \
  --batch-size 32 \
  --lr 1e-4 \
  --hidden-dim 768 \
  --depth 12 \
  --num-heads 12 \
  --cross-attn \
  --cond-dropout 0.1 \
  --ema-decay 0.999 \
  --latent-stats-json "{STATS_JSON}" \
  --latent-stats-mode channel \
  --t-sampler logit-normal \
  --flow-shift 3.0 \
  --logit-mean 0.0 \
  --logit-std 1.0 \
  --y-norm \
  --y-norm-scale-factor 0.01 \
  --grad-clip 0.1 \
  --log-every 50 \
  --loss-window 200 \
  --log-csv "{TRAIN_DIR}/loss.csv" \
  --save-every 1000 \
  --checkpoint-dir "{DRIVE_CKPT_DIR}/checkpoints" \
  --preview-every 1000 \
  --preview-cache-record "{FIRST_CACHE}" \
  --preview-out-dir "{TRAIN_DIR}/previews" \
  --preview-t 0.8 \
  --preview-steps 32 \
  --hf-token true \
  --save "{DRIVE_CKPT_DIR}/latent_dit_final.pt"
```

Sampling automatically reuses the checkpoint's flow shift, latent stats, and y-norm settings:

```python
!python tools/sample_latent_dit.py \
  --checkpoint "{DRIVE_CKPT_DIR}/latent_dit_final.pt" \
  --condition-cache "{FIRST_CACHE}" \
  --out "{TRAIN_DIR}/sample_heun_cfg2.png" \
  --steps 96 \
  --sampler heun \
  --cfg-scale 2.0 \
  --weights auto \
  --seed 0 \
  --hf-token true
```

```python
!python tools/sample_latent_dit.py \
  --checkpoint "{DRIVE_CKPT_DIR}/latent_dit_final.pt" \
  --condition-cache "{FIRST_CACHE}" \
  --out "{TRAIN_DIR}/sample_heun_cfg15.png" \
  --steps 64 \
  --sampler heun \
  --cfg-scale 1.5 \
  --weights auto \
  --seed 0 \
  --hf-token true
```

If Euler and Heun both remain smudged, check latent scale and move to a 10k cache:

```python
from pathlib import Path

CACHE_DIR = "/content/boomer_cache_10k"
TRAIN_DIR = "/content/boomer_train_10k_cross_cfg_ema_norm"
DRIVE_CKPT_DIR = "/content/drive/MyDrive/Boomer/checkpoints/boomer_train_10k_cross_cfg_ema_norm"
STATS_JSON = "/content/boomer_cache_10k_stats.json"
```

```python
!python tools/cache_hf_dataset.py \
  --dataset BLIP3o/BLIP3o-Pretrain-JourneyDB \
  --split train \
  --streaming \
  --limit 10000 \
  --batch-size 8 \
  --out-dir "{CACHE_DIR}" \
  --hf-token true
```

```python
!python tools/inspect_latent_stats.py "{CACHE_DIR}" --json-out "{STATS_JSON}"
```

```python
import json

stats = json.load(open(STATS_JSON))
FIRST_CACHE = str(next(Path(CACHE_DIR).glob("*.pt")))
stats["global_std"], stats["channel_std_min"], stats["channel_std_max"], FIRST_CACHE
```

```python
!python tools/overfit_latent_dit.py \
  --cache "{CACHE_DIR}" \
  --steps 30000 \
  --batch-size 32 \
  --lr 2e-4 \
  --cross-attn \
  --cond-dropout 0.1 \
  --ema-decay 0.999 \
  --latent-stats-json "{STATS_JSON}" \
  --latent-stats-mode channel \
  --log-every 50 \
  --loss-window 200 \
  --log-csv "{TRAIN_DIR}/loss.csv" \
  --save-every 1000 \
  --checkpoint-dir "{DRIVE_CKPT_DIR}/checkpoints" \
  --preview-every 1000 \
  --preview-cache-record "{FIRST_CACHE}" \
  --preview-out-dir "{TRAIN_DIR}/previews" \
  --preview-t 0.3 \
  --preview-steps 8 \
  --hf-token true \
  --save "{DRIVE_CKPT_DIR}/latent_dit_final.pt"
```

```python
!python tools/sample_latent_dit.py \
  --checkpoint "{DRIVE_CKPT_DIR}/latent_dit_final.pt" \
  --condition-cache "{FIRST_CACHE}" \
  --out "{TRAIN_DIR}/sample_heun_cfg2.png" \
  --steps 64 \
  --sampler heun \
  --cfg-scale 2.0 \
  --weights auto \
  --seed 0 \
  --hf-token true
```

If samples are still blurry, run a known-latent high-noise denoise grid before retraining:

```python
!python tools/denoise_cached_latent.py \
  --checkpoint "{DRIVE_CKPT_DIR}/latent_dit_final.pt" \
  --cache-record "{FIRST_CACHE}" \
  --out-dir "{TRAIN_DIR}/denoise_grid" \
  --t-values 0.3,0.6,0.8,0.95 \
  --steps 32 \
  --weights auto \
  --seed 0 \
  --hf-token true
```

If `t=0.8` or `t=0.95` still fails, train a high-noise-biased run:

```python
HIGH_T_DIR = "/content/boomer_train_2k_channel_norm_high_t"
HIGH_T_CKPT_DIR = "/content/drive/MyDrive/Boomer/checkpoints/boomer_train_2k_channel_norm_high_t"
```

```python
!python tools/overfit_latent_dit.py \
  --cache "{CACHE_DIR}" \
  --steps 20000 \
  --batch-size 32 \
  --lr 2e-4 \
  --cross-attn \
  --cond-dropout 0.1 \
  --ema-decay 0.999 \
  --latent-stats-json "{STATS_JSON}" \
  --latent-stats-mode channel \
  --t-sampler high \
  --t-bias 2.0 \
  --log-every 50 \
  --loss-window 200 \
  --log-csv "{HIGH_T_DIR}/loss.csv" \
  --save-every 1000 \
  --checkpoint-dir "{HIGH_T_CKPT_DIR}/checkpoints" \
  --preview-every 1000 \
  --preview-cache-record "{FIRST_CACHE}" \
  --preview-out-dir "{HIGH_T_DIR}/previews" \
  --preview-t 0.8 \
  --preview-steps 32 \
  --hf-token true \
  --save "{HIGH_T_CKPT_DIR}/latent_dit_final.pt"
```

To resume from a saved Drive checkpoint, add `--resume` and keep `--steps` as the total target step:

```python
RESUME = f"{DRIVE_CKPT_DIR}/checkpoints/latent_dit_step_010000.pt"
```

```python
!python tools/overfit_latent_dit.py \
  --cache "{CACHE_DIR}" \
  --resume "{RESUME}" \
  --steps 20000 \
  --batch-size 32 \
  --lr 2e-4 \
  --cond-dropout 0.1 \
  --ema-decay 0.999 \
  --log-every 50 \
  --loss-window 200 \
  --log-csv "{TRAIN_DIR}/loss_resume.csv" \
  --save-every 1000 \
  --checkpoint-dir "{DRIVE_CKPT_DIR}/checkpoints" \
  --preview-every 1000 \
  --preview-cache-record "{FIRST_CACHE}" \
  --preview-out-dir "{TRAIN_DIR}/previews_resume" \
  --preview-t 0.3 \
  --preview-steps 8 \
  --hf-token true \
  --save "{DRIVE_CKPT_DIR}/latent_dit_final.pt"
```

Sample and decode from the overfit checkpoint:

```bash
python tools/sample_latent_dit.py \
  --checkpoint /content/boomer_debug_latent_dit.pt \
  --condition-cache /content/boomer_debug_cache/00000000_d8dfa99fb3d6.pt \
  --out /content/boomer_debug_sample.png \
  --steps 32 \
  --seed 0 \
  --hf-token true
```

In Colab notebook-style cells:

```python
from pathlib import Path

FIRST_CACHE = str(next(Path("/content/boomer_cache_512").glob("*.pt")))
```

```python
!python tools/sample_latent_dit.py \
  --checkpoint /content/boomer_train_512_cross/latent_dit_final.pt \
  --condition-cache "{FIRST_CACHE}" \
  --out /content/boomer_train_512_cross/sample_final.png \
  --steps 64 \
  --seed 0 \
  --hf-token true
```

Denoise a known cached latent:

```bash
python tools/denoise_cached_latent.py \
  --checkpoint /content/boomer_debug_latent_dit.pt \
  --cache-record /content/boomer_debug_cache/00000000_d8dfa99fb3d6.pt \
  --out-dir /content/boomer_denoise_debug \
  --t 0.3 \
  --steps 8 \
  --seed 0 \
  --hf-token true
```
