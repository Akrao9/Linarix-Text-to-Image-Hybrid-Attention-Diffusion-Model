# Boomer V3 — REPA Training Log

Run: **`boomer-v3-repa-5m`** · 704M flow-matching FLA DiT · DC-AE f32c32 latents · Qwen3.5-2B text · DINOv2 REPA
Hardware: **RTX PRO 6000 (96 GB, sm_120 Blackwell), 175 GB RAM, Colab Pro+** · torch 2.12 / cu130

---

## Data

- **Corpus:** [`jasperai/monet`](https://huggingface.co/datasets/jasperai/monet) → cached to `hf://datasets/Akrao9/boomer-monet-cache/commoncatalog-cc-by/*.tar` (see [BOOMER_MONET_CACHE_CARD.md](BOOMER_MONET_CACHE_CARD.md)).
- **Cache complete:** ~5M samples (commoncatalog-cc-by), 4 GB WDS shards.
- **Latents:** DC-AE f32c32, pre-scaled (scaling_factor 0.41407). Verified **global std ≈ 0.93, mean ≈ 0.07** → no extra normalization (`--latent-mean/--latent-std` left at 0/1).
- **Captions:** **InternVL-3-8B** (gemini column is NULL on commoncatalog-cc-by; fallback chain selects it). 100% coverage, **0% truncated** at `--text-seq-len 224` (instr+template overhead ~48 tok, full p99 ≈ 178).
- **REPA target:** DINOv2 ViT-G/14 CLS (1536-d), precomputed in cache.

## Model (V3 architecture)

`hidden=896 depth=24 heads=14 mlp_ratio=2.5`, mixer `fla_gdn2` (bidirectional, short-conv k=4),
image self-attn every 6 blocks (+2D RoPE), Sana cross-attn (qk-norm), parallel-block, dual-stream depth 2,
multimodal coord-ids, **no abs-pos-embed** (coord-ids provide position), REPA dim 1536 @ block 12.
**Trainable: 704M, 100% from scratch.** VAE / Qwen / DINOv2 are frozen/precomputed (not in the graph).

## Final training config (LOCKED by benchmarking)

```
--batch-size 128 --gradient-checkpointing      # required; see benchmarks
--lr 1e-4 --beta1 0.9 --beta2 0.999 --fused-adamw --grad-clip 0.3
--lr-scheduler linear-warmup-constant --warmup-steps 1000
--max-steps 100000 --cond-dropout 0.1
--t-sampler plateau-logit-normal --flow-shift 1.5 --min-snr-gamma 0
--t-sampler-final uniform --t-sampler-final-step 80000     # texture-polish phase 80k-100k
--repa-dim 1536 --repa-weight 0.5 --repa-anneal-steps 50000 # REPA anneals 0.5 -> 0 by 50k
--ema-decay 0.9999 --dtype bf16
--image-attn-backend flash2                                # FA2; needs the prebuilt wheel
--compile --compile-mode max-autotune-no-cudagraphs
--save-every-steps 5000 (+ async Drive upload, keep newest 2 local)
--skip-peek (latent 32x16, text 224x2048)                  # avoids 4GB startup download
```

## Benchmarks & decisions

| Config | ms/step | img/s | peak GB | verdict |
|---|---|---|---|---|
| batch 128 + grad-ckpt (eager) | 1991 | **64.3** | **20.1** | ✅ chosen |
| batch 256 + grad-ckpt (eager) | 4073 | 62.9 | 31.4 | ✗ no throughput gain (compute-bound) |
| batch 128, NO ckpt | — | — | OOM @94 | ✗ activations exceed 95 GB |
| batch 96, NO ckpt | — | — | OOM | ✗ no-checkpoint not viable |

- **Compute-bound, not memory-bound:** step time scales linearly with batch → bigger batch buys no throughput. Batch 128 is the sweet spot.
- **Gradient checkpointing is mandatory** at batch 128 (cuts activations 94→20 GB) and the 20 GB peak leaves huge headroom for a safe unattended run.
- **Attention is only 2.7% of CUDA time** (profiled): FA2/flash2 is a negligible win, cross-attn stays sdpa. Compute is matmul/MLP (~33%) + short-conv (~19%) + FLA mixer (~13%); `--compile` fuses ~27% elementwise/cast overhead (the real speed lever).
- **Throughput estimate:** ~64 img/s eager → faster compiled. 100k steps × 128 = 12.8M images ≈ **40–45 hr** (compiled). Spans multiple Colab sessions → resume from Drive.

## Pre-flight verification (all ✅)

- Model builds (704M), batch 128 + ckpt fits (20 GB), 20-step smoke ticks `train/flow_loss`≈1.8, `train/repa_loss`≈1.0.
- Latents std≈0.93, captions clean, flash2 runs on sm_120.
- All code on remote `Akrao9/Boomer-Text-to-image-Hybdrid-Linear-Diffusion-Model` (main).

## Monitoring (W&B `boomer-monet`)

- `train/flow_loss` — denoising MSE (~1.8 init, should descend past warmup)
- `train/repa_loss` — DINOv2 cosine alignment (~1.0 init, descends)
- `train/repa_w` — REPA weight (0.5 → 0 by step 50k)
- `train/total_loss` — actual optimized objective (flow + repa_w·repa)
- `train/t_polish` — flips 0→1 at step 80k (sampler → uniform)

## Checkpoints & resume

- Local `/content/ckpt_boomer-v3-repa-5m` every 5k → async upload to `Drive/MyDrive/Boomer/checkpoints/boomer-v3-repa-5m/`, newest 2 kept local.
- **Resume after disconnect:** remount Drive, copy a `.ckpt` back, add `--resume <path>`. Do NOT wipe the checkpoint dir when resuming.

## Open items / future

- **1024-res training:** needs a re-cache at higher resolution (current cache is 512px / 16×16 latents). Token math: 1024px → 32×32 = 1024 tokens, so batch ~32 ≈ batch 128's token count at 512; batch 48–64 may fit with checkpointing.
- Per-channel latent normalization (one channel std 2.34 vs 0.80 median) — optional refinement for production, not needed for prototype.
- Caption mixing (InternVL / ShareGPT4V / Florence) for diversity — production run.
- If LR scaling is desired at other batch sizes, use the √2 rule for AdamW.
