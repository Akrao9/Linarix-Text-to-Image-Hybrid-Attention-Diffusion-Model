# BoomerMain4 100k Training Report

Date: 2026-05-25

## Summary

BoomerMain4 is the first run that crossed from "denoising curiosity" into a real text-to-image base model. The model is not finished, but by 88k-100k it could generate recognizable dogs, strong landscapes, and partially coherent complex vintage street/person/owl scenes from pure noise.

The most important conclusion: the earlier `t=1.0` latent MSE preview was too pessimistic as a primary generation metric. It measured reconstruction toward a fixed cached latent, while the actual samples showed valid alternate generations. For future evaluation, use sample grids, prompt hit-rate, CLIP-style alignment, and later KID/FID, not MSE alone.

## Final Run Identity

Drive root:

- `/content/drive/MyDrive/BoomerMain4`

Checkpoint directory:

- `/content/drive/MyDrive/BoomerMain4/checkpoints/boomer_fla_journeydb_rope`

Important checkpoint:

- `/content/drive/MyDrive/BoomerMain4/checkpoints/boomer_fla_journeydb_rope/latent_dit_step_100000.pt`
- `/content/drive/MyDrive/BoomerMain4/checkpoints/boomer_fla_journeydb_rope/latent_dit_step_100000.safetensors`

Best current sampling default:

- Weights: `ema`
- Sampler: `stork2`
- Steps: `32`
- CFG: `1.5` for people/complex prompts, `2.0` for simple objects and landscapes
- CFG rescale: `0.5` for CFG above `1.0`

## Model And Training Config

Architecture:

- Model type: `boomer_fla`
- Parameters: about `669.8M`
- Hidden dim: `1152`
- Depth: `28`
- Heads: `16`
- MLP ratio: `2.5`
- Mixer: `fla_gated_deltanet`
- FLA mode: `chunk`
- Short conv: enabled
- Conv size: `4`
- Image attention every: `4`
- Image attention backend: `sdpa`
- Image RoPE: enabled
- Cross attention backend: `sdpa`
- Cross attention QK norm: enabled

Training:

- Batch size: `80`
- Steps: `100000`
- LR: `1e-4`
- Scheduler: `linear-warmup-cosine`
- Warmup: `1000`
- Min LR ratio: `0.1`
- Dtype: `bf16`
- Flow shift: `1.5`
- T sampler: `plateau-logit-normal`
- Logit mean/std: `0.0 / 1.0`
- Min-SNR gamma: `5.0`
- Timestep weighting: disabled
- Grad clip: `0.3`
- Cond dropout: `0.1`
- EMA decay: `0.9992`
- EMA update every: `8`
- Fused AdamW: enabled
- Compile: enabled
- Compile mode: `max-autotune-no-cudagraphs`
- Preview t: `1.0`

Runtime:

- First compile step took about `267s`
- Normal step time after compile: about `0.83s` to `0.85s`
- Total run time: about `20-21h`

## Important Code Changes

Training and data:

- Added GPU-side dtype casting after host-to-device transfer so BF16 conversion does not happen on CPU.
- Added `--min-snr-gamma`.
- Added `--compile` and `--compile-mode`.
- Added true `plateau-logit-normal` timestep sampling.
- Kept `bounded-logit-normal` as a separate sampler.
- Updated rolling WDS loader to shuffled full-shard epoch coverage instead of repeated random shard replacement.

Flow/sampling:

- Added flow Min-SNR loss weighting:
  - `snr = ((1 - sigma) / sigma)^2`
  - `weight = min(snr, gamma) / snr`
- Implemented true plateau logit-normal:
  - find logit-normal mode `t_star`
  - sample truncated logit-normal on `[0, t_star]`
  - sample uniform plateau on `[t_star, 1]`
  - choose region by normalized probability mass
- Optimized inverse normal CDF using:
  - `sqrt(2) * erfinv(2p - 1)`

Model:

- Removed CUDA host sync in condition dropout:
  - no `bool(drop.any())`
  - no clone-heavy dropout path
  - uses `torch.where`
- Cached `t_embedder(timesteps)` once per forward.
- Precomputed RoPE2D broadcast tables.
- Hoisted SDPA cross-attention additive mask construction out of each block.
- Added adaLN-Gaussian internal modulation:
  - `t_block[1].weight`: normal std `0.02`
  - block `scale_shift_table`: normal std `0.02`
  - image-attn `image_attn_scale_shift_table`: normal std `0.02`
  - final output path remains zero initialized.

Tests:

- `python -m pytest tests/test_flow_and_model.py tests/test_denoise_and_preview.py -q`
- Latest known result after these changes: `26 passed`

## Loss Behavior

The loss is not directly comparable to earlier runs because Min-SNR and the timestep distribution changed.

Early:

- Step 1: `loss=1.908953`, `loss_ma=1.908953`
- Step 1000: `loss=1.203779`, `loss_ma=1.235047`

Mid/late:

- Step 4000: `loss=1.125454`, `loss_ma=1.155298`
- Step 20000: `loss=1.054560`, `loss_ma=1.074737`
- Step 56000: `loss=1.142130`, `loss_ma=1.067926`
- Step 72000: LR about `0.000026404`
- Step 72300: `loss=0.985587`, `loss_ma=1.044022`
- Step 93700: `loss=0.979234`, `loss_ma=1.050133`
- Step 93750: `loss=1.153008`, `loss_ma=1.067730`

Interpretation:

- Stable training.
- No NaNs or divergence observed.
- Loss kept hovering around `1.04-1.10` late in training.
- The model was still visually improving even when scalar loss and MSE preview were not strongly optimistic.

## Preview MSE Metrics

Preview used EMA weights and `t=1.0`, so the noised MSE was fixed at `2.093993`.

| Step | Denoised MSE | Ratio |
| ---: | ---: | ---: |
| 4000 | 1.660367 | 0.793 |
| 8000 | 1.534898 | 0.733 |
| 12000 | 1.556072 | 0.743 |
| 16000 | 1.572173 | 0.751 |
| 20000 | 1.618640 | 0.773 |
| 24000 | 1.672853 | 0.799 |
| 28000 | 1.737461 | 0.830 |
| 32000 | 1.797294 | 0.858 |
| 36000 | 1.840229 | 0.879 |
| 40000 | 1.870312 | 0.893 |
| 44000 | 1.894289 | 0.905 |
| 48000 | 1.912903 | 0.914 |
| 52000 | 1.927630 | 0.921 |
| 56000 | 1.936689 | 0.925 |
| 60000 | 1.943053 | 0.928 |

Interpretation:

- The fixed-latent MSE metric improved early, then worsened.
- This did not match visual sample quality.
- The metric should remain a debug diagnostic, not the headline generation metric.

## Visual Progress

Step 56k:

- Preview showed coherent monochrome figure/background structure.
- Distorted face and weak prompt binding, but no longer pure collapse.

Step 72k:

- Preview became more semantically organized.
- Human figure was more grounded.
- Background animal/owl form was clearer.
- Street/vintage framing was more stable.

Step 88k:

- Dog EMA seed 3 was a major milestone:
  - unmistakably dog-like
  - coherent portrait lighting
  - visible fur, eyes, ears, muzzle
- Complex woman/owl prompt still showed weak object binding but good style and scene mood.

Step 100k eval grids:

- Landscape prompts were the strongest.
- Dog prompts became broadly recognizable.
- Complex prompts remained partial and surreal.

## 100k Evaluation Grid Results

Evaluation setup:

- Checkpoint: step `100000`
- Weights: `ema`
- Sampler: `stork2`
- Steps: `32`
- Seeds: `0-7`
- Prompts:
  - simple dog
  - medium landscape
  - complex woman/cloud/owl/street
- CFG tested:
  - `1.0`
  - `1.5`
  - `2.0`

### CFG 1.0

Dog:

- Several seeds recognizable but softer and less directed.
- Some seeds drifted into generic animal/cat/tiger-like forms.

Landscape:

- Strong, coherent landscape generations.
- High seed hit-rate.

Complex woman/owl:

- Strong monochrome vintage street mood.
- Person/scene often present.
- Owl/cloud relationship not reliably solved.

### CFG 1.5

Dog:

- Better identity than CFG `1.0`.
- More seeds produced dog/animal portraits.

Landscape:

- Very strong.
- Better prompt adherence and clarity than CFG `1.0`.

Complex woman/owl:

- Slightly sharper/directive, but object binding still not solved.

### CFG 2.0

Dog:

- Strong and usable.
- Similar or slightly better than CFG `1.5` for simple object identity.

Landscape:

- Excellent.
- Not meaningfully better than CFG `1.5`.

Complex woman/owl:

- Sharpened the same surreal layouts.
- Did not solve multi-object compositional binding.

STORK steps:

- `50` steps looked almost identical to `32` steps.
- Current bottleneck is model structure/object binding, not sampler step count.

## Recommended Sampling Settings

Default:

```text
--weights ema
--sampler stork2
--steps 32
--stork-substeps 5
--cfg-scale 1.5
--cfg-rescale 0.5
```

Use CFG `2.0` for:

- simple objects
- animals
- landscapes
- prompts where identity is more important than avoiding distortion

Use CFG `1.5` for:

- people
- complex prompts
- prompts where over-guidance may amplify weird anatomy

Avoid defaulting to `50` steps for routine evaluation. It costs more and did not visibly improve the tested grids.

## Current Capability

The model can now:

- generate recognizable animals from pure noise
- generate strong landscapes
- separate simple/medium prompt categories
- produce coherent style, lighting, and broad composition
- preserve a distinctive vintage/illustrative aesthetic

The model still struggles with:

- anatomy
- faces
- clean object boundaries
- object counting
- complex compositional binding
- robust prompt following across all seeds
- distinguishing some animal categories cleanly

Best description:

```text
BoomerMain4 100k is an early usable base model, not a finished image model.
```

## Seed Behavior

Seed dependence remains significant.

Important observation:

- At 60k, only some seeds were promising.
- By 88k/100k, more seeds became recognizable for simple prompts.
- The best seed demonstrates model capability.
- The average seed demonstrates model maturity.

Future eval should track hit-rate:

```text
dog: how many of 8 seeds are clearly dog-like?
landscape: how many of 8 are valid landscapes?
portrait: how many of 8 are coherent people?
complex: how many of 8 satisfy the main object relationships?
```

## Evaluation Lessons

Do not use fixed-latent MSE as the primary preview metric for generation.

MSE answers:

```text
Can the model reconstruct this exact cached latent from this exact noise?
```

Generation needs:

```text
Can the model produce a plausible image from noise that matches the text?
```

Better metrics:

- visual grids
- seed hit-rate
- prompt separation
- CLIP text-image similarity
- diversity metrics
- later KID/FID over many samples

Recommendation:

- Keep MSE as a debug line only.
- Add CLIP score for preview samples.
- Run grid eval every `10k-20k` in continuation runs.

## Fine-T2I Continuation Notes

Fine-T2I is a good next dataset once cached into the same latent/text format.

For a 1.16M subset:

- One epoch at batch size `80` is about `14500` steps.
- `40k` steps is about `2.8` epochs.
- `60k` steps is about `4.1` epochs.
- `75k` steps is about `5.2` epochs.

Recommended continuation:

```text
resume from BoomerMain4 100k
dataset: Fine-T2I 1.16M cached subset
lr: 3e-5 or 5e-5
steps: 40k-60k first
scheduler horizon: 75k or 100k
min_lr_ratio: 0.2
flow_shift: 1.5
t_sampler: plateau-logit-normal
min_snr_gamma: 5.0
grad_clip: 0.3
```

Use `3e-5` if prioritizing safety and avoiding style drift.

Use `5e-5` if prioritizing faster prompt-alignment learning.

## Fine-T2I Token Length

Fine-T2I prompts are long, especially enhanced prompts. A text cache length of `340` tokens is reasonable if supported end to end.

Requirements:

- text embeddings shape can be `[340, 1536]` or `[B, 340, 1536]`
- attention masks are also length `340`
- model cross-attention does not assume `300`
- sample/preview tools load condition caches with variable token length
- avoid mixing old `300` token JourneyDB caches with `340` token Fine-T2I caches unless padding/variable length is confirmed

Cost:

- `340` vs `300` is about `13%` more text length in cross-attention.
- Total runtime increase should be less than `13%` because image blocks, FLA blocks, and MLPs dominate much of the compute.

Recommendation:

- Use `340` tokens for Fine-T2I enhanced prompts.
- Keep each training run internally consistent in token length.

## Next Actions

Immediate:

1. Keep BoomerMain4 100k as the baseline.
2. Preserve the 100k eval grids.
3. Use EMA, STORK2, 32 steps, CFG `1.5/2.0` for routine sampling.

Next training:

1. Prefer Fine-T2I 1.16M continuation if the cache is ready.
2. Otherwise do same-data continuation from 100k:
   - LR `2e-5` or `3e-5`
   - `50k` more steps
   - scheduler horizon `75k`
   - min LR ratio `0.2`

Do not change architecture yet. The current architecture works. The remaining bottleneck is data/training maturity, especially object binding and seed robustness.
