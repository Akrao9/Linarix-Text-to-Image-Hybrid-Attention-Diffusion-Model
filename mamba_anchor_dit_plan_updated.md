# MambaAnchor-DiT: Build Plan

**Goal:** Build a text-to-image latent flow model from scratch using Flame/FLA, Sana-style DC-AE f32c32 latents, Gemma text conditioning, Mamba/FLA sequence blocks, anchored attention, and mixed BF16/FP8 training on an RTX PRO 6000 Blackwell 96GB GPU.

**Core idea:** Start with text-to-image generation only, but design the system so it can later support image-conditioned generation, video/future-frame generation, and world-model conditioning.

---

## 1. Final Research Direction

### Working title

**MambaAnchor-DiT: Efficient Latent Image Generation with State-Space Mixing and Anchored Global Attention**

### One-sentence pitch

MambaAnchor-DiT combines efficient Mamba/FLA-style sequence mixing with lightweight anchored attention to improve the speed-quality tradeoff of latent text-to-image generation.

### Main hypothesis

Pure attention is expensive at high resolution. Pure Mamba/linear sequence mixing may be efficient but can struggle with global image composition. A hybrid architecture can use:

- **Mamba/FLA blocks** for efficient long-range token mixing
- **Anchored attention** for global semantic coherence
- **DC-AE f32c32 latents** for Sana-compatible high-quality reconstruction and a proven high-resolution pipeline
- **Gemma text conditioning** to match Sana's released training/inference path
- **FP8/BF16 mixed precision** for faster training on Blackwell

---

## 2. What to Build First

Do **not** start with a full multimodal model.

Start with:

```text
text prompt → image
```

Later extend to:

```text
text + image → edited image
past frames + text/actions → future frame/video
```

This keeps the first version clean and measurable.

---

## 3. Framework and Reference Repo Choice

### Main build framework

Use Flame as the main local library/framework dependency:

```bash
git clone https://github.com/fla-org/flame.git
git clone https://github.com/fla-org/flash-linear-attention.git
```

Current local workspace layout:

```text
/Users/ak/Documents/genaiexperiments/ImageModel/
  Boomer/   # project plan and Boomer-specific notes
  flame/    # local Flame repo; library/framework dependency
  Sana/     # local Sana repo; image-generation reference
```

Flame is the right base library because it is a minimal efficient training framework built on TorchTitan and integrates with FLA and Transformers. It is language-model oriented by default, so the main engineering task is to build Boomer-specific latent image flow-matching modules around Flame rather than rewriting the whole training stack.

Use Flame from the sibling checkout by installing it editable or putting it on `PYTHONPATH`:

```bash
pip install -e ../flame
```

If Flash Linear Attention is not checked out as a separate sibling repo, use Flame's dependency path first. Clone `flash-linear-attention` separately only if local kernel edits or deeper FLA block work become necessary.

### Main image-generation reference

Use Sana as the primary code reference beside Flame:

```bash
git clone https://github.com/NVlabs/Sana.git
```

Sana is the most relevant reference because it already targets efficient high-resolution text-to-image generation and uses deep-compression autoencoders. Study Sana for:

- DC-AE loading and latent scaling
- high-resolution text-to-image configs
- text encoder handling
- sampling and inference pipeline structure
- validation image generation
- dataset/preprocessing conventions

### Minimal reference stack

The main stack should be:

```text
1. Flame / FLA: training framework and sequence-block integration
2. Sana: image-generation, DC-AE, and high-resolution pipeline reference
```

This is enough to start properly. Do not overcomplicate the codebase by trying to follow many repositories at once.

### Optional reference only

Keep Diffusers bookmarked only for sanity-checking schedulers, validation sampling, checkpoint/resume behavior, and common text-to-image training utilities. Do not make it a main dependency unless needed.

```text
Optional reference: Hugging Face Diffusers
Use only for: scheduler logic, validation sampling, checkpointing patterns
```

### Why not build inside LeWM?

LeWM is a world-model/control codebase. It is useful later for recurrent latent world modeling, partial observability, and future-frame/video conditioning.

For the image generator, Flame/FLA is the better base.

Use LeWM later only as a reference for:

- recurrent latent dynamics
- partial observability
- planning/world-model experiments
- action-conditioned video generation

---

## 4. Autoencoder Choice

### Recommended v0

Use:

```text
DC-AE f32c32 Sana 1.1
mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers
```

Use the same f32c32 path as Sana first because it is the most practical way to copy/adapt Sana's code and configs.

For a 1024×1024 image:

```text
1024 / 32 = 32
latent grid = 32 × 32
tokens = 1024
latent channels = 32
```

This has more tokens than f64, but it preserves more spatial detail and follows Sana's proven high-resolution setup.

For a 512×512 image:

```text
512 / 32 = 16
latent grid = 16 × 16
tokens = 256
latent channels = 32
```

This makes 512px the best first serious debugging resolution.

### Optional later experiment

After the f32c32 pipeline works, test:

```text
DC-AE f64c128
```

For 1024×1024:

```text
1024 / 64 = 16
latent grid = 16 × 16
tokens = 256
latent channels = 128
```

This may be useful for lower token count Mamba/anchor experiments, but it should not block the first build.

### Recommendation

Use Sana's f32c32 autoencoder for v0/v1. Treat f64 as a later research variant.

---

## 5. Text Encoder Choice

### v0 debugging

Use:

```text
google/gemma-4-E2B-it
```

Reason:

- newer Gemma text encoder with Hugging Face Transformers support
- Apache-2.0 licensed model card
- avoids adding a separate T5-specific conditioning path first
- gives stronger prompt embeddings than a tiny debug encoder
- can still be cached before generator training

### v1 serious model

Use:

```text
Gemma 4 E2B with cached embeddings
```

Keep the same text encoder while scaling the image generator so architecture changes are easier to measure.

### Optional later comparison

After the Gemma path works, compare against:

```text
T5-large, T5-XL, or T5-XXL cached
```

Use this only as an ablation or prompt-following experiment.

### Optional later

Add:

```text
CLIP/OpenCLIP pooled embedding
```

Use CLIP as a global style/semantic signal, not as the main text encoder.

### Recommended sequence

```text
1. Gemma 4 E2B
2. cached Gemma 4 embeddings
3. optional T5 ablation
4. optional CLIP pooled embedding
```

---

## 6. Dataset Plan

### First streamable dataset

Use:

```text
JourneyDB/JourneyDB
```

Why:

- image-generation-style prompts
- millions of image-text pairs
- good for text-to-image training
- usable through Hugging Face streaming

### Add later

Use:

```text
opendiffusionai/cc12m-2mp-realistic
lance-format/laion-1m
```

### Recommended training mixture later

```text
70% JourneyDB
20% CC12M-style cleaned/realistic data
10% filtered LAION-style data
```

### Data pipeline

Do not repeatedly encode images/text during every training step.

Use a two-stage pipeline:

```text
Stage A:
stream image-caption data
resize/crop image
encode image with DC-AE
encode text with Gemma
save latent shards

Stage B:
train generator from cached latent shards
```

Each cached sample should contain:

```text
image_latent
text_embedding
attention_mask
caption
metadata
```

---

## 6.5 Colab and Google Drive Checkpointing Plan

Because training will run in Colab on an RTX PRO 6000 96GB GPU, treat the Colab VM as disposable.

Use Google Drive for persistent checkpoints:

```text
/content/drive/MyDrive/Boomer/
  checkpoints/
  latent_shards/
  sample_grids/
  logs/
  configs/
```

Recommended workflow:

```text
1. Mount Google Drive at notebook startup.
2. Keep active training writes on local Colab disk when possible.
3. Periodically sync checkpoints, sample grids, logs, and configs to Drive.
4. On startup, auto-detect the latest Drive checkpoint and resume.
5. Keep only the most recent N checkpoints plus important milestone checkpoints.
```

Each checkpoint should include:

```text
model weights
EMA weights
optimizer state
LR scheduler state
global step
epoch/data cursor if available
random seeds
training config
model config
latest validation sample grid
```

Drive can be slower than local disk, so avoid writing every training step directly to Drive. Prefer local checkpoint writes followed by scheduled Drive syncs.

Recommended initial policy:

```text
save local checkpoint every 1,000 steps
sync to Google Drive every 5,000 steps
save validation sample grid every 1,000 steps
keep last 3 regular checkpoints
keep milestone checkpoints permanently
```

---

## 7. Model Architecture

### Model name

```text
MambaAnchor-DiT
```

### Inputs

```text
x_t: noisy image latent
t: flow/diffusion timestep
c: text condition tokens
```

### Output

```text
v_pred: predicted velocity
```

### High-level architecture

```text
noisy latent x_t
   ↓
latent/token projection
   ↓
add timestep conditioning
   ↓
condition on Gemma text tokens
   ↓
N × MambaAnchor blocks
   ↓
final projection
   ↓
velocity prediction
```

---

## 8. MambaAnchor Block

Each block should look like:

```text
x = x + Mamba_or_FLA_Mixer(norm(x), t, c)
x = x + AnchorAttention(norm(x), anchors, t, c)
x = x + MLP(norm(x), t, c)
```

### Block components

#### 1. Normalization

Use:

```text
RMSNorm or LayerNorm
```

Use AdaLN-style timestep conditioning if possible.

#### 2. Sequence mixer

Use one of:

```text
Mamba-3-style block
FLA GLA/DeltaNet-style block
Mamba-2-compatible fallback
```

The FLA version may be easier to implement first inside Flame.

#### 3. Anchored attention

Use anchors to provide global structure.

Start with:

```text
learned anchors
```

Then try:

```text
pooled latent anchors
text-conditioned anchors
```

#### 4. MLP

Use:

```text
SwiGLU MLP
MLP ratio 4
```

---

## 9. Anchored Attention Design

### v0: learned anchors

```text
num_anchors = 32 or 64
anchors are learned parameters
image tokens attend to anchors
anchors summarize global information
```

### v1: pooled anchors

```text
image tokens → downsample/pool → anchor tokens
```

This makes anchors image-aware.

### v2: text-conditioned anchors

```text
Gemma text embedding → anchor initialization/modulation
```

This should improve prompt alignment.

### Recommended starting config

```text
num_anchors: 32
anchored_attention_every: 3 blocks
full_attention_every: 0
```

For larger model:

```text
num_anchors: 64
anchored_attention_every: 2 blocks
```

---

## 10. Precision Plan

### Start with BF16

First run:

```text
BF16 only
```

Do this until:

- loss decreases
- samples decode correctly
- no NaNs
- checkpointing works
- sampler works

### Then enable FP8

Use FP8 for:

- linear projections
- QKV projections
- MLP layers
- internal block projections
- possibly Mamba/FLA projection layers

Keep these in BF16 or FP32:

- loss computation
- normalization layers
- timestep embeddings
- text encoder
- DC-AE encoder/decoder
- final output head at first

### Rule

FP8 is an optimization, not the first thing to debug.

---

## 11. Training Objective

Use flow matching with velocity prediction.

Pseudo-process:

```text
x0 = clean image latent
eps = random Gaussian noise
t = random timestep
x_t = interpolate between x0 and eps
target_velocity = eps - x0 or flow-specific target
model predicts velocity
loss = MSE(v_pred, target_velocity)
```

### Loss

```text
L = MSE(v_pred, v_target)
```

Optional later:

- SNR weighting
- resolution-aware weighting
- perceptual losses only if doing decoder fine-tuning

---

## 12. Sampler

Start with:

```text
Euler sampler
```

Then add:

```text
Heun sampler
DPM-style sampler if useful
```

Support:

- CFG scale
- negative prompts
- EMA weights
- seed control
- batch generation

---

## 13. Model Configs

### v0 debug model

```yaml
model:
  name: MambaAnchor-DiT-S
  hidden_dim: 768
  depth: 16
  mlp_ratio: 4
  num_anchors: 32
  anchored_attention_every: 4
  precision: bf16

autoencoder:
  type: AutoencoderDC
  pretrained: mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers
  compression: 32
  latent_channels: 32
  frozen: true

text_encoder:
  name: google/gemma-4-E2B-it
  frozen: true
  max_length: 300

training:
  objective: flow_matching
  prediction_type: velocity
  ema: true
```

### v1 serious model

```yaml
model:
  name: MambaAnchor-DiT-B
  hidden_dim: 1024
  depth: 24
  mlp_ratio: 4
  num_anchors: 32
  anchored_attention_every: 3
  precision: bf16_fp8_mixed

autoencoder:
  type: AutoencoderDC
  pretrained: mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers
  compression: 32
  latent_channels: 32
  frozen: true

text_encoder:
  name: google/gemma-4-E2B-it
  frozen: true
  max_length: 300

training:
  objective: flow_matching
  prediction_type: velocity
  ema: true
```

### v2 larger model

```yaml
model:
  name: MambaAnchor-DiT-L
  hidden_dim: 1536
  depth: 28
  mlp_ratio: 4
  num_anchors: 64
  anchored_attention_every: 2
  precision: bf16_fp8_mixed

autoencoder:
  type: AutoencoderDC
  pretrained: mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers
  compression: 32
  latent_channels: 32
  frozen: true

text_encoder:
  name: google/gemma-4-E2B-it_cached
  frozen: true
  max_length: 300

training:
  objective: flow_matching
  prediction_type: velocity
  ema: true
```

---

## 14. Baselines

You need baselines to make this publishable.

### Minimum baselines

```text
B1: your previous HybridLinearDIT
B2: FLA/Mamba-only DiT
B3: anchored-attention-only DiT
B4: Mamba/FLA + anchored attention
```

### Optional baselines

```text
B5: small full-attention DiT
B6: local-window attention DiT
B7: Mamba-2 fallback version
B8: Mamba-3/MIMO version
```

### Main comparison

```text
quality vs speed vs memory
```

Do not claim only better quality. Your likely strength is better efficiency at similar quality.

---

## 15. Evaluation

Track:

```text
FID
CLIP score
aesthetic score
GenEval or prompt-composition benchmark if available
training throughput
samples/sec
VRAM usage
loss curve
sample grid over time
```

### Main table

```text
Model                         FID ↓   CLIP ↑   VRAM ↓   samples/sec ↑
HybridLinearDIT
FLA/Mamba-only
Anchor-only
MambaAnchor-DiT
```

### Qualitative tests

Use prompt groups:

```text
single object
multiple objects
spatial relations
style prompts
text rendering
human/animal
landscape
high-detail scene
```

---

## 16. Build Order

### Step 1: Create a Flame-backed Boomer project

Keep Boomer-specific code in the current `Boomer/` project folder and use Flame as a sibling library dependency:

```text
Boomer/
```

Use the sibling repos like this:

```text
Boomer/   # Boomer model, configs, data pipeline, training scripts
../flame/ # installed as editable local library
../Sana/  # reference for DC-AE, sampling, and high-res T2I patterns
```

Do not put Boomer-specific experiments directly into the Flame checkout unless the change is a reusable Flame improvement.

### Step 2: Add latent image dataset support

Implement:

```text
Hugging Face streaming loader
image resize/crop
caption extraction
dataset filtering
```

### Step 3: Add latent caching

Implement:

```text
DC-AE encode
Gemma encode
save shards
load shards efficiently
```

### Step 4: Add simple DiT baseline

Before Mamba/anchors, train a simple model to confirm:

```text
data works
loss works
sampler works
decoder works
```

### Step 5: Add FLA/Mamba mixer

Replace attention/mixing with FLA/Mamba-style blocks.

### Step 6: Add learned anchors

Implement anchored attention.

### Step 7: Combine into MambaAnchor block

Train small model.

### Step 8: Add FP8

Only after BF16 works.

### Step 9: Scale to 1024

Use Sana f32c32 latents and a larger model.

### Step 10: Improve text conditioning

Keep Gemma as the default path. Add cached Gemma embeddings, Sana-style CHI prompt handling, and only then consider T5 as an ablation.

---

## 17. Multimodal Extension Later

Do not start with full multimodal.

After text-to-image works, add:

### Stage 1: image-conditioned generation

```text
input: text + source image latent
output: target image latent
```

Use for:

```text
image editing
variation
inpainting
style transfer
```

### Stage 2: video/future-frame generation

```text
input: past frames + text/action
output: future frame/video latent
```

This connects to the LeWM-M3 world-model idea.

### Stage 3: world-model conditioning

```text
past frames/actions → recurrent world latent → generator condition → future frame
```

This becomes:

```text
action-conditioned video generation under partial observability
```

---

## 18. Risks and Fixes

### Risk: FP8 instability

Fix:

```text
start BF16
enable FP8 only in selected linears
keep norms/loss/output head BF16
```

### Risk: f32c32 has more tokens than f64

Fix:

```text
start at 512px where f32c32 gives 256 tokens
use FLA/Mamba blocks before scaling model depth too aggressively
keep f64c128 as a later low-token ablation
```

### Risk: pure Mamba/FLA weak global composition

Fix:

```text
increase anchor count
add text-conditioned anchors
add occasional low-resolution full attention
```

### Risk: data pipeline bottleneck

Fix:

```text
cache DC-AE latents and Gemma embeddings
use WebDataset-style shards
prefetch aggressively
```

### Risk: Gemma conditioning path adds complexity

Fix:

```text
copy Sana's tokenizer/text-encoder handling first
cache Gemma embeddings before generator training
only add T5 or CLIP after the Gemma path works
```

---

## 19. Milestones

### Milestone 1

```text
BF16 simple latent DiT trains and samples at 256/512
```

### Milestone 2

```text
Mamba/FLA-only model trains
```

### Milestone 3

```text
anchored attention works
```

### Milestone 4

```text
MambaAnchor-DiT beats or matches baseline at better speed/memory
```

### Milestone 5

```text
FP8 mixed precision stable
```

### Milestone 6

```text
1024 generation with Sana f32c32 works
```

### Milestone 7

```text
text-to-image model supports strong prompts
```

### Milestone 8

```text
image-conditioned or video/future-frame extension begins
```

---

## 20. Final Recommended Stack

```text
Framework: Flame/FLA
Main reference repo: NVLabs/Sana
Autoencoder: Sana AutoencoderDC f32c32
Autoencoder weights: mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers
Text encoder v0: Gemma 4 E2B
Text encoder path: google/gemma-4-E2B-it
Text encoder v1: cached Gemma 4 embeddings
Text encoder v2: optional T5/CLIP ablations
Precision: BF16 first, then FP8 selected layers
Backbone: Mamba/FLA mixer + anchored attention
Objective: flow matching velocity prediction
Dataset v0: JourneyDB
Dataset v1: JourneyDB + CC12M cleaned/realistic
Sampler: Euler first, Heun later
```

---

## 21. Source Notes

Useful references checked while making this plan:

- Flame: minimal efficient framework built on TorchTitan, with FLA and Transformers integration  
  https://github.com/fla-org/flame

- Flash Linear Attention: efficient implementations and Flame training examples  
  https://github.com/fla-org/flash-linear-attention

- Sana: efficient high-resolution text-to-image codebase and the closest practical reference for this project  
  https://github.com/NVlabs/Sana

- DC-AE: high-compression autoencoders for efficient high-resolution diffusion models  
  https://arxiv.org/html/2410.10733v1

- Optional text encoder ablation: distilled T5-base can retain much of T5-XXL image quality while being much smaller  
  https://arxiv.org/abs/2503.19897

- Diffusers text-to-image training: optional reference for scheduler and validation pipeline details  
  https://huggingface.co/docs/diffusers/en/training/text2image

- Scale-DiT-style anchored/global guidance: useful conceptual reference for scalable high-resolution generation  
  https://arxiv.org/abs/2510.16325

---

## 22. One-Line Plan

Build a **text-to-image latent flow model** in Flame using **Sana f32c32 DC-AE latents**, **Gemma conditioning**, **Mamba/FLA sequence mixing**, and **anchored attention**, then extend it later into image-conditioned and video/world-model generation.
