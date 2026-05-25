# Boomer Run Progress 1

Date: 2026-05-23

## Context

This log tracks the current Colab run launched from the notebook in the Boomer workspace.

Notebook checked locally:

- Workspace file found: `Boomermainun.ipynb`
- User referred to it as `Boomermainrun.ipynb`; the exact local filename has no `r` before `un`.
- Notebook has 16 cells.
- No saved Python exception outputs were present in the notebook.
- Relevant local tests passed before evaluating the run:
  - `python -m pytest tests/test_flow_and_model.py tests/test_denoise_and_preview.py -q`
  - Result: 23 passed

## Notebook Run Setup

The notebook flow is:

1. Change to `/content`.
2. Unzip `/content/Boomer.zip`.
3. Change to `/content/Boomer`.
4. Install Python dependencies:
   - `transformers`
   - `accelerate`
   - `diffusers`
   - `datasets`
   - `huggingface_hub`
   - `safetensors`
   - `webdataset`
   - `pytest`
   - `einops`
   - `ninja`
   - `packaging`
   - `wandb`
5. Install FLA-related deps:
   - `flash-linear-attention`
   - `einops`
   - `ninja`
   - `packaging`
6. Run `tools/probe_sequence_mixers.py`.
7. Prepare STORK under `/content/STORK` if not already present.
8. Log in to Hugging Face via Colab userdata `HF_TOKEN`.
9. Log in to W&B via Colab userdata `WANDB_API_KEY`.
10. Use PyTorch SDPA for both image attention and cross attention.
11. Run a CUDA BF16 SDPA smoke test.
12. Mount Google Drive.
13. Prepare stats, checkpoint, preview, preview record, and loss CSV directories.
14. Download JourneyDBCache local shards for stats and preview records.
15. Compute channel latent stats.
16. Build five fixed preview `.pt` records.
17. Inspect the BoomerFLA model.
18. Enable TF32 for fp32 paths.
19. Launch long training.

## Drive And Local Paths

Drive root:

- `/content/drive/MyDrive/BoomerMain`

Stats:

- `/content/drive/MyDrive/BoomerMain/stats/journeydb_latent_stats.json`

Checkpoints:

- `/content/drive/MyDrive/BoomerMain/checkpoints/boomer_fla_journeydb_rope`

Previews:

- `/content/drive/MyDrive/BoomerMain/previews/boomer_fla_journeydb_rope`

Preview records:

- `/content/drive/MyDrive/BoomerMain/preview/records`
- `preview_00.pt` through `preview_04.pt`

Loss CSV:

- `/content/drive/MyDrive/BoomerMain/logs/boomer_fla_journeydb_rope_loss.csv`

Local WDS stats shards:

- `/content/wds_stats`

Local rolling WDS shards:

- `/content/wds_roll`

## Dataset And Stats

Training dataset:

- Hugging Face dataset repo: `akrao9/JourneyDBCache`
- WDS glob: `hf://datasets/akrao9/JourneyDBCache/train/*.tar`

Stats shard download:

- `STATS_SHARD_COUNT = 22`
- Commented estimate: about 50k samples and about 48 GB
- Stats max records: `50_000`

Latent normalization in active run:

- Mode: channel
- `latent_mean`: 32 channels
- Mean of channel means: `0.070316`
- Min channel mean: `-0.238635`
- Max channel mean: `0.872752`
- `latent_std`: 32 channels
- Mean of channel stds: `0.887407`
- Min channel std: `0.718289`
- Max channel std: `2.319551`

## Model Inspection And Training Config

Active model:

- `model_type = boomer_fla`
- Parameters: `669,808,000`
- Device: `cuda`
- Dtype: `torch.bfloat16`

Architecture settings:

- `hidden_dim = 1152`
- `depth = 28`
- `num_heads = 16`
- `mlp_ratio = 2.5`
- `mixer_type = fla_gated_deltanet`
- `fla_mode = chunk`
- `use_short_conv = true`
- `conv_size = 4`
- `image_attn_every = 4`
- `image_attn_backend = sdpa`
- `image_rope = true`
- `image_rope_theta = 10000`
- `cross_attn_backend = sdpa`
- `cross_attn_qk_norm = true`
- `y_norm = true`
- `y_norm_scale_factor = 0.01`

Training settings:

- Batch size: `80`
- DataLoader workers: `0`
- Pin memory: `False`
- Steps target: `100000`
- LR: `1e-4`
- LR scheduler: `linear-warmup-cosine`
- Warmup steps: `1000`
- Min LR ratio: `0.1`
- Flow shift: `3.0`
- T sampler: `logit-normal`
- Timestep weighting: enabled
- EMA decay: `0.96`
- EMA update every: `8`
- Grad clip: `1.0`
- Conditioning dropout: `0.1`
- Optimizer: fused AdamW enabled
- Log every: `50`
- Save every: `4000`
- Preview every: `4000`
- Preview sampler: `stork2`
- Preview steps: `16`
- Preview STORK derivative order: `1`
- Preview STORK substeps: `5`
- Seed: `0`
- W&B enabled
- W&B project: `boomer`
- W&B run name: `boomer_fla_journeydb_rope`

Streaming WDS settings:

- Records label: `streaming`
- WDS shuffle buffer: `1000`
- WDS shard shuffle: `0`
- WDS rolling shards: `25`
- WDS local dir: `/content/wds_roll`

## W&B

W&B initialized successfully.

- Account shown in log: `ak-rao-bandi2`
- Entity/project URL shown:
  - `https://wandb.ai/ak-rao-bandi2-akrao/boomer`
- Run URL shown:
  - `https://wandb.ai/ak-rao-bandi2-akrao/boomer/runs/ine70qjr`
- Local W&B run dir:
  - `/content/Boomer/wandb/run-20260523_202347-ine70qjr`

## Training Progress Summary

Initial training:

- Step 1:
  - Loss: `4.040142`
  - Loss MA: `4.040142`
  - Step time: `61.97s`
  - Startup overhead caused very large initial ETA.

Early improvement:

- Step 50:
  - Loss: `3.584660`
  - Loss MA: `3.753084`
  - Step time: `0.86s`
- Step 100:
  - Loss: `3.720034`
  - Loss MA: `3.810030`
  - Step time: `0.90s`
- Step 500:
  - Loss: `2.348719`
  - Loss MA: `2.595834`
  - Step time: `0.91s`
- Step 1000:
  - Loss: `2.436875`
  - Loss MA: `2.475830`
  - Step time: `0.88s`

By 4000:

- Step 4000:
  - Loss: `2.144530`
  - Loss MA: `2.313665`
  - Step time: `0.88s`
  - Elapsed: `1.0h`
  - ETA: `24.8h`

By 8000:

- Step 8000:
  - Loss: `2.331772`
  - Loss MA: `2.213756`
  - Step time: `0.88s`
  - Elapsed: `2.1h`
  - ETA: `24.0h`

Latest pasted log:

- Step 11800:
  - Loss: `2.414409`
  - Loss MA: `2.257407`
  - Step time: `0.88s`
  - Elapsed: `3.1h`
  - ETA field was cut off in the pasted log.

Overall loss behavior:

- Loss dropped quickly from about `4.04` to the mid-2s in the first few hundred steps.
- From roughly step 4000 onward, loss mostly oscillated around `2.15` to `2.35`.
- Occasional lower single-step losses appeared, for example:
  - Step 6150: `1.931394`
  - Step 6650: `1.936680`
- No NaNs or instability were reported.

Throughput:

- After startup, most steps were about `0.86s` to `0.96s`.
- Occasional slower steps appeared:
  - Step 4450: `3.73s`
  - Step 6350: `1.16s`
  - Step 6550: `1.14s`
  - Step 6750: `1.13s`
  - Step 6950: `1.11s`
  - Step 7100: `1.19s`
  - Step 7150: `1.09s`
  - Step 7350: `1.08s`
- These were interpreted as likely shard rolling, download, Drive I/O, or background sync blips because training immediately returned to normal step times.

## Checkpoints

Step 4000 checkpoint:

- Resume checkpoint:
  - `/content/drive/MyDrive/BoomerMain/checkpoints/boomer_fla_journeydb_rope/latent_dit_step_004000.pt`
- Inference checkpoint:
  - `/content/drive/MyDrive/BoomerMain/checkpoints/boomer_fla_journeydb_rope/latent_dit_step_004000.safetensors`

Step 8000 checkpoint:

- Resume checkpoint:
  - `/content/drive/MyDrive/BoomerMain/checkpoints/boomer_fla_journeydb_rope/latent_dit_step_008000.pt`
- Inference checkpoint:
  - `/content/drive/MyDrive/BoomerMain/checkpoints/boomer_fla_journeydb_rope/latent_dit_step_008000.safetensors`

Resume note:

- Use the `.pt` checkpoint for training resume.
- The `.safetensors` file is for inference/model weights, not optimizer resume.

## Preview Metrics

Preview setup:

- Preview every `4000` steps.
- Preview uses EMA weights.
- Preview samples: `5`
- Preview t: `0.30`
- Sampler: `stork2`
- Steps: `16`

Step 4000 preview:

- Weights: `ema`
- Samples: `5`
- `latent_mse_noised = 0.188459`
- `latent_mse_denoised = 0.128766`

Step 8000 preview:

- Weights: `ema`
- Samples: `5`
- `latent_mse_noised = 0.188459`
- `latent_mse_denoised = 0.126395`

Step 12000 preview:

- Weights: `ema`
- Samples: `5`
- `latent_mse_noised = 0.188459`
- `latent_mse_denoised = 0.125243`
- Next logged step:
  - Step 12050 loss: `2.249559`
  - Step 12050 loss MA: `2.280576`
  - Step time: `0.95s`
  - Elapsed: `3.1h`
  - ETA: `23.0h`

Step 16000 preview:

- Weights: `ema`
- Samples: `5`
- `latent_mse_noised = 0.188459`
- `latent_mse_denoised = 0.125025`
- Next logged step:
  - Step 16050 loss: `2.278848`
  - Step 16050 loss MA: `2.162051`
  - Step time: `0.95s`
  - Elapsed: `4.2h`
  - ETA: `22.0h`

Step 20000 high-t preview after resume:

- This run used the updated shuffled-epoch rolling WDS loader and `--preview-t 0.8`.
- Weights: `ema`
- Samples: `5`
- `latent_mse_noised = 1.340156`
- `latent_mse_denoised = 1.242928`
- MSE ratio:
  - `1.242928 / 1.340156 = 0.9274`
- Next logged steps:
  - Step 20050 loss: `2.199187`, loss MA: `2.224410`, step time: `0.85s`, elapsed: `1.1h`, ETA: `21.4h`
  - Step 20100 loss: `2.178588`, loss MA: `2.197908`, step time: `0.94s`, elapsed: `1.1h`, ETA: `21.4h`
  - Step 20150 loss: `2.334675`, loss MA: `2.173838`, step time: `0.94s`, elapsed: `1.1h`, ETA: `21.4h`
  - Step 20200 loss: `2.330291`, loss MA: `2.220926`, step time: `0.87s`, elapsed: `1.1h`, ETA: `21.3h`
  - Step 20250 loss: `2.367746`, loss MA: `2.243919`, step time: `0.86s`, elapsed: `1.1h`, ETA: `21.3h`
  - Step 20300 loss: `2.189728`, loss MA: `2.213652`, step time: `0.94s`, elapsed: `1.1h`, ETA: `21.3h`
- Visual result:
  - The t=0.8 preview is now a recognizable coarse scene/person/owl-like composition.
  - It remains soft and distorted, but it is no longer just a collapsed abstract high-noise smear.
- Interpretation:
  - High-t denoising is now coming online.
  - Compared with the earlier manual t=0.8 diagnostic at step 12000 (`ratio ~= 0.9928`), step 20000 improved to `ratio ~= 0.9274`.
  - This supports continuing training and tracking high-t ratios rather than judging only global loss.

Metric interpretation:

- Denoising improved from step 4000 to 16000, but only slightly:
  - `0.128766 -> 0.126395 -> 0.125243 -> 0.125025`
- The preview metric indicates incremental cleanup rather than a major semantic jump.
- After switching preview to t=0.8, the absolute MSE numbers are larger by design because the input is much noisier.
- The important t=0.8 signal is the ratio, which improved from about `0.993` at step 12000 to about `0.927` at step 20000.

## Preview Image Observations

Prompt for the shown preview:

> A woman holding a cloud in her hand with open eyes on a street in 1870, with a big owl in the background. The photograph is captured using Ilford HP5 and Kodak Tri-X films, on a 35mm medium format, showcasing a vintage black and white aesthetic with grainy texture, reminiscent of the photographic style of Atget in Paris.

Step 4000 t=0.30 denoised image:

- Visual result:
  - Strong vintage black-and-white photographic look.
  - Street background is present.
  - Woman subject is clear.
  - Open eyes are clear.
  - Owl is convincing with visible feather detail.
  - Texture and grain fit the requested old photographic aesthetic.
- Misses:
  - The cloud in the hand is absent.
  - The owl appears in the foreground, not as a big owl in the background.
  - The woman/body/owl composition is surreal and fused.
  - It reads more like uncanny photo collage than controlled 1870 street staging.
- Interpretation:
  - Strong early denoising and concept binding for step 4000.
  - Object placement and separability are not yet solved.

Step 8000 t=0.30 denoised image:

- User observation:
  - Looks the same as step 4000.
  - Texture improved slightly.
- Interpretation:
  - This is expected for the current preview mode.
  - The preview starts from the original cached image latent, adds only `t=0.30` noise, and denoises it.
  - At `t=0.30`, much of the original structure remains, so the preview mostly tests cleanup/reconstruction, not fresh prompt-to-image generation.

Step 12000 t=0.30 denoised image:

- User observation:
  - Still only texture improved.
  - Composition remains effectively the same as steps 4000 and 8000.
- Interpretation:
  - This confirms the t=0.30 preview is acting mainly as a texture/detail denoising monitor.
  - It is not currently a useful signal for semantic rearrangement, object placement, or fresh generation ability.

## Warnings Seen

Warnings during preview/checkpoint period:

- Diffusers Flax classes deprecation warning:
  - Interpreted as harmless library warning.
- Hugging Face `local_dir_use_symlinks` deprecation warning:
  - Interpreted as harmless.
- STORK warning at `/content/STORK/STORKScheduler.py:711` about using `torch.tensor(sourceTensor)`:
  - Repeated during preview.
  - Interpreted as noisy but not a training blocker.

## Current Assessment

Run health:

- Training is stable.
- W&B initialized and synced.
- Checkpoint saving works.
- Preview generation works.
- Throughput is good for a 669M parameter BF16 model at batch size 80.
- No crashes, NaNs, or argument/config mismatches reported.

Learning signal:

- The model learned useful denoising quickly.
- Step 4000 already produced a recognizable denoised vintage street/woman/owl image.
- Step 8000 improved texture slightly but did not substantially change composition.
- Loss plateaued in the low-to-mid 2s after the initial drop.

Important interpretation:

- The periodic preview is a controlled denoising preview, not a full generative sample.
- Similar composition between step 4000 and step 8000 is expected because `t=0.30` keeps much of the original image structure.
- Actual prompt-to-image quality should be evaluated with `tools/sample_latent_dit.py` from random noise.

## Suggested Sampling Test

Use `tools/sample_latent_dit.py` to test true generation from the step 8000 checkpoint.

Suggested EMA sample:

```python
%cd /content/Boomer

PROMPT = """A woman holding a cloud in her hand with open eyes on a street in 1870, with a big owl in the background. The photograph is captured using Ilford HP5 and Kodak Tri-X films, on a 35mm medium format, showcasing a vintage black and white aesthetic with grainy texture, reminiscent of the photographic style of Atget in Paris."""

!python tools/sample_latent_dit.py \
  --checkpoint "{CKPT_DIR}/latent_dit_step_008000.pt" \
  --out "{PREVIEW_DIR}/sample_step008000_cfg2.png" \
  --prompt "{PROMPT}" \
  --steps 32 \
  --sampler stork2 \
  --stork-substeps 5 \
  --cfg-scale 2.0 \
  --cfg-rescale 0.5 \
  --weights ema \
  --hf-token true
```

Suggested live-model comparison:

```python
!python tools/sample_latent_dit.py \
  --checkpoint "{CKPT_DIR}/latent_dit_step_008000.pt" \
  --out "{PREVIEW_DIR}/sample_step008000_model_cfg2.png" \
  --prompt "{PROMPT}" \
  --steps 32 \
  --sampler stork2 \
  --stork-substeps 5 \
  --cfg-scale 2.0 \
  --cfg-rescale 0.5 \
  --weights model \
  --hf-token true
```

Reason:

- EMA can be conservative early.
- The live model can be messier but may reveal whether semantics are moving faster than the EMA preview shows.

## True Sampling Result

Step 8000 EMA fresh prompt sample:

- Command path succeeded after the earlier Gemma4/Transformers issue was resolved in the Colab environment.
- Output file:
  - `/content/sample_step008000_cfg2.png`
- Checkpoint/sample settings:
  - Checkpoint: step 8000 `.pt`
  - Weights: `ema`
  - Sampler: `stork2`
  - Steps: `32`
  - CFG scale: `2.0`
  - CFG rescale: `0.5`
  - Model type: `boomer_fla`
  - Flow shift: `3.0`
  - `y_norm = True`
  - `y_norm_scale_factor = 0.01`
  - Latent normalizer loaded from checkpoint:
    - Mean: 32 channels, mean `0.070316`, min `-0.238635`, max `0.872752`
    - Std: 32 channels, mean `0.887407`, min `0.718289`, max `2.319551`
- Prompt:
  - Same woman/cloud/owl/1870 vintage black-and-white street prompt used for preview discussion.

Visual result:

- The sample is true random-noise generation, not a t=0.30 denoise preview.
- It produced a black-and-white, high-contrast image with some photographic texture.
- It did not yet produce a coherent woman, cloud, street, or owl composition.
- The image is abstract/collapsed with face-like and animal-like fragments, strong blur, and distorted structure.

Interpretation:

- The generation pipeline works end to end:
  - Checkpoint loads.
  - Prompt conditioning loads.
  - STORK sampling runs.
  - VAE decode succeeds.
  - Latent normalization is being read from the checkpoint.
- The model is not yet a useful pure generator at step 8000.
- The t=0.30 denoise previews are much cleaner because they preserve most of the original cached latent structure.
- This confirms the earlier diagnosis: preview quality is showing local denoising ability, while fresh generation still needs more training and/or sampling/config exploration.

Suggested immediate comparisons:

- Sample step 12000 or 16000 once checkpoints exist.
- Compare `--weights ema` against `--weights model`.
- Compare CFG values:
  - `--cfg-scale 1.0`
  - `--cfg-scale 1.5`
  - `--cfg-scale 2.0`
- Keep `--cfg-rescale 0.5` for CFG values above 1.
- If images remain collapsed at all CFG values, treat it as undertraining rather than a sampler-only problem.

Step 12000 dog prompt sampling:

- Prompt: `a photograph of a dog`
- EMA sample:
  - Weights: `ema`
  - Sampler: `stork2`
  - Steps: `34`
  - CFG scale: `1.0`
  - CFG rescale: `0.0`
  - Output: `/content/sample_step012000_cfg4dog.png`
  - Result: collapsed face/fur-like image, no coherent dog.
- Live model sample:
  - Weights: `model`
  - Sampler: `stork2`
  - Steps: `34`
  - CFG scale: `1.0`
  - CFG rescale: `0.0`
  - Output: `/content/sample_step012000_cfg4dogwe.png`
  - Result: essentially the same collapsed output as EMA.
- Interpretation:
  - Prompt complexity is not the issue.
  - CFG is not the issue because CFG scale 1 disables guidance.
  - EMA lag is not the issue because live model and EMA look nearly identical.
  - Per-channel latent normalization is confirmed in the sample logs.

Step 12000 high-t denoise diagnostic:

- Cache caption: `A nostalgic pastimes scene.`
- Weights: `model`
- Sampler: `stork2`
- Steps: `32`
- Output dir: `/content/denoise_step012000_t080`
- `t = 0.80`
- `latent_mse_noised = 1.405084`
- `latent_mse_denoised = 1.394950`
- Visual result:
  - Still heavily abstract/collapsed.
  - Some color/texture and broad local structure appear, but original scene structure is not recovered.
- Interpretation:
  - This is the strongest evidence so far that the model has not learned high-noise/global denoising yet.
  - Full random-noise generation fails because the model cannot yet recover from the high-noise regime.
  - The earlier t=0.30 previews looked good because the original latent structure was still mostly present.

Step 12000 condition-cache full-noise sampling:

- Purpose:
  - Isolate whether fresh prompt encoding is the cause of collapsed random samples.
- Command used cached conditioning from:
  - `/content/drive/MyDrive/BoomerMain/preview/records/preview_00.pt`
- Caption from cache:
  - `A nostalgic pastimes scene.`
- Settings:
  - Weights: `ema`
  - Sampler: `stork2`
  - Steps: `32`
  - CFG scale: `1.0`
  - CFG rescale: `0.0`
  - Seed: `0`
  - Output: `/content/sample_step012000_conditioncache_ema_cfg1.png`
- Result:
  - Output remained abstract/collapsed.
  - It visually matched the high-noise/random failure mode rather than the clean t=0.30 preview.
- Interpretation:
  - Fresh prompt encoding is not the primary cause.
  - Cached text embeddings also fail under full random-noise sampling.
  - The issue is still best explained by high-noise denoising/generation not being learned yet at step 12000.

Cached-vs-fresh text encoding check:

- Preview record caption:
  - `A nostalgic pastimes scene.`
- Cached embedding:
  - Shape: `[300, 1536]`
  - Dtype: `torch.float16`
  - Mean: `0.056470`
  - Std: `1.859423`
- Fresh embedding:
  - Shape: `[1, 300, 1536]`
  - Dtype: `torch.bfloat16`
  - Mean: `0.057456`
  - Std: `1.869608`
- Raw diff before squeezing batch dimension:
  - Max diff: `4.000000`
  - Mean diff: `0.054263`
  - Relative diff: `0.040708`
  - Masks match: `False`
  - Shapes match: `False`
- Interpretation:
  - The shape mismatch is expected because fresh `encode_caption` returns a batch dimension.
  - The mask mismatch check was also affected by shape mismatch: cached mask is `[300]`, fresh mask is `[1, 300]`.
  - Cached conditioning already failed under full-noise sampling, so fresh text encoding differences are not the primary cause of random-sample collapse.

Settled diagnosis after isolation tests:

- Sampling pipeline is functioning:
  - Per-channel latent normalization is loaded and applied.
  - Text tokenization/masking matches between cached and fresh encoding.
  - Fresh prompt encoding differences are small and not outcome-dominating.
  - CFG is not the cause because CFG scale 1.0 also collapses.
  - EMA lag is not the cause because live model and EMA samples are nearly identical.
  - Cached conditioning also collapses under full random-noise sampling.
- The key failure mode is high-noise denoising:
  - Low-t preview at `t=0.30` is coherent because much of the original latent structure remains.
  - High-t diagnostic at `t=0.80` barely reduced MSE:
    - `1.405084 -> 1.394950`
  - Full random-noise generation fails because it must traverse the high-t region where the vector field is not useful yet.
- Interpretation:
  - The model is learning the expected diffusion/flow curriculum.
  - Easier low-noise refinement appears first.
  - Hard high-noise/global recovery develops later.
  - Step 12000 is too early to expect good random-noise generation.

Recommended progress metric going forward:

- Track denoise MSE reduction by timestep at each major checkpoint.
- Suggested t values:
  - `0.20`
  - `0.50`
  - `0.80`
  - `0.95`
- Log both input and output MSE:
  - `latent_mse_noised_t020`
  - `latent_mse_denoised_t020`
  - `latent_mse_noised_t050`
  - `latent_mse_denoised_t050`
  - `latent_mse_noised_t080`
  - `latent_mse_denoised_t080`
  - `latent_mse_noised_t095`
  - `latent_mse_denoised_t095`
- Also track the ratio:
  - `mse_ratio = latent_mse_denoised / latent_mse_noised`
- Current t=0.80 ratio at step 12000:
  - `1.394950 / 1.405084 = 0.9928`
  - This means essentially no high-t denoising yet.
- Expected useful signal:
  - Ratio should begin dropping at higher t values as high-noise competence develops.
  - Random-noise samples should improve only after t=0.80 and t=0.95 diagnostics show meaningful reduction.

Training recommendation:

- Continue training toward the 100k target.
- Do not over-interpret random-noise samples at step 12k.
- Re-test random-noise generation around step 40k and 60k.
- Use high-t reconstruction diagnostics as the main progress monitor in the meantime.

Resume after step 16000:

- User killed the run after the step 16000 checkpoint/preview period.
- Code change made before resume:
  - `tools/overfit_latent_dit.py` now moves CPU batch tensors to the target device first, then casts floating tensors to the training dtype.
  - This keeps fp16-to-bf16 conversion on GPU instead of CPU when using pinned memory / async H2D transfer.
- Verification:
  - `python -m pytest tests/test_flow_and_model.py tests/test_denoise_and_preview.py -q`
  - Result: 23 passed.
  - `python tools/overfit_latent_dit.py --help` parsed successfully.
- Resume recommendation:
  - Resume from `/content/drive/MyDrive/BoomerMain/checkpoints/boomer_fla_journeydb_rope/latent_dit_step_016000.pt`.
  - Keep `--steps 100000`.
  - Change preview diagnostics to high-t with `--preview-t 0.8`.

Rolling WDS coverage update:

- Code change made:
  - `RollingLocalWdsCache` now covers remote shards in shuffled epochs instead of choosing each replacement shard randomly with replacement.
  - It still keeps only the configured local rolling window on disk, for example 25 shards.
  - It shuffles the full remote shard list, consumes every shard once, then reshuffles for the next pass.
  - Prefetching now pulls from this epoch queue as well.
- Why:
  - Better dataset coverage.
  - Avoids repeating some shards before many others are seen.
  - Makes it meaningful to think in approximate full-dataset passes.
- Verification:
  - `python -m pytest tests/test_wds_rolling.py tests/test_wds_latents.py -q`
  - Result: 10 passed.
  - Focused model/preview tests also still passed: 23 passed.
- Note:
  - A currently running Colab process will not pick up this change unless the updated source file is present before the Python process starts.
  - Restart/resume after uploading the updated Boomer code if strict shuffled shard epochs are required immediately.

## Operational Notes

Disk:

- Stats download can use about 48 GB in `/content/wds_stats`.
- Rolling training shards can use about 50 GB in `/content/wds_roll`.
- Combined `/content` disk pressure can be significant in Colab.
- After stats and preview records are built, deleting old stats shards may help if `/content` disk is tight.

Notebook unzip:

- The notebook uses `!unzip -q /content/Boomer.zip`.
- If `/content/Boomer` already exists, unzip can prompt or hang.
- Safer future command:
  - `!unzip -oq /content/Boomer.zip`

Stats recompute:

- `RECOMPUTE_STATS = True` forces stats recomputation.
- Once the Drive stats JSON is trusted, set it to `False` to avoid rebuilding.

FLA probe:

- Keep running `tools/probe_sequence_mixers.py` before training.
- If `GatedDeltaNet` fails there, the BoomerFLA training cell will fail.

## Next Checks

Recommended immediate checks:

1. Let the run continue to at least step 16000.
2. Compare the next preview metric against:
   - Step 4000: `0.128766`
   - Step 8000: `0.126395`
   - Step 12000: `0.125243`
3. Run true prompt sampling from the step 12000 `.pt` checkpoint.
4. Compare EMA and model weights for early generation behavior.
5. Keep watching for:
   - NaNs
   - checkpoint save failures
   - preview failures
   - sustained step-time slowdown
   - `/content` disk pressure

## Sana Repo Check

Local Sana repos inspected:

- `../Sana`
- `../STORK/external/Sana`

Relevant Sana defaults:

- 512px Sana config uses `predict_flow_v: true`, `noise_schedule: linear_flow`, `flow_shift: 3.0`, `weighting_scheme: logit_normal`, `logit_mean: 0.0`, `logit_std: 1.0`.
- 1024px 1600M configs also use `flow_shift: 3.0` and `weighting_scheme: logit_normal`.
- 600M 1024px config uses `flow_shift: 4.0`.
- 4096px config uses `flow_shift: 6.0`.
- Base/sample/CI configs sometimes use `flow_shift: 1.0`.

Important loss-path finding:

- Sana samples timesteps with logit-normal, then calls `train_diffusion.training_losses(model, clean_images, timesteps, ...)`.
- In the normal `train.py` path, it does not pass `timestep_weight=True`.
- Sana's `training_losses(...)` has an optional `timestep_weight=False` argument, but the ordinary image training path leaves it off.
- Therefore, Boomer's current `--timestep-weight` is not part of the standard Sana-like training setup.

Implication:

- If matching Sana more closely in a future fresh ablation, keep `--t-sampler logit-normal --logit-mean 0.0 --logit-std 1.0 --flow-shift 3.0`, but remove `--timestep-weight`.
- Current run should still continue unless high-t metrics stop improving; the step 20000 high-t ratio already improved meaningfully versus step 12000.

## BoomerMain2 Fresh Full Run, No Timestep Weight

Fresh full run setup:

- Drive root changed by user to `BoomerMain2`.
- Removed `--timestep-weight`.
- Kept Sana-like `--t-sampler logit-normal --logit-mean 0.0 --logit-std 1.0 --flow-shift 3.0`.
- Preview changed to pure-noise stress test: `--preview-t 1.0`.
- EMA changed to slower full-run compromise: `--ema-decay 0.9992 --ema-update-every 8`.
- Local rolling WDS cache cleared before run.

Early loss:

- Step 1: loss `1.954617`, loss_ma `1.954617`, step_s `25.08`.
- Step 50: loss `2.019329`, loss_ma `2.028998`, step_s `0.86`.
- Step 100: loss `1.997601`, loss_ma `2.035077`, step_s `0.90`.

Pure-noise preview metrics (`t=1.0`, noised MSE fixed at `2.093993`):

| Step | Denoised MSE | Ratio |
| ---: | ---: | ---: |
| 4000 | 1.728077 | 0.825 |
| 8000 | 1.420080 | 0.678 |
| 12000 | 1.615210 | 0.771 |
| 16000 | 1.968392 | 0.940 |
| 20000 | 2.239205 | 1.069 |
| 24000 | 2.308835 | 1.103 |
| 28000 | 2.244578 | 1.072 |
| 32000 | 2.148707 | 1.026 |
| 36000 | 2.073675 | 0.990 |
| 40000 | 2.032198 | 0.971 |
| 44000 | 2.010623 | 0.960 |

Step 44000 visual:

- Pure-noise EMA preview is still distorted but has clear face/person structure and broad scene organization.
- Metrics worsened after the early 8000 checkpoint, then recovered steadily from 24000 through 44000.
- Since the preview uses slow EMA and `t=1.0`, it is a harsh high-noise generation test; raw model vs EMA should be compared around 44000/48000 to check EMA lag.

Step 48000:

- Preview metric: noised `2.093993`, denoised `1.997278`, ratio `0.954`.
- Dog prompt full sample was run with both `weights=model` and `weights=ema`; user reported both dog images looked the same.
- Cached-condition full sample was run with both `weights=model` and `weights=ema`; user reported cache sample looked effectively the same as dog prompt.
- Dog prompt seeds `0..4` were run with EMA, CFG 1.0, 32 STORK2 steps; user reported different images across seeds.
- `sample_latent_dit.py` reproduced the old preview-like image for `preview_00`, EMA, CFG 1.0, seed 0, 16 STORK2 steps.
- Interpretation: EMA lag, fresh prompt encoding, and standalone sampler mismatch are not the main bottlenecks at 48k. Seed diversity exists, but conditioning from full noise is still weak and seed/basin effects dominate.

Min-SNR code update:

- Added `--min-snr-gamma` to `tools/overfit_latent_dit.py`.
- Default `0.0` disables it, preserving existing behavior.
- For flow noising `x_t=(1-sigma)x0+sigma*noise`, SNR is computed as `((1-sigma)/sigma)^2`.
- Loss multiplier is `min(SNR, gamma) / SNR`, which downweights low-noise/high-SNR samples and leaves high-noise/low-SNR samples at weight `1`.
- This is separate from `--timestep-weight`; do not use both for the Min-SNR ablation.
- Verification: `python -m pytest tests/test_flow_and_model.py -q` passed, 19 tests. `python tools/overfit_latent_dit.py --help` parsed and showed `--min-snr-gamma`.

BoomerFLA model speed cleanup:

- `apply_condition_dropout` already used the no-host-sync `torch.where` path locally: no `bool(drop.any())`, no clone path.
- `RoPE2D` now precomputes broadcasted height/width cos/sin tables once in `__init__` instead of rebuilding token indices and cat/index tables every forward.
- `BoomerFLADiT.forward` now computes `t_embedder(timesteps)` once and reuses it for both the block modulation and final modulation.
- SDPA cross-attention now accepts a prebuilt additive attention bias; `BoomerFLADiT.forward` builds it once per forward instead of rebuilding it in every block. Bool mask is still preserved for xformers and `nn.MultiheadAttention`.
- Verification: `python -m py_compile boomer/models/boomer_fla_dit.py` passed. `python -m pytest tests/test_flow_and_model.py tests/test_denoise_and_preview.py -q` passed, 24 tests.

Plateau logit-normal sampler:

- Added `--t-sampler bounded-logit-normal` for the earlier bounded/stretched logit-normal variant.
- Replaced `--t-sampler plateau-logit-normal` with the true plateau density from the user-provided formula:
  - find the logit-normal mode `t_star`
  - sample truncated logit-normal on `[0, t_star]`
  - sample a uniform plateau on `[t_star, 1]`
  - choose between the two regions using their normalized probability masses
- `--plateau-low` and `--plateau-high` now apply only to `bounded-logit-normal`; they are not used by true `plateau-logit-normal`.
- For `flow_shift=1.5`, `logit_mean=0.0`, `logit_std=1.0`, true plateau produced shifted sigma mean about `0.660`, median about `0.688`.
- Verification: `python -m pytest tests/test_flow_and_model.py tests/test_denoise_and_preview.py -q` passed, 26 tests. `python tools/overfit_latent_dit.py --help` shows both `bounded-logit-normal` and `plateau-logit-normal`.
- Follow-up optimization: `_standard_normal_icdf` now uses the closed-form `sqrt(2) * erfinv(2p - 1)` instead of constructing `torch.distributions.Normal(...).icdf(...)` during sampling. Tests still pass, 26 tests.

BoomerFLA adaLN-Gaussian init:

- Changed block modulation init from pure adaLN-Zero to small Gaussian:
  - `t_block[1].weight`: normal std `0.02`
  - per-block `scale_shift_table`: normal std `0.02`
  - image-attn `image_attn_scale_shift_table`: normal std `0.02`
- Kept final output path zeroed:
  - `final_t_block[1].weight/bias`: zero
  - `out_proj.weight/bias`: zero
- This lets internal modulation start non-degenerate while preserving initial zero model output.
- Verification: `python -m py_compile boomer/models/boomer_fla_dit.py` passed. `python -m pytest tests/test_flow_and_model.py tests/test_denoise_and_preview.py -q` passed, 26 tests.

BoomerMain4 run:

- Config:
  - true `plateau-logit-normal`
  - `flow_shift=1.5`
  - `min_snr_gamma=5.0`
  - `grad_clip=0.3`
  - adaLN-Gaussian block modulation init
  - compile `max-autotune-no-cudagraphs`
  - preview `t=1.0`
- Step time after compile: mostly `0.83`-`0.85s`.
- `t=1.0` preview metrics, noised MSE fixed at `2.093993`:

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

- Interpretation at 56k:
  - Much better than the earlier no-timestep-weight run at the same 20k-56k range.
  - But the `t=1.0` preview ratio bottomed around 8k, then worsened steadily.
  - Need raw-model vs EMA and sample-grid checks before deciding whether this is EMA lag / preview mismatch / real high-noise regression.
