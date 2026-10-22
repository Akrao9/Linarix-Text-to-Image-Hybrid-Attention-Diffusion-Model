# Training

`train.py` is the training entry point. These are the V8 experiment settings;
notebooks and saved experiment outputs stay local.

## Environment

Training uses Linux, an NVIDIA GPU, and CUDA-compatible PyTorch. The original V8
experiments ran on an RTX PRO 6000 Blackwell with 96 GB of VRAM. The full recipes
are not intended for a CPU or a small GPU without adjusting batch size.

Install PyTorch for your CUDA version first, then:

```bash
python -m pip install -e '.[train,cmuon,dev]'
python -m pip install --no-deps "git+https://github.com/fla-org/flash-linear-attention.git@3c4c54ae7397d37130d7101edd0f4eb596af896d"
python -m pip install --no-build-isolation causal-conv1d
python -m pip install --no-build-isolation flash-attn
```

The CUDA extensions must match the installed PyTorch and CUDA versions. FLA is
pinned because Block AttnRes requires `fla.ops.attnres.fused_attnres`. The `cmuon`
extra pins the Dion revision used by the optimizer integration.

Set `HF_TOKEN` or run `hf auth login` for data/model access. Set `WANDB_API_KEY`
when enabling W&B logging. Credentials belong in environment variables or Colab
secrets.

Training previews use STORK and its coefficient files. Place the upstream
repository beside this checkout:

```bash
git clone https://github.com/ZT220501/STORK.git ../STORK
```

## Phases

The MONET phases stream [`jasperai/monet`](https://huggingface.co/datasets/jasperai/monet)
`v1.2.0` directly from the Hugging Face Hub during training. The loaders read remote
Parquet or TAR shards as needed; a separate local training-cache build is not
required. FineT2I likewise streams
[`ma-xu/fine-t2i`](https://huggingface.co/datasets/ma-xu/fine-t2i) from the Hub.

The commands below use the notebook's model and training settings, with the 512px
REPA schedule recorded from the training history. Checkpoint paths and W&B names
use portable names; Drive copying is optional. Run them from the project root in
a Bash shell. Effective batch is batch size × accumulation on one GPU.

| Phase | Data | Image size | Batch × accumulation | LR | Configured step limit |
| --- | --- | --- | --- | --- | --- |
| `512` | MONET Parquet latents, six-source mix | 512 | 64 × 2 | 1e-4 | 300,000 |
| `1024-monet` | MONET synthetic images | 1024 | 16 × 4 | 2e-5 | 50,000 |
| `1024-finet2i` | FineT2I synthetic / curated, 75% / 25% | 1024 | 16 × 4 | 2e-5 | 35,000 |

The checkpoint sequence used for the published model is:

| New phase | Source checkpoint | Initial weights |
| --- | --- | --- |
| `1024-monet` | 512px step 245,000 | EMA; reset optimizer, scheduler, EMA shadow, and step |
| `1024-finet2i` | 1024px MONET step 47,500 | EMA; reset optimizer, scheduler, EMA shadow, and step |
| Published export | 1024px FineT2I step 35,000 | EMA |

512 training reads MONET's precomputed DC-AE latents. Both 1024 phases stream images
and run the frozen DC-AE encoder on the GPU. Qwen encodes captions on the GPU in
every phase. At 512px, REPA weight was `0.25` for the first 100,000 optimizer steps,
then training resumed with REPA weight `0`. Both 1024px phases use REPA weight `0`.

## 512px MONET

Start from scratch with the six-source latent mix. Adjust batch size and
accumulation for the available VRAM. The first launch below stops at step 100,000
with a constant REPA weight of `0.25` (`--repa-anneal-steps 0`). `--monet-dino`
explicitly enables MONET's precomputed DINOv2 targets for the alignment loss.

```bash
export WANDB_RUN_ID=linarix-512
export WANDB_RESUME=allow

python -u train.py \
  --monet-direct --monet-dino --devices 1 \
  --cache-wds 'hf://datasets/jasperai/monet/v1.2.0/synthetic/*/*/*.parquet,hf://datasets/jasperai/monet/v1.2.0/cc12m/*/*.parquet,hf://datasets/jasperai/monet/v1.2.0/megalith10m/*/*.parquet,hf://datasets/jasperai/monet/v1.2.0/laion/*/*.parquet,hf://datasets/jasperai/monet/v1.2.0/coyo/*/*.parquet,hf://datasets/jasperai/monet/v1.2.0/commoncatalog-cc-by/*/*.parquet' \
  --monet-weights 0.15,0.10,0.15,0.30,0.15,0.15 \
  --monet-caption-probs '0,0,0,0,1;.25,.25,.25,.25,0;.25,.25,.25,.25,0;.25,.25,.25,.25,0;.25,.25,.25,.25,0;.25,.25,.25,.25,0' \
  --text-encoder-id Qwen/Qwen3.5-4B --text-encoder-layers 9,18,27 --text-encoder-max-length 320 \
  --text-encoder-instruction --compile-text-encoder --hf-token true --latent-channels 32 --latent-size 16 \
  --extra-input-channels 33 --latent-mean 0.0 --latent-std 1.0 --num-workers 16 --prefetch-factor 4 \
  --persistent-workers --pin-memory --wds-shuffle-buffer 20000 --parquet-read-batch-size 64 \
  --parquet-range-mib 128 --hidden-dim 1152 --depth 28 --ddt-decoder-depth 6 --num-heads 18 --mlp-ratio 2.5 \
  --mixer-type fla_gdn2 --fla-mode chunk --fla-bidirectional --use-short-conv --conv-size 4 \
  --image-attn-every 0 --image-attn-backend flash2 --image-rope --image-rope-theta 10000 \
  --cross-attn-backend sdpa --image-attn-qk-rmsnorm --cross-attn-qk-rmsnorm --joint-attn-backend sdpa \
  --ddt-norm-impl native --shared-adaln --fourier-timestep --hilbert-order --parallel-block \
  --dual-stream-depth 4 --multimodal-coord-ids --joint-refresh-attn --refresh-layers 7,11,15,21 \
  --refresh-no-mixer --block-attnres --sandwich-rmsnorm --rope-ntk-factor 1.0 --no-abs-pos-embed --y-norm \
  --y-norm-scale-factor 0.01 --patch-size 1 --repa-dim 1536 --repa-layer 7 --repa-weight 0.25 \
  --repa-anneal-steps 0 --batch-size 64 --accumulate 2 --optimizer cmuon --lr 1e-4 --weight-decay 0.0 \
  --beta1 0.9 --beta2 0.95 --fused-adamw --grad-clip 1.0 --lr-scheduler linear-warmup-constant \
  --warmup-steps 5000 --cond-dropout 0.1 --t-sampler logit-normal --logit-mean 0.0 --logit-std 1.0 \
  --flow-shift 1.5 --min-snr-gamma 0 --t-sampler-final uniform --t-sampler-final-step 300000 \
  --ema-decay 0.9999 --ema-update-every 1 --dtype bf16 --max-steps 100000 --seed 42 --compile \
  --compile-mode default --checkpoint-dir checkpoints/v8-512 --save-every-steps 5000 \
  --checkpoint-local-retain 2 --checkpoint-keep-uploaded 10 --sample-every-steps 10000 \
  --sample-prompts-file eval_prompts.txt --sample-weights ema --sample-steps 20 --sample-sampler stork4 \
  --sample-stork-derivative-order 1 --sample-stork-substeps 9 --sample-flow-shift 5 --sample-cfg-scale 3.0 \
  --sample-cfg-rescale 0.5 --sample-cfg-interval-low 0.1 --sample-cfg-interval-high 0.9 --sample-seed 42 \
  --log-every-steps 50 --wandb --wandb-project linarix --wandb-run-name linarix-512
```

### Continue 512px with REPA disabled

At step 100,000, resume the same run with `--repa-weight 0`. This is a full resume:
retain optimizer, scheduler, EMA, and global step. Keep `--repa-dim 1536` and the
same model layout so checkpoint tensor shapes match. The continuation keeps the
notebook's overall `--max-steps 300000` limit; the checkpoint used to seed 1024px
was step 245,000.

```bash
export WANDB_RUN_ID=linarix-512
export WANDB_RESUME=allow

python -u train.py \
  --monet-direct --devices 1 \
  --cache-wds 'hf://datasets/jasperai/monet/v1.2.0/synthetic/*/*/*.parquet,hf://datasets/jasperai/monet/v1.2.0/cc12m/*/*.parquet,hf://datasets/jasperai/monet/v1.2.0/megalith10m/*/*.parquet,hf://datasets/jasperai/monet/v1.2.0/laion/*/*.parquet,hf://datasets/jasperai/monet/v1.2.0/coyo/*/*.parquet,hf://datasets/jasperai/monet/v1.2.0/commoncatalog-cc-by/*/*.parquet' \
  --monet-weights 0.15,0.10,0.15,0.30,0.15,0.15 \
  --monet-caption-probs '0,0,0,0,1;.25,.25,.25,.25,0;.25,.25,.25,.25,0;.25,.25,.25,.25,0;.25,.25,.25,.25,0;.25,.25,.25,.25,0' \
  --text-encoder-id Qwen/Qwen3.5-4B --text-encoder-layers 9,18,27 --text-encoder-max-length 320 \
  --text-encoder-instruction --compile-text-encoder --hf-token true --latent-channels 32 --latent-size 16 \
  --extra-input-channels 33 --latent-mean 0.0 --latent-std 1.0 --num-workers 16 --prefetch-factor 4 \
  --persistent-workers --pin-memory --wds-shuffle-buffer 20000 --parquet-read-batch-size 64 \
  --parquet-range-mib 128 --hidden-dim 1152 --depth 28 --ddt-decoder-depth 6 --num-heads 18 --mlp-ratio 2.5 \
  --mixer-type fla_gdn2 --fla-mode chunk --fla-bidirectional --use-short-conv --conv-size 4 \
  --image-attn-every 0 --image-attn-backend flash2 --image-rope --image-rope-theta 10000 \
  --cross-attn-backend sdpa --image-attn-qk-rmsnorm --cross-attn-qk-rmsnorm --joint-attn-backend sdpa \
  --ddt-norm-impl native --shared-adaln --fourier-timestep --hilbert-order --parallel-block \
  --dual-stream-depth 4 --multimodal-coord-ids --joint-refresh-attn --refresh-layers 7,11,15,21 \
  --refresh-no-mixer --block-attnres --sandwich-rmsnorm --rope-ntk-factor 1.0 --no-abs-pos-embed --y-norm \
  --y-norm-scale-factor 0.01 --patch-size 1 --repa-dim 1536 --repa-layer 7 --repa-weight 0 \
  --repa-anneal-steps 0 --batch-size 64 --accumulate 2 --optimizer cmuon --lr 1e-4 --weight-decay 0.0 \
  --beta1 0.9 --beta2 0.95 --fused-adamw --grad-clip 1.0 --lr-scheduler linear-warmup-constant \
  --warmup-steps 5000 --cond-dropout 0.1 --t-sampler logit-normal --logit-mean 0.0 --logit-std 1.0 \
  --flow-shift 1.5 --min-snr-gamma 0 --t-sampler-final uniform --t-sampler-final-step 300000 \
  --ema-decay 0.9999 --ema-update-every 1 --dtype bf16 --max-steps 300000 --seed 42 --compile \
  --compile-mode default --checkpoint-dir checkpoints/v8-512 --save-every-steps 5000 \
  --checkpoint-local-retain 2 --checkpoint-keep-uploaded 10 --sample-every-steps 10000 \
  --sample-prompts-file eval_prompts.txt --sample-weights ema --sample-steps 20 --sample-sampler stork4 \
  --sample-stork-derivative-order 1 --sample-stork-substeps 9 --sample-flow-shift 5 --sample-cfg-scale 3.0 \
  --sample-cfg-rescale 0.5 --sample-cfg-interval-low 0.1 --sample-cfg-interval-high 0.9 --sample-seed 42 \
  --log-every-steps 50 --wandb --wandb-project linarix --wandb-run-name linarix-512 \
  --resume checkpoints/v8-512/boomer-step=100000.ckpt
```

## Prepare an EMA seed

Define this function once in the same Bash shell before launching either 1024
phase. It mirrors the notebook's EMA promotion: every floating-point model tensor
must have a matching EMA tensor, and the output contains model weights and
provenance only. The commands below keep `--compile`, matching the source runs'
`model._orig_mod.*` state keys.

```bash
promote_ema() {
  python - "$1" "$2" <<'PY'
from pathlib import Path
import sys
import torch

source_path, output_path = map(Path, sys.argv[1:])
if source_path.resolve() == output_path.resolve():
    raise ValueError("The seed output must differ from the source checkpoint.")
checkpoint = torch.load(source_path, map_location="cpu", weights_only=False)
state, ema = checkpoint.get("state_dict"), checkpoint.get("ema")
if not isinstance(state, dict) or not isinstance(ema, dict) or not ema:
    raise ValueError("The source checkpoint must contain model state and EMA.")
model_keys = {
    key for key, value in state.items()
    if key.startswith("model.") and torch.is_floating_point(value)
}
if model_keys != {f"model.{key}" for key in ema}:
    raise ValueError("EMA must cover every floating-point model tensor exactly.")
promoted = dict(state)
changed = 0
for key, value in ema.items():
    target = f"model.{key}"
    if state[target].shape != value.shape:
        raise ValueError(f"EMA shape mismatch: {target}")
    changed += not torch.equal(state[target], value)
    promoted[target] = value
if changed == 0:
    raise ValueError("EMA is identical to live weights; check the source run's EMA.")
output_path.parent.mkdir(parents=True, exist_ok=True)
torch.save({
    "state_dict": promoted,
    "source_checkpoint": str(source_path),
    "source_step": checkpoint.get("global_step"),
    "weights_source": "ema",
}, output_path)
print(f"Promoted {len(ema)} EMA tensors; wrote {output_path}")
PY
}
```

## 1024px MONET

Seed from the 512px step-245,000 EMA. Stream only MONET's `synthetic` image TARs,
use original captions, and require a native short side of at least 1024px.
The latent grid is 32 × 32; RoPE factor is 2.0, batch is 16 × 4, and DC-AE encoding
uses chunks of 4 images. The uniform timestep switch is configured at step 48,000
of the 50,000-step limit; the checkpoint used for FineT2I was step 47,500.

```bash
export WANDB_RUN_ID=linarix-1024-monet
export WANDB_RESUME=allow

promote_ema \
  checkpoints/v8-512/boomer-step=245000.ckpt \
  checkpoints/seeds/v8-1024-monet-ema.ckpt

python -u train.py \
  --monet-image-direct --devices 1 --cache-wds 'hf://datasets/jasperai/monet/v1.2.0/synthetic/**/*.tar' \
  --monet-caption-probs 0,0,0,0,1 --monet-image-resolution 1024 --monet-image-min-short-side 1024 \
  --monet-dino --text-encoder-id Qwen/Qwen3.5-4B --text-encoder-layers 9,18,27 \
  --text-encoder-max-length 320 --text-encoder-instruction --compile-text-encoder --hf-token true \
  --latent-channels 32 --latent-size 32 --extra-input-channels 33 --latent-mean 0.0 --latent-std 1.0 \
  --num-workers 14 --prefetch-factor 4 --persistent-workers --pin-memory --wds-shuffle-buffer 512 \
  --hidden-dim 1152 --depth 28 --ddt-decoder-depth 6 --num-heads 18 --mlp-ratio 2.5 --mixer-type fla_gdn2 \
  --fla-mode chunk --fla-bidirectional --use-short-conv --conv-size 4 --image-attn-every 0 \
  --image-attn-backend flash2 --image-rope --image-rope-theta 10000 --cross-attn-backend sdpa \
  --cross-attn-every 1 --image-attn-qk-rmsnorm --cross-attn-qk-rmsnorm --joint-attn-backend sdpa \
  --ddt-norm-impl native --shared-adaln --fourier-timestep --hilbert-order --parallel-block \
  --dual-stream-depth 4 --multimodal-coord-ids --joint-refresh-attn --refresh-layers 7,11,15,21 \
  --refresh-no-mixer --block-attnres --sandwich-rmsnorm --rope-ntk-factor 2.0 --no-abs-pos-embed --y-norm \
  --y-norm-scale-factor 0.01 --patch-size 1 --repa-dim 1536 --repa-layer 7 --repa-weight 0 \
  --repa-anneal-steps 0 --batch-size 16 --accumulate 4 --optimizer cmuon --lr 2e-5 --weight-decay 0.0 \
  --beta1 0.9 --beta2 0.95 --grad-clip 1.0 --lr-scheduler linear-warmup-constant --warmup-steps 1000 \
  --cond-dropout 0.1 --t-sampler logit-normal --logit-mean 0.0 --logit-std 1.0 --flow-shift 3.0 \
  --min-snr-gamma 0 --t-sampler-final uniform --t-sampler-final-step 48000 --ema-decay 0.9999 \
  --ema-update-every 1 --dtype bf16 --max-steps 50000 --seed 42 --compile --compile-mode default \
  --compile-vae --edit-encode-chunk 4 --edit-scaling-factor 0.41407 \
  --checkpoint-dir checkpoints/v8-1024-monet --save-every-steps 2500 --checkpoint-local-retain 2 \
  --checkpoint-keep-uploaded 10 --sample-every-steps 5000 --sample-prompts-file eval_prompts.txt \
  --sample-weights ema --sample-steps 20 --sample-sampler stork4 --sample-stork-derivative-order 1 \
  --sample-stork-substeps 14 --sample-flow-shift 5 --sample-cfg-scale 3.0 --sample-cfg-rescale 0.5 \
  --sample-cfg-interval-low 0.1 --sample-cfg-interval-high 0.9 --sample-scaling-factor 0.41407 \
  --sample-seed 42 --log-every-steps 50 --wandb --wandb-project linarix --wandb-run-name linarix-1024-monet \
  --resume checkpoints/seeds/v8-1024-monet-ema.ckpt --resume-weights-only
```

## 1024px FineT2I

Seed from the 1024px MONET step-47,500 EMA. Stream
`synthetic_enhanced_prompt_square_resolution` and `curated` with weights 0.75/0.25.
Caption sampling uses `--t2i-short-prob 0.5`. The model, batch, learning rate, and
VAE chunk size match the MONET 1024 phase; this phase runs to step 35,000.

`--max-screened-per-shard "64,0"` applies a rejection screen to the synthetic
source and reads curated shards to the end. Preserve that per-source setting.

```bash
export WANDB_RUN_ID=linarix-1024-finet2i
export WANDB_RESUME=allow

promote_ema \
  checkpoints/v8-1024-monet/boomer-step=47500.ckpt \
  checkpoints/seeds/v8-1024-finet2i-ema.ckpt

python -u train.py \
  --t2i-direct --devices 1 --t2i-repo ma-xu/fine-t2i \
  --t2i-data-dir synthetic_enhanced_prompt_square_resolution,curated --t2i-weights 0.75,0.25 \
  --t2i-caption-mode sample --t2i-short-prob 0.5 --t2i-resolution 1024 --t2i-min-short-side 1024 \
  --text-encoder-id Qwen/Qwen3.5-4B --text-encoder-layers 9,18,27 --text-encoder-max-length 320 \
  --text-encoder-instruction --compile-text-encoder --hf-token true --latent-channels 32 --latent-size 32 \
  --extra-input-channels 33 --latent-mean 0.0 --latent-std 1.0 --num-workers 14 --prefetch-factor 4 \
  --persistent-workers --pin-memory --wds-shuffle-buffer 512 --hidden-dim 1152 --depth 28 \
  --ddt-decoder-depth 6 --num-heads 18 --mlp-ratio 2.5 --mixer-type fla_gdn2 --fla-mode chunk \
  --fla-bidirectional --use-short-conv --conv-size 4 --image-attn-every 0 --image-attn-backend flash2 \
  --image-rope --image-rope-theta 10000 --cross-attn-backend sdpa --cross-attn-every 1 \
  --image-attn-qk-rmsnorm --cross-attn-qk-rmsnorm --joint-attn-backend sdpa --ddt-norm-impl native \
  --shared-adaln --fourier-timestep --hilbert-order --parallel-block --dual-stream-depth 4 \
  --multimodal-coord-ids --joint-refresh-attn --refresh-layers 7,11,15,21 --refresh-no-mixer \
  --block-attnres --sandwich-rmsnorm --rope-ntk-factor 2.0 --no-abs-pos-embed --y-norm \
  --y-norm-scale-factor 0.01 --patch-size 1 --repa-dim 1536 --repa-layer 7 --repa-weight 0 \
  --repa-anneal-steps 0 --batch-size 16 --accumulate 4 --optimizer cmuon --lr 2e-5 --weight-decay 0.0 \
  --beta1 0.9 --beta2 0.95 --grad-clip 1.0 --lr-scheduler linear-warmup-constant --warmup-steps 1000 \
  --cond-dropout 0.1 --t-sampler logit-normal --logit-mean 0.0 --logit-std 1.0 --flow-shift 3.0 \
  --min-snr-gamma 0 --t-sampler-final uniform --t-sampler-final-step 35000 --ema-decay 0.9999 \
  --ema-update-every 1 --dtype bf16 --max-steps 35000 --seed 42 --compile --compile-mode default \
  --compile-vae --edit-encode-chunk 4 --max-screened-per-shard 64,0 --edit-scaling-factor 0.41407 \
  --checkpoint-dir checkpoints/v8-1024-finet2i --save-every-steps 2500 --checkpoint-local-retain 2 \
  --checkpoint-keep-uploaded 10 --sample-every-steps 5000 --sample-prompts-file eval_prompts.txt \
  --sample-weights ema --sample-steps 20 --sample-sampler stork4 --sample-stork-derivative-order 1 \
  --sample-stork-substeps 14 --sample-flow-shift 5 --sample-cfg-scale 3.0 --sample-cfg-rescale 0.5 \
  --sample-cfg-interval-low 0.1 --sample-cfg-interval-high 0.9 --sample-scaling-factor 0.41407 \
  --sample-seed 42 --log-every-steps 50 --wandb --wandb-project linarix \
  --wandb-run-name linarix-1024-finet2i \
  --resume checkpoints/seeds/v8-1024-finet2i-ema.ckpt --resume-weights-only
```

## Resume and checkpoint copying

EMA promotion is for the first launch of a new phase. To continue an interrupted
phase, run its same training command but replace the final
`--resume checkpoints/seeds/... --resume-weights-only` with a checkpoint from that
phase, for example:

```bash
# Final resume arguments for the MONET command:
--resume checkpoints/v8-1024-monet/boomer-step=2500.ckpt

# Final resume arguments for the FineT2I command:
--resume checkpoints/v8-1024-finet2i/boomer-step=2500.ckpt
```

Omit `--resume-weights-only` on a full resume so optimizer, scheduler, global step,
and EMA are restored. Keep the rest of the phase's command unchanged. For an
interrupted 512 run, choose the REPA-enabled or REPA-disabled command according
to the checkpoint step, and resume from `checkpoints/v8-512/boomer-step=STEP.ckpt`.
At or after step 100,000, use the REPA-disabled continuation.

Keep the phase's `WANDB_RUN_ID` when resuming so W&B continues the same run.
Use a new ID and checkpoint directory for a separate experiment.

The launch commands write to local `--checkpoint-dir` directories. To copy
checkpoints asynchronously to mounted Drive storage, add
`--checkpoint-upload-remote /path/to/mounted-drive/checkpoints/RUN` to the command.
Mount Drive before training. The notebook's retention settings are preserved:
2 local step checkpoints and 10 uploaded step checkpoints when copying is enabled.

## Sample and export

```bash
python tools/sample_latent_dit.py \
  --checkpoint checkpoints/run/boomer-step=35000.ckpt \
  --weights ema \
  --prompt "a lighthouse on a rocky cliff at golden hour" \
  --out outputs/lighthouse.png \
  --sampler stork4 --steps 20 --stork-derivative-order 1 --stork-substeps 14 \
  --cfg-scale 4 --cfg-rescale 0.5 --flow-shift 5 \
  --cfg-interval-low 0.1 --cfg-interval-high 0.9 \
  --dtype bf16 --device cuda --hf-token true

python tools/export_to_hf.py \
  --checkpoint checkpoints/run/boomer-step=35000.ckpt \
  --weights ema --out outputs/Linarix-v2 \
  --model-name Linarix-v2 --repo-id YOUR_USERNAME/YOUR_MODEL \
  --stork-dir ../STORK --save-dtype bf16
```

Export validates the weight/configuration match and writes a standalone
Diffusers pipeline. Uploading is opt-in with `--push`.

## Checks

```bash
python -m pytest tests/ -q
python tools/gen_modeling_file.py --check
```

On the training GPU, `python tools/attnres_parity.py` checks the fused residual
router against its reference. `python tools/attn_backend_probe.py` probes the
attention backends. The recipes use SDPA for joint attention.
