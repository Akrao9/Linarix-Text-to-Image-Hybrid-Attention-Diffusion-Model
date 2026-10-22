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
required. The FineT2I phase likewise streams
[`ma-xu/fine-t2i`](https://huggingface.co/datasets/ma-xu/fine-t2i) from the Hub.

These are the command settings extracted from the V8 experiment notebook.
Effective batch is per-device batch × gradient accumulation on one GPU.

| Phase | Data | Image size | Batch × accumulation | LR | Steps |
| --- | --- | --- | --- | --- | --- |
| `512` | MONET parquet latents, six-source mix | 512 | 64 × 2 | 1e-4 | 300,000 |
| `1024-monet` | MONET synthetic images | 1024 | 16 × 4 | 2e-5 | 50,000 |
| `1024-finet2i` | FineT2I synthetic / curated, 75% / 25% | 1024 | 16 × 4 | 2e-5 | 35,000 |
| `2048` | MONET diffusion-aesthetic-4k images | 2048 | 4 × 16 | 5e-6 | 1,000 |

The intended sequence is `512` → `1024-monet` → `1024-finet2i` → `2048`.
The source runs used step 245,000 from 512, step 47,500 from 1024 MONET, and
step 35,000 from FineT2I to seed the next phase. The 2048 phase remains experimental.

512 training reuses MONET's precomputed DC-AE latents. The higher-resolution
phases decode source images and run the frozen DC-AE encoder in chunks on the GPU.
Qwen encodes captions on the GPU in every phase. REPA is disabled in these recipes
(`--repa-weight 0`), including the MONET image phase.

## Start a 512px run

This command uses the V8 model layout and streams MONET's six-source latent mix.
Adjust batch size and accumulation for the available VRAM.

```bash
python -u train.py \
  --monet-direct --devices 1 \
  --cache-wds "hf://datasets/jasperai/monet/v1.2.0/synthetic/*/*/*.parquet,hf://datasets/jasperai/monet/v1.2.0/cc12m/*/*.parquet,hf://datasets/jasperai/monet/v1.2.0/megalith10m/*/*.parquet,hf://datasets/jasperai/monet/v1.2.0/laion/*/*.parquet,hf://datasets/jasperai/monet/v1.2.0/coyo/*/*.parquet,hf://datasets/jasperai/monet/v1.2.0/commoncatalog-cc-by/*/*.parquet" \
  --monet-weights "0.15,0.10,0.15,0.30,0.15,0.15" \
  --monet-caption-probs "0,0,0,0,1;.25,.25,.25,.25,0;.25,.25,.25,.25,0;.25,.25,.25,.25,0;.25,.25,.25,.25,0;.25,.25,.25,.25,0" \
  --text-encoder-id Qwen/Qwen3.5-4B --text-encoder-layers "9,18,27" \
  --text-encoder-max-length 320 --text-encoder-instruction \
  --latent-channels 32 --latent-size 16 --extra-input-channels 33 \
  --hidden-dim 1152 --depth 28 --ddt-decoder-depth 6 --num-heads 18 --mlp-ratio 2.5 \
  --mixer-type fla_gdn2 --fla-mode chunk --fla-bidirectional --use-short-conv --conv-size 4 \
  --image-attn-every 0 --image-attn-backend flash2 --image-rope \
  --cross-attn-backend sdpa --image-attn-qk-rmsnorm --cross-attn-qk-rmsnorm \
  --joint-attn-backend sdpa --ddt-norm-impl native \
  --shared-adaln --fourier-timestep --hilbert-order --parallel-block \
  --dual-stream-depth 4 --multimodal-coord-ids --joint-refresh-attn \
  --refresh-layers "7,11,15,21" --refresh-no-mixer --block-attnres \
  --sandwich-rmsnorm --rope-ntk-factor 1.0 --no-abs-pos-embed \
  --y-norm --y-norm-scale-factor 0.01 --patch-size 1 \
  --repa-dim 1536 --repa-layer 7 --repa-weight 0 \
  --batch-size 64 --accumulate 2 --num-workers 16 --prefetch-factor 4 \
  --persistent-workers --pin-memory --wds-shuffle-buffer 20000 \
  --parquet-read-batch-size 64 --parquet-range-mib 128 \
  --optimizer cmuon --lr 1e-4 --weight-decay 0 --beta1 0.9 --beta2 0.95 \
  --grad-clip 1 --lr-scheduler linear-warmup-constant --warmup-steps 5000 \
  --cond-dropout 0.1 --t-sampler logit-normal --flow-shift 1.5 --min-snr-gamma 0 \
  --t-sampler-final uniform --t-sampler-final-step 300000 \
  --ema-decay 0.9999 --dtype bf16 --max-steps 300000 --seed 42 \
  --compile --compile-text-encoder \
  --checkpoint-dir checkpoints/v8-512 --save-every-steps 5000 \
  --sample-every-steps 10000 --sample-prompts-file eval_prompts.txt --sample-weights ema \
  --sample-sampler stork4 --sample-steps 20 --sample-stork-derivative-order 1 \
  --sample-stork-substeps 9 --sample-flow-shift 5 --sample-cfg-scale 3 \
  --sample-cfg-rescale 0.5 --sample-cfg-interval-low 0.1 --sample-cfg-interval-high 0.9 \
  --hf-token true
```

## Checkpoints

To start a later phase, use `--resume PATH --resume-weights-only`. This loads the
checkpoint's model state; optimizer, scheduler, EMA shadow, and step count start
fresh. To seed from EMA, first promote the source checkpoint's `ema` tensors into
its `state_dict` under matching `model.*` keys. Keep the same `--compile` setting
as the checkpoint so the Lightning state keys match.

To continue an interrupted phase, use `--resume PATH` without
`--resume-weights-only`. This restores training state. Use a checkpoint from the
same phase.

`--checkpoint-dir` is the local output directory. The optional
`--checkpoint-upload-remote` enables asynchronous checkpoint copying and can
point to a mounted Drive directory.

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
