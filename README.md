# Linarix

Hybrid linear-attention diffusion for text-to-image generation.

Linarix combines a bidirectional GatedDeltaNet-2 condition encoder with a
full-attention DDT velocity decoder. The denoiser has approximately 890M parameters;
text conditioning comes from frozen Qwen3.5-4B features. Image decoding uses
[SANA 1.1 DC-AE f32c32](https://huggingface.co/mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers).

[Model weights](https://huggingface.co/akrao9/Linarix-v2) ·
[Demo](https://huggingface.co/spaces/akrao9/Linarix-v2-T2I) ·
[Training](docs/training.md)


<img width="2048" height="1536" alt="showcase" src="https://github.com/user-attachments/assets/13021b6c-1bec-4c4f-a8f1-d83375a16849" />

<!-- Paste the showcase image from the Hugging Face model card here. -->

## Setup

For inference, use Linux with an NVIDIA GPU and a CUDA-compatible PyTorch
installation:

```bash
python -m pip install -U "diffusers>=0.38.0" transformers accelerate safetensors torchvision scipy
python -m pip install "flash-linear-attention @ git+https://github.com/fla-org/flash-linear-attention.git@3c4c54ae7397d37130d7101edd0f4eb596af896d"
```

FLA is required for the linear mixers and Block AttnRes. Image attention uses
PyTorch SDPA. The Hugging Face model bundles STORK and its coefficient files;
inference needs no separate `flash-attn` or STORK installation. Training setup is
documented in the [training guide](docs/training.md).

## Generate

Authenticate with `hf auth login` if the model requires access approval.

```python
import torch
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    "akrao9/Linarix-v2",
    custom_pipeline="pipeline_boomer",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
).to("cuda")

image = pipe(
    "a lighthouse on a rocky cliff above crashing waves at golden hour",
    seed=42,
)[0]
image.save("lighthouse.png")
```

The published checkpoint uses EMA weights and generates at 1024px. Defaults are
STORK-4, 20 steps, `derivative_order=1`, `substeps=14`, `cfg_scale=4.0`,
`cfg_rescale=0.5`, guidance interval `[0.1, 0.9]`, and `flow_shift=5.0`.

Qwen and DC-AE download from their upstream repositories on first use. Pass a list
of prompts for batched inference. VAE slicing is enabled by default. For throughput
benchmarks, pass `offload_text_encoder=False` to keep the text encoder on the GPU.

## Train

The MONET training phases streamed
[`jasperai/monet`](https://huggingface.co/datasets/jasperai/monet), version `v1.2.0`,
directly from Hugging Face during training. The 512px phase read MONET's precomputed
DC-AE latents from Parquet shards. The 1024px MONET phase streamed image TAR shards
and encoded the images with DC-AE on the GPU. Captions were encoded with frozen
Qwen during training.

At 512px, REPA weight was `0.25` for the first 100,000 steps. Training then resumed
with REPA weight `0`; the 1024px phases also used REPA weight `0`.

The 1024px fine-tuning phase used
[`ma-xu/fine-t2i`](https://huggingface.co/datasets/ma-xu/fine-t2i), also streamed
from Hugging Face.

Launch commands: [512px MONET](docs/training.md#512px-monet),
[1024px MONET](docs/training.md#1024px-monet), and
[1024px FineT2I](docs/training.md#1024px-finet2i).

## Code

- [Model and data code](boomer/) — models, data loaders, flow matching, and sampling.
- [`train.py`](train.py) — training entry point; components live in [`boomer/training/`](boomer/training/).
- [`tools/`](tools/) — caching, sampling, evaluation, and Hugging Face export.
- [`tests/`](tests/) — regression coverage.
- [`Space/`](Space/) — Hugging Face demo.

Architecture details are in [docs/architecture.md](docs/architecture.md).
Historical cache dataset cards and run notes live in [docs/](docs/).

## Development

```bash
python -m pip install -e '.[train,dev]'
python -m pytest tests/ -q
python tools/gen_modeling_file.py --check
```

Edit the [model source](boomer/models/); regenerate the
[export model](modeling_boomer_fla.py) with `python tools/gen_modeling_file.py`
after model changes.

## License

[MIT](LICENSE) for the project's original code. Model weights and
[third-party components](docs/third-party.md) have their own terms.
