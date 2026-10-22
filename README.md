# Boomer

Hybrid linear-attention diffusion for text-to-image generation.

Boomer V8 combines a bidirectional GatedDeltaNet-2 condition encoder with a
full-attention DDT velocity decoder. The denoiser has approximately 890M parameters;
text conditioning comes from frozen Qwen3.5-4B features and images use DC-AE latents.

[Model weights](https://huggingface.co/Akrao9/Linarix-v2) ·
[Demo](https://huggingface.co/spaces/akrao9/Linarix-v2-T2I) ·
[Training](docs/training.md)

## Setup

Use Linux with an NVIDIA GPU and a CUDA-compatible PyTorch installation. Install
the project and the FLA revision used by V8:

```bash
python -m pip install -e .
python -m pip install --no-deps "git+https://github.com/fla-org/flash-linear-attention.git@3c4c54ae7397d37130d7101edd0f4eb596af896d"
python -m pip install --no-build-isolation causal-conv1d
```

CUDA extensions must match your Python, PyTorch, and CUDA versions. See the
[training guide](docs/training.md) for the training environment and optional dependencies.

## Generate

Authenticate with `hf auth login` if the model requires access approval.

```python
import torch
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    "Akrao9/Linarix-v2",
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

The export supplies the resolution, text readout, and sampling defaults. The
published Linarix-v2 checkpoint generates at 1024px; 2048px finishing is experimental.

## Train

The [training guide](docs/training.md) covers the 512px MONET, 1024px MONET,
1024px FineT2I, and experimental 2048px phases.

## Code

- [`boomer/`](boomer/) — models, data loaders, flow matching, and sampling.
- [`train.py`](train.py) — Lightning training entry point.
- [`tools/`](tools/) — caching, sampling, evaluation, and Hugging Face export.
- [`tests/`](tests/) — regression coverage.
- [`Space/`](Space/) — Hugging Face demo.

Architecture details are in [docs/architecture.md](docs/architecture.md).
Earlier dataset cards and run notes live in [docs/](docs/).

## Development

```bash
python -m pip install -e '.[train,dev]'
python -m pytest tests/ -q
python tools/gen_modeling_file.py --check
```

Edit the model under `boomer/models/`; regenerate `modeling_boomer_fla.py` with
`python tools/gen_modeling_file.py` after model changes.

## License

[MIT](LICENSE) for the project's original code. Model weights and
[third-party components](docs/third-party.md) have their own terms.
