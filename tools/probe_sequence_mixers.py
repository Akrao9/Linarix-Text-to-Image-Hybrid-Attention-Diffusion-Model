"""Probe optional sequence mixer libraries for Boomer.

Run this in Colab before choosing the Mamba/FLA backend for the main model.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch


def maybe_add_repo(path: str | None, *, marker: str, label: str) -> None:
    if not path:
        return
    repo_path = Path(path).expanduser().resolve()
    if not repo_path.exists():
        print(f"{label}: missing {repo_path}")
        return
    if not (repo_path / marker).is_dir():
        print(f"{label}: {repo_path} does not contain {marker}/")
        return
    sys.path.insert(0, str(repo_path))
    print(f"{label}: added {repo_path}")


def try_import(name: str) -> Any | None:
    try:
        module = importlib.import_module(name)
    except Exception as exc:
        print(f"{name}: missing ({type(exc).__name__}: {exc})")
        return None
    version = getattr(module, "__version__", "")
    print(f"{name}: FOUND {version}".rstrip())
    return module


def try_class(module_name: str, class_name: str) -> type | None:
    module = try_import(module_name)
    if module is None:
        return None
    cls = getattr(module, class_name, None)
    if cls is None:
        print(f"{module_name}.{class_name}: missing")
        return None
    print(f"{module_name}.{class_name}: FOUND")
    return cls


def instantiate_and_forward(
    label: str,
    factory: Callable[[], torch.nn.Module],
    *,
    device: str,
    dtype: torch.dtype,
) -> None:
    try:
        module = factory().to(device=device, dtype=dtype).eval()
        x = torch.randn(2, 16, 64, device=device, dtype=dtype)
        with torch.inference_mode():
            y = module(x)
        if isinstance(y, tuple):
            y = y[0]
        print(f"{label}: forward ok input={tuple(x.shape)} output={tuple(y.shape)} dtype={y.dtype} device={y.device}")
    except Exception as exc:
        print(f"{label}: forward failed ({type(exc).__name__}: {exc})")


def instantiate_fla_and_forward(
    module_name: str,
    class_name: str,
    *,
    device: str,
    dtype: torch.dtype,
) -> None:
    cls = try_class(module_name, class_name)
    if cls is None:
        return
    try:
        signature = inspect.signature(cls)
        common_kwargs = {
            "hidden_size": 64,
            "hidden_dim": 64,
            "d_model": 64,
            "num_heads": 4,
            "num_attention_heads": 4,
            "head_dim": 16,
            "mode": "chunk",
            "attn_mode": "chunk",
            "chunk_size": 64,
            "use_short_conv": True,
            "conv_size": 4,
            "use_gate": True,
        }
        kwargs = {name: value for name, value in common_kwargs.items() if name in signature.parameters}
        instantiate_and_forward(
            f"{module_name}.{class_name}",
            lambda: cls(**kwargs),
            device=device,
            dtype=dtype,
        )
        print(f"{module_name}.{class_name}: kwargs={kwargs}")
    except Exception as exc:
        print(f"{module_name}.{class_name}: probe failed ({type(exc).__name__}: {exc})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe optional Mamba/FLA sequence mixer imports.")
    parser.add_argument(
        "--mamba-repo",
        default=str(Path(__file__).resolve().parents[2] / "mamba"),
        help="Optional local state-spaces/mamba checkout. Default: sibling ../mamba.",
    )
    parser.add_argument(
        "--flame-repo",
        default=str(Path(__file__).resolve().parents[2] / "flash-linear-attention"),
        help="Optional local fla-org/flash-linear-attention checkout. Default: sibling ../flash-linear-attention.",
    )
    args = parser.parse_args()
    maybe_add_repo(args.mamba_repo, marker="mamba_ssm", label="local_mamba_repo")
    maybe_add_repo(args.flame_repo, marker="fla", label="local_fla_repo")

    print(f"torch={torch.__version__}")
    print(f"cuda_available={torch.cuda.is_available()}")
    print(f"torch_cuda={getattr(torch.version, 'cuda', None)}")
    if torch.cuda.is_available():
        print(f"cuda_device={torch.cuda.get_device_name(0)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    print(f"probe_device={device} probe_dtype={dtype}")

    print("\n== raw imports ==")
    for name in [
        "mamba_ssm",
        "mamba_ssm.modules.mamba3",
        "mamba3",
        "mamba",
        "causal_conv1d",
        "fla",
        "fla.layers",
        "flash_attn",
    ]:
        try_import(name)

    print("\n== mamba_ssm classes ==")
    mamba3_cls = try_class("mamba_ssm.modules.mamba3", "Mamba3")
    if mamba3_cls is not None:
        instantiate_and_forward(
            "mamba_ssm.modules.mamba3.Mamba3",
            lambda: mamba3_cls(
                d_model=64,
                d_state=128,
                headdim=64,
                is_mimo=False,
                chunk_size=64,
                dtype=dtype,
            ),
            device=device,
            dtype=dtype,
        )

    mamba_cls = try_class("mamba_ssm.modules.mamba_simple", "Mamba")
    if mamba_cls is not None:
        instantiate_and_forward(
            "mamba_ssm.modules.mamba_simple.Mamba",
            lambda: mamba_cls(d_model=64, d_state=16, d_conv=4, expand=2),
            device=device,
            dtype=dtype,
        )

    for module_name, class_name in [
        ("mamba_ssm.modules.mamba2", "Mamba2"),
        ("mamba_ssm.modules.mamba2", "Mamba2Simple"),
    ]:
        cls = try_class(module_name, class_name)
        if cls is not None:
            instantiate_and_forward(
                f"{module_name}.{class_name}",
                lambda cls=cls: cls(d_model=64, d_state=64, d_conv=4, expand=2),
                device=device,
                dtype=dtype,
            )

    print("\n== fla classes ==")
    for module_name, class_name in [
        ("fla.layers", "GatedDeltaNet"),
        ("fla.layers", "LinearAttention"),
        ("fla.layers", "GatedLinearAttention"),
        ("fla.layers", "DeltaNet"),
        ("fla.layers", "HGRN2"),
        ("fla.layers", "Mamba2"),
        ("fla.layers", "RetNet"),
        ("fla.layers", "BasedLinearAttention"),
    ]:
        instantiate_fla_and_forward(module_name, class_name, device=device, dtype=dtype)


if __name__ == "__main__":
    main()
