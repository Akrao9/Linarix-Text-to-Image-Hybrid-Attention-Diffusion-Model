"""STORK flow-matching scheduler loader — self-contained for HuggingFace distribution.

unshift_sigma inlined from boomer/sana_flow.py. No boomer package import needed.
Requires: torch. STORK repo must be pip-installed or available on sys.path.
  pip install git+https://github.com/Algomancer/STORK.git
or clone alongside Boomer:
  git clone https://github.com/Algomancer/STORK.git
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import torch


# ── inlined from boomer/sana_flow.py ──────────────────────────────────────────
def unshift_sigma(sigma: torch.Tensor, shift: float = 1.0) -> torch.Tensor:
    """Invert Sana/FlowMatch sigma shift."""
    if shift <= 0.0:
        raise ValueError(f"flow shift must be positive, got {shift}")
    if shift == 1.0:
        return sigma
    return sigma / (shift - (shift - 1.0) * sigma).clamp_min(1e-12)


# ── STORK loader ───────────────────────────────────────────────────────────────
def _stork_candidates() -> list[Path]:
    """Common locations where STORKScheduler.py might live."""
    here = Path(__file__).resolve().parent
    return [
        here,                    # bundled alongside this file (HF snapshot dir)
        here.parent / "STORK",
        here / "STORK",
        Path("/content/STORK"),
        Path("/tmp/STORK"),
    ]


def _find_stork_path() -> Path:
    # 1. Already importable?
    try:
        import STORKScheduler  # noqa: F401
        return Path(STORKScheduler.__file__)
    except ImportError:
        pass

    # 2. Search known candidate directories
    for candidate_dir in _stork_candidates():
        p = candidate_dir / "STORKScheduler.py"
        if p.is_file():
            if str(candidate_dir) not in sys.path:
                sys.path.insert(0, str(candidate_dir))
            return p

    # 3. Try pip-installed stork package
    try:
        import stork
        p = Path(stork.__file__).parent / "STORKScheduler.py"
        if p.is_file():
            return p
    except ImportError:
        pass

    searched = "\n  ".join(str(d / "STORKScheduler.py") for d in _stork_candidates())
    raise FileNotFoundError(
        "Could not find STORKScheduler.py. Options:\n"
        "  pip install git+https://github.com/ZT220501/STORK.git\n"
        "or clone it next to Boomer:\n"
        "  git clone https://github.com/ZT220501/STORK.git\n"
        f"Searched:\n  {searched}"
    )


def make_stork_scheduler(
    *,
    steps: int,
    device: str | torch.device,
    flow_shift: float,
    solver_order: int,
    derivative_order: int,
    substeps: int,
    start_sigma: float | None = None,
) -> Any:
    stork_path = _find_stork_path()
    spec = importlib.util.spec_from_file_location("STORKScheduler", stork_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load STORK scheduler from {stork_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    STORKScheduler = module.STORKScheduler

    scheduler = STORKScheduler(
        shift=flow_shift,
        solver_order=solver_order,
        prediction_type="flow_prediction",
        derivative_order=derivative_order,
        s=substeps,
    )
    if start_sigma is not None:
        start = torch.tensor(float(start_sigma), dtype=torch.float32)
        base_start = unshift_sigma(start, flow_shift)
        base_sigmas = torch.linspace(float(base_start.item()), 0.0, steps + 1, dtype=torch.float32)[:-1].tolist()
    else:
        base_sigmas = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float32)[:-1].tolist()

    scheduler.set_timesteps(num_inference_steps=steps, device=device, sigmas=base_sigmas)
    scheduler.dt_list = scheduler.dt_list.to(device=device)
    scheduler.sigmas = scheduler.sigmas.to(device=device)
    return scheduler
