"""Load and configure the external STORK flow-matching scheduler."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import torch

from boomer.sana_flow import unshift_sigma


def stork_repo_candidates() -> list[Path]:
    boomer_root = Path(__file__).resolve().parents[1]
    return [
        boomer_root.parent / "STORK",  # ../STORK sibling of the Boomer repo
        Path("/content/STORK"),
        Path("/content/Boomer").resolve().parent / "STORK",
    ]


def maybe_add_sibling_stork_repo() -> None:
    for path in stork_repo_candidates():
        if (path / "STORKScheduler.py").is_file() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


def _stork_scheduler_path() -> Path:
    maybe_add_sibling_stork_repo()
    for path in stork_repo_candidates():
        candidate = path / "STORKScheduler.py"
        if candidate.is_file():
            return candidate
    searched = ", ".join(str(path / "STORKScheduler.py") for path in stork_repo_candidates())
    raise FileNotFoundError(f"Could not find STORKScheduler.py. Checked: {searched}")


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
    stork_path = _stork_scheduler_path()
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
