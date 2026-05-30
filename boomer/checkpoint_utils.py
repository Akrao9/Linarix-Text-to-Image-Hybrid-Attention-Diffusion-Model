"""Shared checkpoint load/save helpers for Boomer training tools."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from boomer.latent_norm import LatentStat
from boomer.models.factory import Config, config_from_dict


def load_torch_file(path: str | Path, map_location: str = "cpu") -> dict[str, Any]:
    """Load a checkpoint or cache record, preferring safe tensor-only loads."""
    path = Path(path)
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        pass
    except Exception:
        pass
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def flatten_state_dict(state_dict: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    return {f"{prefix}.{name}": tensor for name, tensor in state_dict.items()}


def unflatten_state_dict(
    tensors: dict[str, torch.Tensor],
    prefix: str,
) -> dict[str, torch.Tensor]:
    prefix_dot = f"{prefix}."
    return {
        name[len(prefix_dot) :]: tensor
        for name, tensor in tensors.items()
        if name.startswith(prefix_dot)
    }


def collect_train_args(
    *,
    ema_decay: float = 0.0,
    ema_update_every: int = 1,
    cond_dropout: float = 0.0,
    latent_mean: LatentStat = 0.0,
    latent_std: LatentStat = 1.0,
    t_sampler: str = "uniform",
    t_bias: float = 2.0,
    flow_shift: float = 1.0,
    logit_mean: float = 0.0,
    logit_std: float = 1.0,
    plateau_low: float = 0.1,
    plateau_high: float = 0.9,
    y_norm: bool = False,
    y_norm_scale_factor: float = 0.01,
    grad_clip: float = 1.0,
    timestep_weight: bool = False,
    min_snr_gamma: float = 0.0,
    lr_scheduler: str = "constant",
    warmup_steps: int = 0,
    min_lr_ratio: float = 0.1,
) -> dict[str, Any]:
    return {
        "ema_decay": ema_decay,
        "ema_update_every": ema_update_every,
        "cond_dropout": cond_dropout,
        "latent_mean": latent_mean,
        "latent_std": latent_std,
        "t_sampler": t_sampler,
        "t_bias": t_bias,
        "flow_shift": flow_shift,
        "logit_mean": logit_mean,
        "logit_std": logit_std,
        "plateau_low": plateau_low,
        "plateau_high": plateau_high,
        "y_norm": y_norm,
        "y_norm_scale_factor": y_norm_scale_factor,
        "grad_clip": grad_clip,
        "timestep_weight": timestep_weight,
        "min_snr_gamma": min_snr_gamma,
        "lr_scheduler": lr_scheduler,
        "warmup_steps": warmup_steps,
        "min_lr_ratio": min_lr_ratio,
    }


def inference_checkpoint_path(path: str | Path) -> Path:
    return Path(path).with_suffix(".safetensors")


def save_resume_checkpoint(
    path: str | Path,
    model: nn.Module,
    config: Config,
    optimizer: torch.optim.Optimizer,
    step: int,
    *,
    ema_model: nn.Module | None = None,
    train_args: dict[str, Any] | None = None,
) -> Path:
    """Save a full PyTorch checkpoint for exact training resume."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint: dict[str, Any] = {
        "model": model.state_dict(),
        "config": config.__dict__,
        "optimizer": optimizer.state_dict(),
        "steps": step,
        "train_args": train_args or {},
    }
    if ema_model is not None:
        checkpoint["ema_model"] = ema_model.state_dict()
    torch.save(checkpoint, path)
    return path


def save_inference_checkpoint(
    path: str | Path,
    model: nn.Module,
    config: Config,
    step: int,
    *,
    ema_model: nn.Module | None = None,
    train_args: dict[str, Any] | None = None,
) -> Path:
    """Save model/EMA weights in safetensors for faster inference-oriented loads."""
    from safetensors.torch import save_file

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tensors = flatten_state_dict(model.state_dict(), "model")
    if ema_model is not None:
        tensors.update(flatten_state_dict(ema_model.state_dict(), "ema_model"))
    metadata = {
        "steps": str(step),
        "config": json.dumps(config.__dict__),
        "train_args": json.dumps(train_args or {}),
    }
    save_file(tensors, path, metadata=metadata)
    return path


def save_training_checkpoints(
    resume_path: str | Path,
    model: nn.Module,
    config: Config,
    optimizer: torch.optim.Optimizer,
    step: int,
    *,
    ema_model: nn.Module | None = None,
    train_args: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Write a full .pt resume checkpoint and a model-only .safetensors inference checkpoint."""
    resume_path = save_resume_checkpoint(
        resume_path,
        model,
        config,
        optimizer,
        step,
        ema_model=ema_model,
        train_args=train_args,
    )
    inference_path = save_inference_checkpoint(
        inference_checkpoint_path(resume_path),
        model,
        config,
        step,
        ema_model=ema_model,
        train_args=train_args,
    )
    return resume_path, inference_path


def capture_training_checkpoint_payload(
    model: nn.Module,
    config: Config,
    optimizer: torch.optim.Optimizer,
    step: int,
    *,
    ema_model: nn.Module | None = None,
    train_args: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Capture checkpoint tensors on the training thread for background disk writes."""
    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "config": config.__dict__,
        "optimizer": optimizer.state_dict(),
        "steps": step,
        "train_args": train_args or {},
    }
    if ema_model is not None:
        payload["ema_model"] = ema_model.state_dict()
    return payload


def write_training_checkpoints_from_payload(
    resume_path: str | Path,
    payload: dict[str, Any],
    config: Config,
    step: int,
    *,
    train_args: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Write .pt and .safetensors checkpoints from a captured payload."""
    resume_path = Path(resume_path)
    resume_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, resume_path)
    inference_path = inference_checkpoint_path(resume_path)
    tensors = flatten_state_dict(payload["model"], "model")
    ema_state = payload.get("ema_model")
    if ema_state is not None:
        tensors.update(flatten_state_dict(ema_state, "ema_model"))
    from safetensors.torch import save_file

    metadata = {
        "steps": str(step),
        "config": json.dumps(config.__dict__),
        "train_args": json.dumps(train_args or {}),
    }
    save_file(tensors, inference_path, metadata=metadata)
    return resume_path, inference_path


@dataclass
class _UploadJob:
    step: int
    paths: tuple[Path, Path]
    log_path: Path
    process: subprocess.Popen[Any] | None = None
    thread: threading.Thread | None = None
    copy_error: str | None = None


def _is_rclone_remote(target: str) -> bool:
    """True for rclone remotes like gdrive:path (not absolute local paths)."""
    return re.match(r"^[A-Za-z0-9][A-Za-z0-9._-]*:", target) is not None


class CheckpointOffloader:
    """Save checkpoints locally, then upload to cloud storage in the background via rclone."""

    def __init__(
        self,
        *,
        upload_remote: str | None = None,
        local_retain: int = 2,
        rclone_bin: str = "rclone",
        drive_chunk_size: str = "128M",
        checkpoint_dir: str | Path | None = None,
    ) -> None:
        self.upload_remote = upload_remote.rstrip("/") if upload_remote else None
        self.local_retain = max(0, local_retain)
        self.rclone_bin = rclone_bin
        self.drive_chunk_size = drive_chunk_size
        self._jobs: list[_UploadJob] = []
        self._saved: dict[int, tuple[Path, Path, bool]] = {}
        self._save_thread: threading.Thread | None = None
        if checkpoint_dir is not None:
            self._register_existing_checkpoints(Path(checkpoint_dir))

    def save(
        self,
        resume_path: str | Path,
        model: nn.Module,
        config: Config,
        optimizer: torch.optim.Optimizer,
        step: int,
        *,
        ema_model: nn.Module | None = None,
        train_args: dict[str, Any] | None = None,
    ) -> tuple[Path, Path]:
        self._poll_jobs()
        running_steps = [
            job.step
            for job in self._jobs
            if not self._job_is_finished(job)
        ]
        if running_steps:
            print(
                f"[step {step}] warning: previous checkpoint upload(s) still running "
                f"(steps={running_steps}); saving locally anyway"
            )
        self._join_save_thread(step=step)

        resume_path = Path(resume_path)
        inference_path = inference_checkpoint_path(resume_path)
        payload = capture_training_checkpoint_payload(
            model,
            config,
            optimizer,
            step,
            ema_model=ema_model,
            train_args=train_args,
        )
        self._saved[step] = (resume_path, inference_path, False)

        def _write_and_upload() -> None:
            write_training_checkpoints_from_payload(
                resume_path,
                payload,
                config,
                step,
                train_args=train_args,
            )
            if self.upload_remote is not None:
                self._start_upload(step, resume_path, inference_path)

        self._save_thread = threading.Thread(
            target=_write_and_upload,
            name=f"checkpoint-save-{step}",
            daemon=False,
        )
        self._save_thread.start()
        return resume_path, inference_path

    def _join_save_thread(self, *, step: int | None = None) -> None:
        if self._save_thread is None:
            return
        if self._save_thread.is_alive():
            if step is not None:
                print(
                    f"[step {step}] warning: previous checkpoint save still running; "
                    "waiting before writing the next checkpoint"
                )
            self._save_thread.join()
        self._save_thread = None

    def finish(self, *, wait_timeout_s: float | None = None) -> None:
        """Wait for the final background save/upload and report failures."""
        self._join_save_thread()
        self.wait_for_uploads(timeout=wait_timeout_s)
        self._poll_jobs()

    def wait_for_uploads(self, timeout: float | None = None) -> None:
        deadline = None if timeout is None else time.monotonic() + timeout
        while any(not self._job_is_finished(job) for job in self._jobs):
            if deadline is not None and time.monotonic() >= deadline:
                print("warning: checkpoint upload wait timed out before training exit")
                return
            self._poll_jobs()
            if any(not self._job_is_finished(job) for job in self._jobs):
                time.sleep(0.25)

    @staticmethod
    def _job_is_finished(job: _UploadJob) -> bool:
        if job.process is not None:
            return job.process.poll() is not None
        if job.thread is not None:
            return not job.thread.is_alive()
        return True

    def _start_upload(self, step: int, resume_path: Path, inference_path: Path) -> None:
        if self.upload_remote is None:
            return
        log_dir = resume_path.parent / ".upload_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"step_{step:06d}.log"

        if _is_rclone_remote(self.upload_remote):
            self._start_rclone_upload(step, resume_path, inference_path, log_path)
        else:
            self._start_local_copy_upload(step, resume_path, inference_path, log_path)

    def _start_rclone_upload(
        self,
        step: int,
        resume_path: Path,
        inference_path: Path,
        log_path: Path,
    ) -> None:
        cmd = [
            self.rclone_bin,
            "copy",
            str(resume_path.parent),
            self.upload_remote,
            "--include",
            resume_path.name,
            "--include",
            inference_path.name,
            "--drive-chunk-size",
            self.drive_chunk_size,
            "-v",
        ]
        with log_path.open("w", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )
        self._jobs.append(
            _UploadJob(
                step=step,
                paths=(resume_path, inference_path),
                process=process,
                log_path=log_path,
            )
        )
        print(f"[step {step}] queued background rclone upload -> {self.upload_remote}")

    def _start_local_copy_upload(
        self,
        step: int,
        resume_path: Path,
        inference_path: Path,
        log_path: Path,
    ) -> None:
        dest_dir = Path(self.upload_remote)
        job = _UploadJob(
            step=step,
            paths=(resume_path, inference_path),
            log_path=log_path,
        )

        def _copy() -> None:
            try:
                dest_dir.mkdir(parents=True, exist_ok=True)
                for src in (resume_path, inference_path):
                    shutil.copy2(src, dest_dir / src.name)
            except OSError as exc:
                job.copy_error = str(exc)

        thread = threading.Thread(
            target=_copy,
            name=f"checkpoint-copy-{step}",
            daemon=False,
        )
        job.thread = thread
        self._jobs.append(job)
        thread.start()
        print(f"[step {step}] queued background copy -> {dest_dir}")

    def _poll_jobs(self) -> None:
        still_running: list[_UploadJob] = []
        for job in self._jobs:
            if not self._job_is_finished(job):
                still_running.append(job)
                continue
            self._finish_job(job)
        self._jobs = still_running

    def _finish_job(self, job: _UploadJob) -> None:
        if job.process is not None:
            return_code = job.process.returncode
            if return_code != 0:
                err_tail = ""
                if job.log_path.exists():
                    err_tail = job.log_path.read_text(encoding="utf-8")[-2000:]
                print(
                    f"warning: checkpoint upload failed for step={job.step} "
                    f"(exit {return_code}); local files kept at {job.paths[0].parent}"
                )
                if err_tail:
                    print(err_tail)
                return
        elif job.copy_error is not None:
            print(
                f"warning: checkpoint copy failed for step={job.step} "
                f"({job.copy_error}); local files kept at {job.paths[0].parent}"
            )
            return

        resume_path, inference_path = job.paths
        if job.step in self._saved:
            self._saved[job.step] = (resume_path, inference_path, True)
        print(f"[step {job.step}] upload complete -> {self.upload_remote}")
        self._prune_local_checkpoints()

    def _register_existing_checkpoints(self, checkpoint_dir: Path) -> None:
        """Scan checkpoint_dir for pre-existing .pt files and prune down to local_retain.

        Called once at init so that orphaned checkpoints from previous or interrupted
        runs are visible to the offloader and get pruned immediately, instead of
        accumulating indefinitely alongside checkpoints from the current run.
        """
        import re

        found: list[tuple[int, Path, Path]] = []
        for pt_path in checkpoint_dir.glob("latent_dit_step_*.pt"):
            m = re.search(r"latent_dit_step_(\d+)\.pt$", pt_path.name)
            if m is None:
                continue
            step = int(m.group(1))
            sf_path = pt_path.with_suffix(".safetensors")
            found.append((step, pt_path, sf_path))

        if not found:
            return

        for step, pt_path, sf_path in found:
            if step not in self._saved:
                # Mark as uploaded=True so _prune_local_checkpoints considers them.
                self._saved[step] = (pt_path, sf_path, True)

        existing = sorted(s for s, *_ in found)
        print(
            f"[checkpoint-offloader] found {len(found)} existing checkpoint(s) in "
            f"{checkpoint_dir}: steps={existing}; pruning to local_retain={self.local_retain}",
            flush=True,
        )
        self._prune_local_checkpoints()

    def _prune_local_checkpoints(self) -> None:
        uploaded_steps = sorted(
            (step for step, (_, _, uploaded) in self._saved.items() if uploaded),
            reverse=True,
        )
        keep_steps = uploaded_steps[: self.local_retain]
        for step in uploaded_steps[self.local_retain :]:
            resume_path, inference_path, _ = self._saved[step]
            for path in (resume_path, inference_path):
                if path.exists():
                    path.unlink()
            del self._saved[step]
            print(f"[step {step}] pruned local checkpoint after upload (keeping {sorted(keep_steps, reverse=True)})")


def load_inference_checkpoint(path: str | Path, map_location: str = "cpu") -> dict[str, Any]:
    from safetensors import safe_open

    path = Path(path)
    tensors: dict[str, torch.Tensor] = {}
    metadata: dict[str, str] = {}
    with safe_open(path, framework="pt", device=map_location) as opened:
        metadata = dict(opened.metadata() or {})
        for key in opened.keys():
            tensors[key] = opened.get_tensor(key)

    model_state = unflatten_state_dict(tensors, "model")
    if not model_state:
        raise ValueError(f"{path} does not contain model.* tensors")

    checkpoint: dict[str, Any] = {
        "model": model_state,
        "config": json.loads(metadata.get("config", "{}")),
        "train_args": json.loads(metadata.get("train_args", "{}")),
        "steps": int(metadata.get("steps", "0")),
    }
    ema_state = unflatten_state_dict(tensors, "ema_model")
    if ema_state:
        checkpoint["ema_model"] = ema_state
    return checkpoint


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> dict[str, Any]:
    path = Path(path)
    if path.suffix == ".safetensors":
        return load_inference_checkpoint(path, map_location=map_location)
    return load_torch_file(path, map_location=map_location)


def config_from_checkpoint(checkpoint: dict[str, Any]) -> Config:
    raw = checkpoint.get("config") or {}
    return config_from_dict(raw)


def select_model_state(checkpoint: dict[str, Any], weights: str) -> tuple[dict[str, torch.Tensor], str]:
    if weights == "ema":
        if "ema_model" not in checkpoint:
            raise ValueError("Checkpoint does not contain EMA weights.")
        return checkpoint["ema_model"], "ema"
    if weights == "auto" and "ema_model" in checkpoint:
        return checkpoint["ema_model"], "ema"
    return checkpoint["model"], "model"


def load_compatible_state_dict(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
) -> list[str]:
    """Load matching checkpoint tensors and skip renamed or reshaped tensors."""
    model_state = model.state_dict()
    compatible_state: dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    for name, value in state_dict.items():
        if name not in model_state:
            skipped.append(name)
            continue
        if tuple(model_state[name].shape) != tuple(value.shape):
            skipped.append(name)
            continue
        compatible_state[name] = value

    model.load_state_dict(compatible_state, strict=False)
    return skipped
