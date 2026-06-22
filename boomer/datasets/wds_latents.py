"""WebDataset latent/text cache loader for Boomer training."""

from __future__ import annotations

import io
import json
import glob
import os
import random
import shlex
import shutil
import tarfile
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from boomer.datasets.latent_shards import LatentRecord, collate_latent_records
from boomer.sana_latent_cache import hf_token_value

_NUMPY_DTYPES: dict[str, np.dtype] = {
    "float16": np.dtype("float16"),
    "bfloat16": np.dtype("uint16"),
    "float32": np.dtype("float32"),
    "float8_e4m3fn": np.dtype("uint8"),
    "uint8": np.dtype("uint8"),
    "uint16": np.dtype("uint16"),
}

# Max seconds to wait on a single shard download before abandoning it and moving on.
# A stuck/slow HF link must not deadlock training: after this budget the rolling cache
# gives up on that shard, clears the wedged in-flight reference, and continues with the
# shards already in the local window (then tries a different remote shard).
ROLLING_WDS_DOWNLOAD_TIMEOUT_S = 300.0


def split_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def npy_to_tensor(value: bytes | np.ndarray) -> torch.Tensor:
    if isinstance(value, np.ndarray):
        array = value
    else:
        array = np.load(io.BytesIO(value), allow_pickle=False)
    return torch.from_numpy(array)


def decode_text(value: bytes | str) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def decode_json(value: bytes | str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return json.loads(decode_text(value))


@dataclass(frozen=True)
class WdsCacheManifest:
    latents_dtype: str = "float16"
    latents_shape: tuple[int, ...] | None = None
    text_dtype: str = "float16"
    text_shape: tuple[int, ...] | None = None
    text_mask_dtype: str = "uint8"
    raw_tensors: bool = False
    dino_dtype: str = "float16"

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> WdsCacheManifest:
        latents_shape = payload.get("latents_shape")
        text_shape = payload.get("text_shape")
        schema = payload.get("schema") or []
        raw_tensors = bool(payload.get("raw_tensors"))
        if not raw_tensors and schema:
            raw_tensors = any(name.endswith(".bin") for name in schema)
        return cls(
            latents_dtype=str(payload.get("latents_dtype", "float16")),
            latents_shape=tuple(latents_shape) if latents_shape else None,
            text_dtype=str(payload.get("text_dtype", "float16")),
            text_shape=tuple(text_shape) if text_shape else None,
            text_mask_dtype=str(payload.get("text_mask_dtype", "uint8")),
            raw_tensors=raw_tensors,
            dino_dtype=str(payload.get("dino_dtype", "float16")),
        )


def parse_manifest_text(text: str) -> WdsCacheManifest:
    return WdsCacheManifest.from_dict(json.loads(text))


def _manifest_paths_near_tar(tar_path: Path) -> list[Path]:
    parent = tar_path.parent
    candidates = sorted(parent.glob("manifest*.json"))
    default = parent / "manifest.json"
    if default not in candidates and default.is_file():
        candidates.insert(0, default)
    return candidates


def _hf_manifest_candidates(raw_path: str) -> list[str]:
    base = raw_path.split("*", 1)[0].rstrip("/")
    parent = str(Path(base).parent) if "/" in base else base
    names = ["manifest.json"]
    names.extend(f"manifest-{tag}.json" for tag in ("r00", "r01", "r02", "r03"))
    candidates = [f"{base.rsplit('/', 1)[0]}/manifest.json"] if "/" in base else []
    candidates.extend(f"{parent}/{name}" for name in names)
    candidates.append(f"{base}/manifest.json")
    # Preserve order while dropping duplicates.
    seen: set[str] = set()
    ordered: list[str] = []
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            ordered.append(candidate)
    return ordered


def load_wds_cache_manifest(
    paths: str | Iterable[str],
    *,
    hf_token: str | bool | None = None,
    urls: list[str] | None = None,
) -> WdsCacheManifest | None:
    if isinstance(paths, str):
        raw_paths = split_csv(paths)
    else:
        raw_paths = [str(path) for path in paths]

    resolved_urls = urls if urls is not None else resolve_wds_urls(paths, hf_token=hf_token)
    for url in resolved_urls:
        if url.startswith("pipe:"):
            continue
        for manifest_path in _manifest_paths_near_tar(Path(url)):
            return parse_manifest_text(manifest_path.read_text(encoding="utf-8"))

    hf_paths = [raw for raw in raw_paths if raw.startswith("hf://")]
    if hf_paths:
        from huggingface_hub import HfFileSystem

        fs = HfFileSystem(token=hf_token)
        for raw in hf_paths:
            for candidate in _hf_manifest_candidates(raw):
                try:
                    if fs.exists(candidate):
                        with fs.open(candidate, "r") as handle:
                            return parse_manifest_text(handle.read())
                except Exception:
                    continue
    return None


def split_urls_for_worker(urls: list[str], worker_id: int, num_workers: int) -> list[str]:
    if num_workers <= 1:
        return urls
    return urls[worker_id::num_workers]


@dataclass(frozen=True)
class HfWdsShardRef:
    repo_id: str
    path_in_repo: str

    @property
    def filename(self) -> str:
        return Path(self.path_in_repo).name


def list_hf_wds_shard_refs(raw_path: str, *, hf_token: str | bool | None = None) -> list[HfWdsShardRef]:
    if not raw_path.startswith("hf://"):
        raise ValueError(f"Expected an hf:// WebDataset glob, got: {raw_path}")
    from huggingface_hub import HfApi, HfFileSystem

    fs = HfFileSystem(token=hf_token)
    try:
        paths = sorted(fs.glob(raw_path))
    except FileNotFoundError:
        paths = []
    if not paths:
        prefix = "hf://datasets/"
        if raw_path.startswith(prefix):
            repo_and_path = raw_path[len(prefix) :]
            repo_id, _, pattern = repo_and_path.partition("/")
            base = pattern.split("*", 1)[0].rstrip("/")
            api = HfApi(token=hf_token if isinstance(hf_token, str) else None)
            print(
                f"Listing HF dataset files for {repo_id} (this can take several minutes) ...",
                flush=True,
            )
            repo_files = sorted(api.list_repo_files(repo_id=repo_id, repo_type="dataset"))
            paths = [
                f"hf://datasets/{repo_id}/{path}"
                for path in repo_files
                if path.startswith(base.rstrip("/") + "/") and path.endswith(".tar")
            ]
            print(f"Found {len(paths)} tar shards under {base}", flush=True)
    refs: list[HfWdsShardRef] = []
    for path in paths:
        resolved = fs.resolve_path(path)
        refs.append(HfWdsShardRef(repo_id=resolved.repo_id, path_in_repo=resolved.path_in_repo))
    if not refs:
        raise FileNotFoundError(f"No WebDataset shards found for: {raw_path}")
    return refs


def _normalize_shard_name(url: str) -> str:
    text = str(url).strip()
    marker = ".tar"
    if marker in text:
        stem = text.split(marker, 1)[0]
        return stem.rsplit("/", 1)[-1] + marker
    return Path(text).name


class RollingLocalWdsCache:
    """Keep a fixed-size local window while covering remote WDS shards by shuffled epochs."""

    def __init__(
        self,
        *,
        remote_paths: str,
        local_dir: str | Path,
        max_local_shards: int = 25,
        hf_token: str | bool | None = None,
        download_fn: Callable[[HfWdsShardRef, Path], Path] | None = None,
        seed: int | None = None,
    ) -> None:
        if max_local_shards < 1:
            raise ValueError("max_local_shards must be at least 1.")
        self.remote_paths = remote_paths
        self.local_dir = Path(local_dir)
        self.max_local_shards = max_local_shards
        self.hf_token = hf_token
        self.remote_shards = list_hf_wds_shard_refs(remote_paths, hf_token=hf_token)
        self._shard_index_by_name = {ref.filename: index for index, ref in enumerate(self.remote_shards)}
        self._download_fn = download_fn or self._default_download
        self._rng = random.Random(seed)
        self._remaining_remote_indices: deque[int] = deque()
        self._lock = threading.Lock()
        self._window: deque[Path] = deque()
        self._window_indices: set[int] = set()
        self._prefetch_thread: threading.Thread | None = None
        self._prefetch_target: Path | None = None
        self._prefetch_remote_index: int | None = None
        self._inflight_download_thread: threading.Thread | None = None
        self.local_dir.mkdir(parents=True, exist_ok=True)
        self._sync_manifest()
        self.fill_initial()

    def _sync_manifest(self) -> None:
        manifest = load_wds_cache_manifest(self.remote_paths, hf_token=self.hf_token)
        if manifest is None:
            return
        manifest_path = self.local_dir / "manifest.json"
        if not manifest_path.exists():
            manifest_path.write_text(
                json.dumps(
                    {
                        "format_version": 2,
                        "format": "boomer_wds_latent_text_cache",
                        "latents_dtype": manifest.latents_dtype,
                        "latents_shape": list(manifest.latents_shape or ()),
                        "text_dtype": manifest.text_dtype,
                        "text_shape": list(manifest.text_shape or ()),
                        "text_mask_dtype": manifest.text_mask_dtype,
                        "raw_tensors": manifest.raw_tensors,
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )

    def _local_path_for(self, shard: HfWdsShardRef) -> Path:
        return self.local_dir / shard.path_in_repo

    @staticmethod
    def _is_valid_shard(path: Path) -> bool:
        return path.is_file() and path.stat().st_size > 0 and tarfile.is_tarfile(path)

    def _wait_for_download_thread(
        self,
        thread: threading.Thread,
        *,
        label: str,
        timeout: float | None = None,
    ) -> None:
        """Block until ``thread`` finishes, or raise ``TimeoutError`` after ``timeout`` seconds.

        ``timeout=None`` waits indefinitely (legacy behaviour). A finite ``timeout`` lets a
        stuck or pathologically slow HF download be abandoned so that one bad shard cannot
        deadlock the (single-process) training loop. The abandoned download is a daemon
        thread and is simply left to die on its own.
        """
        poll = 60
        waited = 0
        while thread.is_alive():
            thread.join(timeout=poll)
            if thread.is_alive():
                waited += poll
                print(
                    f"[rolling-wds] {label} still downloading ({waited}s, one download at a time) ...",
                    flush=True,
                )
                if timeout is not None and waited >= timeout:
                    print(
                        f"[rolling-wds] {label} exceeded {int(timeout)}s — abandoning this shard "
                        "and continuing with the rest of the local window.",
                        flush=True,
                    )
                    raise TimeoutError(f"{label} exceeded {int(timeout)}s download budget")

    def _download_remote_index_sync(self, remote_index: int, *, label: str | None = None) -> Path:
        """Download one shard, waiting for any prior in-flight download to finish first."""
        with self._lock:
            prior = self._inflight_download_thread
        if prior is not None and prior.is_alive():
            try:
                self._wait_for_download_thread(
                    prior,
                    label=label or "prior shard",
                    timeout=ROLLING_WDS_DOWNLOAD_TIMEOUT_S,
                )
            except TimeoutError:
                # Prior download is wedged; stop waiting on it (it stays a daemon thread)
                # and drop the reference so it no longer blocks this or future downloads.
                with self._lock:
                    if self._inflight_download_thread is prior:
                        self._inflight_download_thread = None

        result: list[Path | None] = [None]
        exc: list[Exception | None] = [None]
        tag = label or f"shard {remote_index}"

        def _run() -> None:
            try:
                result[0] = self._download_remote_index(remote_index)
            except Exception as error:
                exc[0] = error

        thread = threading.Thread(
            target=_run,
            name=f"wds-dl-{remote_index}",
            daemon=True,
        )
        with self._lock:
            self._inflight_download_thread = thread
        thread.start()
        try:
            self._wait_for_download_thread(thread, label=tag, timeout=ROLLING_WDS_DOWNLOAD_TIMEOUT_S)
        except TimeoutError:
            # This shard's download is wedged. Clear the in-flight reference and re-raise so
            # the caller counts it as a failed shard and tries a different one.
            with self._lock:
                if self._inflight_download_thread is thread:
                    self._inflight_download_thread = None
            raise
        with self._lock:
            if self._inflight_download_thread is thread:
                self._inflight_download_thread = None
        if exc[0] is not None:
            raise exc[0]
        if result[0] is None:
            raise RuntimeError(f"[rolling-wds] download thread for {tag} finished without a path")
        return result[0]

    def _default_download(self, shard: HfWdsShardRef, dest: Path) -> Path:
        import time
        from huggingface_hub import hf_hub_download

        if self._is_valid_shard(dest):
            return dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        # 60 s inactivity timeout per chunk — prevents indefinite hangs on stalled connections.
        os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
        token = self.hf_token if isinstance(self.hf_token, str) else None

        last_exc: Exception | None = None
        for attempt in range(3):
            if attempt > 0:
                wait = 20 * attempt  # 20 s, 40 s
                print(
                    f"[rolling-wds] retry {attempt}/2 for {shard.path_in_repo} in {wait}s ...",
                    flush=True,
                )
                time.sleep(wait)
                if self._is_valid_shard(dest):
                    return dest
            try:
                # Download directly under local_dir. huggingface_hub does not use
                # ~/.cache/huggingface when local_dir is set; only small metadata lives
                # under local_dir/.cache/huggingface/.
                downloaded = Path(
                    hf_hub_download(
                        repo_id=shard.repo_id,
                        filename=shard.path_in_repo,
                        repo_type="dataset",
                        token=token,
                        local_dir=str(self.local_dir),
                    )
                )
                if downloaded != dest:
                    with self._lock:
                        if self._is_valid_shard(dest):
                            return dest
                    if dest.exists():
                        dest.unlink()
                    downloaded.replace(dest)
                if not self._is_valid_shard(dest):
                    raise OSError(f"downloaded shard failed tar validation: {shard.path_in_repo}")
                return dest
            except Exception as exc:
                last_exc = exc
                print(
                    f"[rolling-wds] download attempt {attempt + 1}/3 failed for "
                    f"{shard.path_in_repo}: {exc}",
                    flush=True,
                )
                if dest.exists():
                    try:
                        dest.unlink()
                    except OSError:
                        pass

        raise OSError(
            f"[rolling-wds] failed to download {shard.path_in_repo} after 3 attempts"
        ) from last_exc

    def _download_remote_index(self, remote_index: int) -> Path:
        shard = self.remote_shards[remote_index % len(self.remote_shards)]
        dest = self._local_path_for(shard)
        return self._download_fn(shard, dest)

    def _remote_index_for_path(self, path: Path) -> int | None:
        return self._shard_index_by_name.get(path.name)

    def _refill_epoch_queue(self, *, exclude: set[int] | None = None) -> None:
        excluded = exclude or set()
        candidates = [index for index in range(len(self.remote_shards)) if index not in excluded]
        if not candidates:
            candidates = list(range(len(self.remote_shards)))
        self._rng.shuffle(candidates)
        self._remaining_remote_indices.extend(candidates)

    def _next_epoch_remote_index(self, *, exclude: set[int] | None = None) -> int:
        excluded = exclude or set()
        while True:
            if not self._remaining_remote_indices:
                self._refill_epoch_queue(exclude=excluded)
            remote_index = self._remaining_remote_indices.popleft()
            if remote_index not in excluded:
                return remote_index

    def fill_initial(self) -> None:
        total = min(self.max_local_shards, len(self.remote_shards))
        print(
            f"Rolling WDS: {len(self.remote_shards)} remote shards; "
            f"downloading initial window of {total} to {self.local_dir} ...",
            flush=True,
        )
        planned: list[tuple[int, int]] = []
        with self._lock:
            self._window.clear()
            self._window_indices.clear()
            self._remaining_remote_indices.clear()
            for index in range(total):
                planned.append((index, self._next_epoch_remote_index()))

        for index, remote_index in planned:
            shard = self.remote_shards[remote_index % len(self.remote_shards)]
            print(
                f"  [{index + 1}/{total}] {shard.path_in_repo}",
                flush=True,
            )
            path = self._download_remote_index(remote_index)
            with self._lock:
                self._window.append(path)
                self._window_indices.add(remote_index)

        print(
            f"Rolling WDS: initial window ready ({len(planned)} shards).",
            flush=True,
        )

    def local_urls(self) -> list[str]:
        with self._lock:
            return [str(path) for path in self._window]

    def current_shard_path(self) -> Path | None:
        with self._lock:
            if not self._window:
                return None
            return self._window[0]

    def _schedule_prefetch(self) -> None:
        with self._lock:
            if self._prefetch_thread is not None:
                return
            inflight = self._inflight_download_thread
            if inflight is not None and inflight.is_alive():
                return
            remote_index = self._next_epoch_remote_index(exclude=set(self._window_indices))
            target = self._local_path_for(self.remote_shards[remote_index])

        def _run() -> None:
            try:
                self._download_remote_index_sync(remote_index, label=f"shard {remote_index} prefetch")
            except Exception as exc:
                print(f"[rolling-wds] prefetch shard {remote_index} failed: {exc}", flush=True)

        thread = threading.Thread(target=_run, name=f"wds-prefetch-{target.name}", daemon=True)
        with self._lock:
            self._prefetch_thread = thread
            self._prefetch_target = target
            self._prefetch_remote_index = remote_index
        thread.start()

    def schedule_prefetch(self) -> None:
        """Start downloading one replacement shard outside the current window."""
        self._schedule_prefetch()

    def advance_after(self, finished_url: str) -> None:
        finished_name = _normalize_shard_name(finished_url)
        with self._lock:
            if not self._window or self._window[0].name != finished_name:
                return
            finished_path = self._window.popleft()
            finished_index = self._remote_index_for_path(finished_path)
            if finished_index is not None:
                self._window_indices.discard(finished_index)
            prefetch_thread = self._prefetch_thread
            prefetch_remote_index = self._prefetch_remote_index
            self._prefetch_thread = None
            self._prefetch_target = None
            self._prefetch_remote_index = None

        remote_index: int | None = None
        next_path: Path | None = None
        if prefetch_thread is not None and prefetch_remote_index is not None:
            remote_index = prefetch_remote_index
            prefetch_dest = self._local_path_for(self.remote_shards[remote_index])
            try:
                self._wait_for_download_thread(
                    prefetch_thread,
                    label=f"shard {remote_index} prefetch",
                    timeout=ROLLING_WDS_DOWNLOAD_TIMEOUT_S,
                )
            except TimeoutError:
                # Prefetch is wedged on a slow/stuck HF link. Abandon it (the daemon thread is
                # left to die on its own) and clear the in-flight reference so the fallback
                # download below is not blocked by it. Training keeps going on the other shards
                # already in the local window.
                with self._lock:
                    self._inflight_download_thread = None
                remote_index = None
                next_path = None
            else:
                if self._is_valid_shard(prefetch_dest):
                    next_path = prefetch_dest
                elif prefetch_dest.exists():
                    prefetch_dest.unlink()
                else:
                    remote_index = None

        if remote_index is None:
            with self._lock:
                remote_index = self._next_epoch_remote_index(exclude=set(self._window_indices))
            next_path = None

        if next_path is None or not self._is_valid_shard(next_path):
            # One download at a time, but each wait is capped at ROLLING_WDS_DOWNLOAD_TIMEOUT_S
            # so a stuck shard is abandoned rather than deadlocking training. Abandoned daemon
            # downloads are left to die; _MAX_SHARD_ERRORS bounds how many we try before bailing
            # out with a clear "resume once connectivity is restored" error.
            _MAX_SHARD_ERRORS = 8
            _excluded: set[int] = set(self._window_indices)

            _errors = 0
            while True:
                if remote_index is None:
                    with self._lock:
                        remote_index = self._next_epoch_remote_index(exclude=_excluded)

                _target_index = remote_index
                try:
                    next_path = self._download_remote_index_sync(
                        _target_index,
                        label=f"shard {_target_index}",
                    )
                    remote_index = _target_index
                    break
                except Exception as exc:
                    _errors += 1
                    print(
                        f"[rolling-wds] shard {_target_index} failed ({exc}) "
                        f"(error {_errors}/{_MAX_SHARD_ERRORS}); trying another ...",
                        flush=True,
                    )
                    _excluded.add(_target_index)
                    remote_index = None
                    if _errors >= _MAX_SHARD_ERRORS:
                        raise RuntimeError(
                            f"[rolling-wds] {_MAX_SHARD_ERRORS} shards failed to download "
                            "— HF may be down. Resume training once connectivity is restored."
                        ) from exc

        with self._lock:
            if finished_path.exists():
                finished_path.unlink()
            self._window.append(next_path)
            self._window_indices.add(remote_index)

        self._schedule_prefetch()


def _pick_sample_field(sample: dict[str, Any], stem: str) -> tuple[str, bytes | np.ndarray]:
    npy_key = f"{stem}.npy"
    bin_key = f"{stem}.bin"
    if npy_key in sample:
        return "npy", sample[npy_key]
    if bin_key in sample:
        return "bin", sample[bin_key]
    raise KeyError(f"Sample missing {npy_key!r} and {bin_key!r}")


def _require_shape(manifest: WdsCacheManifest | None, attr: str, stem: str) -> tuple[int, ...]:
    shape = getattr(manifest, attr, None) if manifest is not None else None
    if shape is None:
        raise ValueError(
            f"Sample uses {stem}.bin but manifest.json has no {attr}. "
            "Re-cache with .npy tensors or place manifest.json next to the shards."
        )
    return tuple(shape)


def _bin_to_numpy(data: bytes, *, dtype_name: str, shape: tuple[int, ...]) -> np.ndarray:
    dtype = _NUMPY_DTYPES.get(dtype_name)
    if dtype is None:
        raise ValueError(f"Unsupported on-disk dtype {dtype_name!r} for raw tensor payload.")
    array = np.frombuffer(data, dtype=dtype)
    expected = int(np.prod(shape))
    if array.size != expected:
        raise ValueError(
            f"Raw tensor payload size mismatch for shape {shape}: "
            f"expected {expected} elements, got {array.size}."
        )
    return array.reshape(shape).copy()


def _array_to_logical_tensor(array: np.ndarray, *, dtype_name: str) -> torch.Tensor:
    if dtype_name == "bfloat16":
        if array.dtype != np.uint16:
            raise ValueError(
                f"bfloat16 cache payload must be stored as uint16 bit patterns, got {array.dtype}."
            )
        return torch.from_numpy(array).view(torch.bfloat16)
    if dtype_name == "float8_e4m3fn":
        if not hasattr(torch, "float8_e4m3fn"):
            raise RuntimeError(
                "Cache dtype is float8_e4m3fn but this PyTorch build lacks float8 support."
            )
        if array.dtype != np.uint8:
            raise ValueError(
                f"float8_e4m3fn cache payload must be stored as uint8 bit patterns, got {array.dtype}."
            )
        # Upcast to bf16 (not fp16): the DiT trains in bf16, so this keeps one dtype
        # end-to-end and avoids a needless fp16->bf16 recast. fp16's extra mantissa is
        # wasted here — the source is fp8 and the consumer is bf16.
        return torch.from_numpy(array).view(torch.float8_e4m3fn).to(torch.bfloat16)
    return torch.from_numpy(array)


def _load_npy_array(value: bytes | np.ndarray) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    return np.load(io.BytesIO(value), allow_pickle=False)


def _latents_to_tensor(
    value: bytes | np.ndarray,
    fmt: str,
    *,
    manifest: WdsCacheManifest | None,
) -> torch.Tensor:
    dtype_name = manifest.latents_dtype if manifest is not None else "float16"
    if fmt == "npy":
        return _array_to_logical_tensor(_load_npy_array(value), dtype_name=dtype_name).contiguous()
    shape = _require_shape(manifest, "latents_shape", "latents")
    payload = value if isinstance(value, (bytes, bytearray)) else bytes(value)
    array = _bin_to_numpy(payload, dtype_name=dtype_name, shape=shape)
    return _array_to_logical_tensor(array, dtype_name=dtype_name).contiguous()


def _text_to_tensor(
    value: bytes | np.ndarray,
    fmt: str,
    *,
    manifest: WdsCacheManifest | None,
) -> torch.Tensor:
    text_dtype = manifest.text_dtype if manifest is not None else "float16"
    if fmt == "npy":
        return _array_to_logical_tensor(_load_npy_array(value), dtype_name=text_dtype).contiguous()
    shape = _require_shape(manifest, "text_shape", "text")
    payload = value if isinstance(value, (bytes, bytearray)) else bytes(value)
    array = _bin_to_numpy(payload, dtype_name=text_dtype, shape=shape)
    return _array_to_logical_tensor(array, dtype_name=text_dtype).contiguous()


def _mask_to_tensor(
    value: bytes | np.ndarray,
    fmt: str,
    *,
    manifest: WdsCacheManifest | None,
) -> torch.Tensor:
    if fmt == "npy":
        return torch.from_numpy(_load_npy_array(value)).contiguous()
    if manifest is not None and manifest.text_shape:
        shape = (manifest.text_shape[0],)
    else:
        shape = _require_shape(manifest, "text_shape", "text_mask")
        shape = (shape[0],)
    dtype_name = manifest.text_mask_dtype if manifest is not None else "uint8"
    payload = value if isinstance(value, (bytes, bytearray)) else bytes(value)
    array = _bin_to_numpy(payload, dtype_name=dtype_name, shape=shape)
    return torch.from_numpy(array).contiguous()


def sample_to_record(
    sample: dict[str, Any],
    *,
    manifest: WdsCacheManifest | None = None,
) -> LatentRecord:
    lat_fmt, latents_value = _pick_sample_field(sample, "latents")
    text_fmt, text_value = _pick_sample_field(sample, "text")
    mask_fmt, mask_value = _pick_sample_field(sample, "text_mask")

    if any(fmt == "bin" for fmt in (lat_fmt, text_fmt, mask_fmt)) and manifest is None:
        raise ValueError(
            "Sample uses raw .bin tensors but no manifest.json was loaded. "
            "Place manifest.json next to the shards or use .npy cache files."
        )

    image_latent = _latents_to_tensor(latents_value, lat_fmt, manifest=manifest)
    text_embedding = _text_to_tensor(text_value, text_fmt, manifest=manifest)
    attention_mask = _mask_to_tensor(mask_value, mask_fmt, manifest=manifest).to(torch.int16)

    dino_embedding = None
    if "dino.npy" in sample or "dino.bin" in sample:
        dino_fmt, dino_value = _pick_sample_field(sample, "dino")
        dino_dtype = manifest.dino_dtype if manifest is not None else "float16"
        if dino_fmt == "npy":
            dino_embedding = _array_to_logical_tensor(
                _load_npy_array(dino_value), dtype_name=dino_dtype
            ).contiguous()
        else:
            shape = _require_shape(manifest, "dino_shape", "dino")
            payload = dino_value if isinstance(dino_value, (bytes, bytearray)) else bytes(dino_value)
            dino_embedding = _array_to_logical_tensor(
                _bin_to_numpy(payload, dtype_name=dino_dtype, shape=shape), dtype_name=dino_dtype
            ).contiguous()

    caption = decode_text(sample["caption.txt"])
    metadata = decode_json(sample["meta.json"])
    return LatentRecord(
        image_latent=image_latent,
        text_embedding=text_embedding,
        attention_mask=attention_mask,
        caption=caption,
        metadata=metadata,
        path=str(sample.get("__url__", "")),
        dino_embedding=dino_embedding,
    )


def resolve_wds_urls(paths: str | Iterable[str], *, hf_token: str | bool | None = None) -> list[str]:
    if isinstance(paths, str):
        raw_paths = split_csv(paths)
    else:
        raw_paths = [str(path) for path in paths]

    urls: list[str] = []
    for raw in raw_paths:
        if raw.startswith("hf://"):
            from huggingface_hub import HfApi, HfFileSystem, get_token, hf_hub_url

            fs = HfFileSystem(token=hf_token)
            try:
                paths = sorted(fs.glob(raw))
            except FileNotFoundError:
                paths = []
            if not paths:
                prefix = "hf://datasets/"
                if raw.startswith(prefix):
                    repo_and_path = raw[len(prefix) :]
                    repo_id, _, pattern = repo_and_path.partition("/")
                    base = pattern.split("*", 1)[0].rstrip("/")
                    api = HfApi(token=hf_token if isinstance(hf_token, str) else None)
                    print(
                        f"Listing HF dataset files for {repo_id} (this can take several minutes) ...",
                        flush=True,
                    )
                    repo_files = sorted(api.list_repo_files(repo_id=repo_id, repo_type="dataset"))
                    paths = [
                        f"hf://datasets/{repo_id}/{path}"
                        for path in repo_files
                        if path.startswith(base.rstrip("/") + "/") and path.endswith(".tar")
                    ]
                    print(f"Found {len(paths)} tar shards under {base}", flush=True)
            for path in paths:
                resolved = fs.resolve_path(path)
                url = hf_hub_url(resolved.repo_id, resolved.path_in_repo, repo_type="dataset")
                resolved_token = get_token() if hf_token is True else hf_token
                if resolved_token:
                    header = shlex.quote(f"Authorization:Bearer {resolved_token}")
                    url = f"pipe:curl -s -L -H {header} {shlex.quote(url)}"
                urls.append(url)
        else:
            matches = sorted(glob.glob(raw)) if any(char in raw for char in "*?[]") else []
            urls.extend(matches or [raw])

    if not urls:
        raise FileNotFoundError(f"No WebDataset shards found for: {paths}")
    return urls


@dataclass
class WdsLatentCacheDataset(Dataset[LatentRecord]):
    """Indexable WebDataset cache loader.

    This loader intentionally materializes the shard index once at startup so
    the existing training loop can use normal shuffling, ``len(dataset)``, and
    ``dataset[0]`` shape probing.
    """

    paths: str | Iterable[str]
    hf_token: str | bool | None = None

    def __post_init__(self) -> None:
        import webdataset as wds

        self.urls = resolve_wds_urls(self.paths, hf_token=self.hf_token)
        self.manifest = load_wds_cache_manifest(self.paths, hf_token=self.hf_token, urls=self.urls)
        self.samples = []
        dataset = wds.WebDataset(self.urls, shardshuffle=False).decode()
        for sample in dataset:
            self.samples.append(sample)
        if not self.samples:
            raise FileNotFoundError(f"No samples found in WebDataset shards for: {self.paths}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> LatentRecord:
        return sample_to_record(self.samples[index], manifest=self.manifest)


@dataclass
class StreamingWdsLatentCacheDataset(IterableDataset[LatentRecord]):
    """Streaming WebDataset cache loader that does not materialize samples."""

    paths: str | Iterable[str]
    hf_token: str | bool | None = None
    shardshuffle: bool | int = False
    shuffle_buffer: int = 0

    def __post_init__(self) -> None:
        self.urls = resolve_wds_urls(self.paths, hf_token=self.hf_token)
        self.manifest = load_wds_cache_manifest(self.paths, hf_token=self.hf_token, urls=self.urls)

    def __iter__(self):
        import webdataset as wds

        worker_info = get_worker_info()
        urls = self.urls
        if worker_info is not None:
            urls = split_urls_for_worker(urls, worker_info.id, worker_info.num_workers)
        if not urls:
            return

        dataset = wds.WebDataset(urls, shardshuffle=self.shardshuffle).decode()
        if self.shuffle_buffer > 0:
            dataset = dataset.shuffle(self.shuffle_buffer)
        for sample in dataset:
            yield sample_to_record(sample, manifest=self.manifest)


@dataclass
class RollingStreamingWdsLatentCacheDataset(IterableDataset[LatentRecord]):
    """Stream through a rolling local window over shuffled remote WDS shard epochs."""

    paths: str | Iterable[str]
    local_dir: str | Path
    rolling_shards: int = 25
    hf_token: str | bool | None = None
    shuffle_buffer: int = 1000
    seed: int | None = None

    def __post_init__(self) -> None:
        if isinstance(self.paths, str):
            raw_paths = split_csv(self.paths)
        else:
            raw_paths = [str(path) for path in self.paths]
        if len(raw_paths) != 1 or not raw_paths[0].startswith("hf://"):
            raise ValueError(
                "Rolling local WDS streaming requires exactly one hf://datasets/... glob in paths."
            )
        self.remote_paths = raw_paths[0]
        if self.shuffle_buffer < 1:
            raise ValueError("Rolling local WDS streaming requires shuffle_buffer >= 1.")
        self.cache = RollingLocalWdsCache(
            remote_paths=self.remote_paths,
            local_dir=self.local_dir,
            max_local_shards=self.rolling_shards,
            hf_token=self.hf_token,
            seed=self.seed,
        )
        self.manifest = load_wds_cache_manifest(
            self.remote_paths,
            hf_token=self.hf_token,
            urls=self.cache.local_urls(),
        )

    def __iter__(self):
        import webdataset as wds

        worker_info = get_worker_info()
        if worker_info is not None and worker_info.num_workers > 1:
            raise RuntimeError(
                "Rolling local WDS streaming requires --num-workers 0 so shard rotation "
                "happens in the training process."
            )

        while True:
            shard_path = self.cache.current_shard_path()
            if shard_path is None:
                return

            self.cache.schedule_prefetch()
            shard_name = shard_path.name
            dataset = wds.WebDataset([str(shard_path)], shardshuffle=False).decode()
            if self.shuffle_buffer > 0:
                dataset = dataset.shuffle(self.shuffle_buffer)

            for sample in dataset:
                yield sample_to_record(sample, manifest=self.manifest)

            self.cache.advance_after(shard_name)


class _PrefetchStopped(Exception):
    """Internal: raised inside the prefetch pool when stop_event fires mid-wait."""


@dataclass
class PrefetchingWdsLatentCacheDataset(IterableDataset[LatentRecord]):
    """Stream remote WDS shards with a PARALLEL download-ahead window — the same
    prefetch pattern the caching tool uses.

    A thread pool keeps ``prefetch_shards`` tars downloading at once into
    ``local_dir``; the main thread yields samples from each shard in order and
    deletes it after use. This keeps the GPU fed even when one shard download is
    slower than consuming a shard (the failure mode of the serial rolling cache on
    large 4 GB shards). Run with ``num_workers=0`` — prefetch is on internal threads.
    """

    paths: str | Iterable[str]
    local_dir: str | Path
    prefetch_shards: int = 8
    download_workers: int = 4
    hf_token: str | bool | None = None
    shuffle_buffer: int = 2000
    seed: int | None = None
    # Background decode: a producer thread runs download->decode->sample_to_record and
    # pushes finished LatentRecords into a bounded queue, so per-sample tensor
    # reconstruction overlaps the GPU step (the consumer's CUDA call releases the GIL
    # while the GPU is busy). Set <= 0 to decode inline on the consumer thread.
    decode_queue_size: int = 512

    def __post_init__(self) -> None:
        if isinstance(self.paths, str):
            raw_paths = split_csv(self.paths)
        else:
            raw_paths = [str(p) for p in self.paths]
        if len(raw_paths) != 1 or not raw_paths[0].startswith("hf://"):
            raise ValueError(
                "PrefetchingWdsLatentCacheDataset requires exactly one hf://datasets/... glob."
            )
        self.remote_paths = raw_paths[0]
        self.refs = list_hf_wds_shard_refs(self.remote_paths, hf_token=self.hf_token)
        self.manifest = load_wds_cache_manifest(self.remote_paths, hf_token=self.hf_token)

    def _download(self, ref: HfWdsShardRef) -> Path:
        import time

        from huggingface_hub import hf_hub_download

        token = self.hf_token if isinstance(self.hf_token, str) else None
        # 60s inactivity timeout per chunk. hf_xet/hf_transfer (if installed) handle
        # connection drops far better than the pure-python backend, but can still stall,
        # so the consumer also enforces a wall-clock budget (see result() below).
        os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
        last_exc: Exception | None = None
        for attempt in range(3):
            if attempt > 0:
                time.sleep(20 * attempt)  # 20s, 40s backoff before retrying
                print(
                    f"[prefetch-wds] retry {attempt}/2 for {ref.path_in_repo} ...",
                    flush=True,
                )
            try:
                # local_dir (not the blob cache) so os.remove actually frees the shard.
                return Path(
                    hf_hub_download(
                        repo_id=ref.repo_id, filename=ref.path_in_repo, repo_type="dataset",
                        token=token, local_dir=str(self.local_dir),
                    )
                )
            except Exception as exc:
                last_exc = exc
                print(
                    f"[prefetch-wds] download attempt {attempt + 1}/3 failed for "
                    f"{ref.path_in_repo}: {exc}",
                    flush=True,
                )
        raise OSError(
            f"[prefetch-wds] failed to download {ref.path_in_repo} after 3 attempts"
        ) from last_exc

    def _shard_stream(self, stop_event):
        """Download-ahead window -> per-sample LatentRecords, looping epochs forever.
        Honors ``stop_event`` so the consumer can shut the producer down cleanly.

        Downloads run on raw DAEMON threads (not a ThreadPoolExecutor) on purpose:
        concurrent.futures joins its worker threads at interpreter exit, which would
        block the process on an in-flight 4GB shard download when the Trainer abandons
        the iterator at max_steps. Daemon threads are killed immediately at exit instead,
        so the process terminates promptly (the abandoned partial download is harmless;
        local_dir is transient)."""
        import queue
        import threading

        import webdataset as wds

        rng = random.Random(self.seed)
        order = list(range(len(self.refs)))

        while not stop_event.is_set():  # IterableDataset: loop epochs; Trainer stops at max_steps
            rng.shuffle(order)
            # Daemon-thread download pool: submit(pos) enqueues a download; result(pos)
            # blocks until that position's shard is on disk. Results/events are only
            # mutated structurally by THIS (producer) thread; workers just fill them.
            task_q: queue.Queue = queue.Queue()
            results: dict[int, object] = {}
            events: dict[int, threading.Event] = {}
            lock = threading.Lock()

            def worker() -> None:
                while not stop_event.is_set():
                    try:
                        pos = task_q.get(timeout=0.5)
                    except queue.Empty:
                        continue
                    if pos is None or stop_event.is_set():
                        return
                    try:
                        res: object = self._download(self.refs[order[pos]])
                    except Exception as exc:  # deliver to result(pos) rather than die
                        res = exc
                    with lock:
                        results[pos] = res
                        ev = events.get(pos)  # may be gone if result() already abandoned pos
                    if ev is not None:
                        ev.set()

            workers = [
                threading.Thread(target=worker, name="wds-dl", daemon=True)
                for _ in range(max(1, self.download_workers))
            ]
            for w in workers:
                w.start()

            def submit(pos: int) -> None:
                if 0 <= pos < len(order) and pos not in events:
                    events[pos] = threading.Event()
                    task_q.put(pos)

            def result(pos: int, budget: float | None = None):
                ev = events[pos]
                waited = 0.0
                while not ev.wait(timeout=0.5):
                    if stop_event.is_set():
                        raise _PrefetchStopped()
                    waited += 0.5
                    if budget is not None and waited >= budget:
                        # Silent TCP stall: the download is wedged with no data and won't
                        # return on its own. Abandon this shard so one bad connection can't
                        # freeze the in-order, single-process training loop. Drop the
                        # bookkeeping (the daemon worker is left to die at exit); skipping a
                        # shard out of thousands is harmless for training.
                        with lock:
                            events.pop(pos, None)
                            results.pop(pos, None)
                        raise TimeoutError(
                            f"shard pos={pos} ({self.refs[order[pos]].filename}) "
                            f"exceeded {int(budget)}s download budget"
                        )
                with lock:
                    res = results.pop(pos)
                events.pop(pos, None)
                if isinstance(res, BaseException):
                    raise res
                return res

            try:
                for pos in range(min(self.prefetch_shards, len(order))):
                    submit(pos)
                for pos in range(len(order)):
                    if stop_event.is_set():
                        return
                    submit(pos + self.prefetch_shards)  # keep the window full even if this one is skipped
                    try:
                        path = result(pos, budget=ROLLING_WDS_DOWNLOAD_TIMEOUT_S)
                    except _PrefetchStopped:
                        return
                    except Exception as exc:  # download failed/abandoned -> skip, don't freeze
                        print(f"[prefetch-wds] skipping shard pos={pos}: {exc}", flush=True)
                        continue
                    dataset = wds.WebDataset([str(path)], shardshuffle=False).decode()
                    if self.shuffle_buffer > 0:
                        dataset = dataset.shuffle(self.shuffle_buffer)
                    for sample in dataset:
                        if stop_event.is_set():
                            return
                        yield sample_to_record(sample, manifest=self.manifest)
                    try:
                        os.remove(path)
                    except FileNotFoundError:
                        pass
            finally:
                # Wake idle workers so they observe stop and exit; they're daemon, so even
                # a worker mid-download never blocks interpreter shutdown.
                for _ in workers:
                    task_q.put(None)

    def __iter__(self):
        import queue
        import threading

        worker_info = get_worker_info()
        if worker_info is not None and worker_info.num_workers > 1:
            raise RuntimeError(
                "PrefetchingWdsLatentCacheDataset requires num_workers=0 "
                "(it prefetches shards on internal threads)."
            )
        Path(self.local_dir).mkdir(parents=True, exist_ok=True)

        stop_event = threading.Event()

        # Inline path: decode on the consumer thread (decode_queue_size <= 0).
        if self.decode_queue_size <= 0:
            try:
                yield from self._shard_stream(stop_event)
            finally:
                stop_event.set()
            return

        # Background-decode path: a producer thread runs download+decode and fills a
        # bounded queue, so sample reconstruction overlaps the GPU step.
        record_queue: queue.Queue = queue.Queue(maxsize=self.decode_queue_size)
        sentinel = object()

        def producer() -> None:
            try:
                for record in self._shard_stream(stop_event):
                    while not stop_event.is_set():
                        try:
                            record_queue.put(record, timeout=1.0)
                            break
                        except queue.Full:
                            continue
                    if stop_event.is_set():
                        return
            except Exception as exc:  # propagate to the consumer rather than hang
                record_queue.put(exc)
            finally:
                record_queue.put(sentinel)

        thread = threading.Thread(
            target=producer, name="wds-prefetch-decode", daemon=True
        )
        thread.start()
        try:
            while True:
                item = record_queue.get()
                if item is sentinel:
                    break
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            stop_event.set()
            # Drain so a blocked producer.put() can return and the thread can exit.
            try:
                while True:
                    record_queue.get_nowait()
            except queue.Empty:
                pass
            thread.join(timeout=5.0)


__all__ = [
    "HfWdsShardRef",
    "PrefetchingWdsLatentCacheDataset",
    "RollingLocalWdsCache",
    "RollingStreamingWdsLatentCacheDataset",
    "StreamingWdsLatentCacheDataset",
    "WdsCacheManifest",
    "WdsLatentCacheDataset",
    "collate_latent_records",
    "list_hf_wds_shard_refs",
    "load_wds_cache_manifest",
    "resolve_wds_urls",
    "sample_to_record",
    "split_urls_for_worker",
]
