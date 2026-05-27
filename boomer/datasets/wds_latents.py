"""WebDataset latent/text cache loader for Boomer training."""

from __future__ import annotations

import io
import json
import glob
import os
import random
import shlex
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

    def _default_download(self, shard: HfWdsShardRef, dest: Path) -> Path:
        from huggingface_hub import hf_hub_download

        if dest.exists() and dest.stat().st_size > 0:
            return dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        # Download directly under local_dir so shards are not duplicated in ~/.cache/huggingface.
        print(f"[wds-roll] downloading {shard.path_in_repo} -> {dest}", flush=True)
        downloaded = Path(
            hf_hub_download(
                repo_id=shard.repo_id,
                filename=shard.path_in_repo,
                repo_type="dataset",
                token=self.hf_token if isinstance(self.hf_token, str) else None,
                local_dir=str(self.local_dir),
            )
        )
        if downloaded != dest:
            if dest.exists():
                dest.unlink()
            downloaded.replace(dest)
        print(f"[wds-roll] ready {dest}", flush=True)
        return dest

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
        with self._lock:
            self._window.clear()
            self._window_indices.clear()
            self._remaining_remote_indices.clear()
            for _ in range(min(self.max_local_shards, len(self.remote_shards))):
                remote_index = self._next_epoch_remote_index()
                self._window.append(self._download_remote_index(remote_index))
                self._window_indices.add(remote_index)

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
            remote_index = self._next_epoch_remote_index(exclude=set(self._window_indices))
            target = self._local_path_for(self.remote_shards[remote_index])

        def _run() -> None:
            self._download_remote_index(remote_index)

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

            remote_index: int | None = None
            next_path: Path | None = None
            if (
                self._prefetch_target is not None
                and self._prefetch_remote_index is not None
                and self._prefetch_thread is not None
            ):
                remote_index = self._prefetch_remote_index
                next_path = self._prefetch_target
                self._prefetch_thread.join()
                self._prefetch_thread = None
                self._prefetch_target = None
                self._prefetch_remote_index = None
            else:
                remote_index = self._next_epoch_remote_index(exclude=set(self._window_indices))
                next_path = self._local_path_for(self.remote_shards[remote_index])

            if next_path is None or remote_index is None:
                return
            if not next_path.exists() or next_path.stat().st_size == 0:
                next_path = self._download_remote_index(remote_index)

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
        return torch.from_numpy(array).view(torch.float8_e4m3fn).to(torch.float16)
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

    caption = decode_text(sample["caption.txt"])
    metadata = decode_json(sample["meta.json"])
    return LatentRecord(
        image_latent=image_latent,
        text_embedding=text_embedding,
        attention_mask=attention_mask,
        caption=caption,
        metadata=metadata,
        path=str(sample.get("__url__", "")),
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

            print(f"[wds-roll] finished {shard_name}; rotating local shard window", flush=True)
            self.cache.advance_after(shard_name)


__all__ = [
    "HfWdsShardRef",
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
