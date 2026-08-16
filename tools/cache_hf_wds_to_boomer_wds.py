"""Stream HF WebDataset image/caption shards into Boomer latent WebDataset shards.

Optimized variant. Drop-in replacement for cache_latents.py with the Boomer
latent WDS schema (latents/text/text_mask + caption.txt + meta.json) and
manifest format, but with the following speed knobs:

  --gpu-decode            Batch-decode JPEGs on GPU via nvJPEG (torchvision).
                          Falls back to PIL for PNG/WEBP/corrupt JPEGs.
  --dali-decode           Use NVIDIA DALI for the JPEG decode/resize/normalize
                          batch path. Keeps WebDataset caption/meta grouping.
  --compile               torch.compile the VAE and text encoder. fixed shapes
                          only — keeps --batch-size and --resolution constant.
  --compile-mode          "default" | "reduce-overhead" | "max-autotune" |
                          "max-autotune-no-cudagraphs". reduce-overhead/max-autotune
                          enable CUDA graphs, which conflict with --async-cpu-transfer +
                          threaded encode (AssertionError); prefer default or
                          max-autotune-no-cudagraphs here.
  --channels-last         Use channels_last memory format for the VAE.
  --source-preset         Built-in HF WebDataset source presets, including
                          FineT2I enhanced square.
  --storage-dtype         "bf16" (default) | "fp16" | "fp32" for latents/text.
  --text-storage-dtype    Override text payload with "fp8" when desired.
  --raw-tensors           Skip the .npy header by default; store raw bytes plus
                          dtype/shape in manifest. Smaller files, faster writes.
  --world-size / --rank   Manual multi-rank sharding (or use torchrun and they
                          are read from env). Each rank handles urls[rank::world].
  --resume                Resume from output_dir/resume.json if present.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import io
import json
import os
import queue
import shlex
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from boomer.sana_latent_cache import (  # noqa: E402
    CacheConfig,
    DEFAULT_SCALING_FACTOR,
    DEFAULT_TEXT_ENCODER_ID,
    DEFAULT_VAE_ID,
    encode_captions,
    encode_image_latent,
    hf_token_value,
    image_to_tensor,
    load_text_encoder,
    load_vae,
)


DEFAULT_SOURCE_GLOB = "hf://datasets/BLIP3o/BLIP3o-Pretrain-JourneyDB/**/*.tar"
FINE_T2I_ENHANCED_SQUARE_GLOB = (
    "hf://datasets/ma-xu/fine-t2i/synthetic_enhanced_prompt_square_resolution/train-*.tar"
)
SOURCE_PRESETS = {
    "journeydb": DEFAULT_SOURCE_GLOB,
    "fine-t2i-enhanced-square": FINE_T2I_ENHANCED_SQUARE_GLOB,
}

# JPEG magic numbers — fast type sniff so we route to nvJPEG only when valid.
_JPEG_SOI = b"\xff\xd8\xff"

# Lazily imported, set once if --gpu-decode is enabled.
_TV_DECODE_JPEG = None
_TV_IMAGE_READ_MODE = None


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def npy_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return buffer.getvalue()


def raw_bytes(array: np.ndarray) -> bytes:
    # Skips the ~80-byte .npy header. Caller must record dtype/shape elsewhere.
    return np.ascontiguousarray(array).tobytes()


def normalize_storage_dtype(value: str) -> str:
    normalized = value.lower()
    aliases = {
        "fp16": "float16",
        "float16": "float16",
        "half": "float16",
        "bf16": "bfloat16",
        "bfloat16": "bfloat16",
        "fp32": "float32",
        "float32": "float32",
        "fp8": "float8_e4m3fn",
        "float8": "float8_e4m3fn",
        "float8_e4m3fn": "float8_e4m3fn",
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported storage dtype: {value}")
    return aliases[normalized]


def storage_dtype_to_numpy(tensor: torch.Tensor, dtype_name: str) -> np.ndarray:
    """Return a NumPy payload for a logical storage dtype.

    NumPy has no portable bfloat16 dtype, so bf16 is stored as uint16 bit
    patterns and decoded using the manifest's logical dtype.
    """
    dtype_name = normalize_storage_dtype(dtype_name)
    detached = tensor.detach()
    if dtype_name == "float16":
        return detached.to(torch.float16).cpu().numpy()
    if dtype_name == "bfloat16":
        return detached.to(torch.bfloat16).view(torch.uint16).cpu().numpy()
    if dtype_name == "float32":
        return detached.to(torch.float32).cpu().numpy()
    if dtype_name == "float8_e4m3fn":
        if not hasattr(torch, "float8_e4m3fn"):
            raise RuntimeError("This PyTorch build does not support float8_e4m3fn storage.")
        return detached.to(torch.float8_e4m3fn).view(torch.uint8).cpu().numpy()
    raise ValueError(f"Unsupported storage dtype: {dtype_name}")


def storage_dtype_to_tensor(tensor: torch.Tensor, dtype_name: str) -> torch.Tensor:
    """Return a tensor payload matching storage_dtype_to_numpy byte layout."""
    dtype_name = normalize_storage_dtype(dtype_name)
    detached = tensor.detach()
    if dtype_name == "float16":
        return detached.to(torch.float16).contiguous()
    if dtype_name == "bfloat16":
        return detached.to(torch.bfloat16).view(torch.uint16).contiguous()
    if dtype_name == "float32":
        return detached.to(torch.float32).contiguous()
    if dtype_name == "float8_e4m3fn":
        if not hasattr(torch, "float8_e4m3fn"):
            raise RuntimeError("This PyTorch build does not support float8_e4m3fn storage.")
        return detached.to(torch.float8_e4m3fn).view(torch.uint8).contiguous()
    raise ValueError(f"Unsupported storage dtype: {dtype_name}")


def tensor_to_storage_npy(tensor: torch.Tensor, dtype_name: str) -> bytes:
    return npy_bytes(storage_dtype_to_numpy(tensor, dtype_name))


def tensor_to_u8_npy(tensor: torch.Tensor) -> bytes:
    return npy_bytes(tensor.detach().cpu().to(torch.uint8).numpy())


def batch_to_u8_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().to(torch.uint8).cpu().numpy()


def batch_to_u8_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to(torch.uint8).contiguous()


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def split_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def make_wds_member_selector(image_keys: Iterable[str], caption_keys: Iterable[str]) -> Callable[[str], bool]:
    """Build a WebDataset tar-member filter for image/caption payloads only."""
    suffixes = {
        key.lower().lstrip(".")
        for key in [*image_keys, *caption_keys]
        if key and key.strip()
    }
    # Be a little generous for common caption/image variants while still
    # dropping unrelated sidecar files before WebDataset groups samples.
    suffixes.update({"jpg", "jpeg", "png", "webp", "txt", "json"})
    ordered = tuple(sorted(suffixes, key=len, reverse=True))

    def select(name: str) -> bool:
        lowered = str(name).lower()
        return any(lowered.endswith(f".{suffix}") for suffix in ordered)

    return select


# ---------------------------------------------------------------------------
# Sample parsing
# ---------------------------------------------------------------------------


def normalize_caption(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                return normalize_caption(json.loads(stripped))
            except Exception:
                return stripped
        return stripped
    if isinstance(value, list) and value:
        return normalize_caption(value[0])
    if isinstance(value, dict):
        for key in ("caption", "text", "prompt", "original_prompt", "generated_text", "txt"):
            if key in value:
                return normalize_caption(value[key])
    return str(value).strip() or None


def normalize_image_pil(value: Any) -> Image.Image | None:
    if value is None:
        return None
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, bytes):
        with Image.open(io.BytesIO(value)) as image:
            return image.convert("RGB")
    return None


def first_present(sample: dict[str, Any], candidates: Iterable[str]) -> tuple[str | None, Any]:
    for key in candidates:
        if key in sample and sample[key] is not None:
            return key, sample[key]
    return None, None


def prepare_source_sample_cpu(
    sample: dict[str, Any],
    *,
    image_keys: Iterable[str],
    caption_keys: Iterable[str],
    resolution: int,
    global_index: int,
) -> dict[str, Any]:
    """CPU path: decode + resize with PIL, produce a normalized image tensor."""
    try:
        _, image_raw = first_present(sample, image_keys)
        _, caption_raw = first_present(sample, caption_keys)
        image = normalize_image_pil(image_raw)
        caption = normalize_caption(caption_raw)
        if image is None or caption is None:
            return {
                "skip": True,
                "global_index": global_index,
                "error_type": "MissingField",
                "error": "missing image or caption",
            }
        return {
            "skip": False,
            "image_tensor": image_to_tensor(image, resolution),
            "caption": caption,
            "global_index": global_index,
            "source_key": sample.get("__key__", ""),
        }
    except Exception as exc:
        return {
            "skip": True,
            "global_index": global_index,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def prepare_source_sample_gpu(
    sample: dict[str, Any],
    *,
    image_keys: Iterable[str],
    caption_keys: Iterable[str],
    resolution: int,
    global_index: int,
) -> dict[str, Any]:
    """GPU-decode path: hand raw JPEG bytes downstream, decode in batch on GPU."""
    try:
        key, image_raw = first_present(sample, image_keys)
        _, caption_raw = first_present(sample, caption_keys)
        caption = normalize_caption(caption_raw)
        if image_raw is None or caption is None:
            return {
                "skip": True,
                "global_index": global_index,
                "error_type": "MissingField",
                "error": "missing image or caption",
            }

        # If bytes start with JPEG SOI marker, defer decode to GPU. Otherwise
        # fall back to PIL on CPU (PNG/WEBP/etc).
        if isinstance(image_raw, (bytes, bytearray)) and bytes(image_raw[:3]) == _JPEG_SOI:
            return {
                "skip": False,
                "jpeg_bytes": bytes(image_raw),
                "image_tensor": None,
                "caption": caption,
                "global_index": global_index,
                "source_key": sample.get("__key__", ""),
            }
        image = normalize_image_pil(image_raw)
        if image is None:
            return {
                "skip": True,
                "global_index": global_index,
                "error_type": "MissingField",
                "error": "could not decode non-jpeg image",
            }
        return {
            "skip": False,
            "jpeg_bytes": None,
            "image_tensor": image_to_tensor(image, resolution),
            "caption": caption,
            "global_index": global_index,
            "source_key": sample.get("__key__", ""),
        }
    except Exception as exc:
        return {
            "skip": True,
            "global_index": global_index,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def prepare_source_chunk(
    items: list[tuple[int, dict[str, Any]]],
    *,
    image_keys: Iterable[str],
    caption_keys: Iterable[str],
    resolution: int,
    gpu_decode: bool,
) -> list[dict[str, Any]]:
    fn = prepare_source_sample_gpu if gpu_decode else prepare_source_sample_cpu
    return [
        fn(
            sample,
            image_keys=image_keys,
            caption_keys=caption_keys,
            resolution=resolution,
            global_index=global_index,
        )
        for global_index, sample in items
    ]


# ---------------------------------------------------------------------------
# Source URL resolution
# ---------------------------------------------------------------------------


def hf_wds_urls(source_glob: str, token: str | bool | None) -> list[str]:
    from fnmatch import fnmatch

    # Local shards (e.g. img2dataset output): return file paths directly.
    # WebDataset reads local .tar paths natively, no curl pipe needed.
    if not source_glob.startswith("hf://"):
        import glob as _glob

        local = sorted(_glob.glob(source_glob))
        if local:
            return local
        # Not a local match and not an hf:// URI — fall through to HF resolution
        # (covers bare "datasets/..." style globs against HfFileSystem).

    from huggingface_hub import HfApi, HfFileSystem, get_token, hf_hub_url

    fs = HfFileSystem(token=token)
    try:
        paths = fs.glob(source_glob)
    except FileNotFoundError:
        paths = []
    if not paths and source_glob.startswith("hf://datasets/"):
        repo_and_path = source_glob[len("hf://datasets/") :]
        repo_id, _, pattern = repo_and_path.partition("/")
        prefix = pattern.split("*", 1)[0].rstrip("/")
        api = HfApi(token=token if isinstance(token, str) else None)
        repo_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
        paths = [
            f"hf://datasets/{repo_id}/{path}"
            for path in repo_files
            if path.startswith(prefix) and fnmatch(path, pattern)
        ]
    files = [fs.resolve_path(path) for path in sorted(paths)]
    urls = [hf_hub_url(file.repo_id, file.path_in_repo, repo_type="dataset") for file in files]
    if not urls:
        raise FileNotFoundError(f"No source tar files matched: {source_glob}")

    resolved_token = get_token() if token is True else token
    if resolved_token:
        # Env var, not interpolation: webdataset embeds the pipe command in every
        # OSError it raises, which prints an inlined bearer token in plaintext.
        os.environ["BOOMER_HF_BEARER"] = resolved_token
        return [
            "pipe:curl -sSL --fail --retry 5 --retry-delay 2 --speed-limit 1 --speed-time 60"
            f' -H "Authorization:Bearer $BOOMER_HF_BEARER" {shlex.quote(url)}'
            for url in urls
        ]
    return urls


def hf_wds_urls_from_list_file(list_file: str | Path, token: str | bool | None) -> list[str]:
    """Resolve a newline-delimited list of local paths, URLs, pipes, or hf:// WDS paths."""
    from huggingface_hub import HfFileSystem, get_token, hf_hub_url

    raw_paths = [
        line.strip()
        for line in Path(list_file).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not raw_paths:
        raise FileNotFoundError(f"No source tar files listed in: {list_file}")

    fs = HfFileSystem(token=token)
    urls: list[str] = []
    for path in raw_paths:
        if path.startswith("hf://"):
            file = fs.resolve_path(path)
            url = hf_hub_url(file.repo_id, file.path_in_repo, repo_type="dataset")
            urls.append(url)
        else:
            urls.append(path)

    resolved_token = get_token() if token is True else token
    if resolved_token:
        # See hf_wds_urls: the token goes through the environment so shard errors
        # cannot print it.
        os.environ["BOOMER_HF_BEARER"] = resolved_token
        return [
            "pipe:curl -sSL --fail --retry 5 --retry-delay 2 --speed-limit 1 --speed-time 60"
            f' -H "Authorization:Bearer $BOOMER_HF_BEARER" {shlex.quote(url)}'
            if url.startswith("http://") or url.startswith("https://")
            else url
            for url in urls
        ]
    return urls


def is_local_source_path(url: str) -> bool:
    return not (
        url.startswith("pipe:")
        or url.startswith("http://")
        or url.startswith("https://")
        or url.startswith("s3://")
        or url.startswith("gs://")
    )


def purge_source_download_cache(source_path: Path) -> None:
    """Drop HF download-staging residue for a consumed shard so it cannot accumulate.

    hf_hub_download(local_dir=...) leaves small lock/incomplete/metadata stubs under
    ``<local_dir>/.cache/huggingface/download``. We delete the shard after reading it,
    but those stubs are otherwise never cleaned -- one per shard over a long run. The
    shard's local_dir is its parent directory (flat rolling-localize layout).
    """
    cache = source_path.parent / ".cache" / "huggingface" / "download"
    if not cache.exists():
        return
    stem = source_path.name
    for pattern in (f"{stem}*.lock", f"{stem}*.incomplete", f"{stem}*.metadata"):
        for leftover in cache.rglob(pattern):
            try:
                leftover.unlink()
            except OSError:
                pass


def cleanup_empty_parents(path: Path, root: Path) -> None:
    root = root.resolve()
    current = path.resolve()
    while current != root and root in current.parents:
        try:
            current.rmdir()
        except OSError:
            break
        current = current.parent


def wait_for_local_source_path(
    path: Path,
    *,
    ready_suffix: str,
    skip_suffix: str | None,
    timeout: float,
    poll_interval: float,
) -> bool:
    """Block until the shard is ready (return True) or marked skipped (return False).

    The rolling-localize sidecar writes a ``<path><skip_suffix>`` marker for a shard
    it gave up downloading; treating that as "move on" lets the cacher tolerate a dead
    shard instead of waiting out the full timeout and crashing the long-lived run.
    """
    ready = Path(f"{path}{ready_suffix}")
    skip = Path(f"{path}{skip_suffix}") if skip_suffix else None
    deadline = None if timeout <= 0 else time.monotonic() + timeout
    while not (path.exists() and ready.exists()):
        if skip is not None and skip.exists():
            return False
        if deadline is not None and time.monotonic() > deadline:
            raise TimeoutError(f"Timed out waiting for local source shard: {path}")
        time.sleep(poll_interval)
    return True


# ---------------------------------------------------------------------------
# GPU decode + resize
# ---------------------------------------------------------------------------


def _ensure_tv_decode() -> None:
    global _TV_DECODE_JPEG, _TV_IMAGE_READ_MODE
    if _TV_DECODE_JPEG is None:
        from torchvision.io import ImageReadMode, decode_jpeg  # noqa: WPS433

        _TV_DECODE_JPEG = decode_jpeg
        _TV_IMAGE_READ_MODE = ImageReadMode


def gpu_decode_and_resize(
    jpeg_bytes_list: list[bytes],
    *,
    resolution: int,
    device: str,
) -> torch.Tensor:
    """Batch-decode JPEGs on GPU via nvJPEG, then resize + normalize.

    Returns float tensor of shape (N, 3, H, W) in [-1, 1] on `device`.
    """
    _ensure_tv_decode()
    assert _TV_DECODE_JPEG is not None and _TV_IMAGE_READ_MODE is not None

    # nvJPEG wants CPU uint8 tensors of raw bytes; it returns CUDA uint8 CHW.
    cpu_tensors = [torch.frombuffer(bytearray(b), dtype=torch.uint8) for b in jpeg_bytes_list]
    decoded = _TV_DECODE_JPEG(
        cpu_tensors,
        mode=_TV_IMAGE_READ_MODE.RGB,
        device=device,
    )
    # decode_jpeg with a list returns CHW uint8 tensors of varying H/W. Bucket images
    # by identical (C, H, W) so each distinct size is resized+cropped as ONE batched
    # call instead of one F.interpolate launch per image. The per-image geometry
    # (resize-to-fill min side, then center crop) is unchanged, so outputs are bit-for-bit
    # identical to the old loop; only kernel-launch overhead is removed.
    buckets: dict[tuple[int, ...], list[int]] = {}
    for i, img in enumerate(decoded):
        buckets.setdefault(tuple(img.shape), []).append(i)

    out: list[torch.Tensor | None] = [None] * len(decoded)
    for shape, idxs in buckets.items():
        group = torch.stack([decoded[i] for i in idxs], dim=0).to(torch.float32)
        _, _, height, width = group.shape
        scale = resolution / min(height, width)
        new_h = round(height * scale)
        new_w = round(width * scale)
        group = F.interpolate(
            group,
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        top = max((new_h - resolution) // 2, 0)
        left = max((new_w - resolution) // 2, 0)
        group = group[:, :, top : top + resolution, left : left + resolution]
        for j, i in enumerate(idxs):
            out[i] = group[j : j + 1]
    batch = torch.cat(out, dim=0)  # type: ignore[arg-type]
    # uint8 [0,255] -> [-1, 1]
    batch = batch.mul_(2.0 / 255.0).sub_(1.0)
    return batch


def cuda_device_id(device: str) -> int:
    if device == "cuda":
        return torch.cuda.current_device()
    if device.startswith("cuda:"):
        return int(device.split(":", 1)[1])
    raise ValueError(f"DALI requires a CUDA device, got {device!r}")


class DaliJpegBatchDecoder:
    """Decode, resize, center-crop, and normalize JPEG batches with DALI."""

    def __init__(
        self,
        *,
        resolution: int,
        device_id: int,
        num_threads: int,
        device_memory_padding: int = 67_108_864,
        host_memory_padding: int = 16_777_216,
    ) -> None:
        try:
            import nvidia.dali.fn as fn  # noqa: F401
            import nvidia.dali.types as types  # noqa: F401
            from nvidia.dali.pipeline import Pipeline  # noqa: F401
            from nvidia.dali.plugin.pytorch.torch_utils import feed_ndarray, to_torch_type
        except Exception as exc:  # pragma: no cover - depends on optional Colab package
            raise RuntimeError(
                "NVIDIA DALI is not available. In CUDA 13 Colab, install it with "
                "`pip install -U --extra-index-url https://pypi.nvidia.com nvidia-dali-cuda130`."
            ) from exc

        self.resolution = resolution
        self.device_id = device_id
        self.num_threads = num_threads
        self.device_memory_padding = device_memory_padding
        self.host_memory_padding = host_memory_padding
        self._pipelines: dict[int, Any] = {}
        self._feed_ndarray = feed_ndarray
        self._to_torch_type = to_torch_type

    def _build_pipeline(self, batch_size: int) -> Any:
        import nvidia.dali.fn as fn
        import nvidia.dali.types as types
        from nvidia.dali.pipeline import Pipeline

        resolution = self.resolution
        device_memory_padding = self.device_memory_padding
        host_memory_padding = self.host_memory_padding

        class _Pipeline(Pipeline):
            def __init__(self, *, batch_size: int, num_threads: int, device_id: int) -> None:
                super().__init__(
                    batch_size=batch_size,
                    num_threads=num_threads,
                    device_id=device_id,
                    exec_async=True,
                    exec_pipelined=False,
                    prefetch_queue_depth=1,
                )

            def define_graph(self) -> Any:
                encoded = fn.external_source(name="encoded", dtype=types.UINT8, ndim=1)
                images = fn.decoders.image(
                    encoded,
                    device="mixed",
                    output_type=types.RGB,
                    device_memory_padding=device_memory_padding,
                    host_memory_padding=host_memory_padding,
                )
                images = fn.resize(
                    images,
                    resize_shorter=resolution,
                    interp_type=types.INTERP_LINEAR,
                    antialias=True,
                )
                return fn.crop_mirror_normalize(
                    images,
                    crop=(resolution, resolution),
                    mean=[127.5, 127.5, 127.5],
                    std=[127.5, 127.5, 127.5],
                    output_layout="CHW",
                    dtype=types.FLOAT,
                )

        pipe = _Pipeline(
            batch_size=batch_size,
            num_threads=self.num_threads,
            device_id=self.device_id,
        )
        pipe.build()
        return pipe

    def discard(self, batch_size: int) -> None:
        self._pipelines.pop(batch_size, None)

    def __call__(self, jpeg_bytes_list: list[bytes]) -> torch.Tensor:
        batch_size = len(jpeg_bytes_list)
        if batch_size <= 0:
            raise ValueError("DALI JPEG batch must be non-empty.")
        pipe = self._pipelines.get(batch_size)
        if pipe is None:
            pipe = self._build_pipeline(batch_size)
            self._pipelines[batch_size] = pipe

        encoded = [np.frombuffer(data, dtype=np.uint8) for data in jpeg_bytes_list]
        # Feed the external batch as a run input. With external_source this
        # requires prefetch_queue_depth=1 or exec_pipelined=False; otherwise
        # DALI can try to prefetch a non-existent next batch.
        (images,) = pipe.run(encoded=encoded)
        tensor = images.as_tensor()
        dtype = self._to_torch_type[tensor.dtype]
        out = torch.empty(
            tensor.shape(),
            dtype=dtype,
            device=torch.device("cuda", self.device_id),
        )
        stream = torch.cuda.current_stream(out.device)
        return self._feed_ndarray(tensor, out, cuda_stream=stream)


class AsyncCpuTransferStager:
    """Stage encoded tensors into pinned CPU memory on a separate CUDA stream."""

    def __init__(self, device: str) -> None:
        self.device = torch.device(device)
        self.stream = torch.cuda.Stream(device=self.device)

    def stage(
        self,
        *,
        items: list[dict[str, Any]],
        latents: torch.Tensor,
        text_embeddings: torch.Tensor,
        attention_masks: torch.Tensor,
        latent_storage_dtype: str,
        text_storage_dtype: str,
    ) -> dict[str, Any]:
        current_stream = torch.cuda.current_stream(self.device)
        self.stream.wait_stream(current_stream)
        sources: list[torch.Tensor] = [latents, text_embeddings, attention_masks]

        with torch.cuda.stream(self.stream):
            latents_gpu = storage_dtype_to_tensor(latents, latent_storage_dtype)
            text_gpu = storage_dtype_to_tensor(text_embeddings, text_storage_dtype)
            masks_gpu = batch_to_u8_tensor(attention_masks)
            sources.extend([latents_gpu, text_gpu, masks_gpu])

            latents_cpu = torch.empty(
                latents_gpu.shape,
                dtype=latents_gpu.dtype,
                device="cpu",
                pin_memory=True,
            )
            text_cpu = torch.empty(
                text_gpu.shape,
                dtype=text_gpu.dtype,
                device="cpu",
                pin_memory=True,
            )
            masks_cpu = torch.empty(
                masks_gpu.shape,
                dtype=masks_gpu.dtype,
                device="cpu",
                pin_memory=True,
            )

            latents_cpu.copy_(latents_gpu, non_blocking=True)
            text_cpu.copy_(text_gpu, non_blocking=True)
            masks_cpu.copy_(masks_gpu, non_blocking=True)

            event = torch.cuda.Event()
            event.record(self.stream)

        return {
            "items": items,
            "latents_cpu": latents_cpu,
            "text_cpu": text_cpu,
            "masks_cpu": masks_cpu,
            "copy_event": event,
            "_cuda_sources": sources,
        }


# ---------------------------------------------------------------------------
# Encoder wrappers
# ---------------------------------------------------------------------------


@torch.inference_mode()
def encode_images_microbatched(
    vae: Any,
    image_tensors: torch.Tensor,
    config: CacheConfig,
    microbatch_size: int,
) -> torch.Tensor:
    outputs = []
    for start in range(0, image_tensors.shape[0], microbatch_size):
        outputs.append(encode_image_latent(vae, image_tensors[start : start + microbatch_size], config))
    return torch.cat(outputs, dim=0)


@torch.inference_mode()
def encode_text_microbatched(
    text_processor: Any,
    text_encoder: Any,
    captions: list[str],
    config: CacheConfig,
    microbatch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # NOTE: captions are encoded in input order (no length sorting). Regrouping captions
    # changes the bf16 GEMM batch dimension, which perturbs cached values at the
    # numerical-noise level; keeping input order makes the cache reproducible.
    # Qwen path: text_processor is None and text_encoder is a QwenReferenceEncoder.
    # Cache the same instruction-prefixed hidden states the connector reads at inference,
    # padded/truncated to config.max_length for fixed-shape storage.
    if text_processor is None:
        from boomer.models.qwen_connector import encode_qwen_text_fixed

        embeddings = []
        masks = []
        for start in range(0, len(captions), microbatch_size):
            hidden, mask = encode_qwen_text_fixed(
                text_encoder, captions[start : start + microbatch_size], config.max_length
            )
            embeddings.append(hidden)
            masks.append(mask)
        return torch.cat(embeddings, dim=0), torch.cat(masks, dim=0)

    embeddings = []
    masks = []
    for start in range(0, len(captions), microbatch_size):
        text_embedding, attention_mask = encode_captions(
            text_processor,
            text_encoder,
            captions[start : start + microbatch_size],
            config,
        )
        embeddings.append(text_embedding)
        masks.append(attention_mask)
    return torch.cat(embeddings, dim=0), torch.cat(masks, dim=0)


# ---------------------------------------------------------------------------
# Hub upload
# ---------------------------------------------------------------------------


class AsyncHubUploader:
    def __init__(
        self,
        *,
        repo_id: str | None,
        repo_path_prefix: str,
        workers: int,
        delete_uploaded: bool,
        create_repo: bool,
        commit_batch_size: int = 8,
        commit_timeout: float = 900.0,
        commit_max_retries: int = 10,
    ) -> None:
        self.repo_id = repo_id
        self.repo_path_prefix = repo_path_prefix.strip("/")
        self.delete_uploaded = delete_uploaded
        # A create_commit that exceeds this wall-clock budget is treated as a silent
        # socket stall (the HF transfer thread can't be cancelled), abandoned, and
        # retried on a fresh connection. Generous because a full batch is a big upload.
        self.commit_timeout = max(30.0, commit_timeout)
        self.commit_max_retries = max(1, commit_max_retries)
        # HF caps repo commits at 256/hour. One commit per shard trips that with
        # small (1 GB) shards, but small shards are required so training streaming
        # doesn't stall. So commit MANY shards per commit instead of shrinking the
        # batch granularity that training needs.
        self.commit_batch_size = max(1, commit_batch_size)
        self._uploaded: set[str] = set()
        self._buffer: list[str] = []          # queued, not yet committed
        self._futures: dict[int, tuple[concurrent.futures.Future, list[str]]] = {}
        self._batch_id = 0
        self._lock = threading.Lock()
        self._executor = None
        self._api = None
        if repo_id:
            from huggingface_hub import HfApi

            self._api = HfApi()
            if create_repo:
                self._api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
            self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers))

    @property
    def enabled(self) -> bool:
        return self.repo_id is not None

    def upload_path(self, local_path: str | Path) -> None:
        if not self.enabled:
            return
        path = str(local_path)
        with self._lock:
            if path in self._uploaded or path in self._buffer:
                return
            if any(path in paths for _, paths in self._futures.values()):
                return
            self._buffer.append(path)
            if len(self._buffer) >= self.commit_batch_size:
                self._kick_batch_locked()
        print(f"[upload] queued {Path(path).name}")

    def _kick_batch_locked(self) -> None:
        """Submit up to commit_batch_size buffered files as ONE commit. Caller holds lock."""
        if not self._buffer:
            return
        assert self._executor is not None
        paths = self._buffer[: self.commit_batch_size]
        del self._buffer[: len(paths)]
        bid = self._batch_id
        self._batch_id += 1
        self._futures[bid] = (self._executor.submit(self._commit_batch_sync, paths), paths)
        print(f"[upload] commit batch {bid}: {len(paths)} shard(s)")

    @staticmethod
    def _run_with_timeout(fn, budget_s):
        """Run ``fn()`` in a daemon thread; raise ``TimeoutError`` if it exceeds
        ``budget_s``. A wedged HF transfer (silent socket stall) can't be cancelled, so
        it is abandoned to die on its own rather than hanging the uploader forever."""
        box = {}

        def runner():
            try:
                box["val"] = fn()
            except BaseException as exc:  # noqa: BLE001 - propagate to caller
                box["exc"] = exc

        th = threading.Thread(target=runner, name="commit-watchdog", daemon=True)
        th.start()
        th.join(budget_s)
        if th.is_alive():
            raise TimeoutError(f"create_commit exceeded {int(budget_s)}s wall-clock budget")
        if "exc" in box:
            raise box["exc"]
        return box.get("val")

    def _commit_batch_sync(self, paths: list[str]) -> None:
        assert self._api is not None and self.repo_id is not None
        import re as _re

        from huggingface_hub import CommitOperationAdd
        from huggingface_hub.errors import HfHubHTTPError

        ops = [
            CommitOperationAdd(
                path_in_repo=(
                    f"{self.repo_path_prefix}/{Path(p).name}"
                    if self.repo_path_prefix
                    else Path(p).name
                ),
                path_or_fileobj=p,
            )
            for p in paths
        ]
        attempt = 0          # 429 / 5xx rate-limit backoff (retried forever)
        net_attempt = 0      # silent stall / network drop (bounded retries)
        while True:
            try:
                # Wrap create_commit in a wall-clock watchdog: a silently-stalled upload
                # socket (no HTTP status, uncancellable transfer thread) would otherwise
                # hang the uploader forever and deadlock the writer on backpressure.
                self._run_with_timeout(
                    lambda: self._api.create_commit(
                        repo_id=self.repo_id,
                        repo_type="dataset",
                        operations=ops,
                        commit_message=f"Add {len(paths)} shard(s)",
                    ),
                    self.commit_timeout,
                )
                break
            except HfHubHTTPError as e:
                status = getattr(getattr(e, "response", None), "status_code", None)
                # 429 = commit rate limit (256/hr), 5xx = transient server error.
                # Both are recoverable: back off and retry forever rather than
                # letting one bad commit kill a multi-hour cache run. Honour the
                # server's "Retry after N seconds" hint when present.
                if status != 429 and not (status and 500 <= status < 600):
                    raise
                attempt += 1
                m = _re.search(r"[Rr]etry after (\d+)", str(e))
                wait_s = int(m.group(1)) + 5 if m else min(90 * attempt, 600)
                print(
                    f"[upload] {status} on batch of {len(paths)}; backing off {wait_s}s "
                    f"(attempt {attempt}) -- rate limit / transient, will retry"
                )
                time.sleep(wait_s)
            except Exception as e:  # noqa: BLE001 - TimeoutError / connection drops / etc.
                # The watchdog abandoned a wedged upload, or a connection dropped (no HTTP
                # status). create_commit is atomic and LFS/Xet blobs dedup by hash, so a
                # fresh retry is safe. Bounded so a genuine bad batch can't loop forever.
                net_attempt += 1
                if net_attempt > self.commit_max_retries:
                    raise
                wait_s = min(30 * net_attempt, 300)
                print(
                    f"[upload] commit stalled/failed ({e!r}); abandon + retry in {wait_s}s "
                    f"(attempt {net_attempt}/{self.commit_max_retries})"
                )
                time.sleep(wait_s)
        if self.delete_uploaded:
            for p in paths:
                try:
                    os.remove(p)
                except FileNotFoundError:
                    pass

    def poll(self, *, wait: bool = False) -> None:
        if not self.enabled:
            return
        with self._lock:
            # If progress is needed but nothing is in flight, commit a PARTIAL batch
            # so disk-backpressure (max_pending_uploads) can never deadlock waiting
            # for a full batch that won't fill.
            if wait and not self._futures and self._buffer:
                self._kick_batch_locked()
            in_flight = [fut for fut, _ in self._futures.values()]
        if wait and in_flight:
            concurrent.futures.wait(in_flight, return_when=concurrent.futures.FIRST_COMPLETED)
        done = []
        with self._lock:
            for bid, (future, paths) in list(self._futures.items()):
                if future.done():
                    done.append((bid, paths))
        for bid, paths in done:
            future, _ = self._futures.pop(bid)
            future.result()
            for p in paths:
                self._uploaded.add(p)
            print(f"[upload] done   batch {bid} ({len(paths)} shard(s))")

    def finish(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            while self._buffer:
                self._kick_batch_locked()
        while self._futures:
            self.poll(wait=True)
        assert self._executor is not None
        self._executor.shutdown(wait=True)


class ParquetShardWriter:
    """Drop-in stand-in for ``wds.ShardWriter`` that writes one Parquet shard per
    ``maxcount``/``maxsize`` window (one row per sample), matching the
    range-streamable cache format consumed by ``ParquetStreamingLatentCacheDataset``.

    Array members (``latents.npy``/``text.npy``/``text_mask.npy``[/``dino.npy``]) are
    stored as their ``.npy`` bytes in binary columns — byte-identical to the tar
    members, self-describing dtype/shape, decoded with ``np.load(BytesIO(col))``.
    ``caption.txt`` becomes a utf-8 string column and ``meta.json`` a JSON-string
    column (this tool emits both as bytes, so they are decoded here). ``post(path)``
    fires per finished shard so ``AsyncHubUploader`` batches upload/delete exactly
    as for tar. ``maxsize`` is the UNCOMPRESSED binary-payload budget; the on-disk
    Parquet file is usually a touch smaller after compression.
    """

    _MEMBER_TO_COLUMN = {
        "latents.npy": "latents",
        "text.npy": "text",
        "text_mask.npy": "text_mask",
        "dino.npy": "dino",
    }

    def __init__(
        self, pattern: str, *, maxcount: int, maxsize: int, start_shard: int = 0,
        post: Any = None, row_group_size: int = 512, compression: str = "zstd",
    ) -> None:
        self.pattern = pattern
        self.maxcount = max(1, maxcount)
        self.maxsize = max(1, maxsize)
        self.shard = start_shard
        self.post = post
        self.row_group_size = max(1, row_group_size)
        self.compression = compression
        self._reset()

    def _reset(self) -> None:
        self._cols: dict[str, list] = {}
        self._schema_cols: list[str] | None = None
        self._count = 0
        self._bytes = 0

    @staticmethod
    def _as_text(value: Any) -> str:
        if isinstance(value, (bytes, bytearray)):
            return value.decode("utf-8")
        if isinstance(value, str):
            return value
        return json.dumps(value, separators=(",", ":"))

    def write(self, sample: dict[str, Any]) -> None:
        rec: dict[str, Any] = {
            "__key__": str(sample["__key__"]),
            "latents": sample["latents.npy"],
            "text": sample["text.npy"],
            "text_mask": sample["text_mask.npy"],
            "caption": self._as_text(sample.get("caption.txt", "")),
            "meta": self._as_text(sample.get("meta.json", "{}")),
        }
        if "dino.npy" in sample:
            rec["dino"] = sample["dino.npy"]
        # Fix the column set from the first row; keep every shard rectangular even if a
        # later/earlier row is missing an optional member (e.g. dino) by padding nulls.
        if self._schema_cols is None:
            self._schema_cols = list(rec.keys())
            self._cols = {c: [] for c in self._schema_cols}
        for c in self._schema_cols:
            self._cols[c].append(rec.get(c))
        for c in rec:
            if c not in self._cols:  # a new optional column appeared mid-shard: backfill
                self._cols[c] = [None] * self._count + [rec[c]]
                self._schema_cols.append(c)
        self._count += 1
        self._bytes += sum(len(sample[m]) for m in self._MEMBER_TO_COLUMN if m in sample)
        if self._count >= self.maxcount or self._bytes >= self.maxsize:
            self.flush()

    def flush(self) -> None:
        if self._count == 0:
            return
        import pyarrow as pa
        import pyarrow.parquet as pq

        path = self.pattern % self.shard
        table = pa.table(self._cols)
        pq.write_table(
            table, path,
            row_group_size=min(self.row_group_size, self._count),
            compression=self.compression,
            use_dictionary=False,  # unique binary blobs: dictionary encoding only bloats
        )
        if self.post is not None:
            self.post(path)
        self.shard += 1
        self._reset()

    def close(self) -> None:
        self.flush()


class UploadingShardWriter:
    def __init__(
        self,
        *,
        output_dir: str | Path,
        split: str,
        maxsize: int,
        start_shard: int,
        uploader: AsyncHubUploader,
        rank_tag: str = "",
        on_shard_close: Callable[[str], None] | None = None,
        output_format: str = "wds",
        maxcount: int = 10_000_000,
        row_group_size: int = 512,
        compression: str = "zstd",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.split = split
        self.uploader = uploader
        self.on_shard_close = on_shard_close
        self.closed_paths: list[Path] = []
        suffix = f"-{rank_tag}" if rank_tag else ""
        if output_format == "parquet":
            pattern = str(self.output_dir / f"{split}{suffix}-%06d.parquet")
            self._writer: Any = ParquetShardWriter(
                pattern,
                maxcount=maxcount,
                maxsize=maxsize,
                start_shard=start_shard,
                post=self._on_close,
                row_group_size=row_group_size,
                compression=compression,
            )
        else:
            import webdataset as wds

            pattern = str(self.output_dir / f"{split}{suffix}-%06d.tar")
            self._writer = wds.ShardWriter(
                pattern,
                maxsize=maxsize,
                start_shard=start_shard,
                post=self._on_close,
            )

    def _on_close(self, path: str) -> None:
        closed = Path(path)
        self.closed_paths.append(closed)
        self.uploader.upload_path(closed)
        self.uploader.poll()
        if self.on_shard_close is not None:
            self.on_shard_close(path)

    def __enter__(self):
        return self

    def write(self, sample: dict[str, Any]) -> None:
        self._writer.write(sample)

    def __exit__(self, exc_type, exc, tb):
        self._writer.close()
        return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stream HF WDS shards into Boomer latent/text WDS cache shards.")
    parser.add_argument(
        "--source-preset",
        choices=sorted(SOURCE_PRESETS),
        help="Built-in source glob preset. Use fine-t2i-enhanced-square for ma-xu/fine-t2i square enhanced prompts.",
    )
    parser.add_argument("--source-glob", default=DEFAULT_SOURCE_GLOB)
    parser.add_argument("--source-list-file", default=None, help="Newline-delimited local/URL/hf:// tar paths.")
    parser.add_argument("--source-name", default=None, help="Human-readable source dataset/path recorded in manifests and sample metadata.")
    parser.add_argument(
        "--wait-local-source-shards",
        action="store_true",
        help="For local source-list paths, wait for '<path>.ready' before opening each shard.",
    )
    parser.add_argument(
        "--delete-local-source-shards-after-read",
        action="store_true",
        help="Delete each local source-list tar after it has been fully read.",
    )
    parser.add_argument("--local-source-ready-suffix", default=".ready")
    parser.add_argument(
        "--local-source-skip-suffix",
        default=".skip",
        help="Marker the rolling localizer writes for a shard it gave up on; the cacher skips it.",
    )
    parser.add_argument("--local-source-wait-timeout", type=float, default=3600.0)
    parser.add_argument("--local-source-poll-interval", type=float, default=0.25)
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=10_000)
    parser.add_argument("--skip-samples", type=int, default=0)
    parser.add_argument("--shuffle", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-encode-batch", type=int, default=64)
    parser.add_argument("--text-encode-batch", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--source-chunk-size", type=int, default=64)
    parser.add_argument(
        "--source-read-queue",
        type=int,
        default=2048,
        help="Samples buffered between the background shard reader and the encode loop. "
        "Bounds CPU RAM (~jpeg_size each); larger smooths shard-boundary stalls.",
    )
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--async-flush", action="store_true")
    parser.add_argument("--async-flush-depth", type=int, default=2)
    parser.add_argument(
        "--async-cpu-transfer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Copy encoded latents/text/masks to pinned CPU memory on a separate CUDA stream "
        "before tar writing (overlaps the GPU->CPU sync with the next encode). On by default; "
        "pass --no-async-cpu-transfer to disable (e.g. on CPU-only runs).",
    )
    parser.add_argument("--shard-size-gb", type=float, default=2.0)
    parser.add_argument("--shard-start-index", type=int, default=0)
    parser.add_argument("--image-keys", type=split_csv, default=["jpg", "jpeg", "png", "webp", "image"])
    parser.add_argument(
        "--caption-keys",
        type=split_csv,
        default=["txt", "text", "caption", "prompt", "json", "metadata"],
    )
    parser.add_argument(
        "--select-source-files",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Filter source tar members to image/caption files before WebDataset grouping.",
    )
    parser.add_argument(
        "--profile-stages",
        action="store_true",
        help="Print lightweight source/encode/write timing summaries with progress logs.",
    )
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--vae-id", default=DEFAULT_VAE_ID)
    parser.add_argument("--text-encoder-id", default=DEFAULT_TEXT_ENCODER_ID)
    parser.add_argument(
        "--text-encoder-kind",
        default="gemma",
        choices=["gemma", "qwen"],
        help="'gemma' (Sana causal-LM path) or 'qwen' (multimodal QwenReferenceEncoder; "
        "stores full instruction-prefixed hidden states matching the connector at inference).",
    )
    parser.add_argument(
        "--qwen-readout-layer",
        type=int,
        default=-1,
        help="Which Qwen hidden_states layer to cache (-1 = last). Only used with --text-encoder-kind qwen.",
    )
    parser.add_argument(
        "--qwen-instruction",
        default=None,
        help="Override the steering instruction prefix. Only used with --text-encoder-kind qwen.",
    )
    parser.add_argument(
        "--no-qwen-instruction",
        action="store_true",
        help="Disable the Qwen steering instruction prefix entirely.",
    )
    parser.add_argument(
        "--text-attn-implementation",
        default=None,
        choices=["eager", "sdpa", "flash_attention_2", "flash_attention_3"],
    )
    parser.add_argument("--max-length", type=int, default=CacheConfig.max_length)
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--scaling-factor", type=float, default=DEFAULT_SCALING_FACTOR)
    parser.add_argument("--hf-token")
    parser.add_argument("--hf-repo-id")
    parser.add_argument("--hf-path-prefix", default=None)
    parser.add_argument("--hf-upload-workers", type=int, default=2)
    parser.add_argument("--create-repo", action="store_true")
    parser.add_argument("--delete-uploaded", action="store_true")
    parser.add_argument("--log-every", type=int, default=1000)

    # Speed knobs
    parser.add_argument(
        "--gpu-decode",
        action="store_true",
        help="Decode JPEGs on GPU via nvJPEG. PNG/WEBP fall back to PIL on CPU.",
    )
    parser.add_argument(
        "--dali-decode",
        action="store_true",
        help="Use NVIDIA DALI for the JPEG decode/resize/crop/normalize batch path. Requires --gpu-decode.",
    )
    parser.add_argument(
        "--dali-num-threads",
        type=int,
        default=8,
        help="DALI CPU worker threads used by the JPEG decode pipeline.",
    )
    parser.add_argument(
        "--dali-decode-batch",
        "--jpeg-decode-batch",
        dest="dali_decode_batch",
        type=int,
        default=64,
        help="Maximum JPEGs per GPU decode call. Applies to DALI and torchvision nvJPEG paths.",
    )
    parser.add_argument(
        "--dali-device-memory-padding",
        type=int,
        default=67_108_864,
        help="nvJPEG device buffer preallocation per DALI decode pipeline (bytes). "
        "Raise for large JPEGs to avoid realloc/OOM spikes.",
    )
    parser.add_argument(
        "--dali-host-memory-padding",
        type=int,
        default=16_777_216,
        help="Host-side decode buffer preallocation per DALI decode pipeline (bytes).",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="torch.compile the VAE and text encoder.",
    )
    parser.add_argument(
        "--compile-mode",
        default="reduce-overhead",
        choices=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"],
    )
    parser.add_argument(
        "--channels-last",
        action="store_true",
        help="Use channels_last memory format for the VAE.",
    )
    parser.add_argument(
        "--storage-dtype",
        default="bf16",
        choices=["fp16", "bf16", "fp32"],
        help="On-disk dtype for both latents and text unless overridden. bf16 stores uint16 bit patterns.",
    )
    parser.add_argument(
        "--latent-storage-dtype",
        default=None,
        choices=["fp16", "bf16", "fp32"],
        help="Override on-disk latent dtype. Defaults to --storage-dtype.",
    )
    parser.add_argument(
        "--text-storage-dtype",
        default=None,
        choices=["fp16", "bf16", "fp32", "fp8"],
        help="Override on-disk text dtype. Defaults to --storage-dtype; fp8 halves payload.",
    )
    parser.add_argument(
        "--raw-tensors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write raw bytes instead of .npy. Smaller files and faster writes; disable with --no-raw-tensors.",
    )
    parser.add_argument(
        "--output-format",
        choices=["wds", "parquet"],
        default="wds",
        help="Shard container. 'parquet' writes range-streamable Parquet shards "
             "(consumed by ParquetStreamingLatentCacheDataset) instead of WebDataset "
             "tar; forces .npy member encoding (overrides --raw-tensors).",
    )
    parser.add_argument(
        "--parquet-row-group-size", type=int, default=512,
        help="Rows per Parquet row group (the range-read unit). 512 ~ the streaming sweet spot.",
    )
    parser.add_argument(
        "--parquet-compression", default="zstd",
        help="Parquet compression codec (zstd/snappy/none). Only used with --output-format parquet.",
    )
    parser.add_argument(
        "--parquet-max-count", type=int, default=10_000_000,
        help="Hard row cap per Parquet shard; shards normally roll on --shard-size-gb first.",
    )
    parser.add_argument("--world-size", type=int, default=None)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from output_dir/resume-r{rank}.json if present.",
    )
    return parser


# ---------------------------------------------------------------------------
# Multi-rank helpers
# ---------------------------------------------------------------------------


def resolve_rank(args: argparse.Namespace) -> tuple[int, int]:
    if args.rank is not None and args.world_size is not None:
        return args.rank, args.world_size
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    return rank, world


def resolve_device(arg_device: str, rank: int) -> str:
    if arg_device == "cuda" and torch.cuda.is_available():
        n = torch.cuda.device_count()
        if n > 1:
            return f"cuda:{rank % n}"
        return "cuda"
    if arg_device.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return arg_device


# ---------------------------------------------------------------------------
# Resume state
# ---------------------------------------------------------------------------


def resume_path(output_dir: Path, rank: int) -> Path:
    return output_dir / f"resume-r{rank:02d}.json"


def load_resume(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def save_resume(path: Path, state: dict[str, int]) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state))
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:  # noqa: C901 — single-file script, complexity is fine.
    args = build_arg_parser().parse_args()
    if args.output_format == "parquet" and args.raw_tensors:
        # Parquet stores arrays as self-describing .npy bytes; raw .bin has no header,
        # and both np.load (loader) and ParquetShardWriter's column map require .npy.
        print("[cache] --output-format parquet: forcing --no-raw-tensors (.npy member encoding)")
        args.raw_tensors = False
    if args.max_samples <= 0:
        raise ValueError("--max-samples must be positive.")
    if args.batch_size <= 0 or args.image_encode_batch <= 0 or args.text_encode_batch <= 0:
        raise ValueError("Batch sizes must be positive.")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative.")
    if args.prefetch_factor <= 0:
        raise ValueError("--prefetch-factor must be positive.")
    if args.source_chunk_size <= 0:
        raise ValueError("--source-chunk-size must be positive.")
    if args.async_flush_depth <= 0:
        raise ValueError("--async-flush-depth must be positive.")
    if args.dali_num_threads <= 0:
        raise ValueError("--dali-num-threads must be positive.")
    if args.dali_decode_batch <= 0:
        raise ValueError("--dali-decode-batch must be positive.")
    if args.dali_device_memory_padding <= 0:
        raise ValueError("--dali-device-memory-padding must be positive.")
    if args.dali_host_memory_padding <= 0:
        raise ValueError("--dali-host-memory-padding must be positive.")
    if args.dali_decode and not args.gpu_decode:
        raise ValueError("--dali-decode requires --gpu-decode.")

    import webdataset as wds

    rank, world_size = resolve_rank(args)
    device = resolve_device(args.device, rank)

    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    pin_memory = args.pin_memory and str(device).startswith("cuda")
    if args.pin_memory and not pin_memory:
        print("warning: --pin-memory is ignored when not encoding on CUDA")
    if args.dali_decode and not str(device).startswith("cuda"):
        raise ValueError("--dali-decode requires a CUDA device.")
    if args.async_cpu_transfer and not str(device).startswith("cuda"):
        # On by default; silently disable on non-CUDA rather than failing the run.
        args.async_cpu_transfer = False
    if args.async_cpu_transfer and not args.async_flush:
        print("warning: --async-cpu-transfer has little benefit without --async-flush")

    source_glob = SOURCE_PRESETS[args.source_preset] if args.source_preset else args.source_glob
    source_descriptor = args.source_name or (f"list:{args.source_list_file}" if args.source_list_file else source_glob)
    latent_storage_dtype = normalize_storage_dtype(args.latent_storage_dtype or args.storage_dtype)
    text_storage_dtype = normalize_storage_dtype(args.text_storage_dtype or args.storage_dtype)

    hf_token = hf_token_value(args.hf_token)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if rank == 0:
        print("Resolving source tar URLs...")
    urls_all = (
        hf_wds_urls_from_list_file(args.source_list_file, hf_token)
        if args.source_list_file
        else hf_wds_urls(source_glob, hf_token)
    )
    # Per-rank URL shard. Stable so resume works.
    urls = urls_all[rank::world_size]
    rank_tag = f"r{rank:02d}" if world_size > 1 else ""

    if rank == 0:
        print(
            f"source_shards_total={len(urls_all)} per_rank={len(urls)} "
            f"world_size={world_size}"
        )
        print(
            "knobs="
            f"workers={args.num_workers} prefetch_factor={args.prefetch_factor} "
            f"chunk={args.source_chunk_size} pin_memory={pin_memory} "
            f"async_flush={args.async_flush}/{args.async_flush_depth} "
            f"async_cpu_transfer={args.async_cpu_transfer} "
            f"gpu_decode={args.gpu_decode} compile={args.compile} "
            f"dali_decode={args.dali_decode} dali_threads={args.dali_num_threads} "
            f"dali_decode_batch={args.dali_decode_batch} "
            f"dali_device_mem_pad={args.dali_device_memory_padding} "
            f"dali_host_mem_pad={args.dali_host_memory_padding} "
            f"channels_last={args.channels_last} latent_dtype={latent_storage_dtype} "
            f"text_dtype={text_storage_dtype} "
            f"raw_tensors={args.raw_tensors} "
            f"select_source_files={args.select_source_files} "
            f"profile_stages={args.profile_stages}"
        )

    dali_decoder: DaliJpegBatchDecoder | None = None
    dali_failed = False
    if args.dali_decode:
        dali_decoder = DaliJpegBatchDecoder(
            resolution=args.resolution,
            device_id=cuda_device_id(device),
            num_threads=args.dali_num_threads,
            device_memory_padding=args.dali_device_memory_padding,
            host_memory_padding=args.dali_host_memory_padding,
        )
        print(
            f"[rank {rank}] DALI JPEG decode enabled "
            f"(resolution={args.resolution}, threads={args.dali_num_threads}, "
            f"device_mem_pad={args.dali_device_memory_padding}, "
            f"host_mem_pad={args.dali_host_memory_padding})."
        )
    cpu_transfer_stager = AsyncCpuTransferStager(device) if args.async_cpu_transfer else None
    if cpu_transfer_stager is not None:
        print(f"[rank {rank}] Async pinned CPU transfer enabled.")

    config = CacheConfig(
        resolution=args.resolution,
        vae_id=args.vae_id,
        text_encoder_id=args.text_encoder_id,
        text_attn_implementation=args.text_attn_implementation,
        max_length=args.max_length,
        dtype=args.dtype,
        device=device,
        scaling_factor=args.scaling_factor,
    )

    print(f"[rank {rank}] Loading encoders on {device}...")
    vae = load_vae(config, hf_token=hf_token)
    if args.text_encoder_kind == "qwen":
        from boomer.models.qwen_connector import QwenConnectorConfig, QwenReferenceEncoder

        instruction = None if args.no_qwen_instruction else (args.qwen_instruction or QwenConnectorConfig.instruction)
        qwen_cfg = QwenConnectorConfig(
            qwen_model_id=args.text_encoder_id,
            readout_layer=args.qwen_readout_layer,
            instruction=instruction,
            max_text_tokens=args.max_length,
            dtype=args.dtype,
            attn_implementation=args.text_attn_implementation,
        )
        text_encoder = QwenReferenceEncoder(qwen_cfg, hf_token=hf_token)
        text_encoder.load()
        text_encoder.to(device)
        text_processor = None  # QwenReferenceEncoder owns its processor
        print(f"[rank {rank}] Qwen encoder ready (hidden_size={text_encoder.hidden_size}, layer={args.qwen_readout_layer}).")
    else:
        text_processor, text_encoder = load_text_encoder(config, hf_token=hf_token)

    if args.channels_last:
        try:
            vae = vae.to(memory_format=torch.channels_last)
        except Exception as exc:
            print(f"[rank {rank}] channels_last on VAE failed, continuing: {exc}")

    if args.compile:
        # Best-effort compile. encode_image_latent / encode_captions wrap the
        # module-level forwards, so we compile the submodule, not the wrappers.
        # The VAE compiles with the requested --compile-mode. "default" is a fast (~1 min)
        # Inductor compile; the DC-AE VAE has many conv layers so "max-autotune-no-cudagraphs"
        # adds a long per-conv kernel search (minutes, can look like a hang) for a modest
        # steady-state gain — opt into it explicitly when you want it.
        vae_mode = args.compile_mode
        try:
            vae_compile_target = getattr(vae, "encoder", vae)
            compiled = torch.compile(vae_compile_target, mode=vae_mode, fullgraph=False)
            if hasattr(vae, "encoder"):
                vae.encoder = compiled
            else:
                vae = compiled
            print(f"[rank {rank}] torch.compile(vae, mode={vae_mode}) applied.")
        except Exception as exc:
            print(f"[rank {rank}] torch.compile on VAE failed, continuing: {exc}")
        if args.text_encoder_kind == "qwen":
            # Compile the inner Qwen transformer (not the wrapper). Force mode="default"
            # + dynamic=True regardless of --compile-mode: captions vary in length and the
            # readout uses output_hidden_states=True, so CUDA-graph modes (reduce-overhead)
            # would recompile/thrash. fullgraph=False tolerates the hidden-states breaks.
            try:
                # Quiet the dynamo graph-breaks/recompiles on the HF FA2 path:
                #  - capture_scalar_outputs: trace the seqlen `.item()` instead of breaking.
                #  - allow_unspec_int_on_nn_module: treat per-layer `layer_idx` as dynamic so
                #    the attention isn't recompiled once per transformer layer (hits the limit).
                import torch._dynamo as _dynamo
                for _flag in ("capture_scalar_outputs", "allow_unspec_int_on_nn_module"):
                    try:
                        setattr(_dynamo.config, _flag, True)
                    except Exception:
                        pass
                # forward() runs the base transformer (lm_head skipped), so compile that
                # exact module; fall back to the full model if the base wasn't located.
                if getattr(text_encoder, "_base_model", None) is not None:
                    text_encoder._base_model = torch.compile(
                        text_encoder._base_model, mode="default", fullgraph=False, dynamic=True
                    )
                    print(f"[rank {rank}] torch.compile(qwen base model, mode=default, dynamic=True) applied.")
                else:
                    text_encoder._model = torch.compile(
                        text_encoder.model, mode="default", fullgraph=False, dynamic=True
                    )
                    print(f"[rank {rank}] torch.compile(qwen full model, mode=default, dynamic=True) applied.")
            except Exception as exc:
                print(f"[rank {rank}] torch.compile on Qwen encoder failed, continuing: {exc}")
        else:
            try:
                text_encoder = torch.compile(
                    text_encoder,
                    mode=args.compile_mode,
                    fullgraph=False,
                    dynamic=True,  # captions can vary in length
                )
                print(f"[rank {rank}] torch.compile(text_encoder, mode={args.compile_mode}) applied.")
            except Exception as exc:
                print(f"[rank {rank}] torch.compile on text_encoder failed, continuing: {exc}")

    repo_prefix = args.hf_path_prefix if args.hf_path_prefix is not None else args.split
    uploader = AsyncHubUploader(
        repo_id=args.hf_repo_id,
        repo_path_prefix=repo_prefix,
        workers=args.hf_upload_workers,
        delete_uploaded=args.delete_uploaded,
        create_repo=args.create_repo,
    )

    # Null embeddings — rank 0 only writes them so we don't duplicate.
    if rank == 0:
        null_text, null_mask = encode_text_microbatched(
            text_processor,
            text_encoder,
            [""],
            config,
            microbatch_size=1,
        )
        null_text_path = output_dir / "null_text.npy"
        null_mask_path = output_dir / "null_mask.npy"
        null_text_path.write_bytes(tensor_to_storage_npy(null_text[0], text_storage_dtype))
        null_mask_path.write_bytes(tensor_to_u8_npy(null_mask[0]))
        uploader.upload_path(null_text_path)
        uploader.upload_path(null_mask_path)

    stats_lock = threading.Lock()
    stage_stats: dict[str, list[float]] = {}

    def record_stage(name: str, seconds: float, units: int = 1) -> None:
        if not args.profile_stages:
            return
        with stats_lock:
            totals = stage_stats.setdefault(name, [0.0, 0.0])
            totals[0] += seconds
            totals[1] += max(units, 1)

    def sync_stage_timer() -> float:
        if args.profile_stages and torch.cuda.is_available():
            torch.cuda.current_stream(device).synchronize()
        return time.time()

    def record_since(name: str, started_at: float, units: int = 1, *, sync: bool = True) -> None:
        if sync and args.profile_stages and torch.cuda.is_available():
            torch.cuda.current_stream(device).synchronize()
        record_stage(name, time.time() - started_at, units)

    def format_stage_stats() -> str:
        if not args.profile_stages:
            return ""
        with stats_lock:
            items = sorted(stage_stats.items())
        return " ".join(
            f"{name}={seconds / max(units, 1.0) * 1000:.2f}ms"
            for name, (seconds, units) in items
        )

    source_member_selector = (
        make_wds_member_selector(args.image_keys, args.caption_keys)
        if args.select_source_files
        else None
    )

    def make_webdataset(source_urls: list[str]):
        dataset_inner = wds.WebDataset(
            source_urls,
            shardshuffle=False,
            handler=wds.warn_and_continue,
            select_files=source_member_selector,
        )
        if args.shuffle > 0:
            dataset_inner = dataset_inner.shuffle(args.shuffle)
        return dataset_inner

    def iter_source_samples():
        """Yield source samples, optionally waiting for and deleting rolling local shards."""
        if not (args.wait_local_source_shards or args.delete_local_source_shards_after_read):
            yield from make_webdataset(urls)
            return

        delete_root = Path(args.source_list_file).parent if args.source_list_file else Path.cwd()
        skip_suffix = args.local_source_skip_suffix or None
        for shard_url in urls:
            local = is_local_source_path(shard_url)
            if local and args.wait_local_source_shards:
                wait_started = time.time()
                ready = wait_for_local_source_path(
                    Path(shard_url),
                    ready_suffix=args.local_source_ready_suffix,
                    skip_suffix=skip_suffix,
                    timeout=args.local_source_wait_timeout,
                    poll_interval=args.local_source_poll_interval,
                )
                record_stage("source_shard_wait", time.time() - wait_started)
                if not ready:
                    # Sidecar gave up on this shard; clear its marker and move on.
                    print(f"[rank {rank}] skipping unavailable source shard: {shard_url}")
                    if skip_suffix:
                        Path(f"{shard_url}{skip_suffix}").unlink(missing_ok=True)
                    cleanup_empty_parents(Path(shard_url).parent, delete_root)
                    continue

            completed = False
            try:
                yield from make_webdataset([shard_url])
                completed = True
            finally:
                if completed and local and args.delete_local_source_shards_after_read:
                    source_path = Path(shard_url)
                    source_path.unlink(missing_ok=True)
                    Path(f"{source_path}{args.local_source_ready_suffix}").unlink(missing_ok=True)
                    purge_source_download_cache(source_path)
                    cleanup_empty_parents(source_path.parent, delete_root)

    # Source dataset — keep raw bytes through to GPU when --gpu-decode.
    dataset = iter_source_samples()

    # Resume state
    rstate_path = resume_path(output_dir, rank)
    rstate = load_resume(rstate_path) if args.resume else {}
    skip_to = max(args.skip_samples, int(rstate.get("n_seen", 0)))
    next_shard = max(args.shard_start_index, int(rstate.get("next_shard", args.shard_start_index)))
    if args.resume and rstate:
        print(f"[rank {rank}] resuming: skip_to={skip_to} next_shard={next_shard}")

    maxsize = int(args.shard_size_gb * (1024**3))
    started = time.time()
    n_seen = 0
    n_written = 0
    n_reserved = 0
    n_skipped = 0
    pending: list[dict[str, Any]] = []
    latent_shape: list[int] | None = None
    text_shape: list[int] | None = None

    encode_futures: set[concurrent.futures.Future[list[dict[str, Any]]]] = set()
    write_futures: set[concurrent.futures.Future[None]] = set()
    encode_executor = (
        concurrent.futures.ThreadPoolExecutor(max_workers=1) if args.async_flush else None
    )
    write_executor = (
        concurrent.futures.ThreadPoolExecutor(max_workers=1) if args.async_flush else None
    )

    def decode_jpeg_batch(jpeg_bytes_list: list[bytes]) -> torch.Tensor:
        """Decode JPEGs, retrying smaller chunks if DALI or nvJPEG rejects a batch."""
        nonlocal dali_failed
        if not jpeg_bytes_list:
            raise ValueError("JPEG decode batch must be non-empty.")
        batch_len = len(jpeg_bytes_list)

        if dali_decoder is not None and not dali_failed:
            try:
                return dali_decoder(jpeg_bytes_list)
            except Exception as exc:
                # nvJPEG/allocator failures on a GPU shared with PyTorch are systemic and
                # leave the DALI pipeline permanently invalid ("no longer valid"). Disable
                # DALI for the rest of the run and fall back to torchvision nvJPEG, instead
                # of thrashing on per-size retries + empty_cache() syncs (which tanks the
                # throughput to a crawl).
                dali_failed = True
                print(
                    f"[dali] decode failed ({type(exc).__name__}: {exc}); "
                    "disabling DALI, using torchvision nvJPEG for the rest of the run."
                )

        try:
            return gpu_decode_and_resize(jpeg_bytes_list, resolution=args.resolution, device=device)
        except Exception:
            if batch_len <= 1:
                raise
            mid = batch_len // 2
            return torch.cat(
                [
                    decode_jpeg_batch(jpeg_bytes_list[:mid]),
                    decode_jpeg_batch(jpeg_bytes_list[mid:]),
                ],
                dim=0,
            )

    def materialize_image_batch(current: list[dict[str, Any]]) -> torch.Tensor:
        """Convert prepared samples into a single image batch on `device`."""
        if args.gpu_decode:
            jpegs = [item["jpeg_bytes"] for item in current if item.get("jpeg_bytes") is not None]
            cpu_fallbacks = [item for item in current if item.get("jpeg_bytes") is None]

            def decode_jpegs(byte_list: list[bytes]) -> torch.Tensor:
                # Chunk only for DALI (its pipeline is built per batch_size); torchvision
                # nvJPEG already batches a whole list internally, so chunking it just adds
                # Python overhead + extra cats.
                if dali_decoder is not None and not dali_failed:
                    chunks = [
                        byte_list[start : start + args.dali_decode_batch]
                        for start in range(0, len(byte_list), args.dali_decode_batch)
                    ]
                    return torch.cat([decode_jpeg_batch(chunk) for chunk in chunks], dim=0)
                return decode_jpeg_batch(byte_list)

            def cpu_batch() -> torch.Tensor:
                cpu = torch.cat([item["image_tensor"] for item in cpu_fallbacks], dim=0)
                if pin_memory:
                    cpu = cpu.pin_memory()
                return cpu.to(device, non_blocking=True)

            # Fast paths: when only one source is present, no interleave is needed.
            if not cpu_fallbacks:
                return decode_jpegs(jpegs)
            if not jpegs:
                return cpu_batch()

            # Mixed batch: decode both, then stitch back into input order.
            jpeg_t = decode_jpegs(jpegs)
            cpu_t = cpu_batch()
            slices = []
            j_idx = 0
            c_idx = 0
            for item in current:
                if item.get("jpeg_bytes") is not None:
                    slices.append(jpeg_t[j_idx : j_idx + 1])
                    j_idx += 1
                else:
                    slices.append(cpu_t[c_idx : c_idx + 1])
                    c_idx += 1
            return torch.cat(slices, dim=0)
        # CPU path
        images = torch.cat([item["image_tensor"] for item in current], dim=0)
        if pin_memory:
            images = images.pin_memory()
        return images.to(device, non_blocking=True)

    def encode_batch_payloads(current: list[dict[str, Any]]) -> list[dict[str, Any]]:
        nonlocal n_written, n_skipped, latent_shape, text_shape
        try:
            stage_started = sync_stage_timer()
            images = materialize_image_batch(current)
            if args.channels_last:
                images = images.contiguous(memory_format=torch.channels_last)
            record_since("image_materialize", stage_started, len(current))

            captions = [item["caption"] for item in current]
            stage_started = sync_stage_timer()
            latents = encode_images_microbatched(vae, images, config, args.image_encode_batch)
            record_since("vae_encode", stage_started, len(current))

            stage_started = sync_stage_timer()
            text_embeddings, attention_masks = encode_text_microbatched(
                text_processor,
                text_encoder,
                captions,
                config,
                args.text_encode_batch,
            )
            record_since("text_encode", stage_started, len(current))
        except Exception as exc:
            print(f"[skip batch size={len(current)}] {type(exc).__name__}: {exc}")
            if len(current) > 1:
                payloads: list[dict[str, Any]] = []
                for item in current:
                    payloads.extend(encode_batch_payloads([item]))
                return payloads
            with stats_lock:
                n_skipped += 1
            return []

        if latent_shape is None:
            latent_shape = list(latents.shape[1:])
        if text_shape is None:
            text_shape = list(text_embeddings.shape[1:])

        stage_started = sync_stage_timer()
        if cpu_transfer_stager is not None:
            payload = cpu_transfer_stager.stage(
                items=current,
                latents=latents,
                text_embeddings=text_embeddings,
                attention_masks=attention_masks,
                latent_storage_dtype=latent_storage_dtype,
                text_storage_dtype=text_storage_dtype,
            )
        else:
            payload = {
                "items": current,
                "latents_np": storage_dtype_to_numpy(latents, latent_storage_dtype),
                "text_np": storage_dtype_to_numpy(text_embeddings, text_storage_dtype),
                "masks_np": batch_to_u8_numpy(attention_masks),
            }
        record_since("cpu_pack", stage_started, len(current), sync=cpu_transfer_stager is None)
        return [payload]

    def payload_arrays(payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        event = payload.get("copy_event")
        if event is not None:
            current = payload["items"]
            wait_started = time.time()
            event.synchronize()
            record_stage("cpu_transfer_wait", time.time() - wait_started, len(current))
            payload["copy_event"] = None
            payload["_cuda_sources"] = None
            return (
                payload["latents_cpu"].numpy(),
                payload["text_cpu"].numpy(),
                payload["masks_cpu"].numpy(),
            )
        return payload["latents_np"], payload["text_np"], payload["masks_np"]

    def write_batch_payloads(payloads: list[dict[str, Any]], sink: Any) -> None:
        nonlocal n_written
        encoder_fn = raw_bytes if args.raw_tensors else npy_bytes

        write_started = time.time()
        written_in_call = 0
        for payload in payloads:
            current = payload["items"]
            latents_np, text_np, masks_np = payload_arrays(payload)
            for index, item in enumerate(current):
                key = f"{item['global_index']:09d}"
                ext = "bin" if args.raw_tensors else "npy"
                sample = {
                    "__key__": key,
                    f"latents.{ext}": encoder_fn(latents_np[index]),
                    f"text.{ext}": encoder_fn(text_np[index]),
                    f"text_mask.{ext}": encoder_fn(masks_np[index]),
                    "caption.txt": item["caption"],
                    "meta.json": json.dumps(
                        {
                            "index": item["global_index"],
                            "source_key": item["source_key"],
                            "source": source_descriptor,
                            "source_glob": source_glob if not args.source_list_file else None,
                            "source_list_file": args.source_list_file,
                            "source_preset": args.source_preset,
                            "split": args.split,
                            "rank": rank,
                        }
                    ).encode("utf-8"),
                }
                sink.write(sample)
                written_in_call += 1
                with stats_lock:
                    n_written += 1
                    written = n_written
                    skipped = n_skipped
                if written % args.log_every == 0:
                    elapsed = max(time.time() - started, 1e-6)
                    print(
                        f"[rank {rank}] wrote {written:,}/{args.max_samples:,} "
                        f"skipped={skipped:,} rate={written / elapsed:.2f} samples/s"
                    )
                    if args.profile_stages:
                        print(f"[rank {rank}] stages {format_stage_stats()}")
        record_stage("tar_write", time.time() - write_started, written_in_call)

    def flush_batch(current: list[dict[str, Any]], sink: Any) -> None:
        write_batch_payloads(encode_batch_payloads(current), sink)

    def drain_writes(*, wait: bool) -> None:
        if not write_futures:
            return
        timeout = None if wait else 0
        done, _ = concurrent.futures.wait(
            write_futures,
            timeout=timeout,
            return_when=concurrent.futures.FIRST_COMPLETED,
        )
        for future in done:
            write_futures.remove(future)
            future.result()

    def submit_write(payloads: list[dict[str, Any]], sink: Any) -> None:
        if not payloads:
            return
        assert write_executor is not None
        while len(write_futures) >= args.async_flush_depth:
            drain_writes(wait=True)
        write_futures.add(write_executor.submit(write_batch_payloads, payloads, sink))
        drain_writes(wait=False)

    def drain_encodes(sink: Any, *, wait: bool) -> None:
        if not encode_futures:
            return
        timeout = None if wait else 0
        done, _ = concurrent.futures.wait(
            encode_futures,
            timeout=timeout,
            return_when=concurrent.futures.FIRST_COMPLETED,
        )
        for future in done:
            encode_futures.remove(future)
            submit_write(future.result(), sink)

    def submit_flush(current: list[dict[str, Any]], sink: Any) -> None:
        if encode_executor is None:
            flush_batch(current, sink)
            return
        while len(encode_futures) >= args.async_flush_depth:
            drain_encodes(sink, wait=True)
        encode_futures.add(encode_executor.submit(encode_batch_payloads, current))
        drain_encodes(sink, wait=False)
        drain_writes(wait=False)

    def flush_pending(sink: Any) -> None:
        nonlocal n_reserved, pending
        if not pending:
            return
        remaining = args.max_samples - n_reserved
        if remaining <= 0:
            pending = []
            return
        current = pending[:remaining]
        pending = pending[remaining:]
        n_reserved += len(current)
        submit_flush(current, sink)

    max_in_flight_chunks = max(1, args.num_workers * args.prefetch_factor)
    in_flight: set[concurrent.futures.Future[list[dict[str, Any]]]] = set()
    source_chunk: list[tuple[int, dict[str, Any]]] = []
    executor = (
        concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers)
        if args.num_workers > 0
        else None
    )

    def handle_prepared(prepared: dict[str, Any], sink: Any) -> None:
        nonlocal n_skipped
        if prepared.get("skip"):
            with stats_lock:
                n_skipped += 1
            print(
                f"[rank {rank}] skip index={prepared.get('global_index')} "
                f"{prepared.get('error_type')}: {prepared.get('error')}"
            )
            return
        if n_reserved + len(pending) >= args.max_samples:
            return
        pending.append(prepared)
        if len(pending) >= args.batch_size or n_reserved + len(pending) >= args.max_samples:
            flush_pending(sink)

    def handle_prepared_chunk(prepared_chunk: list[dict[str, Any]], sink: Any) -> None:
        for prepared in prepared_chunk:
            if n_reserved + len(pending) >= args.max_samples:
                break
            handle_prepared(prepared, sink)

    def drain_ready(sink: Any, *, wait: bool) -> None:
        if not in_flight:
            return
        timeout = None if wait else 0
        done, _ = concurrent.futures.wait(
            in_flight,
            timeout=timeout,
            return_when=concurrent.futures.FIRST_COMPLETED,
        )
        for future in done:
            in_flight.remove(future)
            handle_prepared_chunk(future.result(), sink)

    def submit_source_chunk(sink: Any) -> None:
        nonlocal source_chunk
        if not source_chunk:
            return
        chunk = source_chunk
        source_chunk = []
        assert executor is not None
        while len(in_flight) >= max_in_flight_chunks:
            drain_ready(sink, wait=True)
            if n_reserved + len(pending) >= args.max_samples:
                return

        def prepare_job() -> list[dict[str, Any]]:
            stage_started = time.time()
            prepared = prepare_source_chunk(
                chunk,
                image_keys=args.image_keys,
                caption_keys=args.caption_keys,
                resolution=args.resolution,
                gpu_decode=args.gpu_decode,
            )
            record_stage("source_prepare", time.time() - stage_started, len(chunk))
            return prepared

        in_flight.add(
            executor.submit(prepare_job)
        )
        drain_ready(sink, wait=False)

    def on_output_shard_closed(_path: str) -> None:
        nonlocal next_shard
        next_shard += 1

    # Cause-1 fix: read source shards on a background thread so the 12 GB tar I/O (the
    # disk read releases the GIL) overlaps GPU encode, instead of the main thread blocking
    # on next(source) between batches and starving the single encode queue. The main loop
    # just drains a bounded queue; "source_next" now measures queue-wait (starvation).
    _SOURCE_SENTINEL = object()
    source_queue: "queue.Queue[Any]" = queue.Queue(maxsize=max(1, args.source_read_queue))
    stop_reading = threading.Event()
    reader_error: list[BaseException] = []

    def source_reader() -> None:
        try:
            for sample in dataset:
                while not stop_reading.is_set():
                    try:
                        source_queue.put(sample, timeout=0.5)
                        break
                    except queue.Full:
                        continue
                else:
                    return  # stop requested; generator finally still deletes the shard
        except BaseException as exc:  # noqa: BLE001 - propagated to the main thread
            reader_error.append(exc)
        finally:
            try:
                source_queue.put(_SOURCE_SENTINEL, timeout=0.5)
            except queue.Full:
                pass

    reader_thread = threading.Thread(target=source_reader, name="source-reader", daemon=True)
    reader_thread.start()

    try:
        with UploadingShardWriter(
            output_dir=output_dir,
            split=args.split,
            maxsize=maxsize,
            start_shard=next_shard,
            uploader=uploader,
            rank_tag=rank_tag,
            on_shard_close=on_output_shard_closed,
            output_format=args.output_format,
            maxcount=args.parquet_max_count,
            row_group_size=args.parquet_row_group_size,
            compression=args.parquet_compression,
        ) as sink:
            while True:
                source_started = time.time()
                sample = source_queue.get()
                record_stage("source_next", time.time() - source_started)
                if sample is _SOURCE_SENTINEL:
                    break
                if n_seen < skip_to:
                    n_seen += 1
                    continue
                if n_reserved + len(pending) >= args.max_samples:
                    break

                global_index = n_seen
                n_seen += 1
                if executor is None:
                    prepared = (prepare_source_sample_gpu if args.gpu_decode else prepare_source_sample_cpu)(
                        sample,
                        image_keys=args.image_keys,
                        caption_keys=args.caption_keys,
                        resolution=args.resolution,
                        global_index=global_index,
                    )
                    handle_prepared(prepared, sink)
                else:
                    source_chunk.append((global_index, sample))
                    if len(source_chunk) >= args.source_chunk_size:
                        submit_source_chunk(sink)

                # Periodically checkpoint resume state.
                if n_seen % max(args.log_every, 1) == 0:
                    save_resume(rstate_path, {"n_seen": n_seen, "next_shard": next_shard})

            if executor is not None:
                submit_source_chunk(sink)
            while in_flight and n_reserved + len(pending) < args.max_samples:
                drain_ready(sink, wait=True)
            flush_pending(sink)
            while encode_futures:
                drain_encodes(sink, wait=True)
            while write_futures:
                drain_writes(wait=True)
            if reader_error:
                raise reader_error[0]
    finally:
        stop_reading.set()
        reader_thread.join(timeout=10)
        if executor is not None:
            for future in in_flight:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)
        if encode_executor is not None:
            for future in encode_futures:
                future.cancel()
            encode_executor.shutdown(wait=True, cancel_futures=True)
        if write_executor is not None:
            for future in write_futures:
                future.cancel()
            write_executor.shutdown(wait=True, cancel_futures=True)
        save_resume(rstate_path, {"n_seen": n_seen, "next_shard": next_shard})

    elapsed = time.time() - started
    manifest = {
        "format_version": 2,
        "format": "boomer_wds_latent_text_cache",
        "text_encoder_kind": args.text_encoder_kind,
        "text_encoder_id": args.text_encoder_id,
        "qwen_readout_layer": args.qwen_readout_layer if args.text_encoder_kind == "qwen" else None,
        "max_length": args.max_length,
        "source": source_descriptor,
        "source_glob": source_glob,
        "source_list_file": args.source_list_file,
        "source_preset": args.source_preset,
        "split": args.split,
        "rank": rank,
        "world_size": world_size,
        "written_samples": n_written,
        "skipped_samples": n_skipped,
        "seen_source_samples": n_seen,
        "max_samples": args.max_samples,
        "skip_samples": args.skip_samples,
        "batch_size": args.batch_size,
        "image_encode_batch": args.image_encode_batch,
        "text_encode_batch": args.text_encode_batch,
        "source_prefetch_workers": args.num_workers,
        "source_prefetch_factor": args.prefetch_factor,
        "source_chunk_size": args.source_chunk_size,
        "select_source_files": args.select_source_files,
        "pin_memory": pin_memory,
        "async_flush": args.async_flush,
        "async_flush_depth": args.async_flush_depth,
        "async_cpu_transfer": args.async_cpu_transfer,
        "gpu_decode": args.gpu_decode,
        "dali_decode": args.dali_decode,
        "dali_num_threads": args.dali_num_threads if args.dali_decode else None,
        "dali_decode_batch": args.dali_decode_batch if args.dali_decode else None,
        "dali_device_memory_padding": (
            args.dali_device_memory_padding if args.dali_decode else None
        ),
        "dali_host_memory_padding": args.dali_host_memory_padding if args.dali_decode else None,
        "torch_compile": args.compile,
        "compile_mode": args.compile_mode if args.compile else None,
        "channels_last": args.channels_last,
        "raw_tensors": args.raw_tensors,
        "shard_size_gb": args.shard_size_gb,
        "output_format": args.output_format,
        "column_encoding": "npy_bytes" if args.output_format == "parquet" else None,
        "parquet_row_group_size": args.parquet_row_group_size if args.output_format == "parquet" else None,
        "parquet_compression": args.parquet_compression if args.output_format == "parquet" else None,
        "schema": (
            ["latents.bin", "text.bin", "text_mask.bin", "caption.txt", "meta.json"]
            if args.raw_tensors
            else ["latents.npy", "text.npy", "text_mask.npy", "caption.txt", "meta.json"]
        ),
        "latents_dtype": latent_storage_dtype,
        "latents_physical_dtype": "uint16" if latent_storage_dtype == "bfloat16" else latent_storage_dtype,
        "latents_shape": latent_shape,
        "text_dtype": text_storage_dtype,
        "text_physical_dtype": (
            "uint16"
            if text_storage_dtype == "bfloat16"
            else "uint8"
            if text_storage_dtype == "float8_e4m3fn"
            else text_storage_dtype
        ),
        "text_shape": text_shape,
        "text_mask_dtype": "uint8",
        "cache_config": asdict(config),
        "null_text_path": "null_text.npy",
        "null_mask_path": "null_mask.npy",
        "image_normalization": "uint8 -> [0,1] -> [-1,1]",
        "elapsed_sec": elapsed,
        "samples_per_sec": n_written / max(elapsed, 1e-6),
    }
    manifest_name = f"manifest-{rank_tag}.json" if rank_tag else "manifest.json"
    manifest_path = output_dir / manifest_name
    write_json(manifest_path, manifest)
    uploader.upload_path(manifest_path)
    uploader.finish()

    print(f"[rank {rank}] Cache complete.")
    print(f"  samples={n_written:,} skipped={n_skipped:,}")
    print(f"  output_dir={output_dir}")
    print(f"  elapsed_min={elapsed / 60:.1f}")
    print(f"  samples_per_sec={manifest['samples_per_sec']:.2f}")
    if args.hf_repo_id:
        print(f"  uploaded_repo={args.hf_repo_id}")
        print(f"  uploaded_prefix={repo_prefix}")


if __name__ == "__main__":
    main()
