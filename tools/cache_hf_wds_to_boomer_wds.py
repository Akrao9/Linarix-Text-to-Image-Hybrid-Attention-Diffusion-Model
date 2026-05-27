"""Stream HF WebDataset image/caption shards into Boomer latent WebDataset shards.

Optimized variant. Drop-in replacement for cache_latents.py with the Boomer
latent WDS schema (latents/text/text_mask + caption.txt + meta.json) and
manifest format, but with the following speed knobs:

  --gpu-decode            Batch-decode JPEGs on GPU via nvJPEG (torchvision).
                          Falls back to PIL for PNG/WEBP/corrupt JPEGs.
  --compile               torch.compile the VAE and text encoder. fixed shapes
                          only — keeps --batch-size and --resolution constant.
  --compile-mode          "default" | "reduce-overhead" | "max-autotune".
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


def tensor_to_storage_npy(tensor: torch.Tensor, dtype_name: str) -> bytes:
    return npy_bytes(storage_dtype_to_numpy(tensor, dtype_name))


def tensor_to_u8_npy(tensor: torch.Tensor) -> bytes:
    return npy_bytes(tensor.detach().cpu().to(torch.uint8).numpy())


def batch_to_u8_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().to(torch.uint8).cpu().numpy()


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def split_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


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
        header = shlex.quote(f"Authorization:Bearer {resolved_token}")
        return [f"pipe:curl -sSL --fail --retry 5 --retry-delay 2 -H {header} {shlex.quote(url)}" for url in urls]
    return urls


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
    # decode_jpeg with a list returns a list of CHW uint8 tensors of varying H/W.
    # Resize each to (resolution, resolution), then stack.
    resized = []
    for img in decoded:
        # Match the Sana CPU path geometry: resize-to-fill, then center crop.
        # The interpolation kernel differs from PIL bicubic, but aspect ratio
        # and crop placement now match resize_crop_to_fill_image().
        _, height, width = img.shape
        scale = resolution / min(height, width)
        new_h = round(height * scale)
        new_w = round(width * scale)
        x = img.unsqueeze(0).to(torch.float32)
        x = F.interpolate(
            x,
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        top = max((new_h - resolution) // 2, 0)
        left = max((new_w - resolution) // 2, 0)
        x = x[:, :, top : top + resolution, left : left + resolution]
        resized.append(x)
    batch = torch.cat(resized, dim=0)
    # uint8 [0,255] -> [-1, 1]
    batch = batch.mul_(2.0 / 255.0).sub_(1.0)
    return batch


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
    ) -> None:
        self.repo_id = repo_id
        self.repo_path_prefix = repo_path_prefix.strip("/")
        self.delete_uploaded = delete_uploaded
        self._uploaded: set[str] = set()
        self._futures: dict[str, concurrent.futures.Future] = {}
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
            if path in self._uploaded or path in self._futures:
                return
            assert self._executor is not None
            self._futures[path] = self._executor.submit(self._upload_path_sync, path)
        print(f"[upload] queued {Path(path).name}")

    def _upload_path_sync(self, local_path: str) -> None:
        assert self._api is not None and self.repo_id is not None
        repo_path = (
            f"{self.repo_path_prefix}/{Path(local_path).name}"
            if self.repo_path_prefix
            else Path(local_path).name
        )
        self._api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=repo_path,
            repo_id=self.repo_id,
            repo_type="dataset",
        )
        if self.delete_uploaded:
            try:
                os.remove(local_path)
            except FileNotFoundError:
                pass

    def poll(self, *, wait: bool = False) -> None:
        if not self.enabled:
            return
        if wait and self._futures:
            concurrent.futures.wait(
                list(self._futures.values()),
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
        done_paths = []
        with self._lock:
            for path, future in list(self._futures.items()):
                if future.done():
                    done_paths.append(path)
        for path in done_paths:
            future = self._futures.pop(path)
            future.result()
            self._uploaded.add(path)
            print(f"[upload] done   {Path(path).name}")

    def finish(self) -> None:
        if not self.enabled:
            return
        while self._futures:
            self.poll(wait=True)
        assert self._executor is not None
        self._executor.shutdown(wait=True)


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
    ) -> None:
        import webdataset as wds

        self.output_dir = Path(output_dir)
        self.split = split
        self.uploader = uploader
        self.on_shard_close = on_shard_close
        self.closed_paths: list[Path] = []
        suffix = f"-{rank_tag}" if rank_tag else ""
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
        return self._writer.__enter__()

    def __exit__(self, exc_type, exc, tb):
        return self._writer.__exit__(exc_type, exc, tb)


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
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--async-flush", action="store_true")
    parser.add_argument("--async-flush-depth", type=int, default=2)
    parser.add_argument("--shard-size-gb", type=float, default=2.0)
    parser.add_argument("--shard-start-index", type=int, default=0)
    parser.add_argument("--image-keys", type=split_csv, default=["jpg", "jpeg", "png", "webp", "image"])
    parser.add_argument(
        "--caption-keys",
        type=split_csv,
        default=["txt", "text", "caption", "prompt", "json", "metadata"],
    )
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--vae-id", default=DEFAULT_VAE_ID)
    parser.add_argument("--text-encoder-id", default=DEFAULT_TEXT_ENCODER_ID)
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
        "--compile",
        action="store_true",
        help="torch.compile the VAE and text encoder.",
    )
    parser.add_argument(
        "--compile-mode",
        default="reduce-overhead",
        choices=["default", "reduce-overhead", "max-autotune"],
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

    source_glob = SOURCE_PRESETS[args.source_preset] if args.source_preset else args.source_glob
    latent_storage_dtype = normalize_storage_dtype(args.latent_storage_dtype or args.storage_dtype)
    text_storage_dtype = normalize_storage_dtype(args.text_storage_dtype or args.storage_dtype)

    hf_token = hf_token_value(args.hf_token)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if rank == 0:
        print("Resolving source tar URLs...")
    urls_all = hf_wds_urls(source_glob, hf_token)
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
            f"gpu_decode={args.gpu_decode} compile={args.compile} "
            f"channels_last={args.channels_last} latent_dtype={latent_storage_dtype} "
            f"text_dtype={text_storage_dtype} "
            f"raw_tensors={args.raw_tensors}"
        )

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
    text_processor, text_encoder = load_text_encoder(config, hf_token=hf_token)

    if args.channels_last:
        try:
            vae = vae.to(memory_format=torch.channels_last)
        except Exception as exc:
            print(f"[rank {rank}] channels_last on VAE failed, continuing: {exc}")

    if args.compile:
        # Best-effort compile. encode_image_latent / encode_captions wrap the
        # module-level forwards, so we compile the submodule, not the wrappers.
        try:
            vae_compile_target = getattr(vae, "encoder", vae)
            compiled = torch.compile(vae_compile_target, mode=args.compile_mode, fullgraph=False)
            if hasattr(vae, "encoder"):
                vae.encoder = compiled
            else:
                vae = compiled
            print(f"[rank {rank}] torch.compile(vae, mode={args.compile_mode}) applied.")
        except Exception as exc:
            print(f"[rank {rank}] torch.compile on VAE failed, continuing: {exc}")
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

    # Source dataset — keep raw bytes through to GPU when --gpu-decode.
    dataset = wds.WebDataset(urls, shardshuffle=False, handler=wds.warn_and_continue)
    if args.shuffle > 0:
        dataset = dataset.shuffle(args.shuffle)

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

    stats_lock = threading.Lock()
    flush_futures: set[concurrent.futures.Future[None]] = set()
    flush_executor = (
        concurrent.futures.ThreadPoolExecutor(max_workers=1) if args.async_flush else None
    )

    def materialize_image_batch(current: list[dict[str, Any]]) -> torch.Tensor:
        """Convert prepared samples into a single image batch on `device`."""
        if args.gpu_decode:
            jpegs = [item["jpeg_bytes"] for item in current if item.get("jpeg_bytes") is not None]
            cpu_fallbacks = [item for item in current if item.get("jpeg_bytes") is None]
            tensors: list[torch.Tensor] = []
            if jpegs:
                tensors.append(
                    gpu_decode_and_resize(jpegs, resolution=args.resolution, device=device)
                )
            if cpu_fallbacks:
                cpu = torch.cat([item["image_tensor"] for item in cpu_fallbacks], dim=0)
                if pin_memory:
                    cpu = cpu.pin_memory()
                tensors.append(cpu.to(device, non_blocking=True))
            # Reorder back to input order so captions line up.
            # Easiest: rebuild in original order by indexing.
            order = []
            j_idx = 0
            c_idx = 0
            jpeg_count = len(jpegs)
            cpu_count = len(cpu_fallbacks)
            for item in current:
                if item.get("jpeg_bytes") is not None:
                    order.append(("j", j_idx))
                    j_idx += 1
                else:
                    order.append(("c", c_idx))
                    c_idx += 1
            jpeg_t = tensors[0] if jpeg_count > 0 else None
            cpu_t = tensors[-1] if cpu_count > 0 and len(tensors) > (1 if jpeg_count > 0 else 0) else (
                tensors[0] if cpu_count > 0 and jpeg_count == 0 else None
            )
            slices = []
            for kind, idx in order:
                if kind == "j":
                    assert jpeg_t is not None
                    slices.append(jpeg_t[idx : idx + 1])
                else:
                    assert cpu_t is not None
                    slices.append(cpu_t[idx : idx + 1])
            return torch.cat(slices, dim=0)
        # CPU path
        images = torch.cat([item["image_tensor"] for item in current], dim=0)
        if pin_memory:
            images = images.pin_memory()
        return images.to(device, non_blocking=True)

    def flush_batch(current: list[dict[str, Any]], sink: Any) -> None:
        nonlocal n_written, n_skipped, latent_shape, text_shape
        try:
            images = materialize_image_batch(current)
            if args.channels_last:
                images = images.contiguous(memory_format=torch.channels_last)
            captions = [item["caption"] for item in current]
            latents = encode_images_microbatched(vae, images, config, args.image_encode_batch)
            text_embeddings, attention_masks = encode_text_microbatched(
                text_processor,
                text_encoder,
                captions,
                config,
                args.text_encode_batch,
            )
        except Exception as exc:
            print(f"[skip batch size={len(current)}] {type(exc).__name__}: {exc}")
            if len(current) > 1:
                for item in current:
                    flush_batch([item], sink)
                return
            with stats_lock:
                n_skipped += 1
            return

        if latent_shape is None:
            latent_shape = list(latents.shape[1:])
        if text_shape is None:
            text_shape = list(text_embeddings.shape[1:])

        latents_np = storage_dtype_to_numpy(latents, latent_storage_dtype)
        text_np = storage_dtype_to_numpy(text_embeddings, text_storage_dtype)
        masks_np = batch_to_u8_numpy(attention_masks)

        encoder_fn = raw_bytes if args.raw_tensors else npy_bytes

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
                        "source_glob": source_glob,
                        "source_preset": args.source_preset,
                        "split": args.split,
                        "rank": rank,
                    }
                ).encode("utf-8"),
            }
            sink.write(sample)
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

    def drain_flushes(*, wait: bool) -> None:
        if not flush_futures:
            return
        timeout = None if wait else 0
        done, _ = concurrent.futures.wait(
            flush_futures,
            timeout=timeout,
            return_when=concurrent.futures.FIRST_COMPLETED,
        )
        for future in done:
            flush_futures.remove(future)
            future.result()

    def submit_flush(current: list[dict[str, Any]], sink: Any) -> None:
        if flush_executor is None:
            flush_batch(current, sink)
            return
        while len(flush_futures) >= args.async_flush_depth:
            drain_flushes(wait=True)
        flush_futures.add(flush_executor.submit(flush_batch, current, sink))
        drain_flushes(wait=False)

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
        in_flight.add(
            executor.submit(
                prepare_source_chunk,
                chunk,
                image_keys=args.image_keys,
                caption_keys=args.caption_keys,
                resolution=args.resolution,
                gpu_decode=args.gpu_decode,
            )
        )
        drain_ready(sink, wait=False)

    def on_output_shard_closed(_path: str) -> None:
        nonlocal next_shard
        next_shard += 1

    try:
        with UploadingShardWriter(
            output_dir=output_dir,
            split=args.split,
            maxsize=maxsize,
            start_shard=next_shard,
            uploader=uploader,
            rank_tag=rank_tag,
            on_shard_close=on_output_shard_closed,
        ) as sink:
            for sample in dataset:
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
            while flush_futures:
                drain_flushes(wait=True)
    finally:
        if executor is not None:
            for future in in_flight:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)
        if flush_executor is not None:
            for future in flush_futures:
                future.cancel()
            flush_executor.shutdown(wait=True, cancel_futures=True)
        save_resume(rstate_path, {"n_seen": n_seen, "next_shard": next_shard})

    elapsed = time.time() - started
    manifest = {
        "format_version": 2,
        "format": "boomer_wds_latent_text_cache",
        "source_glob": source_glob,
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
        "pin_memory": pin_memory,
        "async_flush": args.async_flush,
        "async_flush_depth": args.async_flush_depth,
        "gpu_decode": args.gpu_decode,
        "torch_compile": args.compile,
        "compile_mode": args.compile_mode if args.compile else None,
        "channels_last": args.channels_last,
        "raw_tensors": args.raw_tensors,
        "shard_size_gb": args.shard_size_gb,
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
