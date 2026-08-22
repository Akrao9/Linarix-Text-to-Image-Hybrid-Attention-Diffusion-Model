"""Stream MONET's full-resolution WebDataset images for on-the-fly training.

Unlike :mod:`boomer.datasets.monet_direct`, which reads MONET's Parquet
thumbnails and precomputed 512px latents, this loader reads the raw ``jpg`` and
``json`` members from MONET's tar shards.  Pixels are resized/cropped here and
encoded by the resident frozen DC-AE in ``train.py``; captions are encoded by
the resident Qwen text encoder.

MONET samples also contain several large ``*.pth`` embeddings.  The tar stream
is deliberately left *undecoded*: only ``jpg``/``json`` are decoded, so CLIP,
SSCD, VAE, and DINO files are never materialized unless the optional DINO REPA
target is explicitly requested.
"""

from __future__ import annotations

import glob as glob_module
import io
import random
from dataclasses import dataclass, field
from typing import Any, Iterator

import numpy as np
import torch

from boomer.datasets.fine_t2i import (
    _EPOCH_END,
    FineT2IStreamingDataset,
    _as_json,
    _square_pixels,
    _to_image,
)
from boomer.datasets.monet_direct import MONET_CAPTION_KEYS

DEFAULT_MONET_IMAGE_PATHS = "hf://datasets/jasperai/monet/v1.2.0/**/*.tar"


@dataclass(frozen=True)
class MonetImageRecord:
    pixels: torch.Tensor
    caption: str
    metadata: dict[str, Any]
    dino_embedding: torch.Tensor | None = None


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _select_caption(
    meta: dict[str, Any],
    *,
    caption_keys: tuple[str, ...],
    caption_probs: tuple[float, ...] | None,
    rng: random.Random | None,
) -> tuple[str, str | None]:
    if caption_probs is None:
        for key in caption_keys:
            value = meta.get(key)
            if value is not None and str(value).strip():
                return str(value).strip(), key
        return "", None

    pool = [
        (key, weight)
        for key, weight in zip(caption_keys, caption_probs)
        if weight > 0 and meta.get(key) is not None and str(meta[key]).strip()
    ]
    if not pool:
        return "", None
    chooser = rng if rng is not None else random
    key = chooser.choices(
        [name for name, _ in pool], weights=[weight for _, weight in pool], k=1
    )[0]
    return str(meta[key]).strip(), key


def _decode_dino(value: Any) -> torch.Tensor | None:
    """Decode MONET's DINO ``.pth`` member only when REPA requests it."""
    if value is None:
        return None
    try:
        if isinstance(value, (bytes, bytearray, memoryview)):
            value = torch.load(
                io.BytesIO(bytes(value)), map_location="cpu", weights_only=True
            )
        if isinstance(value, dict):
            value = next((item for item in value.values() if torch.is_tensor(item)), None)
        if value is None:
            return None
        if torch.is_tensor(value):
            return value.detach().float().reshape(-1)
        return torch.from_numpy(np.asarray(value, dtype=np.float32)).reshape(-1)
    except (RuntimeError, TypeError, ValueError, EOFError):
        return None


def monet_image_example_to_record(
    example: dict[str, Any],
    *,
    resolution: int = 1024,
    caption_keys: tuple[str, ...] = MONET_CAPTION_KEYS,
    caption_probs: tuple[float, ...] | None = None,
    rng: random.Random | None = None,
    min_short_side: int = 0,
    min_aesthetic: float | None = None,
    max_nsfw: float | None = None,
    max_watermark: float | None = None,
    dino_key: str | None = None,
) -> MonetImageRecord | None:
    """Convert one raw MONET WDS sample to square uint8 pixels + caption."""
    meta = _as_json(example.get("json"))

    aesthetic = _safe_float(meta.get("aesthetic_laion-improved"))
    nsfw = _safe_float(meta.get("nsfw_jasperai"))
    watermark = _safe_float(meta.get("wk_jasperai"))
    if min_aesthetic is not None and (aesthetic is None or aesthetic < min_aesthetic):
        return None
    if max_nsfw is not None and nsfw is not None and nsfw > max_nsfw:
        return None
    if max_watermark is not None and watermark is not None and watermark > max_watermark:
        return None

    caption, caption_key = _select_caption(
        meta,
        caption_keys=caption_keys,
        caption_probs=caption_probs,
        rng=rng,
    )
    if not caption:
        return None

    # Metadata lets low-resolution samples be rejected without JPEG decode.
    width = _safe_float(meta.get("width"))
    height = _safe_float(meta.get("height"))
    if min_short_side and width and height and min(width, height) < min_short_side:
        return None

    image = _to_image(example.get("jpg"))
    if image is None:
        return None
    try:
        image = image.convert("RGB")
        native_size = tuple(image.size)
        if min_short_side and min(native_size) < min_short_side:
            return None
        pixels = _square_pixels(image, resolution)
    except Exception as exc:  # noqa: BLE001 - one corrupt JPEG must not kill a run
        print(f"[monet-image-direct] drop undecodable image: {exc}", flush=True)
        return None

    dino = None
    if dino_key is not None:
        # group_by_keys normally retains the complete suffix. Tolerate variants for
        # local/mocked WebDataset readers as well.
        dino = _decode_dino(
            example.get(f"{dino_key}.pth", example.get(dino_key))
        )

    return MonetImageRecord(
        pixels=pixels,
        caption=caption,
        metadata={
            "key": str(example.get("__key__", "")),
            "source": meta.get("source"),
            "size": native_size,
            "caption_key": caption_key,
        },
        dino_embedding=dino,
    )


def collate_monet_image_records(records: list[MonetImageRecord]) -> dict[str, Any]:
    dinos = [record.dino_embedding for record in records]
    dino_batch = (
        torch.stack(dinos, dim=0)
        if dinos and all(dino is not None for dino in dinos)
        else None
    )
    return {
        "pixels": torch.stack([record.pixels for record in records], dim=0),
        "caption": [record.caption for record in records],
        "metadata": [record.metadata for record in records],
        "dino_embedding": dino_batch,
    }


@dataclass
class MonetImageStreamingDataset(FineT2IStreamingDataset):
    """Weighted, distributed MONET tar streaming with native-image decode."""

    paths: str = DEFAULT_MONET_IMAGE_PATHS
    log_tag: str = "monet-image-direct"
    caption_keys: tuple[str, ...] = MONET_CAPTION_KEYS
    caption_probs: Any = None
    glob_weights: Any = None
    dino_key: str | None = None
    min_aesthetic: float | None = None
    max_nsfw: float | None = None
    max_watermark: float | None = None
    _glob_files_cache: dict[str, list[str]] = field(
        init=False, repr=False, default_factory=dict
    )
    _unmixed_sources: tuple[str, ...] = field(
        init=False, repr=False, default=()
    )

    def __post_init__(self) -> None:
        # Reuse FineT2I's worker/rank split, retry loop, and reservoir shuffle.
        self.data_dir = self.paths
        self.data_dir_weights = self.glob_weights
        self.data_files = None
        super().__post_init__()
        if len(self._sources) > 1 and self._weights is None:
            # Match monet-direct's unweighted behavior: flatten every glob into one
            # shard stream (natural proportions by shard/row count), rather than the
            # FineT2I base class's single-source path selecting only entry zero.
            self._unmixed_sources = self._sources
            self._sources = ()

        if isinstance(self.caption_keys, str):
            self.caption_keys = tuple(
                key.strip() for key in self.caption_keys.split(",") if key.strip()
            )

        def check(group: Any) -> tuple[float, ...]:
            values = tuple(float(value) for value in group)
            if len(values) != len(self.caption_keys):
                raise ValueError(
                    f"caption_probs group has {len(values)} weights for "
                    f"{len(self.caption_keys)} caption_keys"
                )
            if min(values) < 0 or sum(values) <= 0:
                raise ValueError(
                    "caption_probs must be non-negative with a positive sum"
                )
            return values

        if isinstance(self.caption_probs, str):
            if ";" in self.caption_probs:
                self.caption_probs = tuple(
                    tuple(float(v) for v in group.split(",") if v.strip()) or None
                    for group in self.caption_probs.split(";")
                )
            else:
                self.caption_probs = tuple(
                    float(v) for v in self.caption_probs.split(",") if v.strip()
                ) or None

        self._per_glob_caption_probs: tuple[tuple[float, ...] | None, ...] | None = None
        if self.caption_probs is not None:
            if all(
                group is None or isinstance(group, (tuple, list))
                for group in self.caption_probs
            ):
                self._per_glob_caption_probs = tuple(
                    check(group) if group is not None else None
                    for group in self.caption_probs
                )
            else:
                self.caption_probs = check(self.caption_probs)

        if self._per_glob_caption_probs is not None:
            if self._weights is None:
                raise ValueError("per-glob caption_probs require glob_weights")
            if len(self._per_glob_caption_probs) != len(self._sources):
                raise ValueError(
                    "caption_probs ';' groups must align 1:1 with paths globs"
                )
        self._active_caption_probs = (
            None if self._per_glob_caption_probs is not None else self.caption_probs
        )

    def _list_shard_files(self, source: str | None) -> list[str]:
        if source is None:
            files = [
                path
                for raw in self._unmixed_sources
                for path in self._list_shard_files(raw)
            ]
            if files:
                return files
            raise FileNotFoundError("MONET image streaming needs at least one tar glob")
        cached = self._glob_files_cache.get(source)
        if cached is not None:
            return cached
        if source.startswith("hf://"):
            matches = sorted(self._hf_filesystem().glob(source))
            files = [m if m.startswith("hf://") else f"hf://{m}" for m in matches]
        elif any(char in source for char in "*?[]"):
            files = sorted(glob_module.glob(source))
        else:
            files = [source]
        files = [path for path in files if path.endswith(".tar")]
        if not files:
            raise FileNotFoundError(f"No MONET tar shards found for: {source}")
        self._glob_files_cache[source] = files
        return files

    def prepare_shards(self) -> None:
        """Resolve every glob once in the parent before DataLoader workers spawn."""
        sources = self._unmixed_sources or self._sources
        for source in sources:
            self._list_shard_files(source)
        # HfFileSystem owns live HTTP clients/locks and should not be pickled into
        # spawned workers. The path lists above are plain strings and are retained.
        self._fs = None

    def _iter_shard(self, path: str) -> Iterator[MonetImageRecord | None]:
        """Group raw tar members without ``decode()`` (which would load every .pth)."""
        import webdataset as wds

        handle = None
        if path.startswith("hf://"):
            handle = self._open_hf_stream(path.removeprefix("hf://"))
            samples = wds.tariterators.group_by_keys(
                wds.tariterators.tar_file_expander(
                    [{"url": path, "stream": handle}]
                )
            )
        elif self._is_repo_relative(path):
            handle = self._open_hf_stream(f"datasets/{self.repo_id}/{path}")
            samples = wds.tariterators.group_by_keys(
                wds.tariterators.tar_file_expander(
                    [{"url": path, "stream": handle}]
                )
            )
        else:
            samples = wds.WebDataset(
                [path],
                shardshuffle=False,
                nodesplitter=None,
                workersplitter=None,
            )
        try:
            yield from self._records_from_samples(samples)
        finally:
            if handle is not None:
                handle.close()

    def _records_from_samples(
        self, samples: Any
    ) -> Iterator[MonetImageRecord | None]:
        for sample in samples:
            example = dict(sample)
            if "jpg" not in example:
                for alt in ("jpeg", "png"):
                    if alt in example:
                        example["jpg"] = example[alt]
                        break
            yield monet_image_example_to_record(
                example,
                resolution=self.resolution,
                caption_keys=tuple(self.caption_keys),
                caption_probs=self._active_caption_probs,
                rng=self._caption_rng,
                min_short_side=self.min_short_side,
                min_aesthetic=self.min_aesthetic,
                max_nsfw=self.max_nsfw,
                max_watermark=self.max_watermark,
                dino_key=self.dino_key,
            )

    def _interleave_sources(
        self, *, offset: int, stride: int, rng: random.Random
    ) -> Iterator[MonetImageRecord]:
        streams: list[Iterator[Any]] = []
        assert self._weights is not None
        for index, source in enumerate(self._sources):
            files = self._cached_shard_files(source)
            mine = self._split_files(files, offset, stride, tag=source)
            print(
                f"[monet-image-direct] mix[{index}] w={self._weights[index]:g} "
                f"{len(mine)}/{len(files)} shard(s) {source}",
                flush=True,
            )
            streams.append(
                self._stream_files_forever(
                    mine,
                    seed_offset=offset + 7919 * (index + 1),
                    tag=source,
                )
            )
        indices = list(range(len(streams)))
        while True:
            index = rng.choices(indices, weights=self._weights, k=1)[0]
            if self._per_glob_caption_probs is not None:
                self._active_caption_probs = self._per_glob_caption_probs[index]
            record = next(streams[index])
            if record is _EPOCH_END:
                record = next(streams[index])
            yield record


__all__ = [
    "DEFAULT_MONET_IMAGE_PATHS",
    "MonetImageRecord",
    "MonetImageStreamingDataset",
    "collate_monet_image_records",
    "monet_image_example_to_record",
]
