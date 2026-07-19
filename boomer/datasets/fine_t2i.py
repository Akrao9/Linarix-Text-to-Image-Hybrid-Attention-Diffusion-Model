"""Stream a single-image text-to-image corpus directly for on-the-fly training.

Built for ma-xu/fine-t2i (--t2i-direct): a **WebDataset** (tar shards of
``jpg`` + ``txt`` + ``json`` per sample), ~6M synthetic + curated real images,
mostly >1K native resolution and randomized aspect ratios. There are NO latents
in the corpus, so training encodes pixels on the fly through the frozen DC-AE
on-GPU (train.py ``encode_pixels``) and the caption through Qwen (same as
--monet-direct text). This avoids building a multi-TB parquet latent cache.

Why HF ``datasets`` streaming instead of the repo's parquet range machinery:
fine-t2i is tar/WebDataset, not parquet, so ``EditPairsStreamingDataset``'s
pyarrow inheritance does not apply. ``datasets.load_dataset(streaming=True)``
decodes the tar/jpg/json natively, streams over HTTP, and shards cleanly across
DDP nodes (``split_dataset_by_node``) and DataLoader workers (``.shard``).

Per-row caption policy (``caption_mode``):
  * ``sample`` — per-row random pick between the SHORT caption (``json.prompt``,
    falling back to ``txt``) and the ENHANCED caption (``json.enhanced_prompt``),
    with ``short_prob`` weighting. Trains both concise and richly-detailed
    conditioning so inference generalizes to varied prompt lengths.
  * ``short`` / ``enhanced`` — always that field (with the same fallbacks).

Each image is resized short-side -> ``resolution`` and center-cropped to a
square ``resolution`` x ``resolution`` (randomized aspect ratios collapse to the
square canvas the DiT's RoPE grid expects). Records are ~3 MB each
(3 x 1024^2 uint8), so train.py clamps the shuffle buffer in t2i-direct mode.
"""

from __future__ import annotations

import io
import json as _json
import random
from dataclasses import dataclass, field
from itertools import islice
from typing import Any, Iterator

import numpy as np
import torch
from PIL import Image
from torch.utils.data import IterableDataset, get_worker_info

DEFAULT_T2I_REPO = "ma-xu/fine-t2i"
DEFAULT_IMAGE_KEY = "jpg"
DEFAULT_TEXT_KEY = "txt"
DEFAULT_JSON_KEY = "json"
# json.* fields carrying the short caption (priority order) and the enhanced one.
SHORT_PROMPT_FIELDS = ("prompt",)
ENHANCED_PROMPT_FIELDS = ("enhanced_prompt",)
AESTHETIC_FIELD = "aesthetic_predictor_v_2_5_score"

_CAPTION_MODES = ("sample", "short", "enhanced")


@dataclass(frozen=True)
class T2IRecord:
    """One t2i sample: square uint8 CHW pixels + caption."""

    pixels: torch.Tensor  # (3, R, R) uint8
    caption: str
    metadata: dict[str, Any]


def _as_json(value: Any) -> dict[str, Any]:
    """The webdataset ``.json`` member as a dict (datasets may hand back a dict or
    the raw JSON string / bytes)."""
    if isinstance(value, dict):
        return value
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8", "replace")
    if isinstance(value, str) and value.strip():
        try:
            parsed = _json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except ValueError:
            return {}
    return {}


def _first_field(meta: dict[str, Any], fields: tuple[str, ...]) -> str:
    for key in fields:
        val = meta.get(key)
        if val is not None and str(val).strip():
            return str(val).strip()
    return ""


def _to_image(value: Any) -> Image.Image | None:
    """The ``jpg`` member as a PIL image. datasets' Image feature usually yields a
    PIL image already; tolerate an {bytes,path} struct or raw bytes too."""
    if isinstance(value, Image.Image):
        return value
    if isinstance(value, dict):
        value = value.get("bytes")
    if isinstance(value, (bytes, bytearray)):
        try:
            return Image.open(io.BytesIO(value))
        except Exception:  # noqa: BLE001
            return None
    return None


def _square_pixels(image: Image.Image, resolution: int) -> torch.Tensor:
    """Resize short side to ``resolution`` and center-crop to a square canvas."""
    width, height = image.size
    if (width, height) != (resolution, resolution):
        scale = resolution / min(width, height)
        new_w, new_h = max(resolution, round(width * scale)), max(resolution, round(height * scale))
        image = image.resize((new_w, new_h), Image.BICUBIC)
        left = (new_w - resolution) // 2
        top = (new_h - resolution) // 2
        image = image.crop((left, top, left + resolution, top + resolution))
    # np.array (copy): PIL hands out a read-only buffer, and torch.from_numpy on a
    # non-writable array is undefined behavior (and warns).
    array = np.array(image, dtype=np.uint8)
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def t2i_example_to_record(
    example: dict[str, Any],
    *,
    resolution: int = 1024,
    image_key: str = DEFAULT_IMAGE_KEY,
    text_key: str = DEFAULT_TEXT_KEY,
    json_key: str = DEFAULT_JSON_KEY,
    caption_mode: str = "sample",
    short_prob: float = 0.5,
    min_short_side: int = 0,
    min_aesthetic: float | None = None,
    rng: random.Random | None = None,
) -> T2IRecord | None:
    """Decode one fine-t2i sample, or None when it fails a filter / is unusable."""
    meta = _as_json(example.get(json_key))

    if min_aesthetic is not None:
        score = meta.get(AESTHETIC_FIELD)
        if score is None or float(score) < min_aesthetic:
            return None

    short = _first_field(meta, SHORT_PROMPT_FIELDS)
    if not short:
        txt = example.get(text_key)
        short = str(txt).strip() if txt is not None else ""
    enhanced = _first_field(meta, ENHANCED_PROMPT_FIELDS) or short
    if caption_mode == "short":
        caption = short
    elif caption_mode == "enhanced":
        caption = enhanced or short
    else:  # sample
        chooser = rng if rng is not None else random
        caption = short if (chooser.random() < short_prob and short) else (enhanced or short)
    if not caption:
        return None

    image = _to_image(example.get(image_key))
    if image is None:
        return None
    try:
        image = image.convert("RGB")
    except Exception as exc:  # noqa: BLE001 - one corrupt image must not kill the stream
        print(f"[t2i-direct] drop undecodable image: {exc}", flush=True)
        return None
    if min_short_side and min(image.size) < min_short_side:
        return None

    return T2IRecord(
        pixels=_square_pixels(image, resolution),
        caption=caption,
        metadata={"size": tuple(image.size), "key": str(example.get("__key__", ""))},
    )


def collate_t2i_records(records: list[T2IRecord]) -> dict[str, Any]:
    """Collate for --t2i-direct batches: single-image pixels + raw captions.

    ``image_latent``/``text_embedding`` are deliberately ABSENT — training_step
    encodes pixels with the resident frozen DC-AE and captions with the resident
    Qwen encoder.
    """
    return {
        "pixels": torch.stack([r.pixels for r in records], dim=0),
        "caption": [r.caption for r in records],
        "metadata": [r.metadata for r in records],
    }


@dataclass
class FineT2IStreamingDataset(IterableDataset):
    """Stream a WebDataset t2i corpus via HF ``datasets``, yielding T2IRecords.

    DDP + DataLoader-worker sharding is applied inside ``__iter__`` (the streaming
    dataset is rebuilt per worker so each fresh interpreter opens its own HF
    session — the same fork-safety reason train.py spawns parquet workers).
    """

    repo_id: str = DEFAULT_T2I_REPO
    split: str = "train"
    config_name: str | None = None
    data_dir: str | None = None
    data_files: Any = None
    hf_token: str | bool | None = None
    resolution: int = 1024
    shuffle_buffer: int = 512
    seed: int = 0
    caption_mode: str = "sample"
    short_prob: float = 0.5
    min_short_side: int = 0
    min_aesthetic: float | None = None
    image_key: str = DEFAULT_IMAGE_KEY
    text_key: str = DEFAULT_TEXT_KEY
    json_key: str = DEFAULT_JSON_KEY
    rank: int | None = None
    world_size: int | None = None
    _caption_rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.caption_mode not in _CAPTION_MODES:
            raise ValueError(
                f"caption_mode must be one of {_CAPTION_MODES}, got {self.caption_mode!r}"
            )
        if not 0.0 <= self.short_prob <= 1.0:
            raise ValueError(f"short_prob must be in [0, 1], got {self.short_prob}")
        self._caption_rng = random.Random(self.seed)

    def _load_stream(self):
        from datasets import load_dataset

        token = self.hf_token if isinstance(self.hf_token, str) else None
        kwargs: dict[str, Any] = {}
        if self.data_dir is not None:
            kwargs["data_dir"] = self.data_dir  # restrict to one subfolder of shards
        if self.data_files is not None:
            kwargs["data_files"] = self.data_files
        return load_dataset(
            self.repo_id,
            name=self.config_name,
            split=self.split,
            streaming=True,
            token=token,
            **kwargs,
        )

    def _sharded_stream(self):
        """Build the streaming dataset and split it across DDP nodes then workers.

        DDP: ``split_dataset_by_node`` distributes shards per node (and falls back
        to row-contiguous splitting internally when nodes outnumber shards).
        Workers: ``.shard`` distributes tar shards per worker so each streams only
        its files (bandwidth-efficient) — but that CRASHES / duplicates when there
        are fewer shards than workers, so guard on ``n_shards`` and fall back to
        row-stride sharding (islice) when the file split is unsafe.
        """
        ds = self._load_stream()
        if self.world_size and self.world_size > 1:
            from datasets.distributed import split_dataset_by_node

            ds = split_dataset_by_node(ds, rank=self.rank or 0, world_size=self.world_size)
        worker = get_worker_info()
        salt = worker.id if worker is not None else 0
        if self.shuffle_buffer > 0 and hasattr(ds, "shuffle"):
            ds = ds.shuffle(seed=self.seed + salt, buffer_size=self.shuffle_buffer)
        if worker is not None and worker.num_workers > 1:
            if getattr(ds, "n_shards", 1) >= worker.num_workers:
                ds = ds.shard(num_shards=worker.num_workers, index=worker.id)
            else:
                # Too few shards to split by file: each worker reads the whole stream
                # but keeps a DISJOINT row stride, so no duplication and no crash.
                return islice(ds, worker.id, None, worker.num_workers)
        return ds

    def __iter__(self) -> Iterator[T2IRecord]:
        # Per-worker RNG so the short/enhanced caption pick is decorrelated across
        # workers yet reproducible given the seed + worker id.
        worker = get_worker_info()
        rng = random.Random(self.seed + (worker.id if worker is not None else 0) + 1)
        for example in self._sharded_stream():
            record = t2i_example_to_record(
                example,
                resolution=self.resolution,
                image_key=self.image_key,
                text_key=self.text_key,
                json_key=self.json_key,
                caption_mode=self.caption_mode,
                short_prob=self.short_prob,
                min_short_side=self.min_short_side,
                min_aesthetic=self.min_aesthetic,
                rng=rng,
            )
            if record is not None:
                yield record


__all__ = [
    "FineT2IStreamingDataset",
    "T2IRecord",
    "collate_t2i_records",
    "t2i_example_to_record",
]
