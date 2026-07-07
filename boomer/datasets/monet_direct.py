"""Stream jasperai/monet parquet shards DIRECTLY for training (no cache build).

monet rows already carry everything Boomer needs except text embeddings:

  * ``embedding_vae-dc-sana1p5-1p6b-1024px-tiling-128-resolution-512x512`` —
    SANA DC-AE f32c32 latents flattened to 8192 doubles == 32x16x16 at 512px
    (verified drop-in for Boomer's VAE; see tools/verify_monet_vae_compat.py).
  * caption columns (Gemini / InternVL / ShareGPT4V / Florence-2 / original) —
    the first non-empty key in priority order is used per row.
  * ``embedding_dinov2-vitg14`` — the REPA alignment target.
  * quality columns (aesthetic / nsfw / watermark) for optional row filtering.

So this dataset reuses :class:`ParquetStreamingLatentCacheDataset`'s HTTP-range
streaming machinery and only swaps the row decode: latent + caption (+ dino) per
record, with ``text_embedding`` left EMPTY. Text is encoded on-GPU during
training by the resident frozen Qwen encoder (train.py ``--monet-direct``), so
the Qwen-caching pipeline is skipped entirely.

Column pushdown: monet rows also carry thumbnails, CLIP/SSCD embeddings and
hashes — dead weight several times the size of what we use. The wanted columns
are resolved once against the first matched file of each glob (schemas differ
slightly per subset, e.g. commoncatalog has no ``caption_original``) and their
intersection is pushed down to the Arrow scanner, so those bytes are never
requested.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from boomer.datasets.latent_shards import LatentRecord
from boomer.datasets.wds_latents import ParquetStreamingLatentCacheDataset

MONET_LATENT_KEY = "embedding_vae-dc-sana1p5-1p6b-1024px-tiling-128-resolution-512x512"
MONET_CAPTION_KEYS = (
    "caption_gemini-2.5-flash-lite",
    "caption_internvl-3-8b",
    "caption_sharegpt4v-7b",
    "caption_florence-2-large",
    "caption_original",
)
MONET_DINO_KEY = "embedding_dinov2-vitg14"
LATENT_CHANNELS = 32  # DC-AE f32c32

# Metadata / filter columns requested when present (cheap scalars, useful for
# provenance and the quality filters).
_META_COLUMNS = (
    "__key__",
    "id",
    "source",
    "aesthetic_laion-improved",
    "nsfw_jasperai",
    "wk_jasperai",
    "sscd_cluster_id",
)

_EMPTY = torch.empty(0)


def flat_to_latent(vec: Any) -> torch.Tensor:
    """monet stores the DC-AE latent flattened; reshape to (32, side, side)."""
    arr = np.asarray(vec, dtype=np.float32)
    n = arr.size
    if n % LATENT_CHANNELS != 0:
        raise ValueError(f"latent length {n} not divisible by {LATENT_CHANNELS}")
    cells = n // LATENT_CHANNELS
    side = int(round(cells**0.5))
    if side * side != cells:
        raise ValueError(f"latent {n} -> {cells} cells is not square")
    return torch.from_numpy(arr.reshape(LATENT_CHANNELS, side, side))


def monet_row_to_record(
    row: dict[str, Any],
    *,
    latent_key: str = MONET_LATENT_KEY,
    caption_keys: tuple[str, ...] = MONET_CAPTION_KEYS,
    dino_key: str | None = MONET_DINO_KEY,
    min_aesthetic: float | None = None,
    max_nsfw: float | None = None,
    max_watermark: float | None = None,
) -> LatentRecord | None:
    """Decode one monet parquet row, or None when it fails a filter / lacks data.

    ``text_embedding``/``attention_mask`` are empty placeholders — training encodes
    the caption on-GPU (records must go through :func:`collate_monet_records`).
    """
    if min_aesthetic is not None:
        a = row.get("aesthetic_laion-improved")
        if a is None or a < min_aesthetic:
            return None
    if max_nsfw is not None:
        v = row.get("nsfw_jasperai")
        if v is not None and v > max_nsfw:
            return None
    if max_watermark is not None:
        v = row.get("wk_jasperai")
        if v is not None and v > max_watermark:
            return None

    latent = row.get(latent_key)
    if latent is None:
        return None
    caption, caption_src = "", None
    for key in caption_keys:
        value = row.get(key)
        if value:
            caption, caption_src = value, key
            break
    if not caption:
        return None

    dino = None
    if dino_key is not None:
        raw = row.get(dino_key)
        if raw is not None:
            dino = torch.from_numpy(np.asarray(raw, dtype=np.float32))

    key = row.get("__key__") or row.get("id") or ""
    return LatentRecord(
        image_latent=flat_to_latent(latent),
        text_embedding=_EMPTY,
        attention_mask=_EMPTY,
        caption=str(caption),
        metadata={
            "key": str(key),
            "source": row.get("source"),
            "caption_key": caption_src,
        },
        path=str(key),
        dino_embedding=dino,
    )


def collate_monet_records(records: list[LatentRecord]) -> dict[str, Any]:
    """Collate for direct-monet batches: raw captions instead of text tensors.

    ``text_embedding`` is deliberately ABSENT — training_step treats its absence as
    "encode batch['caption'] with the resident text encoder".
    """
    dinos = [record.dino_embedding for record in records]
    dino_batch = (
        torch.stack(dinos, dim=0) if dinos and all(d is not None for d in dinos) else None
    )
    return {
        "image_latent": torch.stack([record.image_latent for record in records], dim=0),
        "caption": [record.caption for record in records],
        "metadata": [record.metadata for record in records],
        "dino_embedding": dino_batch,
    }


@dataclass
class MonetDirectStreamingDataset(ParquetStreamingLatentCacheDataset):
    """Range-stream jasperai/monet parquet shards, yielding latent+caption records.

    Inherits every streaming/reconnect/memory knob from the Boomer parquet cache
    loader; only the row decode (and the absence of a cache manifest) differ.
    """

    latent_key: str = MONET_LATENT_KEY
    caption_keys: tuple[str, ...] = MONET_CAPTION_KEYS
    dino_key: str | None = MONET_DINO_KEY
    min_aesthetic: float | None = None
    max_nsfw: float | None = None
    max_watermark: float | None = None

    def __post_init__(self) -> None:
        if isinstance(self.caption_keys, str):  # tolerate a single CSV string
            self.caption_keys = tuple(
                key.strip() for key in self.caption_keys.split(",") if key.strip()
            )
        self._columns: list[str] | None = None
        super().__post_init__()

    def _load_manifest(self):
        return None  # monet is a foreign dataset; there is no Boomer cache manifest

    def _dataset_columns(self) -> list[str] | None:
        """Resolve the pushdown column list ONCE against the first matched file of
        each glob (footer read only), intersecting availability across globs so a
        subset missing an optional caption column can't crash the shared stream."""
        if self._columns is not None:
            return self._columns
        import pyarrow.parquet as pq

        available: set[str] | None = None
        fs = None
        for raw in self._globs:
            first = self._first_file_for_glob(raw)
            if first.startswith("hf://"):
                if fs is None:
                    from huggingface_hub import HfFileSystem

                    fs = HfFileSystem(token=self.hf_token)
                with fs.open(first.removeprefix("hf://"), "rb") as handle:
                    names = set(pq.read_schema(handle).names)
            else:
                names = set(pq.read_schema(first).names)
            available = names if available is None else (available & names)
        assert available is not None  # _first_file_for_glob raises on empty globs

        if self.latent_key not in available:
            raise RuntimeError(
                f"[monet-direct] latent column {self.latent_key!r} not present in every "
                f"glob's schema; have {sorted(available)}"
            )
        captions = [key for key in self.caption_keys if key in available]
        if not captions:
            raise RuntimeError(
                f"[monet-direct] none of the caption columns {list(self.caption_keys)} "
                f"are present in every glob's schema; have {sorted(available)}"
            )
        columns = [self.latent_key, *captions]
        if self.dino_key is not None and self.dino_key in available:
            columns.append(self.dino_key)
        columns.extend(c for c in _META_COLUMNS if c in available)
        self._columns = list(dict.fromkeys(columns))
        print(
            f"[monet-direct] column pushdown: {len(self._columns)} of "
            f"{len(available)} columns (captions: {captions})",
            flush=True,
        )
        return self._columns

    def _first_file_for_glob(self, raw: str) -> str:
        import glob as _glob

        if raw.startswith("hf://"):
            from huggingface_hub import HfFileSystem

            fs = HfFileSystem(token=self.hf_token)
            matched = sorted(fs.glob(raw))
            if matched:
                first = matched[0]
                return first if first.startswith("hf://") else f"hf://{first}"
        elif any(ch in raw for ch in "*?[]"):
            matched = sorted(_glob.glob(raw))
            if matched:
                return matched[0]
        else:
            return raw
        raise FileNotFoundError(f"[monet-direct] no parquet files match {raw!r}")

    def _row_to_record(self, row: dict[str, Any]) -> LatentRecord | None:
        return monet_row_to_record(
            row,
            latent_key=self.latent_key,
            caption_keys=tuple(self.caption_keys),
            dino_key=self.dino_key,
            min_aesthetic=self.min_aesthetic,
            max_nsfw=self.max_nsfw,
            max_watermark=self.max_watermark,
        )


__all__ = [
    "LATENT_CHANNELS",
    "MONET_CAPTION_KEYS",
    "MONET_DINO_KEY",
    "MONET_LATENT_KEY",
    "MonetDirectStreamingDataset",
    "collate_monet_records",
    "flat_to_latent",
    "monet_row_to_record",
]
