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

Why a per-file reader instead of ``datasets.load_dataset``: monet is a foreign,
heterogeneous corpus. ``load_dataset("parquet", data_files=[...])`` infers ONE
feature schema from the first shard and hard-casts every later shard to it — but
monet's shards genuinely disagree (e.g. ~9 of 713 cc12m shards store the VAE
latent column as ``string`` rather than ``list<double>``), so that cast raises
``Couldn't cast array of type string to List(...)`` the moment a mismatched shard
is streamed. So this loader reads each shard independently with pyarrow over an
HF range stream — every shard keeps its own schema, and a shard whose latent
column isn't a list (unusable) is skipped rather than crashing the run. This
mirrors how ``tools/cache_monet_parquet_to_boomer_wds.py`` stays robust.

Column pushdown: monet rows also carry thumbnails, CLIP/SSCD embeddings and
hashes — dead weight several times the size of what we use. ``iter_batches`` is
given only the columns present in THAT shard's schema, so those bytes are never
requested (true per-file Parquet projection, not a cross-file intersection).
"""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np
import torch
from torch.utils.data import get_worker_info

from boomer.datasets.latent_shards import LatentRecord
from boomer.datasets.wds_latents import (
    ParquetStreamingLatentCacheDataset,
    _is_transient_stream_error,
    release_parquet_worker_memory,
    split_shuffle_buffer_for_worker,
)

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
        self._fs: Any = None  # lazily-built HfFileSystem, reused across shards
        super().__post_init__()

    def _load_manifest(self):
        return None  # monet is a foreign dataset; there is no Boomer cache manifest

    def _wanted_columns(self, available: set[str]) -> list[str]:
        """Per-shard Parquet projection: the columns we consume that THIS shard has.
        Missing optional captions/dino are simply not requested (schemas vary by
        subset), so a shard is never rejected for lacking an optional column."""
        columns = [self.latent_key]
        columns.extend(key for key in self.caption_keys if key in available)
        if self.dino_key is not None and self.dino_key in available:
            columns.append(self.dino_key)
        columns.extend(c for c in _META_COLUMNS if c in available)
        return list(dict.fromkeys(columns))

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

    # -- per-file pyarrow streaming ----------------------------------------- #
    def _hf_filesystem(self):
        if self._fs is None:
            from huggingface_hub import HfFileSystem

            self._fs = HfFileSystem(token=self.hf_token)
        return self._fs

    def _open_parquet(self, path: str) -> tuple[Any, Any]:
        """Open one shard as (ParquetFile, closeable-handle-or-None). hf:// paths open
        an fsspec range stream (HTTP Range reads, xet disabled); locals open directly."""
        import pyarrow.parquet as pq

        if path.startswith("hf://"):
            handle = self._hf_filesystem().open(path.removeprefix("hf://"), "rb")
            return pq.ParquetFile(handle), handle
        return pq.ParquetFile(path), None

    def _iter_shard(self, path: str) -> Iterator[LatentRecord | None]:
        """Yield decoded records (or None for filtered rows) from one shard. Shards
        whose latent column is missing or not a list type — the handful of monet
        shards that store embeddings as strings — are skipped with a warning."""
        pf, handle = self._open_parquet(path)
        try:
            schema = pf.schema_arrow
            available = set(schema.names)
            if self.latent_key not in available:
                print(f"[monet-direct] skip {path}: no {self.latent_key} column", flush=True)
                return
            latent_type = str(schema.field(self.latent_key).type)
            if "list" not in latent_type:
                # e.g. ~9/713 cc12m shards store the VAE latent as `string`; unusable.
                print(f"[monet-direct] skip {path}: latent column is {latent_type}, not a list",
                      flush=True)
                return
            want = self._wanted_columns(available)
            for batch in pf.iter_batches(columns=want, batch_size=max(1, int(self.read_batch_size))):
                cols = {c: batch.column(c).to_pylist() for c in batch.schema.names}
                for i in range(batch.num_rows):
                    yield self._row_to_record({c: cols[c][i] for c in cols})
        finally:
            if handle is not None:
                handle.close()

    def __iter__(self) -> Iterator[LatentRecord]:
        worker = get_worker_info()
        num_workers = worker.num_workers if worker is not None else 1
        worker_id = worker.id if worker is not None else 0

        # Range reads, not Xet whole-file reconstruction (bounds per-worker RSS).
        os.environ["HF_HUB_DISABLE_XET"] = "1"
        os.environ["HF_XET_HIGH_PERFORMANCE"] = "0"

        files = self._list_files()
        my_files = files[worker_id::num_workers] if num_workers > 1 else list(files)
        if not my_files:  # more workers than shards: let this worker read them all
            my_files = list(files)
        buffer_cap = split_shuffle_buffer_for_worker(self.shuffle_buffer, num_workers)
        print(
            f"[monet-direct worker {worker_id}/{num_workers}] {len(my_files)} shard(s) "
            f"shuffle_buffer={buffer_cap} (global={self.shuffle_buffer}) "
            f"read_batch_size={self.read_batch_size} trim_every={self.memory_trim_every}",
            flush=True,
        )

        buffer: list[LatentRecord] = []
        epoch = 0
        reconnects = 0
        rows = 0
        while True:  # IterableDataset: loop epochs forever; Trainer stops at max_steps
            order = list(my_files)
            random.Random(self.seed + worker_id + epoch).shuffle(order)
            yielded = 0
            for path in order:
                try:
                    shard = self._iter_shard(path)
                    while True:
                        try:
                            record = next(shard)
                        except StopIteration:
                            break
                        reconnects = 0  # a successful read clears the failure budget
                        if record is None:
                            continue
                        rows += 1
                        # Streaming reservoir shuffle: once the buffer is full, emit a
                        # random resident and replace it with the incoming record.
                        if buffer_cap > 0:
                            if len(buffer) < buffer_cap:
                                buffer.append(record)
                            else:
                                j = random.randrange(len(buffer))
                                out, buffer[j] = buffer[j], record
                                yielded += 1
                                yield out
                        else:
                            yielded += 1
                            yield record
                        if self.memory_trim_every > 0 and rows % self.memory_trim_every == 0:
                            release_parquet_worker_memory()
                except Exception as exc:  # noqa: BLE001 - classify transient vs fatal
                    if not _is_transient_stream_error(exc):
                        raise
                    reconnects += 1
                    if reconnects > self.stream_reconnect_max_retries:
                        raise RuntimeError(
                            f"[monet-direct] {reconnects} consecutive transient errors "
                            f"streaming {self.paths}; giving up. Last: {exc!r}"
                        ) from exc
                    delay = min(self.stream_reconnect_backoff * 2.0 ** (reconnects - 1), 60.0)
                    print(
                        f"[monet-direct worker {worker_id}] transient error on {path} "
                        f"({type(exc).__name__}: {exc}); skipping shard "
                        f"(retry {reconnects}/{self.stream_reconnect_max_retries}) after {delay:.0f}s",
                        flush=True,
                    )
                    time.sleep(delay)
                    continue
            # Drain the shuffle buffer at epoch end so its rows aren't starved forever.
            random.shuffle(buffer)
            for record in buffer:
                yielded += 1
                yield record
            buffer.clear()
            if yielded == 0:
                raise RuntimeError(
                    f"[monet-direct] streamed 0 usable rows for {self.paths} across "
                    f"{len(my_files)} shard(s) ({rows} rows read; all filtered or "
                    "string-latent shards?) — refusing to spin an empty epoch loop."
                )
            epoch += 1


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
