"""Stream jasperai/monet parquet shards DIRECTLY for training (no cache build).

monet rows already carry everything Boomer needs except text embeddings:

  * ``embedding_vae-dc-sana1p5-1p6b-1024px-tiling-128-resolution-512x512`` —
    SANA DC-AE f32c32 latents flattened to 8192 doubles == 32x16x16 at 512px
    (verified drop-in for Boomer's VAE; see tools/verify_monet_vae_compat.py).
  * caption columns (Gemini / InternVL / ShareGPT4V / Florence-2 / original) —
    by default the first non-empty key in priority order is used per row;
    ``caption_probs`` instead samples ONE caption per row among the row's
    non-empty columns with the given relative probabilities (nano-t2i style),
    so training sees monet's full multi-caption diversity.
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

Split discipline (WebDataset-style): shards are partitioned first by DDP rank,
then by DataLoader worker — every (rank, worker) pair reads a disjoint slice —
and rows pass through a per-worker reservoir shuffle. ``glob_weights`` mixes the
comma-separated globs in ``paths`` at row level: each glob loops its own shard
slice forever and every emitted row is drawn from glob *i* with probability
``w_i / sum(w)`` (nano_mix.yaml-style subset proportions).
"""

from __future__ import annotations

import glob as glob_module
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

# Emitted by _stream_files_forever between full passes over its shard slice, so the
# reservoir-shuffle consumer can drain its buffer at epoch boundaries.
_EPOCH_END = object()


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
    caption_probs: tuple[float, ...] | None = None,
    rng: random.Random | None = None,
    dino_key: str | None = MONET_DINO_KEY,
    min_aesthetic: float | None = None,
    max_nsfw: float | None = None,
    max_watermark: float | None = None,
) -> LatentRecord | None:
    """Decode one monet parquet row, or None when it fails a filter / lacks data.

    Caption policy: without ``caption_probs`` the first non-empty ``caption_keys``
    entry wins (priority order). With ``caption_probs`` (aligned with
    ``caption_keys``) the row's caption is sampled among its non-empty columns
    with those relative weights — nano-t2i's captioners_probabilities idea — so
    repeated passes over a row see different captions.

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
    if caption_probs is not None:
        # Weighted random pick among the captions this row actually has (zero-weight
        # keys are never chosen). Weights are renormalized over the available pool.
        pool = [(key, p) for key, p in zip(caption_keys, caption_probs) if p > 0 and row.get(key)]
        if pool:
            chooser = rng if rng is not None else random
            caption_src = chooser.choices(
                [key for key, _ in pool], weights=[p for _, p in pool], k=1
            )[0]
            caption = row[caption_src]
    else:
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
    # Caption sampling policy, aligned with caption_keys. Forms:
    #   * None — strict priority order (first non-empty key wins);
    #   * one weight tuple / CSV string — one policy for every row;
    #   * ';'-joined CSV groups (or a tuple of tuple|None) aligned 1:1 with the
    #     globs in `paths` — nano_mix pairs each subset with its own captioner
    #     probabilities (an empty group = priority order for that glob). This
    #     form requires glob_weights (mix mode), where rows are drawn per glob.
    caption_probs: Any = None
    dino_key: str | None = MONET_DINO_KEY
    min_aesthetic: float | None = None
    max_nsfw: float | None = None
    max_watermark: float | None = None
    # Row-level subset mixing: weights aligned with the comma-separated globs in
    # `paths`. None = single combined shard list (no mixing).
    glob_weights: tuple[float, ...] | None = None
    # DDP shard split. None = auto-resolve (torch.distributed, then RANK/WORLD_SIZE
    # env, then single-process). BoomerData passes the Trainer's values explicitly.
    rank: int | None = None
    world_size: int | None = None

    def __post_init__(self) -> None:
        if isinstance(self.caption_keys, str):  # tolerate a single CSV string
            self.caption_keys = tuple(
                key.strip() for key in self.caption_keys.split(",") if key.strip()
            )
        def _check_group(group: tuple[float, ...]) -> tuple[float, ...]:
            group = tuple(float(v) for v in group)
            if len(group) != len(self.caption_keys):
                raise ValueError(
                    f"caption_probs group has {len(group)} weights for "
                    f"{len(self.caption_keys)} caption_keys; they must align 1:1"
                )
            if min(group) < 0 or sum(group) <= 0:
                raise ValueError("caption_probs must be non-negative with a positive sum")
            return group

        if isinstance(self.caption_probs, str):
            if ";" in self.caption_probs:  # per-glob groups
                self.caption_probs = tuple(
                    (tuple(float(v) for v in group.split(",") if v.strip()) or None)
                    for group in self.caption_probs.split(";")
                )
            else:
                self.caption_probs = tuple(
                    float(v) for v in self.caption_probs.split(",") if v.strip()
                ) or None
        # Classify: a tuple of tuples/None is per-glob; a tuple of floats is global.
        self._per_glob_caption_probs: tuple[tuple[float, ...] | None, ...] | None = None
        if self.caption_probs is not None:
            if all(g is None or isinstance(g, (tuple, list)) for g in self.caption_probs):
                self._per_glob_caption_probs = tuple(
                    _check_group(g) if g is not None else None for g in self.caption_probs
                )
            else:
                self.caption_probs = _check_group(self.caption_probs)
        if isinstance(self.glob_weights, str):
            self.glob_weights = tuple(
                float(v) for v in self.glob_weights.split(",") if v.strip()
            )
        self._fs: Any = None  # lazily-built HfFileSystem, reused across shards
        self._caption_rng: random.Random | None = None  # per-(rank,worker); set in __iter__
        # The policy applied to the row currently being decoded. Global policy by
        # default; the mix interleave swaps in the drawn glob's group per row.
        self._active_caption_probs = (
            None if self._per_glob_caption_probs is not None else self.caption_probs
        )
        super().__post_init__()  # parses paths -> self._globs
        if self.glob_weights is not None:
            self.glob_weights = tuple(float(v) for v in self.glob_weights)
            if len(self.glob_weights) != len(self._globs):
                raise ValueError(
                    f"glob_weights has {len(self.glob_weights)} weights for "
                    f"{len(self._globs)} globs in paths; they must align 1:1"
                )
            if min(self.glob_weights) < 0 or sum(self.glob_weights) <= 0:
                raise ValueError("glob_weights must be non-negative with a positive sum")
        if self._per_glob_caption_probs is not None:
            if self.glob_weights is None:
                raise ValueError(
                    "per-glob caption_probs (';' groups) require glob_weights: without "
                    "the mix interleave, rows aren't attributable to a glob"
                )
            if len(self._per_glob_caption_probs) != len(self._globs):
                raise ValueError(
                    f"caption_probs has {len(self._per_glob_caption_probs)} ';' groups "
                    f"for {len(self._globs)} globs in paths; they must align 1:1"
                )

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
            caption_probs=self._active_caption_probs,
            rng=self._caption_rng,
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

    # -- distributed split + streaming -------------------------------------- #
    def _resolve_rank_world(self) -> tuple[int, int]:
        """DDP rank/world for the shard split. Explicit fields (set by BoomerData from
        the Trainer) win; else torch.distributed if initialized; else torchrun env."""
        if self.rank is not None and self.world_size is not None:
            return int(self.rank), int(self.world_size)
        try:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                return dist.get_rank(), dist.get_world_size()
        except Exception:  # noqa: BLE001 - fall through to env
            pass
        return int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))

    def _list_glob_files(self, raw: str) -> list[str]:
        """One glob's shard list (the parent's ``_list_files`` flattens ALL globs into
        one list; the weighted mix needs them separately)."""
        if raw.startswith("hf://"):
            matched = sorted(self._hf_filesystem().glob(raw))
            files = [m if m.startswith("hf://") else f"hf://{m}" for m in matched]
        elif any(ch in raw for ch in "*?[]"):
            files = sorted(glob_module.glob(raw))
        else:
            files = [raw]
        if not files:
            raise FileNotFoundError(f"No parquet shards found for mix glob: {raw}")
        return files

    @staticmethod
    def _split_files(files: list[str], offset: int, stride: int, tag: str) -> list[str]:
        """This (rank, worker)'s disjoint shard slice. If there are more readers than
        shards the slice can be empty; fall back to all shards (duplication across
        readers beats a starved worker crashing the empty-epoch guard)."""
        mine = files[offset::stride] if stride > 1 else list(files)
        if not mine:
            print(
                f"[monet-direct] {tag}: more readers ({stride}) than shards "
                f"({len(files)}); reading ALL shards (rows duplicate across readers)",
                flush=True,
            )
            mine = list(files)
        return mine

    def _stream_files_forever(
        self, files: list[str], *, seed_offset: int, tag: str
    ) -> Iterator[Any]:
        """Infinite record stream over ``files``: fresh shard order every pass,
        transient-error shard skip on a reconnect budget, and a hard error if a full
        pass yields zero usable rows. Emits ``_EPOCH_END`` between passes so the
        caller can drain its shuffle buffer at epoch boundaries."""
        epoch = 0
        reconnects = 0
        rows = 0
        while True:  # IterableDataset: loop passes forever; Trainer stops at max_steps
            order = list(files)
            random.Random(self.seed + seed_offset + epoch).shuffle(order)
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
                            f"streaming {tag}; giving up. Last: {exc!r}"
                        ) from exc
                    delay = min(self.stream_reconnect_backoff * 2.0 ** (reconnects - 1), 60.0)
                    print(
                        f"[monet-direct] transient error on {path} "
                        f"({type(exc).__name__}: {exc}); skipping shard "
                        f"(retry {reconnects}/{self.stream_reconnect_max_retries}) after {delay:.0f}s",
                        flush=True,
                    )
                    time.sleep(delay)
                    continue
            if yielded == 0:
                raise RuntimeError(
                    f"[monet-direct] streamed 0 usable rows for {tag} across "
                    f"{len(files)} shard(s) ({rows} rows read; all filtered or "
                    "string-latent shards?) — refusing to spin an empty epoch loop."
                )
            yield _EPOCH_END
            epoch += 1

    def _interleave_globs(
        self, *, offset: int, stride: int, rng: random.Random
    ) -> Iterator[LatentRecord]:
        """Row-level weighted interleave of the per-glob shard streams (nano_mix-style
        subset proportions). Each glob loops its own shard slice forever; every
        emitted row is drawn from glob i with probability weights[i] / sum(weights),
        so small subsets recycle faster instead of capping the mix."""
        streams: list[Iterator[Any]] = []
        for i, raw in enumerate(self._globs):
            files = self._list_glob_files(raw)
            mine = self._split_files(files, offset, stride, tag=raw)
            print(
                f"[monet-direct] mix[{i}] w={self.glob_weights[i]:g} "
                f"{len(mine)}/{len(files)} shard(s) {raw}",
                flush=True,
            )
            # A distinct per-glob seed offset decorrelates shard orders across globs
            # that share one (seed, rank, worker).
            streams.append(
                self._stream_files_forever(mine, seed_offset=offset + 7919 * (i + 1), tag=raw)
            )
        weights = [float(w) for w in self.glob_weights]
        indices = list(range(len(streams)))
        per_glob_probs = self._per_glob_caption_probs
        while True:
            k = rng.choices(indices, weights=weights, k=1)[0]
            if per_glob_probs is not None:
                # Row decode happens synchronously inside next(); swap in this glob's
                # captioner policy first (nano_mix: per-subset captioner probs).
                self._active_caption_probs = per_glob_probs[k]
            record = next(streams[k])
            if record is _EPOCH_END:
                # No global epochs in mix mode — the glob just wrapped around; a pass
                # always yields >= 1 record, so the next item is a real row.
                record = next(streams[k])
            yield record

    def __iter__(self) -> Iterator[LatentRecord]:
        worker = get_worker_info()
        num_workers = worker.num_workers if worker is not None else 1
        worker_id = worker.id if worker is not None else 0
        rank, world = self._resolve_rank_world()
        # WebDataset-style split discipline: shuffle shards -> split by rank ->
        # split by worker -> stream rows -> shuffle samples. Slicing files at
        # [rank * num_workers + worker_id :: world * num_workers] gives every
        # (rank, worker) pair a disjoint shard slice.
        stride = max(1, world) * max(1, num_workers)
        offset = rank * max(1, num_workers) + worker_id

        # Range reads, not Xet whole-file reconstruction (bounds per-worker RSS).
        os.environ["HF_HUB_DISABLE_XET"] = "1"
        os.environ["HF_XET_HIGH_PERFORMANCE"] = "0"

        # Per-(rank, worker) rng for the probabilistic caption pick; the large prime
        # multiplier decorrelates it from the shard-order rngs sharing this seed.
        self._caption_rng = random.Random((self.seed + 1) * 100003 + offset)
        buffer_cap = split_shuffle_buffer_for_worker(self.shuffle_buffer, max(1, num_workers))
        print(
            f"[monet-direct rank {rank}/{world} worker {worker_id}/{num_workers}] "
            f"shuffle_buffer={buffer_cap} (global={self.shuffle_buffer}) "
            f"read_batch_size={self.read_batch_size} trim_every={self.memory_trim_every}"
            + (f" mix_weights={list(self.glob_weights)}" if self.glob_weights else ""),
            flush=True,
        )

        if self.glob_weights is not None:
            mix_rng = random.Random((self.seed + 2) * 100019 + offset)
            stream = self._interleave_globs(offset=offset, stride=stride, rng=mix_rng)
        else:
            files = self._list_files()
            mine = self._split_files(files, offset, stride, tag=f"rank {rank} worker {worker_id}")
            print(
                f"[monet-direct rank {rank}/{world} worker {worker_id}/{num_workers}] "
                f"{len(mine)}/{len(files)} shard(s)",
                flush=True,
            )
            stream = self._stream_files_forever(mine, seed_offset=offset, tag=str(self.paths))

        buffer: list[LatentRecord] = []
        for item in stream:
            if item is _EPOCH_END:
                # Drain the shuffle buffer at epoch end so its rows aren't starved
                # forever (single-stream mode only; the mix interleave never ends).
                random.shuffle(buffer)
                yield from buffer
                buffer.clear()
                continue
            # Streaming reservoir shuffle: once the buffer is full, emit a random
            # resident and replace it with the incoming record.
            if buffer_cap > 0:
                if len(buffer) < buffer_cap:
                    buffer.append(item)
                else:
                    j = random.randrange(len(buffer))
                    out, buffer[j] = buffer[j], item
                    yield out
            else:
                yield item


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
