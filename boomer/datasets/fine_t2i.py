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

Subset mixing: ``data_dir`` accepts several subfolders (a comma-separated string
or a sequence), and ``data_dir_weights`` gives each one a relative draw
probability -- the nano_mix / ``--monet-weights`` idea for this corpus. Unset
weights = natural proportions, where a small subset is swamped by a large one
(fine-t2i's ``curated`` is 168k real photographs against 1.5M synthetic
renders, so it lands near 10% unless it is weighted up). The mix is built with
``datasets.interleave_datasets(..., stopping_strategy="all_exhausted")`` so an
up-weighted small subset recycles instead of ending the epoch early.
"""

from __future__ import annotations

import io
import json as _json
import random
import time
from dataclasses import dataclass, field
from typing import Any, Iterator

import numpy as np
import torch
from PIL import Image
from torch.utils.data import IterableDataset, get_worker_info

from boomer.datasets.wds_latents import _is_transient_stream_error

DEFAULT_T2I_REPO = "ma-xu/fine-t2i"
DEFAULT_IMAGE_KEY = "jpg"
DEFAULT_TEXT_KEY = "txt"
DEFAULT_JSON_KEY = "json"
# Shard extensions worth enumerating when a restart needs an explicit file order.
_SHARD_SUFFIXES = (".tar", ".parquet")
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


def _parse_sources(value: Any) -> tuple[str, ...]:
    """Normalize ``data_dir`` to a tuple of subfolder names (empty = whole repo)."""
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(part.strip() for part in value.split(",") if part.strip())
    return tuple(str(part).strip() for part in value if str(part).strip())


def _parse_weights(value: Any, num_sources: int) -> tuple[float, ...] | None:
    """Validate mixing weights against the resolved subfolder list."""
    if value is None:
        return None
    if isinstance(value, str):
        value = [part for part in value.split(",") if part.strip()]
    weights = tuple(float(part) for part in value)
    if not weights:
        return None
    if num_sources < 2:
        raise ValueError(
            "data_dir_weights needs at least two data_dir entries to mix; got "
            f"{num_sources}"
        )
    if len(weights) != num_sources:
        raise ValueError(
            f"data_dir_weights has {len(weights)} weights for {num_sources} data_dir "
            "entries; they must align 1:1"
        )
    if min(weights) < 0 or sum(weights) <= 0:
        raise ValueError("data_dir_weights must be non-negative with a positive sum")
    return weights


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
    # One subfolder, or several to mix: a comma-separated string or a sequence.
    data_dir: Any = None
    # Relative draw probabilities aligned 1:1 with the data_dir entries; None =
    # natural proportions (each subfolder contributes in proportion to its size).
    data_dir_weights: Any = None
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
    # Seconds of total silence from the streaming reader before the stream is
    # declared stalled and rebuilt. ``datasets`` streaming has no read deadline,
    # so a dead HTTP connection otherwise parks the worker forever (observed: 7h
    # at 0% GPU with no error). Generous by design — a slow shard is not a stall.
    stall_timeout_s: float = 300.0
    # Socket-level read deadline, the mechanism monet_direct relies on: it inherits
    # ParquetStreamingLatentCacheDataset, which sets HF_HUB_DOWNLOAD_TIMEOUT
    # (wds_latents.py:1496), so a half-open connection raises ReadTimeout in SECONDS,
    # is classified transient, and costs one skipped shard. fine_t2i only referenced
    # that env var in a comment and never set it, so its reads blocked forever and the
    # only backstop was the 300s watchdog below -- whose remedy is a full stream
    # rebuild, orders of magnitude more expensive than skipping a shard. Setting this
    # is what keeps the watchdog a last resort instead of the primary mechanism.
    download_timeout_s: float = 60.0
    # Reconnect budget shared by stalls AND transient stream errors, mirroring
    # monet_direct's stream_reconnect_* (wds_latents.py:1463) so both streaming
    # paths survive the same network weather.
    # 20 was too tight: a single Hub outage burned the budget and killed a 6-hour run
    # even though 11 of 12 workers were healthy (PyTorch has no way to continue with a
    # dead worker). At the 600s cap this is ~8h of trying before giving up, which is the
    # right trade when the alternative is losing the run.
    stall_max_restarts: int = 40
    stall_reconnect_backoff: float = 5.0
    _caption_rng: random.Random = field(init=False, repr=False)
    _sources: tuple[str, ...] = field(init=False, repr=False, default=())
    _weights: tuple[float, ...] | None = field(init=False, repr=False, default=None)
    # source subfolder -> its shard file list (one HfApi call, reused per restart).
    _file_cache: dict = field(init=False, repr=False, default_factory=dict)

    def __post_init__(self) -> None:
        if self.caption_mode not in _CAPTION_MODES:
            raise ValueError(
                f"caption_mode must be one of {_CAPTION_MODES}, got {self.caption_mode!r}"
            )
        if not 0.0 <= self.short_prob <= 1.0:
            raise ValueError(f"short_prob must be in [0, 1], got {self.short_prob}")
        self._sources = _parse_sources(self.data_dir)
        self._weights = _parse_weights(self.data_dir_weights, len(self._sources))
        if len(self._sources) > 1 and self.data_files is not None:
            raise ValueError(
                "data_files applies to a single stream; it cannot be combined with a "
                "multi-entry data_dir mix"
            )
        self._caption_rng = random.Random(self.seed)

    def _load_one(self, data_dir: str | None, data_files: Any = None):
        import os

        from datasets import load_dataset

        # Set inside the worker process, not __post_init__: with spawn the dataset is
        # re-created per worker, and this must be in the env of whichever process
        # actually opens the sockets. setdefault so an explicit env still wins.
        if self.download_timeout_s > 0:
            os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", str(int(self.download_timeout_s)))
        token = self.hf_token if isinstance(self.hf_token, str) else None
        kwargs: dict[str, Any] = {}
        files = data_files if data_files is not None else self.data_files
        if files is not None:
            kwargs["data_files"] = files
        if data_dir is not None and data_files is None:
            # An explicit per-restart file list is already repo-rooted (it carries
            # the subfolder prefix), so passing data_dir too would double it.
            kwargs["data_dir"] = data_dir  # restrict to one subfolder of shards
        return load_dataset(
            self.repo_id,
            name=self.config_name,
            split=self.split,
            streaming=True,
            token=token,
            **kwargs,
        )

    def _list_shard_files(self, source: str | None) -> list[str] | None:
        """Every shard file under one source subfolder, sorted the way ``datasets``
        resolves them. None if listing fails — callers then fall back to letting
        ``load_dataset`` resolve the shards itself."""
        try:
            from huggingface_hub import HfApi

            token = self.hf_token if isinstance(self.hf_token, str) else None
            names = HfApi(token=token).list_repo_files(self.repo_id, repo_type="dataset")
        except Exception as exc:  # noqa: BLE001 - degrade to the old behaviour
            print(
                f"[t2i-direct] shard listing failed ({type(exc).__name__}: {exc}); "
                "restarts will replay from the head of the stream",
                flush=True,
            )
            return None
        prefix = f"{source.strip('/')}/" if source else ""
        files = sorted(
            name
            for name in names
            if name.startswith(prefix) and name.endswith(_SHARD_SUFFIXES)
        )
        return files or None

    def _shard_files_for_restart(self, source: str | None, restart: int) -> list[str] | None:
        """Shard order for one source on rebuild number ``restart``.

        Restart 0 returns None, i.e. keeps ``load_dataset``'s own resolution, so the
        happy path is byte-for-byte what it always was. Later restarts pass an
        explicitly permuted file list so the rebuilt stream does not re-read the
        rows the dead one already delivered. A bumped RNG salt cannot achieve this:
        with ``shuffle_buffer == 0`` no ``.shuffle()`` runs at all, so every source
        restarts at its own row 0 and only the interleave draw order changes.

        The permutation is seeded by ``restart`` ONLY, never by worker id. All
        workers must agree on the order, because ``datasets`` splits shards across
        workers by stride (``files[worker_id::num_workers]``); per-worker orders
        would make those strides overlap and duplicate rows.
        """
        if restart <= 0:
            return None
        if self.data_files is not None:
            # Respect a caller-supplied list; permute it rather than the subfolder.
            if not isinstance(self.data_files, (list, tuple)):
                return None  # a glob or dict — leave resolution to load_dataset
            order = list(self.data_files)
        else:
            if source not in self._file_cache:
                self._file_cache[source] = self._list_shard_files(source)
            files = self._file_cache[source]
            if not files:
                return None
            order = list(files)
        random.Random(self.seed + 100003 * restart).shuffle(order)
        return order

    def _load_one_resilient(self, source: str | None, files: Any):
        """``_load_one`` with the explicit per-restart shard list, degrading to the
        default resolution if ``load_dataset`` rejects that list (an unexpected
        builder inference from bare ``.tar`` paths would otherwise raise a
        NON-transient error and kill the worker during recovery — the one moment it
        must not). The failed listing is poisoned so later restarts skip it."""
        if files is None:
            return self._load_one(source, None)
        try:
            return self._load_one(source, files)
        except Exception as exc:  # noqa: BLE001 - recovery must not be fragile
            print(
                f"[t2i-direct] explicit shard list rejected for {source!r} "
                f"({type(exc).__name__}: {exc}); falling back to default resolution "
                "— this restart may replay the head of the stream",
                flush=True,
            )
            self._file_cache[source] = None
            return self._load_one(source, None)

    def _load_stream(self, *, seed_salt: int = 0, restart: int = 0):
        """One subfolder's stream, or a weighted row-level mix of several.

        ``stopping_strategy="all_exhausted"`` keeps an up-weighted small subset
        recycling until the largest one is spent, instead of "first_exhausted"
        ending the pass as soon as the smallest runs dry (fine-t2i's curated set
        is ~9x smaller than the synthetic square split, so first_exhausted would
        cut every epoch to curated's length).
        """
        if len(self._sources) <= 1:
            source = self._sources[0] if self._sources else None
            return self._load_one_resilient(
                source, self._shard_files_for_restart(source, restart)
            )

        from datasets import interleave_datasets

        streams = [
            self._load_one_resilient(source, self._shard_files_for_restart(source, restart))
            for source in self._sources
        ]
        probabilities = None
        if self._weights is not None:
            total = sum(self._weights)
            probabilities = [weight / total for weight in self._weights]
        return interleave_datasets(
            streams,
            probabilities=probabilities,
            seed=self.seed + seed_salt,
            stopping_strategy="all_exhausted",
        )

    def _sharded_stream(self, restart: int = 0):
        """Build the streaming dataset and split it across DDP nodes.

        DDP: ``split_dataset_by_node`` distributes shards per node (and falls back
        to row-contiguous splitting internally when nodes outnumber shards).

        Workers: do NOT shard here. ``datasets`` already splits an IterableDataset
        across DataLoader workers on its own when it is iterated inside one, so an
        extra ``.shard(num_workers, worker_id)`` COMPOSES with that automatic split
        and leaves each worker 1/N**2 of the rows. Measured on an 8-worker loader:
        aggregate coverage fell to 14% of the dataset per pass (34% at 3 workers)
        with the manual shard, versus ~100% at both worker counts without it. No
        duplication either way — the manual split was pure loss.

        Note ``.shuffle()`` collapses ``num_shards`` (independent of buffer size),
        and ``datasets`` caps active workers at ``num_shards`` — so a non-zero
        ``shuffle_buffer`` costs worker parallelism. Set ``--wds-shuffle-buffer 0``
        to keep every worker busy when the loader is the bottleneck.
        """
        worker = get_worker_info()
        salt = (worker.id if worker is not None else 0) + restart
        # Salt the mix RNG per worker so two workers do not draw the same subset
        # sequence from the weighted interleave. ``restart`` additionally moves a
        # rebuilt-after-stall stream off the draw order that stalled, and permutes
        # its shard order so it does not re-read the head it already consumed.
        ds = self._load_stream(seed_salt=salt, restart=restart)
        if self.world_size and self.world_size > 1:
            from datasets.distributed import split_dataset_by_node

            ds = split_dataset_by_node(ds, rank=self.rank or 0, world_size=self.world_size)
        if self.shuffle_buffer > 0 and hasattr(ds, "shuffle"):
            ds = ds.shuffle(seed=self.seed + salt, buffer_size=self.shuffle_buffer)
        return ds

    def _examples_with_stall_guard(self, tag: str) -> Iterator[Any]:
        """Yield raw examples, rebuilding the stream when the reader goes silent.

        ``datasets`` streaming enforces no read deadline, so a half-open HTTP
        connection leaves the worker blocked in a socket read indefinitely. The
        DataLoader then waits on that worker and training stops with no error, no
        traceback and 0% GPU. ``wds_latents.py`` guards its own path for the same
        reason (ROLLING_WDS_DOWNLOAD_TIMEOUT_S, and the HF_HUB_DOWNLOAD_TIMEOUT it
        sets at lines 443/1104); ``fine_t2i`` had no equivalent.

        A daemon thread does the reading. If nothing arrives for
        ``stall_timeout_s`` the thread is abandoned — it stays blocked in the
        kernel, but it is a daemon holding only its socket, and it exits on its own
        if the read ever returns — and the stream is rebuilt over a freshly
        permuted shard order (``_shard_files_for_restart``) so it does not re-read
        the head it already consumed.

        The retry budget counts CONSECUTIVE failures and resets the moment a
        rebuilt stream delivers a row, matching ``monet_direct``'s
        ``reconnects = 0`` on every successful read.
        """
        import queue as _queue
        import threading

        end_marker = object()
        consecutive = 0  # failures since the stream last delivered a row
        restart = 0  # lifetime rebuilds; only ever increases, so shard order moves
        total = 0  # rows delivered across every attempt
        while True:
            handoff: _queue.Queue = _queue.Queue(maxsize=4)
            wanted = threading.Event()
            wanted.set()
            delivered = 0

            def _produce(sink=handoff, live=wanted, bump=restart) -> None:
                try:
                    for example in self._sharded_stream(restart=bump):
                        while live.is_set():
                            try:
                                sink.put(example, timeout=1.0)
                                break
                            except _queue.Full:
                                continue
                        if not live.is_set():
                            return
                    sink.put(end_marker)
                except BaseException as exc:  # surface in the consumer thread
                    try:
                        sink.put(exc, timeout=5.0)
                    except _queue.Full:
                        pass

            reader = threading.Thread(
                target=_produce, name=f"fine-t2i-reader-{tag}", daemon=True
            )
            reader.start()
            reason = ""
            while True:
                try:
                    item = handoff.get(timeout=self.stall_timeout_s)
                except _queue.Empty:
                    reason = f"no data for {self.stall_timeout_s:g}s (stalled)"
                    break
                if item is end_marker:
                    if total == 0:
                        # monet_direct refuses to spin an empty epoch loop
                        # (monet_direct.py:551); a worker that silently yields
                        # nothing is data loss, not an exhausted corpus.
                        raise RuntimeError(
                            f"[t2i-direct] {tag}: stream ended having yielded 0 rows "
                            f"from {self.repo_id} (data_dir={self.data_dir!r}) — "
                            "wrong subfolder, or every row filtered out?"
                        )
                    return
                if isinstance(item, BaseException):
                    # Transient network faults are retried the way monet_direct
                    # retries them; genuine data/code errors must still surface.
                    if not _is_transient_stream_error(item):
                        raise item
                    reason = f"{type(item).__name__}: {item}"
                    break
                delivered += 1
                total += 1
                yield item

            wanted.clear()
            # monet_direct clears its reconnect budget on every successful read
            # (monet_direct.py:500). Mirror that: the budget counts CONSECUTIVE
            # failures, not failures over the lifetime of the run. Counting the
            # lifetime kills a multi-day job on its 21st unrelated blip even though
            # all twenty earlier ones recovered fully — and compounds the backoff
            # to the 60s cap for incidents days apart.
            consecutive = 0 if delivered > 0 else consecutive + 1
            restart += 1
            if consecutive > self.stall_max_restarts:
                raise RuntimeError(
                    f"[t2i-direct] {tag}: stream failed {consecutive} times in a row "
                    f"with no data in between (last: {reason}) — giving up rather "
                    "than hanging"
                )
            # Cap at 600s, not 60s: the failure this recovers from is usually Hub-side
            # rate limiting, which clears in minutes. Retrying every 60s prolongs it.
            delay = min(self.stall_reconnect_backoff * 2.0 ** max(0, consecutive - 1), 600.0)
            # Jitter, because every worker stalls on the same outage and would otherwise
            # rebuild in lockstep -- 12 simultaneous re-resolutions is a thundering herd
            # that causes the rate limiting that causes the next stall. monet_direct is
            # inherently staggered (each worker is mid-shard at a different offset), so
            # it never needed this.
            delay *= random.uniform(0.5, 1.5)
            budget = (
                f"consecutive failure {consecutive}/{self.stall_max_restarts}"
                if consecutive
                else f"budget reset by {delivered} delivered row(s)"
            )
            print(
                f"[t2i-direct] {tag}: {reason} after {delivered} row(s) — rebuilding "
                f"stream in {delay:.0f}s ({budget}, lifetime restart {restart})",
                flush=True,
            )
            time.sleep(delay)

    def __iter__(self) -> Iterator[T2IRecord]:
        # Per-worker RNG so the short/enhanced caption pick is decorrelated across
        # workers yet reproducible given the seed + worker id.
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        rng = random.Random(self.seed + worker_id + 1)
        for example in self._examples_with_stall_guard(tag=f"worker {worker_id}"):
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
