"""Stream a single-image text-to-image corpus directly for on-the-fly training.

Built for ma-xu/fine-t2i (--t2i-direct): a **WebDataset** (tar shards of
``jpg`` + ``txt`` + ``json`` per sample), ~6M synthetic + curated real images,
mostly >1K native resolution and randomized aspect ratios. There are NO latents
in the corpus, so training encodes pixels on the fly through the frozen DC-AE
on-GPU (train.py ``encode_pixels``) and the caption through Qwen (same as
--monet-direct text). This avoids building a multi-TB parquet latent cache.

Shard streaming mirrors ``MonetDirectStreamingDataset``: list tar shards once,
split by (rank, worker) with WebDataset-style stride, open **one shard at a
time**, and on a transient Hub/network error **skip that shard** (drop the dead
HTTP client, backoff, continue) instead of rebuilding the whole stream. An
application-level stall watchdog is unnecessary once each shard open has a
socket deadline (``HF_HUB_DOWNLOAD_TIMEOUT``).

Per-row caption policy (``caption_mode``):
  * ``sample`` — per-row random pick between the SHORT caption (``json.prompt``,
    falling back to ``txt``) and the ENHANCED caption (``json.enhanced_prompt``),
    with ``short_prob`` weighting. Trains both concise and richly-detailed
    conditioning so inference generalizes to varied prompt lengths.
  * ``short`` / ``enhanced`` — always that field (with the same fallbacks).

Each image is resized short-side -> ``resolution`` and center-cropped to a
square ``resolution`` x ``resolution``. Records are ~3 MB each
(3 x 1024^2 uint8), so train.py clamps the shuffle buffer in t2i-direct mode.

Subset mixing: ``data_dir`` accepts several subfolders (a comma-separated string
or a sequence), and ``data_dir_weights`` gives each one a relative draw
probability -- the nano_mix / ``--monet-weights`` idea for this corpus.
"""

from __future__ import annotations

import gc
import io
import json as _json
import os
import random
import shlex
import time
from dataclasses import dataclass, field
from typing import Any, Iterator

import numpy as np
import torch
from PIL import Image
from torch.utils.data import IterableDataset, get_worker_info

from boomer.datasets.wds_latents import (
    _is_transient_stream_error,
    split_shuffle_buffer_for_worker,
)

DEFAULT_T2I_REPO = "ma-xu/fine-t2i"
DEFAULT_IMAGE_KEY = "jpg"
DEFAULT_TEXT_KEY = "txt"
DEFAULT_JSON_KEY = "json"
_SHARD_SUFFIXES = (".tar", ".parquet")
SHORT_PROMPT_FIELDS = ("prompt",)
ENHANCED_PROMPT_FIELDS = ("enhanced_prompt",)
AESTHETIC_FIELD = "aesthetic_predictor_v_2_5_score"

_CAPTION_MODES = ("sample", "short", "enhanced")

# Emitted between full passes so the reservoir-shuffle consumer can drain.
_EPOCH_END = object()


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


def _safe_float(value: Any) -> float | None:
    """Parse a numeric filter field; garbage JSON must drop the row, not the worker."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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
        score = _safe_float(meta.get(AESTHETIC_FIELD))
        if score is None or score < min_aesthetic:
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
    native_size = tuple(image.size)
    if min_short_side and min(native_size) < min_short_side:
        return None
    try:
        pixels = _square_pixels(image, resolution)
    except Exception as exc:  # noqa: BLE001
        print(f"[t2i-direct] drop undecodable image: {exc}", flush=True)
        return None

    return T2IRecord(
        pixels=pixels,
        caption=caption,
        metadata={"size": native_size, "key": str(example.get("__key__", ""))},
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
    """Stream a WebDataset t2i corpus shard-by-shard (monet-direct discipline).

    DDP + DataLoader-worker sharding is applied inside ``__iter__`` by slicing the
    explicit shard file list (``offset = rank * num_workers + worker_id``). Each
    worker opens one tar at a time; a transient Hub error skips that shard instead
    of rebuilding the whole corpus stream.
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
    # Socket-level read deadline (HF Hub / curl). Same role as monet_direct's
    # HF_HUB_DOWNLOAD_TIMEOUT: a half-open connection raises in seconds and is
    # classified as a transient shard skip.
    download_timeout_s: float = 60.0
    # Consecutive transient shard skips before giving up (resets on a successful read).
    stream_reconnect_max_retries: int = 20
    stream_reconnect_backoff: float = 5.0
    # Best-effort GC cadence for long-lived workers holding ~3 MB pixel records.
    memory_trim_every: int = 256
    # Subclasses can reuse the streaming engine while keeping diagnostics honest.
    log_tag: str = "t2i-direct"
    _fs: Any = field(init=False, repr=False, default=None)  # lazy HfFileSystem
    _caption_rng: random.Random = field(init=False, repr=False)
    _sources: tuple[str, ...] = field(init=False, repr=False, default=())
    _weights: tuple[float, ...] | None = field(init=False, repr=False, default=None)
    # source subfolder -> its shard file list (repo-relative paths, or absolute locals).
    _file_cache: dict[str | None, list[str] | None] = field(
        init=False, repr=False, default_factory=dict
    )

    def __post_init__(self) -> None:
        if self.caption_mode not in _CAPTION_MODES:
            raise ValueError(
                f"caption_mode must be one of {_CAPTION_MODES}, got {self.caption_mode!r}"
            )
        if not 0.0 <= self.short_prob <= 1.0:
            raise ValueError(f"short_prob must be in [0, 1], got {self.short_prob}")
        if self.stream_reconnect_max_retries < 0:
            raise ValueError("stream_reconnect_max_retries must be >= 0")
        if self.memory_trim_every < 0:
            raise ValueError("memory_trim_every must be >= 0")
        self._sources = _parse_sources(self.data_dir)
        self._weights = _parse_weights(self.data_dir_weights, len(self._sources))
        if len(self._sources) > 1 and self.data_files is not None:
            raise ValueError(
                "data_files applies to a single stream; it cannot be combined with a "
                "multi-entry data_dir mix"
            )
        self._caption_rng = random.Random(self.seed)

    # -- auth / listing ------------------------------------------------------- #
    def _token(self) -> str | None:
        if self.hf_token is True:
            from huggingface_hub import get_token

            return get_token()
        return self.hf_token if isinstance(self.hf_token, str) else None

    def _list_shard_files(self, source: str | None) -> list[str]:
        """Shard paths for one source: repo-relative for HF, absolute for locals."""
        if self.data_files is not None:
            if isinstance(self.data_files, (list, tuple)):
                files = [str(p) for p in self.data_files]
            else:
                files = [str(self.data_files)]
            if not files:
                raise FileNotFoundError(f"No shards in data_files for {self.repo_id}")
            return files

        # Local directory / glob (offline / mirrored corpus).
        if os.path.isdir(self.repo_id) or any(ch in self.repo_id for ch in "*?[]"):
            import glob as glob_module

            root = self.repo_id
            if source:
                root = os.path.join(root, source)
            if os.path.isdir(root):
                files = sorted(
                    p
                    for p in glob_module.glob(os.path.join(root, "**", "*"), recursive=True)
                    if p.endswith(_SHARD_SUFFIXES)
                )
            else:
                files = sorted(
                    p for p in glob_module.glob(root) if p.endswith(_SHARD_SUFFIXES)
                )
            if not files:
                raise FileNotFoundError(f"No tar/parquet shards under {root}")
            return files

        try:
            from huggingface_hub import HfApi

            names = HfApi(token=self._token()).list_repo_files(
                self.repo_id, repo_type="dataset"
            )
        except Exception as exc:  # noqa: BLE001
            raise FileNotFoundError(
                f"Could not list shards for {self.repo_id}: {exc!r}"
            ) from exc
        prefix = f"{source.strip('/')}/" if source else ""
        files = sorted(
            name
            for name in names
            if name.startswith(prefix) and name.endswith(_SHARD_SUFFIXES)
        )
        if not files:
            raise FileNotFoundError(
                f"No tar/parquet shards under {self.repo_id!r} "
                f"(data_dir={source!r})"
            )
        return files

    def _cached_shard_files(self, source: str | None) -> list[str]:
        if source not in self._file_cache or self._file_cache[source] is None:
            self._file_cache[source] = self._list_shard_files(source)
        files = self._file_cache[source]
        assert files is not None
        return files

    def _shard_url(self, path: str) -> str:
        """Local path as-is; HF repo-relative path -> authenticated curl pipe URL
        (same convention as ``resolve_wds_urls`` so webdataset can stream it)."""
        if not self._is_repo_relative(path):
            return path
        from huggingface_hub import hf_hub_url

        url = hf_hub_url(self.repo_id, path, repo_type="dataset")
        token = self._token()
        if token:
            # The token goes through the environment, NOT into the command string.
            # webdataset embeds the whole pipe command in every OSError it raises, so
            # an interpolated bearer token is printed in plaintext on each shard error
            # -- and shard errors are routine here. `$BOOMER_HF_BEARER` is expanded by
            # the shell that gopen spawns (shell=True), so the secret never appears in
            # the command, the exception, or the log.
            os.environ["BOOMER_HF_BEARER"] = token
            # --max-time is a deadline on the WHOLE transfer, which is wrong for a
            # lazily-consumed stream: webdataset reads a shard as training eats it, so
            # the pipe legitimately stays open for minutes and curl killed every shard
            # at the deadline with exit 28. Bound the handshake instead, and abort only
            # on a genuinely stalled socket (< 1 KiB/s sustained for the same window).
            timeout = max(1, int(self.download_timeout_s))
            return (
                "pipe:curl -s -L"
                f" --connect-timeout {timeout}"
                f" --speed-limit 1024 --speed-time {timeout}"
                ' -H "Authorization:Bearer $BOOMER_HF_BEARER"'
                f" {shlex.quote(url)}"
            )
        return url

    @staticmethod
    def _is_repo_relative(path: str) -> bool:
        """Whether ``path`` names a file inside the HF repo (vs a local file or URL)."""
        return not (
            os.path.isabs(path)
            or path.startswith(("hf://", "http://", "https://", "pipe:"))
            or os.path.exists(path)
        )

    def _hf_filesystem(self):
        if self._fs is None:
            from huggingface_hub import HfFileSystem

            self._fs = HfFileSystem(token=self._token())
        return self._fs

    # -- per-shard iteration -------------------------------------------------- #
    def _iter_shard(self, path: str) -> Iterator[T2IRecord | None]:
        """Yield decoded records (or None for filtered rows) from one tar shard."""
        import webdataset as wds

        handle = None
        if self._is_repo_relative(path):
            # Same transport monet-direct uses: an fsspec HTTP-range stream through
            # huggingface_hub's own client, instead of a `pipe:curl` subprocess.
            # Three things this buys over the curl pipe, all of which bit us:
            #   * the read deadline is per-socket-read (HF_HUB_DOWNLOAD_TIMEOUT, set in
            #     __iter__), not a ceiling on the whole transfer, so a shard consumed
            #     lazily over many minutes is not killed mid-stream;
            #   * the token is a Python argument, so it cannot reach a command line,
            #     an exception string, or the log;
            #   * a dropped connection resumes with an HTTP Range read rather than
            #     restarting the shard.
            handle = self._hf_filesystem().open(
                f"datasets/{self.repo_id}/{path}", "rb"
            )
            samples = wds.filters.decode()(
                wds.tariterators.group_by_keys(
                    wds.tariterators.tar_file_expander(
                        [{"url": path, "stream": handle}]
                    )
                )
            )
        else:
            # Locals and explicit URLs keep the plain webdataset open. We already
            # sliced shards by (rank, worker); disable WDS's own splitters so a
            # single-URL open is not emptied by split_by_worker.
            samples = wds.WebDataset(
                [self._shard_url(path)],
                shardshuffle=False,
                nodesplitter=None,
                workersplitter=None,
            ).decode()
        try:
            yield from self._records_from_samples(samples)
        finally:
            if handle is not None:
                handle.close()

    def _records_from_samples(self, samples: Any) -> Iterator[T2IRecord | None]:
        for sample in samples:
            # webdataset may keep extension keys (jpg/json) or strip them; normalize.
            example = dict(sample)
            if self.image_key not in example:
                for alt in (f"{self.image_key}.jpg", f"{self.image_key}.jpeg", "jpeg", "png"):
                    if alt in example:
                        example[self.image_key] = example[alt]
                        break
            if self.json_key not in example and f"{self.json_key}.json" in example:
                example[self.json_key] = example[f"{self.json_key}.json"]
            if self.text_key not in example and f"{self.text_key}.txt" in example:
                example[self.text_key] = example[f"{self.text_key}.txt"]
            yield t2i_example_to_record(
                example,
                resolution=self.resolution,
                image_key=self.image_key,
                text_key=self.text_key,
                json_key=self.json_key,
                caption_mode=self.caption_mode,
                short_prob=self.short_prob,
                min_short_side=self.min_short_side,
                min_aesthetic=self.min_aesthetic,
                rng=self._caption_rng,
            )

    # -- distributed split + forever streams --------------------------------- #
    def _resolve_rank_world(self) -> tuple[int, int]:
        if self.rank is not None and self.world_size is not None:
            return int(self.rank), int(self.world_size)
        try:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                return dist.get_rank(), dist.get_world_size()
        except Exception:  # noqa: BLE001
            pass
        return int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))

    def _split_files(
        self, files: list[str], offset: int, stride: int, tag: str
    ) -> list[str]:
        """This (rank, worker)'s disjoint shard slice. More readers than shards →
        fall back to all shards (duplication beats a starved empty worker)."""
        mine = files[offset::stride] if stride > 1 else list(files)
        if not mine:
            print(
                f"[{self.log_tag}] {tag}: more readers ({stride}) than shards "
                f"({len(files)}); reading ALL shards (rows duplicate across readers)",
                flush=True,
            )
            mine = list(files)
        return mine

    def _stream_files_forever(
        self, files: list[str], *, seed_offset: int, tag: str
    ) -> Iterator[Any]:
        """Infinite record stream over ``files``: fresh shard order every pass,
        transient-error shard skip on a reconnect budget, hard error if a full
        pass yields zero usable rows. Emits ``_EPOCH_END`` between passes."""
        epoch = 0
        reconnects = 0
        rows = 0
        while True:
            order = list(files)
            random.Random(self.seed + seed_offset + epoch).shuffle(order)
            yielded = 0
            transient_skips = 0
            for path in order:
                try:
                    shard = self._iter_shard(path)
                    while True:
                        try:
                            record = next(shard)
                        except StopIteration:
                            break
                        reconnects = 0
                        if record is None:
                            continue
                        rows += 1
                        yielded += 1
                        yield record
                        if self.memory_trim_every > 0 and rows % self.memory_trim_every == 0:
                            gc.collect()
                except Exception as exc:  # noqa: BLE001 - classify transient vs fatal
                    if not _is_transient_stream_error(exc):
                        raise
                    reconnects += 1
                    transient_skips += 1
                    # The cached HfFileSystem may hold a dead HTTP client (the classic
                    # "client has been closed" mode); drop it so the next shard open
                    # builds a fresh filesystem instead of failing identically forever.
                    # Same reset monet-direct does, and the reason its stream survives
                    # a network blip that would otherwise burn the whole retry budget.
                    self._fs = None
                    if reconnects > self.stream_reconnect_max_retries:
                        raise RuntimeError(
                            f"[{self.log_tag}] {reconnects} consecutive transient errors "
                            f"streaming {tag}; giving up. Last: {exc!r}"
                        ) from exc
                    delay = min(
                        self.stream_reconnect_backoff * 2.0 ** (reconnects - 1), 60.0
                    )
                    # Small jitter so workers that stall together don't hammer the Hub
                    # in lockstep (monet is naturally staggered mid-shard; we are not).
                    delay *= random.uniform(0.75, 1.25)
                    print(
                        f"[{self.log_tag}] transient error on {path} "
                        f"({type(exc).__name__}: {exc}); skipping shard "
                        f"(retry {reconnects}/{self.stream_reconnect_max_retries}) "
                        f"after {delay:.0f}s",
                        flush=True,
                    )
                    time.sleep(delay)
                    continue
            if yielded == 0:
                if transient_skips > 0:
                    print(
                        f"[{self.log_tag}] pass over {tag} yielded 0 rows with "
                        f"{transient_skips} transient shard skip(s); retrying pass "
                        f"(reconnect budget {reconnects}/"
                        f"{self.stream_reconnect_max_retries})",
                        flush=True,
                    )
                    epoch += 1
                    continue
                raise RuntimeError(
                    f"[{self.log_tag}] streamed 0 usable rows for {tag} across "
                    f"{len(files)} shard(s) ({rows} rows read; all filtered?) "
                    "— refusing to spin an empty epoch loop."
                )
            yield _EPOCH_END
            epoch += 1

    def _interleave_sources(
        self, *, offset: int, stride: int, rng: random.Random
    ) -> Iterator[T2IRecord]:
        """Row-level weighted interleave of per-source forever streams (monet mix)."""
        streams: list[Iterator[Any]] = []
        for i, source in enumerate(self._sources):
            files = self._cached_shard_files(source)
            mine = self._split_files(files, offset, stride, tag=source)
            print(
                f"[{self.log_tag}] mix[{i}] w={self._weights[i]:g} "
                f"{len(mine)}/{len(files)} shard(s) {source}",
                flush=True,
            )
            streams.append(
                self._stream_files_forever(
                    mine, seed_offset=offset + 7919 * (i + 1), tag=source
                )
            )
        weights = [float(w) for w in self._weights]  # type: ignore[arg-type]
        indices = list(range(len(streams)))
        while True:
            k = rng.choices(indices, weights=weights, k=1)[0]
            record = next(streams[k])
            if record is _EPOCH_END:
                record = next(streams[k])
            yield record

    def __iter__(self) -> Iterator[T2IRecord]:
        worker = get_worker_info()
        num_workers = worker.num_workers if worker is not None else 1
        worker_id = worker.id if worker is not None else 0
        rank, world = self._resolve_rank_world()
        stride = max(1, world) * max(1, num_workers)
        offset = rank * max(1, num_workers) + worker_id

        os.environ["HF_HUB_DISABLE_XET"] = "1"
        os.environ["HF_XET_HIGH_PERFORMANCE"] = "0"
        if self.download_timeout_s > 0:
            os.environ.setdefault(
                "HF_HUB_DOWNLOAD_TIMEOUT", str(int(self.download_timeout_s))
            )

        self._caption_rng = random.Random((self.seed + 1) * 100003 + offset)
        buffer_cap = split_shuffle_buffer_for_worker(
            self.shuffle_buffer, max(1, num_workers)
        )
        print(
            f"[{self.log_tag} rank {rank}/{world} worker {worker_id}/{num_workers}] "
            f"repo={self.repo_id} shuffle_buffer={buffer_cap} "
            f"(global={self.shuffle_buffer}) trim_every={self.memory_trim_every}"
            + (f" mix_weights={list(self._weights)}" if self._weights else ""),
            flush=True,
        )

        if self._weights is not None:
            mix_rng = random.Random((self.seed + 2) * 100019 + offset)
            stream: Iterator[Any] = self._interleave_sources(
                offset=offset, stride=stride, rng=mix_rng
            )
        else:
            source = self._sources[0] if self._sources else None
            files = self._cached_shard_files(source)
            mine = self._split_files(
                files, offset, stride, tag=f"rank {rank} worker {worker_id}"
            )
            print(
                f"[{self.log_tag} rank {rank}/{world} worker {worker_id}/{num_workers}] "
                f"{len(mine)}/{len(files)} shard(s)"
                + (f" data_dir={source}" if source else ""),
                flush=True,
            )
            stream = self._stream_files_forever(
                mine, seed_offset=offset, tag=str(source or self.repo_id)
            )

        buffer: list[T2IRecord] = []
        for item in stream:
            if item is _EPOCH_END:
                random.shuffle(buffer)
                yield from buffer
                buffer.clear()
                continue
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
    "FineT2IStreamingDataset",
    "T2IRecord",
    "collate_t2i_records",
    "t2i_example_to_record",
]
