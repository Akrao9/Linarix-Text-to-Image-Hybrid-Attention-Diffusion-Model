"""--t2i-direct: single-image WebDataset streaming — square crop, caption
policy (short/enhanced sampling), filters, collate contract, and monet-style
per-shard streaming (explicit worker shard slice + reservoir shuffle)."""

from __future__ import annotations

import random
import sys
from itertools import islice
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from boomer.datasets.fine_t2i import (  # noqa: E402
    FineT2IStreamingDataset,
    T2IRecord,
    UnusableSourceError,
    collate_t2i_records,
    t2i_example_to_record,
)


def _example(width=1200, height=800, prompt="a red cube", enhanced="a detailed red cube on grass",
             aesthetic=6.0, txt=None, key="k0") -> dict:
    img = Image.new("RGB", (width, height), (200, 30, 30))
    meta = {}
    if prompt is not None:
        meta["prompt"] = prompt
    if enhanced is not None:
        meta["enhanced_prompt"] = enhanced
    if aesthetic is not None:
        meta["aesthetic_predictor_v_2_5_score"] = aesthetic
    ex = {"jpg": img, "json": meta, "__key__": key}
    if txt is not None:
        ex["txt"] = txt
    return ex


def test_record_decode_square_and_dtype() -> None:
    rec = t2i_example_to_record(_example(1200, 800), resolution=256, rng=random.Random(0))
    assert isinstance(rec, T2IRecord)
    assert rec.pixels.shape == (3, 256, 256)  # non-square source -> square canvas
    assert rec.pixels.dtype == torch.uint8
    assert rec.metadata["size"] == (1200, 800)


def test_caption_modes_and_fallbacks() -> None:
    ex = _example(prompt="short cap", enhanced="a long enhanced caption")
    assert t2i_example_to_record(ex, resolution=64, caption_mode="short").caption == "short cap"
    assert t2i_example_to_record(
        ex, resolution=64, caption_mode="enhanced"
    ).caption == "a long enhanced caption"

    # sample mode spans both fields over many rows.
    seen = {
        t2i_example_to_record(ex, resolution=64, caption_mode="sample",
                              rng=random.Random(i)).caption
        for i in range(40)
    }
    assert seen == {"short cap", "a long enhanced caption"}

    # short_prob extremes pin the pick.
    assert t2i_example_to_record(
        ex, resolution=64, caption_mode="sample", short_prob=1.0, rng=random.Random(3)
    ).caption == "short cap"
    assert t2i_example_to_record(
        ex, resolution=64, caption_mode="sample", short_prob=0.0, rng=random.Random(3)
    ).caption == "a long enhanced caption"

    # No json.prompt -> fall back to the txt member for the short caption.
    ex2 = _example(prompt=None, enhanced=None, txt="from txt member")
    assert t2i_example_to_record(ex2, resolution=64, caption_mode="short").caption == "from txt member"
    # Enhanced falls back to short when absent.
    ex3 = _example(prompt="only short", enhanced=None)
    assert t2i_example_to_record(ex3, resolution=64, caption_mode="enhanced").caption == "only short"


def test_filters_drop_rows() -> None:
    # aesthetic below threshold -> dropped.
    assert t2i_example_to_record(_example(aesthetic=4.0), resolution=64, min_aesthetic=5.0) is None
    assert t2i_example_to_record(_example(aesthetic=6.0), resolution=64, min_aesthetic=5.0) is not None
    # missing aesthetic + filter on -> dropped (cannot verify quality).
    assert t2i_example_to_record(_example(aesthetic=None), resolution=64, min_aesthetic=5.0) is None
    # garbage aesthetic must drop the row, not raise.
    assert t2i_example_to_record(
        _example(aesthetic="not-a-number"), resolution=64, min_aesthetic=5.0
    ) is None
    # short side below the floor -> dropped before any upscale.
    assert t2i_example_to_record(_example(300, 200), resolution=1024, min_short_side=512) is None
    assert t2i_example_to_record(_example(700, 600), resolution=1024, min_short_side=512) is not None
    # no caption anywhere -> dropped.
    assert t2i_example_to_record(
        _example(prompt=None, enhanced=None), resolution=64
    ) is None


def test_raw_bytes_and_struct_image_tolerated() -> None:
    import io

    img = Image.new("RGB", (800, 800), (0, 128, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    raw = {"jpg": buf.getvalue(), "json": {"prompt": "p"}, "__key__": "b"}
    struct = {"jpg": {"bytes": buf.getvalue(), "path": "x.jpg"}, "json": {"prompt": "p"}}
    assert t2i_example_to_record(raw, resolution=64) is not None
    assert t2i_example_to_record(struct, resolution=64) is not None


def test_collate_contract_for_training_step() -> None:
    recs = [t2i_example_to_record(_example(), resolution=32, rng=random.Random(i)) for i in range(3)]
    batch = collate_t2i_records(recs)
    assert batch["pixels"].shape == (3, 3, 32, 32)
    assert batch["pixels"].dtype == torch.uint8
    assert len(batch["caption"]) == 3
    # No pre-encoded keys: training_step must route to on-the-fly DC-AE + Qwen.
    assert "image_latent" not in batch
    assert "text_embedding" not in batch


def test_stream_iteration_through_mocked_shards(monkeypatch) -> None:
    """Monet-style path: list shards, open one at a time, decode rows."""
    examples = [_example(prompt=f"cap {i}", enhanced=f"enhanced {i}", key=f"k{i}") for i in range(5)]

    ds = FineT2IStreamingDataset(
        resolution=32,
        shuffle_buffer=0,
        caption_mode="short",
        data_files=["shard_a.tar", "shard_b.tar"],
        stream_reconnect_max_retries=0,
    )
    def fake_iter(path: str):
        if path != "shard_a.tar":
            return
        for ex in examples:
            yield t2i_example_to_record(ex, resolution=32, caption_mode="short")

    monkeypatch.setattr(ds, "_iter_shard", fake_iter)

    records = list(islice(iter(ds), 5))
    assert len(records) == 5
    assert {r.caption for r in records} == {f"cap {i}" for i in range(5)}
    for r in records:
        assert r.pixels.shape == (3, 32, 32)
        assert r.pixels.dtype == torch.uint8


def test_dataset_splits_shards_by_worker(monkeypatch) -> None:
    """Like monet-direct: each DataLoader worker owns a disjoint shard slice."""
    import boomer.datasets.fine_t2i as m

    shard_rows = {
        "s0.tar": [_example(prompt="from-s0", key="a")],
        "s1.tar": [_example(prompt="from-s1", key="b")],
    }

    def run_worker(worker_id, num_workers):
        ds = FineT2IStreamingDataset(
            resolution=16,
            shuffle_buffer=0,
            caption_mode="short",
            data_files=["s0.tar", "s1.tar"],
            stream_reconnect_max_retries=0,
        )
        monkeypatch.setattr(
            ds,
            "_iter_shard",
            lambda path: (
                t2i_example_to_record(ex, resolution=16, caption_mode="short")
                for ex in shard_rows[path]
            ),
        )
        monkeypatch.setattr(
            m,
            "get_worker_info",
            lambda: SimpleNamespace(id=worker_id, num_workers=num_workers),
        )
        # One epoch only: take until first _EPOCH_END drain finishes (1 row each).
        return [r.caption for r in islice(iter(ds), 1)]

    assert run_worker(0, 2) == ["from-s0"]
    assert run_worker(1, 2) == ["from-s1"]


def test_transient_shard_error_skips_and_continues(monkeypatch) -> None:
    """A transient Hub error on one shard must skip it, not rebuild/abort."""
    import boomer.datasets.fine_t2i as m

    good = [_example(prompt="ok", key="g")]

    def flaky_iter(path: str):
        if path == "bad.tar":
            raise ConnectionError("temporary hub blip")
        yield t2i_example_to_record(good[0], resolution=16, caption_mode="short")

    ds = FineT2IStreamingDataset(
        resolution=16,
        shuffle_buffer=0,
        caption_mode="short",
        data_files=["bad.tar", "good.tar"],
        stream_reconnect_max_retries=3,
        stream_reconnect_backoff=0.0,
        seed=0,
    )
    monkeypatch.setattr(ds, "_iter_shard", flaky_iter)
    monkeypatch.setattr(m.time, "sleep", lambda *_: None)

    records = list(islice(iter(ds), 1))
    assert len(records) == 1
    assert records[0].caption == "ok"


def test_empty_after_filter_is_fatal(monkeypatch) -> None:
    ds = FineT2IStreamingDataset(
        resolution=16,
        shuffle_buffer=0,
        caption_mode="short",
        data_files=["emptyish.tar"],
        min_aesthetic=9.9,
        stream_reconnect_max_retries=0,
    )
    monkeypatch.setattr(
        ds,
        "_iter_shard",
        lambda path: (
            t2i_example_to_record(
                _example(aesthetic=1.0), resolution=16, caption_mode="short", min_aesthetic=9.9
            )
            for _ in range(3)
        ),
    )
    with pytest.raises(RuntimeError, match="0 usable rows"):
        list(islice(iter(ds), 1))


def test_dataset_validation() -> None:
    with pytest.raises(ValueError, match="caption_mode"):
        FineT2IStreamingDataset(caption_mode="verbose")
    with pytest.raises(ValueError, match="short_prob"):
        FineT2IStreamingDataset(short_prob=1.5)
    with pytest.raises(ValueError, match="data_dir_weights"):
        FineT2IStreamingDataset(data_dir="a,b", data_dir_weights="1")


def _synthetic_tar(count: int = 3) -> bytes:
    """A minimal WebDataset shard: one .jpg + one .json per sample."""
    import io
    import json
    import tarfile

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for i in range(count):
            image = Image.new("RGB", (32, 32), color=(i * 10, 0, 0))
            jpg = io.BytesIO()
            image.save(jpg, format="JPEG")
            payloads = (
                (f"s{i}.jpg", jpg.getvalue()),
                (f"s{i}.json", json.dumps({"prompt": f"cap {i}"}).encode()),
            )
            for name, blob in payloads:
                info = tarfile.TarInfo(name)
                info.size = len(blob)
                tf.addfile(info, io.BytesIO(blob))
    return buf.getvalue()


def test_repo_shards_stream_through_hf_filesystem_not_a_curl_pipe() -> None:
    """Repo-relative shards open via HfFileSystem range reads (monet-direct's
    transport). The curl pipe bounded the WHOLE transfer with --max-time, so a
    lazily-consumed 1024px shard was killed mid-read every time."""
    import io

    opened: list[str] = []
    closed: list[bool] = []

    class _Handle(io.BytesIO):
        def close(self) -> None:
            closed.append(True)
            super().close()

    class _FS:
        def open(self, path: str, mode: str, block_size: int | None = None):
            opened.append(path)
            assert mode == "rb"
            # A 5 MiB (fsspec default) read-ahead made HF truncate the response
            # mid-body under parallel streams; the range read must stay bounded.
            assert block_size == 1 << 20, block_size
            return _Handle(_synthetic_tar())

    ds = FineT2IStreamingDataset(
        repo_id="ma-xu/fine-t2i",
        resolution=32,
        shuffle_buffer=0,
        caption_mode="short",
        data_files=["curated/train-000092.tar"],
    )
    ds._fs = _FS()

    records = [r for r in ds._iter_shard("curated/train-000092.tar") if r is not None]

    assert len(records) == 3
    assert {r.caption for r in records} == {f"cap {i}" for i in range(3)}
    # fsspec path is datasets/<repo_id>/<path-in-repo>, as monet-direct builds it.
    assert opened == ["datasets/ma-xu/fine-t2i/curated/train-000092.tar"]
    assert closed, "shard handle must be closed when the generator finishes"


def test_shard_url_never_embeds_the_bearer_token() -> None:
    """webdataset puts the pipe command into every OSError it raises, so an
    interpolated token is printed in plaintext on each shard error."""
    ds = FineT2IStreamingDataset(
        repo_id="ma-xu/fine-t2i",
        hf_token="hf_TOTALLYSECRETVALUE",
        resolution=32,
    )
    url = ds._shard_url("curated/train-000092.tar")

    assert "hf_TOTALLYSECRETVALUE" not in url
    assert "$BOOMER_HF_BEARER" in url
    # --max-time is a ceiling on the entire transfer; it must not come back.
    assert "--max-time" not in url
    assert "--connect-timeout" in url and "--speed-time" in url


def test_transient_error_drops_the_cached_filesystem() -> None:
    """A dead HTTP client would otherwise fail every later shard identically."""
    ds = FineT2IStreamingDataset(
        repo_id="ma-xu/fine-t2i",
        resolution=32,
        shuffle_buffer=0,
        data_files=["a.tar", "b.tar"],
        stream_reconnect_max_retries=5,
        stream_reconnect_backoff=0.0,
    )
    ds._fs = object()
    seen: list[object] = []

    def fake_iter(path: str):
        seen.append(ds._fs)
        raise OSError("Connection reset by peer")
        yield  # pragma: no cover - generator marker

    ds._iter_shard = fake_iter  # type: ignore[method-assign]

    stream = ds._stream_files_forever(["a.tar", "b.tar"], seed_offset=0, tag="t")
    with pytest.raises(RuntimeError, match="transient errors"):
        for _ in range(50):
            next(stream)

    assert seen[0] is not None
    assert seen[1] is None, "cached HfFileSystem must be dropped after a transient error"


def test_barren_shard_is_skipped_not_fatal():
    """One 0% shard is data, not a config error: MONET buckets by resolution, so a
    usable source can still hand a worker a shard where nothing clears the filter."""
    rec = t2i_example_to_record(_example(), resolution=32, caption_mode="short")
    shards = {"bad.tar": [None] * 500, "good.tar": [rec]}

    ds = FineT2IStreamingDataset(
        repo_id="ma-xu/fine-t2i", resolution=32, shuffle_buffer=0,
        max_screened_per_shard=10, max_barren_shards=20,
    )
    ds._fs = None
    ds._iter_shard = lambda path: iter(shards[path])  # type: ignore[method-assign]

    got = next(ds._stream_files_forever(["bad.tar", "good.tar"], seed_offset=0, tag="s"))
    assert got is rec, "a barren shard must be abandoned and the next one tried"


def test_a_healthy_shard_is_not_truncated_by_one_filtered_row():
    """The screen counts CONSECUTIVE rejects, not rows inspected.

    Counting cumulatively truncated every healthy shard: fine-t2i's synthetic shards
    yielded ~64 good rows, hit their first filtered row with the counter already past
    the cap, and were abandoned -- logged as "none passed the record filters" while
    they had in fact passed 64. The tell in the log was the row count drifting (64, 65,
    ... 90) where a real barren shard breaks at exactly the cap, and every skip
    reporting "1 barren in a row" because each accepted row reset the streak.
    """
    rec = t2i_example_to_record(_example(), resolution=32, caption_mode="short")
    # 20 good rows, one filtered row, 20 more good rows -- all 40 must come through.
    rows = [rec] * 20 + [None] + [rec] * 20

    ds = FineT2IStreamingDataset(
        repo_id="ma-xu/fine-t2i", resolution=32, shuffle_buffer=0,
        max_screened_per_shard=10, max_barren_shards=20,
    )
    ds._fs = None
    ds._iter_shard = lambda path: iter(rows)  # type: ignore[method-assign]

    stream = ds._stream_files_forever(["a.tar"], seed_offset=0, tag="s")
    got = [next(stream) for _ in range(40)]
    assert all(r is rec for r in got), "one filtered row must not end a productive shard"


def test_a_long_rejected_run_ends_the_shard_without_calling_it_barren():
    """Bounded cost still applies mid-shard, but it is not evidence about the source."""
    rec = t2i_example_to_record(_example(), resolution=32, caption_mode="short")
    shards = {
        "a.tar": [rec] + [None] * 100,  # yields, then a rejected run past the cap
        "b.tar": [rec],
    }

    ds = FineT2IStreamingDataset(
        repo_id="ma-xu/fine-t2i", resolution=32, shuffle_buffer=0,
        # A source that only ever yields once per shard must never trip the fatal
        # guard: max_barren_shards=1 would fire immediately if such a shard counted.
        max_screened_per_shard=10, max_barren_shards=1,
    )
    ds._fs = None
    ds._iter_shard = lambda path: iter(shards[path])  # type: ignore[method-assign]

    stream = ds._stream_files_forever(["a.tar", "b.tar"], seed_offset=0, tag="s")
    assert next(stream) is rec
    assert next(stream) is rec, "a shard that yielded is exhausted, not barren"


def test_source_with_no_usable_partition_raises_instead_of_streaming_forever():
    """The failure that hung the 1024 monet-image runs, now bounded.

    _stream_files_forever skips None internally, so the interleaver's next() on a
    source that never yields would never return and the worker would download at full
    line rate forever, producing no batch and no error.
    """
    ds = FineT2IStreamingDataset(
        repo_id="ma-xu/fine-t2i", resolution=32, shuffle_buffer=0,
        min_short_side=4096,
        max_screened_per_shard=5, max_barren_shards=3,
    )
    ds._fs = None
    ds._iter_shard = lambda path: iter([None] * 50)  # type: ignore[method-assign]

    stream = ds._stream_files_forever(["a.tar", "b.tar", "c.tar"], seed_offset=0, tag="cc12m")
    with pytest.raises(UnusableSourceError) as excinfo:
        next(stream)

    msg = str(excinfo.value)
    assert "cc12m" in msg, msg
    assert "min_short_side=4096" in msg, msg


def test_a_yield_resets_the_barren_shard_streak():
    """Occasional barren shards must never accumulate into a false fatal."""
    rec = t2i_example_to_record(_example(), resolution=32, caption_mode="short")
    shards = {"bad.tar": [None] * 50, "good.tar": [rec]}
    ds = FineT2IStreamingDataset(
        repo_id="ma-xu/fine-t2i", resolution=32, shuffle_buffer=0,
        max_screened_per_shard=5, max_barren_shards=2,
    )
    ds._fs = None
    ds._iter_shard = lambda path: iter(list(shards[path]))  # type: ignore[method-assign]

    # bad, good, bad, good, bad: the streak never reaches 2 unbroken with no output.
    order = ["bad.tar", "good.tar", "bad.tar", "good.tar", "bad.tar"]
    got = list(islice(ds._stream_files_forever(order, seed_offset=0, tag="mixed"), 2))
    assert all(g is rec for g in got)


def test_unusable_source_is_not_mistaken_for_a_transient_network_error():
    """It is raised inside the transient-error handler, which classifies by substring.

    A retry classification here would swallow the guard and restore the hang.
    """
    from boomer.datasets.wds_latents import _is_transient_stream_error

    exc = UnusableSourceError(
        "[t2i-direct] screened 2000 rows from cc12m without a single one passing "
        "the record filters (min_short_side=1024, min_aesthetic=None)."
    )
    assert not _is_transient_stream_error(exc)


def test_the_guard_does_not_fire_on_a_merely_selective_source():
    """A source that accepts rarely is slow, not broken; it must keep streaming."""
    rec = t2i_example_to_record(_example(), resolution=32, caption_mode="short")
    assert rec is not None

    ds = FineT2IStreamingDataset(
        repo_id="ma-xu/fine-t2i", resolution=32, shuffle_buffer=0,
        max_screened_per_shard=100,
    )
    ds._fs = None
    # 99 rejects, then an accept: under the threshold, so no raise.
    ds._iter_shard = lambda path: iter([None] * 99 + [rec])  # type: ignore[method-assign]

    got = next(ds._stream_files_forever(["a.tar"], seed_offset=0, tag="picky"))
    assert got is rec
