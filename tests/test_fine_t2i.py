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
