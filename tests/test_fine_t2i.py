"""--t2i-direct: single-image WebDataset streaming — square crop, caption
policy (short/enhanced sampling), filters, collate contract, and the streamed
iteration through a mocked HF datasets stream."""

from __future__ import annotations

import random
import sys
from itertools import islice
from pathlib import Path

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


def test_stream_iteration_through_mocked_datasets(monkeypatch) -> None:
    examples = [_example(prompt=f"cap {i}", enhanced=f"enhanced {i}", key=f"k{i}") for i in range(5)]

    ds = FineT2IStreamingDataset(resolution=32, shuffle_buffer=0, caption_mode="short")
    # Bypass the real HF stream: _load_stream returns our in-memory examples.
    monkeypatch.setattr(ds, "_load_stream", lambda: list(examples))

    records = list(islice(iter(ds), 5))
    assert len(records) == 5
    assert {r.caption for r in records} == {f"cap {i}" for i in range(5)}
    for r in records:
        assert r.pixels.shape == (3, 32, 32)
        assert r.pixels.dtype == torch.uint8


def test_worker_shard_fallback_is_disjoint(monkeypatch) -> None:
    """With too few shards to split by file (a plain list has none), workers must
    still get DISJOINT rows via the row-stride fallback — no duplication, no crash."""
    from types import SimpleNamespace

    import boomer.datasets.fine_t2i as m

    examples = [_example(prompt=f"cap {i}", key=f"k{i}") for i in range(10)]

    def run_worker(worker_id, num_workers):
        ds = FineT2IStreamingDataset(resolution=16, shuffle_buffer=0, caption_mode="short")
        monkeypatch.setattr(ds, "_load_stream", lambda: list(examples))
        monkeypatch.setattr(
            m, "get_worker_info",
            lambda: SimpleNamespace(id=worker_id, num_workers=num_workers),
        )
        return [r.caption for r in ds]

    w0 = run_worker(0, 2)
    w1 = run_worker(1, 2)
    assert set(w0).isdisjoint(w1)  # no row seen by two workers
    assert sorted(w0 + w1) == sorted(f"cap {i}" for i in range(10))  # every row covered


def test_dataset_validation() -> None:
    with pytest.raises(ValueError, match="caption_mode"):
        FineT2IStreamingDataset(caption_mode="verbose")
    with pytest.raises(ValueError, match="short_prob"):
        FineT2IStreamingDataset(short_prob=1.5)
