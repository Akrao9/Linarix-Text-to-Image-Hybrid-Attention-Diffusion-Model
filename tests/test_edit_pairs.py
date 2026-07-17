"""--edit-direct: OmniEdit pair streaming — shared edit-aware crop, row decode,
collate contract, and parquet projection through the real dataset class."""

from __future__ import annotations

import io
import random
import sys
from itertools import islice
from pathlib import Path

import pytest
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from boomer.datasets.edit_pairs import (  # noqa: E402
    EditPairsStreamingDataset,
    EditRecord,
    collate_edit_records,
    edit_row_to_record,
    register_source_to_target,
    shared_edit_crop_box,
)


def _png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _pair(width=192, height=96, edit_box=(150, 20, 180, 60)) -> tuple[Image.Image, Image.Image]:
    """A wide gray source and a target differing only inside ``edit_box``."""
    src = Image.new("RGB", (width, height), (128, 128, 128))
    tgt = src.copy()
    patch = Image.new("RGB", (edit_box[2] - edit_box[0], edit_box[3] - edit_box[1]), (255, 0, 0))
    tgt.paste(patch, (edit_box[0], edit_box[1]))
    return src, tgt


def _row(src: Image.Image, tgt: Image.Image, **overrides) -> dict:
    row = {
        "src_img": {"bytes": _png_bytes(src), "path": "s.png"},
        "edited_img": {"bytes": _png_bytes(tgt), "path": "t.png"},
        "edited_prompt_list": ["add red box", "Add a large red box to the right side."],
        "task": "addition",
        "o_score": 9.0,
        "omni_edit_id": "row-1",
    }
    row.update(overrides)
    return row


def test_shared_crop_centers_on_the_edit() -> None:
    src, tgt = _pair()  # 192x96, edit at x in [150, 180)
    left, top, right, bottom = shared_edit_crop_box(src, tgt)
    assert (right - left) == (bottom - top) == 96  # square, short side
    assert 0 <= left and right <= 192
    # A naive center crop ([48, 144)) would cut the edit off entirely; the
    # edit-aware crop must contain the edited region.
    assert left <= 150 and right >= 180

    # Identical pair: falls back to the center crop.
    left, _, right, _ = shared_edit_crop_box(src, src.copy())
    assert (left, right) == (48, 144)

    # Square frame: full-frame window, no crop decision to make.
    square = src.resize((96, 96))
    assert shared_edit_crop_box(square, square) == (0, 0, 96, 96)


def test_row_decode_shapes_prompts_and_alignment() -> None:
    src, tgt = _pair()
    rng = random.Random(0)
    rec = edit_row_to_record(_row(src, tgt), resolution=64, rng=rng)
    assert isinstance(rec, EditRecord)
    assert rec.source_pixels.shape == rec.target_pixels.shape == (3, 64, 64)
    assert rec.source_pixels.dtype == torch.uint8
    assert rec.caption in ("add red box", "Add a large red box to the right side.")
    assert rec.metadata["task"] == "addition"
    # Identical crop on both sides: outside the edit the pair must match exactly,
    # inside it the red patch must actually be visible in the target.
    diff = (rec.target_pixels.int() - rec.source_pixels.int()).abs().sum(dim=0)
    assert float(diff.sum()) > 0
    assert float((diff > 0).float().mean()) < 0.5  # the edit, not a global shift

    assert edit_row_to_record(_row(src, tgt), resolution=64, prompt_pick="first").caption == "add red box"
    assert edit_row_to_record(
        _row(src, tgt), resolution=64, prompt_pick="last"
    ).caption == "Add a large red box to the right side."


def test_row_filters() -> None:
    src, tgt = _pair()
    row = _row(src, tgt)
    assert edit_row_to_record(row, resolution=64, min_o_score=9.5) is None
    assert edit_row_to_record(row, resolution=64, tasks=("removal",)) is None
    assert edit_row_to_record(row, resolution=64, tasks=("addition",)) is not None
    assert edit_row_to_record(row, resolution=64, min_short_side=128) is None
    assert edit_row_to_record(_row(src, tgt, edited_prompt_list=[]), resolution=64) is None
    # Irreconcilable aspect ratios (>1.5x apart) are dropped.
    assert edit_row_to_record(
        _row(src, tgt.resize((60, 96))), resolution=64  # 2.0 vs 0.625 aspect
    ) is None
    # Raw bytes (no {bytes,path} struct) are tolerated.
    raw = _row(src, tgt)
    raw["src_img"] = _png_bytes(src)
    raw["edited_img"] = _png_bytes(tgt)
    assert edit_row_to_record(raw, resolution=64) is not None


def test_size_mismatched_pair_is_registered_not_dropped() -> None:
    """OmniEdit's common case: source at original res, edited at generator res
    (same aspect). The source is registered onto the target grid, and the
    edit-aware crop still lands on the edited region."""
    src, tgt = _pair()  # 192x96
    big_src = src.resize((384, 192), Image.BICUBIC)  # 2x original-res source
    rec = edit_row_to_record(_row(big_src, tgt), resolution=64, prompt_pick="first")
    assert rec is not None
    diff = (rec.target_pixels.int() - rec.source_pixels.int()).abs().sum(dim=0)
    assert float(diff.sum()) > 0  # the edit is visible
    assert float((diff > 20).float().mean()) < 0.5  # aligned: no global shift

    # register helper directly: same-aspect -> pure resize to target dims.
    registered = register_source_to_target(big_src, tgt)
    assert registered.size == tgt.size
    # mild aspect mismatch -> center-crop + resize, still target dims.
    slightly_wide = src.resize((240, 96), Image.BICUBIC)
    registered = register_source_to_target(slightly_wide, tgt)
    assert registered.size == tgt.size
    # irreconcilable aspect -> None.
    assert register_source_to_target(src.resize((96, 192)), tgt) is None
    # exact match -> returned unchanged.
    assert register_source_to_target(src, tgt) is src


def test_collate_contract_for_training_step() -> None:
    src, tgt = _pair()
    recs = [edit_row_to_record(_row(src, tgt), resolution=32) for _ in range(3)]
    batch = collate_edit_records(recs)
    assert batch["source_pixels"].shape == (3, 3, 32, 32)
    assert batch["target_pixels"].shape == (3, 3, 32, 32)
    assert batch["source_pixels"].dtype == torch.uint8
    assert len(batch["caption"]) == 3
    # No image_latent / text_embedding keys: training_step's .get() must route to
    # on-the-fly DC-AE + Qwen encoding.
    assert "image_latent" not in batch
    assert "text_embedding" not in batch


def test_parquet_stream_end_to_end(tmp_path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    n = 4
    image_type = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
    rows_src, rows_tgt, prompts = [], [], []
    for i in range(n):
        src, tgt = _pair(edit_box=(150, 20, 180, 40 + 8 * i))
        rows_src.append({"bytes": _png_bytes(src), "path": f"s{i}.png"})
        rows_tgt.append({"bytes": _png_bytes(tgt), "path": f"t{i}.png"})
        prompts.append([f"edit {i}", f"detailed edit {i}"])
    table = pa.table({
        "omni_edit_id": pa.array([f"id{i}" for i in range(n)]),
        "task": pa.array(["addition"] * n),
        "src_img": pa.array(rows_src, type=image_type),
        "edited_img": pa.array(rows_tgt, type=image_type),
        "edited_prompt_list": pa.array(prompts, type=pa.list_(pa.string())),
        "o_score": pa.array([9.0] * n),
        "width": pa.array([192] * n, type=pa.int64()),
        "height": pa.array([96] * n, type=pa.int64()),
        "sc_reasoning": pa.array(["unused column"] * n),
    })
    shard = tmp_path / "train-00000.parquet"
    pq.write_table(table, shard)

    ds = EditPairsStreamingDataset(
        paths=str(shard), shuffle_buffer=0, resolution=32, prompt_pick="first",
        rank=0, world_size=1,
    )
    import pyarrow.parquet as _pq

    schema = _pq.read_schema(str(shard))
    assert ds._shard_usable(str(shard), schema)
    want = ds._wanted_columns(set(schema.names))
    assert "src_img" in want and "edited_img" in want and "edited_prompt_list" in want
    assert "sc_reasoning" not in want  # projection drops dead-weight columns

    records = list(islice(iter(ds), n))
    assert len(records) == n
    for rec in sorted(records, key=lambda r: r.caption):
        assert rec.source_pixels.shape == (3, 32, 32)
        assert rec.target_pixels.dtype == torch.uint8
        assert rec.caption.startswith("edit ")
    # A shard lacking the pair columns is skipped, not crashed on.
    bad = pa.table({"foo": pa.array([1, 2])})
    bad_path = tmp_path / "bad.parquet"
    pq.write_table(bad, bad_path)
    assert not ds._shard_usable(str(bad_path), _pq.read_schema(str(bad_path)))


def test_prompt_pick_validation() -> None:
    with pytest.raises(ValueError, match="prompt_pick"):
        EditPairsStreamingDataset(paths="x.parquet", prompt_pick="detailed")
