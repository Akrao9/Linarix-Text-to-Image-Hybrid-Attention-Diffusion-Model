"""--cache-captions-only: stream a Boomer cache's latents + raw captions while
ignoring its baked text/text_mask (built with a mismatched encoder, e.g. the
finet2i-1024 cache's 2048-d text vs v6's 1536-d) so train.py encodes on the fly."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from boomer.datasets.latent_shards import LatentRecord, collate_latent_records  # noqa: E402
from boomer.datasets.wds_latents import (  # noqa: E402
    ParquetStreamingLatentCacheDataset,
    sample_to_record,
)


def _npy_bytes(array: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, array)
    return buf.getvalue()


def _sample(with_text: bool = True) -> dict:
    sample = {
        "__url__": "",
        "latents.npy": _npy_bytes(np.random.rand(32, 32, 32).astype(np.float32)),
        "caption.txt": "a lighthouse at dusk",
        "meta.json": '{"index": 7}',
    }
    if with_text:
        sample["text.npy"] = _npy_bytes(np.random.rand(320, 2048).astype(np.float16))
        sample["text_mask.npy"] = _npy_bytes(np.ones(320, dtype=np.uint8))
    return sample


def test_sample_to_record_captions_only_skips_text() -> None:
    # text members present: still skipped (decode cost + wrong encoder)
    rec = sample_to_record(_sample(with_text=True), captions_only=True)
    assert rec.text_embedding is None and rec.attention_mask is None
    assert tuple(rec.image_latent.shape) == (32, 32, 32)
    assert rec.caption == "a lighthouse at dusk"

    # text members absent (parquet column projection): must not KeyError
    rec = sample_to_record(_sample(with_text=False), captions_only=True)
    assert rec.text_embedding is None
    assert rec.metadata == {"index": 7}

    # default path unchanged
    rec = sample_to_record(_sample(with_text=True))
    assert tuple(rec.text_embedding.shape) == (320, 2048)
    assert rec.attention_mask is not None


def test_collate_omits_text_batch_for_captions_only_records() -> None:
    def rec(text: torch.Tensor | None, mask: torch.Tensor | None) -> LatentRecord:
        return LatentRecord(
            image_latent=torch.randn(32, 4, 4),
            text_embedding=text,
            attention_mask=mask,
            caption="c",
            metadata={},
            path="",
        )

    batch = collate_latent_records([rec(None, None), rec(None, None)])
    assert batch["text_embedding"] is None  # -> training_step encode_captions branch
    assert batch["attention_mask"] is None
    assert batch["image_latent"].shape[0] == 2
    assert batch["caption"] == ["c", "c"]

    full = collate_latent_records(
        [rec(torch.randn(8, 16), torch.ones(8)) for _ in range(2)]
    )
    assert full["text_embedding"].shape == (2, 8, 16)
    assert full["attention_mask"].shape == (2, 8)


def test_parquet_captions_only_projects_columns_and_decodes(tmp_path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    n = 4
    table = pa.table({
        "__key__": pa.array([f"k{i}" for i in range(n)]),
        "latents": pa.array(
            [_npy_bytes(np.full((32, 32, 32), float(i), dtype=np.float32)) for i in range(n)],
            type=pa.binary(),
        ),
        "text": pa.array(
            [_npy_bytes(np.zeros((320, 2048), dtype=np.uint8)) for _ in range(n)],
            type=pa.binary(),
        ),
        "text_mask": pa.array(
            [_npy_bytes(np.ones(320, dtype=np.uint8)) for _ in range(n)],
            type=pa.binary(),
        ),
        "caption": pa.array([f"caption {i}" for i in range(n)]),
        "meta": pa.array(['{"index": %d}' % i for i in range(n)]),
    })
    shard = tmp_path / "train-000000.parquet"
    pq.write_table(table, shard)

    ds = ParquetStreamingLatentCacheDataset(
        paths=str(shard), shuffle_buffer=0, captions_only=True, read_batch_size=2
    )
    cols = ds._captions_only_columns([str(shard)])
    assert "text" not in cols and "text_mask" not in cols
    assert "latents" in cols and "caption" in cols

    stream = ds._make_stream(0, shuffle_buffer=0)
    records = [ds._row_to_record(row) for row in stream]
    assert len(records) == n
    for i, rec in enumerate(sorted(records, key=lambda r: r.caption)):
        assert rec.text_embedding is None and rec.attention_mask is None
        assert tuple(rec.image_latent.shape) == (32, 32, 32)
        assert rec.caption == f"caption {i}"
        assert rec.metadata == {"index": i}


def test_build_config_text_dims_fall_back_to_args_when_captions_only() -> None:
    import argparse

    import train

    record = LatentRecord(
        image_latent=torch.randn(32, 32, 32),
        text_embedding=None,
        attention_mask=None,
        caption="c",
        metadata={},
        path="",
    )
    args = argparse.Namespace(
        text_dim=1536, text_seq_len=300, latent_channels=32, latent_size=16
    )
    # only dim fields are consumed before BoomerFLADiTConfig defaults kick in
    cfg = train.build_config(_with_model_defaults(args), record)
    assert cfg.latent_channels == 32 and cfg.latent_size == 32  # from the record
    assert cfg.text_dim == 1536 and cfg.text_seq_len == 300  # from the encoder args


def _with_model_defaults(ns):
    """Fill the argparse fields build_config reads beyond the dims."""
    import train

    parser = train.build_parser() if hasattr(train, "build_parser") else None
    if parser is not None:
        defaults = parser.parse_args(["--cache-wds", "x"])
        for key, value in vars(ns).items():
            setattr(defaults, key, value)
        return defaults
    return ns
