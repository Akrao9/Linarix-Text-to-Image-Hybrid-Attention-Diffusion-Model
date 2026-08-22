"""MONET full-resolution WebDataset streaming tests."""

from __future__ import annotations

import io
import json
import random
import sys
import tarfile
from itertools import islice
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from boomer.datasets.monet_direct import MONET_CAPTION_KEYS  # noqa: E402
from boomer.datasets.monet_image_direct import (  # noqa: E402
    MonetImageStreamingDataset,
    collate_monet_image_records,
    monet_image_example_to_record,
)


def _jpeg(width: int = 1200, height: int = 800) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (40, 120, 200)).save(buffer, format="JPEG")
    return buffer.getvalue()


def _example(**overrides):
    meta = {
        "caption_gemini-2.5-flash-lite": "gemini caption",
        "caption_original": "original prompt",
        "width": 1200,
        "height": 800,
        "source": "laion",
        "aesthetic_laion-improved": 6.0,
        "nsfw_jasperai": 0.1,
        "wk_jasperai": 0.05,
    }
    meta.update(overrides.pop("meta", {}))
    example = {"jpg": _jpeg(), "json": meta, "__key__": "sample-0"}
    example.update(overrides)
    return example


def test_record_decodes_full_image_and_caption() -> None:
    record = monet_image_example_to_record(_example(), resolution=256)
    assert record is not None
    assert record.pixels.shape == (3, 256, 256)
    assert record.pixels.dtype == torch.uint8
    assert record.caption == "gemini caption"
    assert record.metadata["caption_key"] == "caption_gemini-2.5-flash-lite"
    assert record.metadata["size"] == (1200, 800)


def test_caption_policy_and_quality_filters() -> None:
    original_only = (0.0, 0.0, 0.0, 0.0, 1.0)
    record = monet_image_example_to_record(
        _example(),
        resolution=64,
        caption_probs=original_only,
        rng=random.Random(0),
    )
    assert record is not None and record.caption == "original prompt"

    assert monet_image_example_to_record(
        _example(), resolution=64, min_aesthetic=7.0
    ) is None
    assert monet_image_example_to_record(
        _example(), resolution=64, max_nsfw=0.05
    ) is None
    assert monet_image_example_to_record(
        _example(), resolution=64, max_watermark=0.01
    ) is None
    assert monet_image_example_to_record(
        _example(), resolution=1024, min_short_side=1024
    ) is None


def test_collate_contract_routes_pixels_to_vae() -> None:
    records = [
        monet_image_example_to_record(
            _example(__key__=f"sample-{index}"), resolution=32
        )
        for index in range(3)
    ]
    batch = collate_monet_image_records(records)
    assert batch["pixels"].shape == (3, 3, 32, 32)
    assert len(batch["caption"]) == 3
    assert batch["dino_embedding"] is None
    assert "image_latent" not in batch
    assert "text_embedding" not in batch


def _add_bytes(tar: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    tar.addfile(info, io.BytesIO(payload))


def test_raw_tar_stream_ignores_unused_pth_members(tmp_path) -> None:
    """A bogus .pth would fail decode(), but must be ignored by this raw reader."""
    path = tmp_path / "00000.tar"
    meta = {
        "caption_original": "a generation prompt",
        "width": 1200,
        "height": 800,
    }
    with tarfile.open(path, "w") as tar:
        _add_bytes(tar, "000001.jpg", _jpeg())
        _add_bytes(tar, "000001.json", json.dumps(meta).encode())
        _add_bytes(tar, "000001.embedding_clip-vit-base-patch32.pth", b"not-a-pth")
        _add_bytes(tar, "000001.embedding_vae-anything.pth", b"also-not-a-pth")

    dataset = MonetImageStreamingDataset(
        paths=str(path),
        resolution=32,
        shuffle_buffer=0,
        caption_probs=(0.0, 0.0, 0.0, 0.0, 1.0),
        stream_reconnect_max_retries=0,
    )
    record = next(dataset._iter_shard(str(path)))
    assert record is not None
    assert record.caption == "a generation prompt"
    assert record.pixels.shape == (3, 32, 32)


def test_per_glob_caption_policy_follows_weighted_source(monkeypatch) -> None:
    dataset = MonetImageStreamingDataset(
        paths="synthetic/*.tar,real/*.tar",
        glob_weights=(1.0, 1.0),
        caption_probs="0,0,0,0,1;.25,.25,.25,.25,0",
        resolution=16,
        shuffle_buffer=0,
    )
    synthetic = _example(
        meta={
            "caption_original": "synthetic original",
            "caption_gemini-2.5-flash-lite": "synthetic recaption",
        }
    )
    real = _example(
        meta={
            "caption_original": "real original",
            "caption_gemini-2.5-flash-lite": "real recaption",
        }
    )
    monkeypatch.setattr(
        dataset,
        "_cached_shard_files",
        lambda source: ["synthetic.tar"] if source.startswith("synthetic") else ["real.tar"],
    )

    def stream(files, **_kwargs):
        example = synthetic if files == ["synthetic.tar"] else real
        while True:
            yield next(dataset._records_from_samples([example]))

    monkeypatch.setattr(dataset, "_stream_files_forever", stream)

    class Alternating:
        value = 0

        def choices(self, *_args, **_kwargs):
            result = [self.value]
            self.value = 1 - self.value
            return result

    records = list(
        islice(
            dataset._interleave_sources(offset=0, stride=1, rng=Alternating()),
            2,
        )
    )
    assert records[0].caption == "synthetic original"
    assert records[1].caption == "real recaption"


def test_caption_key_count_stays_compatible() -> None:
    assert len(MONET_CAPTION_KEYS) == 5


def test_cli_flags_parse() -> None:
    from train import build_parser

    args = build_parser().parse_args(
        [
            "--monet-image-direct",
            "--cache-wds",
            "hf://datasets/jasperai/monet/v1.2.0/synthetic/**/*.tar",
            "--monet-image-resolution",
            "1024",
            "--monet-image-min-short-side",
            "768",
            "--monet-weights",
            "1",
        ]
    )
    assert args.monet_image_direct is True
    assert args.monet_image_resolution == 1024
    assert args.monet_image_min_short_side == 768
