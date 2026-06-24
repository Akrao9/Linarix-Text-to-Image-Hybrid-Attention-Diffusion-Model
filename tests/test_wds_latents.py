import io
import json

import numpy as np
import pytest
import torch

from boomer.datasets.wds_latents import (
    WdsCacheManifest,
    sample_to_record,
    split_shuffle_buffer_for_worker,
    split_urls_for_worker,
)
from tools.cache_hf_wds_to_boomer_wds import npy_bytes, raw_bytes, storage_dtype_to_numpy


def _sample(**fields):
    payload = {
        "caption.txt": "hello",
        "meta.json": json.dumps({"index": 0}).encode("utf-8"),
        "__url__": "train-000000.tar",
    }
    payload.update(fields)
    return payload


def test_split_urls_for_worker():
    urls = [f"shard-{index:03d}.tar" for index in range(8)]
    assert split_urls_for_worker(urls, 0, 4) == ["shard-000.tar", "shard-004.tar"]
    assert split_urls_for_worker(urls, 1, 4) == ["shard-001.tar", "shard-005.tar"]
    assert split_urls_for_worker(urls, 3, 4) == ["shard-003.tar", "shard-007.tar"]
    assert split_urls_for_worker(urls, 0, 1) == urls


def test_split_shuffle_buffer_for_worker_treats_value_as_global_budget():
    assert split_shuffle_buffer_for_worker(2000, 1) == 2000
    assert split_shuffle_buffer_for_worker(2000, 6) == 334
    assert split_shuffle_buffer_for_worker(3, 6) == 1
    assert split_shuffle_buffer_for_worker(0, 6) == 0


def test_sample_to_record_npy_fp16():
    latents = np.random.randn(32, 16, 16).astype(np.float16)
    text = np.random.randn(300, 1152).astype(np.float16)
    mask = np.zeros((300,), dtype=np.uint8)
    mask[:10] = 1
    record = sample_to_record(
        _sample(
            **{
                "latents.npy": npy_bytes(latents),
                "text.npy": npy_bytes(text),
                "text_mask.npy": npy_bytes(mask),
            }
        )
    )
    assert record.image_latent.shape == (32, 16, 16)
    assert record.text_embedding.shape == (300, 1152)
    assert record.attention_mask.shape == (300,)
    assert torch.allclose(record.image_latent, torch.from_numpy(latents), atol=0, rtol=0)


def test_sample_to_record_npy_fp8_text():
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("float8 not available in this PyTorch build")
    latents = np.random.randn(32, 16, 16).astype(np.float16)
    text_fp16 = torch.randn(300, 1152, dtype=torch.float16)
    text_fp8 = text_fp16.to(torch.float8_e4m3fn).view(torch.uint8).cpu().numpy()
    mask = np.zeros((300,), dtype=np.uint8)
    manifest = WdsCacheManifest(
        latents_shape=(32, 16, 16),
        text_shape=(300, 1152),
        text_dtype="float8_e4m3fn",
    )
    record = sample_to_record(
        _sample(
            **{
                "latents.npy": npy_bytes(latents),
                "text.npy": npy_bytes(text_fp8),
                "text_mask.npy": npy_bytes(mask),
            }
        ),
        manifest=manifest,
    )
    assert record.text_embedding.dtype == torch.bfloat16
    assert record.text_embedding.shape == (300, 1152)


def test_sample_to_record_npy_bf16_with_manifest():
    manifest = WdsCacheManifest(
        latents_shape=(32, 16, 16),
        text_shape=(300, 1152),
        latents_dtype="bfloat16",
        text_dtype="bfloat16",
    )
    latents = torch.randn(32, 16, 16, dtype=torch.float32)
    text = torch.randn(300, 1152, dtype=torch.float32)
    mask = np.zeros((300,), dtype=np.uint8)
    record = sample_to_record(
        _sample(
            **{
                "latents.npy": npy_bytes(storage_dtype_to_numpy(latents, "bfloat16")),
                "text.npy": npy_bytes(storage_dtype_to_numpy(text, "bfloat16")),
                "text_mask.npy": npy_bytes(mask),
            }
        ),
        manifest=manifest,
    )
    assert record.image_latent.dtype == torch.bfloat16
    assert record.text_embedding.dtype == torch.bfloat16
    assert torch.allclose(record.image_latent.float(), latents.to(torch.bfloat16).float(), atol=0, rtol=0)


def test_sample_to_record_npy_bf16_rejects_wrong_physical_dtype():
    manifest = WdsCacheManifest(
        latents_shape=(32, 16, 16),
        text_shape=(300, 1152),
        latents_dtype="bfloat16",
        text_dtype="bfloat16",
    )
    latents = np.random.randn(32, 16, 16).astype(np.float16)
    text = storage_dtype_to_numpy(torch.randn(300, 1152), "bfloat16")
    mask = np.zeros((300,), dtype=np.uint8)
    with pytest.raises(ValueError, match="bfloat16 cache payload must be stored as uint16"):
        sample_to_record(
            _sample(
                **{
                    "latents.npy": npy_bytes(latents),
                    "text.npy": npy_bytes(text),
                    "text_mask.npy": npy_bytes(mask),
                }
            ),
            manifest=manifest,
        )


def test_sample_to_record_raw_bin_with_manifest():
    manifest = WdsCacheManifest(
        raw_tensors=True,
        latents_shape=(32, 16, 16),
        text_shape=(300, 1152),
        text_dtype="float16",
    )
    latents = np.random.randn(32, 16, 16).astype(np.float16)
    text = np.random.randn(300, 1152).astype(np.float16)
    mask = np.zeros((300,), dtype=np.uint8)
    record = sample_to_record(
        _sample(
            **{
                "latents.bin": raw_bytes(latents),
                "text.bin": raw_bytes(text),
                "text_mask.bin": raw_bytes(mask),
            }
        ),
        manifest=manifest,
    )
    assert record.image_latent.shape == (32, 16, 16)
    assert record.text_embedding.shape == (300, 1152)
    assert record.attention_mask.shape == (300,)


def test_sample_to_record_raw_bin_bf16_with_manifest():
    manifest = WdsCacheManifest(
        raw_tensors=True,
        latents_shape=(32, 16, 16),
        text_shape=(300, 1152),
        latents_dtype="bfloat16",
        text_dtype="bfloat16",
    )
    latents = torch.randn(32, 16, 16, dtype=torch.float32)
    text = torch.randn(300, 1152, dtype=torch.float32)
    mask = np.zeros((300,), dtype=np.uint8)
    record = sample_to_record(
        _sample(
            **{
                "latents.bin": raw_bytes(storage_dtype_to_numpy(latents, "bfloat16")),
                "text.bin": raw_bytes(storage_dtype_to_numpy(text, "bfloat16")),
                "text_mask.bin": raw_bytes(mask),
            }
        ),
        manifest=manifest,
    )
    assert record.image_latent.dtype == torch.bfloat16
    assert record.text_embedding.dtype == torch.bfloat16


def test_sample_to_record_bin_requires_manifest():
    latents = np.random.randn(32, 16, 16).astype(np.float16)
    with pytest.raises(ValueError, match="manifest.json"):
        sample_to_record(_sample(**{"latents.bin": raw_bytes(latents), "text.bin": b"", "text_mask.bin": b""}))
