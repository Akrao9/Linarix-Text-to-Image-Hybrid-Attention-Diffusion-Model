"""Regression tests for the v6-branch review fixes."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from boomer.datasets.monet_direct import monet_row_to_record  # noqa: E402
from boomer.models.boomer_fla_dit import (  # noqa: E402
    BoomerFLADiTConfig,
    BoomerFLAJointRefreshBlock,
    migrate_glumbconv_state_dict,
)
from boomer.models.factory import create_model  # noqa: E402


def _v6_kwargs(**overrides):
    base = dict(
        hidden_dim=64,
        depth=4,
        num_heads=4,
        mlp_ratio=2.0,
        mixer_type="fallback",
        cross_attention_backend="sdpa",
        image_attention_every=2,
        dual_stream_depth=1,
        multimodal_coord_ids=True,
        joint_refresh_attention=True,
        use_abs_pos_embed=False,
    )
    base.update(overrides)
    return base


# --------------------------------------------------------------------------- #
# GLUMBConv legacy-key migration (EMA shadows, export state)
# --------------------------------------------------------------------------- #
def test_migrate_glumbconv_state_dict_renames_and_reshapes() -> None:
    state = {
        "blocks.0.mlp.inverted_conv.conv.weight": torch.randn(256, 64, 1, 1),
        "blocks.0.mlp.inverted_conv.conv.bias": torch.randn(256),
        "blocks.0.mlp.point_conv.conv.weight": torch.randn(64, 128, 1, 1),
        "blocks.0.norm1.weight": torch.randn(64),  # untouched bystander
    }
    n = migrate_glumbconv_state_dict(state)
    assert n == 3
    assert state["blocks.0.mlp.inverted_fc.0.weight"].shape == (256, 64)
    assert state["blocks.0.mlp.inverted_fc.0.bias"].shape == (256,)
    assert state["blocks.0.mlp.point_fc.weight"].shape == (64, 128)
    assert "blocks.0.mlp.inverted_conv.conv.weight" not in state
    assert "blocks.0.norm1.weight" in state
    # idempotent on already-migrated dicts
    assert migrate_glumbconv_state_dict(state) == 0


def test_legacy_checkpoint_loads_through_module_hook() -> None:
    model = create_model(BoomerFLADiTConfig(**_v6_kwargs(joint_refresh_attention=False)))
    state = model.state_dict()
    legacy = {}
    for key, value in state.items():
        if key.endswith("mlp.inverted_fc.0.weight"):
            key = key.replace("inverted_fc.0.weight", "inverted_conv.conv.weight")
            value = value.unsqueeze(-1).unsqueeze(-1)
        elif key.endswith("mlp.inverted_fc.0.bias"):
            key = key.replace("inverted_fc.0.bias", "inverted_conv.conv.bias")
        elif key.endswith("mlp.point_fc.weight"):
            key = key.replace("point_fc.weight", "point_conv.conv.weight")
            value = value.unsqueeze(-1).unsqueeze(-1)
        legacy[key] = value
    missing, unexpected = model.load_state_dict(legacy, strict=True)
    assert not missing and not unexpected


# --------------------------------------------------------------------------- #
# Sandwich RMSNorm no longer wraps exactly-zero projections
# --------------------------------------------------------------------------- #
def test_sandwich_rmsnorm_reinits_wrapped_projections() -> None:
    sandwich = create_model(BoomerFLADiTConfig(**_v6_kwargs(sandwich_rmsnorm=True)))
    plain = create_model(BoomerFLADiTConfig(**_v6_kwargs(sandwich_rmsnorm=False)))

    # sandwich on: wrapped projections carry a small normal init (norm of an
    # exactly-zero output would amplify grads by rsqrt(eps) and snap on step 1)
    assert sandwich.blocks[2].mlp.point_fc.weight.abs().sum() > 0
    assert sandwich.blocks[2].cross_attn.proj.weight.abs().sum() > 0
    refresh = sandwich.blocks[1]
    assert isinstance(refresh, BoomerFLAJointRefreshBlock)
    assert refresh.image_out_proj.weight.abs().sum() > 0
    assert refresh.text_out_proj.weight.abs().sum() > 0
    # the output head keeps the zero init either way (zero output at init)
    assert torch.all(sandwich.out_proj.weight == 0)

    # sandwich off: classic adaLN-Zero zero-init is preserved exactly
    assert torch.all(plain.blocks[2].mlp.point_fc.weight == 0)
    assert torch.all(plain.blocks[2].cross_attn.proj.weight == 0)
    assert torch.all(plain.blocks[1].image_out_proj.weight == 0)


# --------------------------------------------------------------------------- #
# monet-direct: corrupt rows are dropped, not fatal
# --------------------------------------------------------------------------- #
def _good_row(**overrides):
    row = {
        "dcae_f32c32": [0.5] * (32 * 4 * 4),
        "caption_short": "a red bicycle",
        "__key__": "row-1",
        "source": "test",
    }
    row.update(overrides)
    return row


def test_corrupt_latent_row_is_dropped_not_fatal() -> None:
    good = monet_row_to_record(_good_row(), latent_key="dcae_f32c32",
                               caption_keys=("caption_short",))
    assert good is not None

    truncated = monet_row_to_record(
        _good_row(dcae_f32c32=[0.5] * 100), latent_key="dcae_f32c32",
        caption_keys=("caption_short",),
    )
    assert truncated is None

    ragged = monet_row_to_record(
        _good_row(dcae_f32c32=[[0.5, 0.1], [0.2]] * 256), latent_key="dcae_f32c32",
        caption_keys=("caption_short",),
    )
    assert ragged is None


def test_corrupt_dino_keeps_row_drops_embedding() -> None:
    record = monet_row_to_record(
        _good_row(dino=[[0.1], [0.2, 0.3]]), latent_key="dcae_f32c32",
        caption_keys=("caption_short",), dino_key="dino",
    )
    assert record is not None
    assert record.dino_embedding is None


def test_iter_shard_numpy_decode_path(tmp_path) -> None:
    # The list<double> columns take the vectorized to_numpy path (no per-row float
    # boxing); confirm rows still decode to correct latent shapes end to end.
    import pyarrow as pa
    import pyarrow.parquet as pq
    from boomer.datasets.monet_direct import MonetDirectStreamingDataset

    n = 4
    table = pa.table({
        "dcae_f32c32": pa.array([[float(i)] * (32 * 4 * 4) for i in range(n)],
                                type=pa.list_(pa.float64())),
        "caption_short": pa.array([f"caption {i}" for i in range(n)]),
        "__key__": pa.array([f"k{i}" for i in range(n)]),
    })
    shard = tmp_path / "shard.parquet"
    pq.write_table(table, shard)

    ds = MonetDirectStreamingDataset(
        paths=str(shard), latent_key="dcae_f32c32", caption_keys=("caption_short",),
        dino_key=None, read_batch_size=2,
    )
    records = [r for r in ds._iter_shard(str(shard)) if r is not None]
    assert len(records) == n
    assert all(tuple(r.image_latent.shape) == (32, 4, 4) for r in records)
    assert {r.caption for r in records} == {f"caption {i}" for i in range(n)}


# --------------------------------------------------------------------------- #
# train.py callbacks (imported lazily: pulls in lightning)
# --------------------------------------------------------------------------- #
def test_checkpoint_retention_orders_by_step_number() -> None:
    import train

    names = ["boomer-step=10000.ckpt", "boomer-step=5000.ckpt", "boomer-step=9000.ckpt"]
    ordered = sorted(names, key=train.CheckpointUploadCallback._step_key)
    assert ordered == ["boomer-step=5000.ckpt", "boomer-step=9000.ckpt", "boomer-step=10000.ckpt"]


def test_ema_restore_migrates_legacy_glumbconv_keys() -> None:
    import train

    cb = train.EMACallback(decay=0.999)
    checkpoint = {
        "ema": {
            "blocks.0.mlp.point_conv.conv.weight": torch.randn(64, 128, 1, 1),
            "blocks.0.norm1.weight": torch.randn(64),
        }
    }
    cb.on_load_checkpoint(None, None, checkpoint)
    assert "blocks.0.mlp.point_fc.weight" in cb.ema
    assert cb.ema["blocks.0.mlp.point_fc.weight"].shape == (64, 128)
    assert "blocks.0.mlp.point_conv.conv.weight" not in cb.ema


def test_optimizer_moments_reshaped_to_current_params() -> None:
    import train

    class FakeSelf:
        model = nn.Sequential(nn.Linear(8, 4, bias=False), nn.Linear(4, 2, bias=False))

    fake = FakeSelf()
    checkpoint = {
        "optimizer_states": [
            {
                "param_groups": [{"params": [0, 1]}],
                "state": {
                    0: {"step": torch.tensor(10.0),
                        "exp_avg": torch.randn(4, 8, 1, 1),
                        "exp_avg_sq": torch.randn(4, 8, 1, 1)},
                    1: {"step": torch.tensor(10.0),
                        "exp_avg": torch.randn(2, 4),
                        "exp_avg_sq": torch.randn(2, 4)},
                },
            }
        ]
    }
    train.BoomerLit.on_load_checkpoint(fake, checkpoint)
    state = checkpoint["optimizer_states"][0]["state"]
    assert state[0]["exp_avg"].shape == (4, 8)
    assert state[0]["exp_avg_sq"].shape == (4, 8)
    assert state[1]["exp_avg"].shape == (2, 4)  # untouched
