"""Direct-monet streaming: row decode, collate, and on-the-fly text-encode plumbing."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from boomer.datasets.monet_direct import (  # noqa: E402
    MONET_CAPTION_KEYS,
    MONET_LATENT_KEY,
    collate_monet_records,
    flat_to_latent,
    monet_row_to_record,
)


def _row(**overrides):
    row = {
        MONET_LATENT_KEY: list(range(32 * 16 * 16)),
        "caption_gemini-2.5-flash-lite": "a gemini caption",
        "caption_original": "an original caption",
        "embedding_dinov2-vitg14": [0.5] * 1536,
        "aesthetic_laion-improved": 5.0,
        "nsfw_jasperai": 0.1,
        "wk_jasperai": 0.05,
        "__key__": "sample-1",
        "source": "cc12m",
    }
    row.update(overrides)
    return row


def test_flat_to_latent_reshape() -> None:
    latent = flat_to_latent([float(i) for i in range(32 * 16 * 16)])
    assert latent.shape == (32, 16, 16)
    assert latent.dtype == torch.float32
    assert latent[0, 0, 1] == 1.0  # row-major reshape

    try:
        flat_to_latent([0.0] * 100)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for a non-divisible latent length")


def test_row_decode_and_caption_priority() -> None:
    record = monet_row_to_record(_row())
    assert record is not None
    assert record.image_latent.shape == (32, 16, 16)
    assert record.caption == "a gemini caption"
    assert record.metadata["caption_key"] == "caption_gemini-2.5-flash-lite"
    assert record.dino_embedding is not None and record.dino_embedding.shape == (1536,)
    assert record.text_embedding.numel() == 0  # text is encoded on-GPU, not here

    # Null/absent preferred captions fall through the priority list.
    record = monet_row_to_record(_row(**{"caption_gemini-2.5-flash-lite": None}))
    assert record is not None and record.caption == "an original caption"

    assert monet_row_to_record(_row(**{
        "caption_gemini-2.5-flash-lite": None, "caption_original": "",
    })) is None
    assert monet_row_to_record(_row(**{MONET_LATENT_KEY: None})) is None


def test_row_quality_filters() -> None:
    assert monet_row_to_record(_row(), min_aesthetic=5.5) is None
    assert monet_row_to_record(_row(), min_aesthetic=4.5) is not None
    # Missing aesthetic fails a min-aesthetic filter (unknown quality is excluded).
    assert monet_row_to_record(_row(**{"aesthetic_laion-improved": None}), min_aesthetic=4.5) is None
    assert monet_row_to_record(_row(), max_nsfw=0.05) is None
    assert monet_row_to_record(_row(), max_watermark=0.01) is None
    # Missing nsfw/watermark scores pass (max filters only act on known values).
    assert monet_row_to_record(_row(**{"nsfw_jasperai": None}), max_nsfw=0.05) is not None
    assert monet_row_to_record(_row(), dino_key=None).dino_embedding is None


def test_collate_monet_records() -> None:
    records = [monet_row_to_record(_row(__key__=f"k{i}")) for i in range(3)]
    batch = collate_monet_records(records)
    assert batch["image_latent"].shape == (3, 32, 16, 16)
    assert batch["caption"] == ["a gemini caption"] * 3
    assert batch["dino_embedding"].shape == (3, 1536)
    assert "text_embedding" not in batch

    # A single dino-less record disables the REPA batch (never a ragged stack).
    records[1] = monet_row_to_record(_row(), dino_key=None)
    assert collate_monet_records(records)["dino_embedding"] is None


class _StubEncoder(nn.Module):
    """Stands in for QwenReferenceEncoder: (prompts, images) -> (hidden, mask)."""

    def __init__(self, hidden_dim: int, seq_len: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.model = nn.Linear(1, 1)  # device probe used by _resident_text_encoder

    def forward(self, prompts, images=None):
        batch = len(prompts)
        hidden = torch.randn(batch, self.seq_len, self.hidden_dim, dtype=torch.bfloat16)
        mask = torch.ones(batch, self.seq_len, dtype=torch.long)
        mask[:, self.seq_len // 2 :] = 0
        return hidden, mask


def test_training_step_encodes_captions_on_the_fly() -> None:
    from boomer.models.boomer_fla_dit import BoomerFLADiTConfig
    from train import BoomerLit

    config = BoomerFLADiTConfig(
        hidden_dim=64, depth=2, num_heads=4, mlp_ratio=2.0,
        mixer_type="fallback", cross_attention_backend="sdpa",
        text_dim=48, text_seq_len=24, use_abs_pos_embed=True,
    )
    module = BoomerLit(
        config,
        monet_direct=True,
        text_encoder_id="stub",
        text_encoder_layer=-2,
        text_encoder_max_length=24,
        text_encoder_instruction=False,
        lr=1e-4, weight_decay=0.0, beta1=0.9, beta2=0.95, fused_adamw=False,
        warmup_steps=0, max_steps=10, cond_dropout=0.5,
        t_sampler="uniform", t_sampler_final=None, t_sampler_final_step=0,
        t_bias=2.0, flow_shift=1.5, logit_mean=0.0, logit_std=1.0,
        plateau_low=0.0, plateau_high=1.0, stretch_q_low=0.0, stretch_q_high=1.0,
        timestep_weight=False, min_snr_gamma=0.0,
        latent_mean=0.0, latent_std=1.0,
        repa_weight=0.0, repa_anneal_steps=0,
        dispersive_weight=0.0, dispersive_temp=0.5, dispersive_metric="l2",
        dispersive_anneal_steps=0,
    )
    module._text_encoder_box.append(_StubEncoder(hidden_dim=48, seq_len=24))
    module.lr_schedulers = lambda: None  # not attached to a Trainer in this test

    batch = {
        "image_latent": torch.randn(2, 32, 16, 16),
        "caption": ["a cat", "a dog"],
        "dino_embedding": None,
    }
    # encode_captions keeps the Qwen hidden states bf16 (no fp32 upcast); in training
    # the Lightning bf16-mixed autocast (or FSDP2's bf16 compute policy) reconciles
    # them with the fp32 master weights, so mirror that here.
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = module.training_step(batch, 0)
    assert loss.ndim == 0 and torch.isfinite(loss)
    # encode_qwen_text_fixed runs under inference_mode; the clone escape must leave
    # ordinary tensors, or backward() raises "Inference tensors cannot be saved...".
    loss.backward()
    assert module.model.caption_embedder.y_proj[0].weight.grad is not None


def _write_monet_parquet(
    path, n: int, *, latent_as_string: bool = False, start: int = 0, prefix: str = "cap"
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    if latent_as_string:  # the ~9/713 cc12m shards that store the latent as a string
        latent = pa.array(["corrupt"] * n, pa.string())
    else:
        latent = pa.array([[float(i)] * (32 * 16 * 16) for i in range(n)], pa.list_(pa.float64()))
    table = pa.table({
        MONET_LATENT_KEY: latent,
        "caption_gemini-2.5-flash-lite": pa.array([f"{prefix} {start + i}" for i in range(n)]),
        "__key__": pa.array([f"k{start + i}" for i in range(n)]),
        "source": pa.array(["cc12m"] * n),
    })
    pq.write_table(table, str(path))


def test_per_file_reader_skips_string_latent_shard(tmp_path) -> None:
    import itertools

    from boomer.datasets.monet_direct import MonetDirectStreamingDataset

    # A good shard (list latents) and a string-latent shard in the same glob: the
    # string shard is what crashed the datasets path; here it must be skipped and the
    # good rows stream through unaffected.
    _write_monet_parquet(tmp_path / "aaa_good.parquet", 4, latent_as_string=False)
    _write_monet_parquet(tmp_path / "zzz_bad.parquet", 3, latent_as_string=True)
    ds = MonetDirectStreamingDataset(
        paths=f"{tmp_path}/*.parquet", shuffle_buffer=0, dino_key=None,
    )
    recs = list(itertools.islice(iter(ds), 4))  # exactly one epoch of the good shard
    assert len(recs) == 4
    assert all(r.image_latent.shape == (32, 16, 16) for r in recs)
    assert sorted(r.caption for r in recs) == ["cap 0", "cap 1", "cap 2", "cap 3"]


def test_per_file_reader_shuffle_buffer(tmp_path) -> None:
    import itertools

    from boomer.datasets.monet_direct import MonetDirectStreamingDataset

    _write_monet_parquet(tmp_path / "s.parquet", 16, latent_as_string=False)
    ds = MonetDirectStreamingDataset(
        paths=f"{tmp_path}/s.parquet", shuffle_buffer=8, dino_key=None, seed=0,
    )
    recs = list(itertools.islice(iter(ds), 16))
    # Reservoir shuffle must neither drop nor duplicate within an epoch's worth of rows.
    assert sorted(r.caption for r in recs) == sorted(f"cap {i}" for i in range(16))


def test_caption_probs_sampling() -> None:
    import collections

    import random as random_module

    # 3:1 weights between the two available captions; the zero-weight and absent keys
    # must never win. One rng seeds the whole sequence, so this is deterministic.
    keys = MONET_CAPTION_KEYS
    probs = (0.75, 0.0, 0.0, 0.25, 0.5)  # florence absent in _row(); original present
    row = _row(**{"caption_florence-2-large": None})
    rng = random_module.Random(0)
    counts: collections.Counter[str] = collections.Counter()
    for _ in range(400):
        record = monet_row_to_record(row, caption_probs=probs, rng=rng)
        counts[record.metadata["caption_key"]] += 1
    assert set(counts) == {"caption_gemini-2.5-flash-lite", "caption_original"}
    # Renormalized over the available pool: gemini 0.75/1.25 = 60%, original 40%.
    assert 0.5 < counts["caption_gemini-2.5-flash-lite"] / 400 < 0.7

    # All positive-weight captions empty => the row is dropped, even though a
    # zero-weight caption exists.
    only_zero_weight = _row(**{"caption_gemini-2.5-flash-lite": None})
    assert monet_row_to_record(
        only_zero_weight, caption_probs=(0.0, 0.0, 0.0, 0.0, 1.0), rng=rng,
    ).caption == "an original caption"
    assert monet_row_to_record(
        only_zero_weight, caption_probs=(1.0, 1.0, 1.0, 1.0, 0.0), rng=rng,
    ) is None
    assert len(keys) == len(probs)


def test_caption_probs_validation(tmp_path) -> None:
    from boomer.datasets.monet_direct import MonetDirectStreamingDataset

    _write_monet_parquet(tmp_path / "s.parquet", 2)
    try:
        MonetDirectStreamingDataset(paths=f"{tmp_path}/s.parquet", caption_probs="0.5,0.5")
    except ValueError as exc:
        assert "align 1:1" in str(exc)
    else:
        raise AssertionError("expected ValueError for a misaligned caption_probs length")
    try:
        MonetDirectStreamingDataset(
            paths=f"{tmp_path}/s.parquet", caption_probs="1,0,0,0,0;0,0,0,0,1",
        )
    except ValueError as exc:
        assert "glob_weights" in str(exc)
    else:
        raise AssertionError("expected ValueError for per-glob probs without glob_weights")


def test_rank_worker_shard_split(tmp_path) -> None:
    import itertools

    from boomer.datasets.monet_direct import MonetDirectStreamingDataset

    # 8 shards x 2 rows with globally unique captions. Two "ranks" (world_size=2, one
    # worker each) must read disjoint shard slices whose union is the full corpus.
    for i in range(8):
        _write_monet_parquet(tmp_path / f"{i}.parquet", 2, start=2 * i)
    seen: list[list[str]] = []
    for rank in range(2):
        ds = MonetDirectStreamingDataset(
            paths=f"{tmp_path}/*.parquet", shuffle_buffer=0, dino_key=None,
            rank=rank, world_size=2, seed=0,
        )
        recs = list(itertools.islice(iter(ds), 8))  # one epoch of this rank's slice
        seen.append(sorted(r.caption for r in recs))
    assert not set(seen[0]) & set(seen[1])
    assert sorted(seen[0] + seen[1]) == sorted(f"cap {i}" for i in range(16))


def test_weighted_glob_interleave(tmp_path) -> None:
    import itertools

    from boomer.datasets.monet_direct import MonetDirectStreamingDataset

    # Two subsets with distinct caption prefixes, mixed 3:1. Each glob loops its own
    # shards forever, so the emitted stream's prefix ratio must track the weights.
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    _write_monet_parquet(tmp_path / "a" / "s.parquet", 4, prefix="alpha")
    _write_monet_parquet(tmp_path / "b" / "s.parquet", 4, prefix="beta")
    ds = MonetDirectStreamingDataset(
        paths=f"{tmp_path}/a/*.parquet,{tmp_path}/b/*.parquet",
        glob_weights="0.75,0.25", shuffle_buffer=0, dino_key=None, seed=0,
    )
    recs = list(itertools.islice(iter(ds), 600))
    alpha = sum(r.caption.startswith("alpha") for r in recs)
    assert 0.65 < alpha / 600 < 0.85  # ~0.75 +- sampling noise at n=600
    assert 0 < 600 - alpha  # both subsets present


def test_monet_direct_cli_flags_parse() -> None:
    from train import build_parser

    args = build_parser().parse_args([
        "--cache-wds", "hf://datasets/jasperai/monet/v1.2.0/cc12m/*/*.parquet",
        "--monet-direct", "--monet-min-aesthetic", "4.5",
        "--monet-weights", "0.5,0.5",
        "--monet-caption-probs", "0,0,0,0,1;.25,.25,.25,.25,0",
        "--text-encoder-id", "Qwen/Qwen3.5-4B",
        "--text-encoder-layer", "-2",
        "--text-encoder-max-length", "256",
        "--fsdp2", "--devices", "8",
    ])
    assert args.monet_direct is True
    assert args.text_encoder_layer == -2
    assert args.monet_min_aesthetic == 4.5
    assert args.monet_dino is True
    assert MONET_CAPTION_KEYS[0] in args.monet_caption_keys
    assert args.monet_weights == "0.5,0.5"
    assert ";" in args.monet_caption_probs
    assert args.fsdp2 is True and args.devices == 8 and args.num_nodes == 1
    assert args.fsdp2_distributed_checkpoint is False
