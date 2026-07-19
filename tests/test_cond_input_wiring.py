"""cond_input plumbing: training_step's fill/edit conditioning, cond-input
dropout, CFG threading in boomer.sampling, and the launch-time validations."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import train  # noqa: E402
from boomer.models.boomer_fla_dit import BoomerFLADiTConfig  # noqa: E402
from boomer.sampling import predict_velocity  # noqa: E402


# --------------------------------------------------------------------------- #
# predict_velocity threading
# --------------------------------------------------------------------------- #
class _SpyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[torch.Tensor | None] = []

    def forward(self, latent, timestep, text, mask, cond_input=None):
        self.calls.append(cond_input)
        return torch.zeros_like(latent)


class _LegacyModel(nn.Module):
    """Pre-cond_input forward signature (mamba_anchor_dit-style)."""

    def forward(self, latent, timestep, text, mask):
        return torch.zeros_like(latent)


def _pv_args(batch: int = 1):
    latent = torch.randn(batch, 4, 8, 8)
    t = torch.rand(batch)
    text = torch.randn(batch, 6, 16)
    mask = torch.ones(batch, 6)
    return latent, t, text, mask, torch.zeros_like(text), torch.zeros_like(mask)


def test_predict_velocity_passes_cond_single_and_cfg() -> None:
    model = _SpyModel()
    cond = torch.randn(1, 5, 8, 8)
    # cfg 1.0 -> single pass, cond passed through unchanged.
    predict_velocity(model, *_pv_args(), cfg_scale=1.0, cond_input=cond)
    assert torch.equal(model.calls[-1], cond)
    # cfg > 1 -> doubled batch; cond rides along in BOTH branches.
    predict_velocity(model, *_pv_args(), cfg_scale=3.0, cond_input=cond)
    doubled = model.calls[-1]
    assert doubled.shape[0] == 2
    assert torch.equal(doubled[0], cond[0]) and torch.equal(doubled[1], cond[0])


def test_predict_velocity_legacy_model_without_cond_kwarg() -> None:
    # cond None must not pass the kwarg at all (legacy forward signatures).
    out = predict_velocity(_LegacyModel(), *_pv_args(), cfg_scale=2.0)
    assert out.shape == (1, 4, 8, 8)


def test_predict_velocity_image_cfg_three_branches() -> None:
    model = _SpyModel()
    cond = torch.randn(1, 5, 8, 8)
    predict_velocity(model, *_pv_args(), cfg_scale=4.0, cond_input=cond, image_cfg_scale=1.5)
    tripled = model.calls[-1]
    # IP2P dual guidance: [no-source, source, source] over one tripled batch.
    assert tripled.shape[0] == 3
    assert float(tripled[0].abs().max()) == 0.0
    assert torch.equal(tripled[1], cond[0]) and torch.equal(tripled[2], cond[0])


class _CondCountingModel(nn.Module):
    """sample_latent-compatible stub that records cond_input on every call."""

    def __init__(self, channels: int = 4, size: int = 8) -> None:
        super().__init__()
        from types import SimpleNamespace

        self.config = SimpleNamespace(latent_channels=channels, latent_size=size)
        self.calls: list[torch.Tensor | None] = []

    def null_condition(self, batch, *, device, dtype, mask_dtype=None, token_num=None):
        text = torch.zeros(batch, token_num or 6, 16, device=device, dtype=dtype)
        mask = torch.zeros(batch, token_num or 6, device=device,
                           dtype=mask_dtype or torch.float32)
        return text, mask

    def forward(self, latent, timestep, text, mask, cond_input=None):
        self.calls.append(cond_input)
        return torch.zeros_like(latent)


@pytest.mark.parametrize("sampler", ["euler", "heun", "stork2"])
def test_sample_latent_threads_cond_through_every_call(sampler) -> None:
    # Regression: the Euler/Heun FIRST velocity call used to drop cond_input,
    # silently unconditioning fill/edit sampling on non-STORK samplers.
    if sampler.startswith("stork"):
        pytest.importorskip("diffusers")  # STORKScheduler imports diffusers
    from boomer.sampling import sample_latent

    model = _CondCountingModel()
    cond = torch.randn(1, 5, 8, 8)
    sample_latent(
        model,
        torch.randn(1, 6, 16),
        torch.ones(1, 6),
        steps=2,
        seed=0,
        dtype=torch.float32,
        device="cpu",
        cfg_scale=2.0,
        sampler=sampler,
        cond_input=cond,
    )
    assert len(model.calls) >= 2
    for call in model.calls:
        assert call is not None, f"{sampler}: a velocity call dropped cond_input"
        assert torch.equal(call[0], cond[0])


# --------------------------------------------------------------------------- #
# training_step conditioning
# --------------------------------------------------------------------------- #
def _tiny_config(**overrides) -> BoomerFLADiTConfig:
    base = dict(
        latent_channels=32,
        latent_size=16,
        text_dim=64,
        text_seq_len=12,
        hidden_dim=64,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
        mixer_type="fallback",
        cross_attention_backend="sdpa",
        image_attention_every=2,
        use_abs_pos_embed=False,
        extra_input_channels=33,
    )
    base.update(overrides)
    return BoomerFLADiTConfig(**base)


def _hp(**overrides) -> dict:
    base = dict(
        latent_mean=0.0,
        latent_std=1.0,
        cond_dropout=0.0,
        t_sampler="uniform",
        t_sampler_final=None,
        t_sampler_final_step=0,
        t_bias=2.0,
        flow_shift=1.0,
        logit_mean=0.0,
        logit_std=1.0,
        plateau_low=0.0,
        plateau_high=1.0,
        stretch_q_low=0.0,
        stretch_q_high=1.0,
        timestep_weight=False,
        min_snr_gamma=0.0,
        repa_weight=0.0,
        dispersive_weight=0.0,
        fill_training=False,
        fill_mask_probs=None,
        cond_input_dropout=0.0,
    )
    base.update(overrides)
    return base


def _spyed_module(config, **hp):
    module = train.BoomerLit(config, hf_token=None, **hp)
    module.log = lambda *a, **k: None  # no trainer attached
    module.lr_schedulers = lambda: None
    seen: dict = {}
    inner = module.model

    def spy(x, t, text, mask, cond_input=None, **kw):
        seen["cond_input"] = cond_input
        return inner.forward(x, t, text, mask, cond_input=cond_input, **kw)

    module.model = _Wrap(inner, spy)
    return module, seen


class _Wrap(nn.Module):
    def __init__(self, inner, fn):
        super().__init__()
        self.inner = inner
        self._fn = fn

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.inner, name)

    def forward(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


def _latent_batch(batch: int = 2) -> dict:
    return {
        "image_latent": torch.randn(batch, 32, 16, 16),
        "text_embedding": torch.randn(batch, 12, 64),
        "attention_mask": torch.ones(batch, 12),
        "caption": ["a"] * batch,
    }


def test_training_step_t2i_passes_no_cond() -> None:
    module, seen = _spyed_module(_tiny_config(), **_hp())
    loss = module.training_step(_latent_batch(), 0)
    assert torch.isfinite(loss)
    assert seen["cond_input"] is None


def test_training_step_fill_builds_masked_cond() -> None:
    module, seen = _spyed_module(
        _tiny_config(), **_hp(fill_training=True, fill_mask_probs="rect=1.0")
    )
    batch = _latent_batch()
    loss = module.training_step(batch, 0)
    assert torch.isfinite(loss)
    cond = seen["cond_input"]
    assert cond is not None and cond.shape == (2, 33, 16, 16)
    mask = cond[:, 32:33]
    assert set(mask.unique().tolist()) <= {0.0, 1.0}
    # Kept region carries x0 exactly; the hole is zeroed.
    assert torch.allclose(cond[:, :32], batch["image_latent"] * (1.0 - mask))


def test_training_step_cond_input_dropout_zeroes_everything() -> None:
    module, seen = _spyed_module(
        _tiny_config(),
        **_hp(fill_training=True, fill_mask_probs="rect=1.0", cond_input_dropout=1.0),
    )
    module.training_step(_latent_batch(), 0)
    assert float(seen["cond_input"].abs().max()) == 0.0


def test_training_step_t2i_direct_encodes_single_image_no_cond() -> None:
    module, seen = _spyed_module(_tiny_config(), **_hp(latent_mean=0.5, latent_std=2.0))
    x0_raw = torch.randn(2, 32, 16, 16)
    calls = []

    def fake_encode(pixels):
        calls.append(pixels.shape)
        return x0_raw

    module.encode_pixels = fake_encode
    batch = {
        "pixels": torch.zeros(2, 3, 512, 512, dtype=torch.uint8),
        "caption": ["a", "b"],
        "text_embedding": torch.randn(2, 12, 64),
        "attention_mask": torch.ones(2, 12),
    }
    loss = module.training_step(batch, 0)
    assert torch.isfinite(loss)
    assert calls == [(2, 3, 512, 512)]  # single image encoded once (no pair)
    assert seen["cond_input"] is None   # plain t2i: no reserved-channel conditioning


def test_training_step_edit_uses_source_latent_and_ones_mask() -> None:
    module, seen = _spyed_module(_tiny_config(), **_hp(latent_mean=0.5, latent_std=2.0))
    src_raw = torch.randn(2, 32, 16, 16)
    tgt_raw = torch.randn(2, 32, 16, 16)
    calls = []

    def fake_encode(pixels):
        calls.append(pixels.shape)
        return tgt_raw if len(calls) == 1 else src_raw

    module.encode_pixels = fake_encode
    batch = {
        "source_pixels": torch.zeros(2, 3, 512, 512, dtype=torch.uint8),
        "target_pixels": torch.zeros(2, 3, 512, 512, dtype=torch.uint8),
        "caption": ["a", "b"],
        "text_embedding": torch.randn(2, 12, 64),
        "attention_mask": torch.ones(2, 12),
    }
    loss = module.training_step(batch, 0)
    assert torch.isfinite(loss)
    assert calls == [(2, 3, 512, 512), (2, 3, 512, 512)]  # target first, then source
    cond = seen["cond_input"]
    assert cond.shape == (2, 33, 16, 16)
    assert torch.equal(cond[:, 32:], torch.ones(2, 1, 16, 16))
    # Source latent is normalized with the SAME latent_mean/std as x0.
    assert torch.allclose(cond[:, :32], (src_raw - 0.5) / 2.0)


# --------------------------------------------------------------------------- #
# launch-time validations
# --------------------------------------------------------------------------- #
def _run_main(argv: list[str]) -> None:
    old = sys.argv
    sys.argv = ["train.py", *argv]
    try:
        train.main()
    finally:
        sys.argv = old


def test_main_rejects_fill_without_reserved_channels() -> None:
    with pytest.raises(SystemExit, match="extra-input-channels 33"):
        _run_main(["--cache-wds", "x.parquet", "--fill-training"])


def test_main_rejects_fill_plus_edit() -> None:
    with pytest.raises(SystemExit, match="separate phases"):
        _run_main(["--cache-wds", "x.parquet", "--fill-training", "--edit-direct",
                   "--extra-input-channels", "33"])


def test_main_rejects_edit_with_captions_only() -> None:
    with pytest.raises(SystemExit, match="pixel pairs"):
        _run_main(["--cache-wds", "x.parquet", "--edit-direct", "--cache-captions-only",
                   "--extra-input-channels", "33"])


def test_main_rejects_edit_resolution_latent_size_mismatch() -> None:
    with pytest.raises(SystemExit, match="--latent-size 32"):
        _run_main(["--cache-wds", "x.parquet", "--edit-direct",
                   "--extra-input-channels", "33", "--latent-size", "16",
                   "--edit-resolution", "1024"])


def test_main_rejects_t2i_plus_edit() -> None:
    with pytest.raises(SystemExit, match="cannot combine"):
        _run_main(["--t2i-direct", "--edit-direct", "--extra-input-channels", "33"])


def test_main_rejects_t2i_resolution_latent_size_mismatch() -> None:
    with pytest.raises(SystemExit, match="--latent-size 32"):
        _run_main(["--t2i-direct", "--latent-size", "16", "--t2i-resolution", "1024"])


def test_main_requires_cache_wds_without_t2i_direct() -> None:
    with pytest.raises(SystemExit, match="--cache-wds is required"):
        _run_main(["--monet-direct"])
