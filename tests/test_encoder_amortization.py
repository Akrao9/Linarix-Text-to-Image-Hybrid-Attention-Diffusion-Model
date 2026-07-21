"""DDT encoder amortization: the forward's encoder_state/return_encoder_state contract
and the sampling-loop reuse schedule.

Correctness anchor: with encoder_reuse_steps=1 (and refresh every step) the amortized
path must be BIT-IDENTICAL to the plain full-model path — reuse only ever approximates
when steps are actually skipped, which is the opt-in speed/quality trade."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from boomer.models.boomer_fla_dit import BoomerFLADiT, BoomerFLADiTConfig  # noqa: E402
from boomer.sampling import EncoderStateCache, sample_latent  # noqa: E402


def _config(**overrides) -> BoomerFLADiTConfig:
    base = dict(
        latent_channels=4, latent_size=8, text_dim=16, text_seq_len=6,
        hidden_dim=32, depth=6, num_heads=4, mixer_type="torch",
        image_attention_every=2, dual_stream_depth=1, joint_refresh_attention=True,
        sandwich_rmsnorm=True, use_abs_pos_embed=False, image_attention_rope=True,
        ddt_decoder_depth=2,
    )
    base.update(overrides)
    return BoomerFLADiTConfig(**base)


def _inputs(config, batch=2):
    latent = torch.randn(batch, config.latent_channels, config.latent_size, config.latent_size)
    t = torch.rand(batch)
    text = torch.randn(batch, config.text_seq_len, config.text_dim)
    mask = torch.ones(batch, config.text_seq_len)
    return latent, t, text, mask


def _trained(config) -> BoomerFLADiT:
    """A model with non-trivial (non-zero-init) head/cond so outputs actually vary."""
    torch.manual_seed(0)
    m = BoomerFLADiT(config).eval()
    torch.nn.init.normal_(m.out_proj.weight, std=0.02)
    for block in m.decoder_blocks:
        torch.nn.init.normal_(block.cond_mod.weight, std=0.02)
    return m


def test_return_and_reuse_encoder_state_is_bit_identical() -> None:
    config = _config()
    model = _trained(config)
    latent, t, text, mask = _inputs(config)

    with torch.no_grad():
        full = model(latent, t, text, mask)
        out1, z = model(latent, t, text, mask, return_encoder_state=True)
        # Feed the cached z back with the SAME (latent, t): reconstructs the full forward.
        reused = model(latent, t, text, mask, encoder_state=z)

    assert z.shape == (latent.shape[0], (config.latent_size ** 2), config.hidden_dim)
    assert torch.equal(full, out1)  # return_encoder_state must not perturb the output
    assert torch.equal(full, reused)  # reuse at the same step == full recompute


def test_reused_z_holds_semantics_when_latent_changes() -> None:
    """Reuse across a step means the SAME z but a fresh noisy latent + t. The decoder
    must actually consume the new latent (output changes) while z is held fixed."""
    config = _config()
    model = _trained(config)
    latent, t, text, mask = _inputs(config)
    with torch.no_grad():
        _, z = model(latent, t, text, mask, return_encoder_state=True)
        other = torch.randn_like(latent)
        out_reuse = model(other, t + 0.1, text, mask, encoder_state=z)
        out_full = model(other, t + 0.1, text, mask)
    assert not torch.allclose(out_reuse, out_full)  # z is stale -> genuinely an approximation
    assert out_reuse.shape == latent.shape


def test_encoder_state_rejects_dense_model_and_bad_shape() -> None:
    dense = BoomerFLADiT(_config(ddt_decoder_depth=0)).eval()
    latent, t, text, mask = _inputs(_config(ddt_decoder_depth=0))
    with pytest.raises(ValueError, match="require a DDT model"):
        dense(latent, t, text, mask, return_encoder_state=True)

    config = _config()
    model = _trained(config)
    latent, t, text, mask = _inputs(config)
    bad = torch.randn(latent.shape[0], 999, config.hidden_dim)
    with pytest.raises(ValueError, match="encoder_state shape"):
        model(latent, t, text, mask, encoder_state=bad)


def test_encoder_state_incompatible_with_repa() -> None:
    config = _config(repa_dim=8)
    model = _trained(config)
    latent, t, text, mask = _inputs(config)
    _, z = model(latent, t, text, mask, return_encoder_state=True)
    with pytest.raises(ValueError, match="inference-only"):
        model(latent, t, text, mask, encoder_state=z, return_repa=True)


def test_cache_refresh_schedule() -> None:
    cache = EncoderStateCache(reuse_steps=3)
    refreshes = [cache.refresh_at(i) for i in range(7)]
    assert refreshes == [True, False, False, True, False, False, True]
    assert EncoderStateCache(reuse_steps=1).reuse_steps == 1
    # reuse_steps=1 always refreshes -> never actually reuses.
    assert all(EncoderStateCache(1).refresh_at(i) for i in range(5))


def test_sample_latent_reuse1_matches_no_amortization() -> None:
    """The end-to-end sampler with encoder_reuse_steps=1 must equal the un-amortized
    run bit-for-bit (same seed) — the reuse plumbing is inert when nothing is skipped."""
    config = _config()
    model = _trained(config)
    text = torch.randn(1, config.text_seq_len, config.text_dim)
    mask = torch.ones(1, config.text_seq_len)
    common = dict(
        model=model, text_embedding=text, attention_mask=mask, steps=6, seed=7,
        dtype=torch.float32, device="cpu", cfg_scale=1.0, sampler="euler", flow_shift=1.0,
    )
    base = sample_latent(**common)
    reuse1 = sample_latent(**common, encoder_reuse_steps=1)
    assert torch.equal(base, reuse1)


def test_sample_latent_amortized_runs_and_differs() -> None:
    """reuse_steps>1 must run end-to-end and produce a (necessarily) different result —
    proof the fast path is exercised, not silently falling back to full every step."""
    config = _config()
    model = _trained(config)
    text = torch.randn(1, config.text_seq_len, config.text_dim)
    mask = torch.ones(1, config.text_seq_len)
    common = dict(
        model=model, text_embedding=text, attention_mask=mask, steps=8, seed=7,
        dtype=torch.float32, device="cpu", cfg_scale=3.0, sampler="euler", flow_shift=1.0,
    )
    base = sample_latent(**common)
    amortized = sample_latent(**common, encoder_reuse_steps=4)
    assert amortized.shape == base.shape
    assert not torch.equal(base, amortized)


def test_sample_latent_amortization_rejects_dense_model() -> None:
    dense = _trained(_config(ddt_decoder_depth=0))  # noqa: F841 - reuse helper
    dense = BoomerFLADiT(_config(ddt_decoder_depth=0)).eval()
    text = torch.randn(1, 6, 16)
    mask = torch.ones(1, 6)
    with pytest.raises(ValueError, match="needs a DDT model"):
        sample_latent(
            model=dense, text_embedding=text, attention_mask=mask, steps=4, seed=0,
            dtype=torch.float32, device="cpu", cfg_scale=1.0, sampler="euler",
            encoder_reuse_steps=2,
        )
