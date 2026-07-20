"""v7 DDT split (--ddt-decoder-depth): encoder/decoder block partition, token-wise
z-conditioning, REPA-on-z default, backward-compat with v6 checkpoints, and the
train.py CLI wiring."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from boomer.models.boomer_fla_dit import (  # noqa: E402
    BoomerFLADDTDecoderBlock,
    BoomerFLADiT,
    BoomerFLADiTConfig,
    BoomerFLADualStreamBlock,
    BoomerFLAJointRefreshBlock,
)


def _config(**overrides) -> BoomerFLADiTConfig:
    """Small CPU config: torch mixer (no fla import), v6-style features on."""
    base = dict(
        latent_channels=4,
        latent_size=8,
        text_dim=16,
        text_seq_len=6,
        hidden_dim=32,
        depth=6,
        num_heads=4,
        mixer_type="torch",
        image_attention_every=2,
        dual_stream_depth=1,
        joint_refresh_attention=True,
        sandwich_rmsnorm=True,
        use_abs_pos_embed=False,
        image_attention_rope=True,
        ddt_decoder_depth=2,
    )
    base.update(overrides)
    return BoomerFLADiTConfig(**base)


def _inputs(config: BoomerFLADiTConfig, batch: int = 2):
    latent = torch.randn(batch, config.latent_channels, config.latent_size, config.latent_size)
    t = torch.rand(batch)
    text = torch.randn(batch, config.text_seq_len, config.text_dim)
    mask = torch.ones(batch, config.text_seq_len)
    return latent, t, text, mask


def test_block_partition_and_types() -> None:
    model = BoomerFLADiT(_config())
    assert model.encoder_depth == 4
    assert len(model.blocks) == 4
    assert len(model.decoder_blocks) == 2
    assert isinstance(model.blocks[0], BoomerFLADualStreamBlock)
    assert all(isinstance(b, BoomerFLADDTDecoderBlock) for b in model.decoder_blocks)
    assert model.decoder_x_embedder is not None
    # Joint-refresh demotion is computed within the ENCODER range only.
    for i, block in enumerate(model.blocks):
        if isinstance(block, BoomerFLAJointRefreshBlock):
            assert 1 <= i < model.encoder_depth


def test_forward_shape_and_zero_init_output() -> None:
    config = _config()
    model = BoomerFLADiT(config).eval()
    latent, t, text, mask = _inputs(config)
    with torch.no_grad():
        out = model(latent, t, text, mask)
    assert out.shape == latent.shape
    # out_proj is zero-init: the model must start as an exact zero predictor, DDT or not.
    assert torch.all(out == 0)


def test_ddt_off_has_no_decoder_keys() -> None:
    """ddt_decoder_depth=0 must be key-for-key identical to a pre-v7 model."""
    model = BoomerFLADiT(_config(ddt_decoder_depth=0))
    assert len(model.blocks) == 6
    assert len(model.decoder_blocks) == 0
    assert model.decoder_x_embedder is None
    keys = model.state_dict().keys()
    assert not any(k.startswith(("decoder_blocks", "decoder_x_embedder")) for k in keys)


def test_gradients_flow_through_encoder_via_z() -> None:
    """The decoder's only path to the encoder is the z conditioning; a loss on the
    output must still reach encoder parameters through it."""
    config = _config()
    model = BoomerFLADiT(config).train()
    # adaLN-Zero: the head and the decoder's cond_mod are zero-init, so init-time grads
    # upstream of them are exactly zero (they un-zero after one optimizer step). Randomize
    # both so this tests graph CONNECTIVITY, not the deliberate zero init.
    torch.nn.init.normal_(model.out_proj.weight, std=0.02)
    for block in model.decoder_blocks:
        torch.nn.init.normal_(block.cond_mod.weight, std=0.02)
    latent, t, text, mask = _inputs(config)
    out = model(latent, t, text, mask)
    (out - latent).pow(2).mean().backward()
    enc_grads = [p.grad for p in model.blocks.parameters() if p.grad is not None]
    dec_grads = [p.grad for p in model.decoder_blocks.parameters() if p.grad is not None]
    assert any(g.abs().sum() > 0 for g in enc_grads)
    assert any(g.abs().sum() > 0 for g in dec_grads)
    assert model.decoder_x_embedder.weight.grad is not None


def test_gradient_checkpointing_decoder_path() -> None:
    config = _config(gradient_checkpointing=True)
    model = BoomerFLADiT(config).train()
    latent, t, text, mask = _inputs(config)
    out = model(latent, t, text, mask)
    (out - latent).pow(2).mean().backward()
    assert model.decoder_blocks[0].cond_mod.weight.grad is not None


def test_repa_defaults_to_encoder_mid_with_ddt() -> None:
    """Official DDT keeps vanilla REPA placement (a mid-shallow encoder block, e.g.
    align_layer 8 for 22en6de) — NOT the encoder output z."""
    config = _config(repa_dim=8, repa_layer=-1)
    model = BoomerFLADiT(config)
    assert model.repa_layer == model.encoder_depth // 2  # encoder mid, in-range
    latent, t, text, mask = _inputs(config)
    out, repa = model(latent, t, text, mask, return_repa=True)
    assert out.shape == latent.shape
    assert repa is not None and repa.shape == (latent.shape[0], 8)
    # Without DDT the historical default is unchanged.
    assert BoomerFLADiT(_config(ddt_decoder_depth=0, repa_dim=8)).repa_layer == 3


def test_extra_input_channels_stay_inert_through_decoder_embedder() -> None:
    config = _config(extra_input_channels=3)
    model = BoomerFLADiT(config).eval()
    # The zero-init head would make ANY two forwards equal; give it real weights so the
    # comparison actually exercises the input path.
    torch.nn.init.normal_(model.out_proj.weight, std=0.02)
    latent, t, text, mask = _inputs(config)
    cond = torch.randn(latent.shape[0], 3, config.latent_size, config.latent_size)
    with torch.no_grad():
        base = model(latent, t, text, mask)
        conditioned = model(latent, t, text, mask, cond_input=cond)
    # Both embedders zero the reserved slice at init: conditioning is a no-op until trained.
    assert torch.equal(base, conditioned)


def test_validation_errors() -> None:
    with pytest.raises(ValueError, match="ddt_decoder_depth"):
        BoomerFLADiT(_config(ddt_decoder_depth=6))  # no encoder blocks left
    with pytest.raises(ValueError, match="ddt_decoder_depth"):
        BoomerFLADiT(_config(ddt_decoder_depth=-1))
    with pytest.raises(ValueError, match="dual_stream_depth"):
        BoomerFLADiT(_config(dual_stream_depth=5))  # exceeds encoder depth 4
    with pytest.raises(ValueError, match="repa_layer"):
        BoomerFLADiT(_config(repa_dim=8, repa_layer=5))  # indexes into the decoder
    with pytest.raises(ValueError, match="dispersive_layer"):
        BoomerFLADiT(_config(dispersive_layer=4))


def test_decoder_norms_are_nonaffine_rmsnorm() -> None:
    """Boomer/DDT hybrid: decoder pre-norms follow official DDT (RMSNorm), non-affine
    because adaLN supplies the affine transform; encoder keeps LayerNorm."""
    block = BoomerFLADDTDecoderBlock(_config(), layer_idx=4)
    assert isinstance(block.norm1, torch.nn.RMSNorm)
    assert isinstance(block.norm2, torch.nn.RMSNorm)
    assert block.norm1.weight is None
    assert isinstance(block.cond_mod, torch.nn.Linear)  # s is pre-activated; no SiLU here


def test_decoder_block_rejects_mismatched_s() -> None:
    config = _config()
    block = BoomerFLADDTDecoderBlock(config, layer_idx=4)
    y = torch.randn(2, 64, config.hidden_dim)
    s = torch.randn(2, 32, config.hidden_dim)
    with pytest.raises(ValueError, match="s shape"):
        block(y, s, height=8, width=8)


def test_train_cli_wires_ddt_decoder_depth() -> None:
    import train as train_module

    parser = train_module.build_parser()
    args = parser.parse_args(
        ["--skip-peek", "--latent-channels", "4", "--latent-size", "8",
         "--text-dim", "16", "--text-seq-len", "6", "--ddt-decoder-depth", "6"]
    )
    config = train_module.build_config(args, record=None)
    assert config.ddt_decoder_depth == 6
    # Default stays off.
    args_off = parser.parse_args(
        ["--skip-peek", "--latent-channels", "4", "--latent-size", "8",
         "--text-dim", "16", "--text-seq-len", "6"]
    )
    assert train_module.build_config(args_off, record=None).ddt_decoder_depth == 0
