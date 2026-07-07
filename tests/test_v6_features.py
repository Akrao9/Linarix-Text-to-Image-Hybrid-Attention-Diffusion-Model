"""v6 feature tests: joint refresh blocks, sandwich RMSNorm, RoPE NTK scaling."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from boomer.models.boomer_fla_dit import (  # noqa: E402
    BoomerFLADiTConfig,
    BoomerFLADualStreamBlock,
    BoomerFLAJointRefreshBlock,
    MultimodalCoordinateRoPE,
    RoPE2D,
)
from boomer.models.factory import create_model  # noqa: E402
from boomer.models.latent_dit import AttentionRMSNorm  # noqa: E402


def _v6_config(**overrides) -> BoomerFLADiTConfig:
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
    return BoomerFLADiTConfig(**base)


def _forward_inputs(batch: int = 2, pad_from: int = 240):
    latent = torch.randn(batch, 32, 16, 16)
    timesteps = torch.rand(batch)
    text = torch.randn(batch, 300, 1536)
    mask = torch.ones(batch, 300)
    mask[:, pad_from:] = 0
    return latent, timesteps, text, mask


def test_joint_refresh_block_selection_and_forward() -> None:
    model = create_model(_v6_config())
    # depth 4, dual prefix 1, every 2 -> candidates at blocks 1 and 3. Block 3 is the
    # final block: no later block would read its text update, so its slot keeps the
    # classic image-attention block; only block 1 becomes a refresh block.
    assert isinstance(model.blocks[0], BoomerFLADualStreamBlock)
    assert isinstance(model.blocks[1], BoomerFLAJointRefreshBlock)
    assert not getattr(model.blocks[2], "joint_refresh", False)
    assert not model.blocks[2].use_image_attention
    assert not getattr(model.blocks[3], "joint_refresh", False)
    assert model.blocks[3].use_image_attention

    latent, timesteps, text, mask = _forward_inputs()
    out = model(latent, timesteps, text, mask)
    assert out.shape == latent.shape


def test_joint_refresh_shared_adaln_forward() -> None:
    model = create_model(_v6_config(shared_adaln=True))
    # Refresh blocks reuse shared_mod (9-way) + shared_dual_text_mod (6-way). The
    # demoted final candidate (block 3) is a classic image-attention block, so the
    # 3-way image-attention projection is still built for it.
    assert model.shared_mod is not None
    assert model.shared_image_attn_mod is not None
    assert model.shared_dual_text_mod is not None

    latent, timesteps, text, mask = _forward_inputs()
    out = model(latent, timesteps, text, mask)
    assert out.shape == latent.shape

    # With a block after the last candidate (depth 5), every candidate becomes a
    # refresh block and no image-only attention projection is needed.
    model = create_model(_v6_config(depth=5, shared_adaln=True))
    assert isinstance(model.blocks[1], BoomerFLAJointRefreshBlock)
    assert isinstance(model.blocks[3], BoomerFLAJointRefreshBlock)
    assert model.shared_image_attn_mod is None


def test_all_parameters_receive_gradients() -> None:
    # DDP/optimizer hygiene: no v6 block may carry parameters outside the graph.
    # (null_text_embedding is exempt: it is only consumed by CFG condition dropout.)
    for shared in (False, True):
        model = create_model(_v6_config(shared_adaln=shared, sandwich_rmsnorm=True, repa_dim=16, repa_layer=1))
        model.train()
        latent, timesteps, text, mask = _forward_inputs()
        out, repa_feat = model(latent, timesteps, text, mask, return_repa=True)
        (out.square().mean() + repa_feat.square().mean()).backward()
        unused = [
            name
            for name, param in model.named_parameters()
            if param.grad is None and "null_text_embedding" not in name
        ]
        assert not unused, f"shared_adaln={shared}: parameters without gradients: {unused}"


def test_joint_refresh_parallel_block_forward() -> None:
    model = create_model(_v6_config(parallel_block=True, sandwich_rmsnorm=True))
    latent, timesteps, text, mask = _forward_inputs()
    out = model(latent, timesteps, text, mask)
    assert out.shape == latent.shape


def test_joint_refresh_padded_text_is_inert() -> None:
    torch.manual_seed(0)
    model = create_model(_v6_config()).eval()
    latent, timesteps, text, mask = _forward_inputs(pad_from=200)
    with torch.no_grad():
        out_clean = model(latent, timesteps, text, mask)
        garbage = text.clone()
        garbage[:, 200:] = torch.randn_like(garbage[:, 200:]) * 100.0
        out_garbage = model(latent, timesteps, garbage, mask)
    assert torch.allclose(out_clean, out_garbage, atol=1e-6, rtol=0.0)


def test_joint_refresh_requires_image_attention_cadence() -> None:
    try:
        create_model(_v6_config(image_attention_every=0))
    except ValueError as exc:
        assert "image_attention_every" in str(exc)
    else:
        raise AssertionError("expected ValueError for joint refresh without image_attention_every")


def test_sandwich_rmsnorm_modules_and_zero_output_head() -> None:
    model = create_model(_v6_config(sandwich_rmsnorm=True))
    assert isinstance(model.blocks[2].sandwich_mixer, AttentionRMSNorm)
    assert isinstance(model.blocks[1].sandwich_joint, AttentionRMSNorm)
    assert isinstance(model.blocks[0].text_sandwich_mlp, AttentionRMSNorm)

    plain = create_model(_v6_config(sandwich_rmsnorm=False))
    assert isinstance(plain.blocks[2].sandwich_mixer, nn.Identity)

    # Zero-init output head must still hold with sandwich norms in the blocks.
    latent, timesteps, text, mask = _forward_inputs()
    with torch.no_grad():
        out = model(latent, timesteps, text, mask)
    assert torch.all(out == 0)


def test_rope2d_ntk_factor_rescales_and_preserves_magnitude() -> None:
    torch.manual_seed(0)
    q = torch.randn(1, 16, 2, 8)
    k = torch.randn(1, 16, 2, 8)
    base = RoPE2D(8, 4, theta=100.0)
    ntk = RoPE2D(8, 4, theta=100.0, ntk_factor=2.0)
    q_base, _ = base(q, k)
    q_ntk, _ = ntk(q, k)
    assert not torch.allclose(q_base, q_ntk)
    assert torch.allclose(q_ntk.norm(dim=-1), q.norm(dim=-1), atol=1e-5)


def test_multimodal_rope_ntk_spares_text_positions() -> None:
    torch.manual_seed(0)
    base = MultimodalCoordinateRoPE(16, image_size=4, text_seq_len=8, theta=100.0)
    ntk = MultimodalCoordinateRoPE(16, image_size=4, text_seq_len=8, theta=100.0, ntk_factor=2.0)
    text_ids = base.text_ids(1, 8, device="cpu")
    image_ids = base.image_ids(1, height=4, width=4, device="cpu")
    ids = torch.cat([text_ids, image_ids], dim=1)
    q = torch.randn(1, 24, 2, 16)
    k = torch.randn(1, 24, 2, 16)
    q_base, k_base = base.apply(q, k, ids)
    q_ntk, k_ntk = ntk.apply(q, k, ids)
    # Text tokens (first 8) keep base frequencies; image tokens get the NTK rescale.
    assert torch.allclose(q_base[:, :8], q_ntk[:, :8], atol=1e-6)
    assert torch.allclose(k_base[:, :8], k_ntk[:, :8], atol=1e-6)
    assert not torch.allclose(q_base[:, 8:], q_ntk[:, 8:])


def test_extra_channels_zero_init_and_zero_feed_with_v6() -> None:
    model = create_model(_v6_config(sandwich_rmsnorm=True, extra_input_channels=5)).eval()
    assert torch.all(model.x_embedder.weight[:, 32:] == 0)

    latent, timesteps, text, mask = _forward_inputs()
    with torch.no_grad():
        out_none = model(latent, timesteps, text, mask, cond_input=None)
        out_zeros = model(latent, timesteps, text, mask, cond_input=torch.zeros(2, 5, 16, 16))
    assert torch.equal(out_none, out_zeros)


def test_repa_path_through_refresh_block() -> None:
    # repa_layer 1 is a joint refresh block; the pooled projection must still flow.
    model = create_model(_v6_config(repa_dim=16, repa_layer=1))
    latent, timesteps, text, mask = _forward_inputs()
    out, repa_feat = model(latent, timesteps, text, mask, return_repa=True)
    assert out.shape == latent.shape
    assert repa_feat is not None
    assert repa_feat.shape == (2, 16)


def test_gradient_checkpointing_through_refresh_block() -> None:
    model = create_model(_v6_config(sandwich_rmsnorm=True, gradient_checkpointing=True))
    model.train()
    latent, timesteps, text, mask = _forward_inputs()
    out = model(latent, timesteps, text, mask)
    out.square().mean().backward()
    assert model.blocks[1].text_qkv.weight.grad is not None
    assert model.blocks[1].image_qkv.weight.grad is not None
