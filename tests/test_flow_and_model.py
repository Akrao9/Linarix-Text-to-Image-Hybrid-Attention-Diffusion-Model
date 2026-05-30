"""Lightweight parity tests for Sana-style flow helpers and LatentDiT."""

from __future__ import annotations

import sys
import argparse
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from boomer.checkpoint_utils import (  # noqa: E402
    CheckpointOffloader,
    load_checkpoint,
    save_inference_checkpoint,
    save_resume_checkpoint,
)
from boomer.datasets.latent_shards import is_cache_record_path, looks_like_cache_record_name  # noqa: E402
from boomer.models.factory import create_model
from boomer.models.latent_dit import LatentDiT, LatentDiTConfig  # noqa: E402
from boomer.models.mamba_anchor_dit import MambaAnchorDiTConfig
from boomer.models.boomer_fla_dit import BoomerFLADiTConfig, FLASelfMixer, FullImageSelfAttention, RoPE2D
from boomer.sana_flow import (  # noqa: E402
    flow_min_snr_loss_weight,
    flow_training_loss_weight,
    noise_latent_at_sigma,
    sample_flow_sigmas,
    shift_sigma,
    unshift_sigma,
)
from boomer.sana_latent_cache import gemma_token_select_index  # noqa: E402
from tools.overfit_latent_dit import apply_resume_train_args, init_wandb_run, lr_multiplier  # noqa: E402
from tools.overfit_latent_dit import build_model_config  # noqa: E402


def test_flow_shift_roundtrip() -> None:
    sigma = torch.tensor([0.0, 0.25, 0.5, 0.99, 1.0])
    shifted = shift_sigma(sigma, 3.0)
    recovered = unshift_sigma(shifted, 3.0)
    assert torch.allclose(recovered, sigma, atol=1e-5)


def test_noise_latent_at_sigma() -> None:
    x0 = torch.randn(2, 32, 16, 16)
    t = shift_sigma(torch.tensor([0.3, 0.8]), 3.0)
    x_t, noise = noise_latent_at_sigma(x0, t)
    assert x_t.shape == x0.shape
    expected = (1.0 - t.view(-1, 1, 1, 1)) * x0 + t.view(-1, 1, 1, 1) * noise
    assert torch.allclose(x_t, expected)


def test_flow_training_loss_weight_clamps_near_zero_sigma() -> None:
    target = torch.randn(4, 32, 16, 16)
    sigma = shift_sigma(torch.tensor([1e-8, 0.01, 0.5, 1.0]), 3.0)
    weight = flow_training_loss_weight(sigma, target, max_weight=100.0)
    assert weight.max().item() <= 100.0 + 1e-6


def test_flow_min_snr_loss_weight_downweights_low_noise() -> None:
    target = torch.randn(4, 32, 16, 16)
    sigma = torch.tensor([0.1, 0.3, 0.5, 0.9])
    weight = flow_min_snr_loss_weight(sigma, target, gamma=5.0).flatten()
    assert weight[0].item() < weight[1].item()
    assert weight[2].item() == 1.0
    assert weight[3].item() == 1.0


def test_bounded_logit_normal_stays_inside_shifted_bounds() -> None:
    sigmas = sample_flow_sigmas(
        1000,
        device="cpu",
        dtype=torch.float32,
        sampler="bounded-logit-normal",
        flow_shift=1.5,
        logit_mean=0.0,
        logit_std=1.0,
        plateau_low=0.2,
        plateau_high=0.8,
    )
    assert sigmas.min().item() >= shift_sigma(torch.tensor(0.2), 1.5).item() - 1e-6
    assert sigmas.max().item() <= shift_sigma(torch.tensor(0.8), 1.5).item() + 1e-6


def test_plateau_logit_normal_has_mode_to_one_plateau() -> None:
    sigmas = sample_flow_sigmas(
        5000,
        device="cpu",
        dtype=torch.float32,
        sampler="plateau-logit-normal",
        flow_shift=1.0,
        logit_mean=0.0,
        logit_std=1.0,
    )
    assert sigmas.min().item() > 0.0
    assert sigmas.max().item() < 1.0
    assert ((sigmas >= 0.5) & (sigmas <= 1.0)).float().mean().item() > 0.45


def test_gemma_token_select_index() -> None:
    indices = gemma_token_select_index(300)
    assert indices[0] == 0
    assert len(indices) == 300


def test_cache_record_path_filter() -> None:
    assert looks_like_cache_record_name("00000000_d8dfa99fb3d6.pt")
    assert not looks_like_cache_record_name("latent_dit_step_010000.pt")
    assert not looks_like_cache_record_name("checkpoints/latent_dit_final.pt")


def test_latent_dit_forward_and_null_dropout() -> None:
    model = LatentDiT(LatentDiTConfig(hidden_dim=64, depth=1, num_heads=4, y_norm=True, cross_attention=True))
    latent = torch.randn(2, 32, 16, 16)
    timesteps = torch.tensor([0.5, 0.5])
    text = torch.randn(2, 300, 1536)
    mask = torch.ones(2, 300)
    caption_tokens = model.caption_embedder(text)
    assert caption_tokens.shape == (2, 300, 64)
    assert model.attention_y_norm is not None
    assert tuple(model.attention_y_norm.weight.shape) == (64,)
    out = model(latent, timesteps, text, mask)
    assert out.shape == latent.shape
    dropped_text, dropped_mask = model.apply_condition_dropout(text, mask, 1.0)
    assert dropped_text.shape == text.shape
    assert dropped_mask.shape == mask.shape


def test_mamba_anchor_dit_fallback_forward() -> None:
    config = MambaAnchorDiTConfig(
        hidden_dim=64,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
        num_anchors=8,
        mixer_type="mlp",
        text_attention_every=1,
        anchor_attention_every=1,
    )
    model = create_model(config)
    latent = torch.randn(2, 32, 16, 16)
    timesteps = torch.tensor([0.25, 0.75])
    text = torch.randn(2, 300, 1536)
    mask = torch.ones(2, 300)
    out = model(latent, timesteps, text, mask)
    assert out.shape == latent.shape


def test_boomer_fla_dit_fallback_forward() -> None:
    config = BoomerFLADiTConfig(
        hidden_dim=64,
        depth=2,
        num_heads=4,
        mlp_ratio=2.5,
        mixer_type="fallback",
    )
    model = create_model(config)
    latent = torch.randn(2, 32, 16, 16)
    timesteps = torch.tensor([0.25, 0.75])
    text = torch.randn(2, 300, 1536)
    mask = torch.ones(2, 300)
    out = model(latent, timesteps, text, mask)
    assert out.shape == latent.shape


def test_boomer_fla_condition_dropout_pads_short_checkpoint_null_condition() -> None:
    config = BoomerFLADiTConfig(
        hidden_dim=64,
        depth=2,
        num_heads=4,
        mlp_ratio=2.5,
        mixer_type="fallback",
        text_seq_len=300,
    )
    model = create_model(config)
    text = torch.randn(2, 340, 1536)
    mask = torch.ones(2, 340)

    dropped_text, dropped_mask = model.apply_condition_dropout(text, mask, 1.0)

    assert dropped_text.shape == text.shape
    assert dropped_mask.shape == mask.shape
    assert torch.all(dropped_text[:, 300:] == 0)
    assert torch.all(dropped_mask[:, :300] == 1)
    assert torch.all(dropped_mask[:, 300:] == 0)


def test_boomer_fla_dit_periodic_image_attention_forward() -> None:
    config = BoomerFLADiTConfig(
        hidden_dim=64,
        depth=4,
        num_heads=4,
        mlp_ratio=2.5,
        mixer_type="fallback",
        image_attention_every=2,
        image_attention_backend="sdpa",
    )
    model = create_model(config)
    latent = torch.randn(2, 32, 16, 16)
    timesteps = torch.tensor([0.25, 0.75])
    text = torch.randn(2, 300, 1536)
    mask = torch.ones(2, 300)
    out = model(latent, timesteps, text, mask)
    assert out.shape == latent.shape
    assert model.blocks[0].use_image_attention is False
    assert model.blocks[1].use_image_attention is True


def test_full_image_self_attention_flash_compute_dtype_for_fp32_cuda() -> None:
    if not torch.cuda.is_available():
        return
    x = torch.zeros(2, 256, 64, device="cuda", dtype=torch.float32)
    assert FullImageSelfAttention._flash_compute_dtype(x) == torch.bfloat16


def test_rope2d_preserves_magnitudes() -> None:
    rope = RoPE2D(head_dim=72, grid_size=16, theta=10000.0)
    q = torch.randn(2, 256, 16, 72)
    k = torch.randn(2, 256, 16, 72)
    q_rot, k_rot = rope(q, k)
    assert torch.allclose(q.norm(dim=-1), q_rot.norm(dim=-1), atol=1e-5)
    assert torch.allclose(k.norm(dim=-1), k_rot.norm(dim=-1), atol=1e-5)


def test_boomer_fla_dit_image_rope_forward() -> None:
    config = BoomerFLADiTConfig(
        hidden_dim=64,
        depth=4,
        num_heads=4,
        mlp_ratio=2.5,
        mixer_type="fallback",
        image_attention_every=2,
        image_attention_backend="sdpa",
        image_attention_rope=True,
        image_rope_theta=10000.0,
    )
    model = create_model(config)
    rope_block = model.blocks[1]
    assert rope_block.use_image_attention is True
    assert rope_block.image_attn.rope is not None

    latent = torch.randn(2, 32, 16, 16)
    timesteps = torch.tensor([0.25, 0.75])
    text = torch.randn(2, 300, 1536)
    mask = torch.ones(2, 300)
    out = model(latent, timesteps, text, mask)
    assert out.shape == latent.shape


def test_boomer_fla_dit_sana_cross_attention_forward() -> None:
    config = BoomerFLADiTConfig(
        hidden_dim=64,
        depth=2,
        num_heads=4,
        mlp_ratio=2.5,
        mixer_type="fallback",
        cross_attention_backend="sdpa",
        cross_attention_qk_norm=True,
    )
    model = create_model(config)
    cross_attn = model.blocks[0].cross_attn
    assert not isinstance(cross_attn, torch.nn.MultiheadAttention)
    assert not isinstance(cross_attn.q_norm, torch.nn.Identity)
    assert not isinstance(cross_attn.k_norm, torch.nn.Identity)
    assert torch.all(cross_attn.proj.weight == 0)
    assert torch.all(cross_attn.proj.bias == 0)

    latent = torch.randn(2, 32, 16, 16)
    timesteps = torch.tensor([0.25, 0.75])
    text = torch.randn(2, 300, 1536)
    mask = torch.ones(2, 300)
    mask[1, 128:] = 0
    out = model(latent, timesteps, text, mask)
    assert out.shape == latent.shape


def test_boomer_fla_parallel_coord_dual_stream_forward() -> None:
    config = BoomerFLADiTConfig(
        hidden_dim=64,
        depth=3,
        num_heads=4,
        mlp_ratio=2.0,
        mixer_type="fallback",
        cross_attention_backend="sdpa",
        cross_attention_qk_norm=True,
        parallel_block=True,
        dual_stream_depth=2,
        multimodal_coord_ids=True,
        use_abs_pos_embed=False,
    )
    model = create_model(config)

    assert model.coord_embedder is not None
    assert getattr(model.blocks[0], "updates_text", False)
    assert getattr(model.blocks[1], "updates_text", False)
    assert not getattr(model.blocks[2], "updates_text", False)

    latent = torch.randn(2, 32, 16, 16)
    timesteps = torch.tensor([0.25, 0.75])
    text = torch.randn(2, 300, 1536)
    mask = torch.ones(2, 300)
    mask[1, 240:] = 0
    out = model(latent, timesteps, text, mask)
    assert out.shape == latent.shape

    wide_latent = torch.randn(2, 32, 16, 24)
    out = model(wide_latent, timesteps, text, mask)
    assert out.shape == wide_latent.shape


def test_boomer_fla_patchifier_forward_unpatches_to_latent_shape() -> None:
    config = BoomerFLADiTConfig(
        hidden_dim=64,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
        mixer_type="fallback",
        cross_attention_backend="sdpa",
        patch_size=2,
        use_abs_pos_embed=False,
    )
    model = create_model(config)

    latent = torch.randn(2, 32, 16, 16)
    timesteps = torch.tensor([0.25, 0.75])
    text = torch.randn(2, 300, 1536)
    mask = torch.ones(2, 300)
    out = model(latent, timesteps, text, mask)
    assert out.shape == latent.shape
    assert model.out_proj.out_features == 32 * 2 * 2


def test_boomer_fla_adaln_gaussian_initialization_keeps_output_head_zero() -> None:
    config = BoomerFLADiTConfig(
        hidden_dim=64,
        depth=2,
        num_heads=4,
        mlp_ratio=2.5,
        mixer_type="fallback",
        image_attention_every=2,
        cross_attention_backend="sdpa",
        cross_attention_qk_norm=True,
    )
    model = create_model(config)

    assert torch.all(model.final_t_block[1].weight == 0)
    assert torch.all(model.final_t_block[1].bias == 0)
    assert torch.all(model.out_proj.weight == 0)
    assert torch.all(model.out_proj.bias == 0)
    for block in model.blocks:
        assert torch.all(block.mod[1].weight == 0)
        assert torch.all(block.mod[1].bias == 0)
        assert block.scale_shift_table.shape[0] == 9
        assert block.scale_shift_table.std().item() > 0.0
        assert isinstance(block.norm3, torch.nn.LayerNorm)
        assert torch.all(block.mlp.point_conv.conv.weight == 0)
        if block.use_image_attention:
            assert torch.all(block.image_attn_mod[1].weight == 0)
            assert torch.all(block.image_attn_mod[1].bias == 0)
            assert block.image_attn_scale_shift_table.std().item() > 0.0

    latent = torch.randn(2, 32, 16, 16)
    timesteps = torch.tensor([0.25, 0.75])
    text = torch.randn(2, 300, 1536)
    mask = torch.ones(2, 300)
    with torch.no_grad():
        out = model(latent, timesteps, text, mask)
    assert torch.all(out == 0)


def test_boomer_fla_cross_qk_norm_rejects_mha_backend() -> None:
    config = BoomerFLADiTConfig(
        hidden_dim=64,
        depth=1,
        num_heads=4,
        mixer_type="fallback",
        cross_attention_backend="mha",
        cross_attention_qk_norm=True,
    )
    try:
        create_model(config)
    except ValueError as exc:
        assert "cross_attention_qk_norm" in str(exc)
    else:
        raise AssertionError("expected ValueError when qk_norm is enabled with mha backend")


def test_fla_bidirectional_mixer_runs_forward_and_reverse_with_separate_layers() -> None:
    class DummyCausalMixer(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x.cumsum(dim=1)

    mixer = FLASelfMixer.__new__(FLASelfMixer)
    torch.nn.Module.__init__(mixer)
    mixer.mixer_fwd = DummyCausalMixer()
    mixer.mixer_bwd = DummyCausalMixer()
    mixer.bidirectional = True

    x = torch.arange(1, 5, dtype=torch.float32).reshape(1, 4, 1)
    out = mixer(x)
    expected = 0.5 * (x.cumsum(dim=1) + x.flip(1).cumsum(dim=1).flip(1))

    assert torch.allclose(out, expected)


def test_weights_only_resume_can_enable_boomer_fla_bidirectional() -> None:
    checkpoint_config = BoomerFLADiTConfig(
        hidden_dim=64,
        depth=2,
        num_heads=4,
        mixer_type="fla_gated_deltanet",
        fla_bidirectional=False,
    )
    checkpoint = {"config": checkpoint_config.__dict__, "train_args": {}}
    args = argparse.Namespace(resume_weights_only=True, fla_bidirectional=True)
    sample_record = argparse.Namespace(
        text_embedding=torch.empty(384, 1536),
        image_latent=torch.empty(32, 16, 16),
    )

    config = build_model_config(args, sample_record, checkpoint)

    assert isinstance(config, BoomerFLADiTConfig)
    assert config.fla_bidirectional is True


def test_linear_warmup_cosine_lr_multiplier() -> None:
    assert lr_multiplier(1, 100, scheduler="linear-warmup-cosine", warmup_steps=10, min_lr_ratio=0.1) == 0.1
    assert lr_multiplier(10, 100, scheduler="linear-warmup-cosine", warmup_steps=10, min_lr_ratio=0.1) == 1.0
    end = lr_multiplier(100, 100, scheduler="linear-warmup-cosine", warmup_steps=10, min_lr_ratio=0.1)
    assert abs(end - 0.1) < 1e-6
    assert lr_multiplier(50, 100, scheduler="constant", warmup_steps=10, min_lr_ratio=0.1) == 1.0


def test_safetensors_inference_checkpoint_roundtrip(tmp_path) -> None:
    model = LatentDiT(LatentDiTConfig(hidden_dim=32, depth=1, num_heads=4))
    config = model.config
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    train_args = {"flow_shift": 3.0, "ema_decay": 0.96, "ema_update_every": 8}
    resume_path = tmp_path / "checkpoint.pt"
    save_resume_checkpoint(
        resume_path,
        model,
        config,
        optimizer,
        42,
        train_args=train_args,
    )
    inference_path = tmp_path / "checkpoint.safetensors"
    save_inference_checkpoint(
        inference_path,
        model,
        config,
        42,
        train_args=train_args,
    )

    resume_loaded = load_checkpoint(resume_path)
    inference_loaded = load_checkpoint(inference_path)

    assert resume_loaded["steps"] == 42
    assert inference_loaded["steps"] == 42
    assert "optimizer" in resume_loaded
    assert "optimizer" not in inference_loaded
    assert inference_loaded["train_args"]["flow_shift"] == 3.0
    assert torch.allclose(
        resume_loaded["model"]["token_proj.weight"],
        inference_loaded["model"]["token_proj.weight"],
    )


def test_resume_train_args_preserve_explicit_overrides() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--t-sampler", default="uniform")
    parser.add_argument("--lr-scheduler", default="constant")
    args = parser.parse_args(["--t-sampler", "high"])

    apply_resume_train_args(
        args,
        parser,
        {"t_sampler": "logit-normal", "lr_scheduler": "linear-warmup-cosine"},
    )

    assert args.t_sampler == "high"
    assert args.lr_scheduler == "linear-warmup-cosine"


def test_init_wandb_run_disabled_by_default() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="boomer")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-mode", default="online")
    args = parser.parse_args([])
    config = BoomerFLADiTConfig(hidden_dim=64, depth=2, num_heads=4, mixer_type="fallback")

    assert init_wandb_run(args, config=config, start_step=0, resume=False) is None


def test_checkpoint_offloader_uploads_and_prunes(tmp_path, monkeypatch) -> None:
    model = LatentDiT(LatentDiTConfig(hidden_dim=32, depth=1, num_heads=4))
    config = model.config
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    offloader = CheckpointOffloader(upload_remote="testremote:checkpoints", local_retain=1)

    class FakeProcess:
        def __init__(self) -> None:
            self._done = False

        def poll(self) -> int | None:
            if self._done:
                return 0
            self._done = True
            return None

        @property
        def returncode(self) -> int:
            return 0

    launched_cmds: list[list[str]] = []

    def fake_popen(cmd, stdout, stderr, text):  # noqa: ANN001
        launched_cmds.append(cmd)
        log_path = Path(cmd[2]) / ".upload_logs" / "fake.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("ok", encoding="utf-8")
        return FakeProcess()

    monkeypatch.setattr("boomer.checkpoint_utils.subprocess.Popen", fake_popen)

    for step in (1000, 2000):
        resume_path = tmp_path / f"latent_dit_step_{step:06d}.pt"
        offloader.save(resume_path, model, config, optimizer, step)
        offloader._poll_jobs()

    offloader.finish()

    assert launched_cmds
    assert launched_cmds[0][0] == "rclone"
    assert launched_cmds[0][1] == "copy"
    assert (tmp_path / "latent_dit_step_002000.pt").exists()
    assert not (tmp_path / "latent_dit_step_001000.pt").exists()
    assert (tmp_path / "latent_dit_step_002000.safetensors").exists()
    assert not (tmp_path / "latent_dit_step_001000.safetensors").exists()


def test_checkpoint_offloader_local_copy(tmp_path) -> None:
    model = LatentDiT(LatentDiTConfig(hidden_dim=32, depth=1, num_heads=4))
    config = model.config
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    dest_dir = tmp_path / "drive" / "checkpoints"
    offloader = CheckpointOffloader(upload_remote=str(dest_dir), local_retain=1)

    for step in (1000, 2000):
        resume_path = tmp_path / f"latent_dit_step_{step:06d}.pt"
        offloader.save(resume_path, model, config, optimizer, step)
        offloader._poll_jobs()

    offloader.finish()

    assert (dest_dir / "latent_dit_step_002000.pt").is_file()
    assert (dest_dir / "latent_dit_step_002000.safetensors").is_file()
    assert not (tmp_path / "latent_dit_step_001000.pt").exists()
