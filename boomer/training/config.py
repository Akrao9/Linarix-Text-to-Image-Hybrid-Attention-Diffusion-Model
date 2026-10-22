"""Model configuration and text-encoder dimension helpers."""

from __future__ import annotations

import argparse
from typing import (
    Any,
    Sequence,
)

from boomer.models.boomer_fla_dit import BoomerFLADiTConfig


def text_encoder_hidden_size(model_id: str, hf_token: str | bool | None) -> int:
    """Read the encoder's hidden size from its HF config (one small config.json fetch),
    so --monet-direct can size the DiT's text_dim before the encoder itself loads."""
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model_id, token=hf_token or None)
    hidden = getattr(cfg, "hidden_size", None)
    if not hidden:
        hidden = getattr(getattr(cfg, "text_config", None), "hidden_size", None)
    if not hidden:
        raise RuntimeError(f"Could not read hidden_size from {model_id}'s config")
    return int(hidden)


def parse_int_tuple(csv: str | Sequence[int] | None) -> tuple[int, ...] | None:
    """'7,11,15,21' -> (7, 11, 15, 21); None/empty -> None. Tolerates an
    already-parsed tuple/list (hparams checkpoint round-trips)."""
    if not csv:
        return None
    if isinstance(csv, (tuple, list)):
        return tuple(int(v) for v in csv) or None
    values = tuple(int(v) for v in csv.split(",") if v.strip())
    return values or None


def build_config(args: argparse.Namespace, record: Any) -> BoomerFLADiTConfig:
    # With --skip-peek, record is None and dims come from the explicit flags (no shard
    # download at startup). Otherwise dims are read from one peeked cache record.
    if record is not None:
        latent_channels = int(record.image_latent.shape[0])
        latent_size = int(record.image_latent.shape[-1])
        if record.text_embedding is not None:
            text_dim = int(record.text_embedding.shape[-1])
            text_seq_len = int(record.text_embedding.shape[-2])
        else:
            # --cache-captions-only: latent dims from the peeked record, text dims from
            # the on-the-fly encoder (main() set these from its config.json).
            text_dim = args.text_dim
            text_seq_len = args.text_seq_len
    else:
        latent_channels = args.latent_channels
        latent_size = args.latent_size
        text_dim = args.text_dim
        text_seq_len = args.text_seq_len
    return BoomerFLADiTConfig(
        latent_channels=latent_channels,
        latent_size=latent_size,
        text_dim=text_dim,
        text_seq_len=text_seq_len,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        y_norm=args.y_norm,
        y_norm_scale_factor=args.y_norm_scale_factor,
        mixer_type=args.mixer_type,
        fla_mode=args.fla_mode,
        fla_feature_map=args.fla_feature_map,
        fla_bidirectional=args.fla_bidirectional,
        use_short_conv=args.use_short_conv,
        conv_size=args.conv_size,
        image_attention_every=args.image_attn_every,
        image_attention_backend=args.image_attn_backend,
        image_attention_qk_rmsnorm=args.image_attn_qk_rmsnorm,
        image_attention_rope=args.image_rope,
        image_rope_theta=args.image_rope_theta,
        cross_attention_backend=args.cross_attn_backend,
        cross_attention_qk_norm=args.cross_attn_qk_norm,
        cross_attention_every=args.cross_attn_every,
        joint_attention_backend=args.joint_attn_backend,
        shared_adaln=args.shared_adaln,
        fourier_timestep_embed=args.fourier_timestep_embed,
        hilbert_mixer_order=args.hilbert_mixer_order,
        parallel_block=args.parallel_block,
        dual_stream_depth=args.dual_stream_depth,
        multimodal_coord_ids=args.multimodal_coord_ids,
        joint_refresh_attention=args.joint_refresh_attention,
        sandwich_rmsnorm=args.sandwich_rmsnorm,
        rope_ntk_factor=args.rope_ntk_factor,
        ddt_decoder_depth=args.ddt_decoder_depth,
        ddt_norm_impl=args.ddt_norm_impl,
        use_abs_pos_embed=args.abs_pos_embed,
        patch_size=args.patch_size,
        extra_input_channels=args.extra_input_channels,
        gradient_checkpointing=args.gradient_checkpointing,
        repa_dim=args.repa_dim,
        repa_layer=args.repa_layer,
        dispersive_layer=args.dispersive_layer,
        refresh_layers=parse_int_tuple(args.refresh_layers),
        refresh_no_mixer=args.refresh_no_mixer,
        block_attnres=args.block_attnres,
        attnres_group_ends=(
            (parse_int_tuple(args.attnres_group_ends) or ()) if args.block_attnres else ()
        ),
    )
