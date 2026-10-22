"""Training startup, component wiring, and fit lifecycle."""

from __future__ import annotations

import os
import sys
from typing import Any

import torch

from boomer.datasets.monet_direct import MONET_DINO_KEY
from boomer.fill_masks import parse_fill_mask_probs
from boomer.sana_latent_cache import hf_token_value
from boomer.training._lightning import (
    pl,
    Callback,
    LearningRateMonitor,
    ModelCheckpoint,
    WandbLogger,
)
from boomer.training.module import BoomerLit
from boomer.training.data import (
    BoomerData,
    peek_record,
)
from boomer.training.config import (
    text_encoder_hidden_size,
    parse_int_tuple,
    build_config,
)
from boomer.training.cli import build_parser
from boomer.training.callbacks.ema import EMACallback
from boomer.training.callbacks.progress import StepUnitProgressBar
from boomer.training.callbacks.previews import (
    SamplePreviewCallback,
    CondSamplePreviewCallback,
)
from boomer.training.callbacks.checkpoints import CheckpointUploadCallback


def main() -> None:
    args = build_parser().parse_args()
    # `--timestep-weight` (Sana sqrt(1+sigma^2)/sigma, upweights detail) and
    # `--min-snr-gamma` (downweights detail) are *alternative* per-timestep loss
    # reweightings, not composable — stacking them multiplies into a product no one
    # designed. Refuse both at once rather than silently compounding.
    if args.timestep_weight and args.min_snr_gamma > 0.0:
        raise SystemExit(
            "Use only one per-timestep loss weighting: --timestep-weight "
            "(Sana sqrt(1+sigma^2)/sigma) OR --min-snr-gamma, not both."
        )
    # --monet-image-direct: MONET's full-resolution jpgs, encoded on the fly.
    if args.monet_image_direct:
        conflicts = [
            name for name, on in (
                ("--monet-direct", args.monet_direct),
                ("--t2i-direct", args.t2i_direct),
                ("--edit-direct", args.edit_direct),
                ("--fill-training", args.fill_training),
                ("--cache-captions-only", args.cache_captions_only),
            ) if on
        ]
        if conflicts:
            raise SystemExit(
                "--monet-image-direct streams plain MONET images on the fly; it cannot "
                f"combine with {', '.join(conflicts)}."
            )
        if args.latent_size * 32 != args.monet_image_resolution:
            raise SystemExit(
                f"--monet-image-resolution {args.monet_image_resolution} needs "
                f"--latent-size {args.monet_image_resolution // 32} (DC-AE f32), got "
                f"--latent-size {args.latent_size}."
            )
        if args.wds_shuffle_buffer > 512:
            print(
                f"[monet-image-direct] clamping --wds-shuffle-buffer "
                f"{args.wds_shuffle_buffer} -> 512 (1024px pixel records are ~3 MB each)",
                flush=True,
            )
            args.wds_shuffle_buffer = 512

    # --t2i-direct: plain single-image on-the-fly t2i. Cannot combine with the
    # latent-cache / captions-only / conditioning phases (it streams raw pixels
    # and builds no cond_input). Checked first so a conflicting combo errors on
    # the conflict, not on a downstream resolution detail.
    if args.t2i_direct:
        conflicts = [
            name for name, on in (
                ("--monet-direct", args.monet_direct),
                ("--monet-image-direct", args.monet_image_direct),
                ("--edit-direct", args.edit_direct),
                ("--fill-training", args.fill_training),
                ("--cache-captions-only", args.cache_captions_only),
            ) if on
        ]
        if conflicts:
            raise SystemExit(
                "--t2i-direct streams plain single images on the fly; it cannot combine "
                f"with {', '.join(conflicts)}."
            )
        if args.latent_size * 32 != args.t2i_resolution:
            # DC-AE f32: latent grid = resolution / 32. A mismatch would build a DiT
            # whose RoPE/token grid disagrees with what encode_pixels produces.
            raise SystemExit(
                f"--t2i-resolution {args.t2i_resolution} needs --latent-size "
                f"{args.t2i_resolution // 32} (DC-AE f32), got --latent-size {args.latent_size}."
            )
        if args.wds_shuffle_buffer > 512:
            # A single-image record is ~3 MB (3 x 1024^2 uint8) vs ~64 KB for a latent;
            # a latent-sized reservoir would be tens of GB per worker. Shard-order
            # shuffling (datasets .shuffle buffer) still decorrelates the stream.
            print(f"[t2i-direct] clamping --wds-shuffle-buffer {args.wds_shuffle_buffer} -> 512 "
                  "(pixel records are ~3 MB each)", flush=True)
            args.wds_shuffle_buffer = 512
    elif not args.cache_wds:
        raise SystemExit("--cache-wds is required unless --t2i-direct (which streams "
                         "from --t2i-repo).")
    # Editing-phase invariants (fail at launch, not at first step).
    if args.fill_training and args.edit_direct:
        raise SystemExit("--fill-training and --edit-direct are separate phases (E1/E2); "
                         "pick one per run.")
    if args.edit_direct and (
        args.monet_direct or args.monet_image_direct or args.cache_captions_only
    ):
        raise SystemExit("--edit-direct streams pixel pairs; it cannot combine with "
                         "--monet-direct, --monet-image-direct, or --cache-captions-only.")
    if args.fill_training or args.edit_direct:
        expected_extra = args.latent_channels + 1
        if args.extra_input_channels != expected_extra:
            raise SystemExit(
                f"--fill-training/--edit-direct condition on [latent ({args.latent_channels}ch) "
                f"+ mask (1ch)]; set --extra-input-channels {expected_extra} "
                f"(got {args.extra_input_channels}). This must also match the seeded checkpoint."
            )
        parse_fill_mask_probs(args.fill_mask_probs)  # fail fast on a bad spec
    if args.edit_direct and args.latent_size * 32 != args.edit_resolution:
        # DC-AE f32: latent grid = resolution / 32. A mismatch would build a DiT whose
        # RoPE/token grid disagrees with what encode_pixels produces every step.
        raise SystemExit(
            f"--edit-resolution {args.edit_resolution} needs --latent-size "
            f"{args.edit_resolution // 32} (DC-AE f32), got --latent-size {args.latent_size}."
        )
    if args.edit_direct and args.wds_shuffle_buffer > 512:
        # A pixel-pair record is ~6 MB (2 x 3 x 1024^2 uint8) vs ~64 KB for a latent
        # record; a latent-sized reservoir would be tens of GB of host RAM per worker
        # group. Shard-order shuffling still decorrelates the stream.
        print(f"[edit-direct] clamping --wds-shuffle-buffer {args.wds_shuffle_buffer} -> 512 "
              "(pixel records are ~6 MB each)", flush=True)
        args.wds_shuffle_buffer = 512
    # Fail CSV-alignment errors at launch, not minutes later inside a spawned
    # DataLoader worker (the dataset re-validates, but this is the friendly path).
    def _csv_len(csv: str) -> int:
        return len([v for v in csv.split(",") if v.strip()])

    if args.monet_caption_probs:
        groups = args.monet_caption_probs.split(";")
        if len(groups) > 1:  # per-glob ';' groups pair with the mix globs
            if not args.monet_weights:
                raise SystemExit(
                    "per-glob --monet-caption-probs (';' groups) require --monet-weights."
                )
            if len(groups) != _csv_len(args.cache_wds):
                raise SystemExit(
                    f"--monet-caption-probs has {len(groups)} ';' groups for "
                    f"{_csv_len(args.cache_wds)} globs in --cache-wds; they must align 1:1."
                )
        for group in groups:
            if group.strip() and _csv_len(group) != _csv_len(args.monet_caption_keys):
                raise SystemExit(
                    f"--monet-caption-probs group '{group}' has {_csv_len(group)} weights "
                    f"for {_csv_len(args.monet_caption_keys)} --monet-caption-keys; "
                    "they must align 1:1."
                )
    if args.monet_weights and _csv_len(args.monet_weights) != _csv_len(args.cache_wds):
        raise SystemExit(
            f"--monet-weights has {_csv_len(args.monet_weights)} weights for "
            f"{_csv_len(args.cache_wds)} globs in --cache-wds; they must align 1:1."
        )

    # Multi-GPU strategy. --fsdp2 = ModelParallelStrategy (FSDP2 via configure_model's
    # fully_shard); otherwise plain DDP when more than one device is requested.
    total_devices = args.devices * args.num_nodes
    strategy: Any = "auto"
    if args.fsdp2:
        if total_devices < 2:
            raise SystemExit("--fsdp2 needs at least 2 GPUs (set --devices / --num-nodes).")
        if args.compile:
            raise SystemExit(
                "--compile with --fsdp2 is not wired up (compile must be applied per block "
                "AFTER fully_shard); drop one of the two."
            )
        if args.fsdp2_distributed_checkpoint and args.checkpoint_upload_remote:
            raise SystemExit(
                "--checkpoint-upload-remote copies single .ckpt FILES; distributed (DCP "
                "directory) checkpoints can't be uploaded by it. Drop one of the two."
            )
        from lightning.pytorch.strategies import ModelParallelStrategy

        strategy = ModelParallelStrategy(
            data_parallel_size=total_devices,  # explicit: 'auto' on one node means
            tensor_parallel_size=1,            # TENSOR parallel across local GPUs
            save_distributed_checkpoint=args.fsdp2_distributed_checkpoint,
        )
        if (args.dtype or args.precision) not in ("fp32", "32-true"):
            print(
                "[fsdp2] precision -> 32-true (bf16 compute + fp32 grad reduce come from "
                "the FSDP2 MixedPrecisionPolicy; fp32 sharded master weights)",
                flush=True,
            )
        args.precision, args.dtype = "32-true", None
        if args.ema_decay > 0.0:
            print("[fsdp2] EMA disabled: the EMA callback shadows full state_dicts and "
                  "is not DTensor-safe.", flush=True)
            args.ema_decay = 0.0
        if args.sample_every_steps > 0:
            print("[fsdp2] in-training sample previews disabled: rank-0-only sampling "
                  "would deadlock FSDP2's collective all-gathers.", flush=True)
            args.sample_every_steps = 0
    elif total_devices > 1:
        from lightning.pytorch.strategies import DDPStrategy

        # static_graph=True silences AccumulateGrad stream-mismatch warnings that show up
        # with gradient checkpointing + DDP (the graph topology is fixed per step).
        strategy = DDPStrategy(static_graph=True)

    pl.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision("high")

    hf_token = hf_token_value(args.hf_token)

    if (
        args.monet_direct or args.monet_image_direct or args.cache_captions_only
        or args.edit_direct or args.t2i_direct
    ):
        # On-the-fly text: text_seq_len is the fixed encode length and text_dim comes
        # from the encoder's config.json. --monet-direct also skips the shard peek
        # entirely (monet's 512px f32c32 latents are the --latent-* defaults);
        # --cache-captions-only keeps the peek for LATENT dims (e.g. 1024px caches);
        # --edit-direct / --t2i-direct have no latents to peek (pixels; dims from flags).
        args.text_seq_len = args.text_encoder_max_length
        args.text_dim = text_encoder_hidden_size(args.text_encoder_id, hf_token)
        fusion_layers = parse_int_tuple(args.text_encoder_layers)
        if fusion_layers:
            # v8 multi-layer fusion: the readout concatenates len(layers) RMSNormed
            # hidden states, so the DiT's text_dim grows by the same factor.
            args.text_dim *= len(fusion_layers)
        tag = (
            "monet-direct" if args.monet_direct
            else "monet-image-direct" if args.monet_image_direct
            else "edit-direct" if args.edit_direct
            else "t2i-direct" if args.t2i_direct
            else "captions-only"
        )
        layer_desc = f"layers {fusion_layers}" if fusion_layers else f"layer {args.text_encoder_layer}"
        print(
            f"[{tag}] on-the-fly text: {args.text_encoder_id} "
            f"({layer_desc}) -> text=({args.text_seq_len},{args.text_dim})",
            flush=True,
        )
    if args.monet_direct or args.monet_image_direct or args.edit_direct or args.t2i_direct:
        args.skip_peek = True

    if args.skip_peek:
        print(f"Skipping dim peek; using explicit dims "
              f"(latent {args.latent_channels}x{args.latent_size}, "
              f"text {args.text_seq_len}x{args.text_dim}).", flush=True)
        record = None
    else:
        print("Peeking one cached record for dims (downloads one shard)...", flush=True)
        record = peek_record(args, hf_token)
    config = build_config(args, record)
    print(
        f"config: latent={config.latent_channels}x{config.latent_size} "
        f"text=({config.text_seq_len},{config.text_dim}) hidden={config.hidden_dim} "
        f"depth={config.depth} mixer={config.mixer_type} extra_in={config.extra_input_channels}",
        flush=True,
    )

    module = BoomerLit(
        config,
        hf_token=hf_token,
        monet_direct=args.monet_direct,
        monet_image_direct=args.monet_image_direct,
        fill_training=args.fill_training,
        fill_mask_probs=args.fill_mask_probs,
        cond_input_dropout=args.cond_input_dropout,
        edit_direct=args.edit_direct,
        edit_vae_id=args.edit_vae_id,
        edit_scaling_factor=args.edit_scaling_factor,
        edit_encode_chunk=args.edit_encode_chunk,
        fsdp2=args.fsdp2,
        text_encoder_id=args.text_encoder_id,
        text_encoder_layer=args.text_encoder_layer,
        # v8 multi-layer fusion. main() already multiplied args.text_dim by
        # len(layers) to size the DiT, so omitting this silently builds a
        # single-layer encoder against a DiT expecting the concatenated width.
        text_encoder_layers=args.text_encoder_layers,
        text_encoder_max_length=args.text_encoder_max_length,
        text_encoder_instruction=args.text_encoder_instruction,
        lr=args.lr,
        weight_decay=args.weight_decay,
        beta1=args.beta1,
        beta2=args.beta2,
        fused_adamw=args.fused_adamw,
        optimizer=args.optimizer,
        came_lr=args.came_lr,
        came_block_size=args.came_block_size,
        came_min_8bit_size=args.came_min_8bit_size,
        muon_lr=args.muon_lr,
        muon_aux_lr=args.muon_aux_lr,
        muon_weight_decay=args.muon_weight_decay,
        muon_momentum=args.muon_momentum,
        muon_ns_steps=args.muon_ns_steps,
        muon_adjust_lr_fn=args.muon_adjust_lr_fn,
        cmuon_lr=args.cmuon_lr,
        cmuon_aux_lr=args.cmuon_aux_lr,
        cmuon_weight_decay=args.cmuon_weight_decay,
        cmuon_momentum=args.cmuon_momentum,
        cmuon_ns_steps=args.cmuon_ns_steps,
        cmuon_chunk_rescale=args.cmuon_chunk_rescale,
        warmup_steps=args.warmup_steps,
        lr_scheduler=args.lr_scheduler,
        min_lr_ratio=args.min_lr_ratio,
        max_steps=args.max_steps,
        cond_dropout=args.cond_dropout,
        t_sampler=args.t_sampler,
        t_sampler_final=args.t_sampler_final,
        t_sampler_final_step=args.t_sampler_final_step,
        t_bias=args.t_bias,
        flow_shift=args.flow_shift,
        sigma_buckets=args.sigma_buckets,
        sigma_bucket_log_every=args.sigma_bucket_log_every,
        logit_mean=args.logit_mean,
        logit_std=args.logit_std,
        plateau_low=args.plateau_low,
        plateau_high=args.plateau_high,
        stretch_q_low=args.stretch_q_low,
        stretch_q_high=args.stretch_q_high,
        timestep_weight=args.timestep_weight,
        min_snr_gamma=args.min_snr_gamma,
        latent_mean=args.latent_mean,
        latent_std=args.latent_std,
        repa_weight=args.repa_weight,
        repa_anneal_steps=args.repa_anneal_steps,
        dispersive_weight=args.dispersive_weight,
        dispersive_temp=args.dispersive_temp,
        dispersive_metric=args.dispersive_metric,
        dispersive_anneal_steps=args.dispersive_anneal_steps,
        compile=args.compile,
        compile_mode=args.compile_mode,
        compile_text_encoder=args.compile_text_encoder,
        compile_vae=args.compile_vae,
    )

    # Optionally seed the model from a checkpoint's weights without restoring the
    # optimizer/scheduler/step (dataset continuation / fine-tuning).
    if args.resume and args.resume_weights_only:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        state = ckpt.get("state_dict", ckpt)
        missing, unexpected = module.load_state_dict(state, strict=False)
        print(f"resume-weights-only: loaded weights "
              f"(missing={len(missing)}, unexpected={len(unexpected)})", flush=True)

    if args.fill_training or args.edit_direct:
        # The reserved-channel embedder slice was zero-init'ed and receives zero
        # gradient while fed zeros, so a pure-t2i seed must still be exactly 0 here
        # (verified on the v6 512 checkpoint). Non-zero = already fill/edit-trained.
        eager = getattr(module.model, "_orig_mod", module.model)
        extra_absmax = float(eager.x_embedder.weight[:, config.latent_channels:].abs().max())
        state = (
            "pristine zero (t2i seed; conditioning pathway lights up from scratch)"
            if extra_absmax == 0.0
            else "non-zero (seed already fill/edit-trained)"
        )
        print(f"[cond-input] x_embedder reserved-slice abs-max={extra_absmax:.3e}: {state}",
              flush=True)
        if args.expect_pristine_cond_input and extra_absmax != 0.0:
            raise SystemExit(
                "--expect-pristine-cond-input: reserved slice is non-zero — this is not a "
                "pure-t2i seed. Drop the flag if resuming E1 / seeding E2 from an E1 checkpoint."
            )

    datamodule = BoomerData(
        cache_wds=args.cache_wds,
        local_dir=args.wds_local_dir,
        prefetch_shards=args.wds_prefetch_shards,
        download_workers=args.wds_download_workers,
        shuffle_buffer=args.wds_shuffle_buffer,
        hf_token=hf_token,
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
        decode_queue_size=args.wds_decode_queue,
        shard_shuffle=args.wds_shard_shuffle,
        pin_memory=args.pin_memory,
        prefetch_factor=args.prefetch_factor,
        stream_block_size=max(1, args.stream_block_size_mib) << 20,
        persistent_workers=args.persistent_workers,
        download_max_retries=args.wds_download_max_retries,
        max_consecutive_skips=args.wds_max_consecutive_skips,
        parquet_read_batch_size=args.parquet_read_batch_size,
        parquet_memory_trim_every=args.parquet_memory_trim_every,
        parquet_prefetch_ranges=args.parquet_prefetch_ranges,
        parquet_range_size_mib=args.parquet_range_size_mib,
        captions_only=args.cache_captions_only,
        monet_direct=args.monet_direct,
        monet_image_direct=args.monet_image_direct,
        monet_image_resolution=args.monet_image_resolution,
        monet_image_min_short_side=args.monet_image_min_short_side,
        monet_latent_key=args.monet_latent_key,
        monet_caption_keys=tuple(
            key.strip() for key in args.monet_caption_keys.split(",") if key.strip()
        ),
        monet_caption_probs=args.monet_caption_probs,  # dataset parses CSV/';' groups
        monet_weights=(
            tuple(float(v) for v in args.monet_weights.split(",") if v.strip())
            if args.monet_weights else None
        ),
        # Only fetch the 1536-d DINO column when REPA actually consumes it: otherwise
        # it is streamed, decoded, collated, and H2D-copied every step just to be
        # discarded by training_step.
        monet_dino_key=(
            MONET_DINO_KEY
            if args.monet_dino and args.repa_dim > 0 and args.repa_weight > 0
            else None
        ),
        monet_min_aesthetic=args.monet_min_aesthetic,
        monet_max_nsfw=args.monet_max_nsfw,
        monet_max_watermark=args.monet_max_watermark,
        edit_direct=args.edit_direct,
        edit_resolution=args.edit_resolution,
        edit_min_o_score=args.edit_min_o_score,
        edit_tasks=(
            tuple(t.strip() for t in args.edit_tasks.split(",") if t.strip())
            if args.edit_tasks else None
        ),
        edit_prompt_pick=args.edit_prompt_pick,
        edit_min_short_side=args.edit_min_short_side,
        edit_max_aspect_mismatch=args.edit_max_aspect_mismatch,
        t2i_direct=args.t2i_direct,
        t2i_repo=args.t2i_repo,
        t2i_config=args.t2i_config,
        t2i_data_dir=args.t2i_data_dir,
        t2i_weights=args.t2i_weights,
        max_screened_per_shard=args.max_screened_per_shard,
        t2i_split=args.t2i_split,
        t2i_resolution=args.t2i_resolution,
        t2i_caption_mode=args.t2i_caption_mode,
        t2i_short_prob=args.t2i_short_prob,
        t2i_min_short_side=args.t2i_min_short_side,
        t2i_min_aesthetic=args.t2i_min_aesthetic,
    )

    callbacks: list[Callback] = [
        ModelCheckpoint(
            dirpath=args.checkpoint_dir,
            every_n_train_steps=args.save_every_steps,
            save_top_k=-1,
            # "link" symlinks last.ckpt at the step file instead of serializing the whole
            # state a SECOND time. At this size one checkpoint is ~15 GB (fp32 params +
            # AdamW moments + EMA shadow), so save_last=True doubled every save into a
            # ~30 GB synchronous stall. Safe against this repo's own bookkeeping: the
            # pruner skips last.ckpt and keeps the newest --checkpoint-local-retain step
            # files (so the link target always exists), and the uploader skips it too, so
            # remote behavior is unchanged. Local filesystems only — which checkpoint_dir is.
            save_last="link",
            filename="boomer-{step}",
        ),
        LearningRateMonitor(logging_interval="step"),
        # In Colab/non-TTY output tqdm reprints a line every refresh, so throttle the
        # progress bar to the same cadence as metric logging instead of every step.
        StepUnitProgressBar(refresh_rate=max(1, args.log_every_steps)),
    ]
    ema_cb = None
    if args.ema_decay > 0.0:
        ema_cb = EMACallback(decay=args.ema_decay, update_every=args.ema_update_every)
        callbacks.append(ema_cb)
    # Previews exist to show what the model can do, so they default to --flow-shift only
    # for back-compat; --sample-flow-shift is how you render them at the inference shift.
    sample_flow_shift = (
        args.flow_shift if args.sample_flow_shift is None else args.sample_flow_shift
    )
    if args.sample_every_steps > 0:
        callbacks.append(SamplePreviewCallback(
            every_n_steps=args.sample_every_steps,
            prompts_cache=args.sample_prompts_cache,
            ema_callback=ema_cb,
            use_ema=args.sample_weights == "ema",
            steps=args.sample_steps,
            sampler=args.sample_sampler,
            stork_substeps=args.sample_stork_substeps,
            stork_derivative_order=args.sample_stork_derivative_order,
            cfg_scale=args.sample_cfg_scale,
            cfg_rescale=args.sample_cfg_rescale,
            cfg_interval=(args.sample_cfg_interval_low, args.sample_cfg_interval_high),
            flow_shift=sample_flow_shift,
            latent_mean=args.latent_mean,
            latent_std=args.latent_std,
            seed=args.sample_seed,
            dtype=args.dtype or "bf16",
            vae_id=args.sample_vae_id,
            scaling_factor=args.sample_scaling_factor,
            max_length=args.text_seq_len,
            hf_token=hf_token,
            encode_on_the_fly=(
                args.monet_direct or args.monet_image_direct
                or args.cache_captions_only or args.edit_direct
                or args.t2i_direct
            ),
            prompts_file=args.sample_prompts_file,
        ))
    if (args.fill_training or args.edit_direct) and args.sample_every_steps > 0 \
            and args.sample_cond_count > 0:
        callbacks.append(CondSamplePreviewCallback(
            mode="fill" if args.fill_training else "edit",
            every_n_steps=args.sample_every_steps,
            count=args.sample_cond_count,
            ema_callback=ema_cb,
            use_ema=args.sample_weights == "ema",
            steps=args.sample_steps,
            sampler=args.sample_sampler,
            stork_substeps=args.sample_stork_substeps,
            stork_derivative_order=args.sample_stork_derivative_order,
            cfg_scale=args.sample_cfg_scale,
            cfg_rescale=args.sample_cfg_rescale,
            cfg_interval=(args.sample_cfg_interval_low, args.sample_cfg_interval_high),
            flow_shift=sample_flow_shift,
            latent_mean=args.latent_mean,
            latent_std=args.latent_std,
            seed=args.sample_seed,
            dtype=args.dtype or "bf16",
            vae_id=args.sample_vae_id,
            scaling_factor=args.sample_scaling_factor,
            hf_token=hf_token,
            fill_mask_probs=parse_fill_mask_probs(args.fill_mask_probs),
            encode_chunk=args.edit_encode_chunk,
            image_cfg_scale=args.sample_image_cfg,
        ))
    # Appended AFTER ModelCheckpoint so a checkpoint written this step already exists on
    # disk when this callback's hook runs and enqueues it for async upload.
    upload_cb: CheckpointUploadCallback | None = None
    if args.checkpoint_upload_remote:
        upload_cb = CheckpointUploadCallback(
            checkpoint_dir=args.checkpoint_dir,
            remote=args.checkpoint_upload_remote,
            local_retain=args.checkpoint_local_retain,
            keep_uploaded=args.checkpoint_keep_uploaded,
            rclone_bin=args.checkpoint_rclone_bin,
            chunk_size=args.checkpoint_upload_chunk_size,
            hf_bin=args.checkpoint_hf_bin,
        )
        callbacks.append(upload_cb)

    precision = args.precision
    if args.dtype is not None:
        precision = {"bf16": "bf16-mixed", "fp16": "16-mixed", "fp32": "32-true"}[args.dtype]

    logger: Any = True
    if args.wandb:
        # A stable `id` (not the display `name`) is what lets W&B resume the same run
        # across Colab sessions. Without it, Lightning's WandbLogger mints a fresh random
        # id every launch, so each relaunch becomes a NEW run even with the same name.
        run_id = args.wandb_run_id or os.environ.get("WANDB_RUN_ID")
        logger = WandbLogger(
            project=args.wandb_project,
            name=args.wandb_run_name,
            id=run_id,
            resume="allow" if run_id else None,
            entity=args.wandb_entity,
            mode=args.wandb_mode,
            save_dir=args.checkpoint_dir,
        )
        # Record the full run config (model + hparams) in the W&B run.
        logger.log_hyperparams({**vars(args), **{f"model/{k}": v for k, v in vars(config).items()}})

    trainer = pl.Trainer(
        accelerator="cpu" if args.device == "cpu" else "gpu",
        devices=args.devices,
        num_nodes=args.num_nodes,
        strategy=strategy,
        precision=precision,
        max_steps=args.max_steps,
        gradient_clip_val=args.grad_clip,
        accumulate_grad_batches=args.accumulate,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=args.log_every_steps,
        enable_checkpointing=True,
        default_root_dir=args.checkpoint_dir,
    )
    # resume-weights-only already loaded the weights above; don't restore trainer state.
    resume_ckpt = None if args.resume_weights_only else args.resume
    fit_error: BaseException | None = None
    try:
        trainer.fit(module, datamodule=datamodule, ckpt_path=resume_ckpt)
    except BaseException as exc:  # noqa: BLE001 - see the teardown note below
        fit_error = exc

    # Flush pending checkpoint uploads synchronously before any exit path. Lightning's
    # on_fit_end async drain can be pre-empted by the teardown crash (final checkpoint saved
    # locally but its Drive copy killed mid-flight), and the os._exit() fast-path below would
    # kill the upload worker — so the newest checkpoint reaches the remote only if we drain
    # here, regardless of how fit() ended.
    if upload_cb is not None:
        try:
            upload_cb.finalize_uploads(trainer)
        except Exception as exc:  # noqa: BLE001
            print(f"[ckpt-upload] finalize failed: {exc}", flush=True)

    completed = trainer.global_step >= args.max_steps and not trainer.interrupted

    # Lightning tears down the persistent DataLoader workers at fit-end. Our streaming
    # IterableDataset loops forever, so a worker is typically blocked in an in-flight HF
    # range read when the shutdown signal lands; it can't unwind the C-level httpx/Arrow
    # call, the interpreter finalizes under it ("PyGILState_Release ... finalizing"), and
    # it abort()s -> "DataLoader worker ... killed by signal: Aborted", which propagates
    # out of fit() as a teardown RuntimeError. When this happens AFTER max_steps was
    # reached, training and checkpoints are already complete, so treat it as success.
    # Re-raise anything that happened BEFORE completion (a real OOM/bug/^C).
    if fit_error is not None and not completed:
        raise fit_error
    if fit_error is not None:
        print(f"[teardown] reached max_steps={args.max_steps}; ignoring DataLoader "
              f"shutdown error: {type(fit_error).__name__}: {fit_error}", flush=True)

    # Close the W&B run cleanly so it reads "finished" instead of lingering as "crashed".
    # Lightning's WandbLogger does NOT reliably call finish() on fit-end. An early-stopped
    # / ^C / OOM session never sets completed, so it stays crashed-but-resumable under
    # WANDB_RESUME=allow.
    if args.wandb and completed:
        import wandb

        if wandb.run is not None:
            wandb.finish()
            print(f"[wandb] run finished cleanly at step {trainer.global_step}", flush=True)

    if completed:
        # Durable state is flushed above: checkpoints uploaded via finalize_uploads(), W&B
        # finished. Skip Python's interpreter finalization, which otherwise re-triggers the
        # same httpx/Arrow worker-thread GIL fatal on the way out and gives a non-zero exit.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
