"""Shared flow-matching sampling loop.

Single source of truth for CFG velocity prediction and the STORK/Euler/Heun
sampling loop, used by both the standalone `tools/sample_latent_dit.py` CLI and
the in-training preview callback in `train.py`. Keeping it here avoids importing
a `tools/` script (not a package) from the trainer and prevents the two copies
from drifting.
"""

from __future__ import annotations

from typing import Iterable

import torch

from boomer.sana_flow import make_inference_sigmas
from boomer.stork_scheduler import make_stork_scheduler


class EncoderStateCache:
    """Holds a DDT condition-encoder output z for reuse across adjacent sampling steps.

    Encoder amortization (DDT paper §inference): the semantic state z changes slowly
    over a sampling trajectory, so recompute it only every ``reuse_steps`` steps and run
    the cheap velocity decoder in between (decoder ≈ 6/28 blocks here). ``reuse_steps=1``
    disables reuse (z recomputed every step) — a plain full-model run.

    The cached z is keyed by batch size: CFG batches cond/uncond together, and the batch
    can change at a cfg_interval boundary (2 -> 1), so a shape mismatch forces a refresh
    rather than feeding a stale-shaped z.

    ``refresh_steps`` overrides the stride with an explicit set of step indices, which is
    what `boomer.encoder_schedule` solves for: z drifts fast early and slowly late, so a
    uniform stride spends its recomputes in the wrong places.

    ``record`` appends every freshly computed z to ``history`` — the calibration pass that
    the schedule solver measures.
    """

    def __init__(
        self,
        reuse_steps: int = 1,
        refresh_steps: Iterable[int] | None = None,
        record: list[torch.Tensor] | None = None,
    ) -> None:
        self.reuse_steps = max(1, reuse_steps)
        self.refresh_steps = None if refresh_steps is None else frozenset(refresh_steps)
        if self.refresh_steps is not None and 0 not in self.refresh_steps:
            raise ValueError(
                f"refresh_steps must include step 0 — there is no earlier state to reuse "
                f"at the first step. Got {sorted(self.refresh_steps)[:8]}..."
            )
        self.history = record
        self.z: torch.Tensor | None = None

    def refresh_at(self, step_index: int) -> bool:
        if self.refresh_steps is not None:
            return step_index in self.refresh_steps
        return self.reuse_steps == 1 or step_index % self.reuse_steps == 0


@torch.inference_mode()
def predict_velocity(
    model: torch.nn.Module,
    latent: torch.Tensor,
    timestep: torch.Tensor,
    text_embedding: torch.Tensor,
    attention_mask: torch.Tensor,
    uncond_text_embedding: torch.Tensor,
    uncond_attention_mask: torch.Tensor,
    cfg_scale: float,
    cfg_rescale: float = 0.0,
    *,
    sigma_value: float | None = None,
    cfg_interval: tuple[float, float] = (0.0, 1.0),
    cond_input: torch.Tensor | None = None,
    image_cfg_scale: float = 0.0,
    encoder_cache: EncoderStateCache | None = None,
    refresh_encoder: bool = True,
) -> torch.Tensor:
    def _run(latent_b, timestep_b, text_b, mask_b, extra_b):
        """One model call, transparently caching / reusing the DDT encoder state."""
        if encoder_cache is None:
            return model(latent_b, timestep_b, text_b, mask_b, **extra_b)
        reuse = (
            not refresh_encoder
            and encoder_cache.z is not None
            and encoder_cache.z.shape[0] == latent_b.shape[0]
        )
        if reuse:
            return model(latent_b, timestep_b, text_b, mask_b, **extra_b, encoder_state=encoder_cache.z)
        out, z = model(
            latent_b, timestep_b, text_b, mask_b, **extra_b, return_encoder_state=True
        )
        encoder_cache.z = z
        if encoder_cache.history is not None:
            # Record only the CONDITIONAL rows. The batch here is [uncond, cond] under
            # CFG and [null, source, source+text] under dual guidance, and it drops to
            # one row outside cfg_interval -- so the conditional tail is the only slice
            # with a shape that is stable across the whole trajectory, which the
            # similarity matrix requires. Held on CPU in fp32: the calibration keeps one
            # state per step and the solver wants precision, not speed.
            base = latent.shape[0]
            encoder_cache.history.append(z[-base:].detach().float().cpu())
        return out

    # Guidance interval (Kynkäänniemi et al. 2024, "Applying Guidance in a Limited
    # Interval"): only apply CFG while sigma is inside [low, high]. Outside it we run
    # the single conditional pass — cheaper (one forward, not two) and avoids the
    # over-saturation/detail loss CFG causes at very low noise, plus the wasted
    # contrast at very high noise where cond/uncond barely differ.
    in_interval = sigma_value is None or cfg_interval[0] <= float(sigma_value) <= cfg_interval[1]
    # Only pass the kwarg when conditioning is present: legacy model classes
    # (mamba_anchor_dit) predate cond_input in their forward signature.
    extra = {} if cond_input is None else {"cond_input": cond_input}

    if cond_input is not None and image_cfg_scale > 0.0 and in_interval:
        # InstructPix2Pix dual guidance (training used INDEPENDENT text/cond
        # dropout, so all three branches are in-distribution):
        #   v = v_00 + s_img * (v_i0 - v_00) + s_txt * (v_it - v_i0)
        # branch 00: no source, no text | i0: source only | it: source + text.
        # s_img pulls toward source faithfulness, s_txt toward the instruction.
        latent_batch = latent.repeat(3, 1, 1, 1)
        timestep_batch = timestep.repeat(3)
        text_batch = torch.cat(
            [uncond_text_embedding, uncond_text_embedding, text_embedding], dim=0
        )
        mask_batch = torch.cat(
            [uncond_attention_mask, uncond_attention_mask, attention_mask], dim=0
        )
        cond_batch = torch.cat(
            [torch.zeros_like(cond_input), cond_input, cond_input], dim=0
        )
        v_00, v_i0, v_it = _run(
            latent_batch, timestep_batch, text_batch, mask_batch, {"cond_input": cond_batch}
        ).chunk(3)
        guided = v_00 + image_cfg_scale * (v_i0 - v_00) + cfg_scale * (v_it - v_i0)
        return _rescale(guided, v_it, cfg_rescale)

    if cfg_scale == 1.0 or not in_interval:
        return _run(latent, timestep, text_embedding, attention_mask, extra)

    latent_batch = latent.repeat(2, 1, 1, 1)
    timestep_batch = timestep.repeat(2)
    text_batch = torch.cat([uncond_text_embedding, text_embedding], dim=0)
    mask_batch = torch.cat([uncond_attention_mask, attention_mask], dim=0)
    # FLUX-Fill-style guidance (image_cfg_scale=0): the image condition rides
    # along in BOTH branches, so CFG steers text adherence while the spatial
    # conditioning stays pinned.
    if cond_input is not None:
        extra = {"cond_input": cond_input.repeat(2, 1, 1, 1)}
    uncond_velocity, cond_velocity = _run(
        latent_batch, timestep_batch, text_batch, mask_batch, extra
    ).chunk(2)
    guided = uncond_velocity + cfg_scale * (cond_velocity - uncond_velocity)
    return _rescale(guided, cond_velocity, cfg_rescale)


def _rescale(guided: torch.Tensor, cond_velocity: torch.Tensor, cfg_rescale: float) -> torch.Tensor:
    """CFG-rescale (Lin et al.): pull the guided velocity's per-sample std back
    toward the conditional branch's std, blended by ``cfg_rescale``."""
    if cfg_rescale <= 0.0:
        return guided
    std_cond = cond_velocity.std(dim=(1, 2, 3), keepdim=True)
    std_guided = guided.std(dim=(1, 2, 3), keepdim=True)
    rescaled = guided * (std_cond / std_guided.clamp_min(1e-5))
    return cfg_rescale * rescaled + (1.0 - cfg_rescale) * guided


@torch.inference_mode()
def sample_latent(
    model: torch.nn.Module,
    text_embedding: torch.Tensor,
    attention_mask: torch.Tensor,
    steps: int,
    seed: int,
    dtype: torch.dtype,
    device: str,
    cfg_scale: float,
    cfg_rescale: float = 0.0,
    sampler: str = "stork2",
    flow_shift: float = 1.0,
    # 1, not 2: STORK's own SANA integration builds flow models as
    # `flow_STORK-4-1st` (solver_order=4, derivative_order=1). The 2 that used to
    # sit here was the scheduler's general-purpose default, and it silently made
    # training previews disagree with tools/sample_latent_dit.py, which passes 1.
    stork_derivative_order: int = 1,
    # 9, not 50: substeps Taylor-extrapolate from one model_output, so error grows
    # with s. STORK's ablation finds 9 optimal for conditional flow matching.
    stork_substeps: int = 9,
    cfg_interval: tuple[float, float] = (0.0, 1.0),
    cond_input: torch.Tensor | None = None,
    image_cfg_scale: float = 0.0,
    encoder_reuse_steps: int = 1,
    encoder_refresh_steps: Iterable[int] | None = None,
    record_encoder_states: list[torch.Tensor] | None = None,
) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(seed)
    config = model.config
    # Encoder amortization: reuse the DDT condition-encoder output across adjacent steps.
    # Only valid on a DDT model (dense models have no separable encoder output).
    #
    # encoder_refresh_steps is the solved, irregular form of encoder_reuse_steps' fixed
    # stride (see boomer.encoder_schedule); it wins when both are given.
    # record_encoder_states collects the per-step z that the solver measures, and forces
    # a cache even at reuse_steps=1 -- refreshing every step, so the trajectory recorded
    # is the un-amortized one.
    encoder_cache: EncoderStateCache | None = None
    want_cache = (
        encoder_reuse_steps > 1
        or encoder_refresh_steps is not None
        or record_encoder_states is not None
    )
    if want_cache:
        if getattr(config, "ddt_decoder_depth", 0) <= 0:
            raise ValueError(
                "encoder amortization needs a DDT model (ddt_decoder_depth > 0); "
                "a dense model recomputes the full stack every step."
            )
        encoder_cache = EncoderStateCache(
            reuse_steps=encoder_reuse_steps,
            refresh_steps=encoder_refresh_steps,
            record=record_encoder_states,
        )
    latent = torch.randn(
        1,
        config.latent_channels,
        config.latent_size,
        config.latent_size,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    text_embedding = text_embedding.to(device=device, dtype=dtype)
    attention_mask = attention_mask.to(device=device)
    if cond_input is not None:
        cond_input = cond_input.to(device=device, dtype=dtype)
    uncond_text_embedding, uncond_attention_mask = model.null_condition(
        1,
        device=device,
        dtype=dtype,
        mask_dtype=attention_mask.dtype,
        token_num=text_embedding.shape[-2],
    )

    if sampler.startswith("stork"):
        solver_order = int(sampler.removeprefix("stork"))
        scheduler = make_stork_scheduler(
            steps=steps,
            device=device,
            flow_shift=flow_shift,
            solver_order=solver_order,
            derivative_order=stork_derivative_order,
            substeps=stork_substeps,
        )
        for index in range(steps):
            sigma = scheduler.sigmas[index].to(device=device, dtype=dtype)
            velocity = predict_velocity(
                model,
                latent,
                sigma.expand(1),
                text_embedding,
                attention_mask,
                uncond_text_embedding,
                uncond_attention_mask,
                cfg_scale,
                cfg_rescale,
                sigma_value=float(sigma),
                cfg_interval=cfg_interval,
                cond_input=cond_input,
                image_cfg_scale=image_cfg_scale,
                encoder_cache=encoder_cache,
                refresh_encoder=encoder_cache is None or encoder_cache.refresh_at(index),
            )
            latent = scheduler.step(velocity, sigma, latent, return_dict=True).prev_sample
        return latent

    sigmas = make_inference_sigmas(steps, device=device, dtype=dtype, flow_shift=flow_shift)
    for index in range(steps):
        sigma = sigmas[index]
        next_sigma = sigmas[index + 1]
        dt = sigma - next_sigma
        timestep = sigma.expand(1)
        velocity = predict_velocity(
            model,
            latent,
            timestep,
            text_embedding,
            attention_mask,
            uncond_text_embedding,
            uncond_attention_mask,
            cfg_scale,
            cfg_rescale,
            sigma_value=float(sigma),
            cfg_interval=cfg_interval,
            cond_input=cond_input,
            image_cfg_scale=image_cfg_scale,
            encoder_cache=encoder_cache,
            refresh_encoder=encoder_cache is None or encoder_cache.refresh_at(index),
        )
        if sampler == "euler":
            latent = latent - velocity * dt
        elif sampler == "heun":
            predictor = latent - velocity * dt
            next_timestep = next_sigma.expand(1)
            next_velocity = predict_velocity(
                model,
                predictor,
                next_timestep,
                text_embedding,
                attention_mask,
                uncond_text_embedding,
                uncond_attention_mask,
                cfg_scale,
                cfg_rescale,
                sigma_value=float(next_sigma),
                cfg_interval=cfg_interval,
                cond_input=cond_input,
                image_cfg_scale=image_cfg_scale,
                encoder_cache=encoder_cache,
                # Heun's corrector is the same step: reuse the z just computed above
                # rather than paying a second encode within one step.
                refresh_encoder=False,
            )
            latent = latent - 0.5 * dt * (velocity + next_velocity)
        else:
            raise ValueError(f"Unsupported sampler: {sampler}")
    return latent


__all__ = ["EncoderStateCache", "predict_velocity", "sample_latent"]
