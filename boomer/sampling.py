"""Shared flow-matching sampling loop.

Single source of truth for CFG velocity prediction and the STORK/Euler/Heun
sampling loop, used by both the standalone `tools/sample_latent_dit.py` CLI and
the in-training preview callback in `train.py`. Keeping it here avoids importing
a `tools/` script (not a package) from the trainer and prevents the two copies
from drifting.
"""

from __future__ import annotations

import torch

from boomer.sana_flow import make_inference_sigmas
from boomer.stork_scheduler import make_stork_scheduler


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
) -> torch.Tensor:
    # Guidance interval (Kynkäänniemi et al. 2024, "Applying Guidance in a Limited
    # Interval"): only apply CFG while sigma is inside [low, high]. Outside it we run
    # the single conditional pass — cheaper (one forward, not two) and avoids the
    # over-saturation/detail loss CFG causes at very low noise, plus the wasted
    # contrast at very high noise where cond/uncond barely differ.
    in_interval = sigma_value is None or cfg_interval[0] <= float(sigma_value) <= cfg_interval[1]
    # Only pass the kwarg when conditioning is present: legacy model classes
    # (mamba_anchor_dit) predate cond_input in their forward signature.
    extra = {} if cond_input is None else {"cond_input": cond_input}
    if cfg_scale == 1.0 or not in_interval:
        return model(latent, timestep, text_embedding, attention_mask, **extra)

    latent_batch = latent.repeat(2, 1, 1, 1)
    timestep_batch = timestep.repeat(2)
    text_batch = torch.cat([uncond_text_embedding, text_embedding], dim=0)
    mask_batch = torch.cat([uncond_attention_mask, attention_mask], dim=0)
    # FLUX-Fill-style guidance: the image condition (masked latent + mask / source
    # latent) rides along in BOTH branches, so CFG steers text adherence while the
    # spatial conditioning stays pinned. (Dual image+text guidance would need a
    # third branch; not worth 1.5x the FLOPs for previews.)
    if cond_input is not None:
        extra = {"cond_input": cond_input.repeat(2, 1, 1, 1)}
    uncond_velocity, cond_velocity = model(
        latent_batch, timestep_batch, text_batch, mask_batch, **extra
    ).chunk(2)
    guided = uncond_velocity + cfg_scale * (cond_velocity - uncond_velocity)
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
    stork_derivative_order: int = 2,
    stork_substeps: int = 50,
    cfg_interval: tuple[float, float] = (0.0, 1.0),
    cond_input: torch.Tensor | None = None,
) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(seed)
    config = model.config
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
            )
            latent = latent - 0.5 * dt * (velocity + next_velocity)
        else:
            raise ValueError(f"Unsupported sampler: {sampler}")
    return latent


__all__ = ["predict_velocity", "sample_latent"]
