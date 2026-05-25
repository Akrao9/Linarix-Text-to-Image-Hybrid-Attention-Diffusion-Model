"""Reverse flow-matching denoise helpers shared by training previews and CLI tools."""

from __future__ import annotations

import torch

from boomer.sana_flow import make_denoise_sigmas
from boomer.stork_scheduler import make_stork_scheduler


@torch.inference_mode()
def denoise_latent(
    model: torch.nn.Module,
    noised: torch.Tensor,
    start_t: float,
    text_embedding: torch.Tensor,
    attention_mask: torch.Tensor,
    steps: int,
    flow_shift: float,
    *,
    sampler: str = "stork2",
    stork_derivative_order: int = 1,
    stork_substeps: int = 50,
) -> torch.Tensor:
    latent = noised
    if sampler.startswith("stork"):
        solver_order = int(sampler.removeprefix("stork"))
        scheduler = make_stork_scheduler(
            start_sigma=start_t,
            steps=steps,
            device=latent.device,
            flow_shift=flow_shift,
            solver_order=solver_order,
            derivative_order=stork_derivative_order,
            substeps=stork_substeps,
        )
        for step in range(steps):
            sigma = scheduler.sigmas[step].to(device=latent.device, dtype=latent.dtype)
            velocity = model(latent, sigma.expand(latent.shape[0]), text_embedding, attention_mask)
            latent = scheduler.step(velocity, sigma, latent, return_dict=True).prev_sample
        return latent

    sigmas = make_denoise_sigmas(
        start_t,
        steps,
        device=latent.device,
        dtype=latent.dtype,
        flow_shift=flow_shift,
    )
    for step in range(steps):
        sigma = sigmas[step]
        next_sigma = sigmas[step + 1]
        dt = sigma - next_sigma
        timestep = sigma.expand(latent.shape[0])
        velocity = model(latent, timestep, text_embedding, attention_mask)
        if sampler == "euler":
            latent = latent - dt * velocity
        elif sampler == "heun":
            predictor = latent - dt * velocity
            next_velocity = model(predictor, next_sigma.expand(latent.shape[0]), text_embedding, attention_mask)
            latent = latent - 0.5 * dt * (velocity + next_velocity)
        else:
            raise ValueError(f"Unsupported sampler: {sampler}")
    return latent
