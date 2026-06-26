"""Sana-style flow-matching schedule and conditioning helpers.

Adapted from NVlabs/Sana (https://github.com/NVlabs/Sana): flow sigma shift,
logit-normal timestep sampling, forward noising, and per-timestep loss weights.
Plateau logit-normal sampling follows FLUX.2 (see _sample_plateau_logit_normal).
"""

from __future__ import annotations

import torch


def _standard_normal_cdf(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(x / torch.sqrt(torch.tensor(2.0, device=x.device, dtype=x.dtype))))


def _standard_normal_icdf(p: torch.Tensor) -> torch.Tensor:
    two = torch.tensor(2.0, device=p.device, dtype=p.dtype)
    return torch.sqrt(two) * torch.erfinv(two * p - 1.0)


def _logit_normal_mode(mu: float, sigma: float, *, device: torch.device | str, dtype: torch.dtype) -> torch.Tensor:
    t_star = torch.sigmoid(torch.tensor(float(mu), device=device, dtype=dtype))
    sigma_sq = float(sigma) * float(sigma)
    for _ in range(20):
        t = t_star.clamp(1e-6, 1.0 - 1e-6)
        logit_t = torch.log(t / (1.0 - t))
        g = logit_t - float(mu) - sigma_sq * (2.0 * t - 1.0)
        dg = 1.0 / (t * (1.0 - t)) - 2.0 * sigma_sq
        t_star = (t - g / dg).clamp(1e-6, 1.0 - 1e-6)
    return t_star


def _logit_normal_pdf(t: torch.Tensor, mu: float, sigma: float) -> torch.Tensor:
    t = t.clamp(1e-6, 1.0 - 1e-6)
    logit_t = torch.log(t / (1.0 - t))
    sigma_t = torch.tensor(float(sigma), device=t.device, dtype=t.dtype)
    normalizer = sigma_t * torch.sqrt(torch.tensor(2.0 * torch.pi, device=t.device, dtype=t.dtype))
    exp_term = torch.exp(-0.5 * ((logit_t - float(mu)) / sigma_t).pow(2))
    return exp_term / (normalizer * t * (1.0 - t))


def _sample_plateau_logit_normal(
    batch_size: int,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
    mu: float,
    sigma: float,
) -> torch.Tensor:
    """Sample from the plateau logit-normal distribution (Black Forest Labs, 2025).

    See FLUX.2 representation comparison: https://bfl.ai/research/representation-comparison
    """
    calc_dtype = torch.float32 if dtype in {torch.float16, torch.bfloat16} else dtype
    t_star = _logit_normal_mode(mu, sigma, device=device, dtype=calc_dtype)
    y_star = torch.log(t_star / (1.0 - t_star))
    p_ln = _logit_normal_pdf(t_star, mu, sigma)
    p_cdf = _standard_normal_cdf((y_star - float(mu)) / float(sigma))
    plateau_mass = ((1.0 - t_star) * p_ln) / (p_cdf + (1.0 - t_star) * p_ln).clamp_min(1e-12)

    branch = torch.rand(batch_size, device=device, dtype=calc_dtype) < plateau_mass
    plateau = t_star + (1.0 - t_star) * torch.rand(batch_size, device=device, dtype=calc_dtype)

    u = torch.rand(batch_size, device=device, dtype=calc_dtype) * p_cdf.clamp_min(1e-12)
    y = float(mu) + float(sigma) * _standard_normal_icdf(u.clamp(1e-6, 1.0 - 1e-6))
    left = torch.sigmoid(y)
    return torch.where(branch, plateau, left).to(dtype=dtype)


def _sample_stretched_logit_normal(
    batch_size: int,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
    mu: float,
    sigma: float,
    q_low: float,
    q_high: float,
) -> torch.Tensor:
    """SANA `stretched_logit_normal`: logit-normal truncated to a quantile window,
    then renormalized back to full [0, 1].

    Cuts the extreme tails beyond the [q_low, q_high] normal quantiles *in z-space*
    via exact inverse-CDF truncation, then stretches the surviving central band so
    its edges land on 0 and 1. Net effect: flattens the peaked logit-normal toward
    uniform — wider sigma coverage (higher variance, more mass pushed off 0.5 toward
    the boundaries) while staying smooth, a tunable interpolation between peaked
    logit-normal (q=0/1) and near-uniform (narrow window). Unlike
    `bounded-logit-normal`, which affine-squeezes the whole curve into a sub-range,
    this keeps the full [0, 1] span. q_low=0, q_high=1 reduces exactly to plain
    logit-normal. Mirrors NVlabs/Sana diffusion/model/respace.py:stretched_logit_normal.
    """
    calc_dtype = torch.float32 if dtype in {torch.float16, torch.bfloat16} else dtype
    mu_t = torch.tensor(float(mu), device=device, dtype=calc_dtype)
    sigma_t = torch.tensor(float(sigma), device=device, dtype=calc_dtype)
    # Quantile window edges in z-space (clamp away from +/-inf at q=0/1).
    q_lo = torch.tensor(min(max(float(q_low), 1e-6), 1.0 - 1e-6), device=device, dtype=calc_dtype)
    q_hi = torch.tensor(min(max(float(q_high), 1e-6), 1.0 - 1e-6), device=device, dtype=calc_dtype)
    z_lo = mu_t + sigma_t * _standard_normal_icdf(q_lo)
    z_hi = mu_t + sigma_t * _standard_normal_icdf(q_hi)
    # Exact inverse-CDF sampling of N(mu, sigma^2) truncated to [z_lo, z_hi].
    phi_a = _standard_normal_cdf((z_lo - mu_t) / sigma_t)
    phi_b = _standard_normal_cdf((z_hi - mu_t) / sigma_t)
    r = torch.rand(batch_size, device=device, dtype=calc_dtype)
    q = (phi_a + r * (phi_b - phi_a)).clamp(1e-6, 1.0 - 1e-6)
    z = mu_t + sigma_t * _standard_normal_icdf(q)
    # Map to sigma-space and stretch [u_lo, u_hi] -> [0, 1].
    u_raw = torch.sigmoid(z)
    u_lo = torch.sigmoid(z_lo)
    u_hi = torch.sigmoid(z_hi)
    u = (u_raw - u_lo) / (u_hi - u_lo).clamp_min(1e-12)
    return u.clamp(0.0, 1.0).to(dtype=dtype)


def shift_sigma(sigma: torch.Tensor, shift: float = 1.0) -> torch.Tensor:
    """Apply Sana/FlowMatch sigma shift."""
    if shift <= 0.0:
        raise ValueError(f"flow shift must be positive, got {shift}")
    if shift == 1.0:
        return sigma
    return shift * sigma / (1.0 + (shift - 1.0) * sigma)


def unshift_sigma(sigma: torch.Tensor, shift: float = 1.0) -> torch.Tensor:
    """Invert Sana/FlowMatch sigma shift."""
    if shift <= 0.0:
        raise ValueError(f"flow shift must be positive, got {shift}")
    if shift == 1.0:
        return sigma
    return sigma / (shift - (shift - 1.0) * sigma).clamp_min(1e-12)


def sample_flow_sigmas(
    batch_size: int,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
    sampler: str,
    bias: float = 2.0,
    flow_shift: float = 1.0,
    logit_mean: float = 0.0,
    logit_std: float = 1.0,
    plateau_low: float = 0.1,
    plateau_high: float = 0.9,
    stretch_q_low: float = 0.0,
    stretch_q_high: float = 1.0,
) -> torch.Tensor:
    """Sample flow noise levels in [0, 1] (already flow-shifted)."""
    if bias <= 0.0:
        raise ValueError(f"timestep bias must be positive, got {bias}")
    if logit_std <= 0.0:
        raise ValueError(f"logit std must be positive, got {logit_std}")
    if not 0.0 <= plateau_low < plateau_high <= 1.0:
        raise ValueError(f"plateau bounds must satisfy 0 <= low < high <= 1, got {plateau_low}, {plateau_high}")
    if not 0.0 <= stretch_q_low < stretch_q_high <= 1.0:
        raise ValueError(f"stretch quantiles must satisfy 0 <= low < high <= 1, got {stretch_q_low}, {stretch_q_high}")
    uniform = torch.rand(batch_size, device=device, dtype=dtype)
    if sampler == "uniform":
        sigma = uniform
    elif sampler == "high":
        sigma = uniform.pow(1.0 / bias)
    elif sampler == "low":
        sigma = uniform.pow(bias)
    elif sampler == "logit-normal":
        logits = torch.randn(batch_size, device=device, dtype=dtype) * logit_std + logit_mean
        sigma = torch.sigmoid(logits)
    elif sampler == "bounded-logit-normal":
        logits = torch.randn(batch_size, device=device, dtype=dtype) * logit_std + logit_mean
        sigma = torch.sigmoid(logits)
        sigma = plateau_low + (plateau_high - plateau_low) * sigma
    elif sampler == "stretched-logit-normal":
        sigma = _sample_stretched_logit_normal(
            batch_size,
            device=device,
            dtype=dtype,
            mu=logit_mean,
            sigma=logit_std,
            q_low=stretch_q_low,
            q_high=stretch_q_high,
        )
    elif sampler == "plateau-logit-normal":
        sigma = _sample_plateau_logit_normal(
            batch_size,
            device=device,
            dtype=dtype,
            mu=logit_mean,
            sigma=logit_std,
        )
    else:
        raise ValueError(f"Unsupported timestep sampler: {sampler}")
    return shift_sigma(sigma, flow_shift).clamp(0.0, 1.0)


def make_inference_sigmas(
    steps: int,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
    flow_shift: float = 1.0,
) -> torch.Tensor:
    if steps < 1:
        raise ValueError(f"steps must be at least 1, got {steps}")
    base = torch.linspace(1.0, 0.0, steps + 1, device=device, dtype=dtype)
    return shift_sigma(base, flow_shift)


def make_denoise_sigmas(
    start_sigma: float,
    steps: int,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
    flow_shift: float = 1.0,
) -> torch.Tensor:
    if not 0.0 <= start_sigma <= 1.0:
        raise ValueError(f"start sigma must be in [0, 1], got {start_sigma}")
    start = torch.tensor(float(start_sigma), device=device, dtype=dtype)
    base_start = unshift_sigma(start, flow_shift)
    base = torch.linspace(float(base_start.item()), 0.0, steps + 1, device=device, dtype=dtype)
    sigmas = shift_sigma(base, flow_shift)
    sigmas[0] = start
    sigmas[-1] = 0
    return sigmas


def noise_latent_at_sigma(
    x0: torch.Tensor,
    sigma: torch.Tensor | float,
    noise: torch.Tensor | None = None,
    *,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward flow noise: x_sigma = (1 - sigma) * x0 + sigma * noise.

    ``sigma`` must already be flow-shifted (same convention as training).
    """
    if noise is None:
        noise = torch.randn(x0.shape, device=x0.device, dtype=x0.dtype, generator=generator)
    if not torch.is_tensor(sigma):
        sigma = torch.tensor(float(sigma), device=x0.device, dtype=x0.dtype)
    sigma = sigma.to(device=x0.device, dtype=x0.dtype)
    if sigma.ndim == 0:
        sigma = sigma.expand(x0.shape[0])
    view_shape = (sigma.shape[0],) + (1,) * (x0.ndim - 1)
    t_view = sigma.reshape(view_shape)
    x_t = (1.0 - t_view) * x0 + t_view * noise
    return x_t, noise


def flow_training_loss_weight(sigma: torch.Tensor, target: torch.Tensor, *, max_weight: float = 100.0) -> torch.Tensor:
    """Sana-style per-timestep MSE weight: sqrt(1 + sigma^2) / sigma."""
    sigma = sigma.to(device=target.device, dtype=torch.float32)
    view_shape = (sigma.shape[0],) + (1,) * (target.ndim - 1)
    sigma = sigma.reshape(view_shape)
    weight = (1.0 + sigma * sigma).sqrt() / sigma.clamp_min(1e-6)
    if max_weight is not None:
        weight = weight.clamp_max(max_weight)
    return weight


def flow_min_snr_loss_weight(sigma: torch.Tensor, target: torch.Tensor, gamma: float) -> torch.Tensor:
    """Min-SNR weight for flow latents: downweight high-SNR/low-noise samples."""
    if gamma <= 0.0:
        raise ValueError(f"min-snr gamma must be positive, got {gamma}")
    sigma = sigma.to(device=target.device, dtype=torch.float32)
    view_shape = (sigma.shape[0],) + (1,) * (target.ndim - 1)
    sigma = sigma.reshape(view_shape).clamp(1e-6, 1.0 - 1e-6)
    alpha = 1.0 - sigma
    snr = (alpha * alpha) / (sigma * sigma)
    return torch.minimum(snr, torch.full_like(snr, float(gamma))) / snr.clamp_min(1e-6)


def normalize_text_embedding(
    text_embedding: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    *,
    enabled: bool = False,
    scale: float = 0.01,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Deprecated fixed RMS norm; prefer ``LatentDiT`` learnable ``attention_y_norm``."""
    if not enabled:
        return text_embedding
    normalized = text_embedding.float()
    normalized = normalized * torch.rsqrt(normalized.pow(2).mean(dim=-1, keepdim=True) + eps)
    normalized = normalized.to(dtype=text_embedding.dtype) * scale
    if attention_mask is not None:
        mask = attention_mask.to(device=text_embedding.device, dtype=text_embedding.dtype)
        normalized = normalized * mask.unsqueeze(-1)
    return normalized
