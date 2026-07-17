"""Latent-grid mask synthesis for --fill-training (FLUX-Fill-style inpainting).

Masks live on the LATENT grid (one cell = vae_downsample px, 32 px for DC-AE
f32c32), value 1 = hole to generate, 0 = keep. training_step builds the model's
reserved conditioning channels from them: ``cond_input = [x0 * (1 - m), m]``.

Mask-type mix (defaults tuned for editing pretext coverage):
  * ``blob``     — union of 2-5 random ellipses (free-form inpaint)
  * ``rect``     — one random rectangle (object replacement)
  * ``outpaint`` — everything OUTSIDE a random interior rectangle (border fill)
  * ``full``     — all ones: "generate everything" == t2i under the fill
                   interface, the retention guard for pure text-to-image
  * ``tiny``     — one small ellipse (touch-up robustness)

The synthesis is deliberately per-sample Python over torch grid ops: at
batch<=64 on a 32x32 grid it is microseconds, and readability wins.
"""

from __future__ import annotations

import torch

DEFAULT_FILL_MASK_PROBS: dict[str, float] = {
    "blob": 0.30,
    "rect": 0.25,
    "outpaint": 0.15,
    "full": 0.10,
    "tiny": 0.20,
}

_MASK_TYPES = tuple(DEFAULT_FILL_MASK_PROBS)


def parse_fill_mask_probs(spec: str | None) -> dict[str, float]:
    """Parse ``"blob=0.3,rect=0.25,..."`` into a validated, normalized prob dict.

    Unlisted types get probability 0; unknown names raise. None/empty = defaults.
    """
    if not spec:
        return dict(DEFAULT_FILL_MASK_PROBS)
    probs = dict.fromkeys(_MASK_TYPES, 0.0)
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        name, _, value = part.partition("=")
        name = name.strip()
        if name not in probs:
            raise ValueError(f"unknown fill mask type {name!r}; valid: {_MASK_TYPES}")
        probs[name] = float(value)
    total = sum(probs.values())
    if total <= 0 or min(probs.values()) < 0:
        raise ValueError(f"fill mask probs must be non-negative with a positive sum: {spec!r}")
    return {k: v / total for k, v in probs.items()}


def _rand(generator: torch.Generator | None, *shape: int) -> torch.Tensor:
    # Mask geometry is sampled on CPU (trivial scalars) and only the final mask
    # tensor moves to the training device; a CPU generator keeps previews
    # deterministic without touching the CUDA training RNG stream.
    return torch.rand(*shape, generator=generator)


def _ellipse(
    ys: torch.Tensor, xs: torch.Tensor, generator: torch.Generator | None,
    *, r_low: float, r_high: float,
) -> torch.Tensor:
    height, width = ys.shape[0], xs.shape[1]
    cy = _rand(generator, 1).item() * height
    cx = _rand(generator, 1).item() * width
    ry = (r_low + _rand(generator, 1).item() * (r_high - r_low)) * height
    rx = (r_low + _rand(generator, 1).item() * (r_high - r_low)) * width
    return (((ys - cy) / max(ry, 1e-6)) ** 2 + ((xs - cx) / max(rx, 1e-6)) ** 2) <= 1.0


def _rect_bounds(
    size: int, generator: torch.Generator | None, *, frac_low: float, frac_high: float
) -> tuple[int, int]:
    extent = max(1, round((frac_low + _rand(generator, 1).item() * (frac_high - frac_low)) * size))
    start = int(_rand(generator, 1).item() * (size - extent + 1))
    return start, start + extent


def _one_mask(
    kind: str,
    height: int,
    width: int,
    generator: torch.Generator | None,
) -> torch.Tensor:
    ys = torch.arange(height, dtype=torch.float32).view(-1, 1)
    xs = torch.arange(width, dtype=torch.float32).view(1, -1)
    if kind == "full":
        return torch.ones(height, width)
    if kind == "rect":
        y0, y1 = _rect_bounds(height, generator, frac_low=0.25, frac_high=0.75)
        x0, x1 = _rect_bounds(width, generator, frac_low=0.25, frac_high=0.75)
        mask = torch.zeros(height, width)
        mask[y0:y1, x0:x1] = 1.0
        return mask
    if kind == "outpaint":
        # Keep a random interior window, generate the rest (border extension).
        y0, y1 = _rect_bounds(height, generator, frac_low=0.25, frac_high=0.70)
        x0, x1 = _rect_bounds(width, generator, frac_low=0.25, frac_high=0.70)
        mask = torch.ones(height, width)
        mask[y0:y1, x0:x1] = 0.0
        return mask
    if kind == "blob":
        count = 2 + int(_rand(generator, 1).item() * 4)  # 2..5 ellipses
        blob = torch.zeros(height, width, dtype=torch.bool)
        for _ in range(count):
            blob |= _ellipse(ys, xs, generator, r_low=0.10, r_high=0.40)
        return blob.float()
    if kind == "tiny":
        return _ellipse(ys, xs, generator, r_low=0.05, r_high=0.15).float()
    raise ValueError(f"unknown fill mask type {kind!r}")


def sample_fill_masks(
    batch: int,
    height: int,
    width: int,
    *,
    probs: dict[str, float] | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Draw a (batch, 1, height, width) mask batch from the mask-type mix.

    ``generator`` (CPU) makes the draw deterministic — used by the preview
    callback for FIXED masks across the run; training passes None.
    """
    probs = probs or DEFAULT_FILL_MASK_PROBS
    names = [k for k, v in probs.items() if v > 0]
    weights = torch.tensor([probs[k] for k in names], dtype=torch.float32)
    picks = torch.multinomial(weights, batch, replacement=True, generator=generator)
    masks = torch.stack(
        [_one_mask(names[int(i)], height, width, generator) for i in picks], dim=0
    ).unsqueeze(1)
    return masks.to(device=device, dtype=dtype)


__all__ = [
    "DEFAULT_FILL_MASK_PROBS",
    "parse_fill_mask_probs",
    "sample_fill_masks",
]
