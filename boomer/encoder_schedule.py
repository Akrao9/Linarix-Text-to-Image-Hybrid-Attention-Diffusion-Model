"""DDT encoder sharing: pick WHICH sampling steps recompute the condition encoder.

`boomer/sampling.py` already amortizes the DDT encoder across adjacent steps, but on a
fixed stride: recompute at 0, N, 2N, ... The semantic state z does not drift at a
constant rate along a trajectory, though -- it moves fast early, when the sample is still
noise and the layout is being decided, and barely at all late, when only detail is left.
A uniform stride spends its recomputes in the wrong places.

DDT (CVPR 2026) replaces the stride with a dynamic program. Measure how similar the
encoder state is between every pair of steps on a calibration trajectory, then choose the
placement of a fixed budget of recomputes that minimises total reuse error. The result is
an irregular schedule -- dense at the start, sparse at the end -- for the same cost.

Reference: sibling repo `ImageModel/DDT`, `src/diffusion/stateful_flow_matching/
sharing_sampling.py` (`sharing_dp`) and the `tools/dp.py` sketch. The recurrence here is
equivalent to theirs — `tests/test_encoder_schedule.py` checks that against a transcription
of it, and separately against brute-force optimality. What differs is form and plumbing:

  * The similarity matrix is one [N, T*C] @ [T*C, N] product instead of an explicit
    per-token [T, N, N] `bmm`, which at 1024px is the difference between a few hundred MB
    and a [50, 50] result.
  * Calibration is separated from sampling. The reference re-solves the DP inside
    `__call__` whenever the cached schedule length looks wrong, so the first sample of
    every run silently pays for a second full trajectory. Here you solve once and pass
    the schedule in.
  * Calibration reuses our own sampling loop, so the trajectory it measures is the one
    you actually sample on — same sampler, shift, and guidance.

Typical use, once a checkpoint exists:

    schedule = calibrate_encoder_schedule(model, text_emb, mask, steps=50,
                                          num_recompute=10, ...)
    sample_latent(..., encoder_refresh_steps=schedule)
"""

from __future__ import annotations

import torch

__all__ = [
    "state_similarity",
    "solve_recompute_schedule",
    "calibrate_encoder_schedule",
    "uniform_schedule",
]


def uniform_schedule(steps: int, num_recompute: int) -> list[int]:
    """The fixed-stride schedule, as a baseline to compare a solved one against."""
    if num_recompute < 1:
        raise ValueError(f"num_recompute must be >= 1, got {num_recompute}")
    stride = max(1, steps // num_recompute)
    return [i for i in range(steps) if i % stride == 0][:num_recompute]


def state_similarity(states: list[torch.Tensor]) -> torch.Tensor:
    """Mean per-token cosine similarity between encoder states, over sampling steps.

    Args:
        states: one tensor per sampling step, each [batch, tokens, hidden] and all the
            same shape.

    Returns:
        [num_steps, num_steps] float64 matrix; entry (a, b) is the cosine similarity
        between the states at steps a and b, averaged over every token and batch row.

    Each token vector is L2-normalised first, so this is a mean of per-token cosines and
    not a cosine of concatenated vectors -- a token that happens to have a large norm
    does not get to dominate. Once normalised, averaging over tokens is a plain inner
    product of the flattened states divided by the token count, which is why no [T, N, N]
    intermediate is needed.
    """
    if not states:
        raise ValueError("no encoder states recorded")
    shapes = {tuple(s.shape) for s in states}
    if len(shapes) > 1:
        raise ValueError(
            f"encoder states must all share one shape, got {sorted(shapes)}. This means "
            "the sampling batch changed mid-trajectory; calibrate with settings that keep "
            "it fixed."
        )
    stacked = torch.stack([s.detach().to(torch.float64) for s in states])
    num_steps = stacked.shape[0]
    stacked = torch.nn.functional.normalize(stacked, dim=-1)
    tokens = stacked[0].numel() // stacked.shape[-1]
    flat = stacked.reshape(num_steps, -1)
    return (flat @ flat.T) / tokens


def solve_recompute_schedule(similarity: torch.Tensor, num_recompute: int) -> list[int]:
    """Choose `num_recompute` steps at which to recompute the encoder, by DP.

    Args:
        similarity: [N, N] step-to-step encoder-state similarity, from `state_similarity`.
        num_recompute: how many of the N steps may recompute. The budget, not a hint.

    Returns:
        Sorted step indices, always starting at 0 (step 0 has nothing to reuse).

    The cost of a schedule is the total reuse error: for each recompute point j, the
    error of serving every step from j up to the next recompute point with j's state,
    where error(k, j) = 1 - similarity(k, j).

        C[m][i] = cheapest way to cover steps 0..i-1 with m recomputes
        C[1][i] = cost of serving all of 0..i-1 from step 0
        C[m][i] = min over j < i of  C[m-1][j] + cost(j serves j..i-1)

    O(num_recompute * N^2), which is nothing at N = 50.
    """
    if similarity.ndim != 2 or similarity.shape[0] != similarity.shape[1]:
        raise ValueError(f"similarity must be square, got {tuple(similarity.shape)}")
    num_steps = int(similarity.shape[0])
    if not 1 <= num_recompute <= num_steps:
        raise ValueError(
            f"num_recompute must be in [1, {num_steps}], got {num_recompute}"
        )
    if num_recompute == num_steps:
        return list(range(num_steps))

    error = (1.0 - similarity.to(torch.float64)).clamp_(min=0.0)
    # error[k][j] is only meaningful for k >= j: a step can only reuse a state computed at
    # or before it. Masking the upper triangle makes the column-wise cumulative sum
    # exactly the segment cost, segment[i][j] = sum over k in [j, i] of error[k][j].
    # (The reference reaches the same values with an in-place double loop bounded by
    # j < i, which starts each chain at the zero diagonal; this is that recurrence
    # written as one cumsum, not a correction to it.)
    error = torch.tril(error)
    segment = error.cumsum(dim=0).tolist()  # segment[i][j] = cost of j serving j..i

    inf = float("inf")
    cost = [[inf] * (num_steps + 1) for _ in range(num_recompute + 1)]
    prev = [[-1] * (num_steps + 1) for _ in range(num_recompute + 1)]
    # Step 0 always recomputes, so the one-recompute row is forced rather than searched.
    for i in range(1, num_steps + 1):
        cost[1][i] = segment[i - 1][0]
        prev[1][i] = 0
    for m in range(2, num_recompute + 1):
        for i in range(m, num_steps + 1):
            best, best_j = inf, -1
            for j in range(m - 1, i):
                value = cost[m - 1][j] + segment[i - 1][j]
                if value < best:
                    best, best_j = value, j
            cost[m][i] = best
            prev[m][i] = best_j

    picked = [num_steps]
    for m in range(num_recompute, 0, -1):
        picked.append(prev[m][picked[-1]])
    picked.reverse()
    return picked[:-1]  # drop the sentinel end marker


def calibrate_encoder_schedule(
    model: torch.nn.Module,
    text_embedding: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    steps: int,
    num_recompute: int,
    seed: int = 0,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    return_similarity: bool = False,
    **sample_kwargs,
):
    """Run one full-encoder trajectory, then solve for the recompute schedule.

    Extra keyword arguments go straight to `sample_latent`, and they SHOULD mirror the
    settings you intend to sample with (sampler, flow_shift, cfg_scale, cfg_interval):
    the schedule is only as good as the trajectory it was measured on. The calibration
    pass itself recomputes the encoder at every step, so it costs one ordinary sample.
    """
    from boomer.sampling import sample_latent

    states: list[torch.Tensor] = []
    sample_latent(
        model,
        text_embedding,
        attention_mask,
        steps=steps,
        seed=seed,
        dtype=dtype,
        device=device,
        encoder_reuse_steps=1,
        record_encoder_states=states,
        **sample_kwargs,
    )
    similarity = state_similarity(states)
    schedule = solve_recompute_schedule(similarity, num_recompute)
    return (schedule, similarity) if return_similarity else schedule
