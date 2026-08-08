"""DDT encoder-sharing schedule: the similarity measure, the DP, and the wiring.

The DP is the part worth pinning down. It is easy to write one that returns a
plausible-looking list of indices while optimising the wrong objective, so these tests
check the cost the schedule actually achieves, not the shape of the output: against
brute-force enumeration at a size small enough to enumerate, and against a transcription
of DDT's published recurrence.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from boomer.encoder_schedule import (  # noqa: E402
    solve_recompute_schedule,
    state_similarity,
    uniform_schedule,
)
from boomer.sampling import EncoderStateCache  # noqa: E402


def _drifting_states(num_steps: int, drift: float = 0.25, tokens: int = 6, dim: int = 8):
    """A trajectory whose state moves fast early and slowly late.

    That asymmetry is the whole reason the DP exists: a uniform stride wastes recomputes
    on the flat tail. Built by walking a fixed direction with a decaying step size.
    """
    torch.manual_seed(0)
    base = torch.randn(1, tokens, dim)
    direction = torch.randn(1, tokens, dim)
    states = []
    position = 0.0
    for i in range(num_steps):
        position += drift / (i + 1)  # large early increments, tiny later ones
        states.append(base + position * direction)
    return states


# --- similarity ----------------------------------------------------------------


def test_similarity_is_symmetric_with_unit_diagonal():
    sim = state_similarity(_drifting_states(8))
    assert sim.shape == (8, 8)
    assert torch.allclose(sim, sim.T)
    assert torch.allclose(sim.diagonal(), torch.ones(8, dtype=sim.dtype), atol=1e-9)


def test_similarity_decays_with_step_distance():
    sim = state_similarity(_drifting_states(8))
    row = sim[0]
    assert all(row[i] >= row[i + 1] - 1e-9 for i in range(7)), row.tolist()


def test_similarity_averages_per_token_cosines_not_concatenated_vectors():
    """One high-norm token must not dominate the measure.

    Two states that differ only in the scale of a single token are identical in
    direction per token, so every per-token cosine is 1. A cosine over concatenated
    vectors would instead be dragged away by that token's magnitude.
    """
    a = torch.ones(1, 4, 3)
    b = a.clone()
    b[0, 0] *= 100.0
    sim = state_similarity([a, b])
    assert sim[0, 1].item() == pytest.approx(1.0, abs=1e-9)


def test_similarity_rejects_ragged_states():
    with pytest.raises(ValueError, match="share one shape"):
        state_similarity([torch.randn(1, 4, 8), torch.randn(2, 4, 8)])


def test_similarity_rejects_empty():
    with pytest.raises(ValueError, match="no encoder states"):
        state_similarity([])


# --- the dynamic program -------------------------------------------------------


def test_full_budget_recomputes_every_step():
    sim = state_similarity(_drifting_states(10))
    assert solve_recompute_schedule(sim, 10) == list(range(10))


def test_schedule_is_sorted_unique_and_starts_at_zero():
    sim = state_similarity(_drifting_states(20))
    for budget in (1, 2, 5, 9, 13):
        schedule = solve_recompute_schedule(sim, budget)
        assert len(schedule) == budget
        assert schedule == sorted(set(schedule))
        assert schedule[0] == 0
        assert max(schedule) < 20


def test_single_recompute_is_step_zero_only():
    sim = state_similarity(_drifting_states(12))
    assert solve_recompute_schedule(sim, 1) == [0]


def test_schedule_front_loads_when_the_state_drifts_early():
    """The point of the whole exercise: beat the uniform stride where drift is uneven."""
    sim = state_similarity(_drifting_states(24))
    solved = solve_recompute_schedule(sim, 6)
    uniform = uniform_schedule(24, 6)
    assert _cost(sim, solved) < _cost(sim, uniform), (
        f"DP schedule {solved} did not beat uniform {uniform}"
    )
    # front-loaded: more than half the budget lands in the first half of the trajectory
    assert sum(1 for s in solved if s < 12) > 3, solved


def _cost(similarity: torch.Tensor, schedule: list[int]) -> float:
    """Total reuse error of a schedule: each step served by the latest recompute <= it."""
    num_steps = similarity.shape[0]
    total = 0.0
    for step in range(num_steps):
        source = max(s for s in schedule if s <= step)
        total += float(1.0 - similarity[step, source])
    return total


def test_dp_result_is_optimal_against_brute_force():
    """Against every possible schedule at a size small enough to enumerate."""
    from itertools import combinations

    num_steps, budget = 9, 4
    sim = state_similarity(_drifting_states(num_steps))
    solved = solve_recompute_schedule(sim, budget)

    best = min(
        ((0,) + rest for rest in combinations(range(1, num_steps), budget - 1)),
        key=lambda s: _cost(sim, list(s)),
    )
    assert _cost(sim, solved) == pytest.approx(_cost(sim, list(best)), abs=1e-9)


def test_segment_cost_covers_exactly_the_served_segment():
    """Steps 0-2 are mutually identical, 3-5 are mutually identical, groups far apart.

    The only zero-cost 2-recompute schedule is [0, 3].
    """
    a, b = torch.zeros(1, 1, 4), torch.zeros(1, 1, 4)
    a[..., 0], b[..., 1] = 1.0, 1.0
    states = [a, a.clone(), a.clone(), b, b.clone(), b.clone()]
    sim = state_similarity(states)
    schedule = solve_recompute_schedule(sim, 2)
    assert schedule == [0, 3]
    assert _cost(sim, schedule) == pytest.approx(0.0, abs=1e-9)


def _reference_solve(similarity, num_recompute):
    """DDT's `sharing_dp` recurrence, transcribed.

    Kept as the equivalence oracle: the vectorised `tril` + `cumsum` in our solver is a
    restatement of their in-place accumulation (whose inner loop is bounded by j < i, so
    each chain starts at the zero diagonal), not a change to the objective.
    """
    n = similarity.shape[0]
    error_map = (1.0 - similarity).tolist()
    for i in range(1, n):
        for j in range(0, i):
            error_map[i][j] = error_map[i - 1][j] + error_map[i][j]
    cost = [[0.0] * (n + 1) for _ in range(num_recompute + 1)]
    prev = [[-1] * (n + 1) for _ in range(num_recompute + 1)]
    for i in range(1, n + 1):
        cost[1][i] = error_map[i - 1][0]
        prev[1][i] = 0
    for step in range(2, num_recompute + 1):
        for i in range(step, n + 1):
            best, best_j = float("inf"), -1
            for j in range(step - 1, i):
                value = cost[step - 1][j] + error_map[i - 1][j]
                if value < best:
                    best, best_j = value, j
            cost[step][i] = best
            prev[step][i] = best_j
    picked = [n]
    for step in range(num_recompute, 0, -1):
        picked.append(prev[step][picked[-1]])
    picked.reverse()
    return picked[:-1]


@pytest.mark.parametrize("num_steps,budget", [(7, 2), (8, 3), (9, 4), (12, 5)])
def test_matches_the_reference_recurrence(num_steps, budget):
    """Our vectorised solver must agree with DDT's published recurrence."""
    sim = state_similarity(_drifting_states(num_steps))
    assert solve_recompute_schedule(sim, budget) == _reference_solve(sim, budget)


def test_matches_the_reference_on_arbitrary_similarity_matrices():
    """Not just on smooth trajectories -- on unstructured symmetric matrices too."""
    for seed in range(50):
        torch.manual_seed(seed)
        n = 8
        raw = torch.rand(n, n)
        sim = (raw + raw.T) / 2
        sim.fill_diagonal_(1.0)
        assert solve_recompute_schedule(sim, 3) == _reference_solve(sim, 3), seed


def test_rejects_impossible_budgets():
    sim = state_similarity(_drifting_states(6))
    for budget in (0, -1, 7):
        with pytest.raises(ValueError, match="num_recompute"):
            solve_recompute_schedule(sim, budget)


def test_rejects_non_square_similarity():
    with pytest.raises(ValueError, match="square"):
        solve_recompute_schedule(torch.randn(3, 5), 2)


# --- cache wiring --------------------------------------------------------------


def test_cache_honours_an_explicit_schedule():
    cache = EncoderStateCache(reuse_steps=4, refresh_steps=[0, 1, 5, 11])
    refreshed = [i for i in range(12) if cache.refresh_at(i)]
    assert refreshed == [0, 1, 5, 11], "explicit schedule must override the stride"


def test_cache_falls_back_to_the_stride():
    cache = EncoderStateCache(reuse_steps=3)
    assert [i for i in range(9) if cache.refresh_at(i)] == [0, 3, 6]


def test_cache_rejects_a_schedule_without_step_zero():
    with pytest.raises(ValueError, match="include step 0"):
        EncoderStateCache(refresh_steps=[1, 4, 7])


def test_uniform_schedule_matches_the_stride_cache():
    cache = EncoderStateCache(reuse_steps=5)
    assert uniform_schedule(20, 4) == [i for i in range(20) if cache.refresh_at(i)]


# --- end to end through the sampler --------------------------------------------


def _tiny_model():
    from boomer.models.boomer_fla_dit import BoomerFLADiT, BoomerFLADiTConfig

    config = BoomerFLADiTConfig(
        latent_channels=4, latent_size=8, text_dim=16, text_seq_len=6,
        hidden_dim=32, depth=6, num_heads=4, mixer_type="torch",
        image_attention_every=2, dual_stream_depth=1, joint_refresh_attention=True,
        sandwich_rmsnorm=True, use_abs_pos_embed=False, image_attention_rope=True,
        ddt_decoder_depth=2,
    )
    torch.manual_seed(0)
    model = BoomerFLADiT(config).eval()
    torch.nn.init.normal_(model.out_proj.weight, std=0.02)
    for block in model.decoder_blocks:
        torch.nn.init.normal_(block.cond_mod.weight, std=0.02)
    return model, config


def _sample_kwargs(config):
    return dict(
        text_embedding=torch.randn(1, config.text_seq_len, config.text_dim),
        attention_mask=torch.ones(1, config.text_seq_len),
        seed=0, dtype=torch.float32, device="cpu",
        cfg_scale=1.0, sampler="euler",
    )


def test_recording_captures_one_state_per_step():
    from boomer.sampling import sample_latent

    model, config = _tiny_model()
    states: list[torch.Tensor] = []
    sample_latent(model, steps=6, record_encoder_states=states, **_sample_kwargs(config))
    assert len(states) == 6
    assert all(s.shape == states[0].shape for s in states)
    assert all(s.device.type == "cpu" for s in states), "states must not pin GPU memory"


def test_recording_does_not_change_the_sample():
    """Calibration must observe the un-amortized trajectory, not perturb it."""
    from boomer.sampling import sample_latent

    model, config = _tiny_model()
    kwargs = _sample_kwargs(config)
    plain = sample_latent(model, steps=6, **kwargs)
    recorded = sample_latent(model, steps=6, record_encoder_states=[], **kwargs)
    assert torch.equal(plain, recorded)


def test_full_budget_schedule_is_bit_identical_to_no_amortization():
    """A schedule that refreshes every step must reproduce the plain sample exactly."""
    from boomer.sampling import sample_latent

    model, config = _tiny_model()
    kwargs = _sample_kwargs(config)
    plain = sample_latent(model, steps=6, **kwargs)
    every = sample_latent(model, steps=6, encoder_refresh_steps=list(range(6)), **kwargs)
    assert torch.equal(plain, every)


def test_calibrate_then_sample_end_to_end():
    from boomer.encoder_schedule import calibrate_encoder_schedule
    from boomer.sampling import sample_latent

    model, config = _tiny_model()
    kwargs = _sample_kwargs(config)
    schedule, similarity = calibrate_encoder_schedule(
        model, kwargs.pop("text_embedding"), kwargs.pop("attention_mask"),
        steps=8, num_recompute=3, return_similarity=True, **kwargs
    )
    assert len(schedule) == 3 and schedule[0] == 0
    assert similarity.shape == (8, 8)

    out = sample_latent(
        model, steps=8, encoder_refresh_steps=schedule, **_sample_kwargs(config)
    )
    assert out.shape == (1, config.latent_channels, config.latent_size, config.latent_size)
    assert torch.isfinite(out).all()


def test_amortization_still_requires_a_ddt_model():
    from boomer.models.boomer_fla_dit import BoomerFLADiT, BoomerFLADiTConfig
    from boomer.sampling import sample_latent

    config = BoomerFLADiTConfig(
        latent_channels=4, latent_size=8, text_dim=16, text_seq_len=6,
        hidden_dim=32, depth=4, num_heads=4, mixer_type="torch", ddt_decoder_depth=0,
    )
    model = BoomerFLADiT(config).eval()
    with pytest.raises(ValueError, match="DDT model"):
        sample_latent(model, steps=4, encoder_refresh_steps=[0, 2],
                      **_sample_kwargs(config))


def test_cli_parses_an_explicit_schedule_list():
    import argparse
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "sample_latent_dit", ROOT / "tools" / "sample_latent_dit.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    args = argparse.Namespace(encoder_schedule="0,3,7,12", steps=16)
    assert module.resolve_encoder_schedule(None, args, {}) == [0, 3, 7, 12]

    args.encoder_schedule = None
    assert module.resolve_encoder_schedule(None, args, {}) is None

    args.encoder_schedule = " , "
    with pytest.raises(SystemExit, match="empty list"):
        module.resolve_encoder_schedule(None, args, {})
