"""Hilbert token ordering: equivalence to the scalar reference, and the size that broke.

The permutation is built with vectorized tensor ops because the original
`for d in range(n*n): perm[d] = ...` was traced by torch.compile into an
`ops.where` chain n*n deep. That compiled at the 1024px phase's 32x32 grid and
blew up Inductor at the 2048px phase's 64x64 one.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from boomer.models.boomer_fla_dit import (  # noqa: E402
    _hilbert_d2xy,
    hilbert_permutation,
)

# 32 = the 1024px latent grid (resumed checkpoints must see an identical order),
# 64 = the 2048px grid that triggered the compile failure.
GRIDS = [2, 4, 8, 16, 32, 64, 128]


@pytest.mark.parametrize("n", GRIDS)
def test_matches_the_scalar_reference_exactly(n):
    """Bit-identical, so a checkpoint trained under the scalar version is unaffected."""
    ref = torch.tensor([(lambda xy: xy[1] * n + xy[0])(_hilbert_d2xy(n, d)) for d in range(n * n)])
    assert torch.equal(hilbert_permutation(n, n), ref)


@pytest.mark.parametrize("n", GRIDS)
def test_is_a_permutation_whose_every_step_is_grid_adjacent(n):
    """The property the ordering exists for: sequential tokens are 2D neighbours, so a
    recurrent mixer's recency signal tracks vertical adjacency instead of jumping a
    full row at each raster wrap."""
    perm = hilbert_permutation(n, n)
    assert torch.equal(torch.sort(perm).values, torch.arange(n * n)), "not a permutation"

    xs, ys = perm % n, perm // n
    steps = (xs[1:] - xs[:-1]).abs() + (ys[1:] - ys[:-1]).abs()
    assert torch.equal(steps, torch.ones_like(steps)), "a step left the 4-neighbourhood"


@pytest.mark.parametrize("n", GRIDS)
def test_argsort_inverts_it(n):
    """`x[:, perm]` reorders into Hilbert order; argsort(perm) must restore raster."""
    perm = hilbert_permutation(n, n)
    x = torch.randn(2, n * n, 3)
    assert torch.equal(x[:, perm][:, torch.argsort(perm)], x)


def test_rejects_grids_it_cannot_map():
    with pytest.raises(ValueError, match="square"):
        hilbert_permutation(32, 64)
    with pytest.raises(ValueError, match="power-of-two"):
        hilbert_permutation(48, 48)


def test_build_cost_is_logarithmic_not_quadratic():
    """Guards the actual regression: the old loop ran n*n Python iterations, each
    emitting a traced scalar write. Op count must scale with log2(n), not n*n."""
    counts = {}
    for n in (32, 64):
        seen = 0
        orig = torch.where

        def counting_where(*args, **kwargs):
            nonlocal seen
            seen += 1
            return orig(*args, **kwargs)

        torch.where = counting_where
        try:
            hilbert_permutation(n, n)
        finally:
            torch.where = orig
        counts[n] = seen

    # 4 `where`s per bit-level, 5 levels at n=32 and 6 at n=64 -- a constant-factor
    # step, where the scalar version would have gone 1024 -> 4096.
    assert counts[32] < 40 and counts[64] < 40, counts
    assert counts[64] - counts[32] <= 8, counts
