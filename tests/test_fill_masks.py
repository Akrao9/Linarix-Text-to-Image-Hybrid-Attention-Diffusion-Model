"""--fill-training mask synthesis: mix parsing, per-type geometry, determinism."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from boomer.fill_masks import (  # noqa: E402
    DEFAULT_FILL_MASK_PROBS,
    parse_fill_mask_probs,
    sample_fill_masks,
)


def test_parse_defaults_and_normalization() -> None:
    assert parse_fill_mask_probs(None) == DEFAULT_FILL_MASK_PROBS
    assert parse_fill_mask_probs("") == DEFAULT_FILL_MASK_PROBS
    probs = parse_fill_mask_probs("blob=1,full=3")
    assert probs["blob"] == pytest.approx(0.25)
    assert probs["full"] == pytest.approx(0.75)
    assert probs["rect"] == 0.0  # unlisted types get zero, not the default

    with pytest.raises(ValueError, match="unknown fill mask type"):
        parse_fill_mask_probs("wedge=1.0")
    with pytest.raises(ValueError, match="positive sum"):
        parse_fill_mask_probs("blob=0")


def test_shapes_dtype_and_binary_values() -> None:
    masks = sample_fill_masks(8, 32, 32, dtype=torch.bfloat16)
    assert masks.shape == (8, 1, 32, 32)
    assert masks.dtype == torch.bfloat16
    unique = set(masks.float().unique().tolist())
    assert unique <= {0.0, 1.0}


def test_full_type_is_all_ones() -> None:
    masks = sample_fill_masks(4, 16, 16, probs={"full": 1.0})
    assert torch.equal(masks, torch.ones(4, 1, 16, 16))


def test_rect_is_a_solid_rectangle() -> None:
    masks = sample_fill_masks(6, 32, 32, probs={"rect": 1.0})
    for mask in masks[:, 0]:
        rows = mask.any(dim=1).nonzero().flatten()
        cols = mask.any(dim=0).nonzero().flatten()
        assert len(rows) > 0 and len(cols) > 0
        # The nonzero bounding box is fully filled -> a single solid rectangle.
        box = mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]
        assert bool(box.all())


def test_outpaint_keeps_a_solid_window() -> None:
    masks = sample_fill_masks(6, 32, 32, probs={"outpaint": 1.0})
    for mask in masks[:, 0]:
        kept = 1.0 - mask
        assert float(mask.sum()) > 0  # something is generated
        rows = kept.any(dim=1).nonzero().flatten()
        cols = kept.any(dim=0).nonzero().flatten()
        assert len(rows) > 0 and len(cols) > 0  # something is kept
        # The kept region is one solid rectangle (the source window).
        box = kept[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]
        assert bool(box.all())
        assert float(kept.mean()) <= 0.5  # keep window is at most 0.7 x 0.7


def test_tiny_masks_are_small() -> None:
    masks = sample_fill_masks(16, 32, 32, probs={"tiny": 1.0})
    assert float(masks.mean(dim=(1, 2, 3)).max()) < 0.35


def test_generator_determinism() -> None:
    a = sample_fill_masks(5, 32, 32, generator=torch.Generator().manual_seed(7))
    b = sample_fill_masks(5, 32, 32, generator=torch.Generator().manual_seed(7))
    c = sample_fill_masks(5, 32, 32, generator=torch.Generator().manual_seed(8))
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_non_square_grid() -> None:
    masks = sample_fill_masks(4, 16, 32, probs={"blob": 0.5, "rect": 0.5})
    assert masks.shape == (4, 1, 16, 32)
