"""--fla-mode is actually wired to the mixer (it was silently ignored for GDN2) and
validated per mixer; and last.ckpt is a symlink rather than a second full serialization."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from boomer.models.boomer_fla_dit import (  # noqa: E402
    FLA_SUPPORTED_MODES,
    BoomerFLADiTConfig,
    FLASelfMixer,
)


def test_supported_mode_table_matches_fla_layer_constraints() -> None:
    # GDN/GDN2 reject fused_chunk; GDN2 is chunk-only because FLA asserts that on the
    # training path, and Boomer trains the same mixer it samples with.
    assert "fused_chunk" not in FLA_SUPPORTED_MODES["fla_gated_deltanet"]
    assert FLA_SUPPORTED_MODES["fla_gdn2"] == ("chunk",)
    assert "fused_chunk" in FLA_SUPPORTED_MODES["fla_linear"]
    assert "fused_chunk" in FLA_SUPPORTED_MODES["fla_gla"]


@pytest.mark.parametrize(
    "mixer,mode",
    [
        ("fla_gdn2", "fused_chunk"),      # previously silently ignored
        ("fla_gdn2", "fused_recurrent"),  # rejected by FLA in training
        ("fla_gated_deltanet", "fused_chunk"),
    ],
)
def test_unsupported_mode_rejected_with_clear_message(mixer, mode) -> None:
    config = BoomerFLADiTConfig(mixer_type=mixer, fla_mode=mode, hidden_dim=32, num_heads=4)
    with pytest.raises(ValueError, match="not supported by mixer_type"):
        FLASelfMixer(config, layer_idx=0)


def test_valid_pairs_pass_validation() -> None:
    """The validation must not fire for legal combinations. Construction itself needs the
    FLA package (Triton), so only assert the check is not what stops us."""
    for mixer, modes in FLA_SUPPORTED_MODES.items():
        for mode in modes:
            config = BoomerFLADiTConfig(mixer_type=mixer, fla_mode=mode, hidden_dim=32, num_heads=4)
            try:
                FLASelfMixer(config, layer_idx=0)
            except ValueError as exc:  # must never be OUR validation error
                assert "not supported by mixer_type" not in str(exc)
            except Exception:
                pass  # missing fla/triton locally is fine


def test_gdn2_actually_receives_the_mode_argument(monkeypatch) -> None:
    """Regression: GatedDeltaNet2 was constructed without mode=, so --fla-mode was inert
    for the mixer the v7 run actually uses. Capture the real constructor kwargs via a
    stubbed fla.layers rather than scraping source."""
    import types

    captured: dict[str, dict] = {}

    def _recorder(name):
        def _ctor(**kwargs):
            captured[name] = kwargs
            return torch_nn_identity()
        return _ctor

    def torch_nn_identity():
        from torch import nn
        return nn.Identity()

    fake_layers = types.ModuleType("fla.layers")
    for cls in ("GatedDeltaNet2", "GatedDeltaNet", "GatedLinearAttention", "LinearAttention"):
        setattr(fake_layers, cls, _recorder(cls))
    fake_pkg = types.ModuleType("fla")
    fake_pkg.layers = fake_layers
    monkeypatch.setitem(sys.modules, "fla", fake_pkg)
    monkeypatch.setitem(sys.modules, "fla.layers", fake_layers)

    config = BoomerFLADiTConfig(
        mixer_type="fla_gdn2", fla_mode="chunk", hidden_dim=32, num_heads=4,
        fla_bidirectional=False,
    )
    FLASelfMixer(config, layer_idx=3)
    assert captured["GatedDeltaNet2"]["mode"] == "chunk"
    # layer_idx stays unset: deliberate, so Dynamo does not specialize per block index.
    assert captured["GatedDeltaNet2"]["layer_idx"] is None


def test_checkpoint_uses_symlinked_last() -> None:
    """save_last='link' symlinks last.ckpt at the step file instead of serializing the
    whole ~15 GB state twice per save. Safe here because the pruner and uploader both
    skip last.ckpt (train.py), so the link target is always a retained step file."""
    import inspect

    import train as train_module

    # Strip comments: the rationale comment names the old save_last=True form.
    code = "\n".join(line.split("#")[0] for line in inspect.getsource(train_module.main).splitlines())
    assert 'save_last="link"' in code
    assert "save_last=True" not in code
