"""Qwen readout: after truncating to the readout layer the connector must stop
requesting output_hidden_states (HF retains EVERY layer output when it is set — GBs
at training shapes — for a tensor we never index) while returning a bit-identical
readout via last_hidden_state."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from boomer.models.qwen_connector import (  # noqa: E402
    QwenConnectorConfig,
    QwenReferenceEncoder,
)

HIDDEN = 8
LAYERS = 6
SEQ = 4


def _states(count: int) -> tuple[torch.Tensor, ...]:
    """Deterministic per-index states so a readout can be identified by its value."""
    return tuple(torch.full((1, SEQ, HIDDEN), float(i)) for i in range(count))


class _FakeBase(nn.Module):
    """The Qwen base transformer: owns .layers and .norm, and returns
    last_hidden_state = norm(hidden_states[-1]) exactly as HF's base model does."""

    def __init__(self, calls: list[tuple[str, bool]]) -> None:
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(HIDDEN, HIDDEN) for _ in range(LAYERS))
        self.norm = nn.Linear(HIDDEN, HIDDEN)  # a real (non-Identity) final norm
        self._calls = calls

    def forward(self, *, output_hidden_states: bool, use_cache: bool, return_dict: bool, **inputs):
        self._calls.append(("base", output_hidden_states))
        states = _states(len(self.layers) + 1)
        return SimpleNamespace(
            last_hidden_state=self.norm(states[-1]),
            hidden_states=states if output_hidden_states else None,
        )


class _FakeCausalLM(nn.Module):
    """The full CausalLM wrapper: holds the base as .model and — like HF — returns
    NO last_hidden_state, so this path always needs output_hidden_states."""

    def __init__(self, calls: list[tuple[str, bool]]) -> None:
        super().__init__()
        self.model = _FakeBase(calls)
        self._calls = calls

    def forward(self, *, output_hidden_states: bool, use_cache: bool, return_dict: bool, **inputs):
        self._calls.append(("full", output_hidden_states))
        states = _states(len(self.model.layers) + 1)
        return SimpleNamespace(hidden_states=states if output_hidden_states else None)


def _encoder(readout_layer: int):
    calls: list[tuple[str, bool]] = []
    enc = QwenReferenceEncoder(QwenConnectorConfig(readout_layer=readout_layer))
    full = _FakeCausalLM(calls)
    enc._model = full
    enc._base_model = full.model
    # forward() only uses _build_inputs for tokenization; feed it a fixed mask.
    enc._build_inputs = lambda prompts, images: {  # type: ignore[method-assign]
        "attention_mask": torch.ones(1, SEQ, dtype=torch.long)
    }
    return enc, full, calls


def test_truncated_readout_skips_output_hidden_states() -> None:
    enc, full, calls = _encoder(readout_layer=-2)
    enc._truncate_to_readout(full)

    assert enc._truncated_to_last is True
    assert len(full.model.layers) == LAYERS - 1  # -2 drops the final layer
    assert isinstance(full.model.norm, nn.Identity)  # so last == the raw readout

    hidden, mask = enc(["a prompt"])
    assert calls == [("base", False)]  # the point: no all-layer materialization
    assert hidden.shape == (1, SEQ, HIDDEN)
    assert mask.shape == (1, SEQ)


def test_truncated_readout_is_bit_identical_to_indexing_hidden_states() -> None:
    """The saving must not change a single value: on the truncated stack
    last_hidden_state has to equal hidden_states[readout_layer], the pre-change read."""
    enc, full, _ = _encoder(readout_layer=-2)
    enc._truncate_to_readout(full)

    new_hidden, _ = enc(["a prompt"])

    # Reproduce the pre-change path: request the states and index the readout layer.
    out = full.model(output_hidden_states=True, use_cache=False, return_dict=True)
    old_hidden = out.hidden_states[enc.config.readout_layer]

    assert torch.equal(new_hidden, old_hidden)
    # And it is genuinely the readout layer, not some other entry.
    assert torch.equal(new_hidden, torch.full((1, SEQ, HIDDEN), float(LAYERS - 1)))


def test_untruncated_final_readout_also_skips_hidden_states() -> None:
    """A genuine final-layer readout keeps the real norm, but last_hidden_state is the
    post-final-norm output the readout wants anyway — so the flag is unnecessary here
    too, and skipping it also removes any chance of drifting onto the version-dependent
    hidden_states[-1]."""
    enc, full, calls = _encoder(readout_layer=-1)
    enc._truncate_to_readout(full)

    assert enc._truncated_to_last is False  # nothing to drop
    assert len(full.model.layers) == LAYERS
    assert not isinstance(full.model.norm, nn.Identity)

    hidden, _ = enc(["a prompt"])
    assert calls == [("base", False)]
    # Post-final-norm, not the raw last layer.
    expected = full.model.norm(_states(LAYERS + 1)[-1])
    assert torch.equal(hidden, expected)


def test_untruncatable_intermediate_readout_still_requests_hidden_states() -> None:
    """If truncation could not find the layer stack, an intermediate readout genuinely
    has to index hidden_states — the optimization must not swallow that case."""
    enc, full, calls = _encoder(readout_layer=-2)
    # Never truncated: _truncated_to_last stays False. Force the intermediate-readout
    # branch the way a structure we failed to recognize would leave it.
    enc._readout_is_final = False
    enc.config.readout_layer = LAYERS - 1

    hidden, _ = enc(["a prompt"])
    assert calls == [("base", True)]
    assert torch.equal(hidden, torch.full((1, SEQ, HIDDEN), float(LAYERS - 1)))


def test_compile_base_model_keeps_eager_for_fallback(monkeypatch) -> None:
    """compile_base_model wraps only the BASE transformer and retains the eager one."""
    enc, full, _ = _encoder(readout_layer=-2)
    enc._truncate_to_readout(full)
    eager = enc._base_model

    # Real torch.compile returns an OptimizedModule (an nn.Module); the encoder is an
    # nn.Module too, so _base_model must stay assignable as a child module.
    sentinel = nn.Identity()
    monkeypatch.setattr(torch, "compile", lambda m, **kw: sentinel)
    assert enc.compile_base_model() is True
    assert enc._base_model is sentinel
    assert enc._eager_base_model is eager  # kept for fallback
    assert enc._base_is_compiled is True
    assert enc.compile_base_model() is True  # idempotent
    assert enc._eager_base_model is eager


def test_compiled_base_failure_falls_back_to_eager_not_causal_lm(monkeypatch) -> None:
    """A compiled-graph failure must retry the EAGER BASE. Falling through to the full
    CausalLM would run lm_head (~1 GiB of logits) for the rest of the run."""
    enc, full, calls = _encoder(readout_layer=-2)
    enc._truncate_to_readout(full)

    class _Boom(nn.Module):
        """Stands in for a compiled graph that blows up at call time."""

        def forward(self, **kwargs):
            raise RuntimeError("inductor exploded")

    monkeypatch.setattr(torch, "compile", lambda m, **kw: _Boom())
    enc.compile_base_model()

    hidden, _ = enc(["a prompt"])
    # The eager BASE served the call; the full CausalLM was never touched.
    assert calls == [("base", False)]
    assert enc._base_failed is False
    assert enc._base_is_compiled is False  # compilation permanently dropped
    assert torch.equal(hidden, torch.full((1, SEQ, HIDDEN), float(LAYERS - 1)))

    enc(["another"])  # subsequent calls stay on the eager base, no re-raise
    assert calls == [("base", False), ("base", False)]


def test_compile_base_model_noop_without_base() -> None:
    enc, _, _ = _encoder(readout_layer=-2)
    enc._base_model = None
    assert enc.compile_base_model() is False


def test_full_model_fallback_requests_hidden_states() -> None:
    """The full CausalLM has no last_hidden_state, so that runner always needs the flag
    even though the stack is truncated."""
    enc, full, calls = _encoder(readout_layer=-2)
    enc._truncate_to_readout(full)
    enc._base_failed = True  # force the runner onto the full model

    hidden, _ = enc(["a prompt"])
    assert calls == [("full", True)]
    # Still the readout layer, via hidden_states this time.
    assert torch.equal(hidden, torch.full((1, SEQ, HIDDEN), float(LAYERS - 1)))
