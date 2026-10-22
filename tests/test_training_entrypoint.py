"""Compatibility of the training script after moving its components into a package."""

from __future__ import annotations

import pickle
import runpy
from pathlib import Path

import pytest

import train
from boomer.training import runner
from boomer.training.callbacks.ema import EMACallback


@pytest.mark.parametrize("module_name, expected_calls", [("__main__", 1), ("train_import", 0)])
def test_entrypoint_runs_training_only_when_executed(monkeypatch, module_name, expected_calls):
    calls = []
    monkeypatch.setattr(runner, "main", lambda: calls.append("fit"))
    namespace = runpy.run_path(str(Path(train.__file__)), run_name=module_name)

    assert calls == ["fit"] * expected_calls
    assert namespace["BoomerLit"] is train.BoomerLit
    assert namespace["build_parser"] is train.build_parser


def test_callback_pickle_from_the_old_train_module_still_loads(monkeypatch):
    callback = EMACallback(decay=0.99, update_every=4)
    callback._last_step = 100_000
    # Existing serialized objects name the old train.EMACallback location.
    with monkeypatch.context() as patch:
        patch.setattr(EMACallback, "__module__", "train")
        legacy_payload = pickle.dumps(callback)

    restored = pickle.loads(legacy_payload)
    assert type(restored) is EMACallback
    assert restored.decay == 0.99
    assert restored.update_every == 4
    assert restored._last_step == 100_000
