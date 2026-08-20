"""Regression tests for the self-contained Hugging Face export contract."""

from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
import types
from pathlib import Path

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]


def _load_export_module():
    spec = importlib.util.spec_from_file_location(
        "boomer_export_tool_for_tests", ROOT / "tools" / "export_to_hf.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_pipeline_module(monkeypatch):
    class FakePipeline(nn.Module):
        def __init__(self):
            super().__init__()

        def register_modules(self, **modules):
            for name, module in modules.items():
                setattr(self, name, module)

        @property
        def _execution_device(self):
            return torch.device("cpu")

    diffusers = types.ModuleType("diffusers")
    diffusers.DiffusionPipeline = FakePipeline
    utils = types.ModuleType("diffusers.utils")
    utils.logging = types.SimpleNamespace(get_logger=logging.getLogger)
    monkeypatch.setitem(sys.modules, "diffusers", diffusers)
    monkeypatch.setitem(sys.modules, "diffusers.utils", utils)
    spec = importlib.util.spec_from_file_location(
        "boomer_pipeline_for_tests", ROOT / "pipeline_boomer.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def _args(**overrides):
    values = dict(
        te_repo=None,
        text_encoder_layer=None,
        text_encoder_layers=None,
        instruction=None,
        max_text_length=None,
        vae_repo=None,
        flow_shift=None,
        scaling_factor=None,
        latent_mean=None,
        latent_std=None,
        sampler=None,
        steps=None,
        stork_derivative_order=None,
        stork_substeps=None,
        cfg_scale=None,
        cfg_rescale=None,
        cfg_interval_low=None,
        cfg_interval_high=None,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


def test_export_recovers_v8_text_contract_from_checkpoint() -> None:
    export = _load_export_module()
    ckpt = {
        "train_args": {
            "text_encoder_id": "Qwen/Qwen3.5-4B",
            "text_encoder_layer": -2,
            "text_encoder_layers": "9,18,27",
            "text_encoder_max_length": 320,
            "text_encoder_instruction": True,
            "sample_flow_shift": 5.0,
            "sample_scaling_factor": 0.41407,
        },
        "config": {"text_seq_len": 320},
    }
    meta = export._resolve_metadata(ckpt, _args())
    assert meta.te_repo == "Qwen/Qwen3.5-4B"
    assert meta.text_encoder_layer == -2
    assert meta.text_encoder_layers == (9, 18, 27)
    assert meta.max_text_length == 320
    assert meta.instruction == export.DEFAULT_INSTRUCTION
    assert meta.flow_shift == 5.0
    # Recommended exported inference defaults, independent of an old preview config.
    assert (meta.sampler, meta.steps) == ("stork4", 20)
    assert (meta.stork_derivative_order, meta.stork_substeps) == (1, 9)
    assert (meta.cfg_scale, meta.cfg_rescale) == (3.0, 0.5)
    assert (meta.cfg_interval_low, meta.cfg_interval_high) == (0.1, 0.9)


def test_export_metadata_explicit_overrides_win() -> None:
    export = _load_export_module()
    meta = export._resolve_metadata(
        {
            "train_args": {"text_encoder_id": "old", "text_encoder_layers": "1,2"},
            "config": {"text_seq_len": 320},
        },
        _args(
            te_repo="new",
            text_encoder_layers="3,6,9",
            flow_shift=7.0,
            sampler="stork2",
            cfg_interval_low=0.0,
            cfg_interval_high=1.0,
        ),
    )
    assert meta.te_repo == "new"
    assert meta.text_encoder_layers == (3, 6, 9)
    assert meta.flow_shift == 7.0
    assert meta.sampler == "stork2"
    assert (meta.cfg_interval_low, meta.cfg_interval_high) == (0.0, 1.0)


def test_max_text_length_follows_text_seq_len_not_train_args() -> None:
    """The cached-text path auto-detects text_seq_len and leaves the CLI flag stale.

    text_seq_len sizes the learned `null_text_embedding`; exporting the shorter
    train_args value makes the pipeline slice trained null tokens off the CFG uncond
    branch, which degrades guidance silently rather than raising.
    """
    export = _load_export_module()
    meta = export._resolve_metadata(
        {
            "train_args": {"text_encoder_max_length": 300, "text_seq_len": 224},
            "config": {"text_seq_len": 320},
        },
        _args(),
    )
    assert meta.max_text_length == 320
    # An explicit flag still wins, for re-exporting an old checkpoint deliberately.
    forced = export._resolve_metadata(
        {"train_args": {}, "config": {"text_seq_len": 320}}, _args(max_text_length=300)
    )
    assert forced.max_text_length == 300


def test_stork_export_fails_closed_and_requires_constants(tmp_path, monkeypatch) -> None:
    export = _load_export_module()
    monkeypatch.setattr(export, "REPO_ROOT", tmp_path / "repo")
    incomplete = tmp_path / "stork"
    incomplete.mkdir()
    (incomplete / "STORKScheduler.py").write_text("# scheduler without constants\n")
    with pytest.raises(SystemExit, match="refusing to create an export"):
        export._copy_stork(tmp_path / "out", incomplete)


def test_exported_qwen_connector_is_standalone() -> None:
    source = (ROOT / "boomer" / "models" / "qwen_connector.py").read_text()
    assert "if TYPE_CHECKING:" in source
    assert "from boomer.models.boomer_fla_dit import BoomerFLADiTConfig" in source
    # The only runtime DiT import belongs to the optional composite model, not the
    # QwenReferenceEncoder path copied into the HF snapshot.
    assert source.count("from boomer.models.boomer_fla_dit import") == 2
    assert '"qwen_connector.py"' in (ROOT / "tools" / "export_to_hf.py").read_text()


def test_hf_pipeline_uses_shared_qwen_and_current_sampling_contract() -> None:
    source = (ROOT / "pipeline_boomer.py").read_text()
    assert '"qwen_connector.py"' in source
    assert "qwen.encode_qwen_text_fixed" in source
    assert "pad_to_max_length=False" in source
    assert 'cfg.get("_sampler", "stork4")' in source
    assert 'cfg.get("_steps", 20)' in source
    assert 'cfg.get("_stork_substeps", 9)' in source
    assert "cfg_interval_low" in source and "cfg_interval_high" in source
    assert 'hf_hub_download("akrao9/Boomer-T2I"' not in source


def test_hf_pipeline_reads_exported_defaults(monkeypatch) -> None:
    pipeline = _load_pipeline_module(monkeypatch)

    class Transformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = nn.Parameter(torch.zeros(()))
            self.config = types.SimpleNamespace(text_dim=7680)
            self._boomer_cfg = {
                "_te_repo": "Qwen/Qwen3.5-4B",
                "_text_encoder_layers": [9, 18, 27],
                "_max_text_length": 320,
                "_sampler": "stork4",
                "_steps": 20,
                "_stork_derivative_order": 1,
                "_stork_substeps": 9,
                "_cfg_scale": 3.0,
                "_cfg_rescale": 0.5,
                "_cfg_interval_low": 0.1,
                "_cfg_interval_high": 0.9,
            }

    pipe = pipeline.BoomerPipeline(Transformer())
    assert pipe._te_repo == "Qwen/Qwen3.5-4B"
    assert pipe.text_encoder_layers == (9, 18, 27)
    assert pipe.max_text_length == 320
    assert pipe.default_sampler == "stork4"
    assert (pipe.default_steps, pipe.default_stork_substeps) == (20, 9)
    assert pipe.default_cfg_interval == (0.1, 0.9)


def test_pipeline_only_loads_code_from_its_own_snapshot(tmp_path, monkeypatch) -> None:
    pipeline = _load_pipeline_module(monkeypatch)
    (tmp_path / "qwen_connector.py").write_text("VALUE = 17\n")
    loaded = pipeline._load_bundled_module("fixture", "qwen_connector.py", str(tmp_path))
    assert loaded.VALUE == 17
    with pytest.raises(FileNotFoundError, match="missing from this model snapshot"):
        pipeline._find_bundled_file("not_there.py", str(tmp_path))


def test_hf_pipeline_applies_cfg_only_inside_exported_interval(monkeypatch) -> None:
    pipeline = _load_pipeline_module(monkeypatch)

    class Transformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = nn.Parameter(torch.zeros(()))
            self.config = types.SimpleNamespace(
                text_dim=8, latent_channels=2, latent_size=2
            )
            self._boomer_cfg = {
                "_sampler": "stork4",
                "_steps": 3,
                "_cfg_scale": 3.0,
                "_cfg_rescale": 0.5,
                "_cfg_interval_low": 0.1,
                "_cfg_interval_high": 0.9,
            }
            self.batch_sizes = []

        def null_condition(self, batch, *, device, dtype, mask_dtype, token_num):
            return (
                torch.zeros(batch, token_num, 8, device=device, dtype=dtype),
                torch.zeros(batch, token_num, device=device, dtype=mask_dtype),
            )

        def forward(self, latent, timestep, text, mask):
            self.batch_sizes.append(latent.shape[0])
            return torch.zeros_like(latent)

    class Scheduler:
        sigmas = torch.tensor([1.0, 0.5, 0.0])

        def step(self, velocity, sigma, latent, return_dict):
            return types.SimpleNamespace(prev_sample=latent)

    class VAE(nn.Module):
        def decode(self, latent, return_dict=False):
            return (torch.zeros(latent.shape[0], 3, 4, 4),)

    transformer = Transformer()
    pipe = pipeline.BoomerPipeline(transformer, text_encoder=nn.Identity(), vae=VAE())
    pipe._encode_qwen_text_fixed = lambda encoder, prompts, max_length: (
        torch.zeros(len(prompts), 4, 8),
        torch.ones(len(prompts), 4, dtype=torch.long),
    )
    monkeypatch.setattr(pipeline, "_make_stork_scheduler", lambda **kwargs: Scheduler())
    result = pipe("test", seed=42, offload_text_encoder=False)
    assert len(result) == 1
    # sigma 1.0 and 0.0 are outside [0.1, 0.9] -> conditional only. sigma 0.5 is
    # inside -> [unconditional, conditional] doubled batch.
    assert transformer.batch_sizes == [1, 2, 1]


def test_generated_hf_loader_is_strict() -> None:
    generated = (ROOT / "modeling_boomer_fla.py").read_text()
    assert "model.load_state_dict(sd, strict=True)" in generated
    assert 'kwargs.pop("torch_dtype", None)' in generated
    assert "model = model.to(dtype=requested_dtype)" in generated
