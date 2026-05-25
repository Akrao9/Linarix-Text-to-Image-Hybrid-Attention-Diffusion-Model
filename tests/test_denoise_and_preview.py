from __future__ import annotations

import csv
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from boomer.datasets.latent_shards import LatentRecord  # noqa: E402
from boomer.denoise import denoise_latent  # noqa: E402
from tools.overfit_latent_dit import normalize_preview_records, open_loss_csv, save_denoise_previews  # noqa: E402


class ConstantVelocityModel(nn.Module):
    def __init__(self, velocity: torch.Tensor) -> None:
        super().__init__()
        self.velocity = velocity

    def forward(
        self,
        latent: torch.Tensor,
        _timestep: torch.Tensor,
        _text_embedding: torch.Tensor,
        _attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.velocity.expand_as(latent)


def test_denoise_latent_euler_runs() -> None:
    noised = torch.randn(1, 32, 16, 16)
    model = ConstantVelocityModel(torch.zeros_like(noised))
    text = torch.randn(1, 300, 1536)
    mask = torch.ones(1, 300, dtype=torch.long)

    denoised = denoise_latent(
        model,
        noised,
        0.5,
        text,
        mask,
        4,
        3.0,
        sampler="euler",
    )
    assert denoised.shape == noised.shape
    assert torch.isfinite(denoised).all()


def test_normalize_preview_records_splits_csv_and_list() -> None:
    assert normalize_preview_records(["a.pt", "b.pt,c.pt"]) == ["a.pt", "b.pt", "c.pt"]
    assert normalize_preview_records(None) == []


def test_open_loss_csv_appends_on_resume(tmp_path: Path) -> None:
    csv_path = tmp_path / "loss.csv"
    csv_path.write_text("step,loss,loss_ma,lr\n1,1.0,1.0,1e-4\n", encoding="utf-8")

    csv_file, writer, appended = open_loss_csv(csv_path, resume=True)
    try:
        assert appended is True
        writer.writerow({"step": 2, "loss": 0.5, "loss_ma": 0.75, "lr": 1e-4})
    finally:
        csv_file.close()

    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    assert [row["step"] for row in rows] == ["1", "2"]


@patch("tools.overfit_latent_dit.load_latent_record")
@patch("tools.overfit_latent_dit.decode_image_latent")
@patch("tools.overfit_latent_dit.tensor_to_image")
@patch("tools.overfit_latent_dit.denoise_latent")
def test_save_denoise_previews_writes_multiple_samples(
    mock_denoise: MagicMock,
    mock_tensor_to_image: MagicMock,
    _mock_decode: MagicMock,
    mock_load_record: MagicMock,
    tmp_path: Path,
) -> None:
    record = LatentRecord(
        image_latent=torch.randn(32, 16, 16),
        text_embedding=torch.randn(300, 1536),
        attention_mask=torch.ones(300, dtype=torch.int16),
        caption="test",
        metadata={},
        path="sample.pt",
    )
    mock_load_record.return_value = record
    mock_denoise.return_value = record.image_latent.unsqueeze(0)
    image = MagicMock()
    mock_tensor_to_image.return_value = image

    model = ConstantVelocityModel(torch.zeros(1, 32, 16, 16))
    cache_config = SimpleNamespace()
    record_paths = [tmp_path / f"sample_{index}.pt" for index in range(2)]

    mse_noised, mse_denoised = save_denoise_previews(
        model=model,
        vae=MagicMock(),
        record_paths=record_paths,
        out_dir=tmp_path / "previews",
        cache_config=cache_config,
        step=4,
        preview_t=0.3,
        preview_steps=4,
        preview_sampler="euler",
        stork_derivative_order=1,
        stork_substeps=5,
        seed=0,
        device="cpu",
        dtype=torch.float32,
        latent_mean=0.0,
        latent_std=1.0,
        flow_shift=3.0,
    )

    assert mock_denoise.call_count == 2
    assert mock_tensor_to_image.call_count == 6
    assert mse_noised >= 0.0
    assert mse_denoised >= 0.0


@patch("boomer.denoise.make_stork_scheduler")
def test_denoise_latent_stork2_uses_scheduler(mock_make_stork: MagicMock) -> None:
    sigmas = torch.tensor([0.8, 0.4, 0.0])
    scheduler = MagicMock()
    scheduler.sigmas = sigmas
    scheduler.step.return_value = SimpleNamespace(prev_sample=torch.zeros(1, 32, 16, 16))
    mock_make_stork.return_value = scheduler

    model = ConstantVelocityModel(torch.zeros(1, 32, 16, 16))
    text = torch.randn(1, 300, 1536)
    mask = torch.ones(1, 300, dtype=torch.long)
    noised = torch.randn(1, 32, 16, 16)

    out = denoise_latent(
        model,
        noised,
        0.8,
        text,
        mask,
        2,
        3.0,
        sampler="stork2",
        stork_derivative_order=1,
        stork_substeps=5,
    )
    assert out.shape == noised.shape
    assert mock_make_stork.call_count == 1
    assert scheduler.step.call_count == 2
