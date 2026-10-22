"""Linarix training entry point and backwards-compatible training imports."""

from boomer.training.callbacks.checkpoints import CheckpointUploadCallback
from boomer.training.callbacks.ema import EMACallback
from boomer.training.callbacks.previews import CondSamplePreviewCallback, SamplePreviewCallback
from boomer.training.callbacks.progress import StepUnitProgressBar
from boomer.training.cli import build_parser
from boomer.training.config import build_config, parse_int_tuple, text_encoder_hidden_size
from boomer.training.data import (
    BoomerData,
    is_parquet_cache,
    is_remote_cache,
    make_wds_dataset,
    peek_record,
)
from boomer.training.module import BoomerLit, dispersive_loss
from boomer.training.runner import main


if __name__ == "__main__":
    main()
