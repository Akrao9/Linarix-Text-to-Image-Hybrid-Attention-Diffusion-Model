"""Dataset loader for Boomer latent cache records."""

from __future__ import annotations

import glob
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import Dataset

from boomer.checkpoint_utils import load_torch_file

CACHE_RECORD_PATTERN = re.compile(r"^\d{8}_[0-9a-f]{12}\.pt$")
CHECKPOINT_NAME_PREFIXES = ("latent_dit",)
EXCLUDED_DIR_NAMES = frozenset({"checkpoints", "checkpoint"})


@dataclass(frozen=True)
class LatentRecord:
    image_latent: torch.Tensor
    text_embedding: torch.Tensor
    attention_mask: torch.Tensor
    caption: str
    metadata: dict[str, Any]
    path: str
    # Optional SSL target for REPA (monet's DINOv2 ViT-G/14 CLS). None when absent.
    dino_embedding: torch.Tensor | None = None


def looks_like_cache_record_name(name: str) -> bool:
    if not name.endswith(".pt"):
        return False
    if name.startswith(CHECKPOINT_NAME_PREFIXES):
        return False
    return CACHE_RECORD_PATTERN.match(name) is not None


def is_cache_record_path(path: Path) -> bool:
    if path.suffix != ".pt":
        return False
    if any(part in EXCLUDED_DIR_NAMES for part in path.parts):
        return False
    if not looks_like_cache_record_name(path.name):
        return False
    return not path.exists() or path.is_file()


def discover_cache_files(paths: str | Path | Iterable[str | Path]) -> list[Path]:
    if isinstance(paths, (str, Path)):
        paths = [paths]

    files: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path).expanduser()
        if path.is_dir():
            for candidate in sorted(path.rglob("*.pt")):
                if is_cache_record_path(candidate):
                    files.append(candidate)
        elif path.is_file():
            if is_cache_record_path(path):
                files.append(path)
            else:
                raise ValueError(f"Not a Boomer cache record path: {path}")
        else:
            matches = [Path(match) for match in sorted(glob.glob(str(path)))]
            for match in matches:
                if match.is_file() and is_cache_record_path(match):
                    files.append(match)

    return sorted(dict.fromkeys(files))


def _drop_singleton_batch(tensor: torch.Tensor, expected_dims: int) -> torch.Tensor:
    if tensor.ndim == expected_dims + 1 and tensor.shape[0] == 1:
        return tensor[0]
    return tensor


def _validate_cache_record(record: dict[str, Any], path: Path) -> None:
    schema_version = record.get("schema_version")
    if schema_version is not None and schema_version not in {1, 2}:
        raise ValueError(f"{path} has unsupported schema_version={schema_version!r}")

    image_latent = record["image_latent"]
    text_embedding = record["text_embedding"]
    attention_mask = record["attention_mask"]

    if image_latent.ndim not in {3, 4}:
        raise ValueError(f"{path} image_latent must be 3D or 4D, got {tuple(image_latent.shape)}")
    if text_embedding.ndim not in {2, 3}:
        raise ValueError(f"{path} text_embedding must be 2D or 3D, got {tuple(text_embedding.shape)}")
    if attention_mask.ndim not in {1, 2, 3, 4}:
        raise ValueError(f"{path} attention_mask has invalid shape {tuple(attention_mask.shape)}")

    image_latent = _drop_singleton_batch(image_latent, expected_dims=3)
    text_embedding = _drop_singleton_batch(text_embedding, expected_dims=2)
    if image_latent.shape[0] != 32 or image_latent.shape[1] != image_latent.shape[2]:
        raise ValueError(f"{path} expected image_latent [C, H, W] with C=32 and H==W, got {tuple(image_latent.shape)}")
    if text_embedding.shape[-1] <= 0:
        raise ValueError(f"{path} text_embedding has empty feature dim")


def load_latent_record(path: str | Path, map_location: str = "cpu") -> LatentRecord:
    path = Path(path)
    record = load_torch_file(path, map_location=map_location)
    required = {"image_latent", "text_embedding", "attention_mask", "caption", "metadata"}
    missing = required.difference(record)
    if missing:
        raise ValueError(f"{path} is missing cache fields: {sorted(missing)}")

    _validate_cache_record(record, path)

    image_latent = _drop_singleton_batch(record["image_latent"], expected_dims=3)
    text_embedding = _drop_singleton_batch(record["text_embedding"], expected_dims=2)
    attention_mask = _drop_singleton_batch(record["attention_mask"], expected_dims=1)

    return LatentRecord(
        image_latent=image_latent.contiguous(),
        text_embedding=text_embedding.contiguous(),
        attention_mask=attention_mask.contiguous(),
        caption=str(record["caption"]),
        metadata=dict(record["metadata"]),
        path=str(path),
    )


class LatentCacheDataset(Dataset[LatentRecord]):
    """PyTorch dataset for `.pt` records produced by `sana_latent_cache.py`."""

    def __init__(self, paths: str | Path | Iterable[str | Path], map_location: str = "cpu") -> None:
        self.files = discover_cache_files(paths)
        if not self.files:
            raise FileNotFoundError(f"No latent cache .pt files found for: {paths}")
        self.map_location = map_location

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> LatentRecord:
        return load_latent_record(self.files[index], map_location=self.map_location)


def collate_latent_records(records: list[LatentRecord]) -> dict[str, Any]:
    dinos = [record.dino_embedding for record in records]
    dino_batch = (
        torch.stack(dinos, dim=0) if dinos and all(d is not None for d in dinos) else None
    )
    return {
        "image_latent": torch.stack([record.image_latent for record in records], dim=0),
        "text_embedding": torch.stack([record.text_embedding for record in records], dim=0),
        "attention_mask": torch.stack([record.attention_mask for record in records], dim=0),
        "caption": [record.caption for record in records],
        "metadata": [record.metadata for record in records],
        "path": [record.path for record in records],
        "dino_embedding": dino_batch,
    }
