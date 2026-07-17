"""Stream instruction-editing image PAIRS directly for training (no cache build).

Built for TIGER-Lab/OmniEdit-Filtered-1.2M (--edit-direct): each parquet row
carries a source image, an edited target image, and instruction strings —

  * ``src_img`` / ``edited_img`` — HF Image structs ({bytes, path}), typically
    1152x768-ish JPEG/PNG, NOT square and NOT 1024.
  * ``edited_prompt_list`` — [short instruction, detailed instruction].
  * ``task`` / ``o_score`` (+ sc/pq scores) — task family and quality scores
    (the Filtered release is already o_score >= 9).

No latents exist in the corpus, so everything is computed on the fly: the
loader decodes + crops + resizes the pixel pair on CPU workers, and train.py
encodes both sides through the frozen DC-AE on-GPU (plus Qwen for the
instruction, same as --monet-direct text).

Edit-aware shared crop: the pair must receive the IDENTICAL square window or
source and target misalign and the model learns spurious global shifts. Worse,
a naive center crop of a 1344x768 frame discards ~40% of the width — when the
edit lives there, the pair degenerates to (image, instruction, same image):
actively harmful "ignore the instruction" supervision. So the crop window is
centered on the |src - target| difference centroid (computed on cheap
thumbnails), clamped to the frame, then applied to BOTH images.

Streaming reuses MonetDirectStreamingDataset's whole per-file pyarrow range
machinery (rank/worker shard split, transient-error skip, reservoir shuffle);
only the schema check, column projection, and row decode differ. NOTE pixel
records are ~6 MB each (2 x 3 x 1024^2 uint8) — a latent-cache-sized shuffle
buffer would be tens of GB, so train.py clamps the buffer in edit mode.
"""

from __future__ import annotations

import io
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from PIL import Image

from boomer.datasets.monet_direct import MonetDirectStreamingDataset

EDIT_SRC_KEY = "src_img"
EDIT_TGT_KEY = "edited_img"
EDIT_PROMPT_KEY = "edited_prompt_list"

# Cheap scalar columns kept for filtering/provenance when the shard has them.
_EDIT_META_COLUMNS = ("omni_edit_id", "task", "o_score", "width", "height")


@dataclass(frozen=True)
class EditRecord:
    """One editing pair: uint8 CHW pixels (identically cropped) + instruction."""

    source_pixels: torch.Tensor  # (3, R, R) uint8
    target_pixels: torch.Tensor  # (3, R, R) uint8
    caption: str  # the edit instruction
    metadata: dict[str, Any]
    path: str


def _to_bytes(value: Any) -> bytes | None:
    """HF Image parquet cell -> raw encoded bytes ({bytes,path} struct or bytes)."""
    if isinstance(value, dict):
        value = value.get("bytes")
    return value if isinstance(value, (bytes, bytearray)) else None


def shared_edit_crop_box(
    source: Image.Image, target: Image.Image
) -> tuple[int, int, int, int]:
    """The square crop window (left, top, right, bottom) both images receive.

    Centered on the source/target difference centroid so the edited region
    survives the square crop; computed on ~1/16-scale RGB thumbnails (a few
    thousand pixels), then clamped inside the frame. Identical-image pairs
    (or square frames) fall back to the center.
    """
    width, height = source.size
    side = min(width, height)
    if width == height:
        return (0, 0, width, height)
    thumb_w, thumb_h = max(1, width // 16), max(1, height // 16)
    a = np.asarray(source.resize((thumb_w, thumb_h), Image.BILINEAR), dtype=np.float32)
    b = np.asarray(target.resize((thumb_w, thumb_h), Image.BILINEAR), dtype=np.float32)
    diff = np.abs(a - b).sum(axis=-1) if a.ndim == 3 else np.abs(a - b)
    total = float(diff.sum())
    if total <= 1e-6:
        cx, cy = width / 2.0, height / 2.0
    else:
        ys, xs = np.mgrid[0:thumb_h, 0:thumb_w]
        cx = float((diff * xs).sum()) / total * (width / thumb_w)
        cy = float((diff * ys).sum()) / total * (height / thumb_h)
    left = min(max(0, int(round(cx - side / 2.0))), width - side)
    top = min(max(0, int(round(cy - side / 2.0))), height - side)
    return (left, top, left + side, top + side)


# Pairs whose aspect ratios disagree by more than this are dropped: registration
# would need a crop so aggressive that source and target likely no longer show
# the same framing (corrupt/mispaired row).
_MAX_ASPECT_MISMATCH = 1.5


def register_source_to_target(
    source: Image.Image, target: Image.Image
) -> Image.Image | None:
    """Bring the source onto the target's pixel grid so the pair can share a crop.

    OmniEdit frequently stores the source at its ORIGINAL resolution while the
    edited image is the generator's output size (often the same aspect, smaller).
    Dropping those pairs would discard most of the corpus, so instead: center-crop
    the source to the target's aspect (a no-op when aspects match), then resize to
    the target's exact dimensions. Returns None when the aspects are irreconcilable.
    """
    if source.size == target.size:
        return source
    src_w, src_h = source.size
    tgt_w, tgt_h = target.size
    src_aspect = src_w / src_h
    tgt_aspect = tgt_w / tgt_h
    ratio = max(src_aspect / tgt_aspect, tgt_aspect / src_aspect)
    if ratio > _MAX_ASPECT_MISMATCH:
        return None
    if src_aspect > tgt_aspect:  # source too wide: crop width
        new_w = max(1, round(src_h * tgt_aspect))
        left = (src_w - new_w) // 2
        source = source.crop((left, 0, left + new_w, src_h))
    elif src_aspect < tgt_aspect:  # source too tall: crop height
        new_h = max(1, round(src_w / tgt_aspect))
        top = (src_h - new_h) // 2
        source = source.crop((0, top, src_w, top + new_h))
    return source.resize((tgt_w, tgt_h), Image.BICUBIC)


def _pixels(image: Image.Image, box: tuple[int, int, int, int], resolution: int) -> torch.Tensor:
    cropped = image.crop(box)
    if cropped.size != (resolution, resolution):
        cropped = cropped.resize((resolution, resolution), Image.BICUBIC)
    # np.array (copy): PIL hands out a read-only buffer, and torch.from_numpy on a
    # non-writable array is undefined-behavior territory (and warns).
    array = np.array(cropped, dtype=np.uint8)
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def edit_row_to_record(
    row: dict[str, Any],
    *,
    resolution: int = 1024,
    src_key: str = EDIT_SRC_KEY,
    tgt_key: str = EDIT_TGT_KEY,
    prompt_key: str = EDIT_PROMPT_KEY,
    prompt_pick: str = "random",
    rng: random.Random | None = None,
    min_o_score: float | None = None,
    tasks: tuple[str, ...] | None = None,
    min_short_side: int = 0,
) -> EditRecord | None:
    """Decode one OmniEdit row, or None when it fails a filter / is unusable.

    ``prompt_pick``: "random" samples among the row's instructions (phrasing
    augmentation: OmniEdit pairs a short and a detailed form), "first"/"last"
    always take that entry.
    """
    if tasks is not None:
        task = row.get("task")
        if task not in tasks:
            return None
    if min_o_score is not None:
        score = row.get("o_score")
        if score is None or score < min_o_score:
            return None

    prompts = row.get(prompt_key)
    prompts = [str(p) for p in (prompts if prompts is not None else []) if str(p).strip()]
    if not prompts:
        return None
    if prompt_pick == "first":
        caption = prompts[0]
    elif prompt_pick == "last":
        caption = prompts[-1]
    else:
        chooser = rng if rng is not None else random
        caption = chooser.choice(prompts)

    src_bytes = _to_bytes(row.get(src_key))
    tgt_bytes = _to_bytes(row.get(tgt_key))
    if src_bytes is None or tgt_bytes is None:
        return None
    key = row.get("omni_edit_id") or ""
    try:
        source = Image.open(io.BytesIO(src_bytes)).convert("RGB")
        target = Image.open(io.BytesIO(tgt_bytes)).convert("RGB")
    except Exception as exc:  # noqa: BLE001 - one corrupt image must not kill the stream
        print(f"[edit-pairs] drop undecodable row {key!r}: {exc}", flush=True)
        return None
    if min(source.size) < min_short_side or min(target.size) < min_short_side:
        return None
    if source.size != target.size:
        # Common in OmniEdit (source at original res, edited at generator res):
        # register the source onto the target grid instead of dropping the row.
        source = register_source_to_target(source, target)
        if source is None:
            print(f"[edit-pairs] drop aspect-mismatched row {key!r}", flush=True)
            return None

    box = shared_edit_crop_box(source, target)
    return EditRecord(
        source_pixels=_pixels(source, box, resolution),
        target_pixels=_pixels(target, box, resolution),
        caption=caption,
        metadata={
            "key": str(key),
            "task": row.get("task"),
            "o_score": row.get("o_score"),
            "crop": box,
            "size": tuple(source.size),
        },
        path=str(key),
    )


def collate_edit_records(records: list[EditRecord]) -> dict[str, Any]:
    """Collate for --edit-direct batches: pixel pairs + raw instructions.

    ``image_latent``/``text_embedding`` are deliberately ABSENT — training_step
    encodes pixels with the resident frozen DC-AE and captions with the resident
    Qwen encoder.
    """
    return {
        "source_pixels": torch.stack([r.source_pixels for r in records], dim=0),
        "target_pixels": torch.stack([r.target_pixels for r in records], dim=0),
        "caption": [r.caption for r in records],
        "metadata": [r.metadata for r in records],
    }


@dataclass
class EditPairsStreamingDataset(MonetDirectStreamingDataset):
    """Range-stream OmniEdit-style parquet shards, yielding EditRecord pairs.

    Inherits monet-direct's per-file pyarrow streaming (rank/worker shard split,
    reconnect budget, reservoir shuffle, glob mixing); overrides the schema
    check, the column projection, and the row decode. The monet caption/dino
    fields are unused.
    """

    resolution: int = 1024
    src_key: str = EDIT_SRC_KEY
    tgt_key: str = EDIT_TGT_KEY
    prompt_key: str = EDIT_PROMPT_KEY
    prompt_pick: str = "random"  # random | first | last
    min_o_score: float | None = None
    tasks: tuple[str, ...] | None = None
    min_short_side: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.tasks, str):  # tolerate a CSV string from the CLI
            self.tasks = tuple(t.strip() for t in self.tasks.split(",") if t.strip()) or None
        if self.prompt_pick not in ("random", "first", "last"):
            raise ValueError(f"prompt_pick must be random|first|last, got {self.prompt_pick!r}")
        super().__post_init__()

    def _shard_usable(self, path: str, schema: Any) -> bool:
        available = set(schema.names)
        missing = {self.src_key, self.tgt_key, self.prompt_key} - available
        if missing:
            print(f"[edit-pairs] skip {path}: missing column(s) {sorted(missing)}", flush=True)
            return False
        return True

    def _wanted_columns(self, available: set[str]) -> list[str]:
        columns = [self.src_key, self.tgt_key, self.prompt_key]
        columns.extend(c for c in _EDIT_META_COLUMNS if c in available)
        return list(dict.fromkeys(columns))

    def _row_to_record(self, row: dict[str, Any]) -> EditRecord | None:  # type: ignore[override]
        return edit_row_to_record(
            row,
            resolution=self.resolution,
            src_key=self.src_key,
            tgt_key=self.tgt_key,
            prompt_key=self.prompt_key,
            prompt_pick=self.prompt_pick,
            rng=self._caption_rng,
            min_o_score=self.min_o_score,
            tasks=self.tasks,
            min_short_side=self.min_short_side,
        )


__all__ = [
    "EditPairsStreamingDataset",
    "EditRecord",
    "collate_edit_records",
    "edit_row_to_record",
    "register_source_to_target",
    "shared_edit_crop_box",
]
