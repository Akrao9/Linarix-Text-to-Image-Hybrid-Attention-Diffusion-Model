"""Directory/mount checkpoint upload: fsync-durable copies and the final flush.

Regression for the Colab-Drive case where a plain copy into a drivefs mount
returns success while the background upload is still in flight, so the last
checkpoint of a run silently never lands. _upload_one now fsyncs each file and
finalize_uploads issues a global sync."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import train  # noqa: E402


def _cb(tmp_path: Path) -> "train.CheckpointUploadCallback":
    local = tmp_path / "local"
    remote = tmp_path / "remote"
    local.mkdir()
    return train.CheckpointUploadCallback(
        checkpoint_dir=str(local), remote=str(remote), local_retain=2
    )


def _write_ckpt(cb, name: str, payload: bytes) -> Path:
    p = cb.checkpoint_dir / name
    p.write_bytes(payload)
    return p


def test_upload_one_copies_and_fsyncs_file_and_dir(tmp_path) -> None:
    cb = _cb(tmp_path)
    Path(cb.remote).mkdir()
    src = _write_ckpt(cb, "boomer-step=100.ckpt", b"weights" * 1000)

    fsynced: list[int] = []
    with mock.patch.object(train.os, "fsync", side_effect=lambda fd: fsynced.append(fd)):
        cb._upload_one(src)

    dest = Path(cb.remote) / src.name
    assert dest.exists()
    assert dest.read_bytes() == src.read_bytes()
    # No leftover .uploading temp; atomic replace landed the final name.
    assert not list(Path(cb.remote).glob("*.uploading"))
    # fsync fired for both the file and the directory entry (durability push).
    assert len(fsynced) >= 2


def test_upload_one_survives_mount_without_fsync(tmp_path) -> None:
    # Some mounts raise on fsync; the copy must still succeed (best-effort).
    cb = _cb(tmp_path)
    Path(cb.remote).mkdir()
    src = _write_ckpt(cb, "boomer-step=200.ckpt", b"x" * 500)
    with mock.patch.object(train.os, "fsync", side_effect=OSError("fsync unsupported")):
        cb._upload_one(src)
    assert (Path(cb.remote) / src.name).read_bytes() == src.read_bytes()


def test_finalize_uploads_flushes_and_syncs(tmp_path) -> None:
    cb = _cb(tmp_path)
    Path(cb.remote).mkdir()
    a = _write_ckpt(cb, "boomer-step=100.ckpt", b"a" * 10)
    b = _write_ckpt(cb, "boomer-step=200.ckpt", b"b" * 10)

    synced = {"n": 0}
    with mock.patch.object(train.os, "sync", side_effect=lambda: synced.__setitem__("n", synced["n"] + 1)):
        cb.finalize_uploads(SimpleNamespace(global_rank=0))

    assert (Path(cb.remote) / a.name).exists()
    assert (Path(cb.remote) / b.name).exists()
    assert cb._uploaded == {a.name, b.name}
    assert synced["n"] == 1  # the belt-and-suspenders mount flush ran


def test_finalize_uploads_skips_sync_for_rclone(tmp_path) -> None:
    # rclone/hf remotes upload synchronously via their CLI: no mount sync needed.
    local = tmp_path / "local"
    local.mkdir()
    cb = train.CheckpointUploadCallback(
        checkpoint_dir=str(local), remote="gdrive:Boomer/ckpts"
    )
    assert cb.use_rclone
    with mock.patch.object(train.os, "sync", side_effect=AssertionError("must not sync")):
        cb.finalize_uploads(SimpleNamespace(global_rank=0))  # no local ckpts -> no-op
