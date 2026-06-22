import threading
import time
from argparse import Namespace
from pathlib import Path

from tools.hf_rolling_localize import cmd_prefetch


def _prefetch_args(tmp_path: Path, shards: list[str], **overrides) -> Namespace:
    source_list = tmp_path / "source.txt"
    source_list.write_text("\n".join(shards) + "\n", encoding="utf-8")
    values = {
        "repo_id": "example/source",
        "source_list_file": str(source_list),
        "local_dir": str(tmp_path / "local"),
        "window": 4,
        "download_workers": 1,
        "poll_interval": 0.001,
        "max_download_attempts": 1,
        "retry_delay": 0.001,
        "download_timeout": 30.0,
        "skip_on_failure": False,
        "skip_suffix": ".skip",
        "max_consecutive_skips": 5,
        "skipped_log": None,
        "token": None,
        "token_env": None,
        "error_file": None,
    }
    values.update(overrides)
    return Namespace(**values)


def test_prefetch_ignores_stale_ready_files_outside_current_admission(tmp_path, monkeypatch):
    args = _prefetch_args(tmp_path, ["00452.tar"])
    local_dir = Path(args.local_dir)
    local_dir.mkdir(parents=True)
    for index in range(453, 457):
        stale = local_dir / f"{index:05d}.tar"
        stale.write_bytes(b"stale")
        Path(f"{stale}.ready").write_text("ready\n", encoding="utf-8")

    calls: list[str] = []

    def fake_download(repo_id, shard, *, repo_type, token, local_dir):  # noqa: ANN001
        calls.append(shard)
        target = Path(local_dir) / shard
        target.write_bytes(b"downloaded")
        return str(target)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)

    assert cmd_prefetch(args) == 0
    assert calls == ["00452.tar"]
    assert (local_dir / "00452.tar").is_file()
    assert Path(f"{local_dir / '00452.tar'}.ready").is_file()


def test_prefetch_retries_transient_download_failure(tmp_path, monkeypatch):
    args = _prefetch_args(tmp_path, ["00001.tar"], max_download_attempts=2)
    attempts = 0

    def flaky_download(repo_id, shard, *, repo_type, token, local_dir):  # noqa: ANN001
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("temporary xet failure")
        target = Path(local_dir) / shard
        target.write_bytes(b"downloaded")
        return str(target)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", flaky_download)

    assert cmd_prefetch(args) == 0
    assert attempts == 2


def test_prefetch_publishes_terminal_error_for_parent(tmp_path, monkeypatch):
    error_file = tmp_path / "prefetch.error"
    args = _prefetch_args(tmp_path, ["00001.tar"], error_file=str(error_file))

    def failed_download(*_args, **_kwargs):
        raise RuntimeError("permanent download failure")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", failed_download)

    assert cmd_prefetch(args) == 1
    error = error_file.read_text(encoding="utf-8")
    assert "00001.tar" in error
    assert "permanent download failure" in error


def test_prefetch_recovers_from_stalled_download(tmp_path, monkeypatch):
    # A download that hangs without raising must be abandoned by the watchdog and
    # retried, not block the head-of-line shard forever (the 00452 timeout cause).
    args = _prefetch_args(
        tmp_path,
        ["00001.tar"],
        max_download_attempts=2,
        download_timeout=0.05,
    )
    attempts = 0

    def stalls_then_succeeds(repo_id, shard, *, repo_type, token, local_dir):  # noqa: ANN001
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            time.sleep(30)  # never returns within the watchdog window
        target = Path(local_dir) / shard
        target.write_bytes(b"downloaded")
        return str(target)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", stalls_then_succeeds)

    assert cmd_prefetch(args) == 0
    assert attempts == 2
    assert (Path(args.local_dir) / "00001.tar").is_file()


def test_prefetch_fails_fast_when_download_never_progresses(tmp_path, monkeypatch):
    error_file = tmp_path / "prefetch.error"
    args = _prefetch_args(
        tmp_path,
        ["00001.tar"],
        max_download_attempts=2,
        download_timeout=0.05,
        error_file=str(error_file),
    )
    started = threading.Event()

    def never_returns(*_args, **_kwargs):
        started.set()
        time.sleep(30)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", never_returns)

    assert cmd_prefetch(args) == 1
    assert started.is_set()
    assert "00001.tar" in error_file.read_text(encoding="utf-8")


def test_prefetch_skips_dead_shard_and_continues(tmp_path, monkeypatch):
    # With --skip-on-failure, a shard that fails every attempt is marked .skip and the
    # sidecar keeps going to the next shard instead of aborting the whole run.
    skipped_log = tmp_path / "skipped.txt"
    args = _prefetch_args(
        tmp_path,
        ["00001.tar", "00002.tar"],
        max_download_attempts=1,
        skip_on_failure=True,
        skipped_log=str(skipped_log),
    )

    def fail_first_shard(repo_id, shard, *, repo_type, token, local_dir):  # noqa: ANN001
        if shard == "00001.tar":
            raise RuntimeError("dead shard")
        target = Path(local_dir) / shard
        target.write_bytes(b"downloaded")
        return str(target)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fail_first_shard)

    assert cmd_prefetch(args) == 0  # run survives
    local_dir = Path(args.local_dir)
    assert Path(f"{local_dir / '00001.tar'}.skip").is_file()  # marker for the cacher
    assert (local_dir / "00002.tar").is_file()                # next shard still fetched
    assert "00001.tar" in skipped_log.read_text(encoding="utf-8")


def test_prefetch_aborts_after_too_many_consecutive_skips(tmp_path, monkeypatch):
    error_file = tmp_path / "prefetch.error"
    args = _prefetch_args(
        tmp_path,
        [f"{i:05d}.tar" for i in range(10)],
        download_workers=1,
        max_download_attempts=1,
        skip_on_failure=True,
        max_consecutive_skips=3,
        error_file=str(error_file),
    )

    def always_fail(*_args, **_kwargs):
        raise RuntimeError("outage")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", always_fail)

    assert cmd_prefetch(args) == 1  # a real outage still aborts
    assert "consecutive" in error_file.read_text(encoding="utf-8").lower()
