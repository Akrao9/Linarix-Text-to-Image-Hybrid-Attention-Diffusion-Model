from pathlib import Path
import io
import tarfile

import pytest

from boomer.datasets.wds_latents import HfWdsShardRef, RollingLocalWdsCache, _normalize_shard_name


def _fake_download(_shard: HfWdsShardRef, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = f"payload:{dest.name}".encode("utf-8")
    with tarfile.open(dest, "w") as archive:
        info = tarfile.TarInfo(name="sample.txt")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    return dest


def test_normalize_shard_name():
    assert _normalize_shard_name("/tmp/train-000010.tar") == "train-000010.tar"
    assert _normalize_shard_name("pipe:curl ::/tmp/train-000010.tar") == "train-000010.tar"


def test_rolling_local_wds_cache_shuffled_initial_window(tmp_path, monkeypatch):
    refs = [HfWdsShardRef("repo", f"train/train-{index:06d}.tar") for index in range(30)]
    monkeypatch.setattr(
        "boomer.datasets.wds_latents.list_hf_wds_shard_refs",
        lambda *_args, **_kwargs: refs,
    )
    monkeypatch.setattr(
        "boomer.datasets.wds_latents.load_wds_cache_manifest",
        lambda *_args, **_kwargs: None,
    )

    cache = RollingLocalWdsCache(
        remote_paths="hf://datasets/repo/train/*.tar",
        local_dir=tmp_path / "window",
        max_local_shards=25,
        seed=0,
        download_fn=_fake_download,
    )
    names = {Path(url).name for url in cache.local_urls()}
    assert len(names) == 25
    assert names.issubset({f"train-{index:06d}.tar" for index in range(30)})
    assert names != {f"train-{index:06d}.tar" for index in range(25)}


def test_rolling_local_wds_cache_default_download_uses_local_dir(tmp_path, monkeypatch):
    refs = [HfWdsShardRef("repo", "train/train-000000.tar")]
    monkeypatch.setattr(
        "boomer.datasets.wds_latents.list_hf_wds_shard_refs",
        lambda *_args, **_kwargs: refs,
    )
    monkeypatch.setattr(
        "boomer.datasets.wds_latents.load_wds_cache_manifest",
        lambda *_args, **_kwargs: None,
    )
    seen: dict[str, object] = {}

    def _fake_hf_hub_download(
        *,
        repo_id,
        filename,
        repo_type,
        token,
        local_dir,
    ):  # noqa: ANN001
        seen["local_dir"] = local_dir
        dest = Path(local_dir) / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        payload = b"payload"
        with tarfile.open(dest, "w") as archive:
            info = tarfile.TarInfo(name="sample.txt")
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
        return str(dest)

    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download",
        _fake_hf_hub_download,
    )

    local_dir = tmp_path / "window"
    cache = RollingLocalWdsCache(
        remote_paths="hf://datasets/repo/train/*.tar",
        local_dir=local_dir,
        max_local_shards=1,
        seed=0,
    )
    dest = local_dir / "train" / "train-000000.tar"
    assert dest.is_file()
    assert seen["local_dir"] == str(local_dir)
    assert cache.local_urls() == [str(dest)]


def test_rolling_local_wds_cache_fill_initial_without_deadlock(tmp_path, monkeypatch):
    """fill_initial must not hold the window lock while _default_download finalizes."""
    refs = [HfWdsShardRef("repo", f"train/train-{index:06d}.tar") for index in range(5)]
    monkeypatch.setattr(
        "boomer.datasets.wds_latents.list_hf_wds_shard_refs",
        lambda *_args, **_kwargs: refs,
    )
    monkeypatch.setattr(
        "boomer.datasets.wds_latents.load_wds_cache_manifest",
        lambda *_args, **_kwargs: None,
    )

    def _locking_download(_shard: HfWdsShardRef, dest: Path) -> Path:
        dest.parent.mkdir(parents=True, exist_ok=True)
        payload = f"payload:{dest.name}".encode("utf-8")
        with tarfile.open(dest, "w") as archive:
            info = tarfile.TarInfo(name="sample.txt")
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
        return dest

    cache = RollingLocalWdsCache(
        remote_paths="hf://datasets/repo/train/*.tar",
        local_dir=tmp_path / "lock",
        max_local_shards=3,
        seed=0,
        download_fn=_locking_download,
    )
    assert len(cache.local_urls()) == 3


def test_rolling_local_wds_cache_epoch_refill(tmp_path, monkeypatch):
    refs = [HfWdsShardRef("repo", f"train/train-{index:06d}.tar") for index in range(30)]
    monkeypatch.setattr(
        "boomer.datasets.wds_latents.list_hf_wds_shard_refs",
        lambda *_args, **_kwargs: refs,
    )
    monkeypatch.setattr(
        "boomer.datasets.wds_latents.load_wds_cache_manifest",
        lambda *_args, **_kwargs: None,
    )

    cache = RollingLocalWdsCache(
        remote_paths="hf://datasets/repo/train/*.tar",
        local_dir=tmp_path / "window",
        max_local_shards=25,
        seed=0,
        download_fn=_fake_download,
    )
    first = Path(cache.local_urls()[0]).name
    cache.advance_after(first)
    names = {Path(url).name for url in cache.local_urls()}
    assert len(names) == 25
    assert first not in names
    assert not list((tmp_path / "window").rglob(first))


def test_rolling_local_wds_cache_covers_epoch_before_repeating(tmp_path, monkeypatch):
    refs = [HfWdsShardRef("repo", f"train/train-{index:06d}.tar") for index in range(5)]
    monkeypatch.setattr(
        "boomer.datasets.wds_latents.list_hf_wds_shard_refs",
        lambda *_args, **_kwargs: refs,
    )
    monkeypatch.setattr(
        "boomer.datasets.wds_latents.load_wds_cache_manifest",
        lambda *_args, **_kwargs: None,
    )

    cache = RollingLocalWdsCache(
        remote_paths="hf://datasets/repo/train/*.tar",
        local_dir=tmp_path / "seq",
        max_local_shards=2,
        seed=123,
        download_fn=_fake_download,
    )
    seen: list[str] = []
    for _ in range(5):
        current = cache.current_shard_path()
        assert current is not None
        seen.append(current.name)
        cache.advance_after(current.name)

    assert len(seen) == 5
    assert len(set(seen)) == 5
    assert all(name.startswith("train-") for name in seen)


def test_rolling_local_wds_cache_reshuffles_after_full_epoch(tmp_path, monkeypatch):
    refs = [HfWdsShardRef("repo", f"train/train-{index:06d}.tar") for index in range(3)]
    monkeypatch.setattr(
        "boomer.datasets.wds_latents.list_hf_wds_shard_refs",
        lambda *_args, **_kwargs: refs,
    )
    monkeypatch.setattr(
        "boomer.datasets.wds_latents.load_wds_cache_manifest",
        lambda *_args, **_kwargs: None,
    )

    cache = RollingLocalWdsCache(
        remote_paths="hf://datasets/repo/train/*.tar",
        local_dir=tmp_path / "wrap",
        max_local_shards=2,
        seed=7,
        download_fn=_fake_download,
    )
    seen: list[str] = []
    for _ in range(4):
        current = cache.current_shard_path()
        assert current is not None
        seen.append(current.name)
        cache.advance_after(current.name)
    urls = cache.local_urls()
    assert len(urls) == 2
    assert len(set(seen[:3])) == 3
    assert seen[3] in {f"train-{index:06d}.tar" for index in range(3)}
