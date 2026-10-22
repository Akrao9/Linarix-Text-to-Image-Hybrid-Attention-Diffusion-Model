"""Training data selection, streaming loaders, and dimension peeking."""

from __future__ import annotations

import argparse
import os
from typing import Any

from torch.utils.data import DataLoader

from boomer.datasets.latent_shards import collate_latent_records
from boomer.datasets.monet_direct import (
    MONET_CAPTION_KEYS,
    MONET_DINO_KEY,
    MONET_LATENT_KEY,
    MonetDirectStreamingDataset,
    collate_monet_records,
)
from boomer.datasets.monet_image_direct import (
    MonetImageStreamingDataset,
    collate_monet_image_records,
)
from boomer.datasets.wds_latents import (
    ParquetStreamingLatentCacheDataset,
    PrefetchingWdsLatentCacheDataset,
    StreamingWdsLatentCacheDataset,
)
from boomer.datasets.edit_pairs import (
    EditPairsStreamingDataset,
    collate_edit_records,
)
from boomer.datasets.fine_t2i import (
    DEFAULT_T2I_REPO,
    FineT2IStreamingDataset,
    collate_t2i_records,
)
from boomer.training._lightning import pl


def is_remote_cache(cache_wds: str) -> bool:
    return "hf://" in cache_wds


def is_parquet_cache(cache_wds: str) -> bool:
    """Parquet cache (range-streamed via `datasets`), local or hf://. Takes precedence
    over the tar-shard paths in make_wds_dataset / the worker-count logic."""
    return ".parquet" in cache_wds


def make_wds_dataset(
    *,
    cache_wds: str,
    local_dir: str,
    prefetch_shards: int,
    download_workers: int,
    shuffle_buffer: int,
    hf_token: str | bool | None,
    seed: int,
    decode_queue_size: int = 512,
    shard_shuffle: int = 0,
    download_max_retries: int = 2,
    max_consecutive_skips: int = 8,
    parquet_read_batch_size: int = 32,
    parquet_memory_trim_every: int = 4096,
    parquet_prefetch_ranges: bool = True,
    parquet_range_size_mib: int = 64,
    captions_only: bool = False,
):
    """Parquet caches range-stream via `datasets` (faster + stall-resilient); remote
    hf:// tar caches use the parallel download-ahead prefetch window (thread-pool into
    local_dir, delete after use); local tar globs read directly with plain streaming."""
    if is_parquet_cache(cache_wds):
        return ParquetStreamingLatentCacheDataset(
            paths=cache_wds,
            hf_token=hf_token,
            shuffle_buffer=shuffle_buffer,
            seed=seed,
            download_max_retries=download_max_retries,
            read_batch_size=parquet_read_batch_size,
            memory_trim_every=parquet_memory_trim_every,
            prefetch_ranges=parquet_prefetch_ranges,
            range_size_mib=parquet_range_size_mib,
            captions_only=captions_only,
        )
    if is_remote_cache(cache_wds):
        return PrefetchingWdsLatentCacheDataset(
            paths=cache_wds,
            local_dir=local_dir,
            prefetch_shards=prefetch_shards,
            download_workers=download_workers,
            hf_token=hf_token,
            shuffle_buffer=shuffle_buffer,
            seed=seed,
            decode_queue_size=decode_queue_size,
            download_max_retries=download_max_retries,
            max_consecutive_skips=max_consecutive_skips,
            captions_only=captions_only,
        )
    return StreamingWdsLatentCacheDataset(
        paths=cache_wds,
        hf_token=hf_token,
        shardshuffle=shard_shuffle if shard_shuffle > 0 else True,
        shuffle_buffer=shuffle_buffer,
        captions_only=captions_only,
    )


class BoomerData(pl.LightningDataModule):
    def __init__(
        self,
        *,
        cache_wds: str,
        local_dir: str,
        prefetch_shards: int,
        download_workers: int,
        shuffle_buffer: int,
        hf_token: str | bool | None,
        batch_size: int,
        seed: int,
        num_workers: int,
        decode_queue_size: int = 512,
        shard_shuffle: int = 0,
        pin_memory: bool = True,
        prefetch_factor: int = 2,
        persistent_workers: bool | None = None,
        stream_block_size: int = 1 << 20,
        download_max_retries: int = 2,
        max_consecutive_skips: int = 8,
        parquet_read_batch_size: int = 32,
        parquet_memory_trim_every: int = 4096,
        parquet_prefetch_ranges: bool = True,
        parquet_range_size_mib: int = 64,
        captions_only: bool = False,
        monet_direct: bool = False,
        monet_image_direct: bool = False,
        monet_image_resolution: int = 1024,
        monet_image_min_short_side: int = 0,
        monet_latent_key: str = MONET_LATENT_KEY,
        monet_caption_keys: tuple[str, ...] = MONET_CAPTION_KEYS,
        monet_caption_probs: Any = None,  # CSV / ';'-grouped str, tuple, or None
        monet_weights: tuple[float, ...] | None = None,
        monet_dino_key: str | None = MONET_DINO_KEY,
        monet_min_aesthetic: float | None = None,
        monet_max_nsfw: float | None = None,
        monet_max_watermark: float | None = None,
        edit_direct: bool = False,
        edit_resolution: int = 1024,
        edit_min_o_score: float | None = None,
        edit_tasks: tuple[str, ...] | None = None,
        edit_prompt_pick: str = "random",
        edit_min_short_side: int = 0,
        edit_max_aspect_mismatch: float = 1.05,
        t2i_direct: bool = False,
        t2i_repo: str = DEFAULT_T2I_REPO,
        t2i_config: str | None = None,
        t2i_data_dir: str | None = None,
        t2i_weights: str | None = None,
        max_screened_per_shard: str | None = None,
        t2i_split: str = "train",
        t2i_resolution: int = 1024,
        t2i_caption_mode: str = "sample",
        t2i_short_prob: float = 0.5,
        t2i_min_short_side: int = 0,
        t2i_min_aesthetic: float | None = None,
    ) -> None:
        super().__init__()
        self.captions_only = captions_only
        self.monet_direct = monet_direct
        self.monet_image_direct = monet_image_direct
        self.monet_image_resolution = monet_image_resolution
        self.monet_image_min_short_side = monet_image_min_short_side
        self.edit_direct = edit_direct
        self.edit_resolution = edit_resolution
        self.edit_min_o_score = edit_min_o_score
        self.edit_tasks = edit_tasks
        self.edit_prompt_pick = edit_prompt_pick
        self.edit_min_short_side = edit_min_short_side
        self.edit_max_aspect_mismatch = edit_max_aspect_mismatch
        self.t2i_direct = t2i_direct
        self.t2i_repo = t2i_repo
        self.t2i_config = t2i_config
        self.t2i_data_dir = t2i_data_dir
        self.t2i_weights = t2i_weights
        self.max_screened_per_shard = max_screened_per_shard
        self.t2i_split = t2i_split
        self.t2i_resolution = t2i_resolution
        self.t2i_caption_mode = t2i_caption_mode
        self.t2i_short_prob = t2i_short_prob
        self.t2i_min_short_side = t2i_min_short_side
        self.t2i_min_aesthetic = t2i_min_aesthetic
        self.monet_latent_key = monet_latent_key
        self.monet_caption_keys = monet_caption_keys
        self.monet_caption_probs = monet_caption_probs
        self.monet_weights = monet_weights
        self.monet_dino_key = monet_dino_key
        self.monet_min_aesthetic = monet_min_aesthetic
        self.monet_max_nsfw = monet_max_nsfw
        self.monet_max_watermark = monet_max_watermark
        self.cache_wds = cache_wds
        self.local_dir = local_dir
        self.prefetch_shards = prefetch_shards
        self.download_workers = download_workers
        self.shuffle_buffer = shuffle_buffer
        self.hf_token = hf_token
        self.batch_size = batch_size
        self.seed = seed
        self.num_workers = num_workers
        self.decode_queue_size = decode_queue_size
        self.shard_shuffle = shard_shuffle
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers
        self.stream_block_size = stream_block_size
        self.download_max_retries = download_max_retries
        self.max_consecutive_skips = max_consecutive_skips
        self.parquet_read_batch_size = parquet_read_batch_size
        self.parquet_memory_trim_every = parquet_memory_trim_every
        self.parquet_prefetch_ranges = parquet_prefetch_ranges
        self.parquet_range_size_mib = parquet_range_size_mib

    def _resume_seed_salt(self) -> int:
        """Salt the stream seed with the resumed global step so a restarted run does not
        replay the identical shard order + shuffle from the head of the stream (over many
        Colab preemptions that systematically oversamples early rows). Same salt for every
        worker, preserving the disjoint file sharding; 0 for a fresh run, deterministic
        given the checkpoint."""
        try:
            if self.trainer is not None:
                return int(self.trainer.global_step)
        except RuntimeError:
            pass  # not attached to a trainer (direct/offline use)
        return 0

    def _trainer_rank_world(self) -> tuple[int | None, int | None]:
        """The Trainer's DDP rank/world, resolved HERE (main process, trainer attached)
        and passed to the dataset explicitly — spawned DataLoader workers have no
        initialized process group to ask."""
        try:
            if self.trainer is not None:
                return int(self.trainer.global_rank), int(self.trainer.world_size)
        except RuntimeError:
            pass  # not attached to a trainer (direct/offline use)
        return None, None

    def _dataset(self, *, seed_salt: int = 0):
        if self.monet_image_direct:
            rank, world = self._trainer_rank_world()
            dataset = MonetImageStreamingDataset(
                paths=self.cache_wds,
                hf_token=self.hf_token,
                resolution=self.monet_image_resolution,
                shuffle_buffer=self.shuffle_buffer,
                seed=self.seed + seed_salt,
                caption_keys=self.monet_caption_keys,
                caption_probs=self.monet_caption_probs,
                glob_weights=self.monet_weights,
                dino_key=self.monet_dino_key,
                min_short_side=self.monet_image_min_short_side,
                min_aesthetic=self.monet_min_aesthetic,
                max_nsfw=self.monet_max_nsfw,
                max_watermark=self.monet_max_watermark,
                stream_block_size=self.stream_block_size,
                max_screened_per_shard_per_source=self.max_screened_per_shard,
                rank=rank,
                world_size=world,
            )
            dataset.prepare_shards()
            return dataset
        if self.t2i_direct:
            rank, world = self._trainer_rank_world()
            return FineT2IStreamingDataset(
                repo_id=self.t2i_repo,
                config_name=self.t2i_config,
                data_dir=self.t2i_data_dir,
                data_dir_weights=self.t2i_weights,
                split=self.t2i_split,
                hf_token=self.hf_token,
                resolution=self.t2i_resolution,
                shuffle_buffer=self.shuffle_buffer,
                seed=self.seed + seed_salt,
                caption_mode=self.t2i_caption_mode,
                short_prob=self.t2i_short_prob,
                min_short_side=self.t2i_min_short_side,
                min_aesthetic=self.t2i_min_aesthetic,
                stream_block_size=self.stream_block_size,
                max_screened_per_shard_per_source=self.max_screened_per_shard,
                rank=rank,
                world_size=world,
            )
        if self.edit_direct:
            rank, world = self._trainer_rank_world()
            return EditPairsStreamingDataset(
                paths=self.cache_wds,
                hf_token=self.hf_token,
                shuffle_buffer=self.shuffle_buffer,
                seed=self.seed + seed_salt,
                download_max_retries=self.download_max_retries,
                read_batch_size=self.parquet_read_batch_size,
                memory_trim_every=self.parquet_memory_trim_every,
                prefetch_ranges=self.parquet_prefetch_ranges,
                range_size_mib=self.parquet_range_size_mib,
                resolution=self.edit_resolution,
                min_o_score=self.edit_min_o_score,
                tasks=self.edit_tasks,
                prompt_pick=self.edit_prompt_pick,
                min_short_side=self.edit_min_short_side,
                max_aspect_mismatch=self.edit_max_aspect_mismatch,
                glob_weights=self.monet_weights,
                rank=rank,
                world_size=world,
            )
        if self.monet_direct:
            rank, world = self._trainer_rank_world()
            return MonetDirectStreamingDataset(
                paths=self.cache_wds,
                hf_token=self.hf_token,
                shuffle_buffer=self.shuffle_buffer,
                seed=self.seed + seed_salt,
                download_max_retries=self.download_max_retries,
                read_batch_size=self.parquet_read_batch_size,
                memory_trim_every=self.parquet_memory_trim_every,
                prefetch_ranges=self.parquet_prefetch_ranges,
                range_size_mib=self.parquet_range_size_mib,
                latent_key=self.monet_latent_key,
                caption_keys=self.monet_caption_keys,
                caption_probs=self.monet_caption_probs,
                glob_weights=self.monet_weights,
                dino_key=self.monet_dino_key,
                min_aesthetic=self.monet_min_aesthetic,
                max_nsfw=self.monet_max_nsfw,
                max_watermark=self.monet_max_watermark,
                rank=rank,
                world_size=world,
            )
        return make_wds_dataset(
            cache_wds=self.cache_wds,
            local_dir=self.local_dir,
            prefetch_shards=self.prefetch_shards,
            download_workers=self.download_workers,
            shuffle_buffer=self.shuffle_buffer,
            hf_token=self.hf_token,
            seed=self.seed + seed_salt,
            decode_queue_size=self.decode_queue_size,
            shard_shuffle=self.shard_shuffle,
            download_max_retries=self.download_max_retries,
            max_consecutive_skips=self.max_consecutive_skips,
            parquet_read_batch_size=self.parquet_read_batch_size,
            parquet_memory_trim_every=self.parquet_memory_trim_every,
            parquet_prefetch_ranges=self.parquet_prefetch_ranges,
            parquet_range_size_mib=self.parquet_range_size_mib,
            captions_only=self.captions_only,
        )

    def train_dataloader(self) -> DataLoader:
        # The rolling remote TAR cache REQUIRES num_workers=0 (single-process shard
        # download). Parquet range-streaming and local globs use multiple workers
        # (datasets shards parquet files across workers automatically).
        parquet = (
            self.monet_direct or self.monet_image_direct or self.edit_direct or self.t2i_direct
            or is_parquet_cache(self.cache_wds)
        )
        if parquet:
            workers = self.num_workers
        elif is_remote_cache(self.cache_wds):
            workers = 0
        else:
            workers = self.num_workers

        # DataLoader spawn workers inherit the environment present when iteration
        # begins. Parquet wants cancellable HTTP Range reads. Xet's reconstruction
        # path has an 8 GiB default buffer limit per process (64 GiB in HP mode), so
        # six workers can consume tens of GiB and wait minutes before yielding row 1.
        # Set both flags before spawn/import so Hugging Face actually selects HTTP.
        if parquet:
            if os.environ.get("HF_HUB_DISABLE_XET") != "1":
                print(
                    "[parquet-wds] disabling Xet reconstruction; using HTTP Range reads",
                    flush=True,
                )
            os.environ["HF_HUB_DISABLE_XET"] = "1"
            os.environ["HF_XET_HIGH_PERFORMANCE"] = "0"
            # Applied when spawn starts fresh worker interpreters. This bounds glibc's
            # per-thread arenas, which otherwise retain large Arrow/Python allocations.
            os.environ.setdefault("MALLOC_ARENA_MAX", "2")
        seed_salt = self._resume_seed_salt()
        if seed_salt:
            print(
                f"[data] resume detected (global_step={seed_salt}); salting stream seed "
                f"{self.seed} -> {self.seed + seed_salt} so the restarted stream does not "
                "replay the same head-of-stream rows",
                flush=True,
            )
        persistent = self.persistent_workers if self.persistent_workers is not None else workers > 0
        kwargs = {}
        if workers > 0:
            # Each worker stages prefetch_factor batches ahead so the pinned-memory +
            # non_blocking H2D copy always has something ready, overlapping decode with compute.
            kwargs["prefetch_factor"] = self.prefetch_factor
            if parquet:
                # FORK-SAFETY: the parquet loader touches HF (HfFileSystem glob / datasets
                # streaming) which leaves live fsspec/`requests` session + connection-pool
                # locks in this process. Default fork() workers inherit that half-initialized
                # state and DEADLOCK on their first range read (silent hang). `spawn` starts
                # each worker as a fresh interpreter that builds its own HF session, so there
                # is nothing poisoned to inherit. The dataset pickles cleanly (manifest dict +
                # str globs + token), and env (HF_TOKEN) propagates to spawned children.
                import multiprocessing as _mp

                kwargs["multiprocessing_context"] = _mp.get_context("spawn")
        return DataLoader(
            self._dataset(seed_salt=seed_salt),
            batch_size=self.batch_size,
            num_workers=workers,
            collate_fn=(
                collate_monet_image_records if self.monet_image_direct
                else collate_t2i_records if self.t2i_direct
                else collate_edit_records if self.edit_direct
                else collate_monet_records if self.monet_direct
                else collate_latent_records
            ),
            pin_memory=self.pin_memory,
            drop_last=True,
            persistent_workers=persistent and workers > 0,
            **kwargs,
        )


def peek_record(args: argparse.Namespace, hf_token: str | bool | None) -> Any:
    """Pull one cached record to read latent/text shapes (downloads one shard if remote)."""
    ds = make_wds_dataset(
        cache_wds=args.cache_wds,
        local_dir=args.wds_local_dir,
        prefetch_shards=1,
        download_workers=1,
        shuffle_buffer=1,
        hf_token=hf_token,
        seed=args.seed,
        decode_queue_size=0,  # one record: decode inline, no producer thread
        # Peek must mirror training decode: with --cache-captions-only the record's
        # text_embedding is None and build_config takes text dims from the encoder.
        captions_only=args.cache_captions_only,
    )
    return next(iter(ds))
