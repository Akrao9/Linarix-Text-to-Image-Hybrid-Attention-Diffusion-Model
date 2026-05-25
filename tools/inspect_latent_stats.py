"""Inspect global statistics for cached Boomer image latents."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterable

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from boomer.datasets.latent_shards import LatentCacheDataset, LatentRecord  # noqa: E402
from boomer.datasets.wds_latents import (  # noqa: E402
    RollingStreamingWdsLatentCacheDataset,
    StreamingWdsLatentCacheDataset,
    resolve_wds_urls,
)
from boomer.sana_latent_cache import hf_token_value  # noqa: E402


def build_wds_dataset(
    *,
    cache_wds: str,
    hf_token: str | bool | None,
    wds_local_dir: str | None,
    wds_rolling_shards: int,
):
    if wds_local_dir and wds_rolling_shards:
        print(
            f"Using rolling local WDS: dir={wds_local_dir} "
            f"window={wds_rolling_shards} remote={cache_wds}"
        )
        return RollingStreamingWdsLatentCacheDataset(
            cache_wds,
            local_dir=wds_local_dir,
            rolling_shards=wds_rolling_shards,
            hf_token=hf_token,
        )

    if cache_wds.startswith("hf://"):
        print(f"Resolving HF shard list for {cache_wds} ...", flush=True)
        started = time.time()
        urls = resolve_wds_urls(cache_wds, hf_token=hf_token)
        print(f"Resolved {len(urls)} shard URLs in {time.time() - started:.1f}s", flush=True)
        if not urls:
            raise FileNotFoundError(f"No shards found for {cache_wds}")

    return StreamingWdsLatentCacheDataset(cache_wds, hf_token=hf_token)


def iter_latent_records(
    *,
    cache: str | None,
    cache_wds: str | None,
    hf_token: str | bool | None,
    max_records: int,
    wds_local_dir: str | None,
    wds_rolling_shards: int,
) -> tuple[Iterable[LatentRecord], int | None]:
    if cache_wds:
        dataset = build_wds_dataset(
            cache_wds=cache_wds,
            hf_token=hf_token,
            wds_local_dir=wds_local_dir,
            wds_rolling_shards=wds_rolling_shards,
        )

        def records() -> Iterable[LatentRecord]:
            for index, record in enumerate(dataset):
                if max_records and index >= max_records:
                    break
                yield record

        return records(), None

    assert cache is not None
    dataset = LatentCacheDataset(cache)
    limit = min(len(dataset), max_records) if max_records else len(dataset)
    return (dataset[index] for index in range(limit)), len(dataset)


def compute_latent_stats(
    records: Iterable[LatentRecord],
    *,
    total_records: int | None,
    target_records: int,
    log_every: int,
) -> dict[str, object]:
    total_sum = torch.tensor(0.0, dtype=torch.float64)
    total_squares = torch.tensor(0.0, dtype=torch.float64)
    total_abs = torch.tensor(0.0, dtype=torch.float64)
    total_count = 0
    global_min = float("inf")
    global_max = float("-inf")
    channel_sum = None
    channel_squares = None
    channel_count = 0
    seen = 0
    started = time.time()
    goal = target_records or total_records

    for record in records:
        latent = record.image_latent.float()
        flat = latent.reshape(-1).double()
        total_sum += flat.sum()
        total_squares += (flat * flat).sum()
        total_abs += flat.abs().sum()
        total_count += flat.numel()
        global_min = min(global_min, float(flat.min().item()))
        global_max = max(global_max, float(flat.max().item()))

        per_channel = latent.reshape(latent.shape[0], -1).double()
        if channel_sum is None:
            channel_sum = torch.zeros(per_channel.shape[0], dtype=torch.float64)
            channel_squares = torch.zeros(per_channel.shape[0], dtype=torch.float64)
        channel_sum += per_channel.sum(dim=1)
        channel_squares += (per_channel * per_channel).sum(dim=1)
        channel_count += per_channel.shape[1]
        seen += 1

        if log_every and (seen == 1 or seen % log_every == 0):
            elapsed = max(time.time() - started, 1e-6)
            rate = seen / elapsed
            if goal:
                remaining = max(goal - seen, 0)
                eta_sec = remaining / rate if rate > 0 else 0.0
                print(
                    f"[stats] records={seen:,}/{goal:,} "
                    f"rate={rate:.1f}/s elapsed={elapsed / 60:.1f}m eta={eta_sec / 60:.1f}m",
                    flush=True,
                )
            else:
                print(
                    f"[stats] records={seen:,} rate={rate:.1f}/s elapsed={elapsed / 60:.1f}m",
                    flush=True,
                )

    if total_count == 0:
        raise ValueError("No latent values found in cache.")

    mean = total_sum / total_count
    variance = (total_squares / total_count) - mean * mean
    std = torch.sqrt(variance.clamp_min(0.0))
    rms = torch.sqrt(total_squares / total_count)
    abs_mean = total_abs / total_count

    assert channel_sum is not None and channel_squares is not None
    channel_mean = channel_sum / channel_count
    channel_var = (channel_squares / channel_count) - channel_mean * channel_mean
    channel_std = torch.sqrt(channel_var.clamp_min(0.0))

    return {
        "records": seen,
        "total_records": total_records if total_records is not None else seen,
        "elements": total_count,
        "global_mean": mean.item(),
        "global_std": std.item(),
        "global_rms": rms.item(),
        "global_abs_mean": abs_mean.item(),
        "global_min": global_min,
        "global_max": global_max,
        "channel_mean_abs_max": channel_mean.abs().max().item(),
        "channel_std_mean": channel_std.mean().item(),
        "channel_std_min": channel_std.min().item(),
        "channel_std_max": channel_std.max().item(),
        "channel_mean": [float(value) for value in channel_mean.tolist()],
        "channel_std": [float(value) for value in channel_std.tolist()],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute mean/std statistics for cached image latents.")
    parser.add_argument("cache", nargs="?", help="Cache directory or .pt file.")
    parser.add_argument(
        "--cache-wds",
        help="WebDataset latent cache shards, local glob or hf://datasets/.../train/*.tar.",
    )
    parser.add_argument(
        "--wds-local-dir",
        help="Local rolling shard directory. Use with --wds-rolling-shards for faster stats.",
    )
    parser.add_argument(
        "--wds-rolling-shards",
        type=int,
        default=0,
        help="Rolling local shard window size for --cache-wds hf:// stats/training.",
    )
    parser.add_argument("--max-records", type=int, default=0, help="Optional cap for a quick estimate.")
    parser.add_argument(
        "--log-every",
        type=int,
        default=1000,
        help="Print progress every N records. 0 disables progress logging.",
    )
    parser.add_argument("--json-out", help="Optional path to write stats as JSON.")
    parser.add_argument("--hf-token", default=None, help="HF token for private WDS datasets.")
    args = parser.parse_args()
    if not args.cache and not args.cache_wds:
        parser.error("provide --cache-wds or a cache directory/.pt path.")
    if args.cache and args.cache_wds:
        parser.error("provide only one of cache path and --cache-wds.")
    if args.wds_rolling_shards and not args.wds_local_dir:
        parser.error("--wds-rolling-shards requires --wds-local-dir.")
    if args.wds_local_dir and not args.wds_rolling_shards:
        parser.error("--wds-local-dir requires --wds-rolling-shards.")
    if args.wds_rolling_shards and not str(args.cache_wds or "").startswith("hf://"):
        parser.error("rolling local stats require an hf://datasets/... --cache-wds glob.")
    if args.log_every < 0:
        parser.error("--log-every must be non-negative.")

    target_records = args.max_records
    if args.cache_wds and target_records == 0:
        target_records = 10000
        print(
            "warning: --cache-wds without --max-records defaults to 10,000 samples. "
            "Pass --max-records 100000 explicitly for the full estimate."
        )

    print(
        f"inspect_latent_stats: target_records={target_records:,} "
        f"source={args.cache_wds or args.cache}",
        flush=True,
    )
    records, total_records = iter_latent_records(
        cache=args.cache,
        cache_wds=args.cache_wds,
        hf_token=hf_token_value(args.hf_token),
        max_records=target_records if target_records > 0 else 0,
        wds_local_dir=args.wds_local_dir,
        wds_rolling_shards=args.wds_rolling_shards,
    )
    print("Streaming samples and accumulating stats ...", flush=True)
    stats = compute_latent_stats(
        records,
        total_records=total_records,
        target_records=target_records,
        log_every=args.log_every,
    )

    print(f"records={stats['records']} total_records={stats['total_records']} elements={stats['elements']}")
    print(f"global_mean={stats['global_mean']:.8f}")
    print(f"global_std={stats['global_std']:.8f}")
    print(f"global_rms={stats['global_rms']:.8f}")
    print(f"global_abs_mean={stats['global_abs_mean']:.8f}")
    print(f"global_min={stats['global_min']:.8f}")
    print(f"global_max={stats['global_max']:.8f}")
    print(f"channel_mean_abs_max={stats['channel_mean_abs_max']:.8f}")
    print(f"channel_std_mean={stats['channel_std_mean']:.8f}")
    print(f"channel_std_min={stats['channel_std_min']:.8f}")
    print(f"channel_std_max={stats['channel_std_max']:.8f}")
    print("recommended_training_flags:")
    print(f"  --latent-mean {stats['global_mean']:.8f} --latent-std {stats['global_std']:.8f}")
    print("recommended_channelwise_training_flags:")
    print("  --latent-stats-json STATS_JSON")

    if args.json_out:
        json_path = Path(args.json_out)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        print(f"saved_json={json_path}")


if __name__ == "__main__":
    main()
