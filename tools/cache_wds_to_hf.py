"""Cache a pre-sharded HF WebDataset (e.g. Spawning/pd12m-full) into Boomer Qwen shards.

pd12m-full already ships images as WebDataset .tar shards (jpg + txt + json), so there
is **no download stage** — we encode directly. This orchestrator processes the source
shards in groups (to amortize Qwen/VAE load), streams each group's cache to an HF
dataset repo, deletes local scratch, and resumes statelessly by checking which groups
already landed in the target repo (survives Colab restarts).

Source: CDLA-Permissive-2.0 (commercial use, attribution required) — keep an ATTRIBUTIONS
note crediting Spawning/pd12m-full.

Example:
    python3 tools/cache_wds_to_hf.py \
      --source-repo Spawning/pd12m-full \
      --hf-repo-id akrao9/pd12m-boomer-qwen-cache \
      --shards-per-group 16 --max-length 256 --text-storage-dtype fp8 --hf-token true
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def hf_token(value: str | None) -> str | bool | None:
    if not value:
        return None
    if value.lower() in {"1", "true", "yes"}:
        return os.environ.get("HF_TOKEN") or True
    return value


def list_source_shards(repo_id: str, token: str | bool | None, suffix: str) -> list[str]:
    from huggingface_hub import HfApi

    api = HfApi(token=token if isinstance(token, str) else None)
    files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    shards = sorted(f for f in files if f.endswith(suffix))
    if not shards:
        raise FileNotFoundError(f"No '{suffix}' shards in {repo_id}")
    return shards


def _repo_files(repo_id: str, token: str | bool | None) -> list[str]:
    from huggingface_hub import HfApi
    from huggingface_hub.utils import RepositoryNotFoundError

    api = HfApi(token=token if isinstance(token, str) else None)
    try:
        return api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    except (RepositoryNotFoundError, Exception):  # noqa: BLE001 - repo may not exist yet
        return []


def discover_completed(files: list[str]) -> set[str]:
    """Group prefixes already present in the target repo (idempotent resume)."""
    return {f.split("/", 1)[0] for f in files if "/manifest" in f and f.startswith("part_")}


def ensure_root_manifest(repo_id: str, token: str | bool | None, prefix: str) -> None:
    """Copy a group's manifest to the repo root so the rolling loader can resolve it.

    The loader's first manifest candidate for a 'part_*/*.tar' glob is
    'hf://datasets/<repo>/manifest.json'. All groups share schema/dtype/shape fields,
    so any group's manifest is a valid root manifest (per-group counts are ignored).
    """
    from huggingface_hub import HfApi, hf_hub_download

    local = hf_hub_download(repo_id, f"{prefix}/manifest.json", repo_type="dataset", token=token)
    HfApi(token=token if isinstance(token, str) else None).upload_file(
        path_or_fileobj=local,
        path_in_repo="manifest.json",
        repo_id=repo_id,
        repo_type="dataset",
    )
    print(f"wrote root manifest.json (from {prefix}) -> rolling loader can now resolve dtypes/shapes")


def run_cacher(args: argparse.Namespace, list_file: Path, cache_dir: Path, prefix: str, create_repo: bool) -> None:
    cacher = Path(__file__).resolve().parent / "cache_hf_wds_to_boomer_wds.py"
    cmd = [
        sys.executable, str(cacher),
        "--source-list-file", str(list_file),
        "--source-name", args.source_repo,
        "--text-encoder-kind", "qwen",
        "--text-encoder-id", args.text_encoder_id,
        "--vae-id", args.vae_id,
        "--resolution", str(args.resolution),
        "--max-length", str(args.max_length),
        "--dtype", args.dtype,
        "--latent-storage-dtype", args.latent_storage_dtype,
        "--text-storage-dtype", args.text_storage_dtype,
        "--batch-size", str(args.batch_size),
        "--image-encode-batch", str(args.image_encode_batch),
        "--text-encode-batch", str(args.text_encode_batch),
        "--num-workers", str(args.num_workers),
        "--prefetch-factor", str(args.prefetch_factor),
        "--source-chunk-size", str(args.source_chunk_size),
        "--source-read-queue", str(args.source_read_queue),
        "--async-flush-depth", str(args.async_flush_depth),
        "--max-samples", str(args.max_samples),
        "--shard-size-gb", str(args.shard_size_gb),
        "--shard-start-index", str(args.shard_start_index),
        "--log-every", str(args.log_every),
        "--output-dir", str(cache_dir),
        "--hf-repo-id", args.hf_repo_id,
        "--hf-path-prefix", prefix,
        "--hf-upload-workers", str(args.hf_upload_workers),
        "--delete-uploaded",
        "--hf-token", args.hf_token or "true",
    ]
    if args.gpu_decode:
        cmd.append("--gpu-decode")
    if args.dali_decode:
        cmd.append("--dali-decode")
        cmd.extend(["--dali-num-threads", str(args.dali_num_threads)])
        cmd.extend(["--jpeg-decode-batch", str(args.dali_decode_batch)])
        cmd.extend(["--dali-device-memory-padding", str(args.dali_device_memory_padding)])
        cmd.extend(["--dali-host-memory-padding", str(args.dali_host_memory_padding)])
    if args.channels_last:
        cmd.append("--channels-last")
    if args.text_attn_implementation:
        cmd.extend(["--text-attn-implementation", args.text_attn_implementation])
    if args.compile:
        cmd.append("--compile")
        cmd.extend(["--compile-mode", args.compile_mode])
    if args.pin_memory:
        cmd.append("--pin-memory")
    if args.async_flush:
        cmd.append("--async-flush")
    if args.async_cpu_transfer:
        cmd.append("--async-cpu-transfer")
    if args.rolling_localize_source_shards:
        cmd.append("--wait-local-source-shards")
        if not args.keep_localized_source_shards:
            cmd.append("--delete-local-source-shards-after-read")
    if not args.select_source_files:
        cmd.append("--no-select-source-files")
    if args.profile_stages:
        cmd.append("--profile-stages")
    if create_repo:
        cmd.append("--create-repo")
    print(f"[{prefix}] caching {args.shards_per_group} shard(s) -> hf://datasets/{args.hf_repo_id}/{prefix}")
    subprocess.run(cmd, check=True)


def localize_source_shards(
    repo_id: str,
    shards: list[str],
    token: str | bool | None,
    local_dir: Path,
) -> list[str]:
    """Download source shards through huggingface_hub/hf_xet and return disposable local paths."""
    from huggingface_hub import hf_hub_download

    local_dir.mkdir(parents=True, exist_ok=True)
    return [
        hf_hub_download(
            repo_id,
            shard,
            repo_type="dataset",
            token=token,
            local_dir=str(local_dir),
        )
        for shard in shards
    ]


def rolling_localized_source_commands(
    shards: list[str],
    source_dir: Path,
    *,
    delete_after_read: bool,
) -> list[str]:
    """Return local paths consumed directly by WebDataset as they become ready."""
    return [str(source_dir / shard) for shard in shards]


def start_rolling_localizer(
    args: argparse.Namespace,
    shards: list[str],
    token: str | bool | None,
    source_dir: Path,
    shard_list_file: Path,
) -> subprocess.Popen:
    """Start a sidecar that keeps a small rolling window of HF shards localized."""
    helper = Path(__file__).resolve().parent / "hf_rolling_localize.py"
    shard_list_file.write_text("\n".join(shards) + "\n", encoding="utf-8")
    source_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(helper),
        "prefetch",
        "--repo-id",
        args.source_repo,
        "--source-list-file",
        str(shard_list_file),
        "--local-dir",
        str(source_dir),
        "--window",
        str(args.rolling_localize_window),
        "--download-workers",
        str(args.rolling_localize_download_workers),
        "--token-env",
        "HF_TOKEN",
    ]
    env = os.environ.copy()
    if isinstance(token, str):
        env["HF_TOKEN"] = token
    print(
        f"[rolling-localize] prefetch window={args.rolling_localize_window} "
        f"workers={args.rolling_localize_download_workers} dir={source_dir}"
    )
    return subprocess.Popen(cmd, env=env)


def stop_rolling_localizer(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def main() -> None:
    p = argparse.ArgumentParser(description="Cache a pre-sharded HF WebDataset into Boomer Qwen shards (resumable).")
    p.add_argument("--source-repo", default="Spawning/pd12m-full")
    p.add_argument("--source-suffix", default=".tar", help="Shard file suffix to match.")
    p.add_argument("--hf-repo-id", required=True, help="Target HF dataset repo for the cache.")
    p.add_argument("--shards-per-group", type=int, default=16, help="Source shards per cacher call (amortizes model load). Keep CONSTANT across resumes.")
    p.add_argument(
        "--no-groups",
        action="store_true",
        help="Run all source shards in one cacher invocation. Output still lands under part_00000.",
    )
    p.add_argument(
        "--localize-source-shards",
        action="store_true",
        help="Download each source group to the HF local cache first via hf_hub_download/hf_xet, then cache from local tar files.",
    )
    p.add_argument(
        "--rolling-localize-source-shards",
        action="store_true",
        help="Keep a rolling local window of source shards while one long cacher process reads them.",
    )
    p.add_argument(
        "--rolling-localize-window",
        type=int,
        default=4,
        help="Number of source tar shards allowed ready/on disk in rolling-localize mode.",
    )
    p.add_argument(
        "--rolling-localize-download-workers",
        type=int,
        default=3,
        help="Concurrent hf_hub_download calls in rolling-localize prefetch sidecar.",
    )
    p.add_argument(
        "--rolling-localize-dir",
        default=None,
        help="Directory for rolling source tar cache. Defaults to <scratch>/<part>_source.",
    )
    p.add_argument(
        "--keep-localized-source-shards",
        action="store_true",
        help="Do not delete rolling/localized source tars after reading. Requires much more disk.",
    )
    p.add_argument("--max-groups", type=int, default=None, help="Process at most this many new groups (pilot).")
    p.add_argument("--scratch-dir", default="/content/_wds_cache_scratch")
    # forwarded to the cacher
    p.add_argument("--text-encoder-id", default="Qwen/Qwen3.5-2B")
    p.add_argument("--vae-id", default="mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers")
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument(
        "--text-attn-implementation",
        default=None,
        choices=["eager", "sdpa", "flash_attention_2", "flash_attention_3"],
    )
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--latent-storage-dtype", default="bf16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--text-storage-dtype", default="fp8")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--image-encode-batch", type=int, default=128)
    p.add_argument("--text-encode-batch", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--prefetch-factor", type=int, default=2)
    p.add_argument("--source-chunk-size", type=int, default=64)
    p.add_argument("--source-read-queue", type=int, default=2048,
                   help="Samples buffered between the background shard reader and encode loop.")
    p.add_argument("--async-flush", action="store_true")
    p.add_argument("--async-flush-depth", type=int, default=2)
    p.add_argument("--async-cpu-transfer", action="store_true")
    p.add_argument("--select-source-files", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--profile-stages", action="store_true")
    p.add_argument("--max-samples", type=int, default=1_000_000_000)
    p.add_argument("--shard-size-gb", type=float, default=2.0)
    p.add_argument("--log-every", type=int, default=1000)
    p.add_argument("--gpu-decode", action="store_true", default=True)
    p.add_argument("--no-gpu-decode", dest="gpu_decode", action="store_false")
    p.add_argument("--dali-decode", action="store_true")
    p.add_argument("--dali-num-threads", type=int, default=8)
    p.add_argument("--dali-decode-batch", "--jpeg-decode-batch", dest="dali_decode_batch", type=int, default=64)
    p.add_argument("--dali-device-memory-padding", type=int, default=67_108_864)
    p.add_argument("--dali-host-memory-padding", type=int, default=16_777_216)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--compile-mode", default="default", choices=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"])
    p.add_argument("--channels-last", action="store_true", default=True)
    p.add_argument("--pin-memory", action="store_true")
    p.add_argument("--hf-upload-workers", type=int, default=2)
    p.add_argument("--hf-token", help="HF token, or 'true' for the logged-in token.")
    p.add_argument("--write-root-manifest", action="store_true",
                   help="Only copy an existing group's manifest to the repo root (fixes a repo for train.py), then exit.")
    p.add_argument("--world-size", type=int, default=1,
                   help="Total parallel sessions. Each takes a disjoint stride slice of source shards.")
    p.add_argument("--rank", type=int, default=0,
                   help="This session's index in [0, world-size). Writes to its own part_<rank> prefix.")
    p.add_argument("--skip-source-shards", type=int, default=0,
                   help="Resume: drop the first N shards of THIS session's slice (already processed).")
    p.add_argument("--shard-start-index", type=int, default=0,
                   help="Resume: number output cache shards starting here (avoid overwriting uploaded shards).")
    args = p.parse_args()

    token = hf_token(args.hf_token)
    if args.localize_source_shards and args.rolling_localize_source_shards:
        sys.exit("Use either --localize-source-shards or --rolling-localize-source-shards, not both.")
    if args.rolling_localize_window <= 0:
        sys.exit("--rolling-localize-window must be positive.")
    if args.rolling_localize_download_workers <= 0:
        sys.exit("--rolling-localize-download-workers must be positive.")

    if args.write_root_manifest:
        files = _repo_files(args.hf_repo_id, token)
        parts = sorted(discover_completed(files))
        if not parts:
            sys.exit(f"No part_* groups with a manifest found in {args.hf_repo_id}.")
        ensure_root_manifest(args.hf_repo_id, token, parts[0])
        return

    if args.world_size < 1:
        sys.exit("--world-size must be >= 1.")
    if not (0 <= args.rank < args.world_size):
        sys.exit(f"--rank must be in [0, {args.world_size}); got {args.rank}.")

    shards = list_source_shards(args.source_repo, token, args.source_suffix)
    if args.world_size > 1:
        total = len(shards)
        shards = shards[args.rank :: args.world_size]  # stride slice = balanced across shard sizes
        print(f"[rank {args.rank}/{args.world_size}] this session handles {len(shards)}/{total} source shards "
              f"-> prefix part_{args.rank:05d}")
    if args.skip_source_shards > 0:
        before = len(shards)
        shards = shards[args.skip_source_shards :]
        print(f"resume: skipping first {args.skip_source_shards} shards of this slice "
              f"({before} -> {len(shards)} remaining); first source shard now {shards[0] if shards else 'NONE'}; "
              f"output cache shards numbered from {args.shard_start_index}.")
    if args.no_groups:
        args.shards_per_group = len(shards)
    groups = [shards[i : i + args.shards_per_group] for i in range(0, len(shards), args.shards_per_group)]
    repo_files = _repo_files(args.hf_repo_id, token)
    completed = discover_completed(repo_files)
    root_present = "manifest.json" in repo_files
    print(f"{len(shards)} source shards -> {len(groups)} groups ({args.shards_per_group}/group); {len(completed)} already in repo.")

    scratch = Path(args.scratch_dir)
    repo_seeded = bool(completed)
    processed = 0
    for gi, group in enumerate(groups):
        if args.max_groups and processed >= args.max_groups:
            print(f"reached --max-groups={args.max_groups}; stopping.")
            break
        if args.world_size > 1:
            # Each session owns a distinct prefix so parallel sessions never collide.
            prefix = f"part_{args.rank:05d}" if args.no_groups else f"part_r{args.rank:02d}_{gi:05d}"
        else:
            prefix = f"part_{gi:05d}"
        if prefix in completed:
            print(f"[{prefix}] already in repo — skip.")
            continue

        list_file = scratch / f"{prefix}.txt"
        cache_dir = scratch / prefix
        source_dir = Path(args.rolling_localize_dir) if args.rolling_localize_dir else scratch / f"{prefix}_source"
        source_index_file = scratch / f"{prefix}_source_shards.txt"
        prefetch_proc: subprocess.Popen | None = None
        scratch.mkdir(parents=True, exist_ok=True)
        print(f"\n=== group {gi + 1}/{len(groups)}: {prefix} ({len(group)} shards) ===")
        if args.rolling_localize_source_shards:
            source_paths = rolling_localized_source_commands(
                group,
                source_dir,
                delete_after_read=not args.keep_localized_source_shards,
            )
            prefetch_proc = start_rolling_localizer(args, group, token, source_dir, source_index_file)
        elif args.localize_source_shards:
            source_paths = localize_source_shards(args.source_repo, group, token, source_dir)
        else:
            source_paths = [f"hf://datasets/{args.source_repo}/{s}" for s in group]
        list_file.write_text(
            "\n".join(source_paths) + "\n", encoding="utf-8"
        )
        try:
            run_cacher(args, list_file, cache_dir, prefix, create_repo=not repo_seeded)
            if prefetch_proc is not None and prefetch_proc.poll() not in (None, 0):
                raise RuntimeError(f"rolling localizer exited with code {prefetch_proc.returncode}")
            repo_seeded = True
            processed += 1
        finally:
            stop_rolling_localizer(prefetch_proc)
            shutil.rmtree(cache_dir, ignore_errors=True)
            if not args.keep_localized_source_shards:
                shutil.rmtree(source_dir, ignore_errors=True)
            list_file.unlink(missing_ok=True)
            source_index_file.unlink(missing_ok=True)
        if not root_present:
            ensure_root_manifest(args.hf_repo_id, token, prefix)
            root_present = True
        print(f"[{prefix}] done; scratch cleared.")

    # Self-heal: if every group was already cached but the root manifest is missing
    # (e.g. a run that finished before this feature), write it so train.py can load.
    if not root_present and completed:
        ensure_root_manifest(args.hf_repo_id, token, sorted(completed)[0])
        root_present = True

    print(f"\nfinished: {processed} new group(s) -> hf://datasets/{args.hf_repo_id}")


if __name__ == "__main__":
    main()
