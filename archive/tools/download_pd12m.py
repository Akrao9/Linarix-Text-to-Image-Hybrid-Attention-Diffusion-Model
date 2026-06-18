"""Download PD12M (Spawning/PD12M) into WebDataset shards for Boomer caching.

PD12M ships **metadata only** (parquet with url/caption/source/license); the
images are URLs on S3. This driver:

  1. pulls the metadata parquet from the HF dataset repo,
  2. (optionally) takes a pilot subset,
  3. runs ``img2dataset`` to download + resize images into WebDataset shards,
     preserving ``id/source/license`` per image for the attribution trail,
  4. writes an aggregated ATTRIBUTIONS manifest (per source / per license).

The output WDS shards (``<id>.jpg`` + ``.txt`` caption + ``.json`` provenance)
feed ``tools/cache_hf_wds_to_boomer_wds.py --text-encoder-kind qwen``.

License: PD12M is CDLA-Permissive-2.0 (commercial use, attribution required) —
keep the emitted ATTRIBUTIONS file with any release.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ID = "Spawning/PD12M"
METADATA_PATTERN = "metadata/*.parquet"
PROVENANCE_COLUMNS = ["id", "source", "license"]


def download_metadata(repo_id: str, pattern: str, dest: Path, token: str | bool | None) -> list[Path]:
    from huggingface_hub import snapshot_download

    local = snapshot_download(
        repo_id,
        repo_type="dataset",
        allow_patterns=[pattern],
        local_dir=str(dest),
        token=token,
    )
    files = sorted(Path(local).glob(pattern))
    if not files:
        raise FileNotFoundError(f"No parquet matched {pattern!r} in {repo_id}")
    print(f"metadata: {len(files)} parquet file(s) under {Path(local) / pattern}")
    return files


def build_url_list(parquet_files: list[Path], limit: int | None, work_dir: Path) -> Path:
    """Return a directory of parquet for img2dataset (subset if --limit)."""
    if not limit:
        # Use the metadata directory as-is.
        return parquet_files[0].parent

    import pyarrow.parquet as pq
    import pyarrow as pa

    rows: list[pa.Table] = []
    remaining = limit
    for path in parquet_files:
        table = pq.read_table(path)
        take = min(remaining, table.num_rows)
        rows.append(table.slice(0, take))
        remaining -= take
        if remaining <= 0:
            break
    subset = pa.concat_tables(rows)
    work_dir.mkdir(parents=True, exist_ok=True)
    out = work_dir / "pd12m_subset.parquet"
    pq.write_table(subset, out)
    print(f"pilot subset: {subset.num_rows} rows -> {out}")
    return work_dir


def _img2dataset(args: argparse.Namespace, url_list: Path, output_folder: Path) -> None:
    try:
        from img2dataset import download
    except ImportError:
        sys.exit(
            "img2dataset is required. Install it:\n"
            "  pip install img2dataset\n"
            "then re-run (or use --skip-download to only build metadata/attribution)."
        )

    output_folder.mkdir(parents=True, exist_ok=True)
    download(
        url_list=str(url_list),
        input_format="parquet",
        url_col="url",
        caption_col="caption",
        save_additional_columns=PROVENANCE_COLUMNS,
        output_format="webdataset",
        output_folder=str(output_folder),
        image_size=args.image_size,
        resize_mode=args.resize_mode,
        resize_only_if_bigger=False,
        number_sample_per_shard=args.samples_per_shard,
        processes_count=args.processes,
        thread_count=args.threads,
        retries=args.retries,
        encode_format="jpg",
        encode_quality=args.jpg_quality,
        distributor="multiprocessing",
    )
    print(f"img2dataset done -> {output_folder}/*.tar")


def run_img2dataset(args: argparse.Namespace, url_list_dir: Path) -> None:
    _img2dataset(args, url_list_dir, Path(args.output_folder))


def _maybe_subset(pq_file: Path, row_limit: int | None, dest_dir: Path) -> Path:
    """Return a row-limited copy of a parquet file (for fast test chunks), else the file itself."""
    if not row_limit:
        return pq_file
    import pyarrow.parquet as pq

    table = pq.read_table(pq_file).slice(0, row_limit)
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / f"{pq_file.stem}_subset.parquet"
    pq.write_table(table, out)
    print(f"[chunk {pq_file.stem}] test subset: {table.num_rows} rows")
    return out


def discover_completed_chunks(repo_id: str, token: str | bool | None) -> set[str]:
    """Chunk prefixes already present in the HF repo (idempotent resume)."""
    from huggingface_hub import HfApi
    from huggingface_hub.utils import RepositoryNotFoundError

    api = HfApi(token=token if isinstance(token, str) else None)
    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    except (RepositoryNotFoundError, Exception):  # noqa: BLE001 - repo may not exist yet
        return set()
    # A chunk/group is "done" once its manifest landed under <prefix>/.
    return {f.split("/", 1)[0] for f in files if "/manifest" in f}


def run_cacher_subprocess(
    args: argparse.Namespace, raw_dir: Path, cache_dir: Path, prefix: str, create_repo: bool
) -> None:
    """Encode one chunk's raw shards and stream the cache to HF (subprocess for CUDA isolation)."""
    cacher = Path(__file__).resolve().parent / "cache_hf_wds_to_boomer_wds.py"
    cmd = [
        sys.executable,
        str(cacher),
        "--source-glob",
        f"{raw_dir}/*.tar",
        "--text-encoder-kind",
        "qwen",
        "--text-encoder-id",
        args.text_encoder_id,
        "--vae-id",
        args.vae_id,
        "--resolution",
        str(args.image_size),
        "--max-length",
        str(args.max_length),
        "--dtype",
        args.dtype,
        "--latent-storage-dtype",
        args.latent_storage_dtype,
        "--text-storage-dtype",
        args.text_storage_dtype,
        "--batch-size",
        str(args.batch_size),
        "--image-encode-batch",
        str(args.image_encode_batch),
        "--text-encode-batch",
        str(args.text_encode_batch),
        "--max-samples",
        str(args.cache_max_samples),
        "--shard-size-gb",
        str(args.shard_size_gb),
        "--output-dir",
        str(cache_dir),
        "--hf-repo-id",
        args.hf_repo_id,
        "--hf-path-prefix",
        prefix,
        "--delete-uploaded",
        "--hf-token",
        args.hf_token or "true",
    ]
    if args.gpu_decode:
        cmd.append("--gpu-decode")
    if args.channels_last:
        cmd.append("--channels-last")
    if create_repo:
        cmd.append("--create-repo")
    print(f"[chunk {prefix}] caching -> hf://datasets/{args.hf_repo_id}/{prefix}")
    subprocess.run(cmd, check=True)


def run_cacher_subprocess_for_source_list(
    args: argparse.Namespace,
    source_paths: list[str],
    cache_dir: Path,
    prefix: str,
    create_repo: bool,
) -> None:
    """Encode an explicit list of WDS source shards and stream the cache to HF."""
    cacher = Path(__file__).resolve().parent / "cache_hf_wds_to_boomer_wds.py"
    cache_dir.mkdir(parents=True, exist_ok=True)
    source_list_path = cache_dir / "source_shards.txt"
    source_list_path.write_text("\n".join(source_paths) + "\n", encoding="utf-8")
    cmd = [
        sys.executable,
        str(cacher),
        "--source-list-file",
        str(source_list_path),
        "--text-encoder-kind",
        "qwen",
        "--text-encoder-id",
        args.text_encoder_id,
        "--vae-id",
        args.vae_id,
        "--resolution",
        str(args.image_size),
        "--max-length",
        str(args.max_length),
        "--dtype",
        args.dtype,
        "--latent-storage-dtype",
        args.latent_storage_dtype,
        "--text-storage-dtype",
        args.text_storage_dtype,
        "--batch-size",
        str(args.batch_size),
        "--image-encode-batch",
        str(args.image_encode_batch),
        "--text-encode-batch",
        str(args.text_encode_batch),
        "--max-samples",
        str(args.cache_max_samples),
        "--shard-size-gb",
        str(args.shard_size_gb),
        "--output-dir",
        str(cache_dir),
        "--hf-repo-id",
        args.hf_repo_id,
        "--hf-path-prefix",
        prefix,
        "--delete-uploaded",
        "--hf-token",
        args.hf_token or "true",
    ]
    if args.gpu_decode:
        cmd.append("--gpu-decode")
    if args.channels_last:
        cmd.append("--channels-last")
    if create_repo:
        cmd.append("--create-repo")
    print(f"[group {prefix}] caching {len(source_paths)} source tar(s) -> hf://datasets/{args.hf_repo_id}/{prefix}")
    subprocess.run(cmd, check=True)


def run_hf_wds_chunked(args: argparse.Namespace, token: str | bool | None) -> None:
    """Cache an already-downloaded HF WebDataset source (e.g. Spawning/pd12m-full) in resumable groups."""
    if not args.hf_repo_id:
        sys.exit("--source-wds-glob requires --hf-repo-id (groups stream to the HF dataset repo).")

    from huggingface_hub import HfFileSystem

    fs = HfFileSystem(token=token)
    matched_paths = sorted(str(path) for path in fs.glob(args.source_wds_glob))
    source_paths = []
    for path in matched_paths:
        file = fs.resolve_path(path)
        source_paths.append(f"hf://datasets/{file.repo_id}/{file.path_in_repo}")
    if not source_paths:
        raise FileNotFoundError(f"No source tar files matched: {args.source_wds_glob}")

    group_size = max(1, args.source_wds_group_size)
    groups = [source_paths[i : i + group_size] for i in range(0, len(source_paths), group_size)]
    completed = discover_completed_chunks(args.hf_repo_id, token)
    print(
        f"HF WDS chunked run: {len(source_paths)} source tar(s), "
        f"{len(groups)} group(s), {len(completed)} already in repo."
    )

    cache_root = Path(args.chunk_cache_dir)
    repo_seeded = bool(completed)
    processed = 0
    for group_index, group_paths in enumerate(groups):
        if args.max_chunks and processed >= args.max_chunks:
            print(f"reached --max-chunks={args.max_chunks}; stopping.")
            break
        prefix = f"{args.source_wds_prefix}_{group_index:05d}"
        if prefix in completed:
            print(f"[group {prefix}] already in repo - skip.")
            continue

        cache_dir = cache_root / prefix
        print(f"\n=== group {group_index + 1}/{len(groups)}: {prefix} ===")
        try:
            run_cacher_subprocess_for_source_list(
                args, group_paths, cache_dir, prefix, create_repo=not repo_seeded
            )
            repo_seeded = True
            processed += 1
        finally:
            shutil.rmtree(cache_dir, ignore_errors=True)
        print(f"[group {prefix}] done; local scratch cleared.")

    print(f"\nHF WDS chunked run finished: {processed} new group(s) -> hf://datasets/{args.hf_repo_id}")


def run_chunked(args: argparse.Namespace, parquet_files: list[Path], token: str | bool | None) -> None:
    if not args.hf_repo_id:
        sys.exit("--chunk requires --hf-repo-id (chunks stream to the HF dataset repo).")

    raw_root = Path(args.chunk_raw_dir)
    cache_root = Path(args.chunk_cache_dir)
    completed = discover_completed_chunks(args.hf_repo_id, token)
    print(f"chunked run: {len(parquet_files)} parquet files, {len(completed)} already in repo.")

    repo_seeded = bool(completed)  # only the first ever chunk creates the repo
    processed = 0
    for index, pq_file in enumerate(parquet_files):
        if args.max_chunks and processed >= args.max_chunks:
            print(f"reached --max-chunks={args.max_chunks}; stopping.")
            break
        prefix = f"chunk_{pq_file.stem}"
        if prefix in completed:
            print(f"[chunk {prefix}] already in repo — skip.")
            continue

        raw_dir = raw_root / pq_file.stem
        cache_dir = cache_root / pq_file.stem
        print(f"\n=== chunk {index + 1}/{len(parquet_files)}: {prefix} ===")
        url_list = _maybe_subset(pq_file, args.chunk_row_limit, raw_root / "_subset")
        try:
            _img2dataset(args, url_list, raw_dir)
            run_cacher_subprocess(args, raw_dir, cache_dir, prefix, create_repo=not repo_seeded)
            repo_seeded = True
            processed += 1
        finally:
            shutil.rmtree(raw_dir, ignore_errors=True)
            shutil.rmtree(cache_dir, ignore_errors=True)
        print(f"[chunk {prefix}] done; local scratch cleared.")

    print(f"\nchunked run finished: {processed} new chunk(s) -> hf://datasets/{args.hf_repo_id}")


def write_attribution(parquet_files: list[Path], limit: int | None, out_dir: Path) -> None:
    """Aggregate source/license counts into a provenance manifest."""
    import pyarrow.parquet as pq

    source_counts: dict[str, int] = {}
    license_counts: dict[str, int] = {}
    seen = 0
    for path in parquet_files:
        table = pq.read_table(path, columns=["source", "license"])
        sources = table.column("source").to_pylist()
        licenses = table.column("license").to_pylist()
        for src, lic in zip(sources, licenses):
            if limit and seen >= limit:
                break
            source_counts[str(src)] = source_counts.get(str(src), 0) + 1
            license_counts[str(lic)] = license_counts.get(str(lic), 0) + 1
            seen += 1
        if limit and seen >= limit:
            break

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset": REPO_ID,
        "dataset_license": "CDLA-Permissive-2.0",
        "total_images_counted": seen,
        "by_source": dict(sorted(source_counts.items(), key=lambda kv: -kv[1])),
        "by_license": dict(sorted(license_counts.items(), key=lambda kv: -kv[1])),
    }
    (out_dir / "pd12m_attribution.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    lines = [
        "# PD12M Attribution",
        "",
        f"Source dataset: **{REPO_ID}** (CDLA-Permissive-2.0).",
        f"Images counted: {seen:,}.",
        "",
        "## By source",
        "",
        "| Source | Images |",
        "|---|---:|",
    ]
    lines += [f"| {src} | {cnt:,} |" for src, cnt in manifest["by_source"].items()]
    lines += ["", "## By license", "", "| License | Images |", "|---|---:|"]
    lines += [f"| {lic} | {cnt:,} |" for lic, cnt in manifest["by_license"].items()]
    (out_dir / "ATTRIBUTIONS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"attribution: wrote {out_dir/'pd12m_attribution.json'} and {out_dir/'ATTRIBUTIONS.md'}")


def hf_token(value: str | None) -> str | bool | None:
    if not value:
        return None
    if value.lower() in {"1", "true", "yes"}:
        return os.environ.get("HF_TOKEN") or True
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Download PD12M into WebDataset shards for Boomer caching.")
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--metadata-pattern", default=METADATA_PATTERN)
    parser.add_argument("--metadata-dir", default="/data/pd12m_metadata", help="Where to store the parquet metadata.")
    parser.add_argument("--output-folder", default="/data/pd12m_wds", help="WebDataset shard output dir.")
    parser.add_argument("--limit", type=int, default=None, help="Pilot: only this many rows (e.g. 50000).")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--resize-mode", default="center_crop", choices=["border", "keep_ratio", "center_crop", "no"])
    parser.add_argument("--samples-per-shard", type=int, default=10000)
    parser.add_argument("--processes", type=int, default=16)
    parser.add_argument("--threads", type=int, default=64)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--jpg-quality", type=int, default=95)
    parser.add_argument("--skip-download", action="store_true", help="Build metadata + attribution only; no image download.")
    parser.add_argument("--skip-attribution", action="store_true")
    parser.add_argument("--hf-token", help="HF token, or 'true' to use the logged-in token.")

    # --- chunked orchestration (download one parquet -> cache+upload -> delete) ---
    chunk = parser.add_argument_group("chunked mode (Colab-friendly: bounded local disk)")
    chunk.add_argument("--chunk", action="store_true", help="Loop per parquet file: download -> cache+upload -> delete raw.")
    chunk.add_argument(
        "--source-wds-glob",
        default=None,
        help="Skip img2dataset and cache an existing HF WebDataset source, e.g. hf://datasets/Spawning/pd12m-full/*.tar.",
    )
    chunk.add_argument("--source-wds-group-size", type=int, default=10, help="Number of source WDS tar files per cache group.")
    chunk.add_argument("--source-wds-prefix", default="pd12m_full", help="Output prefix stem for --source-wds-glob groups.")
    chunk.add_argument("--max-chunks", type=int, default=None, help="Process at most this many new chunks (pilot).")
    chunk.add_argument("--chunk-row-limit", type=int, default=None, help="Cap rows per chunk (fast test, e.g. 2000).")
    chunk.add_argument("--chunk-raw-dir", default="/content/_pd12m_raw", help="Scratch dir for raw JPEG shards (auto-deleted).")
    chunk.add_argument("--chunk-cache-dir", default="/content/_pd12m_cache", help="Scratch dir for cache shards (auto-deleted after upload).")
    chunk.add_argument("--hf-repo-id", default=None, help="Target HF dataset repo for the cache (required for --chunk).")
    chunk.add_argument("--text-encoder-id", default="Qwen/Qwen3.5-2B")
    chunk.add_argument("--vae-id", default="mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers")
    chunk.add_argument("--max-length", type=int, default=160, help="Qwen token cache length.")
    chunk.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    chunk.add_argument("--latent-storage-dtype", default="bf16", choices=["fp16", "bf16", "fp32"])
    chunk.add_argument("--text-storage-dtype", default="fp8", help="On-disk dtype for the Qwen text payload (fp8 halves size).")
    chunk.add_argument("--batch-size", type=int, default=256)
    chunk.add_argument("--image-encode-batch", type=int, default=64)
    chunk.add_argument("--text-encode-batch", type=int, default=16)
    chunk.add_argument(
        "--cache-max-samples",
        type=int,
        default=1_000_000_000,
        help="Per-chunk cap passed to the Boomer cache writer. Large default means full downloaded chunk.",
    )
    chunk.add_argument("--shard-size-gb", type=float, default=2.0, help="Target cache shard size in GiB.")
    chunk.add_argument("--gpu-decode", action="store_true", default=True)
    chunk.add_argument("--no-gpu-decode", dest="gpu_decode", action="store_false")
    chunk.add_argument("--channels-last", action="store_true", default=True)
    args = parser.parse_args()

    token = hf_token(args.hf_token)
    if args.source_wds_glob:
        run_hf_wds_chunked(args, token)
        return

    metadata_dir = Path(args.metadata_dir)
    parquet_files = download_metadata(args.repo_id, args.metadata_pattern, metadata_dir, token)

    if not args.skip_attribution:
        write_attribution(parquet_files, None if args.chunk else args.limit, Path(args.output_folder))

    if args.chunk:
        run_chunked(args, parquet_files, token)
        return

    if args.skip_download:
        print("--skip-download set; stopping after metadata + attribution.")
        return

    url_list_dir = build_url_list(parquet_files, args.limit, metadata_dir / "_subset")
    run_img2dataset(args, url_list_dir)
    print(
        "\nNext: cache to Boomer shards:\n"
        f"  python3 tools/cache_hf_wds_to_boomer_wds.py \\\n"
        f"    --source-glob '{args.output_folder}/*.tar' \\\n"
        f"    --text-encoder-kind qwen --text-encoder-id Qwen/Qwen3.5-2B \\\n"
        f"    --vae-id mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers \\\n"
        f"    --resolution {args.image_size} --max-length 160 \\\n"
        f"    --out-dir /data/pd12m_qwen_cache --hf-token true"
    )


if __name__ == "__main__":
    main()
