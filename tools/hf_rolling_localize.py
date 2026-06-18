"""Rolling HF shard localization helpers for WebDataset pipe readers.

`prefetch` downloads source tar shards with huggingface_hub/hf_xet into a small
ready window. `cat` waits for one ready shard, streams it to stdout for
WebDataset, then optionally deletes it so disk use stays bounded.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path


def read_lines(path: str | Path) -> list[str]:
    return [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def ready_count(local_dir: Path) -> int:
    return sum(1 for _ in local_dir.rglob("*.ready"))


def cleanup_empty_parents(path: Path, root: Path) -> None:
    root = root.resolve()
    current = path.resolve()
    while current != root and root in current.parents:
        try:
            current.rmdir()
        except OSError:
            break
        current = current.parent


def token_value(args: argparse.Namespace) -> str | None:
    if args.token:
        return args.token
    if args.token_env:
        return os.environ.get(args.token_env)
    return os.environ.get("HF_TOKEN")


def cmd_prefetch(args: argparse.Namespace) -> int:
    from huggingface_hub import hf_hub_download

    local_dir = Path(args.local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    token = token_value(args)
    shards = read_lines(args.source_list_file)

    lock = threading.Lock()
    in_flight = 0
    next_index = 0
    failed: BaseException | None = None

    def download_shard(index: int, shard: str) -> None:
        nonlocal in_flight, failed
        try:
            target = local_dir / shard
            ready = Path(f"{target}.ready")
            target.parent.mkdir(parents=True, exist_ok=True)
            print(f"[rolling-localize] download {index}: {shard}", file=sys.stderr, flush=True)
            downloaded = Path(
                hf_hub_download(
                    args.repo_id,
                    shard,
                    repo_type="dataset",
                    token=token,
                    local_dir=str(local_dir),
                )
            )
            if downloaded != target:
                target = downloaded
                ready = Path(f"{target}.ready")
            ready.write_text("ready\n", encoding="utf-8")
            print(f"[rolling-localize] ready {index}: {target}", file=sys.stderr, flush=True)
        except BaseException as exc:
            with lock:
                if failed is None:
                    failed = exc
            raise
        finally:
            with lock:
                in_flight -= 1

    pending: set = set()
    with ThreadPoolExecutor(max_workers=args.download_workers) as pool:
        while next_index < len(shards) or pending:
            if failed is not None:
                break

            with lock:
                while (
                    failed is None
                    and next_index < len(shards)
                    and in_flight < args.download_workers
                    and ready_count(local_dir) + in_flight < args.window
                ):
                    index = next_index + 1
                    shard = shards[next_index]
                    next_index += 1
                    in_flight += 1
                    pending.add(pool.submit(download_shard, index, shard))

            if failed is not None:
                break
            if not pending:
                time.sleep(args.poll_interval)
                continue

            done, pending = wait(pending, timeout=args.poll_interval, return_when=FIRST_COMPLETED)
            for future in done:
                try:
                    future.result()
                except BaseException:
                    break

    if failed is not None:
        print(f"[rolling-localize] download failed: {failed}", file=sys.stderr, flush=True)
        return 1

    print("[rolling-localize] all requested shards prefetched", file=sys.stderr, flush=True)
    return 0


def cmd_cat(args: argparse.Namespace) -> int:
    path = Path(args.path)
    ready = Path(args.ready or f"{path}.ready")
    deadline = None if args.timeout <= 0 else time.monotonic() + args.timeout

    while not (path.exists() and ready.exists()):
        if deadline is not None and time.monotonic() > deadline:
            print(f"[rolling-localize] timeout waiting for {path}", file=sys.stderr, flush=True)
            return 2
        time.sleep(args.poll_interval)

    try:
        with path.open("rb") as handle:
            shutil.copyfileobj(handle, sys.stdout.buffer, length=args.block_size)
        sys.stdout.buffer.flush()
    except BrokenPipeError:
        pass
    finally:
        if args.delete:
            path.unlink(missing_ok=True)
            ready.unlink(missing_ok=True)
            if args.local_dir:
                cleanup_empty_parents(path.parent, Path(args.local_dir))
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rolling HF shard localization helper.")
    sub = parser.add_subparsers(dest="command", required=True)

    prefetch = sub.add_parser("prefetch")
    prefetch.add_argument("--repo-id", required=True)
    prefetch.add_argument("--source-list-file", required=True)
    prefetch.add_argument("--local-dir", required=True)
    prefetch.add_argument("--window", type=int, default=4)
    prefetch.add_argument(
        "--download-workers",
        type=int,
        default=3,
        help="Concurrent hf_hub_download calls while respecting --window.",
    )
    prefetch.add_argument("--poll-interval", type=float, default=1.0)
    prefetch.add_argument("--token")
    prefetch.add_argument("--token-env", default="HF_TOKEN")
    prefetch.set_defaults(func=cmd_prefetch)

    cat = sub.add_parser("cat")
    cat.add_argument("--path", required=True)
    cat.add_argument("--ready")
    cat.add_argument("--local-dir")
    cat.add_argument("--delete", action="store_true")
    cat.add_argument("--timeout", type=float, default=3600.0)
    cat.add_argument("--poll-interval", type=float, default=0.25)
    cat.add_argument("--block-size", type=int, default=8 * 1024 * 1024)
    cat.set_defaults(func=cmd_cat)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
