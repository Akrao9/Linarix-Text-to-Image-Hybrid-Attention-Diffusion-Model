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
import traceback
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path


def read_lines(path: str | Path) -> list[str]:
    return [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def shard_is_ready(path: Path) -> bool:
    return path.exists() and Path(f"{path}.ready").exists()


def download_with_watchdog(call, timeout: float):
    """Run a blocking download with a wall-clock watchdog.

    huggingface_hub/hf_xet can stall indefinitely on a half-open connection
    without ever raising, so the retry loop -- which only fires on exceptions --
    never triggers and the head-of-line shard never becomes ready. Running the
    call in a daemon side-thread and joining with a deadline converts such a
    silent stall into a TimeoutError the retry loop can act on. The stalled
    thread is left to die with the process (daemon) instead of blocking exit.
    """
    if timeout is None or timeout <= 0:
        return call()
    box: dict[str, object] = {}

    def runner() -> None:
        try:
            box["result"] = call()
        except BaseException as exc:  # noqa: BLE001 - re-raised to the caller below
            box["error"] = exc

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        raise TimeoutError(f"download stalled beyond {timeout:g}s with no progress")
    if "error" in box:
        raise box["error"]  # type: ignore[misc]
    return box["result"]  # type: ignore[return-value]


def purge_partial_download(local_dir: Path, shard: str) -> None:
    """Remove leftover lock/incomplete files for a shard before retrying it.

    huggingface_hub stages downloads under ``<local_dir>/.cache/huggingface/download``
    with ``.lock``/``.incomplete``/``.metadata`` sidecars. If a prior attempt was
    abandoned mid-flight, a stale lock or corrupt partial can wedge the retry; clearing
    them forces a clean attempt (cheap: shards re-fetch in seconds on a healthy link).
    """
    cache = local_dir / ".cache" / "huggingface" / "download"
    if not cache.exists():
        return
    stem = Path(shard).name
    for pattern in (f"{stem}*.lock", f"{stem}*.incomplete", f"{stem}*.metadata"):
        for leftover in cache.rglob(pattern):
            try:
                leftover.unlink()
            except OSError:
                pass


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
    # Bound the per-read socket timeout so the download raises on a stalled
    # connection instead of blocking forever, and disable hf_xet so that timeout is
    # actually honored. hf_xet runs the transfer in a Rust thread the Python watchdog
    # cannot cancel: an abandoned xet download keeps holding the shard's .lock, so the
    # next retry blocks on it and stalls too -- turning one network stall into a cascade
    # that burns every attempt. The plain HTTP path respects the timeout, raises, and
    # releases its lock, so retries get a clean shot. Both are setdefault, so an
    # explicit env from the caller still wins.
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import hf_hub_download

    local_dir = Path(args.local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    token = token_value(args)
    shards = read_lines(args.source_list_file)

    lock = threading.Lock()
    in_flight = 0
    next_index = 0
    failed: BaseException | None = None
    consecutive_skips = 0
    total_skipped = 0
    # Only shards admitted by this invocation belong to its window. Counting every
    # .ready file below local_dir can deadlock a resumed run: later shards left by an
    # interrupted prefetch fill the window while the first missing shard is never
    # scheduled.
    admitted_ready: set[Path] = set()

    def append_skipped(shard: str) -> None:
        if not args.skipped_log:
            return
        log_path = Path(args.skipped_log)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{shard}\n")

    def publish_failure(message: str) -> None:
        if not args.error_file:
            return
        error_path = Path(args.error_file)
        error_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(f"{error_path}.tmp")
        temporary.write_text(message, encoding="utf-8")
        temporary.replace(error_path)

    def download_shard(index: int, shard: str) -> Path | None:
        """Download one shard. Returns its path, or None when the shard was skipped.

        On a terminal failure with --skip-on-failure, write a .skip marker (so the
        consuming cacher moves past the shard instead of waiting out its timeout),
        record it in the skipped log, and keep going. Only a run of
        --max-consecutive-skips failures -- i.e. a real outage rather than one bad
        shard -- aborts the sidecar.
        """
        nonlocal in_flight, failed, consecutive_skips, total_skipped
        try:
            target = local_dir / shard
            ready = Path(f"{target}.ready")
            skip_marker = Path(f"{target}{args.skip_suffix}")
            target.parent.mkdir(parents=True, exist_ok=True)
            skip_marker.unlink(missing_ok=True)  # clear a stale skip from a prior run
            # Clear any lock/partial left by a previous (possibly killed) run so the
            # first attempt is not wedged by a cross-run leftover.
            purge_partial_download(local_dir, shard)
            last_exc: BaseException | None = None
            for attempt in range(1, args.max_download_attempts + 1):
                try:
                    print(
                        f"[rolling-localize] download {index}: {shard} "
                        f"(attempt {attempt}/{args.max_download_attempts})",
                        file=sys.stderr,
                        flush=True,
                    )
                    downloaded = Path(
                        download_with_watchdog(
                            lambda: hf_hub_download(
                                args.repo_id,
                                shard,
                                repo_type="dataset",
                                token=token,
                                local_dir=str(local_dir),
                            ),
                            args.download_timeout,
                        )
                    )
                    if downloaded != target:
                        target = downloaded
                        ready = Path(f"{target}.ready")
                    ready.write_text("ready\n", encoding="utf-8")
                    print(f"[rolling-localize] ready {index}: {target}", file=sys.stderr, flush=True)
                    with lock:
                        consecutive_skips = 0
                    # The file now lives at target; drop the staging lock/partial/metadata
                    # so .cache/huggingface/download does not accumulate across the run.
                    purge_partial_download(local_dir, shard)
                    return target
                except Exception as exc:
                    last_exc = exc
                    if attempt >= args.max_download_attempts:
                        break
                    delay = args.retry_delay * attempt
                    print(
                        f"[rolling-localize] retry {index} in {delay:g}s after "
                        f"{type(exc).__name__}: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
                    time.sleep(delay)
                    purge_partial_download(local_dir, shard)

            # All attempts exhausted.
            if not args.skip_on_failure:
                with lock:
                    if failed is None:
                        failed = last_exc
                        failure_message = (
                            f"rolling source download failed for {shard}:\n"
                            f"{''.join(traceback.format_exception(last_exc)) if last_exc else ''}"
                        )
                        publish_failure(failure_message)
                        print(f"[rolling-localize] {failure_message}", file=sys.stderr, flush=True)
                if last_exc is not None:
                    raise last_exc
                raise RuntimeError(f"download failed for {shard}")

            purge_partial_download(local_dir, shard)
            skip_marker.write_text("skip\n", encoding="utf-8")
            with lock:
                total_skipped += 1
                consecutive_skips += 1
                append_skipped(shard)
                give_up = consecutive_skips > args.max_consecutive_skips
                run_len = consecutive_skips
            print(
                f"[rolling-localize] SKIP {index}: {shard} after {args.max_download_attempts} "
                f"attempts ({type(last_exc).__name__ if last_exc else 'Unknown'}); "
                f"consecutive_skips={run_len}",
                file=sys.stderr,
                flush=True,
            )
            if give_up:
                with lock:
                    if failed is None:
                        failed = RuntimeError(
                            f"{run_len} consecutive source-shard download failures "
                            f"(>{args.max_consecutive_skips}); aborting (last: {shard})"
                        )
                        publish_failure(str(failed))
                        print(f"[rolling-localize] {failed}", file=sys.stderr, flush=True)
            return None
        finally:
            with lock:
                in_flight -= 1

    pending: set = set()
    with ThreadPoolExecutor(max_workers=args.download_workers) as pool:
        while next_index < len(shards) or pending:
            if failed is not None:
                break

            with lock:
                admitted_ready = {path for path in admitted_ready if shard_is_ready(path)}
                while (
                    failed is None
                    and next_index < len(shards)
                    and in_flight < args.download_workers
                    and len(admitted_ready) + len(pending) < args.window
                ):
                    index = next_index + 1
                    shard = shards[next_index]
                    next_index += 1
                    target = local_dir / shard
                    if shard_is_ready(target):
                        admitted_ready.add(target)
                        print(
                            f"[rolling-localize] reuse {index}: {target}",
                            file=sys.stderr,
                            flush=True,
                        )
                        continue
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
                    result = future.result()
                except BaseException:
                    break
                if result is not None:
                    admitted_ready.add(result)

    if failed is not None:
        print(f"[rolling-localize] download failed: {failed}", file=sys.stderr, flush=True)
        return 1

    print(
        f"[rolling-localize] all requested shards prefetched ({total_skipped} skipped)",
        file=sys.stderr,
        flush=True,
    )
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
    prefetch.add_argument(
        "--max-download-attempts",
        type=int,
        default=5,
        help="Maximum attempts for each source shard before failing the sidecar.",
    )
    prefetch.add_argument(
        "--retry-delay",
        type=float,
        default=5.0,
        help="Base retry delay in seconds; multiplied by the attempt number.",
    )
    prefetch.add_argument(
        "--download-timeout",
        type=float,
        default=600.0,
        help=(
            "Per-attempt wall-clock seconds before a stalled (non-raising) download "
            "is abandoned and treated as a failed attempt. 0 disables the watchdog."
        ),
    )
    prefetch.add_argument(
        "--skip-on-failure",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="On terminal shard failure, mark it skipped and continue instead of aborting.",
    )
    prefetch.add_argument(
        "--skip-suffix",
        default=".skip",
        help="Marker suffix written next to a shard that could not be downloaded.",
    )
    prefetch.add_argument(
        "--max-consecutive-skips",
        type=int,
        default=5,
        help="Abort once this many shards fail back-to-back (a real outage, not one bad shard).",
    )
    prefetch.add_argument(
        "--skipped-log",
        help="Append the name of each skipped shard here for a later re-fetch pass.",
    )
    prefetch.add_argument("--token")
    prefetch.add_argument("--token-env", default="HF_TOKEN")
    prefetch.add_argument(
        "--error-file",
        help="Write the first terminal download traceback here for the parent process to detect.",
    )
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
