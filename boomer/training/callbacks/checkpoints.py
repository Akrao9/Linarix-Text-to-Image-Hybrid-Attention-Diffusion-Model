"""Asynchronous checkpoint copying, upload, retention, and final flush."""

from __future__ import annotations

import os
import queue
import re
import shutil
import subprocess
import threading
from pathlib import Path

from boomer.training._lightning import Callback


class CheckpointUploadCallback(Callback):
    """Copy finished local checkpoints to a remote in a BACKGROUND thread, then keep
    only the newest ``local_retain`` checkpoints on local disk.

    Lightning's ModelCheckpoint writes locally (fast, reliable). This callback must be
    placed AFTER ModelCheckpoint in the callbacks list, so by the time its hook runs the
    .ckpt file exists on disk; it then enqueues the file for an async copy. Training is
    never blocked on the upload, and a Drive-mount stall can't corrupt the local save or
    pause the step. Local pruning only deletes checkpoints that have already uploaded, so
    a slow upload never loses a checkpoint.

    ``remote`` forms:
      * ``hf://buckets/<owner>/<bucket>`` — an HF storage bucket, uploaded with the
        ``hf`` CLI (``hf sync``). Each checkpoint is staged alone (hardlink) into a
        temp dir and synced, so one call never re-uploads or deletes earlier ones.
      * an rclone remote (``gdrive:Boomer/ckpts``).
      * a directory path such as a mounted Drive folder or an hf-mount'ed bucket
        (``hf-mount start bucket akrao9/Boomerv6-ckpt ./local`` then ``./local``);
        copied via tmp-file + rename so a reader never sees a partial file (with a
        direct-copy fallback for FUSE mounts that reject rename).
    """

    def __init__(
        self,
        *,
        checkpoint_dir: str,
        remote: str,
        local_retain: int = 2,
        keep_uploaded: int = 0,
        rclone_bin: str = "rclone",
        chunk_size: str = "128M",
        hf_bin: str = "hf",
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.remote = remote
        self.local_retain = max(1, int(local_retain))
        self.keep_uploaded = int(keep_uploaded)  # 0 = keep all uploaded
        self.rclone_bin = rclone_bin
        self.chunk_size = chunk_size
        self.hf_bin = hf_bin
        # hf:// = HF bucket via the hf CLI; rclone remotes look like "name:path";
        # anything else is a plain directory (mount).
        self.use_hf = remote.startswith("hf://")
        self.use_rclone = not self.use_hf and ":" in remote and not os.path.isabs(remote)
        if not self.use_hf and not self.use_rclone:
            remote_dir = Path(remote).resolve()
            local_dir = self.checkpoint_dir.resolve()
            if remote_dir == local_dir or local_dir.is_relative_to(remote_dir):
                raise ValueError(
                    "--checkpoint-upload-remote must be a DIFFERENT directory from "
                    "--checkpoint-dir: checkpoints save to fast local disk first, then "
                    "the background thread copies them to the remote/mount. Saving "
                    "straight into a network mount would block every training step "
                    "on network I/O."
                )
        self._q: queue.Queue = queue.Queue()
        self._uploaded: set[str] = set()       # filenames confirmed on the remote
        self._enqueued: set[str] = set()        # filenames already handed to the worker
        self._inflight: str | None = None       # filename the worker is copying RIGHT NOW
        self._lock = threading.Lock()
        self._worker: threading.Thread | None = None
        self._stop = threading.Event()

    # ---- worker plumbing -------------------------------------------------- #
    def on_fit_start(self, trainer, pl_module) -> None:
        if trainer.global_rank != 0 or self._worker is not None:
            return  # rank 0 only; single worker
        if self.use_hf:
            if shutil.which(self.hf_bin) is None:
                # Fail at fit start, not at the first save 2000 steps in.
                raise RuntimeError(
                    f"[ckpt-upload] '{self.hf_bin}' CLI not found for {self.remote}; "
                    "install it first: curl -LsSf https://hf.co/cli/install.sh | bash"
                )
            print(f"[ckpt-upload] hf sync -> {self.remote}", flush=True)
        elif self.use_rclone:
            print(f"[ckpt-upload] rclone -> {self.remote}", flush=True)
        else:
            os.makedirs(self.remote, exist_ok=True)
            print(f"[ckpt-upload] copy -> {self.remote}", flush=True)
        self._worker = threading.Thread(target=self._run, name="ckpt-upload", daemon=True)
        self._worker.start()

    def _run(self) -> None:
        while not (self._stop.is_set() and self._q.empty()):
            try:
                path = self._q.get(timeout=1.0)
            except queue.Empty:
                continue
            # Publish the in-flight name so finalize_uploads() can tell "not uploaded
            # yet" from "being uploaded right now" and never write the same destination
            # concurrently with this thread.
            with self._lock:
                self._inflight = Path(path).name
            try:
                self._upload_one(Path(path))
                with self._lock:
                    self._uploaded.add(Path(path).name)
                self._prune_local()
                if self.keep_uploaded > 0:
                    self._prune_remote()
            except Exception as exc:  # never let an upload error kill the worker
                # Drop it from _enqueued so the next _scan_and_enqueue re-queues the
                # (still-local) checkpoint instead of leaving it stranded until fit end.
                with self._lock:
                    self._enqueued.discard(Path(path).name)
                print(f"[ckpt-upload] FAILED {Path(path).name} (will retry): {exc}", flush=True)
            finally:
                with self._lock:
                    self._inflight = None
                self._q.task_done()

    def _upload_one(self, local: Path) -> None:
        if not local.exists():
            return
        if self.use_hf:
            import tempfile

            # Stage the ONE finished checkpoint alone (hardlink: same fs, no copy
            # cost) and sync that dir, so `hf sync` never walks in-progress files in
            # checkpoint_dir and one upload can't touch previously uploaded ckpts.
            with tempfile.TemporaryDirectory(dir=str(local.parent)) as stage:
                link = Path(stage) / local.name
                try:
                    os.link(str(local), str(link))
                except OSError:  # cross-device / fs without hardlinks
                    shutil.copyfile(str(local), str(link))
                proc = subprocess.run(
                    [self.hf_bin, "sync", stage, self.remote],
                    capture_output=True, text=True,
                )
                if proc.returncode != 0:
                    raise RuntimeError(
                        f"hf sync exited {proc.returncode}: {proc.stderr.strip()[-500:]}"
                    )
        elif self.use_rclone:
            dest = f"{self.remote.rstrip('/')}/{local.name}"
            cmd = [self.rclone_bin, "copyto", str(local), dest,
                   "--drive-chunk-size", self.chunk_size, "--no-traverse"]
            subprocess.run(cmd, check=True, capture_output=True)
        else:
            dest = Path(self.remote) / local.name
            tmp = Path(self.remote) / (local.name + ".uploading")
            try:
                # copyfile, not copy2: FUSE-backed mounts (hf-mount buckets, some
                # Drive mounts) often reject the chmod/utime metadata step.
                self._copy_and_fsync(local, tmp)       # copy to tmp + force durable...
                os.replace(str(tmp), str(dest))        # ...then atomic swap into place
            except OSError as exc:
                # Mounts without rename support: copy straight to the final name. This
                # is non-atomic, so if the copy is interrupted, delete the partial file
                # rather than leave a truncated boomer-step=N.ckpt that the resume
                # scanner would pick and fail to load; the retry re-uploads it.
                print(f"[ckpt-upload] tmp+rename failed on {self.remote} ({exc}); "
                      f"copying {local.name} directly", flush=True)
                tmp.unlink(missing_ok=True)
                try:
                    self._copy_and_fsync(local, dest)
                except BaseException:
                    Path(dest).unlink(missing_ok=True)
                    raise
            self._fsync_dir(Path(self.remote))         # persist the rename itself
        print(f"[ckpt-upload] uploaded {local.name}", flush=True)

    @staticmethod
    def _copy_and_fsync(src: Path, dst: Path) -> None:
        """Copy then fsync the destination. On a Drive/drivefs FUSE mount a plain
        copy only fills the client's local cache — the background upload to Google
        can still be in flight when the runtime is torn down, so the last
        checkpoint of a run silently never lands (the copy call returns success
        regardless). fsync forces drivefs to commit the bytes upstream before we
        move on, which is the subprocess-callable equivalent of Colab's
        drive.flush_and_unmount() (that API needs the notebook kernel and can't be
        called from this spawned process). No-op cost on real local disks."""
        with open(src, "rb") as fsrc, open(dst, "wb") as fdst:
            shutil.copyfileobj(fsrc, fdst)
            fdst.flush()
            try:
                os.fsync(fdst.fileno())
            except OSError:
                pass  # some mounts don't support fsync on files; best effort

    @staticmethod
    def _fsync_dir(path: Path) -> None:
        # fsync the directory so the rename/new entry is durable too (drivefs and
        # POSIX both need this for the metadata, not just the file contents).
        try:
            fd = os.open(str(path), os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        except OSError:
            pass  # directory fsync unsupported on this mount; best effort

    # ---- pruning ---------------------------------------------------------- #
    @staticmethod
    def _step_key(name: str) -> tuple[int, str]:
        """Retention order = training-step order parsed from 'boomer-step=N.ckpt'.

        mtime is unreliable (uploads finish out of order; copies reset it) and a
        lexicographic name sort misorders unpadded steps (10000 < 5000), either of
        which makes retention delete the NEWEST checkpoint. Unparseable names sort
        first (pruned before any step checkpoint)."""
        m = re.search(r"step=(\d+)", name)
        return (int(m.group(1)) if m else -1, name)

    def _local_step_ckpts(self) -> list[Path]:
        # All step checkpoints except the rolling 'last.ckpt' (which is overwritten).
        return sorted(
            (p for p in self.checkpoint_dir.glob("*.ckpt") if p.name != "last.ckpt"),
            key=lambda p: self._step_key(p.name),
        )

    def _prune_local(self) -> None:
        ckpts = self._local_step_ckpts()
        # keep the newest local_retain; delete older ones ONLY if already uploaded.
        for p in ckpts[: max(0, len(ckpts) - self.local_retain)]:
            with self._lock:
                done = p.name in self._uploaded
            if done:
                try:
                    p.unlink()
                    print(f"[ckpt-upload] pruned local {p.name}", flush=True)
                except FileNotFoundError:
                    pass

    def _prune_remote(self) -> None:
        if self.use_hf:
            # No known bucket-side delete verb in the hf CLI yet; keep every
            # uploaded checkpoint in the bucket (announce once).
            if not getattr(self, "_hf_prune_warned", False):
                self._hf_prune_warned = True
                print(
                    f"[ckpt-upload] --checkpoint-keep-uploaded is not supported for "
                    f"hf:// buckets; keeping all uploads in {self.remote}",
                    flush=True,
                )
            return
        if self.use_rclone:
            out = subprocess.run([self.rclone_bin, "lsf", self.remote],
                                 capture_output=True, text=True)
            names = sorted(
                (n for n in out.stdout.split() if n.endswith(".ckpt") and n != "last.ckpt"),
                key=self._step_key,
            )
            for n in names[: max(0, len(names) - self.keep_uploaded)]:
                subprocess.run([self.rclone_bin, "deletefile", f"{self.remote.rstrip('/')}/{n}"],
                               capture_output=True)
        else:
            files = sorted(
                (p for p in Path(self.remote).glob("*.ckpt") if p.name != "last.ckpt"),
                key=lambda p: self._step_key(p.name),
            )
            for p in files[: max(0, len(files) - self.keep_uploaded)]:
                try:
                    p.unlink()
                except FileNotFoundError:
                    pass

    # ---- hook: enqueue new checkpoints ------------------------------------ #
    def _scan_and_enqueue(self, trainer) -> None:
        if trainer.global_rank != 0:
            return
        for p in self.checkpoint_dir.glob("*.ckpt"):
            if p.name == "last.ckpt":
                continue
            with self._lock:
                seen = p.name in self._enqueued
            if not seen:
                with self._lock:
                    self._enqueued.add(p.name)
                self._q.put(str(p))

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        # Runs after ModelCheckpoint's hook (this callback is appended later), so any
        # checkpoint written this step already exists on disk.
        self._scan_and_enqueue(trainer)

    def on_fit_end(self, trainer, pl_module) -> None:
        # Make sure the final checkpoint(s) are enqueued, then drain before exit.
        self._scan_and_enqueue(trainer)
        if self._worker is None:
            return
        print("[ckpt-upload] draining pending uploads...", flush=True)
        self._q.join()
        self._stop.set()
        self._worker.join(timeout=30.0)

    def finalize_uploads(self, trainer, *, worker_grace: float = 900.0) -> None:
        """Synchronously ensure every local step checkpoint is on the remote.

        Safe to call from the main thread after fit(). Lightning's on_fit_end async drain
        can be pre-empted by the DataLoader-teardown crash (the final checkpoint is saved
        locally but its Drive copy is killed mid-flight), and the os._exit() fast-path would
        kill the worker thread too — so this is the reliable last flush before exit.

        ``worker_grace`` is deliberately long: one checkpoint is ~15 GB, and a Drive /
        rclone copy of that routinely runs for many minutes. The old 120 s timeout
        returned while the worker was STILL copying (it is a daemon thread, so nothing
        stopped it), and because a file only enters ``_uploaded`` AFTER its copy returns,
        the synchronous pass below then wrote the same destination concurrently — two
        writers on one path, i.e. a truncated/corrupt remote checkpoint, and most likely
        on the newest file. If the grace still expires we skip the in-flight file rather
        than race it: leaving it to the (still running) worker can only succeed or leave
        it absent, never corrupt.
        """
        if getattr(trainer, "global_rank", 0) != 0:
            return
        # Stop the async worker, letting it finish any in-flight copy first.
        if self._worker is not None:
            self._stop.set()
            # Take everything still QUEUED away from the worker before waiting. _stop
            # alone does not stop it pulling more work (its loop exits only once the
            # queue is ALSO empty), so without this it keeps starting fresh copies past
            # the grace and races the synchronous pass below on those files too — not
            # just on the one in flight.
            drained = 0
            while True:
                try:
                    self._q.get_nowait()
                except queue.Empty:
                    break
                self._q.task_done()
                drained += 1
            if drained:
                print(f"[ckpt-upload] reclaimed {drained} queued upload(s) from the "
                      "worker; flushing them synchronously", flush=True)
            with self._lock:
                pending = self._inflight
            if pending is not None:
                print(f"[ckpt-upload] waiting up to {worker_grace:.0f}s for the in-flight "
                      f"copy of {pending} to finish...", flush=True)
            self._worker.join(timeout=worker_grace)
            if self._worker.is_alive():
                with self._lock:
                    stuck = self._inflight
                print(f"[ckpt-upload] worker STILL copying {stuck or '?'} after "
                      f"{worker_grace:.0f}s; leaving it to that thread and skipping it "
                      "here (racing it would corrupt the remote copy)", flush=True)
            else:
                self._worker = None
        # Synchronously copy anything not yet confirmed on the remote, except a file the
        # worker thread may still be writing (see the grace note above).
        for ckpt in sorted(self._local_step_ckpts(), key=lambda p: p.stat().st_mtime):
            with self._lock:
                already_done = ckpt.name in self._uploaded
                being_copied = ckpt.name == self._inflight
            if already_done or being_copied:
                continue
            try:
                self._upload_one(ckpt)
                with self._lock:
                    self._uploaded.add(ckpt.name)
            except Exception as exc:  # noqa: BLE001 - a flush failure must not abort exit
                print(f"[ckpt-upload] final flush FAILED {ckpt.name}: {exc}", flush=True)
        # Directory/mount remotes: a final global sync forces any drivefs upload
        # queue to drain before the process (and often the whole runtime) exits.
        # _upload_one already fsync'd each file; this is the belt-and-suspenders
        # push for anything the mount buffered lazily. hf/rclone remotes upload
        # synchronously via their CLI, so they need nothing here.
        if not self.use_hf and not self.use_rclone:
            try:
                os.sync()
                print("[ckpt-upload] os.sync() flushed mount buffers", flush=True)
            except OSError:
                pass
