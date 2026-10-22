"""Progress reporting in optimizer-step units."""

from __future__ import annotations

from boomer.training._lightning import TQDMProgressBar


class StepUnitProgressBar(TQDMProgressBar):
    """Progress bar in OPTIMIZER-STEP units: global_step / max_steps.

    Lightning's default bar counts micro-batches, and for our infinite streaming
    caches it derives the total from max_steps -- which is in optimizer steps, a
    unit mismatch under --accumulate > 1 (upstream fix: Lightning PR #21802). Rather
    than reconciling both sides to batch units, display the run in the units every
    other knob here already uses (warmup, anneals, save cadence, max-steps): the
    numerator is global_step, the denominator max_steps, and it/s means optimizer
    steps/s. global_step is restored across ckpt resume, so a resumed session shows
    absolute progress (e.g. 40000/100000); `initial` is anchored at session start so
    tqdm's rate/ETA reflect only this session's throughput, not the pre-resume jump.
    """

    @property
    def total_train_batches(self) -> int | float:
        if self.trainer.max_steps is not None and self.trainer.max_steps > 0:
            return self.trainer.max_steps
        return super().total_train_batches

    def on_train_epoch_start(self, trainer, *args) -> None:
        super().on_train_epoch_start(trainer, *args)
        self.train_progress_bar.initial = trainer.global_step
        self.train_progress_bar.n = trainer.global_step

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        n = trainer.global_step
        if self._should_update(n, self.train_progress_bar.total):
            self.train_progress_bar.n = n
            self.train_progress_bar.refresh()
            self.train_progress_bar.set_postfix(self.get_metrics(trainer, pl_module))
