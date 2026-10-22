"""Exponential moving average of the trainable model."""

from __future__ import annotations

import torch

from boomer.models.boomer_fla_dit import migrate_glumbconv_state_dict
from boomer.training._lightning import (
    pl,
    Callback,
)


class EMACallback(Callback):
    def __init__(self, decay: float = 0.9999, update_every: int = 1) -> None:
        self.decay = decay
        self.update_every = max(1, update_every)
        self.ema: dict[str, torch.Tensor] = {}
        # Last global_step an EMA update ran for; see on_train_batch_end.
        self._last_step = -1

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not self.ema:  # fresh run (not resumed)
            self.ema = {
                k: v.detach().clone().float()
                for k, v in pl_module.model.state_dict().items()
                if torch.is_floating_point(v)
            }
        # Provisional anchor; the authoritative one is taken in on_train_start (see there).
        self._last_step = trainer.global_step

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # Anchor the step tracker at the true starting step so a resume does not spend
        # one spurious update on the pre-resume weights. This MUST NOT be done only in
        # on_fit_start: Lightning restores the loop state (and with it global_step) in
        # restore_training_state(), which runs AFTER the on_fit_start hooks, so a resumed
        # run still reads global_step == 0 there. on_train_start runs inside _run_stage(),
        # after the restore, so it sees the real step.
        self._last_step = trainer.global_step

    @torch.no_grad()
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        # This hook fires once per MICRO-batch, but global_step only advances on
        # optimizer steps. Under --accumulate N that is N calls per step: updating on
        # each one applies decay**N per optimizer step, silently shortening the EMA
        # horizon by a factor of N (--accumulate 4 turned decay=0.9999 into 0.9996).
        # It also broke --ema-update-every > 1, which read a global_step that is
        # constant across an accumulation window: the modulo matched either all N
        # micro-batches or none, so the cadence tracked step parity, not the setting.
        # Gate on global_step ADVANCING so one optimizer step == one EMA update, and
        # keep the modulo in the units the flag is documented in (optimizer steps).
        step = trainer.global_step
        if step == self._last_step:
            return
        self._last_step = step
        if step % self.update_every != 0:
            return
        sd = pl_module.model.state_dict()
        for key, ema_v in self.ema.items():
            target = sd[key].detach().float()
            if ema_v.device != target.device:
                # On resume the EMA shadow is restored on CPU while the live model is
                # already on GPU; realign once (later steps are a cheap device compare).
                ema_v = ema_v.to(target.device)
                self.ema[key] = ema_v
            ema_v.mul_(self.decay).add_(target, alpha=1.0 - self.decay)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint) -> None:
        checkpoint["ema"] = self.ema

    def on_load_checkpoint(self, trainer, pl_module, checkpoint) -> None:
        if "ema" in checkpoint:
            ema = {k: v.float() for k, v in checkpoint["ema"].items()}
            # The module-level GLUMBConv migration only covers Module.load_state_dict;
            # the EMA shadow is a raw dict and must be migrated explicitly or the first
            # update after resuming a pre-refactor checkpoint KeyErrors.
            n = migrate_glumbconv_state_dict(ema)
            if n:
                print(f"[ema] migrated {n} legacy GLUMBConv entr{'y' if n == 1 else 'ies'}",
                      flush=True)
            self.ema = ema
