"""Text-to-image and conditioned sampling previews during training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from boomer.fill_masks import sample_fill_masks
from boomer.latent_norm import denormalize_latent
from boomer.sampling import sample_latent
from boomer.sana_latent_cache import (
    CacheConfig,
    decode_image_latent,
    extract_latent,
    load_vae,
    tensor_to_image,
    vae_scaling_factor,
)
from boomer.training._lightning import (
    Callback,
    WandbLogger,
)
from boomer.training.callbacks.ema import EMACallback


class SamplePreviewCallback(Callback):
    """Every N optimizer steps, sample a fixed set of prompts and log the images to W&B.

    Designed to add ~zero steady-state VRAM and never disturb training:
      * Reuses the resident DiT — no second model copy.
      * Samples on the UNCOMPILED module (``model._orig_mod``), so torch.compile's
        training graph is never re-traced for the inference shapes.
      * Text is PRE-encoded (see tools/encode_prompts.py), so Qwen is never loaded
        here; only the small DC-AE decoder is touched, and it's parked on CPU between
        previews and moved to GPU just for the decode.
      * Optionally swaps in EMA weights for the preview, then restores the live weights
        (backup kept on CPU to avoid a GPU spike).
      * The whole preview is wrapped in try/except: a preview failure (e.g. a transient
        OOM) logs a warning and training continues — a preview must never kill a run.
    """

    def __init__(
        self,
        *,
        every_n_steps: int,
        prompts_cache: str,
        ema_callback: "EMACallback | None",
        use_ema: bool,
        steps: int,
        sampler: str,
        stork_substeps: int,
        stork_derivative_order: int = 1,
        cfg_scale: float,
        cfg_rescale: float,
        cfg_interval: tuple[float, float],
        flow_shift: float,
        latent_mean: float,
        latent_std: float,
        seed: int,
        dtype: str,
        vae_id: str,
        scaling_factor: float,
        max_length: int,
        hf_token: str | bool | None,
        encode_on_the_fly: bool = False,
        prompts_file: str | None = None,
    ) -> None:
        self.every_n_steps = every_n_steps
        self.prompts_cache = prompts_cache
        # --monet-direct: pre-encoded prompt caches were built with the OLD encoder
        # (wrong dim/layer), so only prompt STRINGS are loaded and the resident
        # training encoder embeds them lazily at the first preview.
        self.encode_on_the_fly = encode_on_the_fly
        self.prompts_file = prompts_file
        self.ema_callback = ema_callback
        self.use_ema = use_ema
        self.steps = steps
        self.sampler = sampler
        self.stork_substeps = stork_substeps
        self.stork_derivative_order = stork_derivative_order
        self.cfg_scale = cfg_scale
        self.cfg_rescale = cfg_rescale
        self.cfg_interval = cfg_interval
        self.flow_shift = flow_shift
        self.latent_mean = latent_mean
        self.latent_std = latent_std
        self.seed = seed
        self.dtype = dtype
        self.hf_token = hf_token
        self._enabled = False
        self._last_preview_step = -1
        self._prompts: list[str] = []
        self._text_embeddings: torch.Tensor | None = None
        self._attention_masks: torch.Tensor | None = None
        self._vae = None
        self._cache_config = CacheConfig(
            vae_id=vae_id,
            max_length=max_length,
            dtype=dtype,
            device="cuda",
            scaling_factor=scaling_factor,
        )

    def on_fit_start(self, trainer, pl_module) -> None:
        if trainer.global_rank != 0:
            return
        if not isinstance(trainer.logger, WandbLogger):
            print("[sample-preview] no WandbLogger; previews disabled.", flush=True)
            return
        if self.encode_on_the_fly:
            if self.prompts_file and Path(self.prompts_file).is_file():
                lines = Path(self.prompts_file).read_text(encoding="utf-8").splitlines()
                self._prompts = [line.strip() for line in lines if line.strip()]
            elif Path(self.prompts_cache).is_file():
                # Reuse only the prompt strings; the cached embeddings belong to the
                # cache-era encoder and would be the wrong dim/representation.
                payload = torch.load(self.prompts_cache, map_location="cpu", weights_only=False)
                self._prompts = list(payload["prompts"])
            if not self._prompts:
                print(f"[sample-preview] no prompts found ({self.prompts_file or self.prompts_cache}); "
                      "previews disabled. Pass --sample-prompts-file with one prompt per line.",
                      flush=True)
                return
            self._enabled = True
            print(f"[sample-preview] {len(self._prompts)} prompts (encoded on the fly by the "
                  f"resident text encoder), every {self.every_n_steps} steps "
                  f"({'EMA' if self.use_ema else 'live'} weights, {self.sampler}, {self.steps} steps).",
                  flush=True)
            return
        path = Path(self.prompts_cache)
        if not path.is_file():
            print(f"[sample-preview] prompts cache not found ({path}); previews disabled. "
                  f"Run tools/encode_prompts.py --out {path} first.", flush=True)
            return
        payload = torch.load(path, map_location="cpu", weights_only=False)
        self._prompts = list(payload["prompts"])
        self._text_embeddings = payload["text_embeddings"]
        self._attention_masks = payload["attention_masks"]
        self._enabled = True
        print(f"[sample-preview] {len(self._prompts)} prompts, every {self.every_n_steps} steps "
              f"({'EMA' if self.use_ema else 'live'} weights, {self.sampler}, {self.steps} steps).",
              flush=True)

    @staticmethod
    def _strip_compile_prefix(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        prefix = "_orig_mod."
        return {(k[len(prefix):] if k.startswith(prefix) else k): v for k, v in state.items()}

    @torch.no_grad()
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if not self._enabled:
            return
        step = trainer.global_step
        # on_train_batch_end fires once per MICRO-batch, but global_step advances
        # only every accumulate_grad_batches steps -- so at an interval boundary the
        # step-modulo is true for all A micro-batches of the window. Dedup so the
        # preview runs exactly once per optimizer step.
        if step == 0 or step % self.every_n_steps != 0 or step == self._last_preview_step:
            return
        self._last_preview_step = step
        try:
            self._run_preview(trainer, pl_module, step)
        except Exception as exc:  # noqa: BLE001 - a preview must never kill training
            print(f"[sample-preview] step {step} failed ({type(exc).__name__}: {exc}); "
                  "skipping and continuing training.", flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _run_preview(self, trainer, pl_module, step: int) -> None:
        import wandb

        if self.encode_on_the_fly and self._text_embeddings is None:
            # One-time: the encoder is frozen and the prompts fixed, so encode once
            # and keep the embeddings on CPU (sliced back to GPU per prompt below).
            emb, mask = pl_module.encode_captions(self._prompts)
            self._text_embeddings = emb.cpu()
            self._attention_masks = mask.cpu()

        device = str(pl_module.device)
        device_type = "cuda" if "cuda" in device else "cpu"
        amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[self.dtype]
        # Sample on the eager module so torch.compile's training graph isn't re-traced.
        eager = getattr(pl_module.model, "_orig_mod", pl_module.model)
        was_training = eager.training
        eager.eval()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Optionally swap in EMA weights (backup the live weights on CPU to avoid a spike).
        backup = None
        if self.use_ema and self.ema_callback is not None and self.ema_callback.ema:
            backup = {k: v.detach().to("cpu", copy=True) for k, v in eager.state_dict().items()}
            eager.load_state_dict(self._strip_compile_prefix(self.ema_callback.ema), strict=False)

        try:
            self._ensure_vae(device)
            images, captions = [], []
            for i, prompt in enumerate(self._prompts):
                text_embedding = self._text_embeddings[i : i + 1]
                attention_mask = self._attention_masks[i : i + 1]
                with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=self.dtype != "fp32"):
                    latent = sample_latent(
                        eager,
                        text_embedding,
                        attention_mask,
                        steps=self.steps,
                        seed=self.seed,
                        dtype=torch.float32,
                        device=device,
                        cfg_scale=self.cfg_scale,
                        cfg_rescale=self.cfg_rescale,
                        sampler=self.sampler,
                        flow_shift=self.flow_shift,
                        stork_substeps=self.stork_substeps,
                        stork_derivative_order=self.stork_derivative_order,
                        cfg_interval=self.cfg_interval,
                    )
                latent = denormalize_latent(latent.float(), self.latent_mean, self.latent_std)
                decoded = decode_image_latent(self._vae, latent, self._cache_config)
                images.append(wandb.Image(tensor_to_image(decoded), caption=prompt))
                captions.append(prompt)
            trainer.logger.experiment.log({"samples": images, "trainer/global_step": step})
            print(f"[sample-preview] logged {len(images)} images at step {step}.", flush=True)
        finally:
            if backup is not None:
                eager.load_state_dict(backup, strict=False)
                del backup
            if was_training:
                eager.train()
            self._park_vae()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _ensure_vae(self, device: str) -> None:
        if self._vae is None:
            self._vae = load_vae(self._cache_config, hf_token=self.hf_token)
        self._vae = self._vae.to(device)

    def _park_vae(self) -> None:
        # Keep the decoder on CPU between previews so it costs ~0 steady-state GPU memory.
        if self._vae is not None:
            self._vae = self._vae.to("cpu")


class CondSamplePreviewCallback(Callback):
    """Every N optimizer steps, regenerate a FIXED set of CONDITIONED samples.

    Conditioning material is captured once from the FIRST training batch (zero
    startup cost — no extra stream, no eval-set plumbing):
      * fill mode: the first K clean latents + captions, with K deterministic
        masks synthesized once (seeded) and reused all run — so previews are
        comparable across steps: watch the kept region lock in early and the
        hole sharpen over training.
      * edit mode: the first K (source, ground-truth target) pixel pairs +
        instructions; mask is all-ones and the source is DC-AE-encoded at
        preview time.
    Logs [conditioning | generated] pairs (fill) or [source | expected target |
    generated] triplets (edit) under "cond_samples", so edit adherence is
    readable against the ground truth. Same never-kill-training / EMA-swap /
    parked-VAE discipline as SamplePreviewCallback.
    """

    def __init__(
        self,
        *,
        mode: str,  # "fill" | "edit"
        every_n_steps: int,
        count: int,
        ema_callback: "EMACallback | None",
        use_ema: bool,
        steps: int,
        sampler: str,
        stork_substeps: int,
        stork_derivative_order: int = 1,
        cfg_scale: float,
        cfg_rescale: float,
        cfg_interval: tuple[float, float],
        flow_shift: float,
        latent_mean: float,
        latent_std: float,
        seed: int,
        dtype: str,
        vae_id: str,
        scaling_factor: float,
        hf_token: str | bool | None,
        fill_mask_probs: dict[str, float] | None = None,
        encode_chunk: int = 8,
        image_cfg_scale: float = 0.0,
    ) -> None:
        self.mode = mode
        self.image_cfg_scale = image_cfg_scale
        self.every_n_steps = every_n_steps
        self.count = max(1, count)
        self.ema_callback = ema_callback
        self.use_ema = use_ema
        self.steps = steps
        self.sampler = sampler
        self.stork_substeps = stork_substeps
        self.stork_derivative_order = stork_derivative_order
        self.cfg_scale = cfg_scale
        self.cfg_rescale = cfg_rescale
        self.cfg_interval = cfg_interval
        self.flow_shift = flow_shift
        self.latent_mean = latent_mean
        self.latent_std = latent_std
        self.seed = seed
        self.dtype = dtype
        self.hf_token = hf_token
        self.fill_mask_probs = fill_mask_probs
        self.encode_chunk = max(1, encode_chunk)
        self._enabled = False
        self._last_preview_step = -1
        self._prompts: list[str] = []
        self._latents: torch.Tensor | None = None  # fill: RAW cache-space (K,C,H,W)
        self._masks: torch.Tensor | None = None  # fill: fixed (K,1,H,W)
        self._pixels: torch.Tensor | None = None  # edit: uint8 (K,3,R,R) sources
        self._target_pixels: torch.Tensor | None = None  # edit: uint8 ground truth
        self._text: torch.Tensor | None = None
        self._text_mask: torch.Tensor | None = None
        self._vae = None
        self._cache_config = CacheConfig(
            vae_id=vae_id, dtype=dtype, device="cuda", scaling_factor=scaling_factor
        )

    def on_fit_start(self, trainer, pl_module) -> None:
        if trainer.global_rank != 0:
            return
        if not isinstance(trainer.logger, WandbLogger):
            print("[cond-preview] no WandbLogger; conditioning previews disabled.", flush=True)
            return
        self._enabled = True
        print(f"[cond-preview] {self.mode} mode: {self.count} fixed conditions from the "
              f"first batch, every {self.every_n_steps} steps "
              f"({'EMA' if self.use_ema else 'live'} weights).", flush=True)

    @torch.no_grad()
    def _capture(self, batch: dict[str, Any]) -> None:
        self._prompts = [str(c) for c in batch["caption"][: self.count]]
        if self.mode == "fill":
            lat = batch["image_latent"][: self.count].detach().float().cpu()
            self._latents = lat
            gen = torch.Generator().manual_seed(self.seed)
            self._masks = sample_fill_masks(
                lat.shape[0], lat.shape[-2], lat.shape[-1],
                probs=self.fill_mask_probs, generator=gen,
            )
            # Cached-text runs carry usable embeddings; captions-only/monet runs
            # leave them None and the resident encoder fills in at first preview.
            text = batch.get("text_embedding")
            if text is not None:
                self._text = text[: self.count].detach().cpu()
                self._text_mask = batch["attention_mask"][: self.count].detach().cpu()
        else:
            self._pixels = batch["source_pixels"][: self.count].detach().cpu()
            self._target_pixels = batch["target_pixels"][: self.count].detach().cpu()
        coverage = ""
        if self._masks is not None:
            coverage = f", mask coverage {[round(float(m.mean()), 2) for m in self._masks]}"
        print(f"[cond-preview] captured {len(self._prompts)} conditions from the first "
              f"batch{coverage}.", flush=True)

    @torch.no_grad()
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if not self._enabled:
            return
        if not self._prompts:
            try:
                self._capture(batch)
            except Exception as exc:  # noqa: BLE001 - never kill training
                print(f"[cond-preview] capture failed ({type(exc).__name__}: {exc}); "
                      "conditioning previews disabled.", flush=True)
                self._enabled = False
                return
        step = trainer.global_step
        if step == 0 or step % self.every_n_steps != 0 or step == self._last_preview_step:
            return
        self._last_preview_step = step
        try:
            self._run_preview(trainer, pl_module, step)
        except Exception as exc:  # noqa: BLE001 - a preview must never kill training
            print(f"[cond-preview] step {step} failed ({type(exc).__name__}: {exc}); "
                  "skipping and continuing training.", flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _encode_source_latents(self, device: str) -> torch.Tensor:
        """edit mode: source pixels -> RAW cache-space latents via the preview VAE."""
        dtype = next(self._vae.parameters()).dtype
        scale = vae_scaling_factor(self._vae, self._cache_config.scaling_factor)
        parts = []
        for part in self._pixels.split(self.encode_chunk):
            images = part.to(device=device, dtype=dtype) / 127.5 - 1.0
            parts.append(extract_latent(self._vae.encode(images)))
        return torch.cat(parts, dim=0).float() * scale

    def _run_preview(self, trainer, pl_module, step: int) -> None:
        import wandb

        if self._text is None:
            emb, mask = pl_module.encode_captions(self._prompts)
            self._text = emb.cpu()
            self._text_mask = mask.cpu()

        device = str(pl_module.device)
        device_type = "cuda" if "cuda" in device else "cpu"
        amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[self.dtype]
        eager = getattr(pl_module.model, "_orig_mod", pl_module.model)
        was_training = eager.training
        eager.eval()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        backup = None
        if self.use_ema and self.ema_callback is not None and self.ema_callback.ema:
            backup = {k: v.detach().to("cpu", copy=True) for k, v in eager.state_dict().items()}
            eager.load_state_dict(
                SamplePreviewCallback._strip_compile_prefix(self.ema_callback.ema), strict=False
            )

        try:
            self._ensure_vae(device)
            # Build the conditioning batch in the NORMALIZED space the model sees,
            # and a raw-space visual of what the model was shown.
            if self.mode == "fill":
                raw = self._latents.to(device)
                masks = self._masks.to(device=device, dtype=raw.dtype)
                norm = (raw - self.latent_mean) / self.latent_std
                cond_all = torch.cat([norm * (1.0 - masks), masks], dim=1)
                visual_latents = raw * (1.0 - masks)  # raw-space holes decode ~gray
            else:
                raw = self._encode_source_latents(device)
                norm = (raw - self.latent_mean) / self.latent_std
                ones = norm.new_ones(norm.shape[0], 1, *norm.shape[-2:])
                cond_all = torch.cat([norm, ones], dim=1)
                visual_latents = None

            images = []
            for i, prompt in enumerate(self._prompts):
                target_pil = None
                if visual_latents is not None:
                    shown = decode_image_latent(
                        self._vae, visual_latents[i : i + 1], self._cache_config
                    )
                    cond_pil = tensor_to_image(shown)
                else:
                    from PIL import Image as _PILImage

                    cond_pil = _PILImage.fromarray(self._pixels[i].permute(1, 2, 0).numpy())
                    target_pil = _PILImage.fromarray(
                        self._target_pixels[i].permute(1, 2, 0).numpy()
                    )
                with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=self.dtype != "fp32"):
                    latent = sample_latent(
                        eager,
                        self._text[i : i + 1],
                        self._text_mask[i : i + 1],
                        steps=self.steps,
                        seed=self.seed,
                        dtype=torch.float32,
                        device=device,
                        cfg_scale=self.cfg_scale,
                        cfg_rescale=self.cfg_rescale,
                        sampler=self.sampler,
                        flow_shift=self.flow_shift,
                        stork_substeps=self.stork_substeps,
                        stork_derivative_order=self.stork_derivative_order,
                        cfg_interval=self.cfg_interval,
                        cond_input=cond_all[i : i + 1],
                        image_cfg_scale=self.image_cfg_scale,
                    )
                latent = denormalize_latent(latent.float(), self.latent_mean, self.latent_std)
                decoded = decode_image_latent(self._vae, latent, self._cache_config)
                images.append(wandb.Image(cond_pil, caption=f"[{self.mode} input] {prompt}"))
                if target_pil is not None:
                    # edit mode: [source | expected target | generated] — adherence
                    # is only readable against the ground-truth edit.
                    images.append(wandb.Image(target_pil, caption=f"[expected] {prompt}"))
                images.append(wandb.Image(tensor_to_image(decoded), caption=prompt))
            trainer.logger.experiment.log({"cond_samples": images, "trainer/global_step": step})
            print(f"[cond-preview] logged {len(self._prompts)} conditioned "
                  f"{'triplets' if self.mode == 'edit' else 'pairs'} at step {step}.",
                  flush=True)
        finally:
            if backup is not None:
                eager.load_state_dict(backup, strict=False)
                del backup
            if was_training:
                eager.train()
            self._park_vae()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _ensure_vae(self, device: str) -> None:
        if self._vae is None:
            self._vae = load_vae(self._cache_config, hf_token=self.hf_token)
        self._vae = self._vae.to(device)

    def _park_vae(self) -> None:
        if self._vae is not None:
            self._vae = self._vae.to("cpu")
