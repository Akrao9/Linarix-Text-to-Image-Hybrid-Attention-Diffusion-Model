"""BoomerPipeline — HuggingFace DiffusionPipeline wrapper for Boomer FLA.

Load with:
    import torch
    from diffusers import DiffusionPipeline

    pipe = DiffusionPipeline.from_pretrained(
        "akrao9/Boomer-T2I",
        custom_pipeline="pipeline_boomer",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to("cuda")

    image = pipe("a photorealistic portrait of a woman with dark hair")[0]

Requires:
    pip install torch diffusers transformers accelerate safetensors
    pip install git+https://github.com/fla-org/flash-linear-attention.git
    # STORKScheduler.py is bundled with the model repo — cite Tan et al. 2025 (arXiv:2505.24210)
"""

from __future__ import annotations

import gc
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Union

import torch
from diffusers import DiffusionPipeline
from diffusers.utils import logging
from PIL import Image

logger = logging.get_logger(__name__)


# ── output type ────────────────────────────────────────────────────────────────
@dataclass
class BoomerOutput:
    images: List[Image.Image]

    def __iter__(self):  return iter(self.images)
    def __getitem__(self, i): return self.images[i]
    def __len__(self):   return len(self.images)


@dataclass
class BoomerRunStats:
    """CUDA memory high-water marks (GB) from the most recent __call__."""

    peak_encode_gb: float = 0.0
    peak_denoise_gb: float = 0.0
    peak_decode_gb: float = 0.0


def _cuda_peak_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024**3)


def _reset_cuda_peak() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


# ── bundled-code loading ──────────────────────────────────────────────────────
def _find_bundled_file(filename: str, snapshot_dir: str | None = None) -> Path:
    """Find code shipped in this model snapshot; never borrow it from another repo."""
    candidates: list[Path] = []
    if snapshot_dir:
        candidates.append(Path(snapshot_dir) / filename)
    candidates.append(Path(__file__).resolve().parent / filename)
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"{filename} is missing from this model snapshot. Re-export the checkpoint; "
        "using code from another cached model could silently change inference."
    )


def _load_bundled_module(name: str, filename: str, snapshot_dir: str | None = None) -> Any:
    path = _find_bundled_file(filename, snapshot_dir)
    module_name = f"boomer_export_{name}_{abs(hash(str(path.resolve())))}"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create an import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# ── STORK scheduler (Tan et al. 2025, arXiv:2505.24210) ──────────────────────
def _find_stork_file(snapshot_dir: str | None = None) -> Path:
    return _find_bundled_file("STORKScheduler.py", snapshot_dir)


def _make_stork_scheduler(
    steps: int,
    device: str | torch.device,
    flow_shift: float,
    solver_order: int,
    derivative_order: int,
    substeps: int,
    snapshot_dir: str | None = None,
) -> Any:
    stork_path = _find_stork_file(snapshot_dir)
    spec = importlib.util.spec_from_file_location("boomer_export_stork", stork_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load bundled STORK scheduler at {stork_path}")
    module = importlib.util.module_from_spec(spec)
    # STORKScheduler imports its sibling STORK_constants package by name.
    parent = str(stork_path.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    sys.modules["boomer_export_stork"] = module
    spec.loader.exec_module(module)

    scheduler = module.STORKScheduler(
        shift=flow_shift,
        solver_order=solver_order,
        prediction_type="flow_prediction",
        derivative_order=derivative_order,
        s=substeps,
    )
    base_sigmas = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float32)[:-1].tolist()
    scheduler.set_timesteps(num_inference_steps=steps, device=device, sigmas=base_sigmas)
    scheduler.dt_list = scheduler.dt_list.to(device=device)
    scheduler.sigmas  = scheduler.sigmas.to(device=device)
    return scheduler


# ── latent helpers ─────────────────────────────────────────────────────────────
def _stat_tensor(v, latent):
    t = torch.as_tensor(v, device=latent.device, dtype=latent.dtype)
    if t.ndim == 1:
        return t.view(1, t.numel(), 1, 1)
    return t

def _denorm(latent, mean, std):
    return latent * _stat_tensor(std, latent) + _stat_tensor(mean, latent)


# ── pipeline ───────────────────────────────────────────────────────────────────
class BoomerPipeline(DiffusionPipeline):
    """
    Text-to-image pipeline for Boomer FLA (Flash Linear Attention DiT).

    The transformer weights live in this repo. VAE and text encoder are
    fetched from their upstream HuggingFace repos on first use.
    """

    model_cpu_offload_seq = "text_encoder->transformer->vae"

    def __init__(self, transformer, vae=None, text_encoder=None, tokenizer=None) -> None:
        super().__init__()
        self.register_modules(
            transformer=transformer,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
        )
        cfg = getattr(transformer, "_boomer_cfg", {})
        self.latent_mean     = cfg.get("_latent_mean",     0.0)
        self.latent_std      = cfg.get("_latent_std",      1.0)
        self.scaling_factor  = cfg.get("_scaling_factor",  0.41407)
        self.flow_shift      = cfg.get("_flow_shift",      1.5)
        self.max_text_length = cfg.get("_max_text_length", 300)
        self.instruction     = cfg.get("_instruction",     None)
        self.text_encoder_layer = int(cfg.get("_text_encoder_layer", -1))
        layers = cfg.get("_text_encoder_layers")
        self.text_encoder_layers = tuple(int(layer) for layer in layers) if layers else None
        self._vae_repo       = cfg.get("_vae_repo",  "mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers")
        self._te_repo        = cfg.get("_te_repo",   "Qwen/Qwen3.5-4B")
        self.default_sampler = cfg.get("_sampler", "stork4")
        self.default_steps = int(cfg.get("_steps", 20))
        self.default_stork_derivative_order = int(cfg.get("_stork_derivative_order", 1))
        self.default_stork_substeps = int(cfg.get("_stork_substeps", 9))
        self.default_cfg_scale = float(cfg.get("_cfg_scale", 3.0))
        self.default_cfg_rescale = float(cfg.get("_cfg_rescale", 0.5))
        self.default_cfg_interval = (
            float(cfg.get("_cfg_interval_low", 0.1)),
            float(cfg.get("_cfg_interval_high", 0.9)),
        )
        self._hf_token       = None
        self._snapshot_dir   = getattr(transformer, "_snapshot_dir", None)
        self._encode_qwen_text_fixed = None
        self.last_run_stats  = BoomerRunStats()
        # VAE memory savers. Slicing decodes one image at a time so batched-decode peak
        # memory stays flat (diffusers best practice: ~free for batches, no single-image
        # cost). Stored as flags because the VAE is lazy-loaded in _ensure_vae — they are
        # (re)applied there, so enable_vae_slicing/tiling work whether called before or
        # after the VAE exists. Slicing defaults on since this pipeline supports batching.
        self._vae_slicing = True
        self._vae_tiling  = False

    # ── lazy component loading ─────────────────────────────────────────────────
    def _ensure_vae(self):
        if self.vae is not None:
            return
        from diffusers import AutoencoderDC
        dtype  = next(self.transformer.parameters()).dtype
        logger.info(f"Loading VAE from {self._vae_repo} ...")
        self.vae = AutoencoderDC.from_pretrained(
            self._vae_repo, torch_dtype=dtype, token=self._hf_token,
        ).to(self._execution_device).eval()
        # Apply any savers requested before the VAE existed (default: slicing on).
        self._apply_vae_memory_savers()

    def _ensure_text_encoder(self):
        if self.text_encoder is not None:
            return
        dtype = next(self.transformer.parameters()).dtype
        logger.info(f"Loading text encoder from {self._te_repo} ...")
        qwen = _load_bundled_module(
            "qwen_connector", "qwen_connector.py", self._snapshot_dir
        )
        dtype_name = {
            torch.bfloat16: "bf16",
            torch.float16: "fp16",
            torch.float32: "fp32",
        }.get(dtype)
        if dtype_name is None:
            raise TypeError(f"Unsupported text-encoder dtype: {dtype}")
        qwen_config = qwen.QwenConnectorConfig(
            qwen_model_id=self._te_repo,
            readout_layer=self.text_encoder_layer,
            readout_layers=self.text_encoder_layers,
            instruction=self.instruction,
            max_text_tokens=self.max_text_length,
            pad_to_max_length=False,
            dtype=dtype_name,
        )
        encoder = qwen.QwenReferenceEncoder(qwen_config, hf_token=self._hf_token)
        encoder.load()
        expected = int(self.transformer.config.text_dim)
        if encoder.readout_dim != expected:
            raise RuntimeError(
                f"Qwen readout emits {encoder.readout_dim} features but the transformer "
                f"expects text_dim={expected}. Export metadata is inconsistent with its "
                "checkpoint."
            )
        self.text_encoder = encoder.to(self._execution_device).eval()
        self._encode_qwen_text_fixed = qwen.encode_qwen_text_fixed

    # ── VAE memory savers (deferred to the lazily-loaded VAE) ──────────────────
    def _apply_vae_memory_savers(self) -> None:
        """Apply the requested slicing/tiling flags to the VAE if it is loaded.

        No-op when the VAE is still None; _ensure_vae() calls this after loading, so
        enable_vae_slicing()/enable_vae_tiling() take effect even when called before the
        VAE exists (the diffusers-stock methods touch self.vae directly and silently
        no-op — or crash — when it is None).
        """
        if self.vae is None:
            return
        slicing = "enable_slicing" if self._vae_slicing else "disable_slicing"
        tiling = "enable_tiling" if self._vae_tiling else "disable_tiling"
        for name in (slicing, tiling):
            fn = getattr(self.vae, name, None)
            if callable(fn):
                try:
                    fn()
                except Exception as exc:  # noqa: BLE001 - saver is best-effort
                    logger.warning(f"VAE {name}() unavailable: {exc!r}")

    def enable_vae_slicing(self) -> None:
        """Decode one image at a time so batched-decode peak memory stays flat."""
        self._vae_slicing = True
        self._apply_vae_memory_savers()

    def disable_vae_slicing(self) -> None:
        self._vae_slicing = False
        self._apply_vae_memory_savers()

    def enable_vae_tiling(self) -> None:
        """Decode each image in overlapping tiles — for very large images / low VRAM."""
        self._vae_tiling = True
        self._apply_vae_memory_savers()

    def disable_vae_tiling(self) -> None:
        self._vae_tiling = False
        self._apply_vae_memory_savers()

    # ── from_pretrained override (direct BoomerPipeline.from_pretrained calls) ─
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):  # type: ignore[override]
        from huggingface_hub import snapshot_download

        token = kwargs.pop("token", None) or kwargs.pop("use_auth_token", None)
        dtype = kwargs.pop("torch_dtype", torch.bfloat16)

        if Path(pretrained_model_name_or_path).is_dir():
            local = Path(pretrained_model_name_or_path)
        else:
            logger.info(f"Downloading snapshot from {pretrained_model_name_or_path} ...")
            local = Path(snapshot_download(
                pretrained_model_name_or_path, token=token,
                ignore_patterns=["*.png", "*.jpg", "*.jpeg"],
            ))

        if str(local) not in sys.path:
            sys.path.insert(0, str(local))

        from modeling_boomer_fla import BoomerFLADiT  # noqa: PLC0415

        transformer = BoomerFLADiT.from_pretrained(str(local), subfolder="transformer")
        transformer = transformer.to(dtype=dtype)
        transformer._snapshot_dir = str(local)

        pipe = cls(transformer=transformer)
        pipe._hf_token = token
        return pipe

    # ── generation ─────────────────────────────────────────────────────────────
    @torch.inference_mode()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        steps: int | None = None,
        seed: int | None = None,
        cfg_scale: float | None = None,
        cfg_rescale: float | None = None,
        sampler: str | None = None,
        derivative_order: int | None = None,
        substeps: int | None = None,
        cfg_interval_low: float | None = None,
        cfg_interval_high: float | None = None,
        offload_text_encoder: bool = True,
        **kwargs,
    ) -> BoomerOutput:
        """
        Parameters
        ----------
        prompt       : str or list[str]
        steps        : denoising steps (exported default: 20)
        seed         : random seed — None (default) picks a random seed each call
        sampler      : ``stork2`` or ``stork4`` (exported default: STORK-4)
        substeps     : STORK internal RKG substeps (exported default: 9)
        offload_text_encoder : unload text encoder after encoding to free VRAM
        """
        import random
        steps = self.default_steps if steps is None else int(steps)
        cfg_scale = self.default_cfg_scale if cfg_scale is None else float(cfg_scale)
        cfg_rescale = self.default_cfg_rescale if cfg_rescale is None else float(cfg_rescale)
        sampler = self.default_sampler if sampler is None else sampler
        derivative_order = (
            self.default_stork_derivative_order
            if derivative_order is None else int(derivative_order)
        )
        substeps = self.default_stork_substeps if substeps is None else int(substeps)
        interval = (
            self.default_cfg_interval[0] if cfg_interval_low is None else float(cfg_interval_low),
            self.default_cfg_interval[1] if cfg_interval_high is None else float(cfg_interval_high),
        )
        if sampler not in {"stork2", "stork4"}:
            raise ValueError("sampler must be 'stork2' or 'stork4'")
        if not 0.0 <= interval[0] <= interval[1] <= 1.0:
            raise ValueError("CFG interval must satisfy 0 <= low <= high <= 1")
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        logger.info(f"seed={seed}")

        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        batch   = len(prompts)
        device  = self._execution_device
        dtype   = next(self.transformer.parameters()).dtype
        run_stats = BoomerRunStats()
        track_vram = torch.cuda.is_available() and str(device) != "cpu"

        # ── Phase 1: encode text ───────────────────────────────────────────────
        if track_vram:
            _reset_cuda_peak()
        self._ensure_text_encoder()
        self.text_encoder.to(device)
        if self._encode_qwen_text_fixed is None:
            raise RuntimeError("Qwen text encoder loaded without its fixed-length encoder")
        text_emb, attn_mask = self._encode_qwen_text_fixed(
            self.text_encoder, prompts, self.max_text_length
        )
        text_emb = text_emb.to(device=device, dtype=dtype)
        attn_mask = attn_mask.to(device=device)
        if track_vram:
            run_stats.peak_encode_gb = _cuda_peak_gb()
        if offload_text_encoder:
            self.text_encoder.to("cpu")
            gc.collect()
            if device != "cpu":
                torch.cuda.empty_cache()

        # ── Phase 2: denoise ───────────────────────────────────────────────────
        if track_vram:
            _reset_cuda_peak()
        self.transformer.to(device)
        uncond_emb, uncond_mask = self.transformer.null_condition(
            batch, device=device, dtype=dtype,
            mask_dtype=attn_mask.dtype, token_num=text_emb.shape[-2],
        )
        sched = _make_stork_scheduler(
            steps=steps, device=device, flow_shift=self.flow_shift,
            solver_order=int(sampler.removeprefix("stork")),
            derivative_order=derivative_order, substeps=substeps,
            snapshot_dir=self._snapshot_dir,
        )
        gen    = torch.Generator(device=device).manual_seed(seed)
        cfg    = self.transformer.config
        latent = torch.randn(
            batch, cfg.latent_channels, cfg.latent_size, cfg.latent_size,
            generator=gen, device=device, dtype=dtype,
        )
        for step in range(steps):
            sigma  = sched.sigmas[step].to(device=device, dtype=dtype)
            in_interval = interval[0] <= float(sigma) <= interval[1]
            if cfg_scale == 1.0 or not in_interval:
                guided = self.transformer(
                    latent, sigma.expand(batch), text_emb, attn_mask
                )
            else:
                lb     = latent.repeat(2, 1, 1, 1)
                tb     = sigma.expand(batch * 2)
                txtb   = torch.cat([uncond_emb, text_emb], dim=0)
                mskb   = torch.cat([uncond_mask, attn_mask], dim=0)
                uv, cv = self.transformer(lb, tb, txtb, mskb).chunk(2)
                guided = uv + cfg_scale * (cv - uv)
            if cfg_rescale > 0.0 and cfg_scale != 1.0 and in_interval:
                sc     = cv.std(dim=(1, 2, 3), keepdim=True)
                sg     = guided.std(dim=(1, 2, 3), keepdim=True)
                guided = (cfg_rescale * guided * (sc / sg.clamp_min(1e-5))
                          + (1.0 - cfg_rescale) * guided)
            latent = sched.step(guided, sigma, latent, return_dict=True).prev_sample
        if track_vram:
            run_stats.peak_denoise_gb = _cuda_peak_gb()

        # ── Phase 3: decode ────────────────────────────────────────────────────
        if track_vram:
            _reset_cuda_peak()
        self._ensure_vae()
        self.vae.to(device)
        latent_dec = _denorm(latent, self.latent_mean, self.latent_std) / self.scaling_factor
        decoded    = self.vae.decode(latent_dec, return_dict=False)[0]
        pixels     = (decoded.float() / 2.0 + 0.5).clamp(0, 1)
        images = [
            Image.fromarray(
                (pixels[i].permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
            )
            for i in range(batch)
        ]
        if track_vram:
            run_stats.peak_decode_gb = _cuda_peak_gb()
        self.last_run_stats = run_stats
        return BoomerOutput(images=images)
