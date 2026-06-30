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


# ── STORK scheduler (Tan et al. 2025, arXiv:2505.24210; solver_order=2 → STORK-2) ─
def _find_stork_file(snapshot_dir: str | None = None) -> Path:
    """Search for STORKScheduler.py across all known locations."""
    candidates: list[Path] = []

    # 1. Snapshot directory passed in from from_pretrained
    if snapshot_dir:
        candidates.append(Path(snapshot_dir) / "STORKScheduler.py")

    # 2. Same directory as this pipeline file (diffusers modules cache)
    candidates.append(Path(__file__).resolve().parent / "STORKScheduler.py")

    # 3. Entire HF hub cache — covers all downloaded model snapshots
    hub_root = Path.home() / ".cache" / "huggingface" / "hub"
    if hub_root.exists():
        candidates.extend(hub_root.rglob("STORKScheduler.py"))

    # 4. Common Colab / local paths
    candidates += [
        Path("/content/STORK/STORKScheduler.py"),
        Path("/content/Boomer/STORKScheduler.py"),
        Path("/content/Boomer/STORK/STORKScheduler.py"),
    ]

    for p in candidates:
        if p.is_file():
            return p

    # Last resort: download directly from the model repo on HuggingFace
    try:
        from huggingface_hub import hf_hub_download
        downloaded = hf_hub_download("akrao9/Boomer-T2I", "STORKScheduler.py")
        return Path(downloaded)
    except Exception:
        pass

    raise FileNotFoundError(
        "STORKScheduler.py not found. Either:\n"
        "  !git clone https://github.com/ZT220501/STORK.git /content/STORK\n"
        "or ensure the model repo is accessible and retry."
    )


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
    spec   = importlib.util.spec_from_file_location("STORKScheduler", stork_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]

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


# ── text encoding ──────────────────────────────────────────────────────────────
@torch.inference_mode()
def _encode_prompts(tokenizer, text_encoder, prompts, max_length, device, dtype, instruction=None):
    # Qwen readout (instruction set): reproduce the training QwenReferenceEncoder exactly —
    # prepend the instruction, render through the chat template, then read the decoder's
    # post-final-norm last_hidden_state, right-padded/truncated to max_length. This MUST
    # match how the CaptionEmbedder was trained, or prompt-following degrades badly.
    if instruction:
        rendered = []
        for prompt in prompts:
            messages = [{"role": "user", "content": f"{instruction}\n\n{prompt}"}]
            rendered.append(
                tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            )
        # Qwen3.5 is a VL model: its AutoProcessor.__call__ is (images, text, ...), so the
        # text MUST be passed as text=; positionally it gets treated as images and the
        # processor tries to fetch_images() the chat string. Fall back to positional for
        # plain tokenizers that don't accept text=.
        tok_kwargs = dict(max_length=max_length, padding="max_length",
                          truncation=True, return_tensors="pt")
        try:
            tokens = tokenizer(text=rendered, **tok_kwargs)
        except TypeError:
            tokens = tokenizer(rendered, **tok_kwargs)
        ids  = tokens["input_ids"].to(device)
        mask = tokens["attention_mask"].to(device)
        out  = text_encoder(ids, attention_mask=mask, output_hidden_states=True,
                            use_cache=False, return_dict=True)
        hidden = out.last_hidden_state
        # Right-padding keeps valid tokens at the front, so a head slice is safe.
        return hidden[:, :max_length].to(dtype=dtype), mask[:, :max_length]

    # Legacy (no instruction, e.g. gemma): plain tokenize + last_hidden_state.
    token_kwargs = dict(max_length=max_length, padding="max_length",
                        truncation=True, return_tensors="pt")
    try:
        tokens = tokenizer(text=prompts, **token_kwargs)
    except TypeError:
        tokens = tokenizer(prompts, **token_kwargs)

    ids  = tokens["input_ids"].to(device)
    mask = tokens["attention_mask"].to(device)
    out  = text_encoder(ids, attention_mask=mask)
    hidden = out[0] if isinstance(out, tuple) else out.last_hidden_state

    idx = [0] + list(range(-max_length + 1, 0))
    return hidden[:, idx].to(dtype=dtype), mask[:, idx]


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
        # Instruction prefix for the Qwen readout (None => legacy plain-tokenize path).
        self.instruction     = cfg.get("_instruction",     None)
        self._vae_repo       = cfg.get("_vae_repo",  "mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers")
        self._te_repo        = cfg.get("_te_repo",   "google/gemma-4-E2B-it")
        self._hf_token       = None
        self._snapshot_dir   = getattr(transformer, "_snapshot_dir", None)
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
        from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
        dtype = next(self.transformer.parameters()).dtype
        logger.info(f"Loading text encoder from {self._te_repo} ...")
        try:
            self.tokenizer = AutoProcessor.from_pretrained(self._te_repo, token=self._hf_token)
        except (ImportError, OSError, ValueError):
            self.tokenizer = AutoTokenizer.from_pretrained(self._te_repo, token=self._hf_token)
        if hasattr(self.tokenizer, "padding_side"):
            self.tokenizer.padding_side = "right"
        elif hasattr(self.tokenizer, "tokenizer"):
            self.tokenizer.tokenizer.padding_side = "right"
        te = AutoModelForCausalLM.from_pretrained(
            self._te_repo, torch_dtype=dtype, token=self._hf_token,
        )
        self.text_encoder = (te.get_decoder() if hasattr(te, "get_decoder") else te).to(
            self._execution_device
        ).eval()

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
        steps: int = 32,
        seed: int | None = None,
        cfg_scale: float = 4.0,
        cfg_rescale: float = 0.5,
        substeps: int = 5,
        offload_text_encoder: bool = True,
        **kwargs,
    ) -> BoomerOutput:
        """
        Parameters
        ----------
        prompt       : str or list[str]
        steps        : denoising steps (32 recommended with STORK-2)
        seed         : random seed — None (default) picks a random seed each call
        cfg_scale    : classifier-free guidance scale (4.0 recommended)
        cfg_rescale  : CFG variance rescale (0.5 recommended)
        substeps     : STORK-2 internal RK micro-steps
        offload_text_encoder : unload text encoder after encoding to free VRAM
        """
        import random
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
        text_emb, attn_mask = _encode_prompts(
            self.tokenizer, self.text_encoder, prompts,
            self.max_text_length, device, dtype, instruction=self.instruction,
        )
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
            solver_order=2, derivative_order=1, substeps=substeps,
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
            lb     = latent.repeat(2, 1, 1, 1)
            tb     = sigma.expand(batch * 2)
            txtb   = torch.cat([uncond_emb, text_emb], dim=0)
            mskb   = torch.cat([uncond_mask, attn_mask], dim=0)
            uv, cv = self.transformer(lb, tb, txtb, mskb).chunk(2)
            guided = uv + cfg_scale * (cv - uv)
            if cfg_rescale > 0.0:
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
