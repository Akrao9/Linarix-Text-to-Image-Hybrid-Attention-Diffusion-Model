"""Qwen3.5-2B -> Boomer conditioning connector.

A frozen, multimodal **Qwen/Qwen3.5-2B** (Apache-2.0) encodes a prompt together
with an optional reference image. A trained **Perceiver resampler** then maps its
hidden states into a fixed number of conditioning tokens shaped exactly like the
Gemma embeddings Boomer was designed around -- ``[B, N, text_dim]`` plus an
all-ones mask -- so :class:`BoomerFLADiT` consumes them with **no changes**.

Design notes
------------
* The encoder is frozen; only the resampler (and optionally the DiT) is trained.
* The resampler owns all the learnable-query logic, so the only thing required of
  Qwen is ``output_hidden_states=True`` -- a stable API across HF VLMs, no
  model-internal surgery.
* The encoder forward is separated from the resampler so Qwen hidden states can be
  **cached offline** during training (run Qwen once per image, never in the loop).

Provenance: the *generator* (DiT) is what carries the "trained only on licensed
data" claim. Qwen3.5-2B (Apache-2.0) and DC-AE (MIT) are licensed *tools*; ship a
NOTICE/ATTRIBUTIONS file crediting both.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch import nn

from boomer.models.boomer_fla_dit import BoomerFLADiT, BoomerFLADiTConfig

DEFAULT_QWEN_ID = "Qwen/Qwen3.5-2B"

# Steering prefix: shapes which attributes Qwen's hidden states emphasise. This is
# the main lever for getting identity/material/geometry into the conditioning.
DEFAULT_INSTRUCTION = (
    "You are an image-generation conditioning encoder. Attend to the prompt and "
    "any reference image and represent the subject's identity, the materials and "
    "surface textures, the 3D geometry and pose, and the overall composition."
)


@dataclass
class QwenConnectorConfig:
    qwen_model_id: str = DEFAULT_QWEN_ID
    out_dim: int = 1536  # must equal BoomerFLADiTConfig.text_dim
    num_query_tokens: int = 256
    connector_depth: int = 4
    connector_heads: int = 12
    connector_mlp_ratio: float = 4.0
    readout_layer: int = -1  # which Qwen hidden_states layer to resample (-1 = last)
    freeze_encoder: bool = True
    instruction: str | None = DEFAULT_INSTRUCTION
    max_text_tokens: int = 512
    # Pad every batch to exactly max_text_tokens (static shape) instead of to the
    # batch's longest caption. Wastes a little compute on pad tokens but gives
    # torch.compile a single shape to specialize, avoiding recompile thrash.
    pad_to_max_length: bool = False
    dtype: str = "bf16"
    attn_implementation: str | None = None


def _resolve_dtype(dtype: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype.lower()]


class QwenReferenceEncoder(nn.Module):
    """Frozen Qwen3.5-2B wrapper returning hidden states for the resampler.

    ``forward`` returns ``(hidden_states [B, L, D_qwen], pad_mask [B, L])`` where
    ``pad_mask`` is 1 for valid tokens. The model is loaded lazily so the module is
    cheap to construct (e.g. when only the resampler is being trained from cache).
    """

    def __init__(self, config: QwenConnectorConfig, hf_token: str | bool | None = None) -> None:
        super().__init__()
        self.config = config
        self._hf_token = hf_token
        self._model: nn.Module | None = None
        # Base transformer (the submodule that owns the decoder layers). Calling it
        # directly for hidden-state readout skips the lm_head matmul (B*L*vocab FLOPs)
        # that the full causal-LM forward would otherwise compute and discard.
        self._base_model: nn.Module | None = None
        self._base_failed = False
        self._processor: Any | None = None
        self.hidden_size: int | None = None
        # True unless we truncate to an intermediate layer. When reading the genuine
        # final layer we take the post-final-norm output (last_hidden_state), which is
        # stable across transformers versions; relying on hidden_states[-1] being normed
        # is NOT (a version change silently shifted the readout scale ~700x).
        self._readout_is_final = True
        # Set once we truncate: the readout is then the LAST layer of the truncated
        # stack and the final norm is an Identity, so last_hidden_state is bit-identical
        # to hidden_states[readout_layer] and output_hidden_states becomes unnecessary.
        self._truncated_to_last = False

    # -- lazy load -----------------------------------------------------------
    def load(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoProcessor, AutoTokenizer

        dtype = _resolve_dtype(self.config.dtype)
        try:
            self._processor = AutoProcessor.from_pretrained(self.config.qwen_model_id, token=self._hf_token)
        except Exception:
            self._processor = AutoTokenizer.from_pretrained(self.config.qwen_model_id, token=self._hf_token)
        # Right-pad so valid tokens are contiguous at the front; this makes
        # fixed-length truncation/padding for caching safe (see encode_qwen_text_fixed).
        if hasattr(self._processor, "padding_side"):
            self._processor.padding_side = "right"
        if hasattr(self._processor, "tokenizer") and hasattr(self._processor.tokenizer, "padding_side"):
            self._processor.tokenizer.padding_side = "right"

        model = None
        last_err: Exception | None = None
        # Text-only Qwen hidden-state caching should prefer the causal-LM path.
        # Keep multimodal/plain fallbacks so the same wrapper still works for
        # processor-backed Qwen variants with image support.
        for loader_name in (
            "AutoModelForCausalLM",
            "AutoModelForImageTextToText",
            "AutoModelForVision2Seq",
            "AutoModelForMultimodalLM",
            "AutoModel",
        ):
            try:
                import transformers

                loader = getattr(transformers, loader_name)
            except AttributeError:
                continue
            try:
                # transformers v5 renamed `torch_dtype` -> `dtype`; prefer the new
                # kwarg and fall back for older versions (matches sana_latent_cache).
                common_kwargs = {"token": self._hf_token}
                if self.config.attn_implementation:
                    common_kwargs["attn_implementation"] = self.config.attn_implementation
                try:
                    model = loader.from_pretrained(
                        self.config.qwen_model_id, dtype=dtype, **common_kwargs
                    )
                except TypeError:
                    try:
                        model = loader.from_pretrained(
                            self.config.qwen_model_id, torch_dtype=dtype, **common_kwargs
                        )
                    except TypeError:
                        # Some model classes do not accept attn_implementation.
                        common_kwargs.pop("attn_implementation", None)
                        model = loader.from_pretrained(
                            self.config.qwen_model_id, torch_dtype=dtype, **common_kwargs
                        )
                break
            except Exception as exc:  # noqa: BLE001 - keep trying loaders
                last_err = exc
        if model is None:
            raise RuntimeError(f"Could not load {self.config.qwen_model_id}: {last_err!r}")

        if self.config.freeze_encoder:
            model.eval()
            model.requires_grad_(False)
        self._model = model
        self.hidden_size = int(
            getattr(getattr(model, "config", None), "hidden_size", 0)
            or getattr(getattr(model, "config", None), "text_config", object()).hidden_size
        )
        # Drop every transformer layer above the readout layer: hidden_states[readout]
        # is unchanged by layers that run after it, so computing them is pure waste.
        self._truncate_to_readout(model)
        # Prefer the base transformer for forward() so the lm_head is never run.
        self._base_model = self._find_base_with_layers(model)
        # Capture the genuine final norm so a final-layer readout can be normed even on
        # the full-model fallback path (CausalLM output has no last_hidden_state). Only
        # meaningful when not truncated (truncation replaces the norm with Identity).
        self._final_norm: nn.Module | None = None
        if self._readout_is_final:
            holder = self._find_base_with_layers(model)
            for norm_attr in ("norm", "final_layernorm", "ln_f"):
                norm = getattr(holder, norm_attr, None) if holder is not None else None
                if norm is not None and not isinstance(norm, nn.Identity):
                    self._final_norm = norm
                    break

    @staticmethod
    def _find_base_with_layers(model: nn.Module) -> nn.Module | None:
        """Return the submodule that owns the decoder ``layers`` ModuleList (the base
        transformer), or ``None`` if it can't be located."""
        candidates = [
            getattr(model, "model", None),
            getattr(getattr(model, "model", None), "model", None),
            getattr(model, "language_model", None),
            getattr(getattr(model, "model", None), "language_model", None),
        ]
        for holder in candidates:
            if holder is None or holder is model:
                continue
            layers = getattr(holder, "layers", None)
            if isinstance(layers, nn.ModuleList) and len(layers) > 0:
                return holder
        return None

    def _truncate_to_readout(self, model: nn.Module) -> None:
        holder = self._find_base_with_layers(model)
        if holder is None:
            return
        layers = holder.layers
        n = len(layers)
        # output_hidden_states yields n+1 tensors (embeddings + each layer output).
        r = self.config.readout_layer
        r_abs = r if r >= 0 else (n + 1 + r)
        if not (0 <= r_abs <= n) or r_abs >= n:
            return  # reading the final layer (or out of range): nothing to drop
        self._readout_is_final = False  # intermediate readout -> raw (un-normed) output
        holder.layers = nn.ModuleList(list(layers)[:r_abs])
        self.config.readout_layer = r_abs  # now the last collected hidden state
        # HF applies the base model's final norm only to the LAST hidden_states entry.
        # At an intermediate readout the full model returns the RAW (un-normed) layer
        # output, so neutralize the final norm to keep the truncated readout identical.
        for norm_attr in ("norm", "final_layernorm", "ln_f"):
            if hasattr(holder, norm_attr) and getattr(holder, norm_attr) is not None:
                setattr(holder, norm_attr, nn.Identity())
                break
        cfg = getattr(model, "config", None)
        for cfg_obj in (cfg, getattr(cfg, "text_config", None)):
            if cfg_obj is not None and hasattr(cfg_obj, "num_hidden_layers"):
                try:
                    cfg_obj.num_hidden_layers = r_abs
                except Exception:
                    pass
        self._truncated_to_last = True
        print(f"[qwen] truncated encoder to {r_abs}/{n} layers (readout={r_abs}); upper layers + lm_head skipped.")

    @property
    def model(self) -> nn.Module:
        if self._model is None:
            self.load()
        assert self._model is not None
        return self._model

    @property
    def processor(self) -> Any:
        if self._processor is None:
            self.load()
        return self._processor

    def _build_inputs(self, prompts: Sequence[str], images: Sequence[Any] | None) -> dict[str, torch.Tensor]:
        """Render prompt (+ instruction, + optional image) via the chat template."""
        proc = self.processor
        batch_messages = []
        for i, prompt in enumerate(prompts):
            content: list[dict[str, Any]] = []
            if images is not None and images[i] is not None:
                content.append({"type": "image"})
            text = prompt if not self.config.instruction else f"{self.config.instruction}\n\n{prompt}"
            content.append({"type": "text", "text": text})
            batch_messages.append([{"role": "user", "content": content}])

        texts = [
            proc.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
            for m in batch_messages
        ]
        image_arg = None
        if images is not None and any(im is not None for im in images):
            image_arg = [im for im in images if im is not None]
        input_kwargs: dict[str, Any] = {
            "text": texts,
            "padding": "max_length" if self.config.pad_to_max_length else True,
            "truncation": True,
            "max_length": self.config.max_text_tokens,
            "return_tensors": "pt",
        }
        if image_arg is not None:
            input_kwargs["images"] = image_arg
        try:
            inputs = proc(**input_kwargs)
        except TypeError:
            input_kwargs.pop("text", None)
            inputs = proc(texts, **input_kwargs)
        return inputs

    # no_grad rather than inference_mode: the outputs feed straight into the training
    # graph as constants (train.py encode_captions); inference-mode tensors would
    # force a full [B, L, D] clone there every step.
    @torch.no_grad()
    def forward(
        self, prompts: Sequence[str], images: Sequence[Any] | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        model = self.model
        device = next(model.parameters()).device
        inputs = self._build_inputs(prompts, images)
        inputs = {k: v.to(device) for k, v in inputs.items() if torch.is_tensor(v)}
        # Text-only readout: run the base transformer directly to skip the lm_head.
        # hidden_states[readout] is identical either way (lm_head doesn't feed back).
        text_only = images is None or not any(im is not None for im in images)
        runner = (
            self._base_model
            if (text_only and self._base_model is not None and not self._base_failed)
            else model
        )
        # output_hidden_states makes HF retain EVERY layer output simultaneously
        # (~100 MiB each at B=64/L=320/D=2560 bf16, so multiple GB for ~36 layers) when
        # we read exactly one of them. On the truncated base-model path that one is the
        # last entry, and the final norm is an Identity, so last_hidden_state is
        # bit-identical — request the states only when we actually have to index them.
        # The full model returns CausalLM output with no last_hidden_state, so it always
        # needs the flag (including the fallback retry below).
        want_hidden = not (self._truncated_to_last and runner is not model)
        try:
            out = runner(**inputs, output_hidden_states=want_hidden, use_cache=False, return_dict=True)
        except Exception:
            if runner is not model:
                self._base_failed = True  # fall back to the full model for the rest of the run
                want_hidden = True
                out = model(**inputs, output_hidden_states=True, use_cache=False, return_dict=True)
            else:
                raise
        if not want_hidden:
            # Truncated base-model path: readout == last layer, final norm is Identity.
            hidden_states = out.last_hidden_state
        elif self._readout_is_final and getattr(out, "last_hidden_state", None) is not None:
            # Post-final-norm output: matches the validated cache (abs-max ~72) regardless
            # of whether this transformers version post-norms hidden_states[-1].
            hidden_states = out.last_hidden_state
        elif self._readout_is_final and self._final_norm is not None:
            # Full-model fallback (no last_hidden_state): apply the final norm explicitly.
            hidden_states = self._final_norm(out.hidden_states[-1])
        else:
            hidden_states = out.hidden_states[self.config.readout_layer]
        pad_mask = inputs.get("attention_mask")
        if pad_mask is None:
            pad_mask = torch.ones(hidden_states.shape[:2], device=device, dtype=torch.long)
        return hidden_states, pad_mask


@torch.no_grad()
def encode_qwen_text_fixed(
    encoder: QwenReferenceEncoder,
    captions: Sequence[str],
    max_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Text-only Qwen encoding padded/truncated to a fixed ``max_length``.

    Returns ``(hidden [B, max_length, D_qwen], mask [B, max_length])`` suitable for
    fixed-shape WebDataset caching. Uses the same :class:`QwenReferenceEncoder` as
    inference, so cached states match the live connector path exactly. Right-padding
    (set in ``load``) keeps valid tokens at the front, so tail pad/truncate is safe.
    """
    hidden, mask = encoder(list(captions), images=None)
    batch, length, dim = hidden.shape
    if length >= max_length:
        return hidden[:, :max_length].contiguous(), mask[:, :max_length].contiguous()
    hidden = torch.cat([hidden, hidden.new_zeros(batch, max_length - length, dim)], dim=1)
    mask = torch.cat([mask, mask.new_zeros(batch, max_length - length)], dim=1)
    return hidden.contiguous(), mask.contiguous()


class _ResamplerBlock(nn.Module):
    """Cross-attend queries -> encoder states, then self-attend, then FFN."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float) -> None:
        super().__init__()
        self.norm_q1 = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm_q2 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm_ff = nn.LayerNorm(dim)
        inner = int(dim * mlp_ratio)
        self.ff = nn.Sequential(nn.Linear(dim, inner), nn.GELU(approximate="tanh"), nn.Linear(inner, dim))

    def forward(self, q: torch.Tensor, kv: torch.Tensor, key_padding_mask: torch.Tensor | None) -> torch.Tensor:
        kv_n = self.norm_kv(kv)
        q = q + self.cross_attn(
            self.norm_q1(q), kv_n, kv_n, key_padding_mask=key_padding_mask, need_weights=False
        )[0]
        sq = self.norm_q2(q)
        q = q + self.self_attn(sq, sq, sq, need_weights=False)[0]
        q = q + self.ff(self.norm_ff(q))
        return q


class PerceiverConnector(nn.Module):
    """Learnable queries resample Qwen hidden states into ``[B, N, out_dim]`` tokens."""

    def __init__(self, config: QwenConnectorConfig, encoder_hidden_size: int) -> None:
        super().__init__()
        self.config = config
        dim = config.out_dim
        self.in_proj = nn.Linear(encoder_hidden_size, dim)
        self.queries = nn.Parameter(torch.randn(1, config.num_query_tokens, dim) * 0.02)
        self.blocks = nn.ModuleList(
            _ResamplerBlock(dim, config.connector_heads, config.connector_mlp_ratio)
            for _ in range(config.connector_depth)
        )
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, encoder_hidden: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch = encoder_hidden.shape[0]
        kv = self.in_proj(encoder_hidden.to(self.in_proj.weight.dtype))
        q = self.queries.expand(batch, -1, -1)
        key_padding_mask = None
        if pad_mask is not None:
            key_padding_mask = pad_mask == 0  # True == ignore
        for block in self.blocks:
            q = block(q, kv, key_padding_mask)
        return self.out_norm(q)


class ReferenceConditionedBoomer(nn.Module):
    """Qwen3.5-2B encoder + Perceiver connector + BoomerFLADiT (from-scratch generator).

    Training contract mirrors ``BoomerFLADiT.forward`` but takes raw prompts/images
    (or cached Qwen hidden states) instead of precomputed text embeddings.
    """

    def __init__(
        self,
        connector_config: QwenConnectorConfig,
        dit_config: BoomerFLADiTConfig,
        hf_token: str | bool | None = None,
    ) -> None:
        super().__init__()
        if connector_config.out_dim != dit_config.text_dim:
            raise ValueError(
                f"connector out_dim={connector_config.out_dim} must equal "
                f"dit text_dim={dit_config.text_dim}"
            )
        self.connector_config = connector_config
        self.encoder = QwenReferenceEncoder(connector_config, hf_token=hf_token)
        # Defer connector construction until the encoder hidden size is known.
        self._connector: PerceiverConnector | None = None
        self.dit = BoomerFLADiT(dit_config)

    def build_connector(self, encoder_hidden_size: int | None = None) -> None:
        if self._connector is not None:
            return
        if encoder_hidden_size is None:
            self.encoder.load()
            encoder_hidden_size = self.encoder.hidden_size
        if not encoder_hidden_size:
            raise RuntimeError("encoder hidden size unknown; load the encoder or pass it explicitly")
        self._connector = PerceiverConnector(self.connector_config, encoder_hidden_size)

    @property
    def connector(self) -> PerceiverConnector:
        if self._connector is None:
            self.build_connector()
        assert self._connector is not None
        return self._connector

    def encode_condition(
        self,
        prompts: Sequence[str] | None = None,
        images: Sequence[Any] | None = None,
        *,
        cached_hidden: torch.Tensor | None = None,
        cached_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return conditioning tokens ``[B, N, out_dim]`` for the DiT."""
        if cached_hidden is not None:
            return self.connector(cached_hidden, cached_mask)
        if prompts is None:
            raise ValueError("provide prompts (+optional images) or cached_hidden")
        hidden, mask = self.encoder(prompts, images)
        return self.connector(hidden, mask)

    def forward(
        self,
        noisy_latent: torch.Tensor,
        timesteps: torch.Tensor,
        prompts: Sequence[str] | None = None,
        images: Sequence[Any] | None = None,
        *,
        cached_hidden: torch.Tensor | None = None,
        cached_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        cond = self.encode_condition(
            prompts, images, cached_hidden=cached_hidden, cached_mask=cached_mask
        )
        attention_mask = torch.ones(
            cond.shape[0], cond.shape[1], device=cond.device, dtype=torch.long
        )
        return self.dit(noisy_latent, timesteps, cond, attention_mask)
