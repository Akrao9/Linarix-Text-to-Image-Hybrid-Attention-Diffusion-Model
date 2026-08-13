"""Pre-encode a fixed set of eval prompts into a tensor file for in-training previews.

Run this ONCE before training. It loads the Qwen text encoder, encodes the prompts
with the SAME settings the latent cache used (encoder id, max_length, instruction),
and saves the embeddings + masks to a .pt. The training run's SamplePreviewCallback
then loads that file and never has to keep Qwen resident — keeping the text encoder
out of the training VRAM budget.

Example:
    python tools/encode_prompts.py --out prompts_cond.pt \
        --text-encoder-id Qwen/Qwen3.5-2B --max-length 320 --hf-token true
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from boomer.models.qwen_connector import (  # noqa: E402
    QwenConnectorConfig,
    QwenReferenceEncoder,
    encode_qwen_text_fixed,
)
from boomer.sana_latent_cache import CacheConfig, hf_token_value  # noqa: E402

# A fixed, diverse eval set: an articulated quadruped (the known structure stressor),
# a quadruped in motion, a human face, and a rigid object in a lit scene. Keeping these
# constant lets you watch the SAME four images evolve across the whole run on W&B.
DEFAULT_PROMPTS = [
    "a golden retriever sitting in a garden, soft morning sunlight",
    "a brown horse galloping across a green meadow",
    "a close-up portrait of an elderly fisherman with a weathered face",
    "a red vintage sports car on a coastal road at sunset",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-encode eval prompts for in-training previews.")
    parser.add_argument("--out", required=True, help="Output .pt path (e.g. prompts_cond.pt).")
    parser.add_argument("--prompt", action="append", dest="prompts",
                        help="Prompt to encode; repeatable. Defaults to the built-in eval set.")
    parser.add_argument("--text-encoder-id", default=CacheConfig.text_encoder_id)
    parser.add_argument("--max-length", type=int, default=CacheConfig.max_length)
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hf-token", help="Hugging Face token, or 'true' to use the logged-in token.")
    args = parser.parse_args()

    prompts = args.prompts or DEFAULT_PROMPTS
    device = args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    hf_token = hf_token_value(args.hf_token)

    if "qwen" not in args.text_encoder_id.lower():
        raise SystemExit(f"This tool only supports Qwen text encoders, got {args.text_encoder_id}")

    qwen_config = QwenConnectorConfig(
        qwen_model_id=args.text_encoder_id,
        readout_layer=-1,
        dtype=args.dtype,
        instruction=QwenConnectorConfig.instruction,
        max_text_tokens=args.max_length,
        # False to match training: encode_qwen_text_fixed zero-pads to max_length
        # afterwards, so the saved cache keeps the same fixed shape either way.
        pad_to_max_length=False,
    )
    encoder = QwenReferenceEncoder(qwen_config, hf_token=hf_token)
    encoder.load()
    encoder.to(device)

    embeddings, masks = [], []
    for prompt in prompts:
        text_embedding, attention_mask = encode_qwen_text_fixed(encoder, [prompt], args.max_length)
        embeddings.append(text_embedding.squeeze(0).cpu())
        masks.append(attention_mask.squeeze(0).cpu())
        print(f"encoded: {prompt!r} -> embedding {tuple(embeddings[-1].shape)}", flush=True)

    payload = {
        "prompts": list(prompts),
        "text_embeddings": torch.stack(embeddings, dim=0),  # [N, L, D]
        "attention_masks": torch.stack(masks, dim=0),       # [N, L]
        "text_encoder_id": args.text_encoder_id,
        "max_length": args.max_length,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    print(f"saved {len(prompts)} encoded prompts -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
