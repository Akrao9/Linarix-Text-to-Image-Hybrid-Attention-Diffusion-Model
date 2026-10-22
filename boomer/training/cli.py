"""Command-line options for Linarix training."""

from __future__ import annotations

import argparse

from boomer.datasets.monet_direct import (
    MONET_CAPTION_KEYS,
    MONET_LATENT_KEY,
)
from boomer.datasets.fine_t2i import DEFAULT_T2I_REPO
from boomer.sana_latent_cache import (
    DEFAULT_SCALING_FACTOR,
    DEFAULT_VAE_ID,
    CacheConfig,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Lightning trainer for Boomer FLA (Qwen-conditioned, from scratch).")
    bo = argparse.BooleanOptionalAction

    # data
    p.add_argument("--cache-wds", default=None,
                   help="WDS glob, e.g. hf://datasets/<repo>/chunk_*/*.tar. Required unless "
                        "--t2i-direct (which streams from --t2i-repo instead).")
    p.add_argument("--wds-local-dir", default="/content/_wds_cache")
    p.add_argument("--wds-prefetch-shards", type=int, default=8,
                   help="hf:// caches: tars to download in parallel ahead of the GPU "
                        "(disk window = this x shard size). Same prefetch as the caching tool.")
    p.add_argument("--wds-download-workers", type=int, default=4,
                   help="hf:// caches: parallel shard-download threads.")
    p.add_argument("--wds-decode-queue", type=int, default=512,
                   help="hf:// caches: bounded record queue for the background decode "
                        "thread (overlaps sample reconstruction with the GPU step at "
                        "num_workers=0). Set 0 to decode inline on the consumer thread.")
    p.add_argument("--wds-download-max-retries", type=int, default=2,
                   help="hf:// caches: re-download a stalled shard up to N times (the stuck "
                        "worker is replaced, not killed) before skipping it.")
    p.add_argument("--wds-max-consecutive-skips", type=int, default=8,
                   help="hf:// caches: abort the run after this many back-to-back shard "
                        "skips, so a dead network fails loudly instead of silently training "
                        "on a shrinking data slice.")
    p.add_argument("--wds-rolling-shards", type=int, default=2,
                   help="(deprecated alias for --wds-prefetch-shards; ignored)")
    p.add_argument("--wds-shuffle-buffer", type=int, default=2000)
    p.add_argument("--wds-shard-shuffle", type=int, default=0,
                   help="LOCAL streaming only: shard shuffle buffer/count. 0 disables.")
    p.add_argument("--num-workers", type=int, default=4,
                   help="DataLoader workers for LOCAL shard globs (forced to 0 for hf:// prefetch cache).")
    p.add_argument("--pin-memory", action=bo, default=True,
                   help="Pin CPU batches for faster async GPU transfer (CUDA only).")
    p.add_argument("--prefetch-factor", type=int, default=2,
                   help="Batches prefetched per worker when --num-workers > 0.")
    p.add_argument("--stream-block-size-mib", type=int, default=1,
                   help="hf:// range-read block for the image-streaming loaders "
                        "(monet-image-direct / t2i-direct). Each cache miss requests "
                        "this much in one Range GET. Above ~2 MiB, HF truncates the "
                        "response under many parallel streams and huggingface_hub "
                        "re-fetches the whole range with backoff, stalling the run.")
    p.add_argument("--max-screened-per-shard", type=str, default=None,
                   help="Consecutive filtered rows a shard may produce before the "
                        "streaming loaders move on. One value for all sources, or a "
                        "comma list aligned with --t2i-data-dir / --cache-wds. 0 reads "
                        "shards to the end. The default (64) screens a FIXED window at "
                        "each shard's head, so a source whose usable rows are sparse "
                        "there yields the same few rows every pass no matter the seed "
                        "-- measured on fine-t2i curated at --t2i-min-short-side 1024. "
                        "Set 0 for such a source; leave the default where the filter "
                        "passes most rows, since 0 removes the cost bound.")
    p.add_argument("--persistent-workers", action=bo, default=None,
                   help="Keep DataLoader workers alive between epochs. Default: on when --num-workers > 0.")
    p.add_argument("--parquet-read-batch-size", type=int, default=32,
                   help="Rows materialized by Arrow at once per Parquet worker. The cache has "
                        "large 512-row groups; 32 bounds transient host RAM without changing "
                        "the DataLoader batch size.")
    p.add_argument("--parquet-memory-trim-every", type=int, default=4096,
                   help="Parquet rows per worker between best-effort Arrow/libc memory "
                        "release calls. 0 disables; default prevents long-run RSS growth.")
    p.add_argument("--parquet-prefetch", dest="parquet_prefetch_ranges", action=bo, default=True,
                   help="Background-prefetch the next Parquet byte range while the current one "
                        "decodes (pyarrow CacheOptions). Skipped automatically on datasets/"
                        "pyarrow versions without fragment_scan_options support.")
    p.add_argument("--parquet-range-mib", dest="parquet_range_size_mib", type=int, default=64,
                   help="Per-request Parquet range size in MiB (pyarrow default 32). Larger = "
                        "fewer requests but ~2x this buffered per worker with prefetch on; "
                        "0 keeps the pyarrow default.")
    p.add_argument("--hf-token", default=None, help="HF token, or 'true' for the logged-in token.")

    # Direct jasperai/monet streaming (no cache build): --cache-wds points at raw monet
    # parquet globs (e.g. hf://datasets/jasperai/monet/v1.2.0/cc12m/*/*.parquet), the
    # precomputed SANA f32c32 latents are used as-is, and captions are encoded on-GPU
    # every step by a resident frozen Qwen encoder (see --text-encoder-*).
    p.add_argument("--monet-direct", action=bo, default=False,
                   help="Stream jasperai/monet parquet shards directly: latents from the "
                        "dataset, text encoded on the fly (skips the caching pipeline).")
    p.add_argument("--monet-image-direct", action=bo, default=False,
                   help="Stream MONET WebDataset tar shards directly: full-resolution jpg "
                        "pixels are DC-AE encoded on the fly and json captions are encoded "
                        "by Qwen. --cache-wds must point at comma-separated MONET *.tar "
                        "globs; --monet-weights/caption-probs retain their usual meaning.")
    p.add_argument("--monet-image-resolution", type=int, default=1024,
                   help="Square output resolution for --monet-image-direct (resize short "
                        "side, then center crop).")
    p.add_argument("--monet-image-min-short-side", type=int, default=0,
                   help="Drop native MONET images below this short-side size before resize; "
                        "0 keeps and upscales them, matching nano-t2i's 1024 recipe.")
    p.add_argument("--cache-captions-only", action=bo, default=False,
                   help="Boomer WDS/parquet caches: IGNORE the cached text/text_mask "
                        "columns (parquet never even downloads them) and encode the raw "
                        "caption column on the fly with --text-encoder-*. For caches whose "
                        "text was baked with a different encoder than the current model, "
                        "e.g. boomer-finet2i-1024-parquet's 2048-d text vs v6's 1536-d.")
    p.add_argument("--monet-latent-key", default=MONET_LATENT_KEY)
    p.add_argument("--monet-caption-keys", default=",".join(MONET_CAPTION_KEYS),
                   help="comma-ordered caption-column priority; first non-empty per row wins "
                        "(unless --monet-caption-probs samples among them instead).")
    p.add_argument("--monet-caption-probs", default=None,
                   help="Caption sampling weights aligned 1:1 with --monet-caption-keys: each "
                        "row picks ONE caption at random among its non-empty columns with "
                        "these relative probabilities (nano-t2i captioners_probabilities); "
                        "zero-weight keys are never used. With --monet-weights, ';'-separated "
                        "groups give each subset glob its own policy (nano_mix style; an "
                        "empty group = priority order). Unset = strict priority order.")
    p.add_argument("--monet-weights", default=None,
                   help="CSV weights aligned 1:1 with the comma-separated globs in --cache-wds: "
                        "each batch row is drawn from glob i with probability w_i/sum(w) "
                        "(row-level stream interleave; nano_mix.yaml-style subset proportions). "
                        "Unset = all globs merged into one stream.")
    p.add_argument("--monet-dino", action=bo, default=True,
                   help="carry monet's DINOv2 embedding into batches (REPA target).")
    p.add_argument("--monet-min-aesthetic", type=float, default=None)
    p.add_argument("--monet-max-nsfw", type=float, default=None)
    p.add_argument("--monet-max-watermark", type=float, default=None)

    # Editing phases: E1 self-supervised inpaint/fill over any latent cache, and
    # E2 instruction editing over pixel-pair corpora (OmniEdit). Both condition the
    # model's reserved input channels (--extra-input-channels = latent_channels + 1).
    p.add_argument("--fill-training", action=bo, default=False,
                   help="E1: synthesize inpainting masks over the latent stream and train "
                        "with cond_input=[x0*(1-m), m] (FLUX-Fill style). Works with any "
                        "cache/monet stream; requires --extra-input-channels = "
                        "latent_channels + 1.")
    p.add_argument("--fill-mask-probs", default=None,
                   help="mask-type mix, e.g. 'blob=0.3,rect=0.25,outpaint=0.15,full=0.1,"
                        "tiny=0.2' (normalized; that mix is the default). 'full' = pure-t2i "
                        "retention samples under the fill interface.")
    p.add_argument("--cond-input-dropout", type=float, default=0.1,
                   help="per-sample probability of zeroing the WHOLE cond_input block "
                        "(exactly the zero-fed t2i pretrain state) -> retains pure t2i and "
                        "enables image-side CFG (--sample-image-cfg / IP2P dual guidance). "
                        "Independent of --cond-dropout on the text; only applies when a "
                        "cond_input exists (fill/edit phases). Set 0 to disable.")
    p.add_argument("--edit-direct", action=bo, default=False,
                   help="E2: stream OmniEdit-style parquet pixel pairs directly; both sides "
                        "are DC-AE-encoded on the fly, source latent + all-ones mask become "
                        "cond_input, the instruction is the caption (encoded by "
                        "--text-encoder-*). No cache build.")
    p.add_argument("--edit-resolution", type=int, default=1024,
                   help="square training resolution for the edit pair (edit-aware shared "
                        "crop, then resize).")
    p.add_argument("--edit-min-o-score", type=float, default=None,
                   help="drop rows below this o_score (OmniEdit-Filtered is already >= 9).")
    p.add_argument("--edit-tasks", default=None,
                   help="CSV whitelist of OmniEdit task names; unset = all tasks.")
    p.add_argument("--edit-prompt-pick", choices=["random", "first", "last"], default="random",
                   help="which entry of edited_prompt_list to train on; 'random' samples "
                        "per pass (short vs detailed phrasing augmentation).")
    p.add_argument("--edit-min-short-side", type=int, default=0,
                   help="drop pairs whose short side is below this many pixels (0 = keep "
                        "all; OmniEdit's minimum is 768).")
    p.add_argument("--edit-max-aspect-mismatch", type=float, default=1.05,
                   help="max source/target aspect-ratio disparity before a pair is dropped "
                        "instead of registered (target center-cropped to the source aspect). "
                        "1.05 tolerates resize jitter only; widen past ~1.1 ONLY if a probe "
                        "shows genuinely reconcilable wider mismatches.")
    p.add_argument("--edit-encode-chunk", type=int, default=8,
                   help="images per DC-AE encode chunk in training_step (bounds encode VRAM).")
    p.add_argument("--edit-vae-id", default=DEFAULT_VAE_ID,
                   help="DC-AE used for on-the-fly pixel encoding (must match the VAE the "
                        "t2i latents were cached with).")
    p.add_argument("--edit-scaling-factor", type=float, default=DEFAULT_SCALING_FACTOR,
                   help="latent scaling factor for on-the-fly encoding (cache parity). Also "
                        "applies to --t2i-direct pixel encoding.")
    p.add_argument("--t2i-direct", action=bo, default=False,
                   help="Plain t2i on the fly: stream a single-image WebDataset (default "
                        "ma-xu/fine-t2i), DC-AE-encode each image on-GPU, caption encoded by "
                        "--text-encoder-*. No cache build, no cond_input. Uses "
                        "--edit-vae-id/--edit-scaling-factor/--edit-encode-chunk for the encode.")
    p.add_argument("--t2i-repo", default=DEFAULT_T2I_REPO,
                   help="HF dataset repo id to stream for --t2i-direct (WebDataset format).")
    p.add_argument("--t2i-config", default=None,
                   help="dataset config name (unset = default config).")
    p.add_argument("--t2i-data-dir", default=None,
                   help="subfolder of shards to stream, e.g. "
                        "synthetic_enhanced_prompt_square_resolution (square 1024 canvases "
                        "with enhanced prompts — no crop loss). COMMA-SEPARATED to mix "
                        "several, e.g. 'synthetic_enhanced_prompt_square_resolution,curated'; "
                        "weight them with --t2i-weights. Note fine-t2i's json carries BOTH "
                        "'prompt' and 'enhanced_prompt' per row, so the enhanced folder alone "
                        "already feeds --t2i-caption-mode sample — the original_prompt folder "
                        "is redundant.")
    p.add_argument("--t2i-weights", default=None,
                   help="comma-separated mixing weights aligned 1:1 with --t2i-data-dir "
                        "entries (e.g. '0.75,0.25'). Unset = natural proportions, where a "
                        "small subset is swamped: fine-t2i's curated (168k real photographs) "
                        "lands near 10% against the 1.5M synthetic renders unless weighted up.")
    p.add_argument("--t2i-split", default="train", help="dataset split to stream.")
    p.add_argument("--t2i-resolution", type=int, default=1024,
                   help="square training resolution (short side resized, center-cropped).")
    p.add_argument("--t2i-caption-mode", choices=["sample", "short", "enhanced"], default="sample",
                   help="'sample' picks per-row between the short (json.prompt/txt) and enhanced "
                        "(json.enhanced_prompt) caption; 'short'/'enhanced' pin one field.")
    p.add_argument("--t2i-short-prob", type=float, default=0.5,
                   help="P(short caption) when --t2i-caption-mode sample.")
    p.add_argument("--t2i-min-short-side", type=int, default=0,
                   help="drop images whose short side is below this many pixels before "
                        "resize (0 = keep all; guards against upscaling tiny images to 1024).")
    p.add_argument("--t2i-min-aesthetic", type=float, default=None,
                   help="drop rows below this json.aesthetic_predictor_v_2_5_score "
                        "(unset = no aesthetic filter).")
    p.add_argument("--sample-cond-count", type=int, default=4,
                   help="fill/edit runs: number of FIXED conditioned previews captured from "
                        "the first training batch and logged with --sample-every-steps "
                        "(0 disables the conditioning preview).")
    p.add_argument("--sample-image-cfg", type=float, default=0.0,
                   help="conditioning previews: IP2P dual-guidance image scale (three-branch "
                        "CFG: uncond / source-only / source+instruction). 0 = classic "
                        "two-branch text CFG with the condition pinned in both branches. "
                        "Typical: 1.2-1.8 once --cond-input-dropout trained the zero branch.")
    p.add_argument("--expect-pristine-cond-input", action=bo, default=False,
                   help="fail launch if the x_embedder reserved slice is NON-zero — guard "
                        "for a fresh E1 start from a pure-t2i seed (leave off for E1 "
                        "resumes / E2 init from an E1 checkpoint).")

    # On-the-fly text encoder (used with --monet-direct / --cache-captions-only / --edit-direct).
    p.add_argument("--text-encoder-id", default="Qwen/Qwen3.5-4B")
    p.add_argument("--text-encoder-layer", type=int, default=-2,
                   help="hidden_states readout layer (-2 = second-to-last, raw/un-normed; "
                        "layers above it are dropped so they are never computed).")
    p.add_argument("--text-encoder-layers", default=None,
                   help="v8 fixed multi-layer fusion: csv of hidden_states layers (e.g. "
                        "'9,18,27'). Each is non-affine-RMSNormed per token and concatenated on "
                        "the feature dim, so text_dim becomes len(layers) x hidden. Overrides "
                        "--text-encoder-layer; the stack is truncated to the deepest entry.")
    p.add_argument("--text-encoder-max-length", type=int, default=300,
                   help="fixed token length for on-the-fly encoding; becomes text_seq_len.")
    p.add_argument("--text-encoder-instruction", action=bo, default=True,
                   help="prepend the standard conditioning instruction before each caption "
                        "(matches the cache pipeline's default).")

    p.add_argument("--latent-mean", type=float, default=0.0)
    p.add_argument("--latent-std", type=float, default=1.0)
    # Skip the startup dim-peek (which downloads one full shard) and use explicit dims.
    # Defaults match the monet DC-AE/Qwen cache: latent (32,16,16), text (224,2048).
    p.add_argument("--skip-peek", action="store_true",
                   help="Skip the dim-peek shard download; use --latent-*/--text-* dims directly.")
    p.add_argument("--latent-channels", type=int, default=32)
    p.add_argument("--latent-size", type=int, default=16)
    p.add_argument("--text-dim", type=int, default=2048)
    p.add_argument("--text-seq-len", type=int, default=224)

    # model
    p.add_argument("--hidden-dim", type=int, default=896)
    p.add_argument("--depth", type=int, default=24)
    p.add_argument("--num-heads", type=int, default=14)
    p.add_argument("--mlp-ratio", type=float, default=2.5)
    p.add_argument("--mixer-type", default="fla_gdn2",
                   choices=["mamba3", "fla_gated_deltanet", "fla_gdn2", "fla_linear", "fla_gla",
                            "fla_delta_net", "fla_hgrn2", "fla_based", "mlp", "fallback"])
    p.add_argument("--fla-mode", default="chunk",
                   choices=["chunk", "fused_chunk", "fused_recurrent"],
                   help="FLA kernel mode. Not every mixer accepts every value and the model "
                        "validates the pair: fla_linear/fla_gla take all three, "
                        "fla_gated_deltanet takes chunk/fused_recurrent, and fla_gdn2 takes "
                        "chunk only (FLA asserts chunk-only on the training path).")
    p.add_argument("--fla-feature-map", default="relu")
    p.add_argument("--fla-bidirectional", action=bo, default=True)
    p.add_argument("--use-short-conv", action="store_true")
    p.add_argument("--conv-size", type=int, default=4)
    p.add_argument("--image-attn-every", type=int, default=6)
    p.add_argument("--image-attn-backend", default="sdpa",
                   choices=["sdpa", "flash2", "flash3", "flash4", "auto"])
    p.add_argument("--image-attn-qk-rmsnorm", "--image-attn-qk-norm", dest="image_attn_qk_rmsnorm",
                   action=bo, default=True,
                   help="Sana-style per-head Q/K RMSNorm inside full image self-attention.")
    p.add_argument("--image-rope", action=bo, default=False,
                   help="Apply 2D RoPE to full image self-attention anchor blocks.")
    p.add_argument("--image-rope-theta", type=float, default=10000.0,
                   help="RoPE frequency base for --image-rope.")
    p.add_argument("--cross-attn-backend", default="sdpa", choices=["sdpa", "xformers", "auto", "mha"])
    p.add_argument("--cross-attn-qk-rmsnorm", "--cross-attn-qk-norm", dest="cross_attn_qk_norm",
                   action=bo, default=True,
                   help="Sana-style Q/K RMSNorm inside text cross-attention.")
    p.add_argument("--cross-attn-every", type=int, default=1,
                   help="Cross-attend to text every Nth single-stream block (1 = every block). "
                        "PixArt-Sigma-style thinning; pilot-ablate before a long run. 0 removes "
                        "cross-attention entirely, leaving joint attention (dual-stream prefix + "
                        "refresh blocks) as the only text pathway — unlike MMDiT, which keeps text "
                        "in the attended sequence in EVERY block, so validate prompt adherence.")
    p.add_argument("--joint-attn-backend", default="sdpa", choices=["sdpa", "flash4"],
                   help="Backend for the JOINT [image; text] sites (dual-stream prefix + "
                        "refresh blocks) — separate from --image-attn-backend, which routes "
                        "the DDT decoder's self-attention. flash4 uses FlashAttention-4/CuTe "
                        "varlen with seqused_k (2.0x over masked SDPA at B=56 S=576 on "
                        "sm_120) in isolation. KNOWN BAD on sm_120 inside the model: illegal "
                        "memory access during backward under gradient checkpointing, while the "
                        "standalone probe passes. Leave at sdpa unless a full checkpointed "
                        "backward has been verified on your exact GPU + torch.")
    p.add_argument("--shared-adaln", action=bo, default=False,
                   help="PixArt-alpha adaLN-single: one shared timestep-modulation projection for "
                        "all blocks (per-block scale_shift_table keeps layer variation). Saves ~27%% "
                        "params at the deployed size.")
    p.add_argument("--fourier-timestep", dest="fourier_timestep_embed", action=bo, default=False,
                   help="DiT-style sinusoidal timestep features before the timestep MLP instead of "
                        "feeding the raw sigma scalar to Linear(1, hidden).")
    p.add_argument("--hilbert-order", dest="hilbert_mixer_order", action=bo, default=False,
                   help="Feed the FLA mixers Hilbert-curve token order (restored afterward) so "
                        "sequence adjacency tracks 2D adjacency. Needs a square power-of-two grid.")
    p.add_argument("--parallel-block", action=bo, default=False,
                   help="FLUX/SD3-style parallel residual branches inside blocks.")
    p.add_argument("--dual-stream-depth", type=int, default=0)
    p.add_argument("--multimodal-coord-ids", action=bo, default=False)
    p.add_argument("--joint-refresh-attn", dest="joint_refresh_attention", action=bo, default=False,
                   help="v6: every --image-attn-every-th post-prefix block becomes a joint refresh "
                        "block — one MMDiT-style attention over [text; image] replaces the "
                        "image-only full attention + that block's cross-attn, and text is updated "
                        "through its own out-proj + FFN (SD3-style renegotiation).")
    p.add_argument("--refresh-layers", default=None,
                   help="v8: explicit joint-refresh block indices (encoder-absolute, zero-based "
                        "csv, e.g. '7,11,15,21'). Overrides the --image-attn-every cadence AND "
                        "the trailing-candidate demotion, so the final encoder block can be a "
                        "refresh block. Implies joint refresh at exactly these depths.")
    p.add_argument("--refresh-no-mixer", action=bo, default=False,
                   help="v8 substitutive refresh: refresh blocks drop the GDN2/FLA mixer — joint "
                        "attention replaces it instead of supplementing it (GLUMBConv stays).")
    p.add_argument("--block-attnres", action=bo, default=False,
                   help="v8: Kimi Block AttnRes (arXiv 2603.15031), paper algorithm. EVERY encoder "
                        "block's input is a learned per-token softmax mixture over [token "
                        "embedding, completed per-group delta sums, current partial sum] (one "
                        "pseudo-query per block); block deltas accumulate into the partial, which "
                        "folds at each --attnres-group-ends index; a dedicated final operator "
                        "aggregates the bank into z.")
    p.add_argument("--attnres-group-ends", default="2,5,8,11,14,17,19,21",
                   help="csv of encoder block indices closing each AttnRes delta group. Default "
                        "gives the paper-backed ~8 groups over 22 encoder blocks "
                        "(3,3,3,3,3,3,2,2); a trailing end at the last block is allowed and "
                        "coincides with the post-loop fold. Routing runs on FLA's fused_attnres "
                        "Triton kernel (FLA_ATTNRES_GLUON=1 opts into the Gluon backend).")
    p.add_argument("--ddt-decoder-depth", type=int, default=0,
                   help="v7: DDT split (arXiv 2504.05741) — the last N of --depth blocks become a "
                        "velocity decoder (full softmax attention + GLUMBConv, conditioned "
                        "token-wise on the encoder output z via adaLN, noisy latent re-embedded, "
                        "final layer conditioned on silu(t+z)). DDT-XL/2 at hidden 1152/depth 28 "
                        "used 6, keeping vanilla REPA placement (--repa-layer 8). 0 = off (v6 "
                        "layout). --repa-layer -1 => encoder mid-depth.")
    p.add_argument("--ddt-norm-impl", default="native", choices=["native", "liger"],
                   help="DDT decoder pre-norm impl. native = nn.RMSNorm + modulate (torch.compile "
                        "fuses it). liger = LigerModulatedRMSNorm, one Triton kernel for "
                        "(1+scale)*rmsnorm(x)+shift; needs liger-kernel installed. Opt in only "
                        "after profiling shows the norm sites matter, and benchmark under "
                        "--compile first (custom autograd can graph-break). Checkpoints are "
                        "interchangeable between impls (both non-affine, no keys).")
    p.add_argument("--sandwich-rmsnorm", action=bo, default=False,
                   help="v6: Lumina-Next sandwich norm — RMSNorm each sublayer output before the "
                        "residual add (the residual stream itself is never normalized).")
    p.add_argument("--rope-ntk-factor", type=float, default=1.0,
                   help="v6: NTK-aware RoPE rescale for image-spatial axes only (text positions "
                        "keep base frequencies). 1.0 for 512px pretrain; ~2.0 for the 1024px "
                        "fine-tune where the token grid doubles.")
    p.add_argument("--abs-pos-embed", action=bo, default=True)
    p.add_argument("--y-norm", action=bo, default=True)
    p.add_argument("--y-norm-scale-factor", type=float, default=0.01, help="Sana uses 0.01.")
    p.add_argument("--patch-size", type=int, default=1)
    p.add_argument("--extra-input-channels", type=int, default=0,
                   help="FLUX-Fill-style reserved channels (33 = masked latent + mask) for future inpainting.")
    p.add_argument("--gradient-checkpointing", "--grad-checkpoint", dest="gradient_checkpointing",
                   action="store_true")

    # REPA representation alignment (needs a cache with dino.npy, e.g. monet).
    p.add_argument("--repa-dim", type=int, default=0, help="SSL target dim; 1536 for DINOv2-ViT-G. 0=off")
    p.add_argument("--repa-layer", type=int, default=-1, help="DiT block to align; -1 => depth//2")
    p.add_argument("--repa-weight", type=float, default=0.0, help="REPA loss weight (e.g. 0.5)")
    p.add_argument("--repa-anneal-steps", type=int, default=0, help="decay REPA weight to 0 by this step")

    # Dispersive loss (arxiv 2506.09027): target-free batch-dispersion regularizer.
    # Works on the existing cache (no dino needed). 0 weight = off.
    p.add_argument("--dispersive-weight", type=float, default=0.0,
                   help="Dispersive loss weight (e.g. 0.25–0.5). 0=off.")
    p.add_argument("--dispersive-layer", type=int, default=-1,
                   help="DiT block whose pooled hidden state to disperse; -1 => depth//2.")
    p.add_argument("--dispersive-temp", type=float, default=0.5,
                   help="Temperature tau in the InfoNCE-style dispersive loss.")
    p.add_argument("--dispersive-metric", default="l2", choices=["l2", "cosine"],
                   help="Pairwise score: 'l2' (neg squared distance, paper default) or 'cosine'.")
    p.add_argument("--dispersive-anneal-steps", type=int, default=0,
                   help="decay dispersive weight to 0 by this step (0 = constant).")

    # optim / flow
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--fused-adamw", action=bo, default=True)
    p.add_argument("--optimizer", default="adamw", choices=["adamw", "came8bit", "muon", "cmuon"],
                   help="came8bit = SANA 1.5's CAME with factored second moments and "
                        "8-bit first moments (~7x less optimizer state than fp32 AdamW). "
                        "Gradient checkpointing does not touch optimizer state, so this "
                        "is the remaining large lever on peak memory. "
                        "muon = torch.optim.Muon (torch >= 2.9) on 2D hidden weights with "
                        "AdamW on norms/biases/embeddings; in production at Kimi K2, "
                        "GLM-4.5, INTELLECT-3. Aimed at quality, and halves optimizer "
                        "state as a side effect (momentum only, no second moment). "
                        "cmuon = diffusion-specific chunked Muon through Microsoft Dion: "
                        "fused QKV, GLUMBConv gate/value, and adaLN rows are "
                        "orthogonalized independently after fused momentum accumulation.")
    p.add_argument("--muon-lr", type=float, default=None,
                   help="LR for the Muon (2D hidden weight) group. Defaults to --lr: "
                        "adjust_lr_fn=match_rms_adamw rescales the update to AdamW's RMS, "
                        "so an AdamW-tuned rate transfers (unlike --came-lr).")
    p.add_argument("--muon-aux-lr", type=float, default=None,
                   help="LR for the AdamW group (norms, biases, embeddings, heads). "
                        "Defaults to --lr.")
    p.add_argument("--muon-weight-decay", type=float, default=0.1,
                   help="Weight decay on the Muon group. NOT --weight-decay: decoupled "
                        "weight decay is one of the two changes 'Muon is Scalable for LLM "
                        "Training' identifies as required to scale Muon, so it defaults "
                        "on (torch's default) even when --weight-decay is 0.")
    p.add_argument("--muon-momentum", type=float, default=0.95)
    p.add_argument("--muon-ns-steps", type=int, default=5,
                   help="Newton-Schulz iterations per step.")
    p.add_argument("--muon-adjust-lr-fn", default="match_rms_adamw",
                   choices=["match_rms_adamw", "original"],
                   help="match_rms_adamw = Moonshot's 0.2*lr*sqrt(max(A,B)) so AdamW LRs "
                        "transfer; original = Keller's sqrt(max(1, A/B)).")
    p.add_argument("--cmuon-lr", type=float, default=None,
                   help="Base LR for CMuon matrices. Defaults to --lr; the paper's "
                        "DC-AE DiT-XL recipe used 2e-4 with Moonlight RMS scaling.")
    p.add_argument("--cmuon-aux-lr", type=float, default=None,
                   help="LR for CMuon's AdamW group (1D params, embeddings, final head). "
                        "Defaults to --lr.")
    p.add_argument("--cmuon-weight-decay", type=float, default=0.0,
                   help="Decoupled decay for CMuon matrices. Paper default is 0.")
    p.add_argument("--cmuon-momentum", type=float, default=0.95)
    p.add_argument("--cmuon-ns-steps", type=int, default=6,
                   help="Newton-Schulz iterations. The CMuon DC-AE DiT-XL recipe uses 6.")
    p.add_argument("--cmuon-chunk-rescale", action=bo, default=True,
                   help="Multiply semantic chunk-group updates by sqrt(number of chunks), "
                        "the paper's default early-convergence acceleration. Disable with "
                        "--no-cmuon-chunk-rescale.")
    p.add_argument("--came-lr", type=float, default=None,
                   help="LR for --optimizer came8bit. REQUIRED: CAME RMS-clips its update "
                        "and has no bias correction, so an AdamW LR is far too large. "
                        "SANA's came8bit configs use 2e-5 where AdamW recipes use ~1e-4.")
    p.add_argument("--came-block-size", type=int, default=2048,
                   help="Quantization block for CAME 8-bit first moments.")
    p.add_argument("--came-min-8bit-size", type=int, default=16384,
                   help="Parameters smaller than this keep an fp32 first moment.")
    p.add_argument("--warmup-steps", type=int, default=1000,
                   help="Linear warmup steps for the warmup-bearing schedulers.")
    p.add_argument("--lr-scheduler", default="linear-warmup-constant",
                   choices=["constant", "linear-warmup-constant", "linear-warmup-cosine"],
                   help="constant=flat (no warmup); linear-warmup-constant=warmup then hold "
                        "(legacy default); linear-warmup-cosine=warmup then cosine decay to "
                        "--min-lr-ratio. Computed from global step, so resume is deterministic.")
    p.add_argument("--min-lr-ratio", type=float, default=0.1,
                   help="Final LR as a fraction of --lr for linear-warmup-cosine decay.")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--accumulate", type=int, default=1)
    p.add_argument("--cond-dropout", type=float, default=0.1)
    p.add_argument("--t-sampler", default="plateau-logit-normal",
                   choices=["uniform", "high", "low", "logit-normal", "bounded-logit-normal",
                            "stretched-logit-normal", "plateau-logit-normal"])
    # "Texture polish" final phase: from --t-sampler-final-step onward, switch the timestep
    # sampler (e.g. to uniform), shifting compute to the low-noise t~0 regime to sharpen
    # textures/edges. Off unless --t-sampler-final is set.
    p.add_argument("--t-sampler-final", default=None,
                   choices=["uniform", "high", "low", "logit-normal", "bounded-logit-normal",
                            "stretched-logit-normal", "plateau-logit-normal"],
                   help="Sampler to switch to for the final polish phase (e.g. uniform).")
    p.add_argument("--t-sampler-final-step", type=int, default=80000,
                   help="global_step at which to switch to --t-sampler-final.")
    # Per-sigma-band flow MSE. Unlike train/flow_loss, a fixed band's mean does not move
    # when the timestep WEIGHTING moves (--flow-shift, the --t-sampler-final switch), so
    # these curves stay comparable across those changes and show which noise regime is
    # still improving. Accumulated across the window, since one step barely populates a band.
    p.add_argument("--sigma-buckets", type=int, default=5,
                   help="Equal-width sigma bands to log the unweighted flow MSE in "
                        "(0 disables). Logged as train/flow_loss_sigma_<lo>_<hi>, with "
                        "train/sigma_frac_<lo>_<hi> giving each band's share of samples.")
    p.add_argument("--sigma-bucket-log-every", type=int, default=100,
                   help="Steps between per-sigma-band flushes (accumulate in between).")
    p.add_argument("--t-bias", type=float, default=2.0)
    p.add_argument("--flow-shift", type=float, default=1.5)
    p.add_argument("--logit-mean", type=float, default=0.0)
    p.add_argument("--logit-std", type=float, default=1.0)
    p.add_argument("--plateau-low", type=float, default=0.0)
    p.add_argument("--plateau-high", type=float, default=1.0)
    # SANA stretched-logit-normal quantile window (normal CDF space). Truncates the
    # logit-normal to [q-low, q-high] quantiles and stretches the central band back to
    # [0,1] — flattens the peaked logit-normal toward uniform (wider sigma coverage).
    # Narrower window => flatter; 0/1 == plain logit-normal (no truncation).
    p.add_argument("--stretch-q-low", type=float, default=0.0)
    p.add_argument("--stretch-q-high", type=float, default=1.0)
    p.add_argument("--timestep-weight", action="store_true")
    p.add_argument("--min-snr-gamma", type=float, default=5.0)

    # ema / trainer
    p.add_argument("--ema-decay", type=float, default=0.9999)
    p.add_argument("--ema-update-every", type=int, default=1)
    p.add_argument("--max-steps", "--steps", dest="max_steps", type=int, default=100000)
    p.add_argument("--precision", default="bf16-mixed")
    p.add_argument("--dtype", default=None, choices=["bf16", "fp16", "fp32"],
                   help="If set, overrides --precision (bf16->bf16-mixed, fp16->16-mixed, fp32->32-true).")
    p.add_argument("--device", default="cuda",
                   help="'cuda' (gpu accelerator) or 'cpu'. Lightning manages device placement.")
    p.add_argument("--devices", type=int, default=1, help="GPUs per node.")
    p.add_argument("--num-nodes", type=int, default=1)
    p.add_argument("--fsdp2", action=bo, default=False,
                   help="Shard the DiT with FSDP2 (fully_shard per block) via Lightning's "
                        "ModelParallelStrategy — pure data parallel (tensor_parallel_size=1, "
                        "data_parallel_size=devices*num_nodes; never 'auto', which would go "
                        "tensor-parallel on a single node). Forces precision 32-true: bf16 "
                        "compute + fp32 grad reduce come from the FSDP2 MixedPrecisionPolicy "
                        "with fp32 sharded master weights. Disables EMA and in-training "
                        "sample previews (neither is FSDP2-safe yet); --compile unsupported.")
    p.add_argument("--fsdp2-distributed-checkpoint", action=bo, default=False,
                   help="Save sharded (DCP directory) checkpoints instead of consolidating "
                        "full weights into a single rank-0 .ckpt file. Keep OFF so resume/"
                        "export/upload tooling keeps working; the consolidated save is cheap "
                        "at Boomer's scale.")
    p.add_argument("--checkpoint-dir", default="/content/ckpt_stage0",
                   help="Local directory for checkpoints (always written here first).")
    p.add_argument("--save-every-steps", "--save-every", dest="save_every_steps", type=int, default=2000)
    # Async checkpoint upload: save locally, copy to Drive/rclone in the background,
    # keep only the newest N local checkpoints (older ones deleted after they upload).
    p.add_argument("--checkpoint-upload-remote", default=None,
                   help="Background upload destination: an HF bucket "
                        "('hf://buckets/akrao9/Boomerv6-ckpt', uploaded via the hf CLI), "
                        "an rclone remote ('gdrive:Boomer/ckpts'), OR a directory such as "
                        "a mounted Drive folder / hf-mount'ed bucket. Checkpoints always "
                        "save to --checkpoint-dir (fast local disk) first and copy over "
                        "asynchronously. Off if unset.")
    p.add_argument("--checkpoint-hf-bin", default="hf",
                   help="hf CLI executable for hf://buckets/... upload remotes "
                        "(install: curl -LsSf https://hf.co/cli/install.sh | bash).")
    p.add_argument("--checkpoint-local-retain", type=int, default=2,
                   help="Keep this many newest checkpoints on local disk; older uploaded ones are deleted.")
    p.add_argument("--checkpoint-keep-uploaded", type=int, default=0,
                   help="Prune the remote to this many newest checkpoints. 0 = keep all uploaded.")
    p.add_argument("--checkpoint-rclone-bin", default="rclone",
                   help="rclone executable for --checkpoint-upload-remote rclone remotes.")
    p.add_argument("--checkpoint-upload-chunk-size", default="128M",
                   help="rclone --drive-chunk-size for Google Drive uploads.")
    p.add_argument("--log-every-steps", "--log-every", dest="log_every_steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", default=None, help="Path to a .ckpt to resume from.")

    # W&B logging (off unless --wandb). train/loss, train/repa_loss, lr, etc. log automatically.
    p.add_argument("--wandb", action="store_true", help="Log training metrics to Weights & Biases.")
    p.add_argument("--wandb-project", default="boomer", help="W&B project name.")
    p.add_argument("--wandb-run-name", default=None, help="Optional W&B run name.")
    p.add_argument("--wandb-entity", default=None, help="Optional W&B entity/team.")
    p.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online",
                   help="W&B mode. Use offline on air-gapped machines and sync later.")
    p.add_argument("--wandb-run-id", default=None,
                   help="Stable W&B run id for resuming the SAME run across sessions. "
                        "Defaults to the WANDB_RUN_ID env var. The run name is just a "
                        "display label and is not a stable key -- set this to resume.")
    # In-training sample previews -> W&B. Pre-encode prompts with tools/encode_prompts.py.
    p.add_argument("--sample-every-steps", type=int, default=0,
                   help="Sample the eval prompts every N optimizer steps and log to W&B. 0 disables.")
    p.add_argument("--sample-prompts-cache", default="prompts_cond.pt",
                   help="Path to the .pt produced by tools/encode_prompts.py (embeddings + masks).")
    p.add_argument("--sample-prompts-file", default=None,
                   help="--monet-direct: plain-text prompt list (one per line) for previews, "
                        "encoded by the resident text encoder. Falls back to the prompt "
                        "STRINGS inside --sample-prompts-cache when unset.")
    p.add_argument("--sample-weights", choices=["ema", "live"], default="ema",
                   help="Which weights the preview samples: EMA shadow (default) or live training weights.")
    p.add_argument("--sample-steps", type=int, default=32, help="Inference steps per preview image.")
    p.add_argument("--sample-sampler", choices=["stork2", "stork4", "euler", "heun"], default="stork2")
    p.add_argument("--sample-stork-substeps", type=int, default=9,
                   help="STORK intra-step RK micro-steps (STORKScheduler s). Not extra model forwards -- the loop Taylor-extrapolates velocity from ONE stored model_output, which is why more is not better: approximation error accumulates with s. The STORK paper's ablation finds s=9 optimal for conditional flow matching (beating 5, 14, 24, 54, 104). The scheduler's own 50 and the SANA example's INTRA_S=100 are general defaults, not the published flow-matching recipe.")
    p.add_argument("--sample-cfg-scale", type=float, default=4.0)
    p.add_argument("--sample-cfg-rescale", type=float, default=0.5)
    p.add_argument("--sample-cfg-interval-low", type=float, default=0.1)
    p.add_argument("--sample-cfg-interval-high", type=float, default=0.9)
    p.add_argument("--sample-seed", type=int, default=0, help="Fixed seed so previews are comparable across steps.")
    p.add_argument("--sample-vae-id", default=CacheConfig.vae_id, help="DC-AE decoder id for previews.")
    p.add_argument("--sample-scaling-factor", type=float, default=CacheConfig.scaling_factor)
    p.add_argument("--sample-stork-derivative-order", type=int, default=1,
                   choices=[1, 2, 3],
                   help="Taylor order for the preview STORK sampler. 1 is what "
                        "STORK's own flow-matching integration uses; previews "
                        "previously inherited the scheduler's general default of 2.")
    p.add_argument("--sample-flow-shift", type=float, default=None,
                   help="Flow shift for the preview sampler. Defaults to --flow-shift, "
                        "which is the TRAINING value -- MONET's reference trains at a low "
                        "shift and samples at 5, so previews rendered at the training shift "
                        "understate the model. Set this to your inference shift so the W&B "
                        "previews match what tools/sample_latent_dit.py produces.")
    p.add_argument("--resume-weights-only", action="store_true",
                   help="With --resume, load model weights only; reset optimizer/scheduler/step.")
    p.add_argument("--compile", action="store_true",
                   help="Wrap the training model in torch.compile. EMA is left uncompiled.")
    p.add_argument("--compile-mode",
                   choices=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"],
                   default="default",
                   help="torch.compile mode. 'default' avoids aggressive Triton autotune candidates "
                        "that can exceed Blackwell's per-SM resource limits.")
    p.add_argument("--compile-vae", action="store_true",
                   help="torch.compile the frozen DC-AE ENCODER (default off). Only "
                        "matters for the on-the-fly encode paths (--monet-image-direct / "
                        "--t2i-direct / "
                        "--edit-direct), where the VAE runs eager and can dominate step "
                        "time. Keep --edit-encode-chunk a divisor of --batch-size so "
                        "every chunk has one shape and Inductor compiles one graph.")
    p.add_argument("--compile-text-encoder", action="store_true",
                   help="EXPERIMENT (default off): torch.compile the truncated Qwen BASE transformer "
                        "(dynamic=True; the CausalLM head is never run on the text path). Compiled "
                        "graphs and workspaces cost extra persistent GPU memory, so do NOT enable it "
                        "on a memory-tight run -- validate in a separate process first, gating on "
                        "END-TO-END step time and peak memory, not Qwen-local numbers. Leave "
                        "--text-encoder-pad-to-max off: captions run ~57 tokens against a 320 cap, so "
                        "forcing a static shape costs ~5.6x the Qwen compute that compilation is "
                        "trying to save.")
    return p
