"""Cache jasperai/monet parquet shards into Boomer's WebDataset latent/text cache.

monet ships precomputed DC-AE f32c32 latents (verified drop-in for Boomer's VAE;
see tools/verify_monet_vae_compat.py) flattened to 8192 floats == 32x16x16 at
native 512x512. So this tool SKIPS image download + VAE encoding entirely: it
reads the latent column straight from parquet and only runs Qwen3.5-2B over the
provided captions, then writes the same WebDataset members the existing pipeline
emits (latents/text/text_mask + caption.txt + meta.json), consumed unchanged by
boomer/datasets/wds_latents.py.

It additionally writes a `dino.npy` member (monet's precomputed DINOv2-ViT-G/14
embedding) so a REPA representation-alignment loss can be added to training later
at no extra caching cost. The loader ignores unknown members, so existing
training keeps working until REPA is wired in.

Parallel fan-out matches the rest of the pipeline: launch one process per GPU
with RANK/WORLD_SIZE set; each rank takes parquet shards `[rank::world_size]`
and writes its own `r{rank}` tar shards + manifest.

Example (single GPU, smoke test on 2 shards, no upload):

    python -m tools.cache_monet_parquet_to_boomer_wds \
        --subset cc12m --max-source-shards 2 \
        --output-dir /content/monet_cache \
        --hf-token "$HF_TOKEN"

Example (8-GPU launcher):

    for r in $(seq 0 7); do
      RANK=$r WORLD_SIZE=8 CUDA_VISIBLE_DEVICES=$r \
        python -m tools.cache_monet_parquet_to_boomer_wds \
          --output-dir /data/monet_cache \
          --upload-repo you/boomer-monet-cache --delete-uploaded \
          --hf-token "$HF_TOKEN" &
    done; wait
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

from boomer.models.qwen_connector import (
    QwenConnectorConfig,
    QwenReferenceEncoder,
    encode_qwen_text_fixed,
)
from boomer.sana_latent_cache import hf_token_value
from tools.cache_hf_wds_to_boomer_wds import (
    AsyncHubUploader,
    normalize_storage_dtype,
    tensor_to_storage_npy,
    tensor_to_u8_npy,
)

DEFAULT_LATENT_KEY = "embedding_vae-dc-sana1p5-1p6b-1024px-tiling-128-resolution-512x512"
DEFAULT_CAPTION_KEY = "caption_gemini-2.5-flash-lite"
DEFAULT_DINO_KEY = "embedding_dinov2-vitg14"
LATENT_CHANNELS = 32  # DC-AE f32c32


# --------------------------------------------------------------------------- #
# Source shard discovery
# --------------------------------------------------------------------------- #
def list_source_parquets(repo_id: str, subset: str | None, token: str | None) -> list[str]:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    prefix = f"v1.1.0/{subset}/" if subset else "v1.1.0/"
    parquets = sorted(f for f in files if f.startswith(prefix) and f.endswith(".parquet"))
    if not parquets:
        raise FileNotFoundError(f"No parquet shards under {prefix} in {repo_id}")
    return parquets


def resolve_rank_world(args: argparse.Namespace) -> tuple[int, int]:
    rank = args.rank if args.rank is not None else int(os.environ.get("RANK", "0"))
    world = args.world_size if args.world_size is not None else int(os.environ.get("WORLD_SIZE", "1"))
    if rank < 0 or world < 1 or rank >= world:
        raise ValueError(f"Invalid rank/world: rank={rank} world={world}")
    return rank, world


# --------------------------------------------------------------------------- #
# Quality filtering (monet ships per-image quality signals)
# --------------------------------------------------------------------------- #
def passes_filters(row: dict[str, Any], args: argparse.Namespace, seen_clusters: set[str]) -> bool:
    if args.min_aesthetic is not None:
        a = row.get("aesthetic_laion-improved")
        if a is None or a < args.min_aesthetic:
            return False
    if args.max_nsfw is not None:
        n = row.get("nsfw_jasperai")
        if n is not None and n > args.max_nsfw:
            return False
    if args.max_watermark is not None:
        w = row.get("wk_jasperai")
        if w is not None and w > args.max_watermark:
            return False
    if args.dedup:
        cid = row.get("sscd_cluster_id")
        if cid:
            if cid in seen_clusters:
                return False
            seen_clusters.add(cid)
    return True


# --------------------------------------------------------------------------- #
# Latent reshape
# --------------------------------------------------------------------------- #
def flat_to_latent(vec: list[float]) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32)
    n = arr.size
    if n % LATENT_CHANNELS != 0:
        raise ValueError(f"latent length {n} not divisible by {LATENT_CHANNELS}")
    cells = n // LATENT_CHANNELS
    side = int(round(cells ** 0.5))
    if side * side != cells:
        raise ValueError(f"latent {n} -> {cells} cells is not square")
    return arr.reshape(LATENT_CHANNELS, side, side)


# --------------------------------------------------------------------------- #
# Multi-subset mixing (build ONE proportionally-interleaved cache in one run)
# --------------------------------------------------------------------------- #
def parse_mix(spec: str) -> list[tuple[str, float]]:
    """'commoncatalog-cc-by=0.3,coyo=0.2,...' -> normalized [(subset, weight), ...].

    Weights are relative; they are normalized to sum to 1 so the caller can pass
    raw proportions, percentages, or sample-count-ish numbers interchangeably."""
    pairs: list[tuple[str, float]] = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "=" in tok:
            name, w = tok.split("=", 1)
        elif ":" in tok:
            name, w = tok.split(":", 1)
        else:
            raise ValueError(f"bad --mix entry {tok!r}; expected subset=weight")
        pairs.append((name.strip(), float(w)))
    if not pairs:
        raise ValueError("--mix is empty")
    total = sum(w for _, w in pairs)
    if total <= 0:
        raise ValueError("--mix weights sum to <= 0")
    return [(n, w / total) for n, w in pairs]


def resolve_columns(available: set[str], args: argparse.Namespace, caption_keys: list[str],
                    extra_cols: list[str]) -> tuple[str, list[str], str | None, list[str]]:
    """Pick latent/caption/dino columns against THIS shard's schema (subsets differ)."""
    latent_col = args.latent_key if args.latent_key in available else next(
        (c for c in sorted(available) if "vae" in c.lower()), None
    )
    if latent_col is None:
        raise RuntimeError(f"No VAE-latent column found; have {sorted(available)}")
    present_caption_keys = [c for c in caption_keys if c in available]
    dino_col = args.dino_key if (not args.no_dino and args.dino_key in available) else None
    want = [latent_col, *present_caption_keys, *[c for c in extra_cols if c in available]]
    if dino_col:
        want.append(dino_col)
    return latent_col, present_caption_keys, dino_col, list(dict.fromkeys(want))


def build_item(
    row: dict[str, Any], *, latent_col: str, present_caption_keys: list[str],
    dino_col: str | None, args: argparse.Namespace, rank: int, shard_tag: str,
    seen_clusters: set[str], counters: dict[str, Any],
) -> dict[str, Any] | None:
    """Filter + resolve caption for one parquet row -> a ready cache item, or None if
    it should be dropped. Updates ``counters`` (seen/filtered/no_latent/no_caption/
    caption_src) in place so callers can report identically to the single-subset path."""
    counters["seen"] += 1
    if not passes_filters(row, args, seen_clusters):
        counters["filtered"] += 1
        return None
    latent = row.get(latent_col)
    if latent is None:
        counters["no_latent"] += 1
        return None
    caption, caption_src = "", None
    for ck in present_caption_keys:
        v = row.get(ck)
        if v:
            caption, caption_src = v, ck
            break
    if not caption:
        counters["no_caption"] += 1
        return None
    counters["caption_src"][caption_src] += 1
    key = row.get("__key__") or row.get("id") or f"r{rank}{shard_tag}n{counters['seen']}"
    item = {
        "key": str(key), "latent": latent, "caption": caption, "caption_src": caption_src,
        "source": row.get("source"), "aesthetic": row.get("aesthetic_laion-improved"),
        "nsfw": row.get("nsfw_jasperai"), "cluster": row.get("sscd_cluster_id"),
    }
    if dino_col:
        item["dino"] = row.get(dino_col)
    return item


def iter_subset_items(
    subset: str, parquets: list[str], *, args: argparse.Namespace, token_str: str | None,
    staging_dir: Path, dl_executor: Any, caption_keys: list[str], extra_cols: list[str],
    rank: int, seen_clusters: set[str], counters: dict[str, Any],
):
    """Lazily yield ready cache items for ONE subset: download its parquets (small
    per-subset look-ahead window on the shared executor), stream rows in bounded
    pyarrow batches (so memory stays flat with many subsets open at once), and
    delete each parquet after read."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    futures: dict[int, Any] = {}

    def submit(i: int) -> None:
        if 0 <= i < len(parquets) and i not in futures:
            futures[i] = dl_executor.submit(
                hf_hub_download, repo_id=args.repo_id, filename=parquets[i],
                repo_type="dataset", token=token_str, local_dir=str(staging_dir),
            )

    ahead = max(1, args.mix_prefetch_per_subset)
    for i in range(min(ahead, len(parquets))):
        submit(i)
    for idx in range(len(parquets)):
        local = futures.pop(idx).result()
        submit(idx + ahead)
        available = set(pq.read_schema(local).names)
        latent_col, present_caption_keys, dino_col, want = resolve_columns(
            available, args, caption_keys, extra_cols
        )
        pf = pq.ParquetFile(local)
        for batch in pf.iter_batches(columns=want, batch_size=args.read_batch):
            cols = {c: batch.column(c).to_pylist() for c in batch.schema.names}
            for i in range(batch.num_rows):
                row = {c: cols[c][i] for c in cols}
                item = build_item(
                    row, latent_col=latent_col, present_caption_keys=present_caption_keys,
                    dino_col=dino_col, args=args, rank=rank, shard_tag=f"{subset[:6]}{idx}",
                    seen_clusters=seen_clusters, counters=counters,
                )
                if item is not None:
                    yield item
        try:
            os.remove(local)
        except FileNotFoundError:
            pass


# --------------------------------------------------------------------------- #
# Text-storage fidelity gate (E4M3 round-trip) + cache fingerprint
# --------------------------------------------------------------------------- #
_STORAGE_TORCH_DTYPE = {
    "bfloat16": "bfloat16", "bf16": "bfloat16",
    "float16": "float16", "fp16": "float16",
    "float8_e4m3fn": "float8_e4m3fn", "fp8": "float8_e4m3fn", "float8": "float8_e4m3fn",
    "float32": "float32", "fp32": "float32",
}

_FIDELITY_REFS = [
    "a photograph of a golden retriever running through a park",
    "an oil painting of a stormy sea at sunset",
    "a labeled diagram of the water cycle with arrows",
    "macro close-up of a dewdrop on a green leaf",
]


def _transformers_version() -> str:
    try:
        import transformers
        return transformers.__version__
    except Exception:
        return "unknown"


def _storage_roundtrip(t: "torch.Tensor", dtype_name: str) -> "torch.Tensor":
    """Cast to the cache storage dtype and back to fp32 — exactly the precision the
    loader sees (storage is a bit-preserving view, so value == .to(dtype).to(fp32))."""
    name = _STORAGE_TORCH_DTYPE.get(dtype_name, "bfloat16")
    return t.to(getattr(torch, name)).to(torch.float32)


def text_fidelity_fingerprint(
    encoder: QwenReferenceEncoder, args: argparse.Namespace, *, min_cosine: float,
) -> dict[str, Any]:
    """Encode reference captions, round-trip through the storage dtype, and verify the
    representation survives. ALSO returns a fingerprint (transformers version + abs-max)
    so a future env drift that silently changes the encoder is caught at cache time.

    Aborts if the round-trip NaNs (out of range) or cosine drops below ``min_cosine`` —
    turning the manual checks from this session into an automatic gate."""
    with torch.inference_mode():
        hidden, mask = encode_qwen_text_fixed(encoder, _FIDELITY_REFS, args.max_length)
    h = hidden.float()
    rt = _storage_roundtrip(hidden, args.text_dtype).float()
    valid = mask.bool().reshape(-1)
    a = h.reshape(-1, h.shape[-1])[valid]
    b = rt.reshape(-1, rt.shape[-1])[valid]
    cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)
    fp = {
        "transformers_version": _transformers_version(),
        "ref_abs_max": round(float(h.abs().max()), 2),
        "fidelity_dtype": args.text_dtype,
        "fidelity_cosine_mean": round(float(cos.mean()), 5),
        "fidelity_cosine_min": round(float(cos.min()), 5),
    }
    print(
        f"[monet-cache] text fidelity: dtype={args.text_dtype} abs-max={fp['ref_abs_max']} "
        f"cosine mean={fp['fidelity_cosine_mean']} min={fp['fidelity_cosine_min']} "
        f"(transformers {fp['transformers_version']})",
        flush=True,
    )
    nan = int(torch.isnan(rt).sum())
    if nan:
        raise RuntimeError(
            f"[monet-cache] text round-trip produced {nan} NaNs at dtype {args.text_dtype}: "
            f"abs-max {fp['ref_abs_max']} exceeds its range (e4m3fn max 448). Use bf16, or fix "
            f"the encoder readout so embeddings are post-norm (~72). Aborting before caching."
        )
    # Gate on the MEAN (aggregate fidelity); per-token min naturally dips with E4M3 even
    # on healthy data, so it's reported but not the gate.
    if float(cos.mean()) < min_cosine:
        raise RuntimeError(
            f"[monet-cache] text fidelity cosine mean {float(cos.mean()):.5f} < {min_cosine} at "
            f"dtype {args.text_dtype}; refusing to cache a degraded representation (encoder/"
            f"transformers drift, or this dtype needs block-wise scaling). Aborting."
        )
    return fp


# --------------------------------------------------------------------------- #
# Parquet shard sink (alternative to WebDataset tar). Parquet streams far better
# from HF — `datasets streaming=True` + large range reads gives the "2x sample/s,
# 100x fewer requests" wins from the HF streaming-datasets blog, and sidesteps the
# silent tar-shard socket stalls the rolling WDS loader has to watchdog around.
# --------------------------------------------------------------------------- #
class ParquetShardWriter:
    """Drop-in stand-in for ``wds.ShardWriter`` that writes one Parquet shard per
    ``maxcount``/``maxsize`` window, one row per sample.

    It accepts the SAME sample dict the tar path builds (``encode_and_write`` is
    unchanged). Array members (``latents.npy``/``text.npy``/``text_mask.npy``/
    ``dino.npy``) are stored as their ``.npy`` bytes in binary columns —
    byte-identical to the tar members, self-describing dtype/shape, decoded with
    ``np.load(BytesIO(col))``. ``caption.txt`` becomes a utf-8 string column and
    ``meta.json`` a JSON-string column. ``post(path)`` fires on each finished shard
    so the existing AsyncHubUploader handles batched upload/delete exactly as for tar.

    ``maxsize`` is the UNCOMPRESSED payload budget (sum of binary member bytes); the
    on-disk Parquet file is usually a touch smaller after compression.
    """

    _MEMBER_TO_COLUMN = {
        "latents.npy": "latents",
        "text.npy": "text",
        "text_mask.npy": "text_mask",
        "dino.npy": "dino",
    }

    def __init__(
        self, pattern: str, *, maxcount: int, maxsize: int, start_shard: int = 0,
        post: Any = None, row_group_size: int = 512, compression: str = "zstd",
    ) -> None:
        self.pattern = pattern
        self.maxcount = max(1, maxcount)
        self.maxsize = max(1, maxsize)
        self.shard = start_shard
        self.post = post
        self.row_group_size = max(1, row_group_size)
        self.compression = compression
        self._reset()

    def _reset(self) -> None:
        self._cols: dict[str, list] = {}
        self._schema_cols: list[str] | None = None
        self._count = 0
        self._bytes = 0

    def write(self, sample: dict[str, Any]) -> None:
        rec: dict[str, Any] = {
            "__key__": str(sample["__key__"]),
            "latents": sample["latents.npy"],
            "text": sample["text.npy"],
            "text_mask": sample["text_mask.npy"],
            "caption": sample.get("caption.txt", ""),
            "meta": json.dumps(sample.get("meta.json", {}), separators=(",", ":")),
        }
        if "dino.npy" in sample:
            rec["dino"] = sample["dino.npy"]
        # Fix the column set from the first row; keep every shard rectangular even if a
        # later/earlier row is missing an optional member (e.g. dino) by padding nulls.
        if self._schema_cols is None:
            self._schema_cols = list(rec.keys())
            self._cols = {c: [] for c in self._schema_cols}
        for c in self._schema_cols:
            self._cols[c].append(rec.get(c))
        for c in rec:
            if c not in self._cols:  # a new optional column appeared mid-shard: backfill
                self._cols[c] = [None] * self._count + [rec[c]]
                self._schema_cols.append(c)
        self._count += 1
        self._bytes += sum(len(sample[m]) for m in self._MEMBER_TO_COLUMN if m in sample)
        if self._count >= self.maxcount or self._bytes >= self.maxsize:
            self.flush()

    def flush(self) -> None:
        if self._count == 0:
            return
        import pyarrow as pa
        import pyarrow.parquet as pq

        path = self.pattern % self.shard
        table = pa.table(self._cols)
        pq.write_table(
            table, path,
            row_group_size=min(self.row_group_size, self._count),
            compression=self.compression,
            use_dictionary=False,  # unique binary blobs: dictionary encoding only bloats
        )
        if self.post is not None:
            self.post(path)
        self.shard += 1
        self._reset()

    def close(self) -> None:
        self.flush()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo-id", default="jasperai/monet")
    p.add_argument("--subset", default="cc12m", help="subset dir under v1.1.0/ (empty = all subsets)")
    # Multi-subset mixing: build one proportionally-interleaved cache in a single run.
    p.add_argument("--mix", default=None,
                   help="weighted subset mix, e.g. "
                        "'commoncatalog-cc-by=0.30,laion-cc-by=0.15,coyo=0.15,"
                        "synthetic-flux1-schnell=0.20,...'. Weights are normalized. "
                        "Overrides --subset and round-robins rows across subsets so every "
                        "OUTPUT shard is proportionally mixed (good per-batch diversity).")
    p.add_argument("--mix-target", type=int, default=5_000_000,
                   help="total samples to write across ALL ranks in --mix mode "
                        "(per-rank target = mix-target / world-size).")
    p.add_argument("--mix-prefetch-per-subset", type=int, default=1,
                   help="parquets to download ahead PER active subset in --mix mode.")
    p.add_argument("--mix-allow-repeat", action="store_true",
                   help="in --mix mode, re-iterate a subset that runs out of data before "
                        "hitting its target (upsampling, e.g. tiny diffusion-aesthetic).")
    p.add_argument("--read-batch", type=int, default=8192,
                   help="rows per pyarrow batch when streaming parquets in --mix mode "
                        "(bounds memory with many subsets open).")
    p.add_argument("--fidelity-min-cosine", type=float, default=0.999,
                   help="abort if the text storage-dtype round-trip drops MEAN cosine below this "
                        "on reference captions (catches fp8 range/precision loss & encoder drift).")
    p.add_argument("--no-fidelity-check", action="store_true",
                   help="skip the startup text-fidelity gate (not recommended for fp8).")
    p.add_argument("--latent-key", default=DEFAULT_LATENT_KEY)
    p.add_argument("--caption-key", default=DEFAULT_CAPTION_KEY,
                   help="preferred caption column")
    p.add_argument("--caption-fallback-keys",
                   default="caption_internvl-3-8b,caption_sharegpt4v-7b,caption_florence-2-large,caption_original",
                   help="comma-ordered captions to use when --caption-key is empty/absent "
                        "(some subsets, e.g. commoncatalog, have no Gemini caption)")
    p.add_argument("--dino-key", default=DEFAULT_DINO_KEY)
    p.add_argument("--no-dino", action="store_true", help="skip the DINOv2 REPA member")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--output-format", choices=["wds", "parquet"], default="wds",
                   help="wds = WebDataset .tar shards (default); parquet = one row per sample "
                        "in .parquet shards (streams far better from HF via `datasets "
                        "streaming=True` + range reads; recommended for new caches).")
    p.add_argument("--parquet-row-group-size", type=int, default=512,
                   help="rows per Parquet row group (parquet format only). ~512 rows of "
                        "~0.65 MB/sample => ~330 MB groups, a good range-read granularity.")
    p.add_argument("--parquet-compression", default="zstd",
                   help="Parquet column compression (parquet format only): zstd|snappy|none. "
                        "fp8 text is near-incompressible, so gains are modest; zstd is cheap.")
    p.add_argument("--shard-size", type=int, default=100000,
                   help="max samples per output shard (upper bound; --max-shard-gb usually triggers first)")
    p.add_argument("--max-shard-gb", type=float, default=4.0,
                   help="rotate the output shard at this many GB (default 4)")
    p.add_argument("--encode-batch", type=int, default=64, help="captions per Qwen forward")
    p.add_argument("--max-length", type=int, default=340, help="must match existing Qwen cache shape")
    p.add_argument("--latent-dtype", default="bfloat16")
    p.add_argument("--text-dtype", default="bfloat16")
    p.add_argument("--dino-dtype", default="float16")
    # Qwen text encoder
    p.add_argument("--qwen-id", default=QwenConnectorConfig.qwen_model_id)
    p.add_argument("--qwen-readout-layer", type=int, default=-1)
    p.add_argument("--qwen-instruction", default=None)
    p.add_argument("--no-qwen-instruction", action="store_true")
    p.add_argument("--dtype", default="bf16")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--compile", action="store_true",
                   help="torch.compile(mode=default, dynamic=True) the Qwen forward for speed")
    # filtering
    p.add_argument("--min-aesthetic", type=float, default=None)
    p.add_argument("--max-nsfw", type=float, default=None)
    p.add_argument("--max-watermark", type=float, default=None)
    p.add_argument("--dedup", action="store_true", help="keep one image per sscd_cluster_id (per rank)")
    # parallel + limits
    p.add_argument("--rank", type=int, default=None)
    p.add_argument("--world-size", type=int, default=None)
    p.add_argument("--max-source-shards", type=int, default=None, help="cap source shards (testing)")
    p.add_argument("--prefetch-shards", type=int, default=10,
                   help="source parquets to download ahead of the GPU (bounded disk window)")
    p.add_argument("--source-staging-dir", default=None,
                   help="where to download source parquets (default <output-dir>/_src). Files are "
                        "deleted after read; using a real dir (not the HF cache) lets that free disk.")
    p.add_argument("--max-pending-uploads", type=int, default=6,
                   help="block when more than this many output shards await upload (bounds disk)")
    # Resume: skip source shards already done and continue output shard numbering.
    p.add_argument("--start-source-shard", type=int, default=0,
                   help="skip the first N of THIS rank's source shards (resume)")
    p.add_argument("--start-output-shard", type=int, default=0,
                   help="first output shard index (set to last written + 1 to resume)")
    # upload
    p.add_argument("--upload-repo", default=None)
    p.add_argument("--repo-path-prefix", default="")
    p.add_argument("--upload-workers", type=int, default=4)
    p.add_argument("--upload-batch", type=int, default=8,
                   help="shards committed per HF commit (HF caps commits at 256/hr; "
                        "batching lets you keep small training-friendly shards). Set "
                        "--max-pending-uploads >= this so full batches form before backpressure.")
    p.add_argument("--delete-uploaded", action="store_true")
    p.add_argument("--no-create-repo", action="store_true")
    p.add_argument("--hf-token", default=None)
    return p.parse_args()


def build_qwen_encoder(args: argparse.Namespace, token) -> QwenReferenceEncoder:
    instruction = None if args.no_qwen_instruction else (args.qwen_instruction or QwenConnectorConfig.instruction)
    cfg = QwenConnectorConfig(
        qwen_model_id=args.qwen_id,
        readout_layer=args.qwen_readout_layer,
        instruction=instruction,
        dtype=args.dtype,
        # Cap the forward at the cache length (don't process tokens we'll discard).
        # When compiling, also pad to that exact length so shapes are static.
        max_text_tokens=args.max_length,
        pad_to_max_length=args.compile,
    )
    encoder = QwenReferenceEncoder(cfg, hf_token=token)
    encoder.load()
    encoder.to(args.device)
    if args.compile:
        # Static seq length (pad_to_max_length) -> only the batch dim varies (full vs
        # the last partial batch), so dynamic=True settles in ~2 compiles instead of
        # thrashing on every distinct caption length.
        if getattr(encoder, "_base_model", None) is not None:
            encoder._base_model = torch.compile(encoder._base_model, mode="default", dynamic=True)
        if getattr(encoder, "_model", None) is not None:
            encoder._model = torch.compile(encoder._model, mode="default", dynamic=True)
        print("[monet-cache] torch.compile(mode=default, dynamic=True) applied to Qwen "
              f"(static seq={args.max_length})")
    print(f"[monet-cache] Qwen ready (hidden_size={encoder.hidden_size}, layer={args.qwen_readout_layer})")
    return encoder


def write_manifest(
    output_dir: Path,
    rank: int,
    *,
    latent_dtype: str,
    text_dtype: str,
    text_shape: tuple[int, int],
    dino_dtype: str | None,
    dino_dim: int | None,
    args: argparse.Namespace,
    uploader: AsyncHubUploader,
    fingerprint: dict[str, Any] | None = None,
    output_format: str = "wds",
) -> None:
    if output_format == "parquet":
        # Arrays are stored as their .npy bytes in binary columns (np.load(BytesIO));
        # caption is a utf-8 string column, meta a JSON-string column.
        schema = ["latents", "text", "text_mask", "caption", "meta"]
        if dino_dim is not None:
            schema.append("dino")
        fmt = "boomer_parquet_latent_text_cache"
    else:
        schema = ["latents.npy", "text.npy", "text_mask.npy", "caption.txt", "meta.json"]
        if dino_dim is not None:
            schema.append("dino.npy")
        fmt = "boomer_wds_latent_text_cache"
    manifest = {
        "format_version": 2,
        "format": fmt,
        "output_format": output_format,
        "latents_dtype": latent_dtype,
        # monet's 512px f32c32 latents are always 32x16x16
        "latents_shape": [LATENT_CHANNELS, 16, 16],
        "text_dtype": text_dtype,
        "text_shape": list(text_shape),
        "text_mask_dtype": "uint8",
        "raw_tensors": False,
        "schema": schema,
        "source": args.repo_id,
        "caption_key": args.caption_key,
        "qwen_model_id": args.qwen_id,
        "qwen_readout_layer": args.qwen_readout_layer,
        "max_length": args.max_length,
    }
    if output_format == "parquet":
        # arrays = np.load(BytesIO(col)); caption = str; meta = JSON string.
        manifest["column_encoding"] = "npy_bytes"
    if dino_dim is not None:
        manifest["dino_dtype"] = dino_dtype
        manifest["dino_shape"] = [dino_dim]
    if fingerprint:
        manifest.update(fingerprint)  # transformers version, ref abs-max, fidelity cosine
    name = "manifest.json" if rank == 0 else f"manifest-r{rank:02d}.json"
    path = output_dir / name
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    uploader.upload_path(path)
    print(f"[monet-cache] wrote {name}")


def write_null_embeddings(
    output_dir: Path,
    encoder: QwenReferenceEncoder,
    args: argparse.Namespace,
    uploader: AsyncHubUploader,
) -> None:
    # CFG null embedding: encode the empty string, same shape as cached text.
    hidden, mask = encode_qwen_text_fixed(encoder, [""], args.max_length)
    null_text = output_dir / "null_text.npy"
    null_mask = output_dir / "null_mask.npy"
    null_text.write_bytes(tensor_to_storage_npy(hidden[0], args.text_dtype))
    null_mask.write_bytes(tensor_to_u8_npy(mask[0]))
    uploader.upload_path(null_text)
    uploader.upload_path(null_mask)
    print("[monet-cache] wrote null_text.npy / null_mask.npy")


def main() -> int:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    args = parse_args()
    token = hf_token_value(args.hf_token)
    token_str = token if isinstance(token, str) else None
    normalize_storage_dtype(args.latent_dtype)  # validate early
    normalize_storage_dtype(args.text_dtype)
    rank, world = resolve_rank_world(args)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    staging_dir = Path(args.source_staging_dir) if args.source_staging_dir else (out_dir / "_src")
    staging_dir.mkdir(parents=True, exist_ok=True)

    my_parquets: list[str] = []
    if not args.mix:  # single-subset path lists one prefix; --mix lists per-subset later
        all_parquets = list_source_parquets(args.repo_id, args.subset or None, token_str)
        my_parquets = all_parquets[rank::world]
        if args.max_source_shards is not None:
            my_parquets = my_parquets[: args.max_source_shards]
        if args.start_source_shard:
            my_parquets = my_parquets[args.start_source_shard:]  # resume: skip already-done shards
        print(f"[monet-cache rank {rank}/{world}] {len(my_parquets)} source parquet shards "
              f"(start_source_shard={args.start_source_shard}, start_output_shard={args.start_output_shard})")

    encoder = build_qwen_encoder(args, token)

    # Gate + fingerprint: verify the text storage dtype round-trips faithfully on this
    # exact encoder/env BEFORE caching, and stamp the fingerprint into the manifest so a
    # future drift fails loudly instead of silently poisoning a run.
    fingerprint: dict[str, Any] = {}
    if not args.no_fidelity_check:
        fingerprint = text_fidelity_fingerprint(encoder, args, min_cosine=args.fidelity_min_cosine)

    uploader = AsyncHubUploader(
        repo_id=args.upload_repo,
        repo_path_prefix=args.repo_path_prefix,
        workers=args.upload_workers,
        delete_uploaded=args.delete_uploaded,
        create_repo=not args.no_create_repo,
        commit_batch_size=args.upload_batch,
    )

    # Caption priority: preferred key first, then fallbacks (subsets vary in which
    # captioners ran). The first non-empty value per row is used.
    caption_keys = [args.caption_key] + [
        c.strip() for c in args.caption_fallback_keys.split(",") if c.strip()
    ]
    extra_cols = ["__key__", "id", "aesthetic_laion-improved", "nsfw_jasperai",
                  "wk_jasperai", "sscd_cluster_id", "source"]

    run_label = "mix" if args.mix else (args.subset or "all")
    is_parquet = args.output_format == "parquet"
    ext = "parquet" if is_parquet else "tar"
    shard_glob = f"*.{ext}"  # backpressure counts only output shards, not manifest/null npy
    pattern = str(out_dir / f"monet-{run_label}-r{rank:02d}-%06d.{ext}")
    if is_parquet:
        sink = ParquetShardWriter(
            pattern,
            maxcount=args.shard_size,
            maxsize=int(args.max_shard_gb * 1e9),
            start_shard=args.start_output_shard,
            post=uploader.upload_path,
            row_group_size=args.parquet_row_group_size,
            compression=args.parquet_compression,
        )
    else:
        import webdataset as wds

        sink = wds.ShardWriter(
            pattern,
            maxcount=args.shard_size,
            maxsize=int(args.max_shard_gb * 1e9),
            start_shard=args.start_output_shard,
            post=uploader.upload_path,
        )

    seen_clusters: set[str] = set()
    manifest_written = False
    n_written = 0
    n_seen = 0
    n_filtered = 0
    n_no_latent = 0
    n_no_caption = 0
    caption_src_counts: Counter[str] = Counter()

    def encode_and_write(buffer: list[dict[str, Any]]) -> None:
        nonlocal n_written, manifest_written
        if not buffer:
            return
        latents = torch.from_numpy(
            np.stack([flat_to_latent(item["latent"]) for item in buffer])
        )  # (B,32,16,16) fp32
        captions = [item["caption"] for item in buffer]
        with torch.inference_mode():
            hidden, mask = encode_qwen_text_fixed(encoder, captions, args.max_length)
        hidden = hidden.detach().cpu()
        mask = mask.detach().cpu()
        has_dino = (not args.no_dino) and all(item.get("dino") is not None for item in buffer)
        dinos = None
        if has_dino:
            dinos = torch.from_numpy(
                np.stack([np.asarray(item["dino"], dtype=np.float32) for item in buffer])
            )

        if not manifest_written:
            text_shape = (int(hidden.shape[1]), int(hidden.shape[2]))
            write_manifest(
                out_dir, rank,
                latent_dtype=args.latent_dtype, text_dtype=args.text_dtype,
                text_shape=text_shape,
                dino_dtype=args.dino_dtype if not args.no_dino else None,
                dino_dim=int(dinos.shape[1]) if dinos is not None else None,
                args=args, uploader=uploader, fingerprint=fingerprint,
                output_format=args.output_format,
            )
            if rank == 0:
                write_null_embeddings(out_dir, encoder, args, uploader)
            manifest_written = True

        for i, item in enumerate(buffer):
            sample = {
                "__key__": item["key"],
                "latents.npy": tensor_to_storage_npy(latents[i], args.latent_dtype),
                "text.npy": tensor_to_storage_npy(hidden[i], args.text_dtype),
                "text_mask.npy": tensor_to_u8_npy(mask[i]),
                "caption.txt": item["caption"],
                "meta.json": {
                    "source": item.get("source") or args.repo_id,
                    "key": item["key"],
                    "caption_key": item.get("caption_src") or args.caption_key,
                    "aesthetic": item.get("aesthetic"),
                    "nsfw": item.get("nsfw"),
                    "sscd_cluster_id": item.get("cluster"),
                },
            }
            if dinos is not None:
                sample["dino.npy"] = tensor_to_storage_npy(dinos[i], args.dino_dtype)
            sink.write(sample)
            n_written += 1

    import concurrent.futures

    # ----------------------------------------------------------------------- #
    # --mix mode: round-robin rows across subsets so each OUTPUT shard is a
    # proportional blend (good per-batch diversity at train time).
    # ----------------------------------------------------------------------- #
    if args.mix:
        specs = parse_mix(args.mix)
        per_rank_total = -(-args.mix_target // world)  # ceil division
        targets = {s: max(1, int(round(w * per_rank_total))) for s, w in specs}
        counters: dict[str, Any] = {
            "seen": 0, "filtered": 0, "no_latent": 0, "no_caption": 0,
            "caption_src": caption_src_counts,
        }
        dl_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max(4, min(16, len(specs) * (args.mix_prefetch_per_subset + 1)))
        )

        def make_gen(subset: str):
            plist = list_source_parquets(args.repo_id, subset, token_str)[rank::world]
            if args.max_source_shards is not None:
                plist = plist[: args.max_source_shards]
            if not plist:
                raise FileNotFoundError(f"no parquets for subset {subset!r} at rank {rank}")
            return iter_subset_items(
                subset, plist, args=args, token_str=token_str, staging_dir=staging_dir,
                dl_executor=dl_executor, caption_keys=caption_keys, extra_cols=extra_cols,
                rank=rank, seen_clusters=seen_clusters, counters=counters,
            )

        gens = {s: make_gen(s) for s, _ in specs}
        written_per = {s: 0 for s, _ in specs}
        active = {s for s, _ in specs}
        buffer: list[dict[str, Any]] = []
        print(f"[monet-cache rank {rank}/{world}] MIX targets (per rank): "
              + ", ".join(f"{s}={targets[s]}" for s, _ in specs), flush=True)

        def flush() -> None:
            nonlocal buffer
            if buffer:
                encode_and_write(buffer)
                buffer = []
            uploader.poll()
            while len(list(out_dir.glob(shard_glob))) > args.max_pending_uploads:
                uploader.poll(wait=True)

        next_log = 50000
        while active:
            # Pull from whichever active subset is furthest BELOW its target ratio,
            # which keeps the running mix proportional even as subsets exhaust.
            s = min(active, key=lambda x: written_per[x] / targets[x])
            try:
                item = next(gens[s])
            except StopIteration:
                if args.mix_allow_repeat and written_per[s] < targets[s]:
                    gens[s] = make_gen(s)  # upsample: re-iterate this subset
                    try:
                        item = next(gens[s])
                    except StopIteration:
                        active.discard(s)
                        continue
                else:
                    if written_per[s] < targets[s]:
                        print(f"[monet-cache rank {rank}] subset {s} exhausted at "
                              f"{written_per[s]}/{targets[s]} (no --mix-allow-repeat)", flush=True)
                    active.discard(s)
                    continue
            buffer.append(item)
            written_per[s] += 1
            if written_per[s] >= targets[s]:
                active.discard(s)
            if len(buffer) >= args.encode_batch:
                flush()
            total_w = sum(written_per.values())
            if total_w >= next_log:
                next_log += 50000
                mix_now = ", ".join(f"{s}={written_per[s]}" for s, _ in specs)
                print(f"[monet-cache rank {rank}] mix progress: written={total_w} "
                      f"[{mix_now}] seen={counters['seen']} filtered={counters['filtered']}",
                      flush=True)
        flush()
        sink.close()
        dl_executor.shutdown(wait=True)
        uploader.finish()
        src_summary = ", ".join(f"{k}={v}" for k, v in caption_src_counts.most_common()) or "none"
        print(
            f"[monet-cache rank {rank}] MIX DONE: {n_written} written "
            + ", ".join(f"{s}={written_per[s]}" for s, _ in specs)
            + f" | caption_src=[{src_summary}] seen={counters['seen']} "
            f"filtered={counters['filtered']} no_latent={counters['no_latent']} "
            f"no_caption={counters['no_caption']}",
            flush=True,
        )
        return 0

    # Prefetch source parquets ahead of the GPU so it never blocks on the ~600 MB
    # download at each shard boundary. Window = args.prefetch_shards (bounded disk).
    dl_executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=max(1, min(4, args.prefetch_shards))
    )
    dl_futures: dict[int, concurrent.futures.Future] = {}

    def submit_download(idx: int) -> None:
        if 0 <= idx < len(my_parquets) and idx not in dl_futures:
            dl_futures[idx] = dl_executor.submit(
                hf_hub_download,
                repo_id=args.repo_id, filename=my_parquets[idx],
                repo_type="dataset", token=token_str,
                # Download to a real staging dir (NOT the HF blob cache); os.remove on the
                # returned file then actually frees disk. With the default cache, removing the
                # snapshot symlink leaves the blob behind and disk grows unbounded.
                local_dir=str(staging_dir),
            )

    for idx in range(min(args.prefetch_shards, len(my_parquets))):
        submit_download(idx)

    for shard_idx, parquet_path in enumerate(my_parquets):
        local = dl_futures.pop(shard_idx).result()   # wait only if not prefetched yet
        submit_download(shard_idx + args.prefetch_shards)  # keep the window full
        available = set(pq.read_schema(local).names)
        # Resolve columns against THIS shard's schema (subsets differ); request only
        # what exists so a missing fallback caption / dino column can't crash read_table.
        latent_col = args.latent_key if args.latent_key in available else next(
            (c for c in sorted(available) if "vae" in c.lower()), None
        )
        if latent_col is None:
            raise RuntimeError(f"No VAE-latent column found in {parquet_path}; have {sorted(available)}")
        present_caption_keys = [c for c in caption_keys if c in available]
        dino_col = args.dino_key if (not args.no_dino and args.dino_key in available) else None
        want = [latent_col, *present_caption_keys, *[c for c in extra_cols if c in available]]
        if dino_col:
            want.append(dino_col)
        want = list(dict.fromkeys(want))  # dedupe, preserve order

        table = pq.read_table(local, columns=want)
        cols = {c: table.column(c).to_pylist() for c in table.column_names}
        n_rows = table.num_rows
        buffer: list[dict[str, Any]] = []
        for i in range(n_rows):
            n_seen += 1
            row = {c: cols[c][i] for c in cols}
            if not passes_filters(row, args, seen_clusters):
                n_filtered += 1
                continue
            latent = row.get(latent_col)
            if latent is None:
                n_no_latent += 1
                continue
            caption = ""
            caption_src = None
            for ck in present_caption_keys:
                v = row.get(ck)
                if v:
                    caption, caption_src = v, ck
                    break
            if not caption:
                n_no_caption += 1
                continue
            caption_src_counts[caption_src] += 1
            key = row.get("__key__") or row.get("id") or f"r{rank}s{shard_idx}n{i}"
            item = {
                "key": str(key),
                "latent": latent,
                "caption": caption,
                "caption_src": caption_src,
                "source": row.get("source"),
                "aesthetic": row.get("aesthetic_laion-improved"),
                "nsfw": row.get("nsfw_jasperai"),
                "cluster": row.get("sscd_cluster_id"),
            }
            if dino_col:
                item["dino"] = row.get(dino_col)
            buffer.append(item)
            if len(buffer) >= args.encode_batch:
                encode_and_write(buffer)
                buffer = []
        encode_and_write(buffer)

        try:
            os.remove(local)  # staging cleanup: drop the source parquet after read
        except FileNotFoundError:
            pass
        uploader.poll()
        # Backpressure: if uploads lag, don't let finished output shards pile up on disk.
        # Count closed-but-present tars (the open one stays); wait for uploads to drain.
        while len(list(out_dir.glob(shard_glob))) > args.max_pending_uploads:
            uploader.poll(wait=True)
        # Report the captioners ACTUALLY used (not just the preferred key), so a
        # present-but-null column like Gemini on commoncatalog-cc-by is visible here.
        src_summary = ", ".join(f"{k}={v}" for k, v in caption_src_counts.most_common()) or "none"
        print(
            f"[monet-cache rank {rank}] shard {shard_idx + 1}/{len(my_parquets)} "
            f"({parquet_path}): caption_src=[{src_summary}] seen={n_seen} "
            f"written={n_written} filtered={n_filtered} no_latent={n_no_latent} no_caption={n_no_caption}",
            flush=True,
        )

    sink.close()
    dl_executor.shutdown(wait=True)
    uploader.finish()
    print(
        f"[monet-cache rank {rank}] DONE: {n_written} samples written "
        f"(filtered={n_filtered} no_latent={n_no_latent} no_caption={n_no_caption} of {n_seen} seen)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
