"""Verify whether jasperai/monet's precomputed SANA-VAE latents are drop-in
compatible with Boomer's DC-AE f32c32 decoder.

The question this answers: monet stored latents encoded with SANA-VAE. SANA-VAE
*is* DC-AE f32c32 -- the same family as Boomer's
`mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers`. If the checkpoint + scaling
convention truly match, we can reuse monet's latents directly and skip the
entire image-download + GPU-VAE-encode half of the caching pipeline.

Three checks, in order of decisiveness:

  1. CHANNELS -- DC-AE f32c32 latents have exactly 32 channels and an HxW grid
     that is image_res / 32. If monet's stored latent is not 32-channel, it is a
     different VAE and the latents are unusable. (Fast, decisive, no GPU.)

  2. DECODE -- run monet's stored latent through Boomer's DC-AE decoder under
     both possible storage conventions (latent stored already-scaled vs. raw).
     A matching latent space reconstructs a clean image; a mismatch yields
     noise/garbage. PNGs are written for visual confirmation.

  3. RE-ENCODE (optional, --with-image) -- when an aligned full-res source image
     is present (webdataset config), re-encode it with Boomer's VAE and compare
     numerically to the stored latent. This catches subtle scaling / checkpoint
     drift that the eyeball decode test can miss. Requires the SAME image the
     latent was computed from at the SAME resolution, so it is meaningful only
     for the full-res webdataset config, NOT the 384px-thumbnail parquet config.

Usage (lightweight, streams a handful of samples -- does NOT pull 68 TB):

    python -m tools.verify_monet_vae_compat \
        --config <parquet-or-wds-config-name> \
        --num-samples 4 \
        --out-dir /tmp/monet_vae_check \
        --hf-token "$HF_TOKEN"

If you do not know the latent/image field names, run once with --list-keys to
print the schema of the first sample, then pass --latent-key / --image-key.
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from boomer.sana_latent_cache import (
    CacheConfig,
    decode_image_latent,
    encode_image_latent,
    hf_token_value,
    image_to_tensor,
    load_vae,
    tensor_to_image,
    vae_scaling_factor,
)

DC_AE_DOWNSAMPLE = 32  # f32 -> 32x spatial compression
DC_AE_CHANNELS = 32    # c32 -> 32 latent channels


# --------------------------------------------------------------------------- #
# Latent / image field parsing (monet stores arrays in a few possible forms)
# --------------------------------------------------------------------------- #
def _looks_like_latent_key(key: str) -> bool:
    k = key.lower()
    return ("latent" in k or "vae" in k) and "clip" not in k and "dino" not in k


def _looks_like_image_key(key: str) -> bool:
    k = key.lower()
    return any(t in k for t in ("image", "jpg", "jpeg", "png", "webp", "thumb"))


def autodetect_key(sample: dict[str, Any], predicate) -> str | None:
    for key in sample:
        if predicate(key):
            return key
    return None


def to_latent_tensor(value: Any) -> torch.Tensor:
    """Coerce a stored latent (npy bytes / list / ndarray / dict) into a 4D
    tensor shaped (1, C, H, W)."""
    arr: np.ndarray
    if isinstance(value, (bytes, bytearray)):
        arr = np.load(io.BytesIO(bytes(value)), allow_pickle=False)
    elif isinstance(value, dict) and "bytes" in value:
        arr = np.load(io.BytesIO(value["bytes"]), allow_pickle=False)
    elif isinstance(value, np.ndarray):
        arr = value
    elif isinstance(value, torch.Tensor):
        arr = value.detach().cpu().numpy()
    elif isinstance(value, (list, tuple)):
        arr = np.asarray(value)
    else:
        raise TypeError(f"Unrecognized latent storage type: {type(value)!r}")

    tensor = torch.from_numpy(np.ascontiguousarray(arr)).float()
    # Normalize rank to (1, C, H, W).
    if tensor.ndim in (1, 2):
        # monet stores the latent flattened into a faiss-style vector. Reshape
        # back to a DC-AE f32c32 grid: C=32, square spatial = sqrt(N/32).
        flat = tensor.flatten()
        n = flat.numel()
        if n % DC_AE_CHANNELS != 0:
            raise ValueError(f"flat latent length {n} not divisible by {DC_AE_CHANNELS} channels")
        cells = n // DC_AE_CHANNELS
        side = int(round(cells ** 0.5))
        if side * side != cells:
            raise ValueError(f"flat latent {n} -> {cells} cells is not a square grid")
        tensor = flat.reshape(1, DC_AE_CHANNELS, side, side)
    elif tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim == 4:
        pass
    else:
        raise ValueError(f"Latent has unexpected rank {tensor.ndim}, shape {tuple(tensor.shape)}")
    return tensor


def to_pil_image(value: Any):
    from PIL import Image

    if isinstance(value, Image.Image):
        return value
    if isinstance(value, (bytes, bytearray)):
        return Image.open(io.BytesIO(bytes(value)))
    if isinstance(value, dict) and "bytes" in value:
        return Image.open(io.BytesIO(value["bytes"]))
    if isinstance(value, dict) and "path" in value and value["path"]:
        return Image.open(value["path"])
    raise TypeError(f"Unrecognized image storage type: {type(value)!r}")


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #
def channel_check(latent: torch.Tensor) -> tuple[bool, str]:
    _, c, h, w = latent.shape
    ok = c == DC_AE_CHANNELS
    implied_res = (h * DC_AE_DOWNSAMPLE, w * DC_AE_DOWNSAMPLE)
    msg = (
        f"latent shape {tuple(latent.shape)}  ->  channels={c} "
        f"({'OK, DC-AE f32c32' if ok else f'MISMATCH, expected {DC_AE_CHANNELS}'}); "
        f"grid {h}x{w} implies source image ~{implied_res[0]}x{implied_res[1]}px"
    )
    return ok, msg


def latent_stats(latent: torch.Tensor) -> str:
    flat = latent.flatten()
    return (
        f"min={flat.min():+.3f} max={flat.max():+.3f} "
        f"mean={flat.mean():+.4f} std={flat.std():.4f} "
        f"abs_mean={flat.abs().mean():.4f}"
    )


def decode_both_conventions(vae, latent: torch.Tensor, cfg: CacheConfig, sf: float):
    """Boomer's decode_image_latent divides by sf before vae.decode. So:
      - if monet stored the SCALED latent (raw*sf), decode_image_latent is correct.
      - if monet stored the RAW latent (no sf), we must NOT divide, i.e. feed
        latent*sf into decode_image_latent so the internal /sf cancels.
    Return (img_if_stored_scaled, img_if_stored_raw)."""
    img_scaled = decode_image_latent(vae, latent, cfg)              # treats stored as scaled
    img_raw = decode_image_latent(vae, latent * sf, cfg)            # treats stored as raw
    return img_scaled, img_raw


def reencode_compare(vae, image, latent: torch.Tensor, cfg: CacheConfig, sf: float) -> str:
    """Re-encode the source image with Boomer's VAE and compare to the stored
    latent under both conventions. Lower error => that convention matches."""
    _, _, h, w = latent.shape
    res = h * DC_AE_DOWNSAMPLE  # match the resolution the stored latent implies
    img_tensor = image_to_tensor(image, res)
    boomer_scaled = encode_image_latent(vae, img_tensor, cfg).float().cpu()  # = raw*sf
    boomer_raw = boomer_scaled / sf

    if boomer_scaled.shape != latent.shape:
        return (
            f"re-encode shape {tuple(boomer_scaled.shape)} != stored {tuple(latent.shape)} "
            f"(resolution/crop mismatch -- numerical compare skipped)"
        )

    stored = latent.float().cpu()

    def err(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
        rel_l2 = (a - b).norm().item() / (b.norm().item() + 1e-8)
        cos = torch.nn.functional.cosine_similarity(
            a.flatten().unsqueeze(0), b.flatten().unsqueeze(0)
        ).item()
        return rel_l2, cos

    rel_s, cos_s = err(stored, boomer_scaled)
    rel_r, cos_r = err(stored, boomer_raw)
    return (
        f"vs Boomer re-encode  [stored==scaled? relL2={rel_s:.3f} cos={cos_s:+.3f}] "
        f"[stored==raw? relL2={rel_r:.3f} cos={cos_r:+.3f}]  "
        f"(cos>0.97 on either => weights match; rel_l2 small => convention found)"
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="jasperai/monet")
    p.add_argument("--config", default=None, help="dataset config name (parquet or webdataset)")
    p.add_argument("--split", default="train")
    p.add_argument("--num-samples", type=int, default=4)
    p.add_argument("--vae-id", default=CacheConfig.vae_id)
    p.add_argument("--vae-subfolder", default=None,
                   help="VAE subfolder, e.g. 'vae' for a full SANA pipeline repo")
    p.add_argument("--scaling-factor", type=float, default=CacheConfig.scaling_factor)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="bf16")
    p.add_argument("--latent-key", default=None, help="override autodetected latent field")
    p.add_argument("--image-key", default=None, help="override autodetected image field")
    p.add_argument("--with-image", action="store_true",
                   help="also run the re-encode numerical compare (needs aligned full-res image)")
    p.add_argument("--list-keys", action="store_true",
                   help="print the first sample's schema and exit")
    p.add_argument("--parquet-file", default=None,
                   help="read samples from a local .parquet file instead of streaming "
                        "(more reliable; download one shard with hf_hub_download first)")
    p.add_argument("--out-dir", default="/tmp/monet_vae_check")
    p.add_argument("--hf-token", default=None)
    return p.parse_args()


def load_samples(args, token):
    """Yield sample dicts either from a local parquet shard (preferred, no
    streaming dependency) or by streaming the HF dataset."""
    if args.parquet_file:
        import pyarrow.parquet as pq

        print(f"[monet-check] reading local parquet {args.parquet_file}")
        table = pq.read_table(args.parquet_file)
        rows = table.slice(0, args.num_samples).to_pylist()
        if not rows:
            raise RuntimeError("parquet file has no rows")
        return rows[0], iter(rows[1:])

    from datasets import load_dataset

    print(f"[monet-check] streaming {args.dataset} config={args.config} split={args.split}")
    ds = load_dataset(
        args.dataset,
        name=args.config,
        split=args.split,
        streaming=True,
        token=token if isinstance(token, str) else None,
    )
    iterator = iter(ds)
    return next(iterator), iterator


def main() -> int:
    args = parse_args()

    token = hf_token_value(args.hf_token)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    first, iterator = load_samples(args, token)

    if args.list_keys:
        print("[monet-check] first-sample schema:")
        for k, v in first.items():
            kind = type(v).__name__
            extra = ""
            if isinstance(v, (bytes, bytearray)):
                extra = f" ({len(v)} bytes)"
            elif isinstance(v, (list, tuple)):
                extra = f" (len {len(v)})"
            print(f"    {k:32s} {kind}{extra}")
        return 0

    latent_key = args.latent_key or autodetect_key(first, _looks_like_latent_key)
    image_key = args.image_key or autodetect_key(first, _looks_like_image_key)
    if latent_key is None:
        print("[monet-check] FATAL: could not autodetect a latent field. "
              "Re-run with --list-keys and pass --latent-key.", file=sys.stderr)
        return 2
    print(f"[monet-check] latent field = {latent_key!r}; image field = {image_key!r}")

    cfg = CacheConfig(vae_id=args.vae_id, vae_subfolder=args.vae_subfolder,
                      device=args.device, dtype=args.dtype,
                      scaling_factor=args.scaling_factor)
    print(f"[monet-check] loading Boomer VAE {cfg.vae_id} on {cfg.device} ...")
    vae = load_vae(cfg, hf_token=token)
    sf = vae_scaling_factor(vae, cfg.scaling_factor)
    cfg_sf = getattr(getattr(vae, "config", None), "scaling_factor", None)
    print(f"[monet-check] Boomer scaling_factor in use = {sf:.5f} "
          f"(vae.config.scaling_factor={cfg_sf}; Boomer default={cfg.scaling_factor})")

    samples = [first]
    for _ in range(args.num_samples - 1):
        try:
            samples.append(next(iterator))
        except StopIteration:
            break

    all_channels_ok = True
    for i, sample in enumerate(samples):
        print(f"\n[monet-check] ---- sample {i} ----")
        try:
            latent = to_latent_tensor(sample[latent_key])
        except Exception as exc:  # noqa: BLE001
            print(f"  could not parse latent: {exc}")
            all_channels_ok = False
            continue

        ok, msg = channel_check(latent)
        all_channels_ok &= ok
        print(f"  {msg}")
        print(f"  stats: {latent_stats(latent)}")

        try:
            img_scaled, img_raw = decode_both_conventions(vae, latent, cfg, sf)
            p_scaled = out_dir / f"sample{i}_decoded_assuming_stored_SCALED.png"
            p_raw = out_dir / f"sample{i}_decoded_assuming_stored_RAW.png"
            tensor_to_image(img_scaled).save(p_scaled)
            tensor_to_image(img_raw).save(p_raw)
            print(f"  decoded -> {p_scaled.name} , {p_raw.name}  (open both; the clean one reveals the convention)")
        except Exception as exc:  # noqa: BLE001
            print(f"  decode failed: {exc}")

        if image_key and image_key in sample:
            try:
                src = to_pil_image(sample[image_key])
                src.convert("RGB").save(out_dir / f"sample{i}_source.png")
            except Exception as exc:  # noqa: BLE001
                print(f"  (could not save source image: {exc})")

        if args.with_image and image_key and image_key in sample:
            try:
                print(f"  {reencode_compare(vae, to_pil_image(sample[image_key]), latent, cfg, sf)}")
            except Exception as exc:  # noqa: BLE001
                print(f"  re-encode compare failed: {exc}")

    print("\n[monet-check] ===== VERDICT =====")
    if all_channels_ok:
        print("  CHANNELS: PASS -- all latents are 32-channel DC-AE f32c32 grids.")
        print("  Next: open the decoded PNGs in", out_dir)
        print("    * If the *_SCALED.png look clean   -> monet stores already-scaled latents; drop-in.")
        print("    * If the *_RAW.png look clean      -> monet stores raw latents; multiply by sf on load.")
        print("    * If BOTH look like noise          -> same channel count but different DC-AE checkpoint;")
        print("                                          re-encode, or switch Boomer's VAE to SANA-1.5's.")
        print("  For a numeric confirmation on full-res (webdataset) samples, re-run with --with-image.")
    else:
        print("  CHANNELS: FAIL -- at least one latent is not 32-channel.")
        print("  monet's latents are NOT DC-AE f32c32 compatible as stored; do not reuse them directly.")
    return 0 if all_channels_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
