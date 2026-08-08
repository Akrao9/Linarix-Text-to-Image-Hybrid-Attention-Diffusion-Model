"""Which attention backends actually work on THIS GPU, at Boomer's three shapes?

v8 has three distinct attention sites and they are not equally flash-eligible:

  * DDT decoder self-attention (6 blocks) — image tokens only, NO mask. Already routed
    by --image-attn-backend {sdpa,flash2,flash3,flash4}, so a working flash kernel is a
    flag change with no code.
  * Joint attention (4 dual-stream + 4 refresh) — [text; image], padded-text key mask.
    Dense flash kernels take no arbitrary mask; the varlen API with seqused_k does, but
    only if valid tokens form a PREFIX, which needs the concat reordered to
    [image; text] so the padding lands at the end.
  * Cross-attention (14 blocks) — same masking situation as joint.

This probe answers, per backend: does it import, does it run fwd+bwd in bf16 on this
device, does it agree with SDPA, and is it faster. Run before wiring any kernel in:

    python tools/attn_backend_probe.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from boomer.models.boomer_fla_dit import (  # noqa: E402
    maybe_add_sibling_flash_attention_repo,
)

maybe_add_sibling_flash_attention_repo()

B, HEADS, HEAD_DIM = 56, 18, 64
IMAGE_TOKENS, TEXT_TOKENS = 256, 320
DTYPE = torch.bfloat16
RTOL = 2e-2


def _load_backends() -> dict:
    out = {}
    try:
        from flash_attn import flash_attn_func as fa2
        from flash_attn import flash_attn_varlen_func as fa2_varlen

        out["flash2"] = fa2
        out["flash2_varlen"] = fa2_varlen
    except Exception as exc:
        out["flash2"] = f"import failed: {type(exc).__name__}: {exc}"
        out["flash2_varlen"] = out["flash2"]
    try:
        from flash_attn.cute.interface import flash_attn_func as fa4
        from flash_attn.cute.interface import flash_attn_varlen_func as fa4_varlen

        out["flash4"] = fa4
        out["flash4_varlen"] = fa4_varlen
    except Exception as exc:
        out["flash4"] = f"import failed: {type(exc).__name__}: {exc}"
    return out


def _bench(fn, iters: int = 20) -> float:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1000


def probe_decoder_site(backends: dict) -> None:
    """Unmasked self-attention over image tokens — the DDT decoder's shape."""
    print(f"\n=== DDT decoder self-attention: B={B} S={IMAGE_TOKENS} "
          f"H={HEADS} D={HEAD_DIM}, no mask ===")
    q, k, v = (torch.randn(B, IMAGE_TOKENS, HEADS, HEAD_DIM, device="cuda",
                           dtype=DTYPE, requires_grad=True) for _ in range(3))

    def sdpa():
        o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                           v.transpose(1, 2)).transpose(1, 2)
        o.sum().backward()
        return o

    ref = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                         v.transpose(1, 2)).transpose(1, 2).detach()
    print(f"  sdpa            {_bench(sdpa):7.2f} ms (fwd+bwd)")

    for name in ("flash2", "flash4"):
        fn = backends.get(name)
        if not callable(fn):
            print(f"  {name:<15} UNAVAILABLE — {fn}")
            continue
        try:
            out = fn(q, k, v)
            out = out[0] if isinstance(out, tuple) else out
            err = (out.detach().float() - ref.float()).abs().max().item()
            out.sum().backward()

            def run(_fn=fn):
                o = _fn(q, k, v)
                o = o[0] if isinstance(o, tuple) else o
                o.sum().backward()
            print(f"  {name:<15} {_bench(run):7.2f} ms (fwd+bwd), max|Δ| vs sdpa {err:.4f}")
        except Exception as exc:
            print(f"  {name:<15} FAILED on this device — {type(exc).__name__}: {exc}")


def probe_joint_site(backends: dict) -> None:
    """Masked [text; image] joint attention, and whether FA4 varlen can replace it.

    The varlen route needs valid tokens as a prefix, so the sequence is reordered to
    [image; text]: image is always valid, padded text lands at the tail, and
    seqused_k = IMAGE_TOKENS + caption_len. Attention is permutation-invariant over
    keys, so reordering does not change the result.
    """
    seq = IMAGE_TOKENS + TEXT_TOKENS
    print(f"\n=== joint attention: B={B} S={seq} (image {IMAGE_TOKENS} + text "
          f"{TEXT_TOKENS}), padded-text mask ===")
    torch.manual_seed(0)
    text_len = torch.randint(20, TEXT_TOKENS + 1, (B,), device="cuda")
    q, k, v = (torch.randn(B, seq, HEADS, HEAD_DIM, device="cuda",
                           dtype=DTYPE, requires_grad=True) for _ in range(3))

    positions = torch.arange(TEXT_TOKENS, device="cuda")[None, :]
    text_valid = positions < text_len[:, None]
    valid = torch.cat([torch.ones(B, IMAGE_TOKENS, dtype=torch.bool, device="cuda"),
                       text_valid], dim=1)
    bias = valid[:, None, None, :]

    def sdpa():
        o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                           v.transpose(1, 2), attn_mask=bias).transpose(1, 2)
        o.sum().backward()
        return o

    ref = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=bias,
    ).transpose(1, 2).detach()
    print(f"  sdpa (masked)   {_bench(sdpa):7.2f} ms (fwd+bwd)")

    fn = backends.get("flash4_varlen")
    if not callable(fn):
        print(f"  flash4 varlen   UNAVAILABLE — {backends.get('flash4')}")
        return
    seqused = (IMAGE_TOKENS + text_len).to(torch.int32)
    try:
        out = fn(q, k, v, seqused_q=seqused, seqused_k=seqused,
                 max_seqlen_q=seq, max_seqlen_k=seq)
        out = out[0] if isinstance(out, tuple) else out
        # compare only where the query is a real token
        m = valid[..., None, None]
        err = ((out.detach().float() - ref.float()).abs() * m).max().item()
        out.sum().backward()

        def run():
            o = fn(q, k, v, seqused_q=seqused, seqused_k=seqused,
                   max_seqlen_q=seq, max_seqlen_k=seq)
            o = o[0] if isinstance(o, tuple) else o
            o.sum().backward()
        ms = _bench(run)
        ok = "MATCHES" if err < RTOL else f"DIVERGES (max|Δ|={err:.4f})"
        print(f"  flash4 varlen   {ms:7.2f} ms (fwd+bwd), {ok} sdpa on valid tokens")
        if err < RTOL:
            print("    -> viable: wiring joint attention through FA4 varlen requires "
                  "reordering the concat to [image; text].")
    except Exception as exc:
        print(f"  flash4 varlen   FAILED — {type(exc).__name__}: {exc}")


def _pack(tensors, valid: torch.Tensor):
    """Padded [B, S, H, D] -> packed [total, H, D] plus cu_seqlens, for the varlen APIs.

    FA2's varlen takes cu_seqlens over a PACKED buffer; it has no seqused_k and no
    concept of a padded batch. FA4 accepts either. Valid tokens are already a prefix per
    row (the sequence is [image; text] with padding at the tail), so a nonzero() over the
    flattened mask yields exactly the packed order.
    """
    b, s = valid.shape
    index = valid.reshape(-1).nonzero(as_tuple=True)[0]
    packed = [t.reshape(b * s, *t.shape[2:]).index_select(0, index) for t in tensors]
    lengths = valid.sum(dim=1)
    cu = torch.zeros(b + 1, dtype=torch.int32, device=valid.device)
    cu[1:] = lengths.cumsum(0).to(torch.int32)
    return packed, cu, index, int(lengths.max())


def _unpack(packed: torch.Tensor, index: torch.Tensor, b: int, s: int) -> torch.Tensor:
    out = packed.new_zeros(b * s, *packed.shape[1:])
    out.index_copy_(0, index, packed)
    return out.view(b, s, *packed.shape[1:])


def probe_joint_packed(backends: dict) -> None:
    """The joint site again, but through the PACKED varlen API both kernels accept.

    Two things this separates that the seqused probe cannot:

      * whether FA4 on this GPU is broken generally, or only on the padded-batch +
        seqused path (the newest, least-exercised combination, and the one v8 hit an
        illegal memory access on). If packed works and seqused does not, that is a
        kernel bug on one path, not "FA4 does not run on sm_120".
      * whether FA2 is usable here at all. It is the only route for FA2, since FA2 has
        no seqused_k -- so the gather/scatter is not optional overhead, it is the price
        of admission, and it is timed separately below.
    """
    seq = IMAGE_TOKENS + TEXT_TOKENS
    print(f"\n=== joint attention via PACKED varlen (cu_seqlens): B={B} S={seq} ===")
    torch.manual_seed(0)
    text_len = torch.randint(20, TEXT_TOKENS + 1, (B,), device="cuda")
    q, k, v = (torch.randn(B, seq, HEADS, HEAD_DIM, device="cuda",
                           dtype=DTYPE, requires_grad=True) for _ in range(3))
    positions = torch.arange(TEXT_TOKENS, device="cuda")[None, :]
    valid = torch.cat([torch.ones(B, IMAGE_TOKENS, dtype=torch.bool, device="cuda"),
                       positions < text_len[:, None]], dim=1)

    ref = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        attn_mask=valid[:, None, None, :],
    ).transpose(1, 2).detach()

    def pack_only():
        (pq, pk, pv), cu, index, maxlen = _pack((q, k, v), valid)
        _unpack(pq, index, B, seq)

    print(f"  pack+unpack alone {_bench(pack_only):7.2f} ms  "
          f"(paid on every joint site, 8x per forward, plus backward)")

    for name, fn in (("flash2 varlen", backends.get("flash2_varlen")),
                     ("flash4 varlen", backends.get("flash4_varlen"))):
        if not callable(fn):
            print(f"  {name:<15} UNAVAILABLE — {fn}")
            continue
        try:
            (pq, pk, pv), cu, index, maxlen = _pack((q, k, v), valid)
            out = fn(pq, pk, pv, cu_seqlens_q=cu, cu_seqlens_k=cu,
                     max_seqlen_q=maxlen, max_seqlen_k=maxlen)
            out = out[0] if isinstance(out, tuple) else out
            full = _unpack(out, index, B, seq)
            err = ((full.detach().float() - ref.float()).abs()
                   * valid[..., None, None]).max().item()
            full.sum().backward()

            def run(_fn=fn):
                (a, b_, c), cu_, idx, ml = _pack((q, k, v), valid)
                o = _fn(a, b_, c, cu_seqlens_q=cu_, cu_seqlens_k=cu_,
                        max_seqlen_q=ml, max_seqlen_k=ml)
                o = o[0] if isinstance(o, tuple) else o
                _unpack(o, idx, B, seq).sum().backward()

            ok = "MATCHES" if err < RTOL else f"DIVERGES (max|Δ|={err:.4f})"
            print(f"  {name:<15} {_bench(run):7.2f} ms (fwd+bwd incl. pack), {ok} sdpa")
        except Exception as exc:
            print(f"  {name:<15} FAILED — {type(exc).__name__}: {exc}")


def main() -> None:
    major, minor = torch.cuda.get_device_capability()
    print(f"device: {torch.cuda.get_device_name()}  sm_{major}{minor}  "
          f"torch {torch.__version__}")
    print("NOTE: FA4/CuTe targets datacenter Blackwell (sm_100); sm_120 is workstation "
          "Blackwell and is not the same target.")
    backends = _load_backends()
    for name in ("flash2", "flash4"):
        status = backends.get(name)
        print(f"  {name}: {'imported' if callable(status) else status}")
    probe_decoder_site(backends)
    probe_joint_site(backends)
    probe_joint_packed(backends)
    print("\nIf a backend FAILED or DIVERGES here, do not route the model through it.")
    print("Reading the joint results: the decoder site is DENSE (no mask), so it exercises")
    print("a different kernel from either varlen path. A backend can pass there and still")
    print("fail both joint probes — that is a usable result, it means --image-attn-backend")
    print("can take it while --joint-attn-backend cannot.")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("needs the training GPU")
    main()
