"""BF16 parity + timing for FLA's fused_attnres at Boomer's v8 shapes.

FLA's own parity suite (tests/ops/test_attnres.py) covers fp16/fp32 up to L=29,
T=8000, D=7186 — but not bf16, which is what Boomer trains in. This checks the
fused Triton kernel against fla.ops.attnres.naive_attnres (the reference the suite
itself trusts) forward AND backward at bf16, D=1152, L=1..9, and the actual v8
batch shapes: (B=56, T=256) for the 512 phase and (B=14, T=1024) for 1024.

Run on the training GPU (SM120) BEFORE the long run:

    python tools/attnres_parity.py

Optionally benchmark the Gluon backend separately (do not enable it for the full
run without this):

    FLA_ATTNRES_GLUON=1 python tools/attnres_parity.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from boomer.models.boomer_fla_dit import maybe_add_sibling_fla_repo  # noqa: E402

maybe_add_sibling_fla_repo()

from fla.ops.attnres import fused_attnres, naive_attnres  # noqa: E402

try:
    from fla.utils import assert_close
except ImportError:
    def assert_close(name, ref, tri, ratio):  # bf16-appropriate fallback
        torch.testing.assert_close(tri, ref, rtol=ratio, atol=ratio)


DEVICE = "cuda"
DTYPE = torch.bfloat16
D = 1152
SHAPES = [(56, 256), (14, 1024)]  # (B, T): v8 512 phase, v8 1024 phase
MAX_SOURCES = 9  # v8 site 21 / final z site with group ends 2,5,8,11,14,17,19,21
RATIO = 8e-3  # bf16 has 8 mantissa bits; FLA's fp16 suite uses per-op ratios of this order


def make_case(n_sources: int, batch: int, tokens: int):
    residuals = [
        torch.randn(batch, tokens, D, device=DEVICE, dtype=DTYPE, requires_grad=True)
        for _ in range(n_sources)
    ]
    query = torch.randn(D, device=DEVICE, dtype=DTYPE, requires_grad=True) * 0.02
    query = query.detach().requires_grad_(True)
    rms_w = torch.ones(D, device=DEVICE, dtype=DTYPE, requires_grad=True)
    return query, residuals, rms_w


def clone_case(query, residuals, rms_w):
    q = query.detach().clone().requires_grad_(True)
    rs = [r.detach().clone().requires_grad_(True) for r in residuals]
    w = rms_w.detach().clone().requires_grad_(True)
    return q, rs, w


def run_parity() -> None:
    torch.manual_seed(0)
    for batch, tokens in SHAPES:
        # with_onorm covers the final z site, which fuses its learned output RMSNorm
        # into the kernel via output_rms_weight; per-block sites pass None.
        for with_onorm in (False, True):
            for n in range(1, MAX_SOURCES + 1):
                q0, r0, w0 = make_case(n, batch, tokens)
                q1, r1, w1 = clone_case(q0, r0, w0)
                ow0 = ow1 = None
                if with_onorm:
                    ow0 = torch.ones(D, device=DEVICE, dtype=DTYPE, requires_grad=True)
                    ow1 = ow0.detach().clone().requires_grad_(True)

                ref = naive_attnres(
                    query=q0, residuals=r0, rms_weight=w0,
                    output_rms_weight=ow0, rms_eps=1e-6,
                )
                out = fused_attnres(
                    query=q1, residuals=r1, rms_weight=w1,
                    output_rms_weight=ow1, rms_eps=1e-6,
                    checkpoint_level=1,
                )
                grad = torch.randn_like(ref)
                ref.backward(grad)
                out.backward(grad)

                tag = f"L={n},B={batch},T={tokens},onorm={with_onorm}"
                assert_close(f"o[{tag}]", ref, out, RATIO)
                assert_close(f"dq[{tag}]", q0.grad, q1.grad, RATIO)
                assert_close(f"dw[{tag}]", w0.grad, w1.grad, RATIO)
                if with_onorm:
                    assert_close(f"dow[{tag}]", ow0.grad, ow1.grad, RATIO)
                for i, (a, b) in enumerate(zip(r0, r1)):
                    assert_close(f"dv{i}[{tag}]", a.grad, b.grad, RATIO)
        print(f"parity OK: B={batch} T={tokens} D={D} bf16, L=1..{MAX_SOURCES} "
              f"fwd+bwd, with and without fused output RMSNorm")


def run_timing(iters: int = 50) -> None:
    """One full v8 step's worth of routing calls (site source counts for 22 encoder
    blocks + final z site under group ends 2,5,8,11,14,17,19,21)."""
    site_counts = [1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 9, 9]
    for batch, tokens in SHAPES:
        cases = [make_case(n, batch, tokens) for n in site_counts if n > 1]
        # The last routed call is the final z site, which fuses its output RMSNorm.
        onorm = torch.ones(D, device=DEVICE, dtype=DTYPE, requires_grad=True)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iters):
            for idx, (q, rs, w) in enumerate(cases):
                out = fused_attnres(
                    query=q, residuals=rs, rms_weight=w,
                    output_rms_weight=onorm if idx == len(cases) - 1 else None,
                    rms_eps=1e-6,
                )
                out.sum().backward()
        torch.cuda.synchronize()
        ms = (time.perf_counter() - start) / iters * 1000
        print(f"timing B={batch} T={tokens}: {ms:.2f} ms per step-equivalent "
              f"({len(cases)} routed sites, fwd+bwd)")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("needs the training GPU; run in Colab, not locally")
    print(f"device: {torch.cuda.get_device_name()}; torch {torch.__version__}")
    run_parity()
    run_timing()
    print("ALL PARITY CHECKS PASSED")
