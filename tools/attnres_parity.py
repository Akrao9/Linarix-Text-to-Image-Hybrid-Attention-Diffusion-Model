"""BF16 parity + timing for FLA's fused_attnres at Boomer's v8 shapes.

FLA's own parity suite (tests/ops/test_attnres.py) covers fp16/fp32 up to L=29,
T=8000, D=7186 — but not bf16, which is what Boomer trains in. This checks the
fused Triton kernel against fla.ops.attnres.naive_attnres (the reference the suite
itself trusts) forward AND backward at D=1152, L=1..9, and the actual v8 batch
shapes: (B=56, T=256) for the 512 phase and (B=14, T=1024) for 1024.

Two dtype regimes are covered:

  * "uniform bf16" — every input bf16. Isolates the kernel's own numerics.
  * "mixed" — WHAT PRODUCTION ACTUALLY RUNS. Under Lightning ``bf16-mixed``,
    parameters and buffers stay fp32 while autocast makes activations bf16, so
    AttnRes receives bf16 residual sources with an fp32 pseudo-query, fp32 key-norm
    buffer, and fp32 output-norm weight. FLA's ``custom_fwd`` carries no
    ``cast_inputs``, so those mixed dtypes reach the Triton kernel untouched — and
    the grads come back matching each input's dtype (dq/dow fp32, dv bf16), which
    is what the fp32 optimizer states require.

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


def make_case(n_sources: int, batch: int, tokens: int, param_dtype=DTYPE):
    """param_dtype=torch.float32 reproduces bf16-mixed: bf16 activations (the residual
    sources) against fp32 parameters/buffers."""
    residuals = [
        torch.randn(batch, tokens, D, device=DEVICE, dtype=DTYPE, requires_grad=True)
        for _ in range(n_sources)
    ]
    query = torch.randn(D, device=DEVICE, dtype=param_dtype) * 0.02
    query = query.detach().requires_grad_(True)
    # Production semantics (BlockAttnRes): the key RMSNorm is the paper's
    # parameter-free norm — a fixed ones buffer, never trained.
    rms_w = torch.ones(D, device=DEVICE, dtype=param_dtype)
    return query, residuals, rms_w


def clone_case(query, residuals, rms_w):
    q = query.detach().clone().requires_grad_(True)
    rs = [r.detach().clone().requires_grad_(True) for r in residuals]
    return q, rs, rms_w.clone()


def check_single_source_is_bypassed() -> None:
    """L=1 is excluded from the sweep below; verify the bypass that makes that valid.

    With one source the depth softmax is identically 1.0, so the output carries no
    dependence on the query and the reference dq is analytically ZERO. FLA's
    assert_close scores relative error (err_rms / (ref_rms + 1e-8)), so any float noise
    in the kernel's dq reduction — ~6e-5 absolute, negligible against real gradients —
    divides by ~0 and reports a ratio in the thousands. That says nothing about the
    kernel, and Boomer never asks it: BlockAttnRes returns the lone source directly.
    """
    from boomer.models.boomer_fla_dit import BlockAttnRes

    mod = BlockAttnRes(D, slot_counts=[1]).to(DEVICE)
    src = torch.randn(2, 8, D, device=DEVICE, dtype=DTYPE)
    out = mod(0, [src])
    assert out is src, "L=1 bypass is gone — restore it or re-include L=1 in the sweep"
    print("L=1 bypass verified (single-source sites never reach the kernel)")


def run_parity(param_dtype=DTYPE, label: str = "uniform bf16") -> None:
    torch.manual_seed(0)
    for batch, tokens in SHAPES:
        # with_onorm covers the final z site, which fuses its learned output RMSNorm
        # into the kernel via output_rms_weight; per-block sites pass None.
        for with_onorm in (False, True):
            # from 2: see check_single_source_is_bypassed
            for n in range(2, MAX_SOURCES + 1):
                q0, r0, w0 = make_case(n, batch, tokens, param_dtype=param_dtype)
                q1, r1, w1 = clone_case(q0, r0, w0)
                ow0 = ow1 = None
                if with_onorm:
                    ow0 = torch.ones(D, device=DEVICE, dtype=param_dtype, requires_grad=True)
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
                tag = f"L={n},B={batch},T={tokens},onorm={with_onorm},{label}"
                # The mixed case must not silently promote the output: downstream
                # blocks and the residual bank expect activation dtype back.
                assert out.dtype == DTYPE, f"{tag}: fused returned {out.dtype}"
                assert ref.dtype == DTYPE, f"{tag}: naive returned {ref.dtype}"

                grad = torch.randn_like(ref)
                ref.backward(grad)
                out.backward(grad)

                assert_close(f"o[{tag}]", ref, out, RATIO)
                assert_close(f"dq[{tag}]", q0.grad, q1.grad, RATIO)
                # Grads must come back in each input's own dtype — fp32 params under
                # bf16-mixed need fp32 grads for the fp32 optimizer states.
                assert q1.grad.dtype == param_dtype, f"{tag}: dq is {q1.grad.dtype}"
                if with_onorm:
                    assert_close(f"dow[{tag}]", ow0.grad, ow1.grad, RATIO)
                    assert ow1.grad.dtype == param_dtype, f"{tag}: dow is {ow1.grad.dtype}"
                for i, (a, b) in enumerate(zip(r0, r1)):
                    assert_close(f"dv{i}[{tag}]", a.grad, b.grad, RATIO)
                    assert b.grad.dtype == DTYPE, f"{tag}: dv{i} is {b.grad.dtype}"
        print(f"parity OK [{label}]: B={batch} T={tokens} D={D}, L=2..{MAX_SOURCES} "
              f"fwd+bwd, with and without fused output RMSNorm")


def run_module_autocast_check() -> None:
    """End-to-end through BlockAttnRes itself under the real bf16 autocast region,
    with fp32 params/buffers exactly as Lightning holds them."""
    from boomer.models.boomer_fla_dit import BlockAttnRes

    batch, tokens = SHAPES[0]
    counts = [1, 5, MAX_SOURCES]
    mod = BlockAttnRes(D, slot_counts=counts, output_norm_sites=(len(counts) - 1,)).to(DEVICE)
    assert mod.key_norm_weight.dtype == torch.float32
    for site, n in enumerate(counts):
        sources = [
            torch.randn(batch, tokens, D, device=DEVICE, dtype=DTYPE, requires_grad=True)
            for _ in range(n)
        ]
        with torch.autocast(device_type="cuda", dtype=DTYPE):
            out = mod(site, sources)
        assert out.dtype == DTYPE, f"site {site}: autocast output is {out.dtype}"
        out.float().square().mean().backward()
        for i, s in enumerate(sources):
            assert s.grad is not None and s.grad.dtype == DTYPE, (site, i)
        if n > 1:
            g = mod.queries[site].grad
            assert g is not None and g.dtype == torch.float32, (site, g)
    onorm_grad = mod.out_norm_weights[str(len(counts) - 1)].grad
    assert onorm_grad is not None and onorm_grad.dtype == torch.float32
    assert mod.key_norm_weight.grad is None, "parameter-free key norm must never get a grad"
    print("BlockAttnRes autocast check OK: bf16 out, fp32 param grads, key norm untouched")


def run_timing(iters: int = 50) -> None:
    """One full v8 step's worth of routing calls (site source counts for 22 encoder
    blocks + final z site under group ends 2,5,8,11,14,17,19,21)."""
    site_counts = [1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 9, 9]
    for batch, tokens in SHAPES:
        # fp32 params against bf16 sources: the production bf16-mixed regime, so the
        # timing reflects whatever the mixed-dtype loads actually cost.
        cases = [
            make_case(n, batch, tokens, param_dtype=torch.float32)
            for n in site_counts if n > 1
        ]
        # The last routed call is the final z site, which fuses its output RMSNorm
        # (the only LEARNED norm in the operator).
        onorm = torch.ones(D, device=DEVICE, dtype=torch.float32, requires_grad=True)
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
    check_single_source_is_bypassed()
    run_parity()  # uniform bf16: the kernel's own numerics
    # bf16-mixed, i.e. what training actually runs: fp32 params/buffers, bf16 sources.
    run_parity(param_dtype=torch.float32, label="mixed fp32-param/bf16-act")
    run_module_autocast_check()
    run_timing()
    print("ALL PARITY CHECKS PASSED")
