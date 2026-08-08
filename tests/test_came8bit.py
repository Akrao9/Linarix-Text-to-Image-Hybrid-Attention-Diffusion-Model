"""CPU regression coverage for boomer.optim.CAME8Bit.

The optimizer is only worth running if three things hold: the quantiser round-trips
without eating the signal, the state really is smaller than AdamW's, and a checkpoint
resumes to the identical trajectory. Everything here is CPU-only and fast.
"""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from boomer.optim import (
    CAME8Bit,
    _dequantize,
    _quantize,
    adamw_state_bytes,
    came_state_bytes,
)


# --- quantiser -----------------------------------------------------------------


@pytest.mark.parametrize("numel", [2048, 4096, 5000, 1, 17])
def test_quantize_roundtrip_shapes(numel):
    """Round-trip preserves shape for both exact and ragged block counts."""
    t = torch.randn(numel)
    codes, scale, bmin = _quantize(t, 2048)
    out = _dequantize(codes, scale, bmin, t.shape, 2048)
    assert out.shape == t.shape


def test_quantize_constant_block_is_exact_not_nan():
    """A zero-range block must not divide by zero.

    This is the state at step 0: exp_avg is all zeros, so every block has
    max == min. Dividing by the raw scale gives 0/0 = NaN, which would poison the
    first momentum update. The clamped divisor sends every code to 0 and
    dequantisation returns bmin exactly.
    """
    for value in (0.0, -3.5, 7.25):
        t = torch.full((4096,), value)
        codes, scale, bmin = _quantize(t, 2048)
        out = _dequantize(codes, scale, bmin, t.shape, 2048)
        assert torch.isfinite(out).all()
        assert torch.equal(out, t)


def test_quantize_error_within_block_resolution():
    """Error is bounded by half a quantisation step of the block's own range."""
    torch.manual_seed(0)
    t = torch.randn(8192)
    codes, scale, bmin = _quantize(t, 2048)
    out = _dequantize(codes, scale, bmin, t.shape, 2048)
    per_block_range = t.view(-1, 2048).amax(dim=1) - t.view(-1, 2048).amin(dim=1)
    tol = (per_block_range / 255.0).max().item()
    assert (out - t).abs().max().item() <= tol


def test_quantize_is_vectorized_not_chunk_list():
    """State must be flat tensors, not a Python list of per-chunk dicts.

    The reference implementation returns one dict per 2048-element chunk, which at
    v8's scale is hundreds of thousands of dicts and millions of tiny kernel launches
    per step. It also does not survive `state_dict()`.
    """
    codes, scale, bmin = _quantize(torch.randn(8192), 2048)
    assert isinstance(codes, torch.Tensor) and codes.dtype == torch.uint8
    assert isinstance(scale, torch.Tensor) and scale.numel() == 4
    assert isinstance(bmin, torch.Tensor) and bmin.numel() == 4


# --- layout decisions ----------------------------------------------------------


def test_factored_layout_selection():
    assert CAME8Bit._factored(torch.Size([256, 128])) is True
    assert CAME8Bit._factored(torch.Size([64, 32, 1, 1])) is True
    # spatial convs and 1D params keep a dense second moment
    assert CAME8Bit._factored(torch.Size([64, 32, 3, 3])) is False
    assert CAME8Bit._factored(torch.Size([512])) is False


def test_large_2d_gets_quantized_first_moment_small_does_not():
    big = nn.Linear(256, 256, bias=False)     # 65536 elements >= min_8bit_size
    small = nn.Linear(16, 16, bias=False)     # 256 elements
    opt = CAME8Bit(list(big.parameters()) + list(small.parameters()), lr=1e-3)
    for m in (big, small):
        m.weight.grad = torch.randn_like(m.weight)
    opt.step()

    assert "exp_avg_q" in opt.state[big.weight]
    assert "exp_avg" not in opt.state[big.weight]
    assert opt.state[big.weight]["exp_avg_q"].dtype == torch.uint8
    # factored: no dense second moment anywhere
    assert "exp_avg_sq" not in opt.state[big.weight]
    assert opt.state[big.weight]["exp_avg_sq_row"].numel() == 256

    assert "exp_avg" in opt.state[small.weight]
    assert opt.state[small.weight]["exp_avg"].dtype == torch.float32


def test_1d_param_keeps_dense_second_moment():
    norm = nn.LayerNorm(64)
    opt = CAME8Bit(norm.parameters(), lr=1e-3)
    norm.weight.grad = torch.randn_like(norm.weight)
    norm.bias.grad = torch.randn_like(norm.bias)
    opt.step()
    assert opt.state[norm.weight]["exp_avg_sq"].shape == norm.weight.shape
    assert "exp_avg_sq_row" not in opt.state[norm.weight]


# --- the actual claim: less memory than AdamW ----------------------------------


def test_state_is_much_smaller_than_adamw():
    """The whole point. Factoring dominates; 8-bit adds to it."""
    model = nn.Sequential(nn.Linear(1024, 1024), nn.Linear(1024, 1024))
    came = came_state_bytes(model)
    adamw = adamw_state_bytes(model)
    assert came < adamw / 5, f"expected >5x saving, got {adamw / came:.1f}x"


def test_state_accounting_matches_reality():
    """The predicted byte count must match what stepping actually allocates."""
    model = nn.Sequential(nn.Linear(512, 512, bias=False), nn.LayerNorm(512))
    opt = CAME8Bit(model.parameters(), lr=1e-3)
    for p in model.parameters():
        p.grad = torch.randn_like(p)
    opt.step()

    actual = 0
    for state in opt.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                actual += value.numel() * value.element_size()
    assert actual == came_state_bytes(model)


# --- optimization behaviour ----------------------------------------------------


def test_converges_on_a_quadratic():
    torch.manual_seed(0)
    layer = nn.Linear(256, 256, bias=False)
    target = torch.randn(32, 256)
    inputs = torch.randn(32, 256)
    opt = CAME8Bit(layer.parameters(), lr=1e-2)

    losses = []
    for _ in range(60):
        opt.zero_grad()
        loss = ((layer(inputs) - target) ** 2).mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] * 0.5, f"no progress: {losses[0]:.4f} -> {losses[-1]:.4f}"
    assert all(torch.isfinite(p).all() for p in layer.parameters())


def test_step_is_finite_with_zero_and_tiny_gradients():
    """Degenerate gradients must not produce NaN through the rsqrt/factor path."""
    layer = nn.Linear(256, 256, bias=False)
    opt = CAME8Bit(layer.parameters(), lr=1e-3)
    for grad in (torch.zeros_like(layer.weight), torch.full_like(layer.weight, 1e-30)):
        layer.weight.grad = grad.clone()
        opt.step()
        assert torch.isfinite(layer.weight).all()


def test_weight_decay_shrinks_a_zero_gradient_parameter():
    layer = nn.Linear(256, 256, bias=False)
    with torch.no_grad():
        layer.weight.fill_(1.0)
    opt = CAME8Bit(layer.parameters(), lr=0.1, weight_decay=0.1)
    layer.weight.grad = torch.zeros_like(layer.weight)
    opt.step()
    assert layer.weight.max().item() < 1.0


def test_rejects_adamw_style_misuse():
    layer = nn.Linear(8, 8)
    with pytest.raises(ValueError):
        CAME8Bit(layer.parameters(), lr=0.0)
    with pytest.raises(ValueError):
        CAME8Bit(layer.parameters(), lr=1e-3, betas=(0.9, 1.5, 0.99))


def test_sparse_gradients_rejected():
    p = nn.Parameter(torch.randn(16, 16))
    opt = CAME8Bit([p], lr=1e-3)
    p.grad = torch.sparse_coo_tensor(torch.tensor([[0], [0]]), torch.tensor([1.0]), (16, 16))
    with pytest.raises(RuntimeError, match="sparse"):
        opt.step()


def test_bf16_gradients_are_promoted():
    """Statistics stay fp32 even if a gradient arrives in bf16.

    v8 trains bf16-mixed with fp32 parameters, so grads are normally fp32 already; this
    guards the branch that would otherwise accumulate second moments in bf16.
    """
    layer = nn.Linear(256, 256, bias=False).to(torch.bfloat16)
    opt = CAME8Bit(layer.parameters(), lr=1e-3)
    layer.weight.grad = torch.randn_like(layer.weight)
    assert layer.weight.grad.dtype == torch.bfloat16
    opt.step()
    assert torch.isfinite(layer.weight).all()
    assert opt.state[layer.weight]["exp_avg_sq_row"].dtype == torch.float32


# --- checkpoint resume ---------------------------------------------------------


def test_state_dict_resume_is_bit_identical():
    """A resumed optimizer must continue on exactly the same trajectory.

    Quantised state is what makes this non-trivial: uint8 codes plus per-block scales
    have to survive `state_dict()` / `load_state_dict()` without being cast to the
    parameter dtype or flattened into the wrong shape.
    """
    torch.manual_seed(0)
    inputs = torch.randn(16, 256)
    target = torch.randn(16, 256)

    def run(resume_at: int | None):
        torch.manual_seed(1)
        layer = nn.Linear(256, 256, bias=False)
        opt = CAME8Bit(layer.parameters(), lr=1e-2)
        for step in range(10):
            if resume_at is not None and step == resume_at:
                snapshot = copy.deepcopy(opt.state_dict())
                weights = layer.weight.detach().clone()
                layer = nn.Linear(256, 256, bias=False)
                with torch.no_grad():
                    layer.weight.copy_(weights)
                opt = CAME8Bit(layer.parameters(), lr=1e-2)
                opt.load_state_dict(snapshot)
            opt.zero_grad()
            ((layer(inputs) - target) ** 2).mean().backward()
            opt.step()
        return layer.weight.detach().clone()

    assert torch.equal(run(None), run(5))


def test_resume_preserves_the_memory_saving():
    """Resuming must not silently inflate the state back to fp32.

    Numerics alone would not catch this: fp32-cast codes dequantise to the same
    values, so training would continue correctly while the feature quietly stopped
    saving anything.
    """
    layer = nn.Linear(256, 256, bias=False)
    opt = CAME8Bit(layer.parameters(), lr=1e-3)
    layer.weight.grad = torch.randn_like(layer.weight)
    opt.step()

    def state_bytes(o):
        return sum(
            v.numel() * v.element_size()
            for s in o.state.values()
            for v in s.values()
            if isinstance(v, torch.Tensor)
        )

    before = state_bytes(opt)
    restored = CAME8Bit(nn.Linear(256, 256, bias=False).parameters(), lr=1e-3)
    restored.load_state_dict(copy.deepcopy(opt.state_dict()))
    assert state_bytes(restored) == before


def test_state_dict_preserves_uint8_codes():
    layer = nn.Linear(256, 256, bias=False)
    opt = CAME8Bit(layer.parameters(), lr=1e-3)
    layer.weight.grad = torch.randn_like(layer.weight)
    opt.step()

    restored = CAME8Bit(nn.Linear(256, 256, bias=False).parameters(), lr=1e-3)
    restored.load_state_dict(copy.deepcopy(opt.state_dict()))
    state = next(iter(restored.state.values()))
    assert state["exp_avg_q"].dtype == torch.uint8, (
        "load_state_dict cast the quantised codes -- resume would corrupt momentum"
    )


# --- trainer wiring ------------------------------------------------------------


def _hparams(**overrides):
    from argparse import Namespace

    base = dict(
        optimizer="came8bit", lr=1e-4, came_lr=2e-5, weight_decay=0.0,
        beta1=0.9, beta2=0.95, fused_adamw=True,
        came_block_size=2048, came_min_8bit_size=16384,
        warmup_steps=0, lr_scheduler="constant", min_lr_ratio=0.1, max_steps=100,
    )
    base.update(overrides)
    return Namespace(**base)


class _TinyLit:
    """Enough of BoomerLit for configure_optimizers, without building a DiT."""

    def __init__(self, hparams):
        import torch.nn as _nn

        self.hparams = hparams
        self.model = _nn.Sequential(_nn.Linear(64, 64), _nn.Linear(64, 64))


def test_configure_optimizers_builds_came8bit():
    import train

    lit = _TinyLit(_hparams())
    out = train.BoomerLit.configure_optimizers(lit)
    optim = out["optimizer"]
    assert isinstance(optim, CAME8Bit)
    assert optim.param_groups[0]["lr"] == 2e-5, "must use --came-lr, not --lr"


def test_configure_optimizers_defaults_to_adamw():
    import train

    lit = _TinyLit(_hparams(optimizer="adamw"))
    optim = train.BoomerLit.configure_optimizers(lit)["optimizer"]
    assert isinstance(optim, torch.optim.AdamW)
    assert optim.param_groups[0]["lr"] == 1e-4


def test_came8bit_without_came_lr_is_rejected():
    """Inheriting the AdamW LR would be ~5x too large and would look like it trained."""
    import train

    lit = _TinyLit(_hparams(came_lr=None))
    with pytest.raises(ValueError, match="came-lr"):
        train.BoomerLit.configure_optimizers(lit)


def test_on_train_start_does_not_push_adamw_lr_into_came_groups():
    """The --lr resume override must target the ACTIVE optimizer's rate.

    on_train_start rewrites param_groups["initial_lr"] and the scheduler base_lrs from
    the CLI so a changed LR wins over restored state. Reading hparams.lr unconditionally
    would reset CAME to the AdamW rate on every resume.
    """
    import train
    from types import SimpleNamespace

    lit = _TinyLit(_hparams())
    optim = train.BoomerLit.configure_optimizers(lit)["optimizer"]
    for group in optim.param_groups:
        group["initial_lr"] = 999.0
    sched = SimpleNamespace(base_lrs=[999.0])
    lit.trainer = SimpleNamespace(
        optimizers=[optim],
        lr_scheduler_configs=[SimpleNamespace(scheduler=sched)],
    )
    train.BoomerLit.on_train_start(lit)

    assert optim.param_groups[0]["initial_lr"] == 2e-5
    assert sched.base_lrs == [2e-5]
