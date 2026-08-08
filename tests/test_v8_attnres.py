"""v8 Block AttnRes: parameter-free key norm, CUDA fail-fast, final-z output norm,
exact block updates, refresh placement, and telemetry cadence under accumulation."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from boomer.models.boomer_fla_dit import (  # noqa: E402
    BlockAttnRes,
    BoomerFLADiT,
    BoomerFLADiTConfig,
    BoomerFLAJointRefreshBlock,
)


def _config(**overrides) -> BoomerFLADiTConfig:
    """Small CPU stand-in for the v8 layout: 22-block encoder shape compressed to 10
    (6 encoder + 4 decoder) with the same feature switches."""
    base = dict(
        latent_channels=4,
        latent_size=8,
        text_dim=16,
        text_seq_len=6,
        hidden_dim=32,
        depth=10,
        num_heads=4,
        mlp_ratio=1.0,
        mixer_type="fallback",
        dual_stream_depth=1,
        ddt_decoder_depth=4,
        joint_refresh_attention=True,
        refresh_layers=(2, 5),
        refresh_no_mixer=True,
        block_attnres=True,
        attnres_group_ends=(1, 3, 5),
        shared_adaln=True,
        sandwich_rmsnorm=True,
        use_abs_pos_embed=False,
        image_attention_every=2,  # must be neutralized by refresh_layers
    )
    base.update(overrides)
    return BoomerFLADiTConfig(**base)


def _break_zero_init(model: BoomerFLADiT, std: float = 0.05) -> None:
    """Perturb every all-zero parameter so forward/backward carry real signal.

    adaLN-Zero leaves the head, the shared modulation projections, the DDT cond_mod
    projections and the refresh out-projections at zero, so a freshly built model
    outputs exactly 0.0 and every gradient norm is 0 — any comparison against it is
    vacuous. This also un-zeros the AttnRes pseudo-queries, which moves routing off
    the uniform average and onto the real softmax path.
    """
    with torch.no_grad():
        for param in model.parameters():
            if torch.count_nonzero(param) == 0:
                param.normal_(std=std)


def _batch(model: BoomerFLADiT, batch: int = 2):
    cfg = model.config
    return (
        torch.randn(batch, cfg.latent_channels, cfg.latent_size, cfg.latent_size),
        torch.rand(batch),
        torch.randn(batch, cfg.text_seq_len, cfg.text_dim),
        torch.ones(batch, cfg.text_seq_len, dtype=torch.long),
    )


# --- key normalization is the paper's parameter-free form -------------------------


def test_key_norm_is_a_fixed_buffer_not_a_parameter() -> None:
    attn = BlockAttnRes(hidden_dim=8, slot_counts=[1, 2, 3])
    assert not isinstance(attn.key_norm_weight, nn.Parameter)
    assert not attn.key_norm_weight.requires_grad
    assert torch.equal(attn.key_norm_weight, torch.ones(8))
    # non-persistent: never lands in a checkpoint, so FLA/paper convention changes
    # can't silently alter a resumed run
    assert not any("key_norm" in k for k in attn.state_dict())
    assert {n for n, _ in attn.named_parameters()} == {"queries.0", "queries.1", "queries.2"}


def test_key_norm_never_receives_gradient() -> None:
    attn = BlockAttnRes(hidden_dim=8, slot_counts=[2])
    sources = [torch.randn(2, 4, 8, requires_grad=True) for _ in range(2)]
    attn(0, sources).square().mean().backward()
    assert attn.key_norm_weight.grad is None
    assert attn.queries[0].grad is not None


def test_routing_matches_flas_naive_reference() -> None:
    attn = BlockAttnRes(hidden_dim=8, slot_counts=[3])
    with torch.no_grad():
        attn.queries[0].normal_(std=0.5)
    sources = [torch.randn(2, 4, 8) for _ in range(3)]
    out = attn(0, sources)

    v = torch.stack(sources).float()
    k = F.rms_norm(v, (8,), torch.ones(8), 1e-6)
    p = torch.einsum("nbtd,d->nbt", k, attn.queries[0].float()).softmax(dim=0)
    assert torch.allclose(out, torch.einsum("nbt,nbtd->btd", p, v), atol=1e-5)


def test_zero_queries_start_at_the_papers_uniform_average() -> None:
    attn = BlockAttnRes(hidden_dim=8, slot_counts=[4])
    attn.collect_weights = True
    sources = [torch.randn(2, 4, 8) for _ in range(4)]
    out = attn(0, sources)
    assert torch.allclose(out, sum(sources) / 4, atol=1e-5)
    assert torch.allclose(attn.last_weights[0], torch.full((4,), 0.25), atol=1e-5)


# --- CUDA fails fast rather than falling back to the naive stack ------------------


def test_cuda_without_fused_kernel_raises() -> None:
    """The naive path allocates torch.stack(...).float() — >1 GiB of transients at the
    nine-source production shape. On CUDA that must be an error, not a silent fallback."""
    attn = BlockAttnRes(hidden_dim=8, slot_counts=[2])
    attn._fused_checked, attn._fused = True, None  # simulate an FLA without fused_attnres

    class FakeCudaSource:
        is_cuda = True
        dtype = torch.bfloat16   # the dtype guard runs first and reads this

    with pytest.raises(RuntimeError, match="fused_attnres"):
        attn(0, [FakeCudaSource(), FakeCudaSource()])


def test_cpu_still_uses_the_naive_path() -> None:
    attn = BlockAttnRes(hidden_dim=8, slot_counts=[2])
    attn._fused_checked, attn._fused = True, None
    out = attn(0, [torch.randn(2, 4, 8) for _ in range(2)])
    assert out.shape == (2, 4, 8) and torch.isfinite(out).all()


# --- final z aggregation carries the learned output RMSNorm -----------------------


def test_only_the_final_z_site_has_an_output_norm() -> None:
    model = BoomerFLADiT(_config())
    encoder_depth = model.encoder_depth
    assert set(model.block_attnres.out_norm_weights.keys()) == {str(encoder_depth)}
    assert len(model.block_attnres.queries) == encoder_depth + 1


def test_z_leaves_the_encoder_at_unit_rms() -> None:
    """Without the norm the convex average shrinks z by ~1/L, so silu(z + t_emb)
    would be dominated by the timestep term."""
    model = BoomerFLADiT(_config())
    _, z = model(*_batch(model), return_encoder_state=True)
    rms = z.float().square().mean(-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3)


def test_output_norm_site_indices_are_validated() -> None:
    with pytest.raises(ValueError, match="output_norm_sites"):
        BlockAttnRes(hidden_dim=8, slot_counts=[1, 2], output_norm_sites=(5,))


# --- blocks hand back exact updates (no bf16 stream-scale subtraction) ------------


@pytest.mark.parametrize("parallel", [False, True])
def test_blocks_return_their_bare_update(parallel: bool) -> None:
    """The AttnRes bank consumes the update itself; recovering it as (out - in) in
    bf16 loses the small contributions the bank exists to preserve."""
    model = BoomerFLADiT(_config(parallel_block=parallel))
    t_emb = model.t_embedder(torch.rand(2))
    kwargs = dict(height=8, width=8)
    tok = torch.randn(2, 64, 32)
    txt = torch.randn(2, 6, 32)
    kpm = torch.zeros(2, 6, dtype=torch.bool)
    bias = ~kpm[:, None, None, :]

    dual = model.blocks[0]
    x, _, upd = dual(
        tok, txt, t_emb, kpm, bias, **kwargs, return_update=True,
        image_t_mod=model.shared_dual_image_mod(t_emb),
        text_t_mod=model.shared_dual_text_mod(t_emb),
    )
    assert torch.allclose(x, tok + upd, atol=1e-5)

    refresh = model.blocks[2]
    assert isinstance(refresh, BoomerFLAJointRefreshBlock)
    x, _, upd = refresh(
        tok, txt, t_emb, kpm, bias, **kwargs, return_update=True,
        t_mod=model.shared_mod(t_emb), text_t_mod=model.shared_dual_text_mod(t_emb),
    )
    assert torch.allclose(x, tok + upd, atol=1e-5)

    plain = model.blocks[4]
    x, upd = plain(
        tok, txt, t_emb, kpm, bias, **kwargs, return_update=True,
        t_mod=model.shared_mod(t_emb),
    )
    assert torch.allclose(x, tok + upd, atol=1e-5)
    # asking for the update must not perturb the block's own output
    assert torch.allclose(
        x, plain(tok, txt, t_emb, kpm, bias, **kwargs, t_mod=model.shared_mod(t_emb)),
        atol=1e-6,
    )


def test_gradient_checkpointing_matches_the_plain_path() -> None:
    """Numerical parity, not just graph connectivity: the AttnRes bank is rebuilt from
    recomputed block updates inside the checkpointed segments, so a mis-threaded
    return_update would show up as a value mismatch here."""
    torch.manual_seed(0)
    model = BoomerFLADiT(_config())
    _break_zero_init(model)
    ckpt = BoomerFLADiT(_config(gradient_checkpointing=True))
    ckpt.load_state_dict(model.state_dict())
    model.train()
    ckpt.train()

    batch = _batch(model)
    a = model(*batch)
    b = ckpt(*batch)
    assert a.abs().max() > 1e-3, "zero-init head would make this comparison vacuous"
    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)

    a.square().mean().backward()
    b.square().mean().backward()
    # Parameters legitimately outside the loss graph, pinned so a future change cannot
    # quietly widen the dead-weight surface:
    #   * null_text_embedding — used by apply_condition_dropout, not by forward
    #   * the LAST encoder block's text half — it is a refresh block by design and
    #     nothing downstream reads text again (v8 accepts this: z leaving through
    #     exact joint global mixing is the point)
    #   * AttnRes site 0 — single source, bypasses routing entirely
    dead_text = f"blocks.{model.encoder_depth - 1}.text"
    expected_dead = {"caption_embedder.null_text_embedding", "block_attnres.queries.0"}

    grads = dict(ckpt.named_parameters())
    compared, dead = 0, set()
    for name, param in model.named_parameters():
        other = grads[name]
        assert (param.grad is None) == (other.grad is None), name
        if param.grad is None or param.grad.abs().max() == 0:
            dead.add(name)
            if param.grad is not None:
                # even a dead branch must agree between the two paths
                assert other.grad.abs().max() == 0, name
            continue
        torch.testing.assert_close(param.grad, other.grad, rtol=1e-4, atol=1e-6, msg=name)
        compared += 1
    assert compared > 200, compared
    assert {n for n in dead if not n.startswith(dead_text)} == expected_dead, sorted(dead)

    # every routing query carries real (non-zero) gradient, final z norm included
    for q, n in zip(ckpt.block_attnres.queries, ckpt.block_attnres.slot_counts):
        if n > 1:
            assert q.grad is not None and q.grad.abs().max() > 0
    assert ckpt.block_attnres.out_norm_weights[str(ckpt.encoder_depth)].grad.abs().max() > 0


def test_slot_counts_follow_the_delta_group_schedule() -> None:
    """[embedding] + completed group sums + the current partial, one site per encoder
    block plus the final z site."""
    model = BoomerFLADiT(_config())
    # ends (1, 3, 5) over a 6-block encoder: the trailing end folds post-loop, so the
    # final z site sees the 3 completed group sums plus the embedding and the partial
    assert model.block_attnres.slot_counts == (1, 2, 2, 3, 3, 4, 4)
    assert model._attnres_fold_ends == frozenset({1, 3})


# --- explicit refresh placement owns global attention outright --------------------


def test_refresh_layers_disables_the_legacy_image_attention_cadence() -> None:
    """image_attention_every=2 would otherwise leave stray full-attention blocks at
    off-refresh depths, breaking the v8 'GDN2 everywhere else' layout."""
    model = BoomerFLADiT(_config())
    assert model._joint_refresh_indices == {2, 5}
    assert not any(getattr(b, "use_image_attention", False) for b in model.blocks)
    assert model.shared_image_attn_mod is None
    # the final encoder block IS a refresh block (impossible under the v6 demotion walk)
    assert isinstance(model.blocks[5], BoomerFLAJointRefreshBlock)


def test_v6_cadence_is_untouched_without_refresh_layers() -> None:
    model = BoomerFLADiT(_config(refresh_layers=None, block_attnres=False,
                                 attnres_group_ends=(), refresh_no_mixer=False))
    assert any(
        getattr(b, "use_image_attention", False) or isinstance(b, BoomerFLAJointRefreshBlock)
        for b in model.blocks
    )
    assert model.block_attnres is None
    assert not any("attnres" in k for k in model.state_dict())


# --- telemetry is off by default and collected once per accumulation window -------


def test_weights_are_not_collected_unless_requested() -> None:
    model = BoomerFLADiT(_config())
    model(*_batch(model))
    assert all(w is None for w in model.block_attnres.last_weights)

    model.block_attnres.collect_weights = True
    model(*_batch(model))
    counts = model.block_attnres.slot_counts
    for w, n in zip(model.block_attnres.last_weights, counts):
        assert w is not None and w.shape == (n,)
        assert torch.allclose(w, torch.full((n,), 1.0 / n), atol=1e-4)


def test_telemetry_collects_once_per_optimizer_step_under_accumulation() -> None:
    """global_step is constant across an accumulation window, so a bare
    `global_step % log_every == 0` test would collect (and log 116 scalars) N times."""
    pytest.importorskip("pytorch_lightning")
    from train import BoomerLit

    module = BoomerLit(
        _config(),
        lr=1e-4, weight_decay=0.0, beta1=0.9, beta2=0.95, fused_adamw=False,
        warmup_steps=0, max_steps=10, cond_dropout=0.0,
        t_sampler="uniform", t_sampler_final=None, t_sampler_final_step=0,
        t_bias=1.0, flow_shift=1.0, logit_mean=0.0, logit_std=1.0,
        plateau_low=0.0, plateau_high=1.0, stretch_q_low=0.0, stretch_q_high=1.0,
        timestep_weight=False, min_snr_gamma=0.0,
        latent_mean=0.0, latent_std=1.0,
        repa_weight=0.0, repa_anneal_steps=0,
        dispersive_weight=0.0, dispersive_temp=0.5, dispersive_metric="l2",
        dispersive_anneal_steps=0,
    )
    module.lr_schedulers = lambda: None
    # LightningModule.global_step reads through to the trainer; the accumulation window
    # is exactly the span over which it stays constant.
    trainer = type("T", (), {"log_every_n_steps": 10, "global_step": 0})()
    module.trainer = trainer
    logged: list[str] = []
    module.log = lambda name, *a, **k: logged.append(name)

    cfg = module.model.config
    batch = {
        "image_latent": torch.randn(2, cfg.latent_channels, cfg.latent_size, cfg.latent_size),
        "text_embedding": torch.randn(2, cfg.text_seq_len, cfg.text_dim),
        "attention_mask": torch.ones(2, cfg.text_seq_len, dtype=torch.long),
        "dino_embedding": None,
    }

    # step 0 is a logging step: 4 accumulation microbatches, one collection
    collected = 0
    for micro in range(4):
        logged.clear()
        module.training_step(batch, micro)
        collected += any(n.startswith("attnres/") for n in logged)
    assert collected == 1, f"AttnRes telemetry logged in {collected} of 4 microbatches"

    # the next logging step collects again (the guard is a sentinel, not a latch)
    trainer.global_step = 10
    logged.clear()
    module.training_step(batch, 0)
    assert any(n.startswith("attnres/") for n in logged)

    # an off-cadence step logs nothing and skips the weight reductions entirely
    trainer.global_step = 13
    logged.clear()
    module.training_step(batch, 0)
    assert not any(n.startswith("attnres/") for n in logged)
    assert not module.model.block_attnres.collect_weights


# --- CLI args must actually reach the LightningModule ------------------------------


def test_every_text_encoder_arg_is_forwarded_to_boomerlit() -> None:
    """--text-encoder-layers was parsed, used to size the DiT's text_dim, and then not
    passed to BoomerLit — so the resident encoder silently built the single-layer path
    at 2560 against a DiT expecting 7680. Nothing failed until the first batch.

    Any argparse dest in this family that main() does not forward is the same bug.
    """
    import ast

    tree = ast.parse((ROOT / "train.py").read_text())

    parsed: set[str] = set()
    forwarded: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        # p.add_argument("--text-encoder-...")
        if isinstance(fn, ast.Attribute) and fn.attr == "add_argument":
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    if arg.value.startswith("--text-encoder-"):
                        parsed.add(arg.value[2:].replace("-", "_"))
        # BoomerLit(..., text_encoder_x=args.text_encoder_x, ...)
        if isinstance(fn, ast.Name) and fn.id == "BoomerLit":
            for kw in node.keywords:
                if kw.arg and kw.arg.startswith("text_encoder_"):
                    forwarded.add(kw.arg)

    assert "text_encoder_layers" in parsed, "flag disappeared from the parser"
    missing = parsed - forwarded
    assert not missing, (
        f"parsed but never forwarded to BoomerLit: {sorted(missing)} — "
        "the resident encoder will fall back to its default for these"
    )


# --- the bank must be single-dtype under autocast ----------------------------------


def test_bank_sources_share_one_dtype_under_autocast() -> None:
    """FLA's pointer table needs one dtype across sources. Under autocast the token
    embedding is bf16 but raw block updates are fp32 — every gate is an fp32
    scale_shift_table parameter added to a bf16 modulation, which promotes the product.
    Unbanked, that reached Triton as 'invalid operands of type pointer<fp32> and
    pointer<bf16>' after a full data-pipeline startup.
    """
    model = BoomerFLADiT(_config())
    seen: list[tuple[int, set]] = []
    real = BlockAttnRes.forward

    def spy(self, site, sources):
        seen.append((site, {s.dtype for s in sources}))
        return real(self, site, sources)

    BlockAttnRes.forward = spy
    try:
        with torch.autocast("cpu", dtype=torch.bfloat16):
            model(*_batch(model))
    finally:
        BlockAttnRes.forward = real

    assert len(seen) == model.encoder_depth + 1, len(seen)
    mixed = [(site, sorted(str(d) for d in dt)) for site, dt in seen if len(dt) > 1]
    assert not mixed, f"mixed-dtype bank at sites: {mixed}"
    assert all(dt == {torch.bfloat16} for _, dt in seen), (
        f"bank must hold the activation dtype: {[(s, sorted(map(str, d))) for s, d in seen]}"
    )


def test_mixed_dtype_sources_raise_a_named_error() -> None:
    attn = BlockAttnRes(hidden_dim=8, slot_counts=[2])
    sources = [torch.randn(2, 4, 8), torch.randn(2, 4, 8, dtype=torch.bfloat16)]
    with pytest.raises(TypeError, match="must share one dtype"):
        attn(0, sources)


# --- cross_attention_every=0: joint attention as the only text pathway -------------


def test_cross_attn_every_zero_removes_all_cross_attention() -> None:
    """Text then reaches the image stream only through joint attention (dual-stream
    prefix + refresh blocks). Unlike MMDiT, plain blocks see no text at all."""
    model = BoomerFLADiT(_config(cross_attention_every=0))
    for i, blk in enumerate(model.blocks):
        assert not getattr(blk, "use_cross_attention", False), f"block {i} kept cross-attn"
        assert getattr(blk, "cross_attn", None) is None, f"block {i} kept cross_attn weights"
    assert not any("cross_attn" in k for k in model.state_dict())
    out = model(*_batch(model))
    assert out.shape[1:] == (4, 8, 8) and torch.isfinite(out).all()


def test_cross_attn_every_zero_still_conditions_on_text() -> None:
    """The dual-stream prefix and refresh blocks must keep text in the graph, or the run
    would silently train an unconditional model."""
    model = BoomerFLADiT(_config(cross_attention_every=0))
    _break_zero_init(model)
    x, t, text, mask = _batch(model)
    text = text.clone().requires_grad_(True)
    model(x, t, text, mask).square().mean().backward()
    assert text.grad is not None and text.grad.abs().max() > 0, "text is disconnected"
    assert model.caption_embedder.y_proj[0].weight.grad.abs().max() > 0

    # and the output actually moves when the caption changes
    with torch.no_grad():
        a = model(x, t, text, mask)
        b = model(x, t, torch.randn_like(text), mask)
    assert (a - b).abs().max() > 1e-4, "output is invariant to the caption"


def test_cross_attn_every_zero_rejected_without_a_joint_pathway() -> None:
    with pytest.raises(ValueError, match="only text pathway"):
        BoomerFLADiT(_config(cross_attention_every=0, dual_stream_depth=0,
                             joint_refresh_attention=False, refresh_layers=None,
                             block_attnres=False, attnres_group_ends=()))


def test_negative_cross_attn_every_still_rejected() -> None:
    with pytest.raises(ValueError, match="must be >= 0"):
        BoomerFLADiT(_config(cross_attention_every=-1))
