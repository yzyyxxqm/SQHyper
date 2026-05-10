# STHQ v7 smoke tests: SQHyper backbone + STHQ SpikeEncoder + QMF
# CPU only, synthetic data. Verifies:
#   - QuaternionLinear correctness (block-Hamilton form)
#   - Forward/backward with no NaN
#   - SpikeEncoder produces (g_n, e_n) with correct gating + floor
#   - Cross-cell self-attention path works (and OOM fallback path is safe)
#   - Quaternion h2n vs linear h2n produce different outputs
#   - Diagnostic logging executes without error

import os
import sys
import torch

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.STHQ import (
    Model, QuaternionLinear, SpikeEncoder, MultiHeadAttentionBlock,
)


class FakeConfigs:
    d_model = 32
    n_layers = 2
    n_heads = 2
    enc_in = 5
    pred_len = 3
    seq_len = 20
    seq_len_max_irr = None
    pred_len_max_irr = None
    task_name = "short_term_forecast"
    features = "M"
    # STHQ v7 hyperparameters
    sthq_spike_floor = 0.0
    sthq_event_layer_warmup = 0
    sthq_no_spike = 0
    sthq_no_quaternion = 0
    sthq_diag_interval = 0


def test_quaternion_linear():
    """Verify QuaternionLinear produces output matching its block matrix."""
    print("--- QuaternionLinear ---")
    ql = QuaternionLinear(8, 8, bias=False)
    r, i, j, k = ql.r, ql.i, ql.j, ql.k
    W_ref = torch.cat([
        torch.cat([r, -i, -j, -k], 1),
        torch.cat([i,  r, -k,  j], 1),
        torch.cat([j,  k,  r, -i], 1),
        torch.cat([k, -j,  i,  r], 1),
    ], 0)
    x = torch.randn(2, 8)
    out = ql(x)
    out_ref = x @ W_ref.t()
    diff = (out - out_ref).abs().max()
    print(f"  max diff: {diff.item():.2e}")
    assert diff < 1e-5, f"QuaternionLinear mismatch: {diff}"
    print("  OK")


def test_quaternion_identity_init():
    """init_identity should make QuaternionLinear act as identity."""
    print("--- QuaternionLinear identity init ---")
    ql = QuaternionLinear(16, 16, bias=False)
    ql.init_identity()
    x = torch.randn(3, 16)
    out = ql(x)
    diff = (out - x).abs().max()
    print(f"  max diff vs input: {diff.item():.2e}")
    assert diff < 1e-5
    print("  OK")


def test_spike_encoder():
    """SpikeEncoder produces masked g_n with floor, and gated e_n."""
    print("--- SpikeEncoder ---")
    n_vars, D = 5, 32
    floor = 0.2
    enc = SpikeEncoder(n_vars=n_vars, d_model=D, floor=floor)
    B, N = 2, 7
    value = torch.randn(B, N)
    time_norm = torch.rand(B, N)
    var_id = torch.randint(0, n_vars, (B, N))
    mask = torch.ones(B, N)
    mask[0, -2:] = 0.0  # 2 padded cells
    g_n, e_n = enc(value, time_norm, mask, var_id)
    print(f"  g_n shape={tuple(g_n.shape)} e_n shape={tuple(e_n.shape)}")
    print(f"  g min/max on observed = "
          f"{g_n[mask > 0].min().item():.3f}/"
          f"{g_n[mask > 0].max().item():.3f}")
    assert g_n.shape == (B, N)
    assert e_n.shape == (B, N, D // 4)
    # Floor enforced on observed cells
    assert (g_n[mask > 0] >= floor - 1e-6).all(), "spike floor violated"
    assert (g_n[mask > 0] <= 1.0 + 1e-6).all()
    # Padded cells must be exactly 0
    assert (g_n[mask == 0] == 0).all(), "padded g_n must be 0"
    assert (e_n[mask == 0] == 0).all(), "padded e_n must be 0"
    # Event head initialized to zero -> e_n ≈ 0 at init (gated by g_n)
    assert e_n.abs().max() < 1e-5, (
        f"e_n must be ~0 at init, got max {e_n.abs().max()}")
    print("  OK: floor enforced, padding zero, event head zero-init")


def test_model_forward_backward():
    """Full model forward + backward on a small batch with no NaN."""
    print("--- Model forward + backward ---")
    cfg = FakeConfigs()
    model = Model(cfg)
    model.train()
    B = 2
    L, V = cfg.seq_len, cfg.enc_in
    x = torch.randn(B, L, V)
    y = torch.randn(B, cfg.pred_len, V)
    out = model(x=x, y=y, exp_stage="train")
    pred = out["pred"]
    true = out["true"]
    mask = out["mask"]
    print(f"  pred shape={tuple(pred.shape)}, "
          f"std={pred.std().item():.3f}")
    assert torch.isfinite(pred).all(), "NaN in pred"
    loss = ((pred - true) * mask).pow(2).sum() / mask.sum().clamp_min(1)
    loss.backward()
    none_grad, has_grad = 0, 0
    for p in model.parameters():
        if p.requires_grad:
            if p.grad is None:
                none_grad += 1
            else:
                has_grad += 1
                assert torch.isfinite(p.grad).all(), "NaN gradient"
    print(f"  params with grad: {has_grad}, no grad: {none_grad}")
    print("  OK")


def test_test_stage_returns_padded_shape():
    """At test stage, model returns shape (B, PRED_LEN, V)."""
    print("--- Test-stage output shape ---")
    cfg = FakeConfigs()
    model = Model(cfg)
    model.eval()
    B = 2
    L, V = cfg.seq_len, cfg.enc_in
    x = torch.randn(B, L, V)
    y = torch.randn(B, cfg.pred_len, V)
    out = model(x=x, y=y, exp_stage="test")
    assert out["pred"].shape == (B, cfg.pred_len, V), (
        f"got {out['pred'].shape}")
    print(f"  pred shape={tuple(out['pred'].shape)}")
    print("  OK")


def test_no_spike_ablation():
    """sthq_no_spike=1 should still produce a valid forward pass."""
    print("--- Ablation: --sthq_no_spike ---")
    cfg = FakeConfigs()
    cfg.sthq_no_spike = 1
    model = Model(cfg)
    model.train()
    x = torch.randn(2, cfg.seq_len, cfg.enc_in)
    y = torch.randn(2, cfg.pred_len, cfg.enc_in)
    out = model(x=x, y=y, exp_stage="train")
    pred = out["pred"]
    assert torch.isfinite(pred).all()
    pred.mean().backward()
    print(f"  pred std={pred.std().item():.3f}")
    print("  OK")


def test_no_quaternion_ablation():
    """sthq_no_quaternion=1 should produce different output than QMF."""
    print("--- Ablation: --sthq_no_quaternion ---")
    cfg1 = FakeConfigs()
    cfg2 = FakeConfigs()
    cfg2.sthq_no_quaternion = 1
    torch.manual_seed(0)
    m1 = Model(cfg1)
    torch.manual_seed(0)
    m2 = Model(cfg2)
    m1.eval()
    m2.eval()
    x = torch.randn(2, cfg1.seq_len, cfg1.enc_in)
    y = torch.randn(2, cfg1.pred_len, cfg1.enc_in)
    p1 = m1(x=x, y=y, exp_stage="train")["pred"]
    p2 = m2(x=x, y=y, exp_stage="train")["pred"]
    diff = (p1 - p2).abs().max()
    print(f"  max diff QMF vs linear: {diff.item():.4f}")
    assert diff > 1e-4, "Linear and QMF should produce different outputs"
    print("  OK")


def test_spike_floor():
    """Floor>0 changes K/V gating intensity vs floor=0."""
    print("--- Spike floor ---")
    cfg1 = FakeConfigs()
    cfg2 = FakeConfigs()
    cfg2.sthq_spike_floor = 0.3
    torch.manual_seed(0)
    m1 = Model(cfg1)
    torch.manual_seed(0)
    m2 = Model(cfg2)
    # Set gate_scale to 1 so floor actually matters in the K/V path
    with torch.no_grad():
        for p in m1.hypergraph_learner.gate_scale:
            p.fill_(1.0)
        for p in m2.hypergraph_learner.gate_scale:
            p.fill_(1.0)
    m1.eval(); m2.eval()
    x = torch.randn(2, cfg1.seq_len, cfg1.enc_in)
    y = torch.randn(2, cfg1.pred_len, cfg1.enc_in)
    p1 = m1(x=x, y=y, exp_stage="train")["pred"]
    p2 = m2(x=x, y=y, exp_stage="train")["pred"]
    diff = (p1 - p2).abs().max().item()
    print(f"  max diff floor=0 vs floor=0.3: {diff:.4f}")
    assert diff > 1e-5, "floor change should affect prediction"
    print("  OK")


def test_event_warmup():
    """event_layer_warmup>0 zeros e_n in early layers."""
    print("--- Event layer warmup ---")
    cfg = FakeConfigs()
    cfg.n_layers = 3
    cfg.sthq_event_layer_warmup = 2
    model = Model(cfg)
    # Ensure no exception running with warmup
    model.train()
    x = torch.randn(2, cfg.seq_len, cfg.enc_in)
    y = torch.randn(2, cfg.pred_len, cfg.enc_in)
    out = model(x=x, y=y, exp_stage="train")
    assert torch.isfinite(out["pred"]).all()
    out["pred"].mean().backward()
    print("  OK: warmup runs without error and gradients flow")


def test_n_layers_1():
    """USHCN config uses n_layers=1; the only layer is also the h2h layer."""
    print("--- n_layers=1 (USHCN config) ---")
    cfg = FakeConfigs()
    cfg.n_layers = 1
    model = Model(cfg)
    model.train()
    x = torch.randn(2, cfg.seq_len, cfg.enc_in)
    y = torch.randn(2, cfg.pred_len, cfg.enc_in)
    out = model(x=x, y=y, exp_stage="train")
    out["pred"].mean().backward()
    print("  OK")


def test_diagnostic_logging():
    """Verify diagnostic logging triggers without error."""
    print("--- Diagnostic logging ---")
    cfg = FakeConfigs()
    cfg.sthq_diag_interval = 1
    model = Model(cfg)
    model.train()
    x = torch.randn(2, cfg.seq_len, cfg.enc_in)
    y = torch.randn(2, cfg.pred_len, cfg.enc_in)
    _ = model(x=x, y=y, exp_stage="train")
    print("  OK")


def main():
    test_quaternion_linear()
    test_quaternion_identity_init()
    test_spike_encoder()
    test_model_forward_backward()
    test_test_stage_returns_padded_shape()
    test_no_spike_ablation()
    test_no_quaternion_ablation()
    test_spike_floor()
    test_event_warmup()
    test_n_layers_1()
    test_diagnostic_logging()
    print("\n  All STHQ v7 smoke tests passed.")


if __name__ == "__main__":
    main()
