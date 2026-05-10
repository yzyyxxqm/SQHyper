"""Smoke test for SQHH v2 (SQHyper backbone + SRI + SQC).

Verifies:
  - Model instantiation and forward pass
  - train / val / test stage output shapes
  - SRI gate values in [0, 1] and event features finite
  - SQC rotation preserves quaternion norm (up to numerical tolerance)
  - Ablations (no_sri, no_sqc, no_qmf) run without error
  - Full vs each ablation produces different outputs (mechanism active)
  - No NaN / Inf in gradients
"""
import sys
from pathlib import Path
import torch

# Add repo root to path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from models.SQHH import (
    Model, QuaternionLinear, SpikeRefractoryIncidence,
    SpikeQuaternionRotation, hamilton_product,
)


class FakeConfigs:
    """Minimal config mimicking ExpConfigs."""
    task_name = "short_term_forecast"
    features = "M"
    seq_len = 30
    pred_len = 3
    seq_len_max_irr = 30
    pred_len_max_irr = 3
    enc_in = 5
    dec_in = 5
    c_out = 5
    d_model = 64   # must be divisible by 4
    n_layers = 2
    n_heads = 2
    loss = "MSE"
    sqhh_no_sri = 0
    sqhh_no_sqc = 0
    sqhh_no_qmf = 0
    sqhh_diag_interval = 100


def make_irregular_batch(B=2, L=30, V=5, seed=0):
    """Make an irregular multivariate batch: ~60% observed, random."""
    torch.manual_seed(seed)
    x = torch.randn(B, L, V)
    mask = (torch.rand(B, L, V) > 0.4).float()
    y = torch.randn(B, 3, V)
    y_mask = (torch.rand(B, 3, V) > 0.3).float()
    return x, mask, y, y_mask


def smoke():
    print("--- SQHH v2 smoke test ---")
    cfg = FakeConfigs()
    torch.manual_seed(42)
    model = Model(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model params: {n_params/1e3:.1f}K")
    print(f"  structure: SQHyper backbone + SRI + SQC")

    x, mask, y, y_mask = make_irregular_batch()

    # ---- Train stage ----
    model.train()
    out = model(x=x, x_mask=mask, y=y, y_mask=y_mask, exp_stage="train")
    assert "pred" in out and "true" in out and "mask" in out
    print(f"  train pred shape: {tuple(out['pred'].shape)} (flattened)")
    assert out["pred"].shape == out["true"].shape == out["mask"].shape
    assert not torch.isnan(out["pred"]).any()

    # backward
    loss = ((out["pred"] - out["true"])**2 * out["mask"]).sum() / (
        out["mask"].sum() + 1e-6)
    loss.backward()
    print(f"  train loss: {loss.item():.4f}  (backward OK)")

    # verify no NaN grads
    bad = []
    for name, p in model.named_parameters():
        if p.grad is not None and (torch.isnan(p.grad).any()
                                   or torch.isinf(p.grad).any()):
            bad.append(name)
    assert not bad, f"NaN/Inf grads in: {bad[:5]}"
    print(f"  all {sum(1 for _ in model.parameters())} param grads finite ✓")

    # ---- Val stage ----
    model.zero_grad()
    out_v = model(x=x, x_mask=mask, y=y, y_mask=y_mask, exp_stage="val")
    assert out_v["pred"].shape == out_v["true"].shape
    print(f"  val pred shape: {tuple(out_v['pred'].shape)}")

    # ---- Test stage (unpacked) ----
    model.eval()
    out_t = model(x=x, x_mask=mask, y=y, y_mask=y_mask, exp_stage="test")
    expected_test_shape = (2, 3, 5)  # (B, pred_len, V)
    assert out_t["pred"].shape == expected_test_shape, (
        f"expected {expected_test_shape}, got {out_t['pred'].shape}")
    print(f"  test pred shape: {tuple(out_t['pred'].shape)} ✓")


def test_sri_output_range():
    print("--- SRI output range ---")
    torch.manual_seed(7)
    D, V, B, N = 64, 5, 2, 20
    sri = SpikeRefractoryIncidence(D, V)
    obs = torch.randn(B, N, D)
    mask = (torch.rand(B, N) > 0.3).float()
    var_id = torch.randint(0, V, (B, N))
    time_norm = torch.rand(B, N)

    # Build variable incidence matrix: (B, V, N)
    vim = torch.zeros(B, V, N)
    for b in range(B):
        for n in range(N):
            vim[b, var_id[b, n], n] = mask[b, n]

    g_n, e_n = sri(obs, mask, vim, var_id, time_norm)
    assert g_n.shape == (B, N)
    assert e_n.shape == (B, N, D // 4)
    # Gate in [0, 1]
    assert (g_n >= 0).all() and (g_n <= 1).all(), (
        f"g_n range [{g_n.min():.4f}, {g_n.max():.4f}] out of bounds")
    # Padded positions masked to 0
    g_n_padded = g_n[mask == 0]
    if g_n_padded.numel() > 0:
        assert (g_n_padded == 0).all(), "Padded gate should be 0"
    print(f"  g_n in [{g_n.min():.4f}, {g_n.max():.4f}] ✓")
    print(f"  e_n shape {tuple(e_n.shape)} finite: {torch.isfinite(e_n).all().item()}")


def test_sqc_preserves_norm():
    print("--- SQC preserves quaternion norm ---")
    torch.manual_seed(13)
    B, N, D = 2, 10, 32
    q = torch.randn(B, N, D)
    spike = torch.rand(B, N)
    sqc = SpikeQuaternionRotation()
    q_rot = sqc(q, spike)

    # Check norm preservation (Hamilton product with unit rotation quat preserves |q|)
    def quat_norm_sq(q):
        Q = q.shape[-1] // 4
        return (q[..., :Q]**2 + q[..., Q:2*Q]**2
                + q[..., 2*Q:3*Q]**2 + q[..., 3*Q:]**2)

    n_before = quat_norm_sq(q)
    n_after = quat_norm_sq(q_rot)
    diff = (n_before - n_after).abs().max().item()
    assert diff < 1e-4, f"Norm not preserved: max diff {diff:.2e}"
    print(f"  max |norm_before - norm_after| = {diff:.2e} ✓")


def test_ablations():
    print("--- Ablations run ---")
    cfg = FakeConfigs()
    x, mask, y, y_mask = make_irregular_batch()

    outputs = {}
    for ablation, kwargs in [
        ("full", {}),
        ("no_sri", {"sqhh_no_sri": 1}),
        ("no_sqc", {"sqhh_no_sqc": 1}),
        ("no_qmf", {"sqhh_no_qmf": 1}),
    ]:
        c = FakeConfigs()
        for k, v in kwargs.items():
            setattr(c, k, v)
        torch.manual_seed(0)  # same init for comparison
        m = Model(c)
        m.eval()
        with torch.no_grad():
            o = m(x=x, x_mask=mask, y=y, y_mask=y_mask, exp_stage="test")
        outputs[ablation] = o["pred"]
        print(f"  {ablation:<10} pred_mean={o['pred'].mean().item():+.4f}  OK")

    # Ensure each ablation changes the output
    for name in ["no_sri", "no_sqc", "no_qmf"]:
        diff = (outputs["full"] - outputs[name]).abs().max().item()
        print(f"  full vs {name}: max diff = {diff:.4f}", end="")
        assert diff > 1e-6, (
            f"Ablation {name} has no effect (diff={diff:.2e})")
        print("  ✓")


def test_diagnostic():
    print("--- Diagnostic logging ---")
    cfg = FakeConfigs()
    cfg.sqhh_diag_interval = 1
    model = Model(cfg)
    model.train()
    x, mask, y, y_mask = make_irregular_batch()
    for _ in range(2):
        out = model(x=x, x_mask=mask, y=y, y_mask=y_mask, exp_stage="train")
    print(f"  diag_step = {int(model._diag_step.item())} ✓")


if __name__ == "__main__":
    smoke()
    test_sri_output_range()
    test_sqc_preserves_norm()
    test_ablations()
    test_diagnostic()
    print("\n  All SQHH v2 smoke tests passed.")
