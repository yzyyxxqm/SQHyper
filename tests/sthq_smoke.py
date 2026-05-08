# Smoke test for STHQ-Net: forward + backward on synthetic data, CPU only.
# Verifies:
#   - Kronecker QuaternionLinear produces same output as cat-based version
#   - Hamilton product is bilinear/correct
#   - Full model forward/backward with no NaN
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.STHQ import (
    Model, QuaternionLinear, hamilton_product,
    _BASIS_A1, _BASIS_AI, _BASIS_AJ, _BASIS_AK,
)


class FakeConfigs:
    d_model = 32
    n_layers = 2
    enc_in = 5
    pred_len = 3
    seq_len = 20
    seq_len_max_irr = None
    pred_len_max_irr = None
    task_name = "short_term_forecast"
    features = "M"
    sthq_k_t = 8
    sthq_k_v = 4
    sthq_omega_min = 0.05
    sthq_omega_max = 0.5
    sthq_use_he_attn_from_layer = 1
    sthq_lambda_tau = 0.0
    sthq_lambda_var = 0.0
    sthq_k_t_per_layer = ""
    sthq_diag_interval = 0


def test_kron_equivalence():
    """Verify Kronecker form of QuaternionLinear is equivalent to cat form."""
    print("--- Kron equivalence ---")
    qlin = QuaternionLinear(8, 8, bias=False)
    R, I, J, K = qlin.R, qlin.I, qlin.J, qlin.K
    # Kron form: assembled inside forward
    x = torch.randn(2, 8)
    out_kron = qlin(x)
    # Cat form: equivalent reference
    W_cat = torch.cat([
        torch.cat([R, -I, -J, -K], 1),
        torch.cat([I,  R, -K,  J], 1),
        torch.cat([J,  K,  R, -I], 1),
        torch.cat([K, -J,  I,  R], 1),
    ], 0)
    out_cat = x @ W_cat.t()
    diff = (out_kron - out_cat).abs().max()
    print(f"max diff Kron vs Cat: {diff.item():.2e}")
    assert diff < 1e-5, f"Kron != Cat: {diff}"
    print("OK")


def test_hamilton_bilinear():
    print("--- Hamilton bilinearity ---")
    Q = 4
    a = torch.randn(2, 4 * Q)
    b = torch.randn(2, 4 * Q)
    c = torch.randn(2, 4 * Q)
    out_lhs = hamilton_product(a + b, c)
    out_rhs = hamilton_product(a, c) + hamilton_product(b, c)
    diff = (out_lhs - out_rhs).abs().max()
    print(f"max diff (a+b)⊗c vs a⊗c+b⊗c: {diff.item():.2e}")
    assert diff < 1e-5
    print("OK")


def test_model():
    print("--- Full model ---")
    torch.manual_seed(0)
    cfg = FakeConfigs()
    model = Model(cfg)
    print(f"params: {sum(p.numel() for p in model.parameters())}")
    B, L, V = 4, cfg.seq_len, cfg.enc_in
    PL = cfg.pred_len
    x = torch.randn(B, L, V)
    x_mask = (torch.rand(B, L, V) > 0.3).float()
    y = torch.randn(B, PL, V)
    y_mask = torch.ones(B, PL, V)

    print("Forward (train) ...")
    model.train()
    out = model(x=x, x_mask=x_mask, y=y, y_mask=y_mask, exp_stage="train")
    print(f"  pred shape: {out['pred'].shape}")
    print(f"  pred mean/std: {out['pred'].mean():.4f} / {out['pred'].std():.4f}")

    print("Backward ...")
    pred, true, mask = out["pred"], out["true"], out["mask"]
    loss = ((pred - true) * mask).pow(2).sum() / (mask.sum() + 1e-6)
    loss.backward()
    print(f"  loss: {loss.item():.4f}")
    nan_grad = False
    none_grad = []
    grad_count = 0
    for name, p in model.named_parameters():
        if p.grad is None:
            none_grad.append(name)
        else:
            grad_count += 1
            if torch.isnan(p.grad).any():
                print(f"  NaN grad in {name}")
                nan_grad = True
    print(f"  gradients on {grad_count} params, none-grad: {len(none_grad)}, NaN: {nan_grad}")
    if none_grad:
        print(f"  none-grad: {none_grad[:5]}")
    assert not nan_grad

    print("Forward (test) ...")
    model.eval()
    with torch.no_grad():
        out_test = model(x=x, x_mask=x_mask, y=y, y_mask=y_mask, exp_stage="test")
    print(f"  test pred shape: {out_test['pred'].shape}")
    assert out_test["pred"].shape == (B, PL, V)


def test_query_receives_messages():
    """Regression test: query positions (forecast targets, spike=0) MUST
    still receive hyperedge messages and have their state updated.

    Bug fixed 2026-05-07: previously m_norm used aggregation-spike-gated
    weights for distribution, causing queries to get zero messages.
    """
    print("--- Query position update ---")
    torch.manual_seed(42)
    cfg = FakeConfigs()
    model = Model(cfg)
    model.eval()

    B, L, V = 2, cfg.seq_len, cfg.enc_in
    PL = cfg.pred_len
    x = torch.randn(B, L, V)
    x_mask = torch.ones(B, L, V)
    y = torch.zeros(B, PL, V)
    y_mask = torch.ones(B, PL, V)

    # Run twice with different observed values; query positions should give
    # different outputs (proving they receive observed-value information).
    out1 = model(x=x, x_mask=x_mask, y=y, y_mask=y_mask, exp_stage="test")
    out2 = model(x=x * 5 + 7, x_mask=x_mask, y=y, y_mask=y_mask, exp_stage="test")
    diff = (out1["pred"] - out2["pred"]).abs().max().item()
    print(f"  query pred diff with different observations: {diff:.4f}")
    assert diff > 1e-3, (
        f"Query positions did NOT respond to observation changes (diff={diff:.2e}). "
        f"They are receiving zero messages — bug regression!"
    )
    print("  OK: queries receive non-trivial messages")


def test_multiscale_kt():
    """Verify per-layer K_t schedule is applied correctly."""
    print("--- Multi-scale K_t ---")
    cfg = FakeConfigs()
    cfg.n_layers = 3
    cfg.sthq_k_t_per_layer = "16,8,4"
    model = Model(cfg)
    kts = [layer.k_t for layer in model.layers]
    print(f"  kt per layer: {kts}")
    assert kts == [16, 8, 4], f"K_t schedule mismatch: {kts}"
    # τ shape per layer should match K_t
    for l, layer in enumerate(model.layers):
        assert layer.tau.numel() == kts[l], (
            f"Layer {l}: tau has {layer.tau.numel()} but expected {kts[l]}"
        )
    # Forward should still work
    B, L, V = 2, cfg.seq_len, cfg.enc_in
    x = torch.randn(B, L, V)
    out = model(x=x, exp_stage="train")
    assert out["pred"].dim() == 2 and out["pred"].shape[0] == B
    assert torch.isfinite(out["pred"]).all()
    print(f"  pred shape: {tuple(out['pred'].shape)}")
    print("  OK: multi-scale K_t accepted and forward works")


def test_diagnostic_logging():
    """Verify diagnostic logging triggers without error."""
    print("--- Diagnostic logging ---")
    cfg = FakeConfigs()
    cfg.sthq_diag_interval = 1  # log every step
    model = Model(cfg)
    model.train()
    B, L, V = 2, cfg.seq_len, cfg.enc_in
    x = torch.randn(B, L, V)
    out = model(x=x, exp_stage="train")  # should print diag once
    out = model(x=x, exp_stage="train")  # and again
    print("  OK: diagnostic logging executed")


def main():
    test_kron_equivalence()
    test_hamilton_bilinear()
    test_model()
    test_query_receives_messages()
    test_multiscale_kt()
    test_diagnostic_logging()
    print("\n✅ All smoke tests passed.")


if __name__ == "__main__":
    main()
