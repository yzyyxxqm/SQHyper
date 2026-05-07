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


def main():
    test_kron_equivalence()
    test_hamilton_bilinear()
    test_model()
    print("\n✅ All smoke tests passed.")


if __name__ == "__main__":
    main()
