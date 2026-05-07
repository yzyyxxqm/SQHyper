# Smoke test for PE-RQH: forward + backward on synthetic data, CPU only.
import os
import sys
import torch

# Add repo root to sys.path so this works from any cwd / install location.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.PERQH import Model


class FakeConfigs:
    """Minimal ExpConfigs stand-in for smoke testing."""
    d_model = 64
    n_layers = 2
    enc_in = 5
    pred_len = 3
    seq_len = 20
    seq_len_max_irr = None
    pred_len_max_irr = None
    task_name = "short_term_forecast"
    features = "M"
    perqh_n_codes = 16
    perqh_top_m = 3
    perqh_tau_init = 1.0
    perqh_tau_min = 0.5
    perqh_tau_decay = 0.999
    perqh_lambda_div = 0.1
    perqh_lambda_commit = 0.01


def main():
    torch.manual_seed(0)
    cfg = FakeConfigs()
    model = Model(cfg)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    B, L, V = 4, cfg.seq_len, cfg.enc_in
    PL = cfg.pred_len
    x = torch.randn(B, L, V)
    # Use masked data to exercise pad_and_flatten path
    x_mask = (torch.rand(B, L, V) > 0.3).float()
    y = torch.randn(B, PL, V)
    y_mask = torch.ones(B, PL, V)

    print("\n--- Forward (train) ---")
    model.train()
    out = model(x=x, x_mask=x_mask, y=y, y_mask=y_mask, exp_stage="train")
    print(f"keys: {list(out.keys())}")
    print(f"pred shape: {out['pred'].shape}")
    print(f"true shape: {out['true'].shape}")
    print(f"mask shape: {out['mask'].shape}")
    print(f"aux_loss: {out.get('aux_loss')}")

    print("\n--- Backward ---")
    pred = out["pred"]
    true = out["true"]
    mask = out["mask"]
    mse = ((pred - true) * mask).pow(2).sum() / (mask.sum() + 1e-6)
    total = mse + out.get("aux_loss", torch.tensor(0.0))
    total.backward()
    print(f"loss: {total.item():.6f} (mse={mse.item():.6f})")

    # check gradients
    has_nan = False
    grad_count = 0
    for name, p in model.named_parameters():
        if p.grad is not None:
            grad_count += 1
            if torch.isnan(p.grad).any():
                print(f"NaN grad in {name}")
                has_nan = True
    print(f"Gradients OK on {grad_count} params, NaN={has_nan}")

    print("\n--- Forward (test) ---")
    model.eval()
    with torch.no_grad():
        out_test = model(x=x, x_mask=x_mask, y=y, y_mask=y_mask, exp_stage="test")
    print(f"test pred shape: {out_test['pred'].shape}")
    print(f"test true shape: {out_test['true'].shape}")
    assert out_test["pred"].shape == (B, PL, V), \
        f"Expected ({B},{PL},{V}), got {out_test['pred'].shape}"
    print("\n✅ Smoke test passed.")


if __name__ == "__main__":
    main()
