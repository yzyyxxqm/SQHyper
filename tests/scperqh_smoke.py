# Smoke test for SC-PERQH: forward + backward on synthetic data, CPU only.
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.SCPERQH import Model


class FakeConfigs:
    d_model = 64
    n_layers = 2
    enc_in = 5
    pred_len = 3
    seq_len = 20
    seq_len_max_irr = None
    pred_len_max_irr = None
    task_name = "short_term_forecast"
    features = "M"
    scperqh_k_time = 8
    scperqh_k_var = 6
    scperqh_k_event = 8
    scperqh_top_t = 2
    scperqh_top_v = 2
    scperqh_top_e = 2
    scperqh_tau_init = 1.0
    scperqh_tau_min = 0.5
    scperqh_tau_decay = 0.999
    scperqh_lambda_div = 0.05
    scperqh_lambda_commit = 0.005


def main():
    torch.manual_seed(0)
    cfg = FakeConfigs()
    model = Model(cfg)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"  time_codes: {model.codebook.time_codes.shape}")
    print(f"  var_codes: {model.codebook.var_codes.shape}")
    print(f"  event_codes: {model.codebook.event_codes.shape}")

    B, L, V = 4, cfg.seq_len, cfg.enc_in
    PL = cfg.pred_len
    x = torch.randn(B, L, V)
    x_mask = (torch.rand(B, L, V) > 0.3).float()
    y = torch.randn(B, PL, V)
    y_mask = torch.ones(B, PL, V)

    print("\n--- Forward (train) ---")
    model.train()
    out = model(x=x, x_mask=x_mask, y=y, y_mask=y_mask, exp_stage="train")
    print(f"keys: {list(out.keys())}")
    print(f"pred shape: {out['pred'].shape}")
    print(f"aux_loss: {out.get('aux_loss')}")

    print("\n--- Backward ---")
    pred, true, mask = out["pred"], out["true"], out["mask"]
    mse = ((pred - true) * mask).pow(2).sum() / (mask.sum() + 1e-6)
    total = mse + out.get("aux_loss", torch.tensor(0.0))
    total.backward()
    print(f"loss: {total.item():.6f} (mse={mse.item():.6f})")

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
    assert out_test["pred"].shape == (B, PL, V)
    print("\n✅ Smoke test passed.")


if __name__ == "__main__":
    main()
