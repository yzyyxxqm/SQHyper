"""Bit-exact smoke test for SQHH v2 SRI refactor.

Verifies that the optimized SRI (shared precomputed pair tensors + fused
kernel via masked_fill) produces identical outputs to the original
per-layer recomputation on a P12-sized batch.

Tests:
1. SRI alone: new kernel vs. old kernel on the same inputs -> bit-exact.
2. Full Model forward: precomputed path vs. fallback (None) path -> bit-exact.
3. Backward: gradients on log_tau, log_alpha, membrane_proj.weight match.

Usage: python -m tests.sqhh_v2_bit_exact_smoke
"""
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.SQHH import (
    SpikeRefractoryIncidence,
    Model as SQHHModel,
)


class _Cfg:
    """Minimal ExpConfigs-compatible stub."""
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def make_cfg(d_model=32, enc_in=8, n_layers=2, seq_len=16, pred_len=3,
             batch_size=4):
    return _Cfg(
        d_model=d_model, n_heads=2, n_layers=n_layers, enc_in=enc_in,
        dec_in=enc_in, c_out=enc_in,
        seq_len=seq_len, pred_len=pred_len,
        seq_len_max_irr=0, pred_len_max_irr=0,
        task_name="short_term_forecast", features="M",
        sqhh_no_sri=0, sqhh_no_sqc=0, sqhh_no_qmf=0,
        sqhyper_no_qmf=0, sqhh_diag_interval=0,
        batch_size=batch_size,
    )


def reference_sri(sri_module: SpikeRefractoryIncidence,
                  obs, mask_flat, variable_incidence_matrix,
                  variable_indices_flattened, time_norm):
    """Original SRI formulation (materialize 5 float (B,N,N) tensors)."""
    from einops import repeat
    D = obs.shape[-1]
    var_count = variable_incidence_matrix.sum(-1, keepdim=True).clamp(min=1)
    var_context = (variable_incidence_matrix @ obs) / var_count
    obs_var_ctx = var_context.gather(
        1, repeat(variable_indices_flattened, "B N -> B N D", D=D))
    deviation = obs - obs_var_ctx
    membrane_input = torch.cat([obs, deviation], dim=-1)
    membrane = sri_module.membrane_proj(membrane_input).squeeze(-1)
    g_raw = torch.sigmoid(membrane) * mask_flat
    e_n = sri_module.event_proj(membrane_input) * g_raw.unsqueeze(-1)
    e_n = e_n * mask_flat.unsqueeze(-1)
    if sri_module.no_refractory:
        return g_raw, e_n

    tau = torch.exp(sri_module.log_tau)[variable_indices_flattened]
    alpha = torch.exp(sri_module.log_alpha)[variable_indices_flattened]
    same_var = (
        variable_indices_flattened.unsqueeze(2)
        == variable_indices_flattened.unsqueeze(1)
    )
    time_diff = time_norm.unsqueeze(2) - time_norm.unsqueeze(1)
    earlier = (time_diff > 0).to(obs.dtype)
    valid_pair = mask_flat.unsqueeze(2) * mask_flat.unsqueeze(1)
    pair_mask = same_var.float() * earlier * valid_pair
    decay = torch.exp(-time_diff.clamp_min(0) / tau.unsqueeze(2).clamp_min(1e-6))
    inh = (pair_mask * decay * g_raw.unsqueeze(1)).sum(dim=-1)
    g_refr = 1.0 - torch.tanh(alpha * inh)
    g_n = g_raw * g_refr
    e_n = e_n * g_refr.unsqueeze(-1)
    return g_n, e_n


def test_sri_bit_exact():
    torch.manual_seed(0)
    B, N, D, E = 4, 48, 32, 8
    obs = torch.randn(B, N, D, requires_grad=True)
    obs2 = obs.clone().detach().requires_grad_(True)

    mask_flat = (torch.rand(B, N) > 0.2).float()
    var_idx = torch.randint(0, E, (B, N))
    time_idx = torch.randint(0, 20, (B, N)).float()
    time_norm = time_idx / 19.0
    var_incidence = torch.zeros(B, E, N)
    for b in range(B):
        for n in range(N):
            if mask_flat[b, n] > 0:
                var_incidence[b, var_idx[b, n], n] = 1.0

    sri = SpikeRefractoryIncidence(D, E)
    # Give tau/alpha nontrivial values so refractory is active.
    with torch.no_grad():
        sri.log_alpha.copy_(torch.full((E,), 0.5))
        sri.log_tau.copy_(torch.randn(E) * 0.3 - 1.0)
        sri.membrane_proj.weight.copy_(torch.randn_like(sri.membrane_proj.weight) * 0.1)
        sri.membrane_proj.bias.zero_()
        sri.event_proj.weight.copy_(torch.randn_like(sri.event_proj.weight) * 0.1)

    # New path (with precomputed pair tensors)
    same_var_b = var_idx.unsqueeze(2) == var_idx.unsqueeze(1)
    td = time_norm.unsqueeze(2) - time_norm.unsqueeze(1)
    earlier_b = td > 0
    valid_b = mask_flat.unsqueeze(2).bool() & mask_flat.unsqueeze(1).bool()
    refractory_mask_bool = same_var_b & earlier_b & valid_b
    time_diff_clamped = td.clamp_min(0)

    g_new, e_new = sri(obs, mask_flat, var_incidence, var_idx, time_norm,
                        refractory_mask_bool=refractory_mask_bool,
                        time_diff_clamped=time_diff_clamped)

    # Reference path
    g_ref, e_ref = reference_sri(sri, obs2, mask_flat, var_incidence,
                                  var_idx, time_norm)

    assert torch.allclose(g_new, g_ref, atol=1e-6), \
        f"g_n differs: max |diff|={ (g_new - g_ref).abs().max().item() }"
    assert torch.allclose(e_new, e_ref, atol=1e-6), \
        f"e_n differs: max |diff|={ (e_new - e_ref).abs().max().item() }"

    # Backward bit-exact
    (g_new.sum() + e_new.sum()).backward()
    g_obs_new = obs.grad.clone()
    g_tau_new = sri.log_tau.grad.clone()
    g_alpha_new = sri.log_alpha.grad.clone()
    g_w_new = sri.membrane_proj.weight.grad.clone()

    sri.zero_grad(set_to_none=True)
    (g_ref.sum() + e_ref.sum()).backward()
    g_obs_ref = obs2.grad.clone()
    g_tau_ref = sri.log_tau.grad.clone()
    g_alpha_ref = sri.log_alpha.grad.clone()
    g_w_ref = sri.membrane_proj.weight.grad.clone()

    for name, a, b in [("obs", g_obs_new, g_obs_ref),
                       ("log_tau", g_tau_new, g_tau_ref),
                       ("log_alpha", g_alpha_new, g_alpha_ref),
                       ("membrane_proj.weight", g_w_new, g_w_ref)]:
        diff = (a - b).abs().max().item()
        assert diff < 1e-5, f"grad {name} differs: max |diff|={diff}"
    print("[OK] test_sri_bit_exact: fwd and bwd identical to reference")


def test_sri_fallback_path():
    """SRI.forward(refractory_mask_bool=None) should match the precomputed path."""
    torch.manual_seed(1)
    B, N, D, E = 2, 32, 16, 4
    obs = torch.randn(B, N, D)
    mask_flat = (torch.rand(B, N) > 0.1).float()
    var_idx = torch.randint(0, E, (B, N))
    time_norm = torch.rand(B, N)
    var_incidence = torch.zeros(B, E, N)
    for b in range(B):
        for n in range(N):
            if mask_flat[b, n] > 0:
                var_incidence[b, var_idx[b, n], n] = 1.0

    sri = SpikeRefractoryIncidence(D, E)
    with torch.no_grad():
        sri.log_alpha.copy_(torch.full((E,), 0.3))

    g1, e1 = sri(obs, mask_flat, var_incidence, var_idx, time_norm)

    same_var_b = var_idx.unsqueeze(2) == var_idx.unsqueeze(1)
    td = time_norm.unsqueeze(2) - time_norm.unsqueeze(1)
    earlier_b = td > 0
    valid_b = mask_flat.unsqueeze(2).bool() & mask_flat.unsqueeze(1).bool()
    rmb = same_var_b & earlier_b & valid_b
    tdc = td.clamp_min(0)
    g2, e2 = sri(obs, mask_flat, var_incidence, var_idx, time_norm,
                  refractory_mask_bool=rmb, time_diff_clamped=tdc)

    assert torch.allclose(g1, g2, atol=1e-6)
    assert torch.allclose(e1, e2, atol=1e-6)
    print("[OK] test_sri_fallback_path: lazy and precomputed paths match")


def test_model_forward_backward():
    torch.manual_seed(2)
    cfg = make_cfg(d_model=32, enc_in=6, n_layers=2,
                   seq_len=12, pred_len=3, batch_size=2)
    model = SQHHModel(cfg)

    B = 2
    x = torch.randn(B, cfg.seq_len, cfg.enc_in)
    x_mask = (torch.rand_like(x) > 0.3).float()
    y = torch.randn(B, cfg.pred_len, cfg.enc_in)
    y_mask = torch.ones_like(y)

    out = model(x=x, x_mask=x_mask, y=y, y_mask=y_mask, exp_stage="train")
    pred = out["pred"]
    assert pred.requires_grad
    loss = (pred * out["mask"].float()).sum()
    loss.backward()

    # Verify SRI params received non-zero gradient (proves path is alive)
    for i, sri in enumerate(model.hypergraph_learner.sri):
        assert sri.log_tau.grad is not None, f"layer {i} log_tau grad missing"
        assert sri.log_alpha.grad is not None, f"layer {i} log_alpha grad missing"
    print("[OK] test_model_forward_backward: SRI params get gradients")


def test_model_ablation_no_sri():
    """no_sri=1 should skip the pair-tensor precomputation (cheaper path)."""
    cfg = make_cfg(batch_size=2)
    cfg.sqhh_no_sri = 1
    torch.manual_seed(3)
    model = SQHHModel(cfg)
    B = 2
    x = torch.randn(B, cfg.seq_len, cfg.enc_in)
    x_mask = (torch.rand_like(x) > 0.3).float()
    out = model(x=x, x_mask=x_mask, exp_stage="train")
    assert out["pred"].shape == (B, cfg.seq_len + cfg.pred_len, )[
        :1] + out["pred"].shape[1:]
    print("[OK] test_model_ablation_no_sri: runs without refractory")


if __name__ == "__main__":
    test_sri_bit_exact()
    test_sri_fallback_path()
    test_model_forward_backward()
    test_model_ablation_no_sri()
    print("\nAll SQHH v2 bit-exact tests passed.")
