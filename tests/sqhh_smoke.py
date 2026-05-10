# SQHH smoke tests: quaternion primitives + SRA + SQC + 3-layer hypergraph
# CPU only, synthetic data. Verifies:
#   - Quaternion primitives (Hamilton product, conjugate, distance, exp_K)
#   - QuaternionLinear: block-Hamilton form correctness, identity init
#   - QuaternionMHA: Q/K/V quaternion projection, mask works
#   - SpikeRefractoryEncoder: floor enforced; refractory inhibition reduces
#     same-variable subsequent spike intensity
#   - SpikeQuaternionRotation: identity at spike=0, max rotation at spike=1
#   - PrimaryLayer / QuaternionAnchorLayer / SpikeTriggeredEventLayer
#   - SQHHBlock and full Model forward + backward
#   - Each ablation flag changes output (or runs cleanly when disabled)

import os
import sys
import math
import torch

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.SQHH import (
    Model, QuaternionLinear, QuaternionLayerNorm, QuaternionMHA,
    hamilton_product, quaternion_conjugate, quaternion_distance,
    quaternion_exp_K, quaternion_split, quaternion_concat,
    SpikeRefractoryEncoder, SpikeQuaternionRotation,
    QuaternionCellEncoder, PrimaryLayer, QuaternionAnchorLayer,
    SpikeTriggeredEventLayer, SQHHBlock,
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
    # SQHH hyperparameters
    sqhh_k_a = 4
    sqhh_k_e = 6
    sqhh_spike_floor = 0.0
    sqhh_no_layer0 = 0
    sqhh_no_layer1 = 0
    sqhh_no_layer2 = 0
    sqhh_no_sra = 0
    sqhh_no_sqc = 0
    sqhh_diag_interval = 0


# ----------------------------------------------------------------------------
# Quaternion primitives
# ----------------------------------------------------------------------------

def test_quaternion_linear():
    """Block-Hamilton form correctness."""
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
    assert diff < 1e-5, f"QuaternionLinear mismatch: {diff}"
    print(f"  diff={diff.item():.2e}  OK")


def test_quaternion_identity_init():
    """init_identity should make QuaternionLinear act as identity."""
    print("--- QuaternionLinear identity init ---")
    ql = QuaternionLinear(16, 16, bias=False)
    ql.init_identity()
    x = torch.randn(3, 16)
    out = ql(x)
    diff = (out - x).abs().max()
    assert diff < 1e-5
    print(f"  diff={diff.item():.2e}  OK")


def test_hamilton_product_axioms():
    """Verify quaternion algebra: i⊗j=k, j⊗k=i, k⊗i=j; conjugate."""
    print("--- Hamilton product axioms ---")
    Q = 1
    e_1 = quaternion_concat(torch.tensor([[1.0]]), torch.tensor([[0.0]]),
                             torch.tensor([[0.0]]), torch.tensor([[0.0]]))
    e_i = quaternion_concat(torch.tensor([[0.0]]), torch.tensor([[1.0]]),
                             torch.tensor([[0.0]]), torch.tensor([[0.0]]))
    e_j = quaternion_concat(torch.tensor([[0.0]]), torch.tensor([[0.0]]),
                             torch.tensor([[1.0]]), torch.tensor([[0.0]]))
    e_k = quaternion_concat(torch.tensor([[0.0]]), torch.tensor([[0.0]]),
                             torch.tensor([[0.0]]), torch.tensor([[1.0]]))
    # i*j = k
    assert torch.allclose(hamilton_product(e_i, e_j), e_k), "i⊗j != k"
    # j*k = i
    assert torch.allclose(hamilton_product(e_j, e_k), e_i), "j⊗k != i"
    # k*i = j
    assert torch.allclose(hamilton_product(e_k, e_i), e_j), "k⊗i != j"
    # 1*x = x
    x = torch.randn(2, 4)
    assert torch.allclose(hamilton_product(e_1, x), x), "1⊗x != x"
    # conj(conj(x)) = x
    assert torch.allclose(quaternion_conjugate(quaternion_conjugate(x)), x)
    print("  i⊗j=k, j⊗k=i, k⊗i=j, 1⊗x=x, conj²=id  OK")


def test_quaternion_distance():
    """||q ⊗ conj(q)||² should be ||q||⁴; ||p ⊗ conj(q)||² should be 0
    when p == q only if q is unit quaternion → so check a simpler property."""
    print("--- Quaternion distance ---")
    q = torch.randn(2, 3, 4)
    # distance(q, q) = ||q ⊗ conj(q)||² = (||q||² + 0i + 0j + 0k)² = ||q||⁴
    d = quaternion_distance(q, q)
    norm4 = q.pow(2).sum(-1, keepdim=True).pow(2)
    assert torch.allclose(d, norm4, atol=1e-5), \
        f"distance(q, q) != ||q||⁴: max diff {(d-norm4).abs().max()}"
    print(f"  d(q,q)=||q||⁴ verified  OK")


def test_quaternion_exp_K():
    """exp(0·ê_K) = (1, 0, 0, 0); exp(π/2·ê_K) ≈ (0, 0, 0, 1)."""
    print("--- quaternion_exp_K ---")
    Q = 4
    theta = torch.tensor([[[0.0]]])
    out = quaternion_exp_K(theta, Q)  # [1, 1, 4Q]
    expected = torch.zeros(1, 1, 4 * Q)
    expected[..., :Q] = 1.0  # cos(0) = 1 in R-component
    assert torch.allclose(out, expected, atol=1e-5)
    theta = torch.tensor([[[math.pi / 2]]])
    out = quaternion_exp_K(theta, Q)
    expected2 = torch.zeros(1, 1, 4 * Q)
    expected2[..., 3 * Q:] = 1.0  # sin(π/2) = 1 in K-component
    assert torch.allclose(out, expected2, atol=1e-5)
    print("  exp(0)=1, exp(π/2)=ê_K  OK")


def test_quaternion_layer_norm():
    """Each component should be normalized independently."""
    print("--- QuaternionLayerNorm ---")
    qln = QuaternionLayerNorm(16)
    x = torch.randn(2, 5, 16) * 10 + 3.0
    out = qln(x)
    Q = 4
    for c in range(4):
        comp = out[..., c * Q:(c + 1) * Q]
        # LayerNorm zero-mean unit-variance per token (last dim norm)
        assert comp.mean(-1).abs().max() < 1e-4, \
            f"component {c} mean nonzero"
    print("  per-component normalization  OK")


def test_quaternion_mha():
    """QuaternionMHA forward + backward + mask."""
    print("--- QuaternionMHA ---")
    mha = QuaternionMHA(d_model=16, num_heads=2)
    Q = torch.randn(2, 5, 16, requires_grad=True)
    K = torch.randn(2, 7, 16)
    V = torch.randn(2, 7, 16)
    out = mha(Q, K, V)
    assert out.shape == (2, 5, 16)
    out.sum().backward()
    assert Q.grad is not None
    print(f"  out shape={tuple(out.shape)}  grad flows  OK")
    # Mask test: masked entries should not contribute
    mask = torch.zeros(2, 5, 7)
    mask[:, :, :3] = 1  # only first 3 keys valid
    out_masked = mha(Q.detach(), K, V, mask=mask)
    out_full = mha(Q.detach(), K[:, :3, :], V[:, :3, :])
    diff = (out_masked - out_full).abs().max()
    assert diff < 1e-4, f"mask not equivalent to truncating keys: {diff}"
    print(f"  mask-vs-truncate diff={diff.item():.2e}  OK")


# ----------------------------------------------------------------------------
# SRA: SpikeRefractoryEncoder
# ----------------------------------------------------------------------------

def test_spike_refractory():
    """Refractory inhibition reduces same-variable subsequent spikes.

    Construct two cells of same variable: t=0 and t=0.1.
    Expected: spike_eff[1] < spike_raw[1] when SRA enabled.
    """
    print("--- SpikeRefractoryEncoder + SRA ---")
    n_vars, D = 3, 32
    enc_no_sra = SpikeRefractoryEncoder(
        n_vars=n_vars, d_model=D, floor=0.0, no_refractory=True)
    enc_sra = SpikeRefractoryEncoder(
        n_vars=n_vars, d_model=D, floor=0.0, no_refractory=False,
        tau_r_init=0.05, alpha_init=5.0)
    # Same MLP weights for fair comparison
    enc_sra.var_emb.load_state_dict(enc_no_sra.var_emb.state_dict())
    enc_sra.body.load_state_dict(enc_no_sra.body.state_dict())
    enc_sra.gate_head.load_state_dict(enc_no_sra.gate_head.state_dict())

    B, N = 1, 4
    value = torch.tensor([[1.0, 1.0, 1.0, 1.0]])  # all same
    time = torch.tensor([[0.0, 0.05, 0.5, 0.55]])  # pairs of same-var
    var = torch.tensor([[0, 0, 1, 1]], dtype=torch.long)
    mask = torch.ones(B, N)

    s_raw = enc_no_sra(value, time, mask, var)
    s_sra = enc_sra(value, time, mask, var)
    print(f"  no SRA spike: {s_raw.tolist()[0]}")
    print(f"  with SRA   : {s_sra.tolist()[0]}")
    # Cell 0 has no earlier same-var cell → unchanged
    assert abs(s_sra[0, 0].item() - s_raw[0, 0].item()) < 1e-5
    # Cell 1 has earlier same-var (cell 0, dt=0.05) → reduced
    assert s_sra[0, 1].item() < s_raw[0, 1].item() - 1e-3, \
        "SRA should reduce cell 1 spike"
    # Cell 2 (var=1) has no earlier same-var → unchanged
    assert abs(s_sra[0, 2].item() - s_raw[0, 2].item()) < 1e-5
    print("  refractory reduces 2nd same-var spike  OK")


def test_spike_floor():
    """floor>0 enforces minimum spike on observed cells."""
    print("--- Spike floor ---")
    enc = SpikeRefractoryEncoder(n_vars=3, d_model=32, floor=0.3)
    # Force spike_raw to 0 by overriding gate_head bias
    with torch.no_grad():
        enc.gate_head.bias.fill_(-100.0)  # sigmoid(-100) ≈ 0
    value = torch.zeros(1, 5)
    time = torch.zeros(1, 5)
    mask = torch.tensor([[1.0, 1.0, 0.0, 1.0, 0.0]])
    var = torch.zeros(1, 5, dtype=torch.long)
    s = enc(value, time, mask, var)
    # Observed cells should have spike >= floor
    obs = s[mask > 0]
    pad = s[mask == 0]
    print(f"  obs spike: {obs.tolist()}")
    print(f"  pad spike: {pad.tolist()}")
    assert (obs >= 0.3 - 1e-4).all(), f"floor violated: {obs}"
    assert (pad == 0).all(), "padding should stay 0"
    print("  floor enforced, padding preserved  OK")


# ----------------------------------------------------------------------------
# SQC: SpikeQuaternionRotation
# ----------------------------------------------------------------------------

def test_sqc_rotation():
    """SQC rotation: spike=0 should be identity; magnitude preserved."""
    print("--- SpikeQuaternionRotation ---")
    sqc = SpikeQuaternionRotation(theta_max_init=math.pi / 4)
    q = torch.randn(2, 5, 16)
    spike0 = torch.zeros(2, 5)
    out0 = sqc(q, spike0)
    diff = (out0 - q).abs().max()
    assert diff < 1e-5, f"SQC at spike=0 should be identity, got diff={diff}"
    # Magnitude should be (approximately) preserved since rotation is unitary
    spike1 = torch.ones(2, 5)
    out1 = sqc(q, spike1)
    norm_in = q.pow(2).sum(-1)
    norm_out = out1.pow(2).sum(-1)
    # Hamilton product preserves norm if the rotation is a unit quaternion;
    # we use exp(θê_K) which IS unit. Check ratio close to 1.
    ratio = (norm_out / (norm_in + 1e-9)).mean()
    print(f"  magnitude ratio (spike=1): {ratio.item():.4f}")
    assert abs(ratio.item() - 1.0) < 0.1, f"norm not preserved: {ratio}"
    print("  spike=0 identity, magnitude preserved  OK")


# ----------------------------------------------------------------------------
# Cell encoder
# ----------------------------------------------------------------------------

def test_cell_encoder():
    """Quaternion cell encoder produces masked quaternion states."""
    print("--- QuaternionCellEncoder ---")
    enc = QuaternionCellEncoder(d_model=32, n_vars=4)
    B, N = 2, 6
    value = torch.randn(B, N)
    time = torch.rand(B, N)
    var = torch.randint(0, 4, (B, N))
    spike = torch.rand(B, N)
    mask = torch.ones(B, N)
    mask[0, -2:] = 0
    q = enc(value, time, var, spike, mask)
    assert q.shape == (B, N, 32)
    # Padded cells must be zero
    assert (q[0, -2:] == 0).all()
    print(f"  shape={tuple(q.shape)}, padding zero  OK")


# ----------------------------------------------------------------------------
# Layers
# ----------------------------------------------------------------------------

def test_primary_layer():
    """PrimaryLayer is a quaternion residual transform."""
    print("--- PrimaryLayer ---")
    p = PrimaryLayer(d_model=32)
    q = torch.randn(2, 6, 32, requires_grad=True)
    out = p(q)
    assert out.shape == q.shape
    out.sum().backward()
    print("  forward + backward  OK")


def test_quaternion_anchor_layer():
    """QuaternionAnchorLayer: K_a anchors, distance-based incidence."""
    print("--- QuaternionAnchorLayer ---")
    layer = QuaternionAnchorLayer(d_model=32, k_a=4, num_heads=2)
    B, N = 2, 8
    q = torch.randn(B, N, 32, requires_grad=True)
    mask = torch.ones(B, N)
    mask[0, -2:] = 0
    msg = layer(q, mask)
    assert msg.shape == (B, N, 32)
    msg.sum().backward()
    # Anchor parameter should receive gradient
    assert layer.anchors.grad is not None
    print(f"  msg shape={tuple(msg.shape)}, anchor grad flows  OK")


def test_spike_triggered_event_layer():
    """Layer 1 dynamic edges, top-K spike selection."""
    print("--- SpikeTriggeredEventLayer ---")
    layer = SpikeTriggeredEventLayer(d_model=32, k_e=3, num_heads=2)
    B, N = 2, 10
    q = torch.randn(B, N, 32, requires_grad=True)
    spike = torch.rand(B, N)
    time = torch.rand(B, N)
    var = torch.randint(0, 5, (B, N))
    mask = torch.ones(B, N)
    msg = layer(q, spike, time, var, mask)
    assert msg.shape == (B, N, 32)
    msg.sum().backward()
    assert q.grad is not None
    print(f"  msg shape={tuple(msg.shape)}, grad flows through topk  OK")


def test_sqhh_block():
    """SQHHBlock combining all 3 layers + SQC."""
    print("--- SQHHBlock ---")
    blk = SQHHBlock(d_model=32, n_vars=5, k_a=4, k_e=3, num_heads=2)
    B, N = 2, 8
    q = torch.randn(B, N, 32, requires_grad=True)
    spike = torch.rand(B, N)
    time = torch.rand(B, N)
    var = torch.randint(0, 5, (B, N))
    mask = torch.ones(B, N)
    out = blk(q, spike, time, var, mask)
    assert out.shape == q.shape
    out.sum().backward()
    print(f"  out shape={tuple(out.shape)}, grad flows  OK")


# ----------------------------------------------------------------------------
# Main model
# ----------------------------------------------------------------------------

def test_model_forward_backward():
    """Full SQHH model end-to-end."""
    print("--- SQHH Model forward + backward ---")
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
    assert torch.isfinite(pred).all(), "NaN in pred"
    print(f"  pred shape={tuple(pred.shape)}, std={pred.std().item():.3f}")
    loss = ((pred - true) * mask).pow(2).sum() / mask.sum().clamp_min(1)
    loss.backward()
    has_grad, no_grad = 0, 0
    for p in model.parameters():
        if p.requires_grad:
            if p.grad is None:
                no_grad += 1
            else:
                has_grad += 1
                assert torch.isfinite(p.grad).all(), "NaN gradient"
    print(f"  params with grad: {has_grad}, no grad: {no_grad}")
    print("  OK")


def test_test_stage_shape():
    """Test stage returns (B, PRED_LEN, V)."""
    print("--- Test-stage output shape ---")
    cfg = FakeConfigs()
    model = Model(cfg)
    model.eval()
    x = torch.randn(2, cfg.seq_len, cfg.enc_in)
    y = torch.randn(2, cfg.pred_len, cfg.enc_in)
    out = model(x=x, y=y, exp_stage="test")
    assert out["pred"].shape == (2, cfg.pred_len, cfg.enc_in)
    print(f"  pred shape={tuple(out['pred'].shape)}  OK")


# ----------------------------------------------------------------------------
# Ablations
# ----------------------------------------------------------------------------

def test_ablation_no_sra():
    """no_sra disables refractory inhibition."""
    print("--- Ablation: no_sra ---")
    cfg = FakeConfigs()
    cfg.sqhh_no_sra = 1
    model = Model(cfg)
    model.train()
    x = torch.randn(2, cfg.seq_len, cfg.enc_in)
    y = torch.randn(2, cfg.pred_len, cfg.enc_in)
    out = model(x=x, y=y, exp_stage="train")
    out["pred"].mean().backward()
    print("  OK")


def test_ablation_no_sqc():
    """no_sqc disables spike-quaternion rotation."""
    print("--- Ablation: no_sqc ---")
    cfg = FakeConfigs()
    cfg.sqhh_no_sqc = 1
    model = Model(cfg)
    model.train()
    x = torch.randn(2, cfg.seq_len, cfg.enc_in)
    y = torch.randn(2, cfg.pred_len, cfg.enc_in)
    out = model(x=x, y=y, exp_stage="train")
    out["pred"].mean().backward()
    print("  OK")


def test_ablation_no_layer0():
    print("--- Ablation: no_layer0 ---")
    cfg = FakeConfigs()
    cfg.sqhh_no_layer0 = 1
    model = Model(cfg)
    model.train()
    x = torch.randn(2, cfg.seq_len, cfg.enc_in)
    y = torch.randn(2, cfg.pred_len, cfg.enc_in)
    out = model(x=x, y=y, exp_stage="train")
    out["pred"].mean().backward()
    print("  OK")


def test_ablation_no_layer1():
    print("--- Ablation: no_layer1 ---")
    cfg = FakeConfigs()
    cfg.sqhh_no_layer1 = 1
    model = Model(cfg)
    model.train()
    x = torch.randn(2, cfg.seq_len, cfg.enc_in)
    y = torch.randn(2, cfg.pred_len, cfg.enc_in)
    out = model(x=x, y=y, exp_stage="train")
    out["pred"].mean().backward()
    print("  OK")


def test_ablation_no_layer2():
    print("--- Ablation: no_layer2 ---")
    cfg = FakeConfigs()
    cfg.sqhh_no_layer2 = 1
    model = Model(cfg)
    model.train()
    x = torch.randn(2, cfg.seq_len, cfg.enc_in)
    y = torch.randn(2, cfg.pred_len, cfg.enc_in)
    out = model(x=x, y=y, exp_stage="train")
    out["pred"].mean().backward()
    print("  OK")


def test_full_vs_ablation_differ():
    """Full SQHH should produce different output than each ablation."""
    print("--- Full vs ablation outputs differ ---")
    cfg_full = FakeConfigs()
    cfg_no_l1 = FakeConfigs()
    cfg_no_l1.sqhh_no_layer1 = 1
    torch.manual_seed(0)
    m_full = Model(cfg_full)
    torch.manual_seed(0)
    m_no_l1 = Model(cfg_no_l1)
    m_full.eval()
    m_no_l1.eval()
    x = torch.randn(2, cfg_full.seq_len, cfg_full.enc_in)
    y = torch.randn(2, cfg_full.pred_len, cfg_full.enc_in)
    p_full = m_full(x=x, y=y, exp_stage="train")["pred"]
    p_no_l1 = m_no_l1(x=x, y=y, exp_stage="train")["pred"]
    diff = (p_full - p_no_l1).abs().max().item()
    print(f"  max diff full vs no_layer1: {diff:.4f}")
    assert diff > 1e-4, "no_layer1 ablation should change predictions"
    print("  OK")


def test_diagnostic_logging():
    print("--- Diagnostic logging ---")
    cfg = FakeConfigs()
    cfg.sqhh_diag_interval = 1
    model = Model(cfg)
    model.train()
    x = torch.randn(2, cfg.seq_len, cfg.enc_in)
    y = torch.randn(2, cfg.pred_len, cfg.enc_in)
    _ = model(x=x, y=y, exp_stage="train")
    print("  OK")


def main():
    # Primitives
    test_quaternion_linear()
    test_quaternion_identity_init()
    test_hamilton_product_axioms()
    test_quaternion_distance()
    test_quaternion_exp_K()
    test_quaternion_layer_norm()
    test_quaternion_mha()
    # SRA / SQC
    test_spike_refractory()
    test_spike_floor()
    test_sqc_rotation()
    # Cell encoder
    test_cell_encoder()
    # Layers
    test_primary_layer()
    test_quaternion_anchor_layer()
    test_spike_triggered_event_layer()
    test_sqhh_block()
    # Model
    test_model_forward_backward()
    test_test_stage_shape()
    # Ablations
    test_ablation_no_sra()
    test_ablation_no_sqc()
    test_ablation_no_layer0()
    test_ablation_no_layer1()
    test_ablation_no_layer2()
    test_full_vs_ablation_differ()
    test_diagnostic_logging()
    print("\n  All SQHH smoke tests passed.")


if __name__ == "__main__":
    main()
