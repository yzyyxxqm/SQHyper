# SQHH: Spike-Quaternion Heterogeneous Hypergraph

**Date**: 2026-05-10
**Status**: Design phase
**Lineage**: STHQ v1-v6 (failed soft-anchor) → v7 (SQHyper-derivative) → **SQHH (paradigm reset within hypergraph)**

---

## 1. Motivation & framing

### Why we are not iterating on STHQ v7

STHQ v7 inherits ~80% of its architecture from HyperIMTS via SQHyper:
binary L+V hyperedges, MultiHeadAttentionBlock, node_self_update,
IrregularityAwareAttention, and the `Linear(3D, 1)` decoder. Only
the spike encoder strength and a couple of training tweaks are
distinct. This is incremental, not novel — a reviewer can correctly
classify v7 as a SQHyper extension.

### Why "redesign the hypergraph" is the right ask

"Hypergraph" is a paradigm with several free design dimensions.
HyperIMTS / SQHyper occupy a single point in that space. A genuine
contribution must occupy a **different point**:

| Dimension | HyperIMTS occupies | SQHH explores |
|-----------|:-:|:-:|
| Layers | 1 (homogeneous) | 3 (heterogeneous) |
| Hyperedge semantics | time/variable groupings (prior) | data-driven, dynamic, geometric |
| Node typing | vanilla vector | quaternion native |
| Edge typing | vanilla vector | quaternion native |
| Incidence | hard 0/1 | quaternion-distance + spike-modulated |
| Edge creation | static (mask-determined) | dynamic (spike-triggered) |
| Attention score | dot product | Hamilton product |

HyperIMTS is recovered as a degenerate special case of SQHH (single
layer, hard incidence, dot product attention). SQHH is therefore
strictly more expressive.

### Why spike + quaternion as innovation pillars

The user's invariant: spike (脉冲) and quaternion (四元数) must be
the central innovation, with hypergraph as the underlying framework.
SQHH is structured so that spike and quaternion are not decorative —
they are responsible for the existence of two of the three layers
(spike → Layer 1; quaternion → Layer 2) and for the cross-layer
coupling (SQC rotation, Hamilton-product attention).

---

## 2. SQHH architecture overview

### Three heterogeneous hyperedge types

```
        Layer 2: Global Pattern Hyperedges (Quaternion Anchors)
        ┌──────────────────────────────────────────────────────┐
        │  K_a learnable quaternion anchors  Q_k ∈ ℍ^Q          │
        │  Incidence: I_2[k, n] = sigmoid(-||q_n ⊗ Q̄_k||² / σ_k²) │
        │  Captures: long-range macro patterns (rhythms, motifs)│
        └──────────────────────────────────────────────────────┘
                              ▲
                              │ Hamilton-product attention
                              ▼
        Layer 1: Local Event Hyperedges (Spike-Triggered)
        ┌──────────────────────────────────────────────────────┐
        │  Top-K_e high-spike cells trigger event edges         │
        │  Incidence: I_1[k, n] = window(t_k, Δt) ·             │
        │                          spike[n] · refractory(...)    │
        │  Captures: local bursts (cardiac events, motion bursts)│
        └──────────────────────────────────────────────────────┘
                              ▲
                              │ spike-driven activation
                              ▼
        Layer 0: Primary Hyperedges (Per-Observation Self-Loops)
        ┌──────────────────────────────────────────────────────┐
        │  Each cell n has its own self-loop edge e_n = {n}     │
        │  Incidence: I_0[n, m] = δ_{n,m}                        │
        │  Captures: raw observation precision (no info loss)   │
        └──────────────────────────────────────────────────────┘
                              ▲
                              │
                       Quaternion Cells q_n ∈ ℍ^Q
                       (R=value, I=time, J=variable, K=spike)
```

### Quaternion typing throughout

Every state vector in SQHH is quaternion-typed (∈ ℍ^Q ≡ ℝ^{4Q}):

- **Cell state** `q_n`: 4 channels carry value, time-phase,
  variable-id, spike intensity
- **Edge state** `h_e`: same 4-channel decomposition
- **Anchor state** `Q_k` (Layer 2): explicit quaternion anchor
- **Attention Q/K/V**: quaternion projections; scores via Hamilton
  product

No flatten-to-vanilla operation anywhere on the message passing
path. Decoder is permitted to use ordinary linear (downstream of
all hypergraph computation).

---

## 3. Mathematical specification

### 3.1 Cell encoding (quaternion-native)

For observation `(t_n, v_n, x_n)` with `v_n ∈ {1, ..., V}`:

```
spike[n]   = SpikeRefractoryEncoder(value, time, mask, var_id)  → scalar
q_n^R      = MLP_R(value_n, mask_n)                                 ∈ ℝ^Q
q_n^I      = sin(time_n · ω) ⊕ cos(time_n · ω)                      ∈ ℝ^Q
q_n^J      = VarEmbed(v_n)                                          ∈ ℝ^Q
q_n^K      = MLP_K(spike[n])                                        ∈ ℝ^Q
q_n        = [q_n^R, q_n^I, q_n^J, q_n^K] ∈ ℍ^Q                     (concat)
```

Initial mixing via `QuaternionLinear`:

```
q_n^(0) = QLin_init(q_n) · mask_n
```

### 3.2 Spike-Refractory Activation (SRA)

SNN-inspired: each cell's effective spike is reduced by recent
spikes from the same variable.

```
spike_raw[n] = sigmoid(SpikeMLP(value, time, mask, var_emb))

# Refractory inhibition: same-variable, earlier-time, decaying kernel
refractory[n] = Σ_{m: var_m = var_n, t_m < t_n}
                 spike_eff[m] · exp(−(t_n − t_m) / τ_r)

spike_eff[n] = max(floor, spike_raw[n] · (1 − tanh(α · refractory[n])))
```

- `τ_r` and `α` are learnable per-variable
- `floor` from `--sthq_spike_floor` (carryover)
- Implementation: causal mask on a per-variable cumulative pass

### 3.3 Spike-Quaternion Coupling (SQC) rotation

Before a cell participates in higher-layer message passing, its
quaternion state is rotated based on its spike intensity. High-spike
cells are rotated towards the K-axis ("event subspace").

```
ê_K     = unit quaternion (0, 0, 0, 1) ∈ ℍ
θ_n     = θ_max · spike_eff[n]
rot_n   = exp(θ_n · ê_K) = cos(θ_n) + sin(θ_n) · ê_K
q_n_rot = rot_n ⊗ q_n
```

`exp(θ · ê_K)` for a unit quaternion is computed in closed form via
`cos(θ) · 1 + sin(θ) · ê_K`. `θ_max` is a learnable scalar in
`[0, π/2]`.

### 3.4 Layer 0 — Primary Edges (per-observation self-loop)

Trivial structure but quaternion-typed:

```
msg_0[n] = QLin_0(q_n_rot)
```

This is essentially a per-cell quaternion residual. Its purpose is
to **guarantee that no observation's information is lost** —
addressing the v6 failure mode where K_t < L caused per-timestep
collapse.

### 3.5 Layer 1 — Spike-Triggered Event Edges (dynamic)

Each forward pass:

1. **Triggering**: select top-K_e cells per batch by `spike_eff`
   (excluding query cells where `y_mask = 1`)
2. **Edge state init**: `h_k^(L1) = QuaternionLinear(q_{n_k}_rot)` for
   trigger cell `n_k`
3. **Incidence (asymmetric: aggregation only)**:
   ```
   I_1_aggr[k, n] = 1[t_n ∈ window(t_k, Δt)]
                   · 1[var_n same family as var_k or *]
                   · spike_eff[n]
                   · refractory_decay(t_n − t_k)
   ```
4. **Aggregation** (Hamilton-product weighted sum):
   ```
   h_k^(L1)_new = h_k^(L1) + Σ_n I_1_aggr[k, n] · (q_n_rot ⊗ Anchor_proj_k)
   ```
5. **Distribution back to cells** (spread over a wider window):
   ```
   I_1_dist[k, n] = 1[t_n ∈ window(t_k, Δt_distribute)] · refractory_decay(...)
   msg_1[n] = QLin_L1( Σ_k I_1_dist[k, n] · h_k^(L1)_new )
   ```

Window sizes `Δt`, decay `τ_r`, K_e are hyperparameters.

### 3.6 Layer 2 — Quaternion Anchor Edges

K_a learnable quaternion anchors `Q_k ∈ ℍ^Q`, init via
`xavier(Q.r) + small_normal(Q.{i,j,k})`.

1. **Quaternion incidence**:
   ```
   d_kn = ||q_n_rot ⊗ conj(Q_k)||²    # quaternion distance
   I_2[k, n] = exp(−d_kn / σ_k²) · mask[n]
   σ_k learnable
   ```
2. **Aggregation**:
   ```
   h_k^(L2) = (Σ_n I_2[k, n] · q_n_rot) / (Σ_n I_2[k, n] + ε)
   h_k^(L2) = QLin_L2_aggr(h_k^(L2))
   ```
3. **Hamilton-product attention back to cells**:
   ```
   Q_n = QLin_q(q_n_rot)
   K_k = QLin_k(h_k^(L2))
   V_k = QLin_v(h_k^(L2))
   score[n, k] = Re(Q_n ⊗ conj(K_k))    # real part of Hamilton product
   α[n, k] = softmax_k(score[n, k] / sqrt(Q))
   msg_2[n] = QLin_o( Σ_k α[n, k] · V_k )
   ```

### 3.7 Layer fusion

Per-layer learnable mixing weights:

```
ω_l = softplus(parameter_l)  # ω_l > 0
msg[n] = ω_0 · msg_0[n] + ω_1 · msg_1[n] + ω_2 · msg_2[n]

q_n^(next) = QuaternionLayerNorm( q_n + msg[n] ) · mask[n]
```

### 3.8 Decoder

After `n_layers` of SQHH, the final cell states are decoded by:

```
pred[n] = Linear_decode( [q_n_final, q_n_initial, time_emb(t_n), var_emb(v_n)] )
```

Same time-aware decoder as STHQ v6 (this part can be vanilla).

---

## 4. Why each layer is necessary

| Failure mode without this layer | Layer that prevents it |
|---|:-:|
| Per-observation precision loss (v6 failure) | Layer 0 |
| Local burst patterns averaged out by global anchors | Layer 1 |
| No long-range / cross-variable global structure | Layer 2 |

### Quantitative coverage analysis (HA, L=3003, V=12)

| Scheme | # edges | Covers per-timestep? | Covers global? | Covers events? |
|--------|:-:|:-:|:-:|:-:|
| HyperIMTS | L + V = 3015 | ✅ (L hard) | ❌ (V hard, no cross-time) | ❌ |
| STHQ v6 | K_t + K_v = 384 + 12 | ❌ (K_t < L) | ✅ (multi-scale ω) | ❌ |
| SQHyper | L + V = 3015 | ✅ | ❌ | partial (SGI) |
| **SQHH** | N (Layer 0) + K_e + K_a | ✅ (N self-loops) | ✅ (K_a anchors) | ✅ (K_e event edges) |

SQHH is the only design that covers all three.

---

## 5. Ablation matrix

We design 7 ablations explicitly:

| Ablation | Layer 0 | Layer 1 | Layer 2 | SRA | SQC | QHA |
|----------|:-:|:-:|:-:|:-:|:-:|:-:|
| **Full SQHH** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| no_layer0 | – | ✓ | ✓ | ✓ | ✓ | ✓ |
| no_layer1 (spike-triggered off) | ✓ | – | ✓ | ✓ | ✓ | ✓ |
| no_layer2 (quat anchor off) | ✓ | ✓ | – | ✓ | ✓ | ✓ |
| no_sra (refractory off) | ✓ | ✓ | ✓ | – | ✓ | ✓ |
| no_sqc (rotation off) | ✓ | ✓ | ✓ | ✓ | – | ✓ |
| no_qha (dot-product attn) | ✓ | ✓ | ✓ | ✓ | ✓ | – |
| HyperIMTS-equivalent | – | – | only V | – | – | – |

Each ablation answers a specific question:
- `no_layer0`: does primary precision matter?
- `no_layer1`: does dynamic event triggering help?
- `no_layer2`: does quaternion anchor routing help?
- `no_sra`: does refractory inhibition help?
- `no_sqc`: does spike-quaternion rotation help?
- `no_qha`: does Hamilton-product attention beat dot-product?

---

## 6. Hyperparameters

| Symbol | Meaning | Default |
|--------|---------|---------|
| K_a | # quaternion anchors (Layer 2) | 16 |
| K_e | # spike-triggered events per batch (Layer 1) | min(32, N/16) |
| Δt | Layer 1 aggregation window (relative) | 0.05 |
| Δt_dist | Layer 1 distribution window (relative) | 0.10 |
| τ_r | refractory time constant (per-var, learnable) | init 0.1 |
| α (SRA) | refractory inhibition strength (learnable) | init 1.0 |
| θ_max | SQC max rotation angle (learnable) | init π/4 |
| floor | spike floor (carryover) | 0.1-0.2 |
| ω_l init | layer mixing weights (learnable) | softplus(0.5) |

---

## 7. Implementation plan

### File structure
```
models/SQHH.py               # main model
tests/sqhh_smoke.py          # CPU smoke
scripts/SQHH/                # 4 dataset scripts
  ├── USHCN.sh
  ├── HumanActivity.sh
  ├── P12.sh
  └── MIMIC_III.sh
```

### Build order (each step gated by smoke test)

1. **Quaternion primitives**: `QuaternionLinear`, `hamilton_product`,
   `quaternion_conjugate`, `quaternion_distance`, `quaternion_exp`,
   `QuaternionLayerNorm`, `QuaternionMHA` (Hamilton-product attention)
   → smoke: shape correctness, gradient flow

2. **SpikeRefractoryEncoder**: `SRA` causal pass with per-variable
   exponential decay
   → smoke: floor enforced, refractory reduces same-var subsequent spikes

3. **SQC rotation module**: `q_rotated = exp(θ · ê_K) ⊗ q`
   → smoke: identity at θ=0, magnitude preservation

4. **Layer 0 (Primary)**: trivial QuaternionLinear self-update
   → smoke: forward + backward

5. **Layer 2 (Quaternion Anchor)**: `compute_quat_distance`,
   `aggregate`, Hamilton-product attention back
   → smoke: aggregation reasonable, K_a=16 forward + backward

6. **Layer 1 (Spike-Triggered)**: top-K_e selection, window incidence,
   refractory weighting, aggregation, distribution
   → smoke: top-K differentiable through gather, edges respect window

7. **Main SQHH model**: combine layers, decoder, forward
   → smoke: end-to-end fwd/bwd, NaN check, ablation flags

### Risk management

- Build Layer 0 + Layer 2 first (simpler), validate they ≥ SQHyper
  on USHCN/HA before adding Layer 1
- Layer 1 is the most complex (dynamic edges), keep it isolated
- Provide an ablation flag for each component so we can debug
  individually

---

## 8. Existing literature differentiation

### Multi-layer / heterogeneous hypergraph
- "Hypergraph Convolutional Networks" (HGCN, 2019): single-layer
- "Hypergraph Attention Networks" (HGAT, 2020): single-layer
- "DHGNN" (Dynamic HGNN, 2019): dynamic but no quaternion / SNN
- **SQHH novelty**: 3-layer heterogeneous structure with explicit
  semantic separation (precision / event / global), spike-driven
  edge creation, quaternion anchors

### Quaternion neural networks
- Quaternion CNN, Quaternion RNN: vanilla quaternion replacing real
- "Quaternion Knowledge Graph Embedding" (QuatE, 2019): static graph
- **SQHH novelty**: first quaternion-typed hypergraph for time series;
  Hamilton-product attention; spike-modulated quaternion rotation

### SNN for time series
- "Spiking Neural Networks for IMTS" (existing work uses SNN as
  encoder, then standard NN downstream)
- **SQHH novelty**: SRA refractory mechanism integrated into hypergraph
  message passing, not just encoder

### IMTS forecasting baselines
- HyperIMTS (ICML 2025): single-layer hypergraph
- GraFITi (AAAI 2024): graph attention
- ContiFormer (NeurIPS 2023): continuous-time attention
- **SQHH** is fundamentally different: only model with multi-layer
  heterogeneous quaternion hypergraph + SNN refractory

### Literature audit (2026-05-10)

ArXiv search confirms novelty of SQHH's specific combination:

| Concept | Closest prior | Distance from SQHH |
|---------|---------------|---------------------|
| Quaternion + time series | 2403.11722 "Quaternion Time Series Compression" (uses quaternion to compress 4 stats per chunk) | Different problem (compression, not forecasting); no hypergraph; no spike |
| Hypergraph + IMTS | 2505.17431 (HyperIMTS) | Single-layer; hard time/var hyperedges; no quaternion; no spike refractory |
| Refractory in SNN | 2507.02960 "Historical Dynamics of Refractory Periods" (Jun 2025); 2509.17769 "Spike-Triggered Threshold Dynamics for Refractory" (Sep 2025) | Both apply refractory **inside an SNN**; SQHH applies refractory inhibition to **hypergraph hyperedge membership weights** in IMTS forecasting — a different domain |
| Heterogeneous (multi-layer) hypergraph + IMTS | none found | SQHH is the first |
| Hamilton-product attention + hypergraph | none found | SQHH is the first |

**Defensible claims** for the paper:

1. *First multi-layer heterogeneous hypergraph for IMTS forecasting*
   (Layer 0 / Layer 1 / Layer 2 with distinct semantics per layer).
2. *First quaternion-typed hypergraph* where cell, hyperedge, and
   attention all preserve Hamilton structure end-to-end.
3. *First application of SNN refractory inhibition to hypergraph
   membership weights* — concurrent SNN-internal work (Tao 2025;
   Li 2025) shows the mechanism is well-motivated and current.
4. *First spike-triggered dynamic hyperedge creation* in IMTS:
   high-spike cells dynamically instantiate event hyperedges per
   batch, replacing static prior incidence.
5. *First Spike-Quaternion Coupling*: spike intensity drives a
   quaternion rotation that brings salient cells into an "event
   subspace" before downstream message passing.

---

## 9. Empirical validation roadmap

### Milestone tests (each gated by ≥ outcome)

| Milestone | Target | Expected duration |
|-----------|--------|:-:|
| M1: Layer 0 + Layer 2 only | ≥ SQHyper on USHCN/HA | Day 7 |
| M2: + Layer 1 | ≥ M1 on all 4 datasets | Day 12 |
| M3: + SRA + SQC + QHA | best results, ablations | Day 17 |
| M4: paper draft | submission-ready | Day 20+ |

### Failure conditions / pivots

If M1 underperforms SQHyper:
→ Layer 2 quaternion anchors are not adding value; pivot to using
   L hard hyperedges as Layer 2 (HyperIMTS-flavored anchor) and
   redirect novelty to Layer 1 + SQC + SRA.

If M2 ≤ M1:
→ Layer 1 spike-triggered edges are not helping; consider keeping
   Layer 1 as ablation rather than default.

If M3 ≤ M2:
→ SRA/SQC/QHA are not contributing; one or two of them is the
   actual bottleneck. Use ablation matrix to identify and drop.

---

## 10. Today's deliverables (no GPU, code-only)

1. ✅ This design document
2. Quaternion primitives + smoke (next)
3. SpikeRefractoryEncoder + smoke
4. Layer 0 + Layer 2 + smoke
5. (defer Layer 1 to next session)

When server returns, M1 validation can begin immediately.
