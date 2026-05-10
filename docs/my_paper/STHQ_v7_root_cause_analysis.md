# STHQ v7 — Root-Cause Analysis & Architectural Reset

**Date**: 2026-05-10
**Trigger**: User shut down the server and requested deep analysis. v6
results showed STHQ losing badly to the in-house SQHyper baseline on
3 of 4 datasets. We needed to know **why**, not iterate again.
**Outcome**: Root cause identified, full architectural reset
implemented as STHQ v7.

---

## 1. The numbers that forced this analysis

| Dataset | **SQHyper** (in-house) | STHQ v6 | HyperIMTS (paper) | STHQ regression |
|---------|:-:|:-:|:-:|:-:|
| USHCN | 0.1910 | **0.1721** | 0.1738 | none (slight win) |
| **HumanActivity** | **0.01733** | 0.0920 | 0.0421 | **5.3× worse** |
| **P12** | **0.3009** | 0.3864 | 0.2996 | **+28%** |
| **MIMIC** | **0.4198** | 0.5481 | 0.4259 | **+30%** |

`SQHyper` is the user's prior model that already **beats HyperIMTS by
~58% on HA and ~1% on MIMIC**. STHQ v1–v6 erased every gain on those
datasets. v6 only "won" on USHCN, and even there only by 1% over the
HyperIMTS paper number.

A model that loses to a baseline you've already built is not a
contribution; it's a regression.

---

## 2. Side-by-side architectural comparison

After reading `models/HyperIMTS.py`, `models/SQHyper.py`, and the v6
`models/STHQ.py` carefully, the following dimension table tells the
whole story:

| Dimension | HyperIMTS | **SQHyper** (HA=0.017) | **STHQ v6** (HA=0.092) |
|-----------|:-:|:-:|:-:|
| Temporal hyperedge count | **L** (one per timestep) | **L** | K_t = 96–384 |
| HA temporal hyperedges | 3003 | 3003 | **384** (8× compression) |
| Temporal incidence | hard 0/1 binary | hard 0/1 | Gaussian soft kernel |
| Variable hyperedge count | enc_in (5–96) | enc_in | K_v = 12–32 |
| Variable incidence | hard 0/1 | hard 0/1 | softmax over learned affinity |
| **Cell-to-cell self-attention** | ✅ `node_self_update` (3D MHA) | ✅ | ❌ none |
| n2h K/V gating | mask-only | mask-only + SGI gate | mask only |
| h2n fusion | `Linear(3D, 1)` | **QMF (Quaternion)** | quaternion + linear hybrid |
| h2h interaction | last-layer IrregularityAwareAttention | same | hyperedge self-attn |

Two issues stand out:

### 2.1 STHQ replaced **hard per-timestep** with **soft K_t anchors**

In HyperIMTS / SQHyper, every actual timestep is its own temporal
hyperedge. Two cells observed at the same timestep are **guaranteed**
to share a hyperedge. The incidence matrix is binary 0/1 derived
directly from `time_indices_flattened == t`.

In STHQ v6, the temporal hyperedges were `K_t` learnable Gaussian
anchors. Membership was `exp(-((t - τ_k)/ω_k)^2)` — a continuous
approximation. For HA where `L = 3003`, `K_t = 384`, **each anchor
must "represent" ~8 timesteps**, and the precise per-timestep grouping
is destroyed.

Per-timestep coupling is **the inductive bias that matters most for
irregular sparse data**. Throwing it away to introduce learnable
"multi-scale anchors" was a regression in disguise.

### 2.2 STHQ removed cell-to-cell attention

HyperIMTS / SQHyper both have `node_self_update`: a 3D-key
multi-head attention that lets every cell directly attend to every
other cell, masked by `x_y_mask × x_y_mask^T`. This is O(N²) but is
the **only path** by which information can flow between two
arbitrary cells without going through the hyperedge bottleneck.

STHQ v6 had no equivalent. Every cell-to-cell interaction had to be
mediated by `K_t + K_v + K_e ≈ 396` hyperedge slots per layer. On
HA with N≈12,000 cells, that's a 30× compression of an already
overconstrained channel.

### 2.3 Why USHCN looked OK

| Dataset | L (seq+pred) | K_t (max layer) | anchors / timestep |
|---------|:-:|:-:|:-:|
| USHCN | 153 | 96 | 0.63 |
| **HA** | 3003 | 384 | **0.13** |
| P12 | 39 | 96 | 2.5 |
| MIMIC | 75 | 96 | 1.3 |

USHCN's seq_len happens to be small enough that `K_t` is on the order
of `L`, so the soft anchors approximate per-timestep grouping
adequately. HA breaks this hard.

---

## 3. Why α / β / γ were stuck

Three adaptive mechanisms in v6 all failed to learn:

### α (Hamilton vs Linear blend) ≈ 0.50 everywhere

```python
msg_h = msg_proj_h(hamilton_product(h, q))   # Hamilton path
msg_l = msg_proj_l(h)                          # Linear path
msg = α * msg_h + (1-α) * msg_l
```

After `LayerNorm`, `q` has bounded norm. The Hamilton product `h ⊗ q`
is a structured rotation/scaling of `h`. Both `msg_proj_*` are
unconstrained `Linear`. **The two paths span the same output
subspace**, so the optimizer has no incentive to break the symmetry.
α stays at sigmoid(0)=0.5.

### β (STEA mixing) ≈ 0.27 (= sigmoid(-1), the init)

STEA's K_e dynamic anchors only "see" a tiny fraction of cells per
batch (HA: K_e=32 of N=12,000 → 0.27%). The fraction of output that
actually depends on β is too small for the optimizer to grow β.

### γ (spike-modulated bandwidth) ≈ 1 (init)

γ requires variance in `spike_intensity` across cells to learn a
useful modulation. On USHCN/HA, spike std is 0.10–0.20 — too narrow
to drive specialization.

**Common pattern**: each adaptive mechanism is added downstream of
**fixed structural commitments that don't reward adaptation**. The
fix isn't more adaptive parameters; it's correcting the
structure.

---

## 4. v7 architectural reset

After confirming the user's preference (option A), STHQ was rebuilt
on top of the **SQHyper backbone** with STHQ's distinctive spike
encoder layered on top:

```
STHQ v7 architecture
├── HypergraphEncoder      (identical to HyperIMTS / SQHyper)
│   ├── L hard temporal hyperedges
│   ├── enc_in hard variable hyperedges
│   └── binary incidence matrices
└── HypergraphLearner (per layer):
    1. SpikeEncoder ──────────────► (g_n, e_n)            [STHQ-distinct]
       • multi-feature: (value, time_norm, mask, var_emb)
       • spike floor: g_n ∈ [floor, 1] on observed cells
    2. node2temporal_hyperedge      [SQHyper, K/V gated by g_n]
    3. node2variable_hyperedge      [SQHyper, K/V gated by g_n]
    4. node_self_update             [SQHyper: cell-to-cell 3D MHA]
    5. QMF Hamilton fusion          [SQHyper, but q_K = e_n from STHQ]
       q = [proj_R(self), proj_I(temp), proj_J(var), event_e_n]
       h2n = QuaternionLinear(q)
    6. (last layer only) h2h IrregularityAwareAttention   [SQHyper]
```

### What's STHQ-distinct

1. **Stronger spike encoder**: SQHyper's SGI took only `(obs,
   deviation)` as input. STHQ v7 takes `(value, time_norm, mask,
   var_emb)`, capturing temporal phase + variable identity intrinsically
   without bootstrapping from variable-context (which is unreliable at
   layer 0 before any message passing has happened).

2. **Spike floor**: every observed cell contributes ≥ `floor` mass to
   K/V gating. This was the v5 fix that solved spike starvation
   (P12: 3% → 29%; MIMIC: 12% → 45%). Critical for sparse medical
   data and worth carrying over into v7.

3. **Event-layer warmup**: optional `event_layer_warmup=N` zeros
   `e_n` for the first N layers, letting QMF first learn temporal/variable
   fusion before integrating event signal. Helps when the spike
   encoder hasn't yet calibrated.

4. **Per-layer learnable `gate_scale`**: residual blend
   `gating = mask + gate_scale * (g_n - mask)`. Init 0 means the
   model starts equivalent to HyperIMTS K/V (no SGI noise on smooth
   data) and learns to turn on gating where it helps. Inherited from
   SQHyper R2; kept here because it works.

### What was removed (and why)

| Removed v6 mechanism | Why |
|---|---|
| `K_t` soft anchors | replaced by L hard hyperedges |
| `K_v` soft variable codes | replaced by enc_in hard hyperedges |
| `K_e` STEA event anchors | redundant with L hyperedges; β never learned |
| Hamilton/Linear hybrid (α) | both paths span same space; α stuck at 0.5 |
| Spike-modulated bandwidth (γ) | no kernel left to modulate |
| τ-repulsion / var-entropy aux losses | no anchors left to regularize |

### Ablation flags

- `--sthq_no_spike 1`: disables SpikeEncoder entirely (mask-only K/V,
  zero event features). Tests whether the new encoder is actually
  contributing.
- `--sthq_no_quaternion 1`: replaces QMF with `Linear(D, D)`. Tests
  whether quaternion structure is contributing.

---

## 5. v7 code & test status

### Files changed

- `models/STHQ.py` — fully rewritten. ~570 lines, down from ~750.
- `models/STHQ_v6_archive.py.bak` — archived v6 for diff reference.
- `tests/sthq_smoke.py` — fully rewritten. **11 tests, all pass.**
- `utils/configs.py` — replaced 13 STHQ flags with 5 (cleaner CLI).
- `utils/ExpConfigs.py` — same field reduction.
- `scripts/STHQ/{USHCN,HumanActivity,P12,MIMIC_III}.sh` —
  rewritten to mirror SQHyper budgets (`itr=5`, `train_epochs=300`,
  `patience=10`) plus STHQ-specific spike floors.

### Smoke test coverage

```
test_quaternion_linear        ✅ block-Hamilton form correct
test_quaternion_identity_init ✅ identity init produces y = x
test_spike_encoder            ✅ floor enforced, padding zero, event head 0-init
test_model_forward_backward   ✅ no NaN, 103 params get gradient
test_test_stage_returns_padded_shape ✅ unpad + reshape correct
test_no_spike_ablation        ✅ ablation runs cleanly
test_no_quaternion_ablation   ✅ QMF != linear (max diff 0.17)
test_spike_floor              ✅ floor=0 vs 0.3 changes prediction
test_event_warmup             ✅ N=2 warmup runs + gradients flow
test_n_layers_1               ✅ USHCN single-layer config works
test_diagnostic_logging       ✅ logger emits without crash
```

### Smoke diagnostic at init

```
[STHQ diag step=1] L0: g=0.881 |e|=0.000 gs=+0.000 |
                   L1: g=0.881 |e|=0.000 gs=+0.000
```

- `g=0.88` ≈ sigmoid(2): gate_head bias init 2.0 means observed cells
  start with strong K/V participation (close to mask-only).
- `|e|=0`: event head zero-init means QMF starts identical to
  3-component fusion (event K-channel is dormant).
- `gs=0`: gate_scale init 0 means K/V gating starts identical to
  HyperIMTS (no SGI noise injected).

The model effectively starts as a **HyperIMTS-equivalent**, then
learns to bring in gating and event signal where they help. This is
the conservative initialization that SQHyper's R2 fix proved
important.

---

## 6. Configuration choices per dataset

| Dataset | seq_len | n_layers | d_model | spike_floor | event_warmup | rationale |
|---------|:-:|:-:|:-:|:-:|:-:|:-|
| USHCN | 150 | 1 | 256 | 0.1 | 0 | smooth climate; light spike floor |
| HA | 3000 | 3 | 128 | 0.1 | 0 | event-rich motion; let spike speak |
| P12 | 36 | 2 | 256 | 0.2 | 0 | sparse medical; needs strong floor |
| MIMIC | 72 | 2 | 256 | 0.2 | 1 | sparse + heterogeneous; warmup for QMF |

These mirror the SQHyper budgets exactly, with the addition of spike
floor (and one warmup layer for MIMIC). The user can run them
unchanged when the server is back up:

```bash
bash scripts/STHQ/USHCN.sh
bash scripts/STHQ/HumanActivity.sh
bash scripts/STHQ/P12.sh
bash scripts/STHQ/MIMIC_III.sh
```

---

## 7. Expected outcome

**Hypothesis**: v7 should at minimum match SQHyper, with potential
improvement from the stronger spike encoder + floor:

| Dataset | SQHyper | **v7 expected** |
|---------|:-:|:-:|
| USHCN | 0.1910 | 0.18-0.19 (close to or slightly better) |
| HA | 0.01733 | 0.017-0.020 (within noise of SQHyper) |
| P12 | 0.3009 | 0.298-0.302 (floor=0.2 may help marginally) |
| MIMIC | 0.4198 | 0.40-0.42 (warmup may help) |

If v7 underperforms SQHyper, the residual gap would tell us whether
the stronger spike encoder is a net positive or net noise — either
result is informative, and ablations (`--sthq_no_spike`) will isolate.

If v7 matches but doesn't beat SQHyper, the right next step is to
**either** scale up the spike-encoder contribution further (deeper MLP,
multi-scale temporal phase features) **or** introduce a genuinely
novel structural component that doesn't replicate SQHyper.

---

## 8. Lessons (for future iterations)

1. **Honor the inductive bias of the data**. IMTS data has a
   discrete, set-valued structure: each observation belongs to
   exactly one (timestep, variable) pair. Replacing this with soft
   continuous approximations (Gaussian kernels, learned anchors)
   throws away the strongest signal.

2. **Don't add adaptive parameters around dead structure**. v6 had
   α, β, γ — all learnable, all stuck at their inits. Each was
   gated downstream of a structural choice the loss couldn't
   penalize. Fix the structure first, then ask whether adaptation
   helps.

3. **Cell-to-cell attention is not optional**. Both HyperIMTS and
   SQHyper have it; STHQ v1–v6 didn't. The hyperedge bottleneck
   alone is not sufficient to route information for sparse irregular
   data — long-tail interactions need a direct path.

4. **Compare to the right baseline**. STHQ kept comparing itself to
   HyperIMTS paper numbers, but the in-house SQHyper baseline was
   already substantially better. Comparing only to the public number
   masked the regression.

5. **Smoke tests should cover ablations**. v6's smoke test verified
   the model runs; it didn't verify any STHQ-distinct mechanism
   actually changed predictions in the expected direction. v7 adds
   tests that explicitly compare with/without spike, with/without
   quaternion, with different floor values.

---

## 9. Status

- ✅ Root cause identified and documented (this file).
- ✅ v7 model implemented (`models/STHQ.py`, ~570 lines).
- ✅ Smoke tests rewritten and passing (11/11).
- ✅ Scripts mirror SQHyper budgets, add STHQ flags.
- ⏸  Server training **paused** by the user — code is ready to
    deploy when the GPU is back online.
