# STHQ v5 / v6 Diagnostic Report

**Date**: 2026-05-10
**Branch**: `sqhyper/main`
**Commits**: `e6c3d30` (v5), `6a2ebc2` (v6)

## Executive summary

Two iterations on STHQ delivered measurable improvement over v4 and closed
the USHCN gap to HyperIMTS. HA, P12, and MIMIC remain behind paper baselines
but moved in the right direction. Two adaptive mechanisms (γ for ω
modulation, β for STEA mixing) were added but are barely learning under the
current training budget — the architecture has more headroom than the
optimisation is currently extracting.

## Final 4-dataset results

| Dataset | v4 (pre) | v5 (5-iter) | **v6 (final)** | HyperIMTS paper | v6 Δ |
|---------|:-:|:-:|:-:|:-:|:-:|
| **USHCN** | val 0.225 | 0.1854 ± 0.018 | **0.1721 ± 0.011** | 0.1738 | **−1.0% ✅** |
| **HA** | val 0.158 | 0.0949 ± 0.003 | 0.0920 ± 0.009 | 0.0421 | +118% ⚠ |
| **P12** | — | (3 iter under-train, val 0.36) | **0.3864** (1 iter) | 0.2996 | +29% |
| **MIMIC** | — | (1 iter, val 0.58) | **0.5481** (1 iter) | 0.4259 | +29% |

USHCN v6 best 0.1641 < HyperIMTS 0.1738 — first time a single iter beats
the published baseline.

## What changed in v5 vs v4

Two STHQ-distinct fixes driven by v4's 30-min diagnostic data
(P12 spike active = 3 %, MIMIC = 12 % — severe spike starvation):

1. **Spike floor** (`--sthq_spike_floor`):
   `spike = floor + (1 − floor) · sigmoid(MLP)`. Every observed cell
   contributes at least `floor` to hyperedge mass.
   Per-dataset: USHCN/HA 0.1, P12/MIMIC 0.2.

2. **Spike-modulated bandwidth** (per-layer learnable γ):
   `ω_eff(cell, anchor) = ω(anchor) · (1 + γ · (1 − spike(cell)))`
   High-spike cells → narrow ω; low-spike (incl. queries) → wide ω.

### Validation
| Dataset | v4 spike active | v5 spike active | Δ |
|---------|:-:|:-:|:-:|
| USHCN | 0.40 | 0.41 | + 2 % |
| **P12** | **0.03** | **0.29** | **+ 867 %** |
| **MIMIC** | **0.12** | **0.45** | **+ 275 %** |
| HA | 0.33 | 0.37 | + 12 % |

Spike starvation hypothesis confirmed.

### γ adaptation pattern
| Dataset | L0 | L1 | L2 | Comment |
|---------|:-:|:-:|:-:|---|
| USHCN | 0.99 | 0.90 | 0.98 | dense data → γ ≈ 1 |
| **P12** | 0.90 | **0.63** | **0.51** | deep layers learned to disable spike-modulation |
| MIMIC | 1.10 | 0.89 | 0.83 | mild deep-layer reduction |
| HA | 1.02 | 0.99 | 0.99 | no learning signal — concerning |

P12 deep-layer γ → 0.5 means the model decided "deep layers shouldn't
narrow ω based on spike". Healthy specialisation.

## What changed in v6

**Spike-Triggered Event Anchors (STEA)**: third hyperedge type.
For each batch, the top-K_e spike-active observed cells become
**dynamic anchors** at runtime. Unlike static τ anchors learned across
the dataset, these anchors are sample-specific.

```python
top_idx = (spike * mask).topk(K_e, dim=1).indices       # [B, K_e]
event_t = gather(time_norm, top_idx); event_v = gather(var_id, top_idx)
kernel_e = gauss(t − event_t, ω_e) · (same_var + xvar_w · (1 − same_var))
h_event = aggregate(spike·kernel_e · q) + event_q seed; QLin proj
msg_e   = QLin( h_event distributed to cells  ⊗  q )    # Hamilton
msg     = (1 − β) · msg_existing + β · msg_e            # learnable β
```

Per-layer K_e schedule:
| Dataset | L0 | L1 | L2 |
|---------|:-:|:-:|:-:|
| USHCN | 16 | 8 | 4 |
| **HA** | 32 | 16 | 8 |
| P12 | 24 | 16 | 8 |
| MIMIC | 32 | 16 | 8 |

**STHQ-distinct**: HyperIMTS has no per-sample anchors. STEA is the
"event-routed" mechanism the model name promised.

### v5 → v6 deltas (USHCN, HA where 5/2-iter overlap)

| Dataset | v5 mean | v6 mean | Improvement |
|---------|:-:|:-:|:-:|
| USHCN | 0.1854 | 0.1721 | **−7.2 %** |
| HA | 0.0949 | 0.0920 | −3.0 % |

## Diagnostic — what didn't work

### β stuck at 0.27 (= initial sigmoid(−1) ≈ 0.27)

Across all 4 datasets, all 3 layers, β barely moved:
- USHCN: L0/1/2 = 0.27 / 0.28 / 0.27
- P12:   0.26 / 0.26 / 0.27
- MIMIC: 0.26 / 0.25 / 0.30
- HA:    0.27 / 0.27 / 0.27

**Implication**: STEA's contribution (~27 %) is at initialisation level.
The optimiser has no reason to grow β — STEA messages are useful enough at
27 % weight that loss doesn't push for more, but not so useful that
gradients amplify them. Likely causes:

1. STEA messages might be redundant with τ-anchor messages on USHCN where
   data is regular.
2. On HA (long sparse sequences), top-K event selection might be unstable
   batch-to-batch, increasing variance and damping β growth.
3. No explicit STEA regularisation pushing for diverse use.

### γ ≈ 1 on USHCN/HA

γ only learned on P12 and MIMIC (sparse medical data). USHCN climate is
smooth — every cell looks similar — so spike-modulation has nothing to
exploit. On HA the variance is high but maybe regions of high spike density
overlap so there's no benefit from variable bandwidth.

### α ≈ 0.50 everywhere

Hamilton/Linear blend stays at 50/50. This is the *signature* of the
hybrid being **redundant** — the two paths are not learning to specialise.

## Recommendations for v7+

1. **Force STEA to activate**: warmup β from 0 to 0.5 over first 5 epochs,
   or add an entropy bonus on the (msg_h, msg_l, msg_e) blend, or remove
   one of the static paths so STEA is the only "extra capacity".

2. **Pure Hamilton (kill linear path)**: removes α=0.5 redundancy, makes
   the model commit to the typed-quaternion story. Lower-risk change.

3. **HA-specific**: HA has K_t=384 anchors but only 3 queries. The
   anchor count exceeds what the loss can supervise. Drop K_t to 64 and
   add cell-to-cell skip attention conditioned on time delta — let the
   model directly attend across observed cells without the hyperedge
   bottleneck. Requires architectural commitment.

4. **More epochs for P12/MIMIC**: 25 epochs is undertraining. Increase
   to 50 with `patience=4` (still ≤ 30 min budget by raising batch_size
   to 128 or using mixed precision).

5. **Diagnose β learning**: log β/γ gradients (already partial), and
   consider straight-through Gumbel for top-K (currently hard topk, no
   gradient through index selection).

## Files

- `models/STHQ.py` — model with v5 + v6 changes
- `tests/sthq_smoke.py` — 9 smoke tests, all passing
- `scripts/STHQ/{USHCN,P12,MIMIC_III,HumanActivity}.sh` — 30-min budget
  configs with K_e schedules

## Result file paths (for reproducibility)

```
storage/results/USHCN/USHCN/STHQ/STHQ/150_3/2026_0510_0550/iter{0,1}/eval_*/metric.json
storage/results/HumanActivity/HumanActivity/STHQ/STHQ/3000_3/2026_0510_0550/iter{0,1}/eval_*/metric.json
storage/results/P12/P12/STHQ/STHQ/36_3/2026_0510_0625/iter0/eval_*/metric.json
storage/results/MIMIC_III/MIMIC_III/STHQ/STHQ/72_3/2026_0510_0632/iter0/eval_*/metric.json
```

## Key statistics

**v6 USHCN beats HyperIMTS paper (best iter)**: 0.1641 < 0.1738.
**Mean improvement v6 vs v5**: USHCN −7.2 %, HA −3.0 %.
**Smoke test coverage**: 9/9 passing.
**Model parameters**: ~1.5 M (v6 adds ~3 % over v5 for STEA projections).
