# Autonomous Worklog: STHQ → beat HyperIMTS
Deadline: 2026-05-08 00:00 UTC (Beijing 8 AM)
Start: 2026-05-07 15:05 UTC (Beijing 11:05 PM)
Available time: ~9 hours

## Targets (to beat, from SQHyper baselines ≈ HyperIMTS)

| Dataset | Target MSE | PE-RQH | SC-PERQH | STHQ fix @ 7min |
|---------|-----------|--------|----------|-----------------|
| USHCN | < 0.191 | 0.218 | 0.197 val | 0.206 val |
| P12 | < 0.301 | 0.40 stuck | 0.41 val | 0.396 val ↓ |
| MIMIC_III | < 0.420 | 0.81 stuck | 0.82 val | 0.824 val |
| HumanActivity | < 0.017 | 0.32 stuck | 0.17 val | 0.183 val ↓ |

## Strategy (budget-aware, infrequent checks)

- Progress checks spaced 40-60 min apart
- Per iteration: check → analyze → fix → push → restart → wait

## Known risk points to monitor after each check

1. Spike collapse (all ~0 or ~1)
2. Hyperedge starvation (some never used)
3. τ collapse (all same position)
4. Hamilton product magnitude issues
5. Training slowing down (vanishing grad)
6. Early stopping on poor plateau

## Pre-planned improvements (in order of likely impact)

| # | Change | Motivation | Implementation cost |
|---|--------|-----------|---------------------|
| I1 | Time-aware decoder: y = MLP(q, pos_emb(time, var)) | Explicit position injection to decoder | Low |
| I2 | Skip connection: decoder sees q_init + q_final | Preserve raw position info | Low |
| I3 | Diagnostic: print spike statistics per layer | Verify spike not collapsed | Low |
| I4 | Add L2 reg on τ to prevent collapse | Prevent degenerate hyperedges | Low |
| I5 | Replace Hamilton with QuaternionLinear(concat) | If Hamilton damps signal | Medium |
| I6 | Add direct cell-cell attention within top cluster | More expressive message passing | Medium |
| I7 | Multi-bandwidth ω within each layer | Capture multi-scale in one layer | Medium |
| I8 | Decoder with Q-dim per-component extraction | Let decoder see typed components | Low |
| I9 | Learning rate warmup + cosine | Smoother training | Low |

## Progress log

### 15:05 UTC — Iteration 1 started
- STHQ with query-distribution bug FIXED running on 4 datasets
- All smoke tests pass including regression test

### 16:17 UTC — Iteration 1 results (82 min runtime)
- USHCN full 5 iters: MSE 0.254/0.229/0.248/0.185/0.198 -> mean 0.223 ± 0.027
  - Best 0.185 beats SQHyper 0.191; mean 17% worse
- P12 iter 2: val 0.357 (was stuck at 0.40)
- MIMIC_III iter 0 (80min!): val 0.578, 7/10 counter (too slow, will be killed)
- HumanActivity iter 4: val 0.162 (5x improvement vs PE-RQH 0.32)

### 16:25 UTC — Iteration 2 started (v2)
Changes:
  - Time-aware decoder: concat(q_final, q_initial, sin/cos(time), var_emb)
  - Dropout 0.1 on msg_proj + decoder
  - Anti-collapse aux loss (τ-repulsion + var-entropy bonus)
  - MIMIC: d_model 256→128, K_t/K_v 32/24→24/16, itr 5→3, patience 10→7
  - P12 patience 10→7
  - Loss: MSE_aux to include aux_loss in total
Next check: 17:10 UTC (after ~45 min, expect USHCN done + MIMIC 1-2 iters)

### 17:10 UTC — 45 min into v2
- USHCN best val 0.194 (similar to v1)
- P12 plateauing at 0.344
- MIMIC iter 0 at 0.608 (descending slowly)
- HA best val 0.138 (better than v1 0.162) ✅

### 17:56 UTC — 90 min into v2 — MAJOR RESULTS
🎉 USHCN: ALL 5 ITERS DONE
  Test MSEs: 0.177, 0.211, 0.182, 0.171, 0.170
  Mean 0.182 ± 0.015
  v1: 0.223 ± 0.027 (best 0.185)
  v2: 0.182 ± 0.015 (best 0.170)
  ✅ BEATS SQHyper baseline 0.191 by 5%
  ✅ Variance halved

🎉 HumanActivity: ALL 5 ITERS DONE
  Test MSEs: 0.085, 0.103, 0.095, 0.103, 0.092
  Mean 0.096 ± 0.008
  vs PE-RQH/SC-PERQH 0.32, much better

P12: iter 1/5, val 0.354 (slower descent)
MIMIC: iter 0/3, val 0.588, counter 3/7

Time analysis:
- 17:56 UTC → 6h 4min remaining
- P12: ~2-3 hours more (5 iters)
- MIMIC: ~2 hours more (3 iters)
- Buffer: ~2 hours for analysis/final touches

Decision: let everything finish. Monitor at 19:30 UTC.

### 23:09 UTC — Iteration 3 started (v3)
v2 results showed gap to HyperIMTS still big on P12/MIMIC/HA.
v3 changes:
  - Hybrid Hamilton + Linear msg path (learnable α)
  - K_t/K_v scaled per dataset:
    USHCN: 24/5 → 48/5
    P12: 24/16 → 48/24
    MIMIC: 24/16 → 48/32, d_model 128 → 192
    HA: 64/12 → 256/12
  - HA omega range tightened

### 2026-05-08 00:40 UTC — User stopping server
v3 status:
  USHCN ✅ ALL 5 ITERS DONE
    Test MSEs: 0.170, 0.165, 0.179, 0.171, 0.174
    Mean 0.172 ± 0.005 (best result of all attempts)
    vs HyperIMTS-like 0.167 ± 0.004 → essentially MATCHED (within 1σ)
  
  HumanActivity: 4 iters early-stopped, 5th in progress (val_best 0.130)
  P12: iter 0 early-stopped (val 0.370), iter 1 just started
  MIMIC: iter 0 still running (val 0.570)

Final STHQ best results compiled:
  USHCN:        0.172 ± 0.005 (v3) vs HyperIMTS 0.167  — MATCHED
  P12:          0.355 ± 0.007 (v2) vs HyperIMTS 0.301  — gap +18%
  MIMIC_III:    0.568 ± 0.014 (v2) vs HyperIMTS 0.394  — gap +44%
  HumanActivity: 0.096 ± 0.008 (v2) vs HyperIMTS 0.042 — gap +130%

Goal "明显优于 HyperIMTS" NOT achieved.
USHCN now matches; P12/MIMIC/HA still significantly behind.

Final analysis written to STHQ_final_analysis.md
