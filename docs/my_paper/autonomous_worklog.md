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
- Next check: 15:45 UTC
