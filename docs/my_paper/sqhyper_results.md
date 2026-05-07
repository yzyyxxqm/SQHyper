# SQHyper Experimental Results (Round 1)

**Date**: 2026-05-07
**Model**: SQHyper (Spike-Quaternion Hypergraph Network)
**Hardware**: RTX 3090 24GB
**Iterations per dataset**: 5

## Completed: USHCN & HumanActivity

### USHCN (5-iter)

| iter | MSE | MAE |
|------|-----|-----|
| 0 | 0.18952 | 0.26674 |
| 1 | 0.18543 | 0.27298 |
| 2 | 0.18754 | 0.26921 |
| 3 | 0.21658 | 0.28998 |
| 4 | 0.21801 | 0.28528 |

**Summary**:
- **MSE**: mean **0.1994 ± 0.0164**, median 0.1895, min 0.1854, max 0.2180
- **MAE**: mean **0.2768 ± 0.0102**

### HumanActivity (5-iter)

| iter | MSE | MAE |
|------|-----|-----|
| 0 | 0.01608 | 0.09249 |
| 1 | 0.01644 | 0.09402 |
| 2 | 0.01678 | 0.09729 |
| 3 | 0.01844 | 0.10220 |
| 4 | 0.01892 | 0.10212 |

**Summary**:
- **MSE**: mean **0.01733 ± 0.00127**, median 0.01678, min 0.01608, max 0.01892
- **MAE**: mean **0.09763 ± 0.00449**

## Comparison vs Baselines

| Dataset | SQHyper (mean ± std) | HyperIMTS paper | QSH-Net (M1) | vs Paper | vs QSH-Net |
|---------|----------------------|------------------|----------------|----------|-------------|
| **USHCN** | 0.1994 ± 0.0164 | 0.1738 ± 0.0078 | 0.1846 ± 0.032 | +14.7% (worse) | +8.0% (worse) |
| **HumanActivity** | **0.01733 ± 0.00127** | 0.0421 ± 0.0021 | 0.0416 ± 0.0002 | **−58.8% (better)** 🎉 | **−58.3% (better)** 🎉 |

**Notes**:
- Stability on USHCN: std 0.0164 is much better than QSH-Net M1's 0.032 (halved variance), but mean slightly worse than M1 baseline.
- Stability on HumanActivity: std 0.00127 (slightly worse than M1's 0.0002 but still very tight); the absolute MSE is dramatically lower.

## In Progress

| Dataset | Status | Saved iters | ETA |
|---------|--------|-------------|-----|
| **P12** | iter4 training | iter0-3 ✓ | ~15 min |
| **MIMIC_III** | iter4 training | iter0-3 ✓ | ~15-25 min |

## Key Observations

1. **HumanActivity dramatic improvement (−58.8%)**: Unprecedented gap vs HyperIMTS paper. Worth verifying — could indicate:
   - SGI captures sparse event signal in HumanActivity sensor data effectively (this dataset has 12 motion-sensor variables with characteristic burst patterns).
   - QMF's quaternion fusion captures structured cross-variable dependencies better than flat linear.
   - Need to confirm with additional ablations.

2. **USHCN slight regression**: SQHyper trades a bit of performance on this 5-variable climate dataset. Variance is much better than M1 (0.016 vs 0.032).

3. **Architecture validation**: SGI gradient flow confirmed live (vs QSH-Net's dead spike branch at init), so the spike branch is genuinely contributing this time.

## Next Steps

- Wait for P12 and MIMIC_III completion.
- Run ablation study (`--sqhyper_no_sgi`, `--sqhyper_no_qmf`) on HumanActivity to confirm the improvement is from the new mechanisms.
- If P12 / MIMIC_III also show large improvements, consider 10-iter runs for stability verification.
