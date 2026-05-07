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

### P12 (5-iter)

| iter | MSE | MAE |
|------|-----|-----|
| 0 | 0.30121 | 0.36077 |
| 1 | 0.30222 | 0.37032 |
| 2 | 0.30095 | 0.36050 |
| 3 | 0.30007 | 0.35873 |
| 4 | 0.30008 | 0.35959 |

**Summary**:
- **MSE**: mean **0.30091 ± 0.00089**, median 0.30095, min 0.30007, max 0.30222
- **MAE**: mean **0.36198 ± 0.00473**

## Updated Comparison Table (Round 1, 3/4 datasets done)

| Dataset | SQHyper R1 | HyperIMTS paper | QSH-Net (M1) | vs Paper | vs QSH-Net |
|---------|------------|------------------|----------------|----------|-------------|
| **USHCN** | 0.1994 ± 0.0164 | 0.1738 ± 0.0078 | 0.1846 ± 0.032 | +14.7% | +8.0% |
| **P12** | 0.30091 ± 0.00089 | 0.2996 ± 0.0003 | 0.3012 ± 0.0012 | +0.4% | -0.1% |
| **HumanActivity** | **0.01733 ± 0.00127** | 0.0421 ± 0.0021 | 0.0416 ± 0.0002 | **−58.8%** 🎉 | **−58.3%** 🎉 |
| MIMIC_III | (iter4 training) | 0.4259 ± 0.0021 | 0.4047 ± 0.030 | — | — |

**Observations on P12**:
- SQHyper essentially matches HyperIMTS paper baseline (+0.4% MSE, within noise)
- Variance very tight (0.00089) — much better than QSH-Net M1 (0.0012)
- Indicates P12 has neither USHCN's bimodal instability nor HA's strong event signal — model behaves "neutrally"

### MIMIC_III (Round 1, 5-iter)

| iter | MSE | MAE |
|------|-----|-----|
| 0 | 0.44437 | 0.40727 |
| 1 | 0.45940 | 0.41452 |
| 2 | 0.39409 | 0.36957 |
| 3 | 0.39035 | 0.37330 |
| 4 | 0.41086 | 0.37539 |

**Summary**:
- **MSE**: mean **0.41981 ± 0.03074**, min 0.39035, max 0.45940

## Round 2: gate_scale fix on USHCN

**Hypothesis**: SGI's K/V gating injects noise on smooth data (USHCN climate). Fix with per-layer learnable `gate_scale` (init=0):
```python
gating = mask + gate_scale * (g_n - mask)
# gate_scale=0 → mask only (HyperIMTS-equivalent K/V)
# gate_scale=1 → full SGI gating
```

### USHCN R2 (5-iter)

| iter | MSE | MAE |
|------|-----|-----|
| 0 | 0.19040 | 0.27205 |
| 1 | 0.18648 | 0.28004 |
| 2 | 0.18954 | 0.27271 |
| 3 | 0.20556 | 0.27316 |
| 4 | 0.18307 | 0.27001 |

**Summary**:
- **MSE**: mean **0.19101 ± 0.00863**, median 0.18954, min 0.18307, max 0.20556

### USHCN R1 vs R2 Comparison

| Variant | MSE | std | Improvement vs R1 |
|---------|-----|-----|-------------------|
| R1 (uniform gating) | 0.1994 | 0.0164 | baseline |
| **R2 (learnable gate_scale)** | **0.1910** | **0.0086** | **−4.2% mean, −47% std** ✅ |

The fix worked as predicted:
- Mean reduced by 4.2% (toward HyperIMTS paper baseline)
- Variance approximately halved (0.0164 → 0.0086) — confirms removing the noisy gating stabilizes the model
- Bimodal failure mode mitigated: R1 had iter3/iter4 spikes to 0.217+, R2 max is only 0.206

## Final Comparison Table (All 4 datasets, R1 + R2 USHCN)

| Dataset | SQHyper (best) | HyperIMTS paper | QSH-Net (M1) | vs Paper | vs QSH-Net |
|---------|----------------|------------------|----------------|----------|-------------|
| **USHCN** (R2) | 0.1910 ± 0.0086 | 0.1738 ± 0.0078 | 0.1846 ± 0.032 | +9.9% | +3.5% |
| **P12** | 0.30091 ± 0.00089 | 0.2996 ± 0.0003 | 0.3012 ± 0.0012 | +0.4% | -0.1% |
| **HumanActivity** | **0.01733 ± 0.00127** | 0.0421 ± 0.0021 | 0.0416 ± 0.0002 | **−58.8%** 🎉 | **−58.3%** 🎉 |
| **MIMIC_III** | 0.41981 ± 0.03074 | 0.4259 ± 0.0021 | 0.4047 ± 0.030 | **−1.4%** ✅ | +3.7% |

**Story emerging**:
- **HA**: massive improvement (−58.8%, −58.3%) — strong event-rich data
- **MIMIC**: matches/slightly beats HyperIMTS paper, slightly behind QSH-Net
- **P12**: matches HyperIMTS paper exactly (saturated regime)
- **USHCN**: closing gap with R2 fix (+9.9% from +14.7%), still room for improvement

**Mean improvement vs HyperIMTS paper across 4 datasets**: 
- HA −58.8%, MIMIC −1.4%, P12 +0.4%, USHCN +9.9%
- Average: −12.5% (heavily driven by HA)
- Median (excluding HA outlier): +0.4% — neutral

**Net assessment**: SQHyper is a **selective improvement**, not uniform. It produces dramatic gains on event-structured high-temporal-density data (HA), and matches baseline elsewhere. R2 fix successfully addressed the USHCN regression.

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
