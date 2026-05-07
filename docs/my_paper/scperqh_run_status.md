# SC-PERQH Stage 1: Parallel Training Status

Launched: 2026-05-07 13:21 UTC

## Architecture

Structured-Codebook PE-RQH:
- K_time codes (Fourier-init): time-frequency templates
- K_var codes (orthogonal-init): per-variable templates
- K_event codes (random-init): learned event prototypes

Each cell receives 3 separate top-k routings, summed and Hamilton-routed via QuaternionLinear.

## Per-dataset codebook sizing

| Dataset | V | seq_len | N (cells) | K_time | K_var | K_event | K_total | K/N % |
|---------|---|---------|-----------|--------|-------|---------|---------|-------|
| USHCN | 5 | 150 | 765 | 24 | 8 | 24 | 56 | 7.3% |
| P12 | 36 | 36 | 1,404 | 16 | 48 | 32 | 96 | 6.8% |
| MIMIC_III | 89 | 72 | 6,675 | 24 | 128 | 48 | 200 | 3.0% |
| HumanActivity | 12 | 3000 | 36,036 | 64 | 16 | 32 | 112 | 0.31% |

K/N ratios are improved 3-7× over PE-RQH baseline thanks to V-aware sizing.

## Process IDs (1 trainer + 10 dataloader workers each)

- USHCN trainer: PID 34141
- P12 trainer: PID 34225
- MIMIC_III trainer: PID 34525
- HumanActivity trainer: PID 34806

## Logs

`/tmp/sqhyper_logs/SCPERQH_{USHCN,P12,MIMIC_III,HumanActivity}.log`

## Compute

GPU: NVIDIA RTX 3090 24 GB
Initial usage: 20 GB / 24 GB (higher than PE-RQH's 15 GB due to larger MIMIC codebook)

## Comparison targets

| Dataset | SQHyper baseline | PE-RQH actual | SC-PERQH target |
|---------|-----------------|---------------|-----------------|
| USHCN   | 0.191 ± 0.009 | 0.218 ± 0.030 | < 0.20 (close to SQHyper, lower variance) |
| P12     | 0.301 | TBD (val~0.30) | < 0.30 (beat SQHyper) |
| MIMIC_III | 0.420 | TBD (val~0.71) | < 0.42 (beat SQHyper) |
| HumanActivity | 0.017 | TBD (catastrophic) | < 0.02 |

The MIMIC_III recovery (from val 0.71 → ~0.42) is the most critical test of whether structured codebook re-introduces the necessary inductive bias.
