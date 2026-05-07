# PE-RQH Stage 1: Parallel Training Status

Launched: 2026-05-07 11:44 UTC (~19:44 +08:00)

## Processes (1 trainer × 10 dataloader workers each)

| Trainer PID | Dataset | Script | K | Layers | Batch | Epochs | LR |
|-------------|---------|--------|---|--------|-------|--------|-----|
| 4843 | USHCN | scripts/PERQH/USHCN.sh | 32 | 2 | 16 | 300 | 1e-3 |
| 4927 | P12 | scripts/PERQH/P12.sh | 64 | 2 | 32 | 300 | 1e-3 |
| 5100 | MIMIC_III | scripts/PERQH/MIMIC_III.sh | 64 | 2 | 32 | 300 | 1e-3 |
| 5383 | HumanActivity | scripts/PERQH/HumanActivity.sh | 64 | 3 | 16 | 300 | 1e-3 |

## Logs

`/tmp/sqhyper_logs/PERQH_{USHCN,P12,MIMIC_III,HumanActivity}.log`

## Compute

GPU: NVIDIA RTX 3090 24 GB
Initial usage: 15.4 GB / 24 GB

## Reference (SQHyper baselines to beat)

| Dataset | SQHyper MSE | Target |
|---------|-------------|--------|
| USHCN   | 0.191       | < 0.21 (allow modest regression) |
| P12     | 0.301       | < 0.31 |
| MIMIC_III | 0.420     | < 0.42 |
| HumanActivity | 0.0173 | < 0.020 |

## Monitoring commands (server)

```bash
# Check still running
ssh server "ps aux | grep main.py | grep -v grep | awk '{print \$2}' | wc -l"
# Should be 1 + 10 = 11 per dataset = 44 total

# Tail one log
ssh server "tail -50 /tmp/sqhyper_logs/PERQH_USHCN.log"

# GPU usage
ssh server "nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader"
```
