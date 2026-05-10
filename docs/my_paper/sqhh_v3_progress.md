# SQHH v3 自主训练进度（2026-05-10 → 2026-05-11）

## 时间线
- 2026-05-10 22:48 (UTC+8): 启动 USHCN/P12/MIMIC v3 训练
- Deadline: 2026-05-11 08:00 (UTC+8)
- 总预算: 9.2h

## 训练配置
所有运行使用 `--use_amp 1` (bf16 autocast) + `--use_compile 0`（compile val 阶段 recompile thrash）。

| 数据集 | itr | patience | d_model | n_layers | bs | lr | 状态 |
|---|---|---|---|---|---|---|---|
| USHCN | 5 | 25 | 256 | 2 | 16 | 1e-3 | 训练中 |
| P12 | 1 | 10 | 256 | 2 | 32 | 1e-3 | 训练中 |
| MIMIC_III | 1 | 10 | 256 | 2 | 32 | 1e-3 | 训练中 |
| HumanActivity | 5 | (前次完成) | 128 | 3 | 32 | 1e-3 | 已完成 (eval_2026_0510_1251) |

## 速度（v3 优化后实测）
- USHCN: 23.85 s/epoch（原 24.8 s/epoch，提升不显著因 USHCN 模型小）
- P12: 277 ms/batch（原 350 ms/batch，提升 ~25%）
- MIMIC_III: 395 ms/batch（原 350 ms/batch，3 进程 GPU 共享略慢）

## HyperIMTS 基线（论文）
| 数据集 | MSE 目标（≤） | MAE | 备注 |
|---|---|---|---|
| USHCN | 0.1738 ± 0.0078 | — | 最难打 |
| HumanActivity | 0.0421 ± 0.0021 | — | SQHyper 已 -58% (0.0173) |
| P12 | 0.2996 ± 0.0003 | — | 平 |
| MIMIC_III | 0.4259 ± 0.0021 | — | SQHyper 已 -1.4% (0.420) |

## 历史 SQHH 结果（commit 1136 run）
- USHCN: 0.1934 ± 0.008 (5 itrs) — **比 HyperIMTS 差 +11%**
- HumanActivity: 0.0207 ± 0.0019 (5 itrs) — **比 HyperIMTS 好 -51%** ✅
- P12 / MIMIC: 训练未完成

## 第一轮结果（v3, bf16）

| 数据集 | SQHH v3 MSE | HyperIMTS | 差距 | 状态 |
|---|---|---|---|---|
| HumanActivity | 0.0207 ± 0.0019 (5 itr, 旧 1136 run) | 0.0421 | **-51%** ✅ | beat |
| MIMIC_III | 0.39078 (1 itr) | 0.4259 | **-8.2%** ✅ | beat |
| P12 | 0.30405 (1 itr) | 0.2996 | +1.5% ❌ | 略差 |
| USHCN | 0.207 ± 0.018 (5 itr) | 0.1738 | +19% ❌ | 差很多 |

### USHCN 5-iter 详细
| iter | MSE |
|---|---|
| iter0 | 0.18587 |
| iter1 | 0.21564 |
| iter2 | 0.23244 |
| iter3 | 0.20111 |
| iter4 | 0.19886 |

iter0 已经接近 HyperIMTS 0.1738（差 7%），但其他 iter 高方差。问题：bf16 在 USHCN 这种平滑小数据集可能伤害收敛 + d_model=256 偏大过拟合。

## 第二轮（v3.1）

### USHCN_v31.sh: fp32 + d_model=192 + lr=5e-4 + patience=30
- 已启动 16:41 UTC
- 目标: 拉低 mean 到 < 0.1738

### P12_v31.sh: bf16 + patience=20 + lr=5e-4
- 已启动 16:04 UTC，best val 0.289（已优于 HyperIMTS 0.2996）
- 目标: test MSE < 0.295
