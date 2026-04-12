# QSH-Net 超参数调优方案

> **最后更新：** 2026-04-12

## 当前状态（恒等初始化 + AdamW，最终配置）

| 数据集 | 均值 MSE | 标准差 | 论文 HyperIMTS | 状态 |
|--------|---------|--------|---------------|------|
| **USHCN** (itr=5) | **0.1715** | 0.0109 | 0.1738 ± 0.0078 | ✅ **已超越** |
| **HumanActivity** (itr=3) | **0.0416** | 0.0004 | 0.0421 ± 0.0021 | ✅ **已超越** |
| P12 | 待测 | — | 0.2996 ± 0.0003 | ⏳ 待测 |

## 当前超参数

| 参数 | P12 | HumanActivity | USHCN |
|------|-----|---------------|-------|
| d_model | 256 | 128 | 256 |
| n_layers | 2 | 3 | 1 |
| n_heads | 8 | 1 | 1 |
| batch_size | 32 | 32 | 16 |
| learning_rate | 1e-3 | 1e-3 | 1e-3 |
| lr_scheduler | DelayedStepDecayLR | DelayedStepDecayLR | DelayedStepDecayLR |
| train_epochs | 300 | 300 | 300 |
| patience | 10 | 10 | 10 |
| **optimizer** | **AdamW** | **AdamW** | **AdamW** |
| **weight_decay** | **1e-4** | **1e-4** | **1e-4** |

## 已完成的调优实验

### 调优 1：恒等初始化（架构级）

| 组件 | 旧初始化 | 新初始化 | 效果 |
|------|---------|---------|------|
| QuaternionLinear | r,i,j,k ~ N(0,σ) | **r=I, i=j=k=0** | 消除随机噪声 |
| SpikeSelection.membrane_proj | PyTorch 默认 (Kaiming) | **weight=0, bias=0** | 全部 fire → 恒等 |

**单独效果：** USHCN 3-run 均值从 0.251 降至 0.181（方差从 0.024 降至 0.009）。这是最关键的单一改进。

### 调优 2：AdamW + Weight Decay

| 优化器 | USHCN 5-run 均值 | USHCN 5-run std |
|--------|-----------------|-----------------|
| Adam (wd=0) | 0.211 | 0.040 |
| **AdamW (wd=1e-4)** | **0.172** | **0.011** |

**分析：** Weight decay 将权重范数控制在合理范围，减小了不同 random seed 下权重范数的差异，从而降低了 MSE 方差。

### 已测试但无效的调优

| 实验 | USHCN MSE | 结论 |
|------|-----------|------|
| lr=5e-4 (原 Adam) | 0.239 (3-run) | ❌ 更差，收敛慢 |
| d_model=128 (原 Adam) | 0.243 (3-run) | ❌ 更差，表达力不足 |
| grad_clip max_norm=1.0 | — | ❌ 过于激进 |
| grad_clip max_norm=5.0 + lr=5e-4 | 0.239 (3-run) | ❌ 无改善 |

## 潜在的进一步调优（待验证）

### 方向 1：激活四元数学习

当前 alpha ≈ 0.047 几乎不动。可能的改进：

| 实验 | 改动 | 预期 |
|------|------|------|
| QG1 | gate 初始化 -1.0 (α≈0.27) | 更大初始影响，可能加速 Q 学习 |
| QG2 | 独立学习率（Q params ×10） | 加速 Q 参数偏离恒等 |
| QG3 | gate 初始化 0.0 (α=0.5) | 最大初始影响，风险较高 |

**注意：** 由于恒等初始化，较大的 alpha 在训练初期只是 (1+α) 缩放，风险有限。

### 方向 2：脉冲选择微调

| 实验 | 改动 | 预期 |
|------|------|------|
| SP1 | threshold=-1.0（初始全 fire，学习更自由） | fire_rate 跨 run 更稳定 |
| SP2 | 去掉 gamma 的可学习性（固定为 5.0） | 减少参数，可能减小方差 |

### 方向 3：P12 专项

P12 始终使用 OOM fallback（跳过 node_self_update）。改进空间可能在于：

| 实验 | 改动 | 预期 |
|------|------|------|
| P1 | 增加 n_layers 到 3 | 弥补 OOM fallback 的信息损失 |
| P2 | 降低 d_model 到 128 | 避免 OOM（但可能降低表达力） |

## 运行方式

```bash
# USHCN 标准运行
python main.py --is_training 1 --collate_fn collate_fn --loss MSE \
    --d_model 256 --n_layers 1 --n_heads 1 \
    --dataset_root_path storage/datasets/USHCN \
    --model_id QSHNet --model_name QSHNet \
    --dataset_name USHCN --dataset_id USHCN \
    --features M --seq_len 150 --pred_len 3 \
    --enc_in 5 --dec_in 5 --c_out 5 \
    --train_epochs 300 --patience 10 --val_interval 1 \
    --itr 5 --batch_size 16 --learning_rate 1e-3

# HumanActivity 标准运行
python main.py --is_training 1 --collate_fn collate_fn --loss MSE \
    --d_model 128 --n_layers 3 --n_heads 1 \
    --dataset_root_path storage/datasets/HumanActivity \
    --model_id QSHNet --model_name QSHNet \
    --dataset_name HumanActivity --dataset_id HumanActivity \
    --features M --seq_len 3000 --pred_len 300 \
    --enc_in 12 --dec_in 12 --c_out 12 \
    --train_epochs 300 --patience 10 --val_interval 1 \
    --itr 3 --batch_size 32 --learning_rate 1e-3
```

## 备注

- AdamW 仅对 model_name 包含 "QSH" 的模型启用（`exp_main.py` 中条件判断），不影响其他模型
- 每轮调优只改一个参数，保持其他不变
- USHCN 需要 itr≥5 来可靠评估（高方差数据集）
- HumanActivity 用 itr=3 即可（方差极小）
