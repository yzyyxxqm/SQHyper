# QSH-Net 当前实现设计文档

> **最后更新：** 2026-04-12
> **状态：** 已验证，USHCN 和 HumanActivity 均优于 HyperIMTS 基线

---

## 1. 总体架构

QSH-Net（Quaternion-Spiking-Hypergraph Network）是一个面向不规则多元时间序列（IMTS）预测的模型。它以 HyperIMTS 的超图消息传递框架为**不变核心**，在消息传递的关键阶段有机融合了脉冲神经元事件检测和四元数 Hamilton 积特征精炼。

### 1.1 核心设计原则：恒等初始化

QSH-Net 的所有增强组件（Spike、Quaternion）在初始化时都精确等价于恒等映射。这意味着：

- **训练起点 = 纯 HyperIMTS**：模型从已验证有效的基线开始，不会因为新组件的随机初始化引入噪声
- **安全退化**：如果 Spike/Quaternion 无法学到有用的东西，它们保持近恒等，不伤害性能
- **渐进学习**：新组件只在训练证明其有益时才逐渐偏离恒等

### 1.2 信号处理流水线

```
obs → [Spike Selection] → [Hypergraph Message Passing] → [Quaternion Refinement] → prediction
         ↑ 用 H 的变量关联矩阵          (不变核心)            ↑ 精炼 H 的融合结果
```

模型由三个主要部分组成：
1. **HypergraphEncoder**：将 IMTS 数据转化为超图表示（观测节点 + 时间超边 + 变量超边）
2. **HypergraphLearner**：在超图上进行多层消息传递，每层融合脉冲选择（前置）和四元数精炼（后置）
3. **Decoder**：将更新后的节点和超边特征映射为预测值

---

## 2. 输入数据处理

### 2.1 输入格式

模型接收以下输入：
- `x`: (B, SEQ_LEN, ENC_IN) — 历史观测值
- `x_mark`: (B, SEQ_LEN, 1) — 历史时间戳
- `x_mask`: (B, SEQ_LEN, ENC_IN) — 历史观测掩码（1=有观测，0=缺失）
- `y`: (B, PRED_LEN, ENC_IN) — 预测目标值
- `y_mark`: (B, PRED_LEN, 1) — 预测时间戳
- `y_mask`: (B, PRED_LEN, ENC_IN) — 预测目标掩码

### 2.2 数据拼接

对于预测任务，历史和预测部分被拼接成完整序列：
- `x_L = cat(x, zeros_like(y))` — 观测值序列，预测位置填零
- `x_y_mask = cat(x_mask, y_mask)` — 完整掩码
- `y_L = cat(zeros_like(x), y)` — 目标值序列，历史位置填零
- `x_y_mark = cat(x_mark, y_mark)` — 完整时间戳

总序列长度 `L = SEQ_LEN + PRED_LEN`。

### 2.3 Flatten 操作

IMTS 数据中每个样本的有效观测数量不同。为了统一处理，所有 3D 张量 (B, L, ENC_IN) 被 flatten 为 2D 张量 (B, N_OBSERVATIONS_MAX)：

- 对于规则时间序列（所有位置都有观测）：直接 reshape，N = L × ENC_IN
- 对于不规则时间序列：使用 `pad_and_flatten` 函数，只保留有效观测，N = max(各样本的有效观测数)

同时生成 `time_indices_flattened` 和 `variable_indices_flattened`，记录每个 flattened 位置对应的原始时间步和变量索引。

---

## 3. HypergraphEncoder（超图编码器）

与 HyperIMTS 完全一致。

### 3.1 观测节点初始化

每个有效观测被初始化为一个 d_model 维的节点向量：

```
input = stack([观测值, 预测指示器], dim=-1)  # (B, N, 2)
observation_nodes = ReLU(Linear(2, d_model)(input)) * mask
```

其中预测指示器 = `1 - x_y_mask + y_mask`，用于区分历史观测（值=0）和预测目标（值=1）。

### 3.2 时间超边初始化

每个时间步 t 对应一个时间超边，用正弦编码初始化：

```
temporal_hyperedges = sin(Linear(1, d_model)(x_y_mark))  # (B, L, d_model)
```

### 3.3 变量超边初始化

每个变量 v 对应一个变量超边，用可学习参数初始化：

```
variable_hyperedges = ReLU(learnable_weights)  # (B, ENC_IN, d_model)
```

所有样本共享同一组初始化参数。

### 3.4 关联矩阵构建

构建两个关联矩阵（incidence matrix），定义超图的拓扑结构：

- `temporal_incidence_matrix`: (B, L, N) — 时间超边 t 连接所有 time_index=t 的节点
- `variable_incidence_matrix`: (B, ENC_IN, N) — 变量超边 v 连接所有 variable_index=v 的节点

无效观测位置（mask=0）的连接被置零。

---

## 4. HypergraphLearner（超图学习器）

超图学习器执行 n_layers 层消息传递。每层包含以下步骤：

### 4.1 步骤一：上下文感知脉冲选择（QSH-Net 增强 1）

**位置：** node-to-hyperedge 消息传递**之前**

**目的：** 利用超图的变量关联矩阵计算每个变量的聚合上下文，判断每个观测是否"偏离"其变量的典型模式。偏离较大的观测被视为"事件"（fire），全信号参与后续消息传递；其他观测被轻微衰减。

**数学公式：**
```
# 1. 利用超图结构计算变量上下文
var_count = variable_incidence_matrix.sum(dim=-1, keepdim=True).clamp(min=1)  # (B, E, 1)
var_context = (variable_incidence_matrix @ obs) / var_count                    # (B, E, D)
obs_var_ctx = var_context.gather(1, variable_indices_expanded)                 # (B, N, D)

# 2. 计算偏差并生成膜电位
deviation = obs - obs_var_ctx                                     # 观测与变量上下文的偏差
membrane = Linear(2D, 1)(cat(obs, deviation)).squeeze(-1)         # 标量膜电位 (B, N)

# 3. 脉冲发放
spike = Heaviside(membrane - threshold)                           # {0, 1}

# 4. 加权输出
attenuation = sigmoid(log_attenuation)                            # 衰减因子 ∈ (0, 1)
weight = spike + (1 - spike) * attenuation                        # firing→1, non-firing→attn
obs_selected = obs * weight.unsqueeze(-1) * mask
```

**恒等初始化策略：**

| 参数 | 初始值 | 效果 |
|------|--------|------|
| `membrane_proj.weight` | **全零** | membrane = 0 对所有观测 |
| `membrane_proj.bias` | **全零** | membrane = 0 对所有观测 |
| `threshold` | 0.0 | `0 >= 0` = True → 所有观测 fire |
| `log_attenuation` | 4.0 | `sigmoid(4) ≈ 0.982` 非 fire 时信号保留 98.2% |
| `gamma` | 5.0 | 替代梯度斜率 |

初始化后的行为：所有 membrane = 0，threshold = 0，`0 >= 0 = True`，所有观测 fire，weight = 1 → **精确恒等映射**。

**反向传播：** Heaviside 函数不可导，使用 sigmoid 导数作为替代梯度（surrogate gradient）。

**与超图的有机连接：** 脉冲决策**依赖超图的变量关联矩阵**来计算每个变量的上下文。这不是一个独立的筛选模块——它利用了超图的结构信息来判断什么是"事件"。

**参数量：** 每层 `Linear(2D, 1)` = 2D+1 个参数，加上 threshold/gamma/log_attenuation 各 1 个 = **2D + 4 个参数**（d_model=256 时仅 516 个参数）。

### 4.2 步骤二：Node-to-Hyperedge 消息传递

使用 spike-selected 的节点特征进行聚合。与 HyperIMTS 一致，但输入是经过脉冲选择的节点。

**Node→Temporal Hyperedge：**
```
Q = temporal_hyperedges                              # (B, L, D)
KV = cat(variable_HE_gathered, obs_selected)         # (B, N, 2D) — 交叉信息注入
mask = temporal_incidence_matrix                      # (B, L, N)
temporal_HE_updated = MultiHeadAttention(Q, KV, mask)
```

**Node→Variable Hyperedge：**
```
Q = variable_hyperedges                              # (B, ENC_IN, D)
KV = cat(temporal_HE_gathered, obs_selected)         # (B, N, 2D) — 交叉信息注入
mask = variable_incidence_matrix                      # (B, ENC_IN, N)
variable_HE_updated = MultiHeadAttention(Q, KV, mask)
```

**第一层特殊处理：** 在第一层中，预测目标位置的节点被从关联矩阵中屏蔽（mask_temp），防止未知的预测目标影响超边的初始聚合。

### 4.3 步骤三：Hyperedge-to-Node 信息聚合

每个节点从它所在的时间超边和变量超边获取信息：

```
tg = temporal_HE.gather(time_indices)                # (B, N, D)
vg = variable_HE.gather(variable_indices)            # (B, N, D)
```

**正常路径（node_self_update 不 OOM 时）：**
```
obs_for_h2n = MultiHeadAttention(
    Q=obs,
    KV=cat(tg, vg, obs),                            # (B, N, 3D)
    mask=obs_mask × obs_mask                          # (B, N, N) — N×N 注意力
)
h2n_input = cat(obs_for_h2n, tg, vg)                 # (B, N, 3D)
```

**OOM Fallback 路径：**
```
obs_for_h2n = obs                                    # 跳过 node_self_update
h2n_input = cat(obs, tg, vg)                         # (B, N, 3D)
```

node_self_update 需要 O(N²) 内存，在大 N 时（如 P12 有 36 变量）会触发 CUDA OOM，自动降级为 fallback 路径。

### 4.4 步骤四：四元数精炼（QSH-Net 增强 2）

**位置：** hyperedge-to-node 融合层

**目的：** 用 Hamilton 积捕获标准线性层无法表达的跨特征组交互（cross-feature-group interactions）。

当前设计为**对线性路径输出的残差精炼**：

```
# HyperIMTS 原始路径（不变）
linear_out = Linear(3D, D)(h2n_input)                # (B, N, D)

# 四元数精炼：Hamilton 积捕获 linear_out 内部的跨特征组交互
quat_out = QuaternionLinear(D, D)(linear_out)        # (B, N, D)

# 加法精炼（非插值融合）
alpha = sigmoid(gate)                                 # gate 初始化为 -3.0 → alpha ≈ 0.047
h2n_out = linear_out + alpha * quat_out
```

**关键：加法精炼 vs 插值融合**

旧版使用插值：`h2n_out = (1-α) * linear + α * quat`——quaternion 在**替代**部分 linear 输出。
当前使用加法：`h2n_out = linear + α * quat`——quaternion 在**补充** linear 输出。

**QuaternionLinear 恒等初始化：**

初始化为 `W_r = I`（单位矩阵），`W_i = W_j = W_k = 0`，此时 W 退化为块对角单位矩阵 → `QuaternionLinear(x) = x`。

**Gate 参数：**

| 参数 | 初始值 | sigmoid 值 | 含义 |
|------|--------|-----------|------|
| `gate` | -3.0 | 0.047 | 四元数精炼的初始影响 ≈ 4.7% |

**参数量：** 每层 `QuaternionLinear(D, D)` = 4 × (D/4)² + D = D²/4 + D 个参数 + 1 个 gate。d_model=256 时每层约 **16,641 个参数**。

**节点更新：**
```
observation_nodes = ReLU((obs + h2n_out) * mask)
```

### 4.5 步骤五：Variable Hyperedge-to-Hyperedge 消息传递

仅在最后一层执行，与 HyperIMTS 完全一致。让变量超边之间直接通信，学习变量间的不规则感知依赖关系。

---

## 5. Decoder（解码器）

与 HyperIMTS 完全一致。

每个节点的最终预测由三部分信息融合得到：

```
pred = Linear(3D, 1)(cat(obs, tg, vg))               # (B, N, 1) → (B, N)
```

---

## 6. 训练配置

### 6.1 优化器

QSH-Net 使用 **AdamW**（而非 Adam），添加 weight_decay = 1e-4 进行 L2 正则化。

### 6.2 学习率调度

使用 `DelayedStepDecayLR`：前 2 个 epoch 保持初始 lr，之后每个 epoch 乘以 0.8。

### 6.3 数据集特定超参数

| 参数 | USHCN | HumanActivity | P12 | MIMIC_III | MIMIC_IV |
|------|-------|---------------|-----|-----------|----------|
| d_model | 256 | 128 | 256 | 256 | 128 |
| n_layers | 1 | 3 | 2 | 2 | 4 |
| n_heads | 1 | 1 | 8 | 4 | 8 |
| seq_len | 150 | 3000 | 36 | 72 | 2160 |
| pred_len | 3 | 300 | 3 | 3 | 3 |
| enc_in (变量数) | 5 | 12 | 36 | 96 | 100 |
| batch_size | 16 | 32 | 32 | 32 | 32 |
| learning_rate | 1e-3 | 1e-3 | 1e-3 | 1e-3 | 1e-3 |
| train_epochs | 300 | 300 | 300 | 300 | 300 |
| patience | 10 | 10 | 10 | 10 | 10 |

---

## 7. 与 HyperIMTS 的精确差异

### 7.1 架构差异

| 组件 | HyperIMTS | QSH-Net | 位置 |
|------|-----------|---------|------|
| n2h 前处理 | 无 | **上下文脉冲选择** | 每层消息传递开始 |
| h2n 融合 | `Linear(3D, D)` | `Linear(3D, D) + α·QuatLinear(D,D)` | 每层 HE-to-node |
| 优化器 | Adam | **AdamW** (weight_decay=1e-4) | 训练循环 |
| 其他所有组件 | — | **完全一致** | — |

### 7.2 额外参数量

| d_model | 每层 Spike | 每层 Quaternion | 每层总计 | 相对于 HyperIMTS |
|---------|-----------|----------------|---------|-----------------|
| 128 | 260 | 4,225 | **4,486** | +1.7% |
| 256 | 516 | 16,641 | **17,158** | +1.6% |

---

## 8. 实验结果（2026-04-14 全数据集）

| 数据集 | HyperIMTS (论文, itr=5) | **QSH-Net** | 改善 vs HyperIMTS |
|--------|------------------------|-------------|-------------------|
| **MIMIC_III** (itr=5) | 0.4259 ± 0.0021 | **0.3933 ± 0.0060** | **-7.7%** ✅ |
| **MIMIC_IV** (itr=4) | 0.2174 ± 0.0009 | **0.2157 ± 0.0022** | **-0.8%** ✅ |
| **HumanActivity** (itr=5) | 0.0421 ± 0.0021 | **0.0416 ± 0.0003** | **-1.2%** ✅ |
| P12 (itr=5) | 0.2996 ± 0.0003 | 0.3006 ± 0.0013 | +0.3% ⚠ |
| USHCN (itr=5) | 0.1738 ± 0.0078 | 0.1870 ± 0.0277 | +7.6% ❌ |

---

## 9. 代码文件结构

```
models/QSHNet.py           # 模型全部实现
├── QuaternionLinear       # 四元数线性层（Hamilton 积 Kronecker 块矩阵），恒等初始化
├── SpikeFunction          # 脉冲函数（前向 Heaviside，反向 sigmoid 替代梯度）
├── SpikeSelection         # 上下文感知脉冲选择，零初始化 → 恒等
├── MultiHeadAttentionBlock# 多头注意力（与 HyperIMTS 一致）
├── IrregularityAwareAttention # 不规则感知注意力（与 HyperIMTS 一致）
├── HypergraphEncoder      # 超图编码器（与 HyperIMTS 一致）
├── HypergraphLearner      # 超图学习器（融合 Spike + Quaternion）
└── Model                  # 顶层模型类

exp/exp_main.py            # 训练循环（AdamW for QSHNet, Adam for others）
scripts/QSHNet/            # 各数据集的运行脚本
configs/QSHNet/            # 各数据集的配置文件
```
