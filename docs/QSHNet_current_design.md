# QSH-Net 历史实现设计文档（2026-04-15 快照）

> **最后更新：** 2026-04-15
> **状态：** 历史设计快照，仅用于保留当时的实现思路与阶段性结果

## 使用说明

这份文档记录的是 **2026-04-15** 时点的 M1 设计与阶段性实验结果。

它的用途是：

- 保留当时的实现思路
- 对照早期结构假设与当前实际演化之间的差异

它**不是**当前状态的最终依据。查询当前模型情况时，优先看：

- [QSHNet_overview.md](/opt/Codes/PyOmniTS/docs/QSHNet_overview.md)
- [QSHNet_architecture_status.md](/opt/Codes/PyOmniTS/docs/QSHNet_architecture_status.md)
- [hyperparameter_tuning_plan.md](/opt/Codes/PyOmniTS/docs/hyperparameter_tuning_plan.md)

---

## 1. 总体架构

QSH-Net（Quaternion-Spiking-Hypergraph Network）是一个面向不规则多元时间序列（IMTS）预测的模型。当前实现采用方案 B 的 **M1 分层协同版本**：保留 HyperIMTS 的超图消息传递框架作为稳定骨架，在消息传递内部引入连续 Spike 路由和条件化 Quaternion 融合。

### 1.1 核心设计原则：安全初始化

当前实现不再追求所有增强分支都“严格精确恒等”，而是追求**尽可能接近主干、且不会在训练初期大幅污染主路径**。这意味着：

- **Spike 主路径精确保真：** `retain_gate` 初始为 1，`obs_base = obs`
- **事件支路初始静默：** `event_proj` 全零初始化，`event_residual_scale = 0`，所以 `obs_event` 与事件残差初始不影响主干
- **Quaternion 弱残差启动：** 条件化 gate 初始化在 `sigmoid(-3) ≈ 0.047` 附近，四元数残差以很弱强度参与，而不是从随机强干预开始
- **渐进学习：** 只有当训练证明这些增强有益时，路由与条件化融合才会偏离初始安全状态

### 1.2 信号处理流水线

```
obs → [Spike Router: base / event split] → [Hypergraph Message Passing + event residual] → [Quaternion-conditioned fusion] → prediction
         ↑ 用 H 的变量关联矩阵                     (不变骨架)                            ↑ 用 event_gate 调节四元数残差
```

模型由三个主要部分组成：
1. **HypergraphEncoder**：将 IMTS 数据转化为超图表示（观测节点 + 时间超边 + 变量超边）
2. **HypergraphLearner**：在超图上进行多层消息传递，每层执行连续 Spike 路由、事件增强残差和条件化四元数融合
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

### 4.1 步骤一：连续 Spike 路由（QSH-Net 增强 1）

**位置：** node-to-hyperedge 消息传递**之前**

**目的：** 利用超图的变量关联矩阵计算每个变量的聚合上下文，将每个观测拆成两条路径：

- **`obs_base`：** 走稳定主干的基础路径
- **`obs_event`：** 只在需要时参与事件增强流的事件路径

当前实现采用**连续 gate**，不再用硬二值 fire 直接控制主路径。

**数学公式：**
```
# 1. 利用超图结构计算变量上下文
var_count = variable_incidence_matrix.sum(dim=-1, keepdim=True).clamp(min=1)  # (B, E, 1)
var_context = (variable_incidence_matrix @ obs) / var_count                    # (B, E, D)
obs_var_ctx = var_context.gather(1, variable_indices_expanded)                 # (B, N, D)

# 2. 计算偏差并生成路由分数
deviation = obs - obs_var_ctx
route_logit = Linear(2D, 1)(cat(obs, deviation)).squeeze(-1)      # (B, N)

# 3. 连续 gate
retain_gate = 1 - (exp(retain_log_scale) - 1) * sigmoid(-route_logit)
event_gate = exp(event_log_scale) * sigmoid(route_logit)

# 4. 输出两条路径
event_features = Linear(2D, D)(cat(obs, deviation))
obs_base = obs * retain_gate.unsqueeze(-1) * mask
obs_event = event_features * event_gate.unsqueeze(-1) * mask
```

**安全初始化策略：**

| 参数 | 初始值 | 效果 |
|------|--------|------|
| `membrane_proj.weight/bias` | **全零** | `route_logit = 0` |
| `retain_log_scale` | 0.0 | `retain_gate = 1` → 主路径精确保真 |
| `event_proj.weight/bias` | **全零** | 事件特征初始为 0 |
| `event_log_scale` | -8.0 | `event_gate ≈ 1.7e-4`，且乘上全零事件特征后输出仍为 0 |
| `threshold` / `gamma` | 0.0 / 5.0 | 当前主路径未直接使用，保留作诊断与后续脉冲语义扩展 |

初始化后的行为：`obs_base = obs`，`obs_event = 0`，因此 SpikeRouter 在训练开始时近似等价于纯主干前处理。

**与超图的有机连接：** 路由分数依赖超图的变量关联矩阵来计算每个变量的上下文。也就是说，事件判断不是外部启发式，而是建立在超图结构之上。

**参数量：** 每层 SpikeRouter 额外参数约为 `2D² + 3D + 5`。当 `d_model=256` 时约为 **131,845** 个参数，主要增长来自 `event_proj(2D→D)`。

### 4.2 步骤二：Node-to-Hyperedge 消息传递 + 事件增强残差

普通 node-to-hyperedge 聚合仍沿用 HyperIMTS 的 attention 主干，但输入从单一路径变为 `obs_base`，同时为 temporal / variable hyperedge 各自添加一条事件增强残差。

**Node→Temporal Hyperedge：**
```
temporal_HE_base = MultiHeadAttention(
    Q=temporal_hyperedges,
    KV=cat(variable_HE_gathered, obs_base),
    mask=temporal_incidence_matrix
)

temporal_event_delta = (temporal_incidence_matrix @ obs_event) / temporal_count
temporal_HE_updated = temporal_HE_base + event_scale * temporal_event_delta
```

**Node→Variable Hyperedge：**
```
variable_HE_base = MultiHeadAttention(
    Q=variable_hyperedges,
    KV=cat(temporal_HE_gathered, obs_base),
    mask=variable_incidence_matrix
)

variable_event_delta = (variable_incidence_matrix @ obs_event) / variable_count
variable_HE_updated = variable_HE_base + event_scale * variable_event_delta
```

其中：

- `event_scale = exp(event_residual_scale) - 1`
- `event_residual_scale` 初始化为 0，因此 `event_scale = 0`
- 在初始化时，事件残差对超边更新**没有任何影响**

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

### 4.4 步骤四：条件化四元数融合（QSH-Net 增强 2）

**位置：** hyperedge-to-node 融合层

**目的：** 用 Hamilton 积捕获标准线性层难以表达的跨特征组交互，并通过 `event_gate` 让四元数残差根据节点状态与事件性进行条件化参与。

当前设计为**线性主路径 + 条件化四元数残差**：

```
# HyperIMTS 原始路径（不变）
linear_out = Linear(3D, D)(h2n_input)                # (B, N, D)

# 四元数残差
quat_out = QuaternionLinear(D, D)(linear_out)        # (B, N, D)

# 条件化 gate
quat_gate = sigmoid(Linear(D+1, 1)(cat(linear_out, event_gate)))
h2n_out = linear_out + quat_gate * quat_out
```

**初始化策略：**

- `QuaternionLinear` 仍采用恒等初始化：`W_r = I`，`W_i = W_j = W_k = 0`
- 条件 gate 的权重初始化为 0，bias 初始化为 -3.0
- 因此 `quat_gate ≈ 0.047`，四元数残差以较弱强度参与

**与旧版差异：**

- 旧版 gate 是全局单标量，所有节点共享同一 `alpha`
- 当前 gate 是节点级条件化标量，输入依赖 `linear_out` 和 `event_gate`
- 这意味着四元数分支不再是全局统一强度，而是按节点自适应启用

**当前实现的现实含义：**

- h2n 融合端不是严格等于旧版主干，而是从一个**弱四元数残差**状态启动
- 当前实验结果表明，这样的条件化设计方向可行，但在部分 seed 上仍存在偶发失稳问题

**参数量：** 每层条件化四元数部分额外参数约为 `D²/4 + 2D + 3`（含 `event_residual_scale` 与条件 gate）。当 `d_model=256` 时约为 **16,899** 个参数。

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
| n2h 前处理 | 无 | **SpikeRouter：`obs_base / obs_event / route_state`** | 每层消息传递开始 |
| 超边更新 | 纯 attention 聚合 | **attention 主路径 + 事件增强残差** | 每层 temporal / variable 更新 |
| h2n 融合 | `Linear(3D, D)` | `Linear(3D, D) + quat_gate(node,event)·QuatLinear(D,D)` | 每层 HE-to-node |
| 优化器 | Adam | **AdamW** (weight_decay=1e-4) | 训练循环 |
| 其他所有组件 | — | **完全一致** | — |

### 7.2 额外参数量

| d_model | 每层 SpikeRouter | 每层 Event/CondQuat | 每层额外总计 |
|---------|------------------|---------------------|----------------|
| 128 | 33,157 | 4,355 | **37,512** |
| 256 | 131,845 | 16,899 | **148,744** |

M1 不再是早期“轻量插件版” QSH-Net。当前参数增长的主要来源是 `event_proj(2D→D)`，这是为了给事件增强流提供足够表达力而引入的结构性代价。

---

## 8. 实验结果（2026-04-15，M1 当前实现）

| 数据集 | 论文 HyperIMTS | 当前 M1 结果 | 结论 |
|--------|----------------|--------------|------|
| USHCN (10 轮) | 0.1738 ± 0.0078 | **0.1846 ± 0.0323** | 均值略优于旧版 QSH，但双峰与异常轮仍在 |
| HumanActivity (5 轮) | 0.0421 ± 0.0021 | **0.0416 ± 0.0002** | 基本持平，安全退化成立 |
| P12 (5 轮) | 0.2996 ± 0.0003 | 0.3012 ± 0.0012 | 略有退步，但幅度较小 |
| MIMIC_III (5 轮) | 0.4259 ± 0.0021 | **0.4047 ± 0.0300** | 仍优于论文基线，但较旧版 QSH 的均值和稳定性退步 |
| MIMIC_IV | 0.2174 ± 0.0009 | 未运行 | 因训练成本高，暂未评估 |

当前实现的整体判断如下：

- **结构方向成立。** M1 证明了分层协同设计可以安全接入 HyperIMTS 主干。
- **当前主要问题是偶发失稳。** USHCN 和 MIMIC_III 都保留了好轮次，但在部分 seed 上会出现明显坏轮。
- **M1 适合作为当前保留版本。** 这版不应回退，但下一阶段目标应是稳定化，而不是立刻扩展 M2。

---

## 9. 代码文件结构

```
models/QSHNet.py           # 模型全部实现
├── QuaternionLinear       # 四元数线性层（Hamilton 积 Kronecker 块矩阵），恒等初始化
├── SpikeFunction          # 早期脉冲函数实现，当前 M1 主路径未直接使用
├── SpikeRouter            # 连续路由器：输出 obs_base / obs_event / route_state
├── SpikeSelection         # 兼容别名，继承自 SpikeRouter
├── MultiHeadAttentionBlock# 多头注意力（与 HyperIMTS 一致）
├── IrregularityAwareAttention # 不规则感知注意力（与 HyperIMTS 一致）
├── HypergraphEncoder      # 超图编码器（与 HyperIMTS 一致）
├── HypergraphLearner      # 超图学习器（融合 Spike 路由 + 事件残差 + 条件 Quaternion）
└── Model                  # 顶层模型类

exp/exp_main.py            # 训练循环（AdamW for QSHNet, Adam for others）
scripts/QSHNet/            # 各数据集的运行脚本
configs/QSHNet/            # 各数据集的配置文件
tests/models/test_QSHNet.py# M1 单元测试：路由初始化、条件 gate、模型前向 smoke test
```
