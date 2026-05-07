# Model Redesign Analysis: From Patch to Principled Architecture

## Phase 1: HyperIMTS 结构性不足分析

通过深入阅读 HyperIMTS 源码 (`models/HyperIMTS.py`) 和当前 QSH-Net (`models/QSHNet.py`)，
我们识别出 HyperIMTS 在**超图结构**和**消息融合**两个层面的结构性假设缺陷。

### 不足 1: 二值关联矩阵 — 观测同质性假设

HyperIMTS 的关联矩阵 H_T ∈ {0,1}^{L×N} 和 H_V ∈ {0,1}^{V×N} 是**纯二值**的：
一个观测要么属于某个超边，要么不属于，没有中间状态。

```python
# HyperIMTS.py line 358-362
temporal_incidence_matrix = (temporal_incidence_matrix == ... - 1).to(torch.float32)
# → 结果是 0 或 1，没有数据依赖的权重
```

**问题**：在 IMTS 中，同一时刻的多个观测具有截然不同的信息量。例如：
- 心率在正常范围内 → 常规观测，对预测贡献有限
- 心率突然飙升 → 事件信号，对预测至关重要

但 H_T 对两者赋予相同的权重 (=1)。在 node→hyperedge 的 attention 聚合中，
虽然 attention weight 会学习区分，但超图结构本身没有这个信息。
这意味着**超图的拓扑结构是数据无关的**，只取决于观测存在性，不取决于观测内容。

### 不足 2: 扁平化多源融合 — 丢失语义结构

HyperIMTS 的 hyperedge→node 融合步骤：

```python
# HyperIMTS.py line 551
observation_nodes = activation(
    observation_nodes + hyperedge2node[i](
        cat([self_attn_out, temporal_gathered, variable_gathered], dim=-1)
    )
)
# hyperedge2node = nn.Linear(3*d_model, d_model)
```

这里 `Linear(3D, D)` 将三个语义不同的来源**扁平拼接后线性投影**：
- `self_attn_out` (D): 节点自身的上下文更新
- `temporal_gathered` (D): 来自时间超边的消息
- `variable_gathered` (D): 来自变量超边的消息

**问题**：单一线性层将 3D 维视为独立标量的线性组合，没有任何归纳偏置来建模三个来源之间的**结构化交叉交互**。
然而，节点更新本质上需要多源信息的耦合：
- 某个变量的时间趋势（temporal×variable 交互）
- 事件信号如何影响特定变量（event×variable 交互）
- 时间模式在自身历史中的位置（temporal×self 交互）

一个 3D→D 的线性层虽然**理论上**能学到任意映射（足够参数），
但它没有**归纳偏置**来优先建模这些交叉交互，需要从数据中从零学起。

### 为什么之前的 QSH-Net 失败了

之前的设计是 "HyperIMTS + spike 附件 + quaternion 附件"：
- Spike Router: 在 message passing **之前**过滤节点 → 本质是一个可选的预处理步骤
- Quaternion Refinement: 在 linear h2n **之后**加残差 → 本质是一个可选的后处理步骤

两者都设计为**身份初始化**，可以完全关闭而不影响模型。
结果正如 ablation 所示：模型学会了不使用它们（3/4 数据集上无差异），
因为 HyperIMTS 骨架已经足够好，附加模块提供的边际收益不足以克服其引入的优化噪声。

**根本原因**：spike 和 quaternion 不是模型的**结构性组成部分**，而是可拆卸的插件。
好的模型设计应该让每个组件都是不可或缺的。

---

## Phase 2: 脉冲和四元数如何**自然涌现**为解决方案

### 洞察 1: 脉冲机制 → 数据自适应超图拓扑

IMTS 的核心特征是**观测异质性**：不同观测携带的信息量差异巨大。
脉冲神经元的 integrate-and-fire 机制天然适合量化这种异质性：

- 每个观测节点根据其偏离上下文的程度计算"膜电位"
- 膜电位通过 sigmoid（或脉冲函数 + 代理梯度）转化为 [0,1] 的激活权重
- 这个权重**直接作为关联矩阵的软权重**：H_soft[l,n] = H_binary[l,n] × g_n

**关键区别**：这不是在消息传递"之前"的滤波器，而是**重新定义了超图结构本身**。
超图的拓扑从"数据无关"变为"数据自适应"——信息量大的观测在超图中具有更强的连接权重。

**理论支撑**：
- 与注意力机制相比：attention 在聚合阶段学习权重，但超图结构仍是固定的。
  Spike-gated incidence 在结构层面就已经编码了观测重要性。
- 与 GNN 中的 edge weight 类比：在图神经网络中，数据依赖的边权重已被广泛验证有效
  (GAT, GIN 等)。我们将这一思想推广到超图的关联矩阵。

### 洞察 2: 四元数代数 → 多源结构化融合

HyperIMTS 的 h2n 融合涉及 3 个语义不同的信息源。
如果我们引入第 4 个源（事件/偏差上下文，由脉冲机制提供），
恰好得到 **4 个语义组**——这正是四元数代数的天然结构：

| 四元数分量 | 语义来源 | 含义 |
|-----------|---------|------|
| R (实部) | 节点自身嵌入 | 观测值的基础表示 |
| I (虚部1) | 时间超边上下文 | 时间维度的聚合信息 |
| J (虚部2) | 变量超边上下文 | 变量维度的聚合信息 |
| K (虚部3) | 事件偏差上下文 | 脉冲机制检测到的异常信号 |

Hamilton 乘积的代数结构自动产生所有成对交叉交互：
- R×I: 观测值如何与时间模式关联
- R×J: 观测值如何与变量特征关联
- I×J: 时间和变量信息的耦合（最重要的交互之一）
- R×K, I×K, J×K: 事件信号如何调制其他三个来源

**关键区别**：这不是在线性层"之后"的残差修正，而是**取代线性层成为主融合机制**。
四元数不是可选的增强，而是模型融合多源信息的核心方式。

**理论支撑**：
- 四元数线性层的参数量是标准线性层的 1/4（4×(D/4)² vs D²），
  但通过 Hamilton 乘积的结构约束，强制建模了所有 6 种成对交互。
- 这是一种**参数效率更高**的归纳偏置：用更少的参数，
  通过代数结构约束来实现标准线性层需要大量数据才能学到的交叉交互。

### 脉冲与四元数的紧耦合

两个机制不是独立的，而是紧密耦合的：
1. **脉冲 → 四元数**：脉冲机制的事件偏差输出直接作为四元数的第 4 分量 (K)，
   为融合提供事件调制信号
2. **四元数 → 脉冲**：四元数融合后的节点更新进入下一层的脉冲机制，
   影响下一层的膜电位计算
3. **超图结构 ↔ 消息融合**：脉冲定义超图拓扑权重，四元数定义消息融合方式，
   两者共同构成了完整的消息传递流水线

---

## Phase 3: 新模型架构设计

### 模型名称: SQHyper (Spike-Quaternion Hypergraph Network)

### 设计哲学

> HyperIMTS 证明了超图是 IMTS 的有效表示。但它的超图结构是静态的（二值关联矩阵），
> 消息融合是非结构化的（扁平线性投影）。我们提出 SQHyper，通过两个紧耦合的机制
> 解决这两个结构性缺陷：脉冲门控使超图拓扑自适应于观测信息量，
> 四元数融合使多源消息传递保持语义结构。

### 3.1 架构总览

```
Input IMTS → Encoder (same as HyperIMTS) → K × SQHyper Layer → Decoder
```

每个 SQHyper Layer 包含：
1. Spike-Gated Incidence (SGI) — 计算软关联权重
2. Weighted Node→Hyperedge Aggregation — 使用软权重的注意力聚合
3. Quaternion Multi-Source Fusion (QMF) — 四元数结构化的 hyperedge→node 融合
4. Residual Node Update

### 3.2 Spike-Gated Incidence (SGI)

**输入**: 当前层的节点嵌入 z_n^(k)，变量关联矩阵 H_V
**输出**: 门控权重 g_n ∈ (0,1]，事件特征 e_n

```
Step 1: 计算变量级上下文
  μ_v = (H_V @ z) / H_V.sum()           # 每个变量的平均嵌入

Step 2: 计算偏差
  d_n = z_n - μ_{v(n)}                    # 节点相对于其变量上下文的偏差

Step 3: 计算膜电位
  u_n = w^T [z_n || d_n] + b              # 标量膜电位 (Linear(2D,1))

Step 4: 计算门控权重（连续松弛的脉冲函数）
  g_n = σ(u_n)                             # sigmoid 替代硬阈值

Step 5: 生成软关联矩阵
  H_T_soft[l,n] = H_T[l,n] * g_n          # 时间关联加权
  H_V_soft[v,n] = H_V[v,n] * g_n          # 变量关联加权

Step 6: 提取事件特征（用于四元数的第 4 分量）
  e_n = W_e [z_n || d_n] + b_e            # 事件特征投影 (Linear(2D, D/4))
```

**与旧设计的区别**：
- 旧：spike 输出 obs_base 和 obs_event，是对节点的两路拷贝
- 新：spike 输出 (1) 作用于关联矩阵的权重 g_n 和 (2) 作为四元数第 4 分量的事件特征 e_n
- 旧：移除 spike → 模型不变（identity init）
- 新：移除 spike → 关联矩阵退化为二值，四元数丢失 K 分量 → 结构性降级

### 3.3 Weighted Node→Hyperedge Aggregation

使用 SGI 产生的软关联矩阵，node→hyperedge 的 attention 被修改为：

```
# 时间超边更新（修改 attention mask/weight）
τ_l^(k+1) = MHA(
    Q = τ_l^(k),
    K = [ν_{v(n)} || z_n * g_n] for n in N_T(l),    # key/value 被 spike 权重调制
    mask = H_T_soft[l, :]                              # 使用软关联作为 attention mask
)

# 变量超边更新（类似）
ν_v^(k+1) = MHA(
    Q = ν_v^(k),
    K = [τ_{l(n)} || z_n * g_n] for n in N_V(v),
    mask = H_V_soft[v, :]
)
```

**关键**：spike 权重 g_n 在两个地方起作用：
1. 调制 key/value 的贡献强度（乘以 g_n）
2. 作为 attention mask 的软权重（通过 H_soft）

### 3.4 Quaternion Multi-Source Fusion (QMF)

**核心创新**：将 hyperedge→node 融合从扁平线性投影改为四元数结构化融合。

```
Step 1: 收集多源信息
  f_self = z_n^(k)                              # 节点自身 (D)
  f_temporal = τ_{l(n)}^(k+1)                   # 时间超边消息 (D)
  f_variable = ν_{v(n)}^(k+1)                   # 变量超边消息 (D)

Step 2: 投影到四元数分量空间
  q_R = W_R(f_self)                              # D → D/4, 实部
  q_I = W_I(f_temporal)                          # D → D/4, 虚部1
  q_J = W_J(f_variable)                          # D → D/4, 虚部2
  q_K = e_n  (from SGI, already D/4)             # 虚部3 = 事件特征

Step 3: 拼接为四元数向量
  q = [q_R | q_I | q_J | q_K]                    # (D)

Step 4: 四元数线性变换 (Hamilton product)
  h2n_out = QuatLinear(q) + bias                  # D → D

Step 5: 节点更新
  z_n^(k+1) = ReLU(z_n^(k) + h2n_out) * m_n
```

**参数分析**：
- 原始 HyperIMTS: Linear(3D, D) = 3D² 参数
- 新 QMF: 4 × Linear(D, D/4) + QuatLinear(D,D) = 4×(D×D/4) + 4×(D/4)² = D² + D²/4 = 5D²/4 参数
- 参数减少约 58%，但通过 Hamilton 乘积的结构约束，编码了所有成对交互

**与旧设计的区别**：
- 旧：h2n_out = Linear(3D,D) + α × QuatLinear(Linear(3D,D))  [quaternion 是残差]
- 新：h2n_out = QuatLinear([proj_R | proj_I | proj_J | event_K])  [quaternion 是主路径]
- 旧：移除 quaternion → h2n_out = Linear(3D,D)（不变）
- 新：移除 quaternion → 需要用 Linear(D,D) 替代，但失去了结构化交叉交互 → 有意义的 ablation

### 3.5 Ablation 设计

新设计的 ablation 不再是"关掉一个可选模块"，而是"用简单替代品替换一个核心组件"：

| 消融变体 | 修改 | 预期效果 |
|---------|------|---------|
| Full SQHyper | 完整模型 | 最佳 |
| w/o SGI (→ binary incidence) | g_n = 1 for all n, e_n = 0 | 超图退化为静态，四元数丢失 K 分量 |
| w/o QMF (→ flat linear) | 用 Linear(3D+D/4, D) 替代四元数融合 | 丢失结构化交叉交互 |
| w/o both | g_n = 1, Linear(3D, D) | 退化为 HyperIMTS |

**为什么这比之前的 ablation 更有说服力**：
- "w/o SGI" 移除了数据自适应拓扑，这是一个结构性降级，不是"关掉一个没用的开关"
- "w/o QMF" 移除了结构化融合，这改变了模型的信息处理方式，不是"移除一个小残差"
- "w/o both" 直接退化为 HyperIMTS → 清晰的 baseline 对比

### 3.6 Self-Attention 的处理

注意 HyperIMTS 在 h2n 之前有一个 node_self_update (self-attention)：

```python
obs_for_h2n = self.node_self_update[i](
    observation_nodes,
    cat([temporal_gathered, variable_gathered, observation_nodes], -1),
    mask)
```

在新设计中，self-attention 可以保留，但其输出作为 q_R 的输入而非直接进入 h2n：
- f_self = SelfAttn(z_n, [τ,ν,z])  → q_R = W_R(f_self)

这样 self-attention 仍然提供节点级的上下文更新，但最终融合通过四元数完成。

### 3.7 初始化策略

- SGI: 膜电位的权重零初始化 → g_n ≈ 0.5（均匀权重，非极端值）
- QMF 投影: 标准 Xavier 初始化
- QuatLinear: R=I, I=J=K=0 → 初始时四元数变换为恒等
- 在训练初期，模型接近 "均匀加权超图 + 恒等四元数融合"，
  随着训练推进，spike 学会区分观测，quaternion 学会交叉交互

---

## Phase 4: 待查新颖性

需要检查以下方向是否已有先例：
1. 软/加权关联矩阵在超图神经网络中的应用
2. 四元数在图/超图消息传递中的应用
3. 脉冲神经元用于生成数据依赖的图权重
4. 多源结构化融合（特别是用代数结构约束的方式）
