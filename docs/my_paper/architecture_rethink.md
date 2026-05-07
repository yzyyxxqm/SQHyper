# 架构反思：为什么 PE-RQH 和 SC-PERQH 都难以提升？

**时间**: 2026-05-07 14:00 UTC（停掉训练后）
**结论**: 不是超参问题，不是初始化问题，是**架构范式选错了**。

---

## 一. 两个版本共享的根本错误

| 阶段 | 输入 → 输出 |
|------|-------------|
| Encode | (value, time, var, mask) → q ∈ R^D |
| Route | q · code → top-k → 加权 Hamilton product → message |
| Update | q ← q + message |
| Decode | q → ŷ |

**关键路径**：cell → code（瓶颈）→ cell

### 1.1 Codebook routing 的三个根本性缺陷

#### (a) Codebook 是全局共享的，不是样本特定的
- Attention 的精髓：**每个样本** 计算自己的 query-key 相关性
- Codebook 的本质：**所有样本** 共享一组固定原型
- 对 IMTS 这种"同一个变量在不同 sample 中模式高度异构"的任务，全局 codes 抹平了 sample 级特异性

#### (b) Cell → code → cell 是**双向有损压缩**
- 信息从 N 个 cells 压缩到 K 个 codes（典型 K << N）
- 再从 K 个 codes 解压回 N 个 cells
- 这个 bottleneck 必然丢信息，除非 K 接近 N
- 但 K 接近 N 时 codebook 就退化成"每个 cell 一个 code"，毫无意义

#### (c) Hamilton product 没有语义对应
- Hamilton product `p ⊗ q`：把 q 在 p 定义的 4D 旋转下变换
- 对 IMTS 而言："时间-变量-值"三元组的旋转是什么意思？
- 这是一个**有数学结构但无任务语义**的操作
- 实验证据：在所有数据集上，quaternion 结构没有带来明显增益

### 1.2 实测证据

| 数据集 | PE-RQH MSE | SC-PERQH val_loss (early) | SQHyper baseline | 改善幅度 |
|--------|------------|---------------------------|------------------|----------|
| USHCN | 0.218 ± 0.030 | 0.197 (val, 1 iter) | 0.191 | 微弱 |
| P12 | 0.40 (val 卡) | 0.41 (val, 36min) | 0.301 | 仍差 |
| MIMIC | 0.81 (val 卡) | 0.82 (val, 36min) | 0.420 | 仍差 |
| HumanActivity | ~0.32 (val 卡) | 0.17 (val, 18min) | ~0.017 | 距离巨大 |

即便 SC-PERQH 比 PE-RQH 好一点，跟 SQHyper baseline 仍有显著距离。**所有数据集**都吃力。

---

## 二. 真正成功的 IMTS 架构有什么共同点？

### 2.1 SQHyper / HyperIMTS（success）

- **直接 cell-cell 通信**：通过共享 hyperedge 的 attention
- **结构化但不死板**：incidence matrix 强制按时间/变量分组，但组内 attention 是输入相关的
- **轻量**：每个 hyperedge 维持一个 D-dim 状态，attention 为 sparse

### 2.2 mTAN / Raindrop / Warpformer（success）

- 显式时间编码（连续位置编码）
- Cell-to-cell attention 或 cell-to-time-grid attention
- 专门处理 missing 模式

### 2.3 共同点

**所有成功的架构都让 cells 之间直接（或几乎直接）地通过 attention 通信。没有任何成功架构使用 codebook 作为信息瓶颈**。

---

## 三. 三个真正不同的方向

下面是**不依赖 HyperIMTS 结构、也不重蹈 codebook routing 错误**的新范式。

### 方向 A：**LCAT-Net (Learned Cluster-Attention Time-series)**

**核心**：保留 codebook 的"动态结构发现"创新，但 codes 不参与信息流——只用来 **分组**，组内做直接 attention。

```
Encode: cell -> q_i ∈ R^D

Group assignment (top-k soft clustering):
  for each cell i:
    g_t(i) = top_k(q_i · time_codes)   # K_t time clusters
    g_v(i) = top_k(q_i · var_codes)    # K_v var clusters

Within-cluster attention:
  for each cluster c in time clusters:
    members = {i : g_t(i) = c}
    msg_t[i] = SoftmaxAttention(q_i, {q_j : j ∈ members})
  similar for var clusters

Combine: q_i' = q_i + MLP([msg_t[i] || msg_v[i]])
Decode: y_i = MLP(q_i')
```

**为什么可能突破**：
- 同一个 cluster 内的 cells **直接互看**（不再通过 code 中转）
- Cluster 是**学出来**的，不是 HyperIMTS 那样固定为时间/变量
- 数学上可解释：cluster ≈ 学到的 hyperedge

**与 HyperIMTS 的差异**：HyperIMTS 用固定 (T+V) 个 hyperedge，LCAT-Net 用 (K_t+K_v) 个 emergent cluster，且 cells 在多个 cluster 中（top-k soft membership）。

**预期效果**：
- 直接 attention 表达力远高于 codebook
- 学习的 cluster 比固定 hyperedge 更适应数据特性
- 不丢失 sample-specific 信息

---

### 方向 B：**ContiGraph (Continuous-Time Cell Graph)**

**核心**：放弃离散 cluster/code，改用 **连续时间核** 在 cell graph 上做信息传播。

```
Encode: cell i has feature q_i, time t_i, var v_i

Edge weight (continuous, sample-specific):
  w_ij = exp(-α |t_i - t_j|^2) · σ(MLP([v_i, v_j]))
  # decay in time, learned variable affinity

Graph attention with edge bias:
  msg_i = Σ_j w_ij · (V_j) / Σ_j w_ij
  q_i' = q_i + MLP(msg_i)

Multi-layer: stack with different α values for multi-scale
```

**为什么可能突破**：
- 完全连续，没有任何离散瓶颈
- α 可学习，每层学不同时间尺度
- 变量 affinity 直接学成 V × V 矩阵
- 完美处理不规则时间（连续核天然适应）

**与 HyperIMTS 的差异**：HyperIMTS 用离散 hyperedge，ContiGraph 用连续核。完全不同的数学。

**与 attention 的差异**：标准 attention 没有时间/变量的 inductive bias，ContiGraph 通过核函数把这两个 prior 显式注入。

**风险**：O(N²) 复杂度对长序列（HumanActivity N=36k）有压力。需要稀疏近似（kNN 或 random feature）。

---

### 方向 C：**TrajLatent (Per-Variable Latent Trajectory)**

**核心**：抛弃 cell-level 表示，改用 **变量级潜在轨迹** 的视角。每个变量有一个连续函数 f_v(t)，观测是 f_v 的样本，预测是 f_v 在新时间点的取值。

```
Per-variable trajectory token: z_v ∈ R^D for v = 1..V

Update from observations (each obs is irregular sample of f_v):
  z_v ← Attention(query=z_v, key=value=encode(t_i, x_i for obs of var v))

Cross-variable coupling:
  z_v ← z_v + Attention(query=z_v, key=value={z_u for u != v})

Decode: y(t, v) = MLP(z_v, t_emb)  
  # decoder takes trajectory token + query time
```

**为什么可能突破**：
- 视角对：IMTS 本质上就是有限观测下的潜在连续轨迹估计
- 跨变量耦合直接显式
- 解码是函数评估，对任意 query time 都能预测
- V 比 N 小一两个数量级（V=89 << N=6675），效率高

**与 HyperIMTS 的差异**：HyperIMTS 把 cell 当节点，TrajLatent 把变量整体当节点。粒度完全不同。

**风险**：每个变量只一个 token，可能容量不够。可以扩展为多 token per 变量（hierarchical）。

---

## 四. 三方向对比

| 维度 | LCAT-Net (A) | ContiGraph (B) | TrajLatent (C) |
|------|--------------|----------------|----------------|
| 核心创新 | 动态学习 cluster + 组内 attention | 连续时间核 + sample-dependent edges | 变量级轨迹 token + 时间解码器 |
| 与 HyperIMTS 距离 | 中等（保留 hyperedge 概念但完全可学） | 远（不再有 hyperedge） | 远（视角完全不同） |
| 复杂度 | O(N · max_cluster_size) | O(N²) 或 O(N · k) sparse | O(V²) + O(V · obs_per_var) |
| 适合数据 | 全部 | 中-短序列 | 高 V，sparse obs |
| 实现难度 | 中 | 中 | 较高 |
| 风险 | cluster 学习可能不稳 | 长序列压力 | V 小数据集容量不足 |

---

## 五. 我的判断

**最有希望的是方向 A (LCAT-Net)**：

1. 保留了"动态结构发现"这个研究亮点（top-k soft clustering）
2. 用直接 attention 替换有损 codebook 路由——突破了 PE-RQH/SC-PERQH 的核心瓶颈
3. 复杂度可控
4. 在所有数据集上都有合理表达力上限

但如果用户希望最大程度跳出当前圈子，**方向 B (ContiGraph)** 才是真正的"另起炉灶"——它根本不用任何离散结构。

方向 C 优雅但可能不够通用。

---

## 六. 问题

**请用户决定方向**：
- A: LCAT-Net（建议）— 保留 cluster 创新但改用直接 attention
- B: ContiGraph — 完全连续，无离散结构
- C: TrajLatent — 变量级轨迹视角
- D: 其他想法（用户提议）

我可以为选定的方向写更详细的设计文档，再实现。
