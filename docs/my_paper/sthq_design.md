# STHQ-Net: Spike-Triggered Hyperedge with Quaternion States

A native hypergraph model for IMTS forecasting, designed from first principles.
Spike, quaternion, and hypergraph each play a non-trivial, non-substitutable role.

---

## 一. 设计原则

1. **每个组件必须有清晰、不可替代的角色**——不能"加一个 spike 分支""加一个 quaternion 调制"这种打补丁的设计
2. **不能套用现有 IMTS 模型的结构**——不来自 HyperIMTS 的 (T+V) 超边、不来自 attention 的 query/key/value 范式、不来自 codebook routing
3. **Hypergraph 是核心，不是装饰**——必须真有"超边"作为一等公民的结构
4. **能稳定训练**——避免 PE-RQH 那种没有 inductive bias 的从零学

---

## 二. 各组件的不可替代角色

### 2.1 Spike (脉冲) 的角色

**Spike 是一个 cell 的"显著性强度"——只有显著的 cell 才有资格参与超边**

不是把 spike 当 gate 加在某个 attention 上，而是：
- 每个观测 cell 经过 spike encoder，产生 spike intensity ∈ [0, 1]
- spike intensity = 0 的 cell 在超边中**没有任何贡献**（spike 直接决定超边成员权重）
- spike intensity 高的 cell **完全决定**超边状态

这个角色是不可替代的，因为：
- IMTS 的核心是稀疏不规则观测，spike 给"哪些观测真正关键"提供原生表达
- mask 只是 binary 的"有/没有"，spike 是 continuous 的"多重要"
- 不是单独 gate 一个 K/V 投影，而是**门控整个超边的形成**

### 2.2 Quaternion (四元数) 的角色

**Quaternion 是 cell 状态的 4 个 typed 组件**：

```
q = (q_r, q_i, q_j, q_k)
  q_r:  value 通道 (实部)        — 该观测的实测值的高维表征
  q_i:  temporal 通道 (i 部)     — 时间相对位置的高维表征
  q_j:  variable 通道 (j 部)     — 变量身份的高维表征
  q_k:  spike-driven 通道 (k 部) — 脉冲强度调制的隐状态
```

每个超边 e 也是一个 quaternion h_e。

**Hamilton product `h_e ⊗ q_i`** 不是装饰，而是**自然的状态组合算子**：
- 它让超边的 4 个 typed 组件分别对 cell 的 4 个组件做 cross-mixing
- 不是简单的 concat + linear（会破坏 typed 结构）
- 不是 elementwise product（不能跨组件传递信息）
- Hamilton 的 cyclic 结构正好让 (time, var, value, spike) 互相融合

参数效率：quaternion linear 有 1/4 的参数量 vs flat linear，IMTS 数据集小，这个 efficiency 真有用。

### 2.3 Hypergraph 的角色

**超边是 cells 的 emergent grouping**——但不是 SC-PERQH 那种"全局学到的 codes"，也不是 HyperIMTS 那种"每个时间一个超边"。

**两种类型的超边，由 spike 触发动态生成**：

#### 2.3.1 Temporal Coincidence Hyperedge (时序并发超边)

K_t 个时间锚点 (τ_k, ω_k)：
- τ_k: 学习的时间中心
- ω_k: 学习的时间带宽

cell i 在超边 k 的成员权重：

```
m_temp(i, k) = spike_i · exp(-(t_i - τ_k)² / 2ω_k²)
```

多尺度：每层用不同的 ω 范围，让浅层捕捉短时并发，深层捕捉长时并发。

**为什么不固定每个时间一个超边（HyperIMTS 风格）？**
- HyperIMTS 是 hard binning，每个观测只属于一个时间槽
- 我们是 soft 高斯窗口，相邻时间的观测**自然共享部分超边**
- (τ, ω) 是学的——数据决定时间锚点应该放在哪、窗口多宽

#### 2.3.2 Variable Affinity Hyperedge (变量亲和超边)

V × K_v 的可学习矩阵 A：
- A[v, k] = 变量 v 对超边 k 的亲和度
- Softmax 归一化后作为软分配

cell i 在变量超边 k 的成员权重：

```
m_var(i, k) = spike_i · softmax(A[var_i, k])
```

**为什么不固定每个变量一个超边？**
- 变量之间有强相关（ICU 中 BP/HR/Temp 应该共享超边）
- A 是学的，让模型发现哪些变量应该聚在一起
- K_v 通常 < V，所以这是**变量降维聚类**

### 2.4 三者的有机组合

**Spike 不是给已有结构加一个 gate**——它是超边成员权重的**乘性核心因子**。没有 spike 这一项，超边就退化为传统 hypergraph。

**Quaternion 不是装饰**——它是 cell 状态的 typed 表达，每个组件对应一个语义维度。Hamilton product 让超边和 cell 的状态自然组合。

**Hypergraph 不是 attention 替代品**——它是超边作为 explicit 中间结构，cells 间通过超边间接但**结构化**地通信。

---

## 三. 完整架构

### 3.1 输入与编码

```
输入: B 个样本，每个有 N 个 cells，每个 cell 有 (value, time_norm, var_id, mask)

Spike Encoder (产生 spike intensity ∈ [0, 1]):
  features = [value, time_norm, var_emb(var_id), mask]   (R^4)
  spike_intensity = mask · σ(MLP(features))               (R)
  
Quaternion State Encoder (产生 4-typed cell state):
  q_r = Linear_value(value · mask)                        (R^Q)
  q_i = sin(time_norm · ω_basis)                          (R^Q, ω_basis 是 log-spaced 频率)
  q_j = var_emb(var_id)                                   (R^Q)
  q_k = spike_intensity · Linear_spike(features)          (R^Q)
  q = QuatLinear([q_r; q_i; q_j; q_k])                    (R^4Q)
```

### 3.2 单层 STHQ Layer

输入: cell 状态 q ∈ R^(B,N,4Q)，spike 强度 s ∈ R^(B,N)，时间 t ∈ R^(B,N)，变量 var ∈ Z^(B,N)，mask ∈ R^(B,N)

```
# 1. 计算超边成员权重 (B,N,K_t) 和 (B,N,K_v)
m_temp[b,i,k] = s[b,i] · exp(-(t[b,i] - τ_k)² / 2ω_k²)
m_var[b,i,k]  = s[b,i] · softmax(A[var[b,i], :])[k]

# 2. 聚合 cell -> 超边 (B,K,4Q)
h_temp[b,k] = Σ_i m_temp[b,i,k] · q[b,i] / (Σ_i m_temp[b,i,k] + ε)
h_var[b,k]  = Σ_i m_var[b,i,k]  · q[b,i] / (Σ_i m_var[b,i,k]  + ε)

# 3. 超边间交互 (可选，让超边互相通信)
H = concat(h_temp, h_var)                              (B, K_t+K_v, 4Q)
H' = MultiHeadQuatAttention(H)                         (quaternion-aware self-attention)
h_temp', h_var' = split(H')

# 4. 超边 -> cell：Hamilton product 形成消息
msg_temp[b,i] = Σ_k norm_m_temp[b,i,k] · (h_temp'[b,k] ⊗ q[b,i])
msg_var[b,i]  = Σ_k norm_m_var[b,i,k]  · (h_var'[b,k]  ⊗ q[b,i])
  其中 norm_m 是 cell 维度归一化的成员权重

# 5. 残差更新
q' = QuatLayerNorm(q + QuatLinear(msg_temp + msg_var))
```

### 3.3 多层堆叠

L 个 STHQ Layer 串联，**每层的 (τ_k, ω_k) 范围不同**：
- 浅层：短时窗口（细粒度并发）
- 深层：长时窗口（粗粒度全局结构）

具体：第 l 层的 ω_k 初始化为 `ω_l_init = ω_min · (ω_max/ω_min)^(l/L)`（log-spaced）

### 3.4 解码

```
# Cell-level prediction
q_final ∈ R^(B,N,4Q)
y_pred = MLP_decoder(q_final).squeeze(-1)              (B, N)
```

---

## 四. 关键参数

| 参数 | 说明 | 默认 |
|------|------|------|
| Q | 每个四元数组件的维度 (D=4Q) | 64 |
| L | STHQ 层数 | 3 |
| K_t | 每层时间锚点数 | 16-64 (按数据集) |
| K_v | 每层变量超边数 | min(V, 16-32) |
| ω_min, ω_max | 时间窗口范围 (相对 seq_len) | 0.02, 0.5 |
| τ_init | 时间锚点初始化 | linspace(0, 1, K_t) |
| A_init | 变量亲和初始化 | 单位矩阵扰动 (前 V 个 code 偏向对应变量) |

---

## 五. 与现有模型的关键差异

| 维度 | HyperIMTS | PE-RQH/SC-PERQH | **STHQ-Net** |
|------|-----------|-----------------|--------------|
| 超边来源 | 固定 (每时间/变量一个) | 学习 D 维 codes | **(τ,ω) 时间锚 + V×K_v 变量亲和** |
| 超边数 | T+V (>1000) | K (32-64) | **K_t+K_v (16-128)** |
| Cell 状态 | 标量 D-dim | quaternion 4Q | **typed quaternion (value,time,var,spike) 4Q** |
| Spike 角色 | 无 | 无（仅 PE-RQH 名字带"event"） | **超边成员权重核心因子** |
| Quaternion 角色 | 无 | Hamilton routing | **typed 状态 + Hamilton 状态组合** |
| 信息流 | cell ↔ 超边 (attention) | cell → code → cell (有损) | **cell → 超边 → cell (Hamilton 复合)** |
| 多尺度 | 无 | 无 | **每层不同 ω 范围** |

---

## 六. 不会重蹈覆辙的设计点

1. **超边成员权重有解析形式**（高斯核 + softmax），不是从零学的离散分配 → 不会出现 PE-RQH 的随机簇问题
2. **Spike intensity 直接乘进权重**，不是 gate 在某个分支上 → spike 真正发挥作用
3. **(τ, ω) 是 2D 学习参数**（不是 D 维 code），超参少、可解释、易稳定
4. **K_v 与 V 同尺度**（不像 SC-PERQH 中 K_var=128 of V=89 失衡）
5. **Hamilton product 在 typed quaternion 上有语义**（不是无意义的几何操作）

---

## 七. 实现估算

- 模型参数量: 与 SQHyper 同级 (~100K-1M)
- 单步复杂度: O(N · (K_t + K_v) · 4Q) — 与 SC-PERQH 相同量级
- 内存峰值: O(B · N · 4Q + B · (K_t+K_v) · 4Q)
- 实现工作量: ~600 行 PyTorch
- Smoke test 时间: 5 分钟
- 4 数据集首轮训练: ~2 小时（与 PE-RQH 同等）

---

## 八. 待用户确认的设计决策

1. **是否用 Hamilton product 还是更简单的 channel-mix**？
   - Hamilton: 数学优雅，参数效率高，但语义抽象
   - Channel-mix: 灵活，可学习任意组合，但参数多
   - **建议**: 用 Hamilton（参数效率对小数据更友好）

2. **超边间是否要加 self-attention**？
   - 加：超边间互相 refine（K^2 attention，K 小所以便宜）
   - 不加：保持简洁，让 cell-超边-cell 直接通信
   - **建议**: 第一层不加，后面层加（浅层结构化、深层混合）

3. **Spike encoder 的网络深度**？
   - 浅 (1 层 MLP)：稳定，但表达力有限
   - 深 (2-3 层)：表达力强，但训练不稳
   - **建议**: 1 层 + sigmoid，并加少量 weight decay

4. **(τ, ω) 是否在不同 sample 间共享**？
   - 共享：模型参数，所有样本同一组锚点
   - 不共享：每个样本动态生成
   - **建议**: 共享（更稳定，且大多数 IMTS 数据集中时间结构是 dataset-level 的）

请用户确认设计后，我开始实现。
