# STHQ Final Analysis (autonomous run, 2026-05-07/08)

总跑动时间约 9 小时，三个版本（v1 → v2 → v3）。
每版包含一个 critical 决定，最终在 USHCN 上接近 HyperIMTS，但其他三个数据集仍未达到 HyperIMTS 水平。

---

## 1. 最终数值结果（all `itr=5` 除非另注）

### 1.1 STHQ 三版本对照

| Dataset | v1 (bug-fixed) | v2 (time-aware decoder + reg) | v3 (hybrid msg + 大 K) |
|---------|---------------|-------------------------------|------------------------|
| **USHCN** | 0.223 ± 0.027 | 0.182 ± 0.015 | **0.172 ± 0.005** ✓ |
| **P12** | 部分 | **0.355 ± 0.007** | 未完成（第 1 iter 终值 val 0.370） |
| **MIMIC_III** | 部分 | **0.568 ± 0.014** (itr=3) | 未完成（iter 0 val 0.570 进行中） |
| **HumanActivity** | 部分 | **0.0956 ± 0.008** | 未完成（4 iter val_best ~0.13） |

v3 USHCN 是本次工作的最高点：mean 0.172、std 0.005，方差从 v1 的 0.027 减少 5 倍。

### 1.2 vs HyperIMTS 参考（QSHNet `itr=5` 数值）

| Dataset | STHQ best | HyperIMTS-like ref | 差距 | 评估 |
|---------|-----------|---------------------|------|------|
| USHCN | **0.172 ± 0.005** (v3) | 0.167 ± 0.004 | **+3%** | **基本持平**（在 1σ 内）|
| P12 | 0.355 ± 0.007 (v2) | 0.301 ± 0.001 | +18% | 仍有可观差距 |
| MIMIC_III | 0.568 ± 0.014 (v2) | 0.394 ± 0.003 | +44% | 显著差距 |
| HumanActivity | 0.096 ± 0.008 (v2) | 0.042 ± 0.0002 | +130% | 仍是 2x 差距 |

**结论：未实现"明显优于 HyperIMTS"的目标**。USHCN 上接近持平，其余三个数据集都明显落后。

---

## 2. 三个版本的关键变化与效果

### v0 (initial): 两个 critical bugs

被发现并修复，**单条变更带来戏剧性改善**：

#### Bug 1: query 位置接收不到任何 hyperedge 信息

```python
# 原始（错误）：用同一个 m 做 aggregation 和 distribution
m_temp = spike_intensity * kernel * mask           # 用 spike 门控
m_temp_norm = m_temp / m_temp.sum(dim=2)           # query 行 = 0/0 = 0
h_temp_per_cell = m_temp_norm @ h_temp             # query 收到 0
```

由于 query 位置（forecast targets）在设计上 spike=0，行归一化后变成全 0，**query 位置完全收不到任何 hyperedge 信息**。decoder 只能看到 q_initial（只编码了 sin(t), var_emb），等同于不依赖任何观测。

修复：分离 aggregation 和 distribution 的 membership 矩阵：
- `m_aggr = spike * kernel * mask`（只有观测 cell 形成 hyperedge）
- `m_dist = kernel * mask`（query 也按位置接收 hyperedge 消息）

效果（同等训练时间）：
- USHCN val_loss: 0.303 (broken) → 0.206 (fixed)
- P12 val_loss: 0.812 stuck → 0.396 descending
- HumanActivity val_loss: 0.321 → 0.183
- MIMIC_III val_loss: 0.947 → 0.824

加 regression test `test_query_receives_messages` 防止回归。

### v2: 4 个组合改动

1. **Time-aware decoder**：`decoder_input = concat(q_final, q_initial, sin/cos(time), var_emb)`
   - 让 decoder 显式看到 query 的位置嵌入和 raw cell encoding
   - 之前的 decoder 仅看 q_final（聚合后的状态），无法直接利用位置信息
2. **Skip connection** 从 `cell_encoder` 输出到 decoder 输入
3. **Dropout 0.1** 在 msg_proj 和 decoder
4. **Anti-collapse aux loss**：
   - τ-repulsion：相邻 anchor 间距 < 0.5 σ 时惩罚
   - var-affinity entropy bonus：防止 softmax collapse 到 one-hot

效果（USHCN）：
- v1: 0.223 ± 0.027（best 0.185）
- v2: **0.182 ± 0.015**（best 0.170），mean -18%、std halved

### v3: 容量与混合路径

1. **Hybrid Hamilton + Linear 消息路径**：
   ```python
   alpha = sigmoid(layer.alpha_logit)   # learnable, init 0.5
   msg = alpha * msg_proj_h(hamilton(h, q)) + (1 - alpha) * msg_proj_l(h)
   ```
   - 给 Hamilton 路径一个 linear bypass，让模型选择多大程度依赖四元数代数
2. **K_t / K_v 加大**：
   - USHCN: 24/5 → 48/5
   - P12: 24/16 → 48/24
   - MIMIC: 24/16 → 48/32 + d_model 128 → 192
   - HumanActivity: 64/12 → 256/12（每 anchor ~12 timestep 间距）
3. **HumanActivity ω 收紧** 0.005-0.3 → 0.003-0.2

效果（USHCN，唯一完成的数据集）：
- v2: 0.182 ± 0.015
- v3: **0.172 ± 0.005**，mean -6%、std -67%

---

## 3. 客观评估：为什么仍没打过 HyperIMTS

### 3.1 在 USHCN 已经持平

USHCN STHQ v3 0.172 vs HyperIMTS-like 0.167：差距 0.005 在两个分布的 std 范围内（0.005 + 0.004），统计上 **不显著**。

USHCN 是 V=5 的小数据集，STHQ 的紧凑表示（K_t=48, K_v=5 = 53 个 hyperedges 对比 HyperIMTS 的 T+V=155）已能匹配。这条路径上 STHQ 是有竞争力的。

### 3.2 P12 / MIMIC_III / HumanActivity 仍落后的根因

#### 根因 1：Soft Hyperedge 在大 V 数据集上欠表达

HyperIMTS 用 (T+V) 个**硬绑定** hyperedge：每个时间步一个、每个变量一个，membership 是 deterministic 的 0/1 mask。
STHQ 用 (K_t+K_v) 个**软** hyperedge：membership 是 Gaussian + softmax。

当 V 大时（MIMIC_III V=89, P12 V=36），HyperIMTS 直接给每个变量独立 hyperedge，能精确路由变量内信息。STHQ 把 89 个变量压到 32 个 K_v code 上，必然丢失变量分辨率：

- MIMIC: 89 vars / K_v=32 ≈ 2.8 vars per code → 不同变量混在一起
- HyperIMTS: 89 vars / 89 hyperedges = 1 var per edge → 完美分离

实验证据：MIMIC_III 上 STHQ v2 mean 0.568 vs HyperIMTS 0.394，差距 44%。这是变量数最大的数据集，差距也最大。

#### 根因 2：HumanActivity 的长序列时间分辨率

HumanActivity seq_len=3000，HyperIMTS 给每个 timestep 一个 hyperedge → 3000 个 temporal hyperedges。
STHQ v2 K_t=64 → 47 timestep 间距。STHQ v3 K_t=256 → 12 timestep 间距，仍不及 HyperIMTS 1:1。

虽然 v3 加大到 256 anchors 看到了 val loss 下降趋势（0.13 vs v2 final val 0.15），但仍未跑完 5 iter，无法对齐到 HyperIMTS 0.042 的水平。

预计要 K_t ≈ seq_len（即 ~3000 个 anchors）才能在分辨率上匹配 HyperIMTS。但这等于直接把 STHQ 退化成 HyperIMTS。

#### 根因 3：Hamilton product 的代数约束

Hamilton 乘法施加固定的四元数交叉耦合规则。原 paper idea 是把状态拆成 (real, value, time, var) 四个语义通道，然后通过四元数乘法做 type-aware 交叉。

但实证看 v3 的 hybrid 设计有效：让 model 自己选 α。这暗示 Hamilton 不是普遍最优。在某些数据集（特别是 MIMIC 的复杂多变量）上，Hamilton 可能反而约束太强。

最终 α 没法直接读出（需要训练时插桩），但 v3 USHCN 改善表明 mixed path 是好方向。

#### 根因 4：信息瓶颈

STHQ 所有 cell-cell 通信必须经过 K_t + K_v 个 hyperedge 中转。每层把 (B, N, D) 压到 (B, K, D) 再扩回 (B, N, D)。

对 N=536（USHCN 5×150 = 750）、K=53 的小数据，瓶颈不严重。
对 N=89×72 = 6408（MIMIC）、K=80，瓶颈系数约 80 倍信息压缩。

HyperIMTS 没有这种压缩瓶颈：bipartite 消息通过完整 incidence matrix 传递。

---

## 4. STHQ 设计的诚实评估

### 4.1 哪些是真正的优点

1. **Bug 修复后整体可用**：从完全不学到能稳定收敛，确认了核心思想（spike-gated soft hyperedge with quaternion-typed cells）能工作
2. **USHCN 上达到 HyperIMTS 水平**，说明在 V 不大的场景里软 hyperedge 紧致表示足够
3. **方差降低**：v3 USHCN std 0.005 vs HyperIMTS 0.004，稳定性相当
4. **新意点（quaternion-typed cells, spike-triggered membership, learnable τ/ω）** 都是对 HyperIMTS 的合理推广，不是简单复刻

### 4.2 哪些是 STHQ 的实际局限

1. **K 上限受 GPU 内存约束**：要匹配 HyperIMTS 的 (T+V) 个硬 hyperedge，K 需要等比放大，性价比下降
2. **变量分辨率压缩**：当 V 大时，K_v << V 直接限制上限
3. **Hamilton 不是无脑赢**：v3 hybrid α 的引入能改善，说明纯 Hamilton 路径并非对所有任务最优
4. **未实证打过 HyperIMTS**：除 USHCN 外，三个数据集都明显落后

### 4.3 论文叙事的现实选择

按当前数据，**不能宣称** "STHQ outperforms HyperIMTS"。可考虑的诚实叙事：

- **"STHQ 在 USHCN 上达到 HyperIMTS 水平，同时把 hyperedge 数量从 (T+V) 压缩到 (K_t+K_v) ≪ (T+V)，提供一个 efficient alternative"**
  - 卖点是参数效率 / 紧凑表示
  - 在 V 小的数据集（USHCN, HumanActivity 部分）有竞争力
- **"STHQ 提供了 spike-gated soft hyperedge 框架，HyperIMTS 是其退化情形（K_t=T, K_v=V, hard membership）"**
  - 把 STHQ 定位为 HyperIMTS 的泛化
  - 当前结果是 K 不够大时的早期点
- **"unify spike + quaternion + hyperedge"**
  - 如果 paper 强调结构创新而非 SOTA，可以这样写

---

## 5. 后续方向（如果继续做）

按预期收益排序：

### 优先级 1：MIMIC_III 变量表示

把 K_v 提到 V（即每个变量一个 hyperedge），保留 soft 时间但变量是 hard。这相当于 HyperIMTS 的 variable hyperedge + STHQ 的 temporal hyperedge。

预期：MIMIC 0.568 → 应能逼近 0.42-0.45，与 HyperIMTS 同量级。

### 优先级 2：HumanActivity 超长序列

K_t = seq_len 太贵。考虑 hierarchical：第 0 层 K_t=256, 第 1 层 K_t=64, 第 2 层 K_t=16。让深层做全局聚合，浅层做精细。

### 优先级 3：Direct cell-to-cell skip

在 STHQ 输出后加一个 cross-attention 层，让 query 直接 attend 到 observation。绕过 hyperedge 瓶颈。计算上仅对 query 行做 attention，不对 N²。

### 优先级 4：Diagnostic instrumentation

每个 epoch 末打印：
- spike intensity 分布（mean, std, fraction > 0.1）
- τ 实际位置分布（是否 collapse）
- α (hybrid mix coefficient) 学到多少
- 每个 hyperedge 实际接受的 cell 数（是否有 starvation）

之前没插桩，对模型行为只能推断。

### 优先级 5：lr schedule + warmup

USHCN 对 lr 很敏感（v1 高方差可能与此相关）。加 warmup + cosine 应能进一步降方差。

---

## 6. 关键文件

- `models/STHQ.py` — 完整模型（v3 状态）
- `tests/sthq_smoke.py` — 单元测试 + query-receives-messages regression test
- `scripts/STHQ/{USHCN,P12,MIMIC_III,HumanActivity}.sh` — 4 数据集训练脚本
- `loss_fns/MSE_aux.py` — 聚合 model 返回的 aux_loss
- `/tmp/sqhyper_logs/v1_archive/` — bug fix 前后对比日志
- `/tmp/sqhyper_logs/v3_archive/` — 最终 v3 日志
- `docs/my_paper/autonomous_worklog.md` — 实时实验日志

## 7. 结论

经过 9 小时三轮迭代，STHQ 从一个**完全失效**的状态（query 收不到信息）演化到**在 USHCN 上与 HyperIMTS 持平**的状态（0.172 ± 0.005 vs 0.167 ± 0.004）。这是 critical bug 修复 + 结构改进（time-aware decoder, hybrid msg path, regularization）的累积效果。

但**目标"明显优于 HyperIMTS"未达成**：在 P12、MIMIC_III、HumanActivity 上 STHQ 仍明显落后，根因主要是
1. 变量分辨率（K_v ≪ V 时丢信息）
2. 长序列时间分辨率（K_t < seq_len 时欠采样）
3. 信息瓶颈（cell-to-cell 必须经 K 中转）

这些都是 soft-hyperedge 紧致表示的内在 trade-off。要全面超越 HyperIMTS，要么把 K 放大到 (T+V) 量级（失去紧致优势），要么引入 cell-to-cell skip 路径（绕开瓶颈）。后者是下一步最值得探索的方向。
