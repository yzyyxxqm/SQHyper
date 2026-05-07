# PE-RQH 第一轮全数据集深度分析

**收集时间**: 2026-05-07 13:00 UTC（启动后 ~80 分钟）
**对照基线**: SQHyper 同环境结果（见 `sqhyper_results.md`）

---

## 一. 运行时间分析

### 1.1 关键发现：**问题不是“per-epoch 慢”**，而是**“需要更多 epoch 才能收敛”**

| 数据集 | SQHyper s/epoch | PE-RQH s/epoch | per-epoch 比 | 备注 |
|--------|-----------------|----------------|--------------|------|
| USHCN | 38–40 s | 33–38 s | **0.95×** | 反而略快 |
| P12 | 44–48 s | 100–110 s | **2.3×** | 显著慢 |
| MIMIC_III | 161–222 s | 175–182 s | **1.0–1.1×** | 近乎相同 |
| HumanActivity | n/a（log 不全） | 22–27 s | n/a | |

### 1.2 总 iter 时间（每 iter = 训练到 early stop + 测试）

| 数据集 | SQHyper iter 平均 | PE-RQH iter 平均 | 比 |
|--------|-------------------|------------------|-----|
| USHCN | ~13 min | ~11 min | **0.85×** |
| P12 | ~17 min | **>80 min** (iter 0 仍未结束) | ≥**5×** |
| MIMIC_III | ~30 min（含一次 67 min 离群） | **>80 min** (iter 0 仍未结束) | ≥**3×** |
| HumanActivity | n/a | 11–28 min | n/a |

### 1.3 真正的慢源：**收敛轨迹**

P12 iter 0 val_loss 轨迹（PE-RQH）：
```
0.51 → 0.40（前 5 min, 快下降）
→ 0.40（30 min 内仅波动，不下降）
→ 0.40 后停滞，触发 early stop（~80 min 处）
```

SQHyper P12 iter 0：约 25 epoch（17 min）即达到 val_loss ≈ 0.30 → MSE 0.301。

**结论：PE-RQH 在 P12 上 *根本无法把 val_loss 降到 SQHyper 水平*，而不是“慢慢能到”**。

MIMIC iter 0 val_loss 轨迹（PE-RQH，~80 min 内 23 epoch）：
```
0.972 → 0.935 → 0.923 → 0.899 → 0.859 → 0.840 → 0.829 → 0.813
```
单调下降，约 -16%。但还远未到 SQHyper baseline (~0.42 域)。

---

## 二. MSE 指标分析

### 2.1 USHCN（已完成）

| iter | 0 | 1 | 2 | 3 | 4 | mean | std |
|------|---|---|---|---|---|------|-----|
| **PE-RQH MSE** | 0.242 | 0.176 | 0.240 | 0.241 | 0.189 | **0.218** | **0.030** |
| **SQHyper MSE** | 0.190 | 0.186 | 0.190 | 0.206 | 0.183 | **0.191** | **0.009** |

**关键现象：PE-RQH 在 USHCN 是双峰分布**

- **好模式**: iter 1 (0.176)、iter 4 (0.189) — *与 SQHyper 同档甚至更优*（best 0.176 < SQHyper best 0.183）
- **坏模式**: iter 0 (0.242)、iter 2 (0.240)、iter 3 (0.241) — 比 SQHyper worst (0.206) 还差 17%
- σ 是 SQHyper 的 **3.3 倍**

**根因推测**：
1. 初始化敏感 — 没有结构先验（时间/变量超边）锚定，初期的 codebook 分配是随机分簇
2. 一旦初期分簇不合理，后续 layer 的 Hamilton 路由就放大错误
3. 而 SQHyper 的双层超边给了对称性约束，初始化误差被正则化掉

### 2.2 P12（iter 0 完成中）

PE-RQH 最佳 val_loss = **0.404**（含 aux ~0.10，估算 MSE val ≈ **0.30**）  
SQHyper test MSE = **0.301**

**乍看持平，但仔细看：**
- val_loss 在 0.405 平台震荡 30+ 分钟无显著下降
- early stopping counter 走到 8/10 才停
- SQHyper 是 *快速* 下降到 0.30 然后稳定

**判断**：P12 上 PE-RQH 有可能 *勉强追上* SQHyper，但代价是 5× 时间。这是“边际持平”，不是“显著超越”。

### 2.3 MIMIC_III（iter 0 中）

PE-RQH 当前最佳 val_loss = **0.813**（估算 MSE val ≈ **0.71**）  
SQHyper test MSE = **0.420**

**~70% 回退**。即使 val_loss 还在缓慢下降，要想从 0.71 → 0.42 需要把当前 trajectory 的进度再走 4-5 倍。这几乎不可能在 patience=10 的限制内做到。

### 2.4 HumanActivity（4/5 iter 已结束，无 MSE 输出）

val_loss 在每个 iter 内快速从 1.36 → ~0.33 收敛。
- baseline SQHyper MSE 估计 ~0.017（HyperIMTS 论文 0.0421）
- val 0.33 含 aux ~0.1，估算 MSE val ~0.23
- 即使去掉 aux，~0.20 vs 0.017 = **10× 回退**

---

## 三. 数据集分布性分析

### 3.1 单元格数 N（决定 codebook 表达需求）

| 数据集 | seq_len | pred_len | enc_in | N = (seq+pred) × V | 倍数 |
|--------|---------|----------|--------|---------------------|------|
| USHCN | 150 | 3 | 5 | 765 | 1× |
| P12 | 36 | 3 | 36 | 1,404 | 2× |
| MIMIC_III | 72 | 3 | 89 | 6,675 | 9× |
| HumanActivity | 3000 | 3 | 12 | 36,036 | **47×** |

### 3.2 与现有设计冲突

PE-RQH 用 **K=32-64 个 codes** 覆盖所有 N 个 cells 的事件类型。

- USHCN: K/N = 32/765 = 4.2% — 似乎够（但实际不稳）
- P12: K/N = 64/1404 = 4.6% — 类似  
- MIMIC: K/N = 64/6675 = 0.96% — **严重不足**
- HumanActivity: K/N = 64/36036 = 0.18% — **极度不足**

**这是一个根本架构问题**：当数据规模上去，固定大小的 codebook 无法表达细粒度事件模式。

### 3.3 为什么 SQHyper 不需要这么多 "codes"

SQHyper 的 hyperedge 数量 = (T + V) 个：
- USHCN: 153 + 5 = 158
- P12: 39 + 36 = 75  
- MIMIC: 75 + 89 = 164
- HumanActivity: 3003 + 12 = 3015

**且每个 hyperedge 由 incidence matrix 强约束** — 它"代表"哪些 cell 是 deterministic 的（按时间或变量）。这是结构先验。

PE-RQH 的 K 个 codes 是 *软分配*，要从数据中 *学出* 谁应该归到哪个 code。**这相当于让模型自己发现"时间维度"和"变量维度"，对小数据集是过分要求**。

---

## 四. 各数据集失败模式归类

| 数据集 | 失败模式 | 主因 |
|--------|----------|------|
| **USHCN** | 高方差双峰（mean 比 SQHyper 差 14%，但 best 持平） | 训练不稳，初始化敏感 |
| **P12** | 收敛到与 SQHyper 同水平但 ~5× 时间 | codebook 收敛慢，无结构锚 |
| **MIMIC_III** | 大幅回退（~70%） | K 太小（<1% N），缓慢但远不到 baseline |
| **HumanActivity** | 灾难性回退（估 ~10×） | K 严重不足（0.18% N），且 N 巨大 |

**共同主因**：抛弃时间/变量的结构性 hyperedge 是过激的，对中-高 V 数据集和长 seq 数据集尤其致命。

---

## 五. 不能怪 PE-RQH 整体设计的方面

1. **梯度稳定**：未发现 NaN（修过一个 Gumbel bug）
2. **理论自洽**：Quaternion routing + VQ codebook 数学上没问题
3. **小数据无回退**：USHCN best iter (0.176) 实际上 *打败* SQHyper best (0.183)
4. **Per-epoch 不慢**：除了 P12，其他都与 SQHyper 同量级

---

## 六. 不能掩盖的设计缺陷

### 6.1 Codebook 容量与数据规模不匹配

固定 K 不能 scale 到大数据。需要要么：
- 让 K 随 N 自适应（但 lookup 复杂度爆炸）
- 引入 **结构性 codes**（codes 与时间/变量绑定的 inductive bias）

### 6.2 缺少结构性引导导致初期分簇随机

VQ codebook 初始随机 → 早期分配近乎噪声 → quaternion routing 在噪声上工作 → 训练崩溃或卡在差解。

### 6.3 单纯靠 commit + diversity loss 训不出有意义的 codes

`commitment_loss = ‖q − code‖²` 把 q 拉向最近 code。
`diversity_loss = log K − H(usage)` 把 usage 推向均匀。

但这两个加起来 *不能保证* code 学到"事件语义"。可能学到的是任意聚类。

---

## 七. 决策选项（不简化结构、不放弃 PE-RQH 的方向上）

### 选项 A：Hybrid（PE-RQH on top of structural backbone）
保留 SQHyper 的时间/变量超边作为骨架，在其上 *叠加* PE-RQH 的 codebook routing 作为软增强。让结构先验给 codebook 提供初始锚点。

**优点**：保持 SQHyper baseline 的下界，codebook 只能改进不能更差  
**缺点**：本质上是 SQHyper + 软调制，离“纯 PE-RQH”原意远了

### 选项 B：结构性 Codebook
让 codes 不是纯学习的随机向量，而是结构化分组：
- K = K_time + K_var + K_event
- K_time 个 codes 与时间频率绑定（位置编码初始化）
- K_var 个 codes 与变量绑定（embedding 初始化）  
- K_event 个 codes 是事件原型（学习）
拼起来 K 仍然是 32–64，但 *每类 codes 有 inductive bias*。

**优点**：保留 codebook 框架，引入足够的结构  
**缺点**：实现复杂，仍可能不如 SQHyper

### 选项 C：Codebook scaling + curriculum
让 K 随数据集动态扩展。从 K=32 开始，根据 usage 自动 split high-usage codes。同时用 SQHyper 模型权重做 warm start。

**优点**：保留架构纯度  
**缺点**：训练流程复杂，warm-start 引入耦合

### 选项 D：实验失败、回到 SQHyper 改进路径
承认 PE-RQH 是一个清晰的 **negative result**，写入 paper 作为 ablation。主线回到 SQHyper：
- 解决 USHCN 双峰问题（训练稳定性）
- 调优 P12（提升超过 0.301）
- MIMIC 改进 K/V gating

**优点**：路径清晰，时间可控  
**缺点**：不再是"完全独立的新模型"

---

## 八. 推荐

我推荐 **选项 B + 部分 C**：保留 PE-RQH 的核心创新（quaternion routing + 软 codebook），但把 codebook 结构化分组（time codes + var codes + event codes），让 inductive bias 重新进入。这样既不放弃创新方向，也不简化结构以求速度。

如果用户希望更激进的纯净度，可以先尝试 **选项 C**（curriculum + scaling）；如果希望最稳，**选项 D** 是最务实的。
