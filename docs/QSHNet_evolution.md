# QSH-Net 模型演进记录

## 基线参照

HyperIMTS 论文报告的 MSE（5轮均值±标准差）：

| 数据集 | MSE |
|--------|-----|
| P12 | 0.2996 ± 0.0003 |
| HumanActivity | 0.0421 ± 0.0021 |
| USHCN | 0.1738 ± 0.0078 |

---

## v1~v3：初始实现阶段

### v1：完整实现设计文档全部 5 个模块
- 模块1：四元数线性层（Kronecker 块矩阵 Hamilton 积 + 分离激活）
- 模块2：脉冲状态空间模型（ZOH 离散化 + 自适应替代梯度 + Gray Code RPE）
- 模块3：有向因果超图（无填充节点表征 + HGNN-AS 软剪枝 + 因果拉普拉斯）
- 模块4：联合时频建模（自适应分块 + NUDFT + iNUDFT 季节性偏差）
- 模块5：DBLoss（EMA 趋势-季节分解损失）

**结果：** GPU OOM（3.6GB 显存不足），batch_size 从 32 降到 2 仍然 OOM。Loss 爆炸到 1e7 级别。

### v2~v3：速度优化
- S-SSM：从逐步 for loop 改为 log-space cumsum 并行化
- 因果拉普拉斯：从 O(N²) 密集矩阵改为 O(N) cumsum
- 去掉 node_self_attention

**结果：** 能跑了，但 MSE 远高于 HyperIMTS。

---

## v4~v5：轻量化重设计

### v4：极简架构
**结果（1轮）：** P12=0.316, Human=0.075, USHCN=0.184

### v5：增强表达力
**结果：** MSE 几乎没有下降。

**教训：** 在弱基础上加组件无法弥补核心消息传递机制的差距。

---

## v6~v7：恢复 HyperIMTS 核心

### v6：恢复 HyperIMTS 级别的消息传递
**结果（1轮）：** P12=0.303, Human=0.044, USHCN=0.228

### v7：完整复刻 HyperIMTS + 外挂组件
**结果（1轮）：** P12=0.303, Human=0.043, USHCN=0.228

**教训：** 外挂的 QuaternionBlock 和 SpikingGate 对 MSE 没有贡献。

---

## v8：深度融合尝试

**结果（1轮）：** 几乎没有变化，Human 略微退步。

**教训：** 四元数替换 Linear 引入了过强的结构约束；因果掩码改变了超图拓扑引入噪声。

---

## 消融实验阶段

### 系统性消融（5 个配置 × 3 个数据集）

| 配置 | P12 MSE | Human MSE | USHCN MSE |
|------|---------|-----------|-----------|
| A: 全部开启 | 0.3004 | 0.0464 | 0.1814 |
| B: w/o 四元数 | 0.3008 | 0.0436 | 0.1649 |
| C: w/o 脉冲 | 0.3009 | 0.0440 | 0.2244 |
| D: w/o 因果掩码 | 0.2992 | 0.0493 | 0.2074 |
| **E: 纯 HyperIMTS** | **0.2994** | **0.0413** | **0.1671** |

**关键发现：** 纯 HyperIMTS 复刻（E）在所有数据集上都是最好或接近最好的。

---

## 有机重设计阶段（2026-04-12）

### 第一步：Context-Aware Spike Selection 重设计
利用超图的 `variable_incidence_matrix` 计算每个变量的均值上下文，标量膜电位 `Linear(2D, 1)` 替代旧版 `Linear(D, D)`。

### 第二步：语义四元数融合（失败）
**结果：** USHCN MSE 从 0.168 恶化到 0.221+。

### 第三步：四元数残差精炼（改善但不稳定）
**结果：** 单次最优 0.165，但 5 轮均值 0.211 ± 0.040。

### 第四步：恒等初始化（突破！）
**关键发现：** 所有 QSH 增强组件必须从精确恒等开始。
**结果（itr=3，无 weight decay）：** 均值 0.181 ± 0.009

### 第五步：AdamW + Weight Decay（稳定化）
**结果（itr=5，全数据集评估 2026-04-14）：**

| 数据集 | 论文 HyperIMTS | **QSH-Net** | 改善 |
|--------|---------------|-------------|------|
| **MIMIC_III** | 0.4259 ± 0.0021 | **0.3933 ± 0.0060** | **-7.7%** ✅ |
| **MIMIC_IV** | 0.2174 ± 0.0009 | **0.2157 ± 0.0022** | **-0.8%** ✅ |
| **HumanActivity** | 0.0421 ± 0.0021 | **0.0416 ± 0.0003** | **-1.2%** ✅ |
| P12 | 0.2996 ± 0.0003 | 0.3006 ± 0.0013 | +0.3% ⚠ |
| USHCN | 0.1738 ± 0.0078 | 0.1870 ± 0.0277 | +7.6% ❌ |

各轮详情：
- MIMIC_III: 0.388, 0.389, 0.402, 0.400, 0.388
- MIMIC_IV: 0.2155, 0.2140, 0.2190, 0.2145
- HumanActivity: 0.0413, 0.0419, 0.0413, 0.0415, 0.0417
- P12: 0.3009, 0.2998, 0.3027, 0.3001, 0.2996
- USHCN: 0.1696, 0.1687, 0.2361(异常), 0.1617, 0.1991

---

## 核心教训总结

### 1. 恒等初始化是最重要的单一改进
新组件从随机初始化开始 = 向基线模型注入噪声。唯一安全的做法是从精确恒等开始。

### 2. 加法精炼 > 插值融合
`linear + α*quat` 不会减弱原始信号。

### 3. Weight decay 对高方差数据集至关重要
AdamW (wd=1e-4) 将 USHCN 5-run std 从 0.040 降到 0.011。

### 4. 不要改变超图核心
v1~v8 的反复失败证明：任何改变 HyperIMTS 核心消息传递机制的尝试都会伤害性能。

---

## USHCN 基础模型优化实验（2026-04-13，已全部回退）

### 全面对比

| 版本 | 均值 | std | 中位数 | 好运行率 |
|------|------|-----|--------|----------|
| HyperIMTS 同环境 | 0.2016 | 0.0352 | 0.1745 | 3/5 |
| **原始 QSH-Net** | **0.1862** | 0.0268 | **0.1725** | **4/5** |
| A: LayerNorm | 0.2015 | 0.0234 | 0.1997 | 3/5 |
| A+B: LN+WarmCos | 0.2843 | 0.0710 | 0.2522 | 0/5 |
| A+C.1: LN+Time | 0.2340 | 0.0277 | 0.2417 | 1/5 |
| A+C.2: LN+Drop | 0.2283 | **0.0170** | 0.2385 | 1/5 |

### 核心结论

1. **正则化越强 → 越稳定但越差**
2. **双峰 ≠ bug，而是 feature**
3. **USHCN 上不要动基础模型**
4. **Early stopping 与慢衰减 LR 不兼容**

所有改动已回退，恢复到产出四数据集结果的原始版本。

---

## 待解决问题

1. USHCN 训练不稳定：itr=5 中出现 0.236 异常轮
2. P12 略高于基线 0.3%
3. 四元数门控 alpha ≈ 0.047 几乎不动——可能需要更高的初始值或独立学习率
4. 脉冲 fire_rate 跨 run 变化很大
5. **MIMIC_III 改善 7.7% 的机理需要深入分析**

---

## 有机融合重设计实验（2026-04-14~15，已全部回退）

### 背景与动机

旧版 QSH-Net 的 Spike 和 Quaternion 是"贴"在超图上的，不是"长"在超图里的：
- SpikeSelection 只在消息传递前做标量加权（fire_rate=16%, attenuation=98.2%，区分度仅 1.8%）
- QuaternionLinear 只在融合后做残差补充（alpha≈0.047 几乎不动）
- 超图的消息传递本身完全没变

目标：让四元数和脉冲有机融合进超图消息传递的核心机制中。

### 设计 A：QuaternionStructuredFusion + SpikeTemperature

**QuaternionStructuredFusion**（替换 Linear(3D, D)）：
- obs/tg/vg 分别投影为四元数的 i/j/k 分量，线性融合作为实部 r
- Hamilton 积（Kronecker 块矩阵）自动产生跨源交互：时间×变量、变量×节点、节点×时间
- 输出取实部

**SpikeTemperature**（替换 SpikeSelection）：
- 脉冲信号调制 n2h 注意力的温度（而非标量加权）
- 偏离变量上下文 → 低温度 → 尖锐注意力；符合上下文 → 高温度 → 平滑注意力

### 实验 A1：D/4 维分量 + D×D Kronecker（首版）

每个四元数分量投影到 D/4 维，Kronecker 块矩阵 (D×D)，子矩阵 (D/4×D/4)。

**USHCN itr=5 结果：** 0.235, 0.246, 0.225, 0.275, 0.169 → 均值 0.230 ± 0.037

**失败原因：** D/4 瓶颈——每个分量只有 64 维（D=256），三源融合信息从 256 维压缩到 64 维，丢失 75% 信息。与之前"语义四元数"失败的原因完全一致。

### 实验 A2：全 D 维分量 + 4D×4D Kronecker

每个分量保持完整 D=256 维，Kronecker 块矩阵 (4D×4D)=(1024×1024)，子矩阵 (D×D)=(256×256)。输出取实部（前 D 维）。

**USHCN itr=3 结果：** 0.166, 0.249, 0.182 → 均值 0.199 ± 0.037

**问题：** 参数量爆炸（4×D² = 262K per layer），且 `proj_r` 的随机初始化破坏了恒等初始化——训练起点不是纯 HyperIMTS。

### 实验 A3：修复恒等初始化（Linear 主路径 + Hamilton 积残差 + gate=0）

保留原始 `Linear(3D, D)` 作为主路径，Hamilton 积作为残差补充，gate 初始化为 0（tanh(0)=0）确保精确恒等初始化。

**USHCN itr=3 结果（含 SpikeTemperature）：** 0.193, 0.212, 0.195 → 均值 0.200 ± 0.010

**分析：** 方差大幅降低（0.010 vs 0.037），恒等初始化修复有效。但均值 0.200 仍比旧版（0.187）差。

### 实验 A4：消融——去掉 SpikeTemperature

只保留 QuaternionStructuredFusion，n2h 注意力回到标准 MultiHeadAttentionBlock。

**USHCN itr=3 结果：** 0.215, 0.161, 0.164 → 均值 0.180 ± 0.030

**关键发现：SpikeTemperature 在伤害性能**（去掉后从 0.200 改善到 0.180）。温度调制在 USHCN（仅 5 变量）上区分度太低，反而引入噪声。

### 实验 A4 全数据集评估

| 数据集 | QuatFusion only | 旧版 QSH-Net | HyperIMTS 论文 |
|--------|----------------|-------------|---------------|
| HumanActivity (itr=5) | 0.0417 ± 0.0003 | 0.0416 ± 0.0003 | 0.0421 ± 0.0021 |
| P12 (itr=5) | 0.3021 ± 0.0015 | 0.3006 ± 0.0013 | 0.2996 ± 0.0003 |
| MIMIC_III (itr=5) | 0.4015 ± 0.0115 | **0.3933 ± 0.0060** | 0.4259 ± 0.0021 |
| USHCN (itr=3) | 0.180 ± 0.030 | 0.187 ± 0.028 | 0.174 ± 0.008 |

**结论：** QuaternionStructuredFusion 在 MIMIC_III 上退步（0.401 vs 0.393），方差翻倍（0.012 vs 0.006）。其他数据集基本持平。

### 核心教训

1. **D/4 瓶颈是致命的**：四元数的 4 分量结构天然要求将 D 维切成 4 份，这在 D=256 时丢失太多信息
2. **恒等初始化必须精确**：任何新的 Linear 层（如 proj_r）如果不和原始 HyperIMTS 的 Linear 共享权重，就会破坏恒等初始化
3. **SpikeTemperature 在低变量数数据集上有害**：温度调制需要足够的变量多样性才能产生有意义的区分
4. **Hamilton 积的结构约束不一定优于自由 Linear**：在 MIMIC_III 上，旧版的 `Linear(3D,D) + α*QuatLinear(linear_out)` 比新版的结构化融合效果更好
5. **旧版的"拼接"设计虽然不够优雅，但恒等初始化 + 加法残差的安全性是其成功的关键**

### 决策

所有新设计实验已回退，恢复到旧版 QSH-Net（SpikeSelection + QuaternionLinear 加法精炼）。该版本在 5 个数据集中 3 个超越 HyperIMTS 基线（MIMIC_III -7.7%, HumanActivity -1.2%, MIMIC_IV -0.8%），是目前效果最好的版本。

---

## 分层协同型 M1 之后的结构试验链（2026-04-16，当前主线）

### 当前背景

旧 M1 的实现方向没有回退，但针对其真实训练动力学，后续已经完成一轮更细的结构试验链。

此时最关键的事实不是「某个超参数没调好」，而是：

1. 旧 M1 的真实训练动力学并不是「Spike + Event + Quaternion 协同」。
2. 旧 M1 更接近 `HyperIMTS + retain 调制 + quaternion 残差`。
3. 如果希望保住论文里的「三元素统一框架」，就必须先让 `event` 分支从死分支变成真实可训练、真实参与训练的分支。

### 阶段 1：bounded-gate 保守稳定化（已回退）

在不改变 M1 总体结构的前提下，曾尝试将 `retain_gate`、`event_gate` 与 `event_residual_scale` 的增长改为更保守的有界形式。

| 数据集 | 轮数 | MSE 均值 ± std | 结论 |
|--------|------|----------------|------|
| USHCN | 10 | **0.2068 ± 0.0375** | 明显退化，整套试验回退 |
| P12 | 5 | **0.3012 ± 0.0012** | 基本不变 |
| MIMIC_III | 5 | **0.3983 ± 0.0066** | 有小幅改善，但不具备统一推广性 |

**结论：** bounded-gate 不是统一主线方案，当前仓库已回到上一版 M1 主线。

### 阶段 2：本地参数诊断确认旧 M1 的真实动力学

在 `exp/exp_main.py` 中加入 epoch 级 `QSHDiag` 之后，对 `USHCN` 与 `HumanActivity` 做了短程本地诊断。

**关键观察：**

1. 旧 M1 中的 `event residual` 分支最初是死分支。
   - `event_proj.weight norm = 0.0`
   - `event_residual_scale = 0.0`
   - `event_log_scale` 长期停留在极低区间

2. 真正持续学习的是：
   - `retain_log_scale`
   - `membrane_proj.weight`
   - `quat_gate`
   - `quat_h2n`

3. USHCN 的坏轮更像是活跃分支被高方差数据放大，而不是 `event` 协同失败。

**结构级结论：**

- 旧 M1 的真实动力学更接近 `HyperIMTS + retain 调制 + quaternion 残差`。
- 如果继续沿论文叙事推进，就不能再把 `event` 当成默认已经工作的分支。

### 阶段 3：方向 B 稳定化试验

#### B1：`retaincap_main`

只对 `retain` 路径做幅度约束，不动 quaternion 与 event 结构。

| 数据集 | 轮数 | MSE 均值 ± std | 结论 |
|--------|------|----------------|------|
| HumanActivity | 3 | **0.0442 ± 0.0018** | 基本稳定 |
| USHCN | 5 | **0.1673 ± 0.0033** | 明显优于旧 M1，说明 `retain` 约束有效 |

#### B2：`retaincap_quatbound`

在 `retain cap` 基础上，再限制 quaternion residual 相对线性主干的残差幅度。

| 数据集 | 轮数 | MSE 均值 ± std | 结论 |
|--------|------|----------------|------|
| HumanActivity | 3 | **0.0426 ± 0.0009** | 安全 |
| USHCN | 5 | **0.1674 ± 0.0093** | 均值几乎不动，方差略放大 |

**结论：** quaternion residual 的简单有界化不是当前最关键的增益点。

### 阶段 4：方向 A 激活 `event` 分支

#### A1：`eventfusion_sigmoid`

只改 `event` 融合强度表达：

- `event_residual_scale` 从近零指数残差，改成 `sigmoid` 受控融合系数
- 初始化到 `sigmoid ≈ 0.1`

| 数据集 | 轮数 | MSE 均值 ± std | 结论 |
|--------|------|----------------|------|
| HumanActivity | 3 | **0.0426 ± 0.0009** | `event` 分支在简单数据上稳定活化 |
| USHCN | 5 | **0.1803 ± 0.0386** | `event` 活了，但高方差尾部明显放大 |

**关键意义：**

- `event_residual_scale` 不再是近零死残差语义
- `event_proj.weight norm` 在 HumanActivity 与 USHCN 上都持续偏离 0
- 说明 `event` 分支第一次真正参与训练

#### A2：`eventnorm_main`

在 `eventfusion_sigmoid` 基础上，只增加 `event` 支路的独立归一化：

- `temporal_event_norm`
- `variable_event_norm`

| 数据集 | 轮数 | MSE 均值 ± std | 结论 |
|--------|------|----------------|------|
| HumanActivity | 3 | **0.0430 ± 0.0013** | 仍然稳定 |
| USHCN | 5 | **0.1653 ± 0.0011** | 在保留 `event` 活性的同时显著压回方差 |

**结论：**

- `event` 支路的问题不只是“强度太弱”，还包括“缺少独立控制”
- `event` 独立归一化是当前最有价值的结构补充

#### A2 扩展验证：`eventnorm_itr10`

为了确认 `itr=5` 不是偶然现象，对 `eventnorm` 版本补做了 `USHCN itr=10`。

| 数据集 | 轮数 | MSE 均值 ± std | 最优 | 最差 | 结论 |
|--------|------|----------------|------|------|------|
| USHCN | 10 | **0.1829 ± 0.0279** | 0.1628 | 0.2353 | 长重复下仍有尾部风险，不能直接视为最终定版 |

**判断：**

- `eventnorm` 明显优于单纯增强融合强度的 `eventfusion_sigmoid`
- 但长重复下仍会出现 `iter5/7/8` 级别的坏轮
- 当前更适合把它写成「可训练、可控、但尚未彻底稳定」的阶段性版本

#### A2.0 失败分支：`eventgain_main`

尝试在 `eventnorm` 基础上，为 `event` 注入额外增加“主干条件化 gain”：

- `event_gain = sigmoid(Linear(main_hyperedge_state))`
- `updated = main_update + event_scale * event_gain * event_delta`

| 数据集 | 轮数 | MSE 均值 ± std | 结论 |
|--------|------|----------------|------|
| HumanActivity | 3 | **0.0424 ± 0.0006** | 简单数据上基本正常 |
| USHCN | 5 | **0.1974 ± 0.0633** | 明显退化，并重新出现重度坏轮 |

**判断：**

- `event` 支路的问题不是“缺少一层主干条件化控制器”
- 额外的条件化 gain 反而会让部分 run 更脆弱
- 这一方向应明确否定

#### A2.0 失败分支：`event_temporal_only`

尝试缩减传播范围，只保留 temporal `event` 注入，去掉 variable `event` 注入。

| 数据集 | 轮数 | MSE 均值 ± std | 结论 |
|--------|------|----------------|------|
| HumanActivity | 3 | **0.0422 ± 0.0005** | 简单数据上保持稳定 |
| USHCN | 5 | **0.1779 ± 0.0365** | 明显不如双支路 `eventnorm_main` |

**判断：**

- 双支路传播本身不是主要问题
- 只保留 temporal 注入并不能自动带来更高稳定性
- `eventnorm_main` 的有效性来自“独立控制 + 双支路表达”，而不是单纯删支路

#### A2.0 失败分支：`eventnorm_clip`

尝试在 `eventnorm_main` 基础上，直接对归一化后的 `event_delta` 做 `tanh` 幅度裁剪。

| 数据集 | 轮数 | MSE 均值 ± std | 结论 |
|--------|------|----------------|------|
| HumanActivity | 3 | **0.0422 ± 0.0005** | 简单数据上基本不受影响 |
| USHCN | 5 | **0.1898 ± 0.0381** | 出现明显双峰坏轮，方向失败 |

**判断：**

- 问题不在于归一化后的 `event_delta` 还残留少量大幅值
- 直接裁剪 `event_delta` 会破坏注入连续性，造成新的两极分化
- 这一步明确说明：应当控制“总注入量”，而不是粗暴裁剪“事件形状”

#### A2.1：`eventscalecap_main`

在 `eventnorm_main` 基础上，不改拓扑，只对 `event_scale` 引入温和上界：

- `event_scale = clamp(sigmoid(event_residual_scale), max=0.12)`

| 数据集 | 轮数 | MSE 均值 ± std | 结论 |
|--------|------|----------------|------|
| HumanActivity | 3 | **0.0430 ± 0.0013** | 整体保持稳定 |
| USHCN | 5 | **0.1663 ± 0.0027** | 与 `eventnorm_main` 接近，说明温和上界是安全保险丝 |

**判断：**

- 这一步没有像 `eventnorm_clip` 那样制造新的双峰坏轮
- 但在 `itr=5` 上也没有明显胜过 `eventnorm_main`
- 因此它更像是一个保守的总量控制，而不是短重复下的强增益点

#### A2.1 扩展验证：`eventscalecap_itr10`

为了确认 `event_scale` 上界是否对长重复稳定性真正有帮助，对 `USHCN` 补做了 `itr=10`。

| 数据集 | 轮数 | MSE 均值 ± std | 最优 | 最差 | 结论 |
|--------|------|----------------|------|------|------|
| USHCN | 10 | **0.1728 ± 0.0222** | 0.1610 | 0.2355 | 明显优于 `eventnorm_itr10`，但尾部坏轮仍未完全消失 |

**判断：**

- 相比 `eventnorm_itr10` 的 `0.1829 ± 0.0279`，温和上界版本均值和方差都更好
- 说明长重复下的一部分尾部风险确实来自 `event` 总注入量过大
- 但 `iter8 = 0.2355` 仍然表明该问题没有被彻底解决
- 因此当前最准确的表述应是：`eventnorm + mild event_scale cap` 是更好的三元素主线候选，但仍不是最终稳定版

### 当前综合结论

1. **如果目标只是追求最稳的短期结果，`retaincap_main` 仍然是最保守的稳定化版本。**
2. **如果目标是保住论文里的三元素统一框架，当前更合适的主线候选已经升级为 `eventnorm + mild event_scale cap`。**
3. **`eventnorm` 已经证明：**
   - `event` 不再是死分支
   - `event` 可以在 HumanActivity 与 USHCN 上真实参与训练
   - `event` 支路需要独立控制，否则会在高方差数据上放大尾部风险
4. **`event_scale` 温和上界进一步证明：**
   - `event` 的长重复尾部风险部分来自总注入量偏大
   - 对总注入量做轻量约束可以改善 `USHCN itr=10` 的均值和方差
   - 但仍不足以彻底消除坏轮
5. **当前最准确的表述不是“三分支已经稳定协同”，而是：**
   - 超图仍然是结构主干
   - 四元数仍然是主要增强分支
   - 脉冲 / event 已经成为真实可训练、可控的事件注入支路，并且在加入轻量总量约束后进一步改善了长重复稳定性，但长重复下仍有尾部不稳定性

## EQHO 开发记录（2026-04-17）

### 背景

在 `eventscalecap_main` 母体之上，尝试引入 `Event-Conditioned Quaternion Hyperedge Operator (EQHO)`，目标不是替换现有超图主干，而是让 `event` 通过 hyperedge-level quaternion refinement 真实参与四元数混合控制。

实现按 3 个阶段推进：

- `A1`：固定 real-only 模板，只增加 hyperedge-level quaternion refinement
- `A2`：保持固定模板，仅让 `event_summary` 控制 `mix_gain`
- `A3`：放开完整 `mix_coef`，允许 `event_summary` 动态控制 `r / i / j / k` mixing

### 本地筛查结果：`A3 = Full EQHO`

运行版本：

- `HumanActivity`：`QSHNet_EQHO_A3`
- `USHCN`：`QSHNet_EQHO_A3`

#### 结果

| 配置 | 数据集 | 轮数 | 结果 |
|------|--------|------|------|
| `QSHNet_EQHO_A3` | HumanActivity | 3 | **0.0418 ± 0.0002** |
| `QSHNet_EQHO_A3` | USHCN | 5 | **0.1979 ± 0.0310** |

对应单轮结果：

- HumanActivity：`0.04183`、`0.04208`、`0.04159`
- USHCN：`0.21483`、`0.17559`、`0.24942`、`0.18205`、`0.16750`

### `QSHDiag` 诊断结论

#### HumanActivity

- `EQHO-temporal coef_r` 长期保持在约 `0.597`
- `EQHO-variable coef_r` 长期保持在约 `0.593`
- `gain_mean` 基本稳定在 `0.0192 ~ 0.0195`
- `temporal residual_norm_mean` 约 `0.06`
- `variable residual_norm_mean` 大约 `0.11 ~ 0.14`

判断：

- `EQHO` 已真实参与前向，不是死结构
- 但其行为更接近温和 refinement，而不是强烈改变主导表达的分支
- `temporal / variable` 两支路有差异，但差异幅度有限

#### USHCN

- `EQHO-temporal coef_r` 快速塌到接近 `0`
- `EQHO-variable coef_r` 同样接近 `0`
- `temporal` 侧主要由 `coef_i / coef_j` 主导
- `variable` 侧长期由 `coef_k` 主导
- `gain_mean` 升到 `0.07 ~ 0.14`
- `summary_mean` 升到 `2.4 ~ 3.5`
- `residual_norm_mean` 升到 `1.4 ~ 1.8`

判断：

- `A3` 不是“没学起来”，而是学得过强
- 完整动态 `mix_coef` 在高方差数据上会把 quaternion hyperedge refinement 推到非实部主导区间
- 该机制会重新放大 `USHCN` 的坏轮风险

### 结论

1. `EQHO` 本身不是无效设计。
   - 它在 `HumanActivity` 与 `USHCN` 上都表现出真实可训练性。

2. `A3` 不能作为当前主线继续推进。
   - `HumanActivity` 稳定且结果不差。
   - 但 `USHCN itr=5` 明显退化到 `0.1979 ± 0.0310`，远差于当前主线候选 `eventscalecap_main` 的 `0.1663 ± 0.0027`。

3. 最值得保留的不是 `A3` 本身，而是新的结构判断：
   - `event-conditioned hyperedge refinement` 有研究价值；
   - 但 `full dynamic r / i / j / k mixing` 对高方差数据过强；
   - 后续若继续做 `EQHO`，应优先回到受限动态化，而不是保留当前 `A3`。

### 当前决策更新

- `eventscalecap_main / eventscalecap_itr10` 继续作为当前统一主线候选
- `EQHO A3` 记为“可训练但失败的探索分支”
- 后续若继续推进 `EQHO`，下一步应优先尝试 `A2.5`
  - 保持 real-dominant 模板
  - 只允许 `mix_coef` 在安全边界内做小幅偏移
  - 不再直接放开完整动态 `mix_coef`

### 本地筛查结果：`A2.5 = Template-Offset + Safety Floor`

在 `A3` 失败后，继续在 `eventscalecap_main` 母体上尝试更保守的受限动态化版本：

- 使用固定 real-dominant 基模板 `mix_coef = [0.70, 0.10, 0.10, 0.10]`
- 仅允许 `event_summary` 预测有界 offset
- 对 `coef_r` 显式设置 safety floor：`coef_r >= 0.55`
- `mix_gain` 路径保持不变

运行版本：

- `HumanActivity`：`QSHNet_EQHO_A25`
- `USHCN`：`QSHNet_EQHO_A25`

#### 结果

| 配置 | 数据集 | 轮数 | 结果 |
|------|--------|------|------|
| `QSHNet_EQHO_A25` | HumanActivity | 3 | **0.0418 ± 0.0002** |
| `QSHNet_EQHO_A25` | USHCN | 5 | **0.2062 ± 0.0264** |

对应单轮结果：

- HumanActivity：`0.04178`、`0.04207`、`0.04159`
- USHCN：`0.23422`、`0.17866`、`0.22763`、`0.21162`、`0.17868`

### `A2.5` 诊断结论

#### HumanActivity

- `EQHO-temporal coef_r` 长期稳定在约 `0.699`
- `EQHO-variable coef_r` 也保持 real-dominant
- `gain_mean` 基本维持在 `0.0192 ~ 0.0196`

判断：

- `A2.5` 没有破坏简单数据上的可训练性
- 说明“real-dominant 模板 + 有界 offset”本身是安全可运行的

#### USHCN

- temporal `coef_r` 被稳定压在 `0.55`
- `coef_i / coef_j / coef_k` 被稳定限制在约 `0.15`
- `A3` 中 `mix_coef` 偏离安全区的问题被显式消除
- 但 temporal `gain_mean` 仍长期维持在 `0.14 ~ 0.15`
- `summary_mean` 仍在 `3+`，个别轮次更高

判断：

- `A2.5` 已经证明 `A3` 的主要灾难确实包含 `mix_coef` 失控这一因素
- 但它没有解决 `mix_gain` 在高方差数据上的放大问题
- 也就是说，`EQHO` 在 `USHCN` 上的主风险已经从“系数漂移”转移为“增益饱和”

### `A2.5` 的结构意义

1. `A2.5` 不是有效候选主线。
   - `USHCN itr=5` 仍退化到 `0.2062 ± 0.0264`；
   - 不仅差于 `eventscalecap_main` 的 `0.1663 ± 0.0027`，也比 `A3` 的 `0.1979 ± 0.0310` 更差。

2. 但它提供了比 `A3` 更强的结构归因证据。
   - `mix_coef` 被锁回安全区后，坏轮仍然存在；
   - 因此 `EQHO` 在 `USHCN` 上的根本风险不能再简单归因于 quaternion mixing 系数漂移。

3. 当前更准确的判断是：
   - `A3` 失败来自 `mix_coef` 与 `mix_gain` 的联合作用；
   - `A2.5` 进一步证明，即使压住 `mix_coef`，只要 `mix_gain` 仍可在 temporal 支路持续接近饱和，`USHCN` 仍会出现显著坏轮。

### `A2.5` 后的决策更新

- `A2.5` 记为“结构归因成功，但实验结果失败”的探索分支
- 后续若继续推进 `EQHO`，下一步不应再围绕 `mix_coef` 做文章
- 更值得验证的唯一结构假设将转向：
  - 在保持 `A2.5` real-dominant 安全模板不动的前提下，
  - 对 `mix_gain` 做单因素约束或重参数化，
  - 检查 `USHCN` 的坏轮是否能进一步被压回

### 本地筛查结果：`A2.6 = A2.5 + gain hard cap`

在 `A2.5` 之后，继续只改一个核心因素：将 `mix_gain` 上界从 `0.15` 压到 `0.08`，其余全部保持不变：

- `mix_coef` 仍使用 `[0.70, 0.10, 0.10, 0.10]`
- `coef_r >= 0.55` 的 safety floor 不变
- `event_summary`、`coef_head`、quaternion residual 主体均不改

运行版本：

- `HumanActivity`：`QSHNet_EQHO_A26`
- `USHCN`：`QSHNet_EQHO_A26`

#### 结果

| 配置 | 数据集 | 轮数 | 结果 |
|------|--------|------|------|
| `QSHNet_EQHO_A26` | HumanActivity | 3 | **0.0418 ± 0.0003** |
| `QSHNet_EQHO_A26` | USHCN | 5 | **0.1846 ± 0.0297** |

对应单轮结果：

- HumanActivity：`0.04177`、`0.04207`、`0.04153`
- USHCN：`0.22667`、`0.16272`、`0.20463`、`0.17026`、`0.15859`

### `A2.6` 诊断结论

#### HumanActivity

- temporal / variable `gain_mean` 都稳定在 `0.0103 ~ 0.0104`
- `coef_r` 继续保持在约 `0.699`
- 结果与 `A2.5` 基本持平

判断：

- `gain hard cap` 没有把简单数据上的 EQHO 表达压死
- 说明把 `mix_gain` 上界直接降到 `0.08` 在小数据上仍是安全的

#### USHCN

- temporal `gain_mean` 被稳定压到 `0.075 ~ 0.080`
- 不再出现 `A2.5` 中 `0.14 ~ 0.15` 的持续饱和
- `coef_r` 继续被锁在 `0.55` 左右
- 但 temporal `summary_mean` 仍然长期在 `5 ~ 11`
- `residual_norm_mean` 仍然长期在 `1.63 ~ 1.76`

判断：

- `A2.6` 已经证明：单独压 `mix_gain` 上界，确实可以改善 `USHCN`
- 但它没有从根本上消除高能量 `event_summary` 驱动的放大问题

### `A2.6` 的结构意义

1. `A2.6` 是当前 `EQHO` 探索里最好的受限版本。
   - 相比 `A2.5` 的 `0.2062 ± 0.0264`，改善到 `0.1846 ± 0.0297`
   - 相比 `A3` 的 `0.1979 ± 0.0310`，也有实质改善

2. 但 `A2.6` 仍然不能进入主线。
   - 它仍明显差于 `eventscalecap_main` 的 `0.1663 ± 0.0027`
   - 并且方差仍明显偏大

3. 当前关于 `EQHO` 的最准确判断进一步更新为：
   - `mix_coef` 漂移不是唯一问题；
   - `mix_gain` 饱和也确实是问题；
   - 但即使同时压住 `mix_coef` 和 `mix_gain` 的显式上界，`event_summary` 驱动的高方差放大仍然存在。

### `A2.6` 后的决策更新

- `A2.6` 记为“比 `A2.5/A3` 更好，但仍未达到主线标准”的探索分支
- 如果后续继续推进 `EQHO`，不应再重复单纯的 `mix_coef` 边界修补
- 也不应再重复只做固定 `mix_gain` 上界压缩
- 下一轮若继续，需要转向更深一层的问题：
  - `event_summary` 的幅值与统计形态本身
  - 而不仅仅是它之后的 `coef` 或 `gain` 投影

### 当前决策

- **保留 `eventnorm + mild event_scale cap` 作为当前三元素统一框架候选版本。**
- **不再回到常规超参数扫描。**
- **后续若继续改动，应优先围绕 `event_scale` 的工作区间与更细的尾部稳定性做单因素控制，而不是继续无差别增强 `event` 强度。**

### 本地筛查结果：`S1 = A2.6 + event_summary output LayerNorm`

在 `A2.6` 之后，尝试只改一个更上游的核心因素：

- 在 `HyperedgeEventSummarizer` 输出端加入 `LayerNorm(cond_dim)`
- 其余保持 `A2.6` 不变：
  - `mix_coef` 仍为 `[0.70, 0.10, 0.10, 0.10]`
  - `coef_r >= 0.55`
  - `mix_gain_max = 0.08`

运行版本：

- `HumanActivity`：`QSHNet_EQHO_S1`
- `USHCN`：`QSHNet_EQHO_S1`

#### 结果

| 配置 | 数据集 | 轮数 | 结果 |
|------|--------|------|------|
| `QSHNet_EQHO_S1` | HumanActivity | 3 | **0.0418 ± 0.0003** |
| `QSHNet_EQHO_S1` | USHCN | 5 | **0.2072 ± 0.0461** |

对应单轮结果：

- HumanActivity：`0.04181`、`0.04206`、`0.04152`
- USHCN：`0.18965`、`0.28756`、`0.20210`、`0.18195`、`0.17460`

### `S1` 诊断结论

#### HumanActivity

- 结果与 `A2.6` 基本持平
- 说明 `event_summary` 输出端的 `LayerNorm` 不会破坏简单数据上的可训练性

#### USHCN

- 首轮 `QSHDiag` 已显示：
  - temporal `summary_mean ≈ 0.20`
  - variable `summary_mean ≈ 0.22`
- 相比 `A2.6` 在 `USHCN` 上长期出现的 `summary_mean = 5 ~ 11`，统计量被显著压低
- 但最终 `USHCN itr=5` 反而退化到 `0.2072 ± 0.0461`
- 并且方差比 `A2.6` 更大，出现了 `0.28756` 的明显坏轮

判断：

- `event_summary` 输出归一化确实改变了统计尺度
- 但这并没有转化为更好的 `USHCN` 最终表现
- 说明问题不能被简单表述为“summary 幅值过大，所以只要直接归一化输出就能解决”

### `S1` 的结构意义

1. `event_summary` 的统计量下降，不等于 `USHCN` 性能改善。
   - `S1` 把 `summary_mean` 从高能区压回了低量级；
   - 但最终 MSE 仍明显差于 `A2.6` 与主线。

2. 因此 “只在 summarizer 输出端加 `LayerNorm`” 不是正确修复方向。
   - 它更像是把表面统计压平了；
   - 但没有修复真正决定泛化效果的结构问题。

3. 当前关于 `EQHO` 的判断需要再更新一层：
   - `mix_coef` 不是主因；
   - `mix_gain` 不是充分解释；
   - `event_summary` 输出尺度本身也不是唯一主因；
   - 更深的问题可能在 `event_summary` 的信息组织方式，而不是单纯幅值。

### `S1` 后的决策更新

- `S1` 记为“统计上压住了 `summary`，但实验结果失败”的探索分支
- 后续若继续推进 `EQHO`，不应再沿着“只做 summarizer 输出归一化”继续细调
- `EQHO` 当前仍然不能替代 `eventscalecap_main`

### 本地筛查结果：`S2 = structured branch fusion`

在 `S1` 之后，继续只改 summarizer 内部融合方式：

- 不再使用 `cat + fuse`
- 改成分路投影后做静态加权聚合
- 其余仍保持 `A2.6` 的 `mix_coef / gain_max` 约束

运行版本：

- `HumanActivity`：`QSHNet_EQHO_S2`
- `USHCN`：`QSHNet_EQHO_S2`

#### 结果

| 配置 | 数据集 | 轮数 | 结果 |
|------|--------|------|------|
| `QSHNet_EQHO_S2` | HumanActivity | 3 | **0.0422 ± 0.0002** |
| `QSHNet_EQHO_S2` | USHCN | 5 | **0.2218 ± 0.0336** |

对应单轮结果：

- HumanActivity：`0.04206`、`0.04245`、`0.04212`
- USHCN：`0.20329`、`0.25454`、`0.22815`、`0.24934`、`0.17376`

### `S2` 诊断结论

- `HumanActivity` 已经出现轻度退化
- `USHCN` 则进一步恶化到比 `S1` 更差的水平
- 首轮 `QSHDiag` 中，`coef_r` 甚至在 variable 支路掉到 `0.56 ~ 0.61` 附近

判断：

- 这种“显式三路加权聚合”虽然改变了信息组织方式
- 但它破坏了 `EQHO` 需要的 real-dominant 安全结构
- 因而是明确失败方向

### 本地筛查结果：`S3 = main summary + bounded residual event/gate`

在 `S2` 失败后，进一步尝试更保守的 summarizer 结构：

- `summary = main_feat + alpha * event_feat + beta * gate_feat`
- 其中 `alpha / beta` 是有界残差系数
- 目标是让 summarizer 本身也遵循“主干主导、事件小残差修正”的主线范式

运行版本：

- `HumanActivity`：`QSHNet_EQHO_S3`
- `USHCN`：`QSHNet_EQHO_S3`

#### 结果

| 配置 | 数据集 | 轮数 | 结果 |
|------|--------|------|------|
| `QSHNet_EQHO_S3` | HumanActivity | 3 | **0.0418 ± 0.0004** |
| `QSHNet_EQHO_S3` | USHCN | 5 | **0.2173 ± 0.0340** |

对应单轮结果：

- HumanActivity：`0.04145`、`0.04219`、`0.04164`
- USHCN：`0.24181`、`0.24650`、`0.18545`、`0.23759`、`0.17537`

### `S3` 诊断结论

- `HumanActivity` 回到了安全线附近，说明 residual-style summarizer 至少没有破坏简单数据
- 但 `USHCN` 仍明显退化，且远差于 `A2.6`
- 尽管 temporal 分支较稳，variable 分支的 `coef_r` 仍会逼近安全下界

判断：

- 把 summarizer 改成 residual-style，仍不足以修复 `USHCN`
- 说明问题并不只是 summarizer 的融合拓扑
- 至少在当前 `EQHO` 设计里，继续围绕 summarizer 小修小补已经没有性价比

### `S2/S3` 后的决策更新

- `S2` 与 `S3` 一起给出更强的否定性结论：
  - 仅围绕 `event_summary` 的输出尺度或内部融合拓扑做改造，不能把 `EQHO` 拉回主线水位
- 因此 `EQHO` 当前最优版本仍然是 `A2.6`
- 但 `A2.6` 依旧明显弱于 `eventscalecap_main`
- 后续不应继续在 summarizer 层做局部结构修补

## `event density` 试验链（2026-04-18）

### 背景

在 `eventscalecap_main` 母体之上，继续只看 `event` 尾部控制，尝试回答一个更窄的问题：

- 如果 `USHCN` 的坏轮来自某些高活跃 route 把真实有效的 `event` 注入放大，
- 那么是否能通过更局部的 `event` 收缩，把坏轮压回去，同时保住 `HumanActivity` 上已经形成的改善？

这轮试验严格遵守单因素原则，只沿着 `event residual / event density` 这条线推进。

### 试验 1：`eventrescap_main`

设计：

- 不改变 `eventnorm + eventscalecap` 的基本结构
- 仅额外约束 `event_scale * event_delta` 的总残差范数
- 将其相对 `main_state` 范数限制在固定比例以内

#### 结果

| 配置 | 数据集 | 轮数 | 结果 |
|------|--------|------|------|
| `eventrescap_main` | HumanActivity | 3 | **0.04185 ± 0.00009** |
| `eventrescap_main` | USHCN | 5 | **0.1834 ± 0.0273** |

对应单轮结果：

- HumanActivity：`0.04181`、`0.04195`、`0.04178`
- USHCN：`0.15558`、`0.22176`、`0.20143`、`0.17132`、`0.16689`

#### 结论

- `HumanActivity` 继续保持改善
- 但 `USHCN` 的均值和方差都明显退化
- 说明“按主干范数对 event residual 做比例硬约束”会把高方差数据上的有效 `event` 一起压掉

因此：

- `eventrescap_main` 记为失败方向，不保留

### 试验 2：`eventdenscap_main`

设计：

- 不再直接约束 residual 范数
- 改为根据 route density 动态衰减 `event_scale`
- 且对 temporal / variable 两条路径同时生效

#### 结果

| 配置 | 数据集 | 轮数 | 结果 |
|------|--------|------|------|
| `eventdenscap_main` | HumanActivity | 3 | **0.04181 ± 0.00011** |
| `eventdenscap_main` | USHCN | 5 | **0.1891 ± 0.0347** |

对应单轮结果：

- HumanActivity：`0.04182`、`0.04191`、`0.04169`
- USHCN：`0.18448`、`0.24986`、`0.17584`、`0.16747`、`0.16795`

#### 结论

- `HumanActivity` 仍然不受伤
- 但 `USHCN` 进一步退化，甚至比 `eventrescap_main` 更差
- 说明“全路径 density-aware 抑制”不是安全保险丝
- 更准确地说，temporal 路径上的 density-aware 收缩本身就会误伤有效注入

因此：

- `eventdenscap_main` 记为失败方向，不保留

### 试验 3：`eventdensvar_main`

设计：

- 保持 temporal 路径与 `eventscalecap_main` 完全一致
- 只在 variable 路径上保留 density-aware `event_scale` 衰减
- 目标是只对更可能放大局部噪声的 variable 注入做温和控制

#### 结果

| 配置 | 数据集 | 轮数 | 结果 |
|------|--------|------|------|
| `eventdensvar_main` | HumanActivity | 3 | **0.04181 ± 0.00011** |
| `eventdensvar_main` | USHCN | 5 | **0.1703 ± 0.0058** |

对应单轮结果：

- HumanActivity：`0.04176`、`0.04194`、`0.04174`
- USHCN：`0.17625`、`0.16607`、`0.17673`、`0.16748`、`0.16481`

#### 结论

- `HumanActivity` 的改善被完整保住
- `USHCN` 相比 `eventscalecap_main` 的 `0.1663 ± 0.0027`，均值略差
- 但它显著好于 `eventrescap_main` 与 `eventdenscap_main`
- 同时方差已经收敛到 `0.0058` 量级，没有再出现严重长尾

因此当前更准确的定位是：

- `eventdensvar_main` 不是新的统一主线
- 但它已经达到“结果可接受、值得保留”的状态
- 若后续继续沿 `event density` 方向推进，应以它为直接起点，而不是回到更激进的全路径收缩

### 这轮试验链带来的新增约束

1. 不要再做全局 `event residual` 比例硬约束。
2. 不要再同时对 temporal / variable 两条路径做 density-aware 收缩。
3. 若继续推进 `event density` 方向，只看 variable 路径。
4. `eventscalecap_main / eventscalecap_itr10` 仍是统一主线母体；
   `eventdensvar_main` 则是当前可接受保留候选。
