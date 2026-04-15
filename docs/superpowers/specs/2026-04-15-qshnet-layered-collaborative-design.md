# QSH-Net 分层协同型重构设计

> **日期：** 2026-04-15
> **状态：** 已完成设计评审，待进入实现计划
> **适用范围：** `/opt/Codes/PyOmniTS`

---

## 1. 设计目标与核心原则

### 1.1 设计目标

新版 QSH-Net 的目标不是替代 HyperIMTS，而是在保留其稳定超图骨架的前提下，引入更有解释性和适应性的增强机制，使模型在不同数据集上具备更强的通用性。

具体目标如下：

- **保住稳定骨架：** 以 HyperIMTS 的超图消息传递作为主干能力来源，保证所有数据集上的可靠性能下限。
- **保住创新点：** 四元数（Quaternion）与脉冲（Spike）继续作为模型核心组成，但职责明确、边界清晰。
- **追求通用最优：** 同一套结构能够根据数据复杂度、事件性和变量规模，自适应调整增强强度。
- **从原理出发可解释：** Spike 回答「什么时候需要增强」，Quaternion 回答「增强时如何做结构化交互」，Hypergraph 回答「信息沿什么结构传播」。

### 1.2 非目标

为避免后续设计再次走偏，本方案明确不追求以下方向：

- **不推翻 HyperIMTS 的基本拓扑：** observation node、temporal hyperedge、variable hyperedge 的三元结构保持不变。
- **不让 Quaternion 替代主线性路径：** 四元数是结构化增强器，不是新的统一主干。
- **不让 Spike 直接统治核心 attention 分布：** Spike 主要承担路由职责，而非直接重写主注意力。
- **不做数据集专用 hack：** 统一结构通过自适应机制适配数据，而不是为不同数据集写不同分支逻辑。

### 1.3 三层职责划分

- **底层：Hypergraph 结构骨架**
  - 负责稳定的结构传播。
  - 提供经过验证的主干能力。
- **中层：Spike 事件驱动路由器**
  - 负责判断哪些节点需要额外增强。
  - 控制增强分支的激活强度。
- **高层：Quaternion 结构化交互增强器**
  - 负责在多源融合瓶颈中建模标准线性层难以表达的跨组交互。

### 1.4 核心原则

- **主干永远可直达：** 所有增强都必须是残差式或门控式，原始 HyperIMTS 路径必须始终存在。
- **创新模块从安全状态启动：** 初始化时整体行为应尽量接近纯 HyperIMTS。
- **增强按节点/样本自适应：** 不同数据、不同节点对增强的需求不同，不能统一加码。
- **先改表征与融合，再改传播规则：** 优先修改信息表达与融合接口，谨慎触碰主干拓扑。
- **通用性优先于局部炫技：** 不能为了少数数据集的局部收益牺牲整体稳定性。

### 1.5 成功标准

- 在 5 个数据集上，整体均值或中位数优于当前旧版 QSH-Net，或至少不退步。
- Quaternion 与 Spike 的作用可解释，不再长期处于「几乎不动」状态。
- 模型能在低变量和高变量数据上自然退化或增强。
- 不依赖大规模破坏 HyperIMTS 主干的结构改写。

---

## 2. 从底层到表层的模块分层设计

### 2.1 新版总体结构

新版 QSH-Net 从简单的「前置 Spike + 后置 Quaternion」改为一条明确的内部机制链：

```text
obs
 -> [底层表征增强]
 -> [Spike 路由器：决定普通流 / 事件增强流]
 -> [Spike-aware n2h 消息传递]
 -> [Hyperedge 表示更新]
 -> [Quaternion-conditioned h2n 融合]
 -> [节点更新]
 -> prediction
```

核心变化包括：

- **Spike 不再只是前置缩放器，而是进入消息传递内部，控制增强路径是否开启。**
- **Quaternion 不再只是通用 residual，而是专门作用于 h2n 的多源交互瓶颈。**
- **超图消息传递从单一主流升级为「稳定主流 + 事件增强流」的双通道结构。**

### 2.2 底层：Observation 表征层重构

当前 observation node 编码过于简单，主要依赖观测值与预测指示器，导致事件判断和时序差异的压力全部堆到后续模块。

新版设计将 observation 表征拆为两部分：

- **`obs_base`：稳定基流表征**
  - 表达观测值、mask、基本时间位置等稳定信息。
  - 直接服务 HyperIMTS 主干。
- **`obs_event`：事件候选表征**
  - 表达与变量上下文的偏差、局部动态、潜在事件性。
  - 主要供 Spike 路由和事件增强流使用。

设计约束：

- `obs_base` 必须单独支撑纯主干路径。
- `obs_event` 初始不能污染主干。
- 事件信息可以强，但必须可关闭。

### 2.3 中层：Spike 从筛选器升级为路由器

旧版 SpikeSelection 的作用过弱，更像前处理技巧。

新版 Spike 的职责是输出一组路由信号，而不是只输出一个 fire/non-fire：

- **基础保留强度：** 控制节点主流信息保留多少。
- **事件增强强度：** 控制节点是否进入事件增强路径。
- **路由置信度：** 提供诊断信号，并可用于后续轻量约束。

在消息传递层面，节点信息被分成两路：

- **普通通道：** 保留主干稳定传播。
- **事件增强通道：** 仅在事件节点上提供额外信息。

新版不直接修改 attention 的 softmax 温度，而是以附加消息流的形式表达事件性，从而降低训练风险。

### 2.4 中高层：Hyperedge 更新容纳事件统计

为了让 Spike 真正改变超图内部语义，temporal hyperedge 和 variable hyperedge 在聚合节点信息时，除了接收普通流，还要接收事件增强流及其轻量统计。

具体目标：

- 时间超边不仅表示「这个时间步有哪些观测」，还表示「这个时间步是否事件密集」。
- 变量超边不仅表示「这个变量的平均状态」，还表示「这个变量当前是否活跃或异常」。

第一版实现中，这类统计只以轻量 residual bias 的方式加入，不直接暴力扩维。

### 2.5 高层：Quaternion 从通用 residual 升级为条件化融合器

旧版 Quaternion 虽然位置正确，但 gate 过于粗糙，导致它长期接近不工作状态。

新版 Quaternion 继续只服务于 `h2n` 融合瓶颈，但其强度不再由单个全局标量决定，而由当前节点的融合状态和事件信号条件化决定。

目标是让 Quaternion 只在以下情况增强：

- `obs / tg / vg` 冲突较大。
- 节点事件性较高。
- 需要更强的跨组交互，而不是简单线性融合。

### 2.6 节点更新层：主路径与增强路径分离再合流

节点最终更新保持「主干优先」：

- `base_update` 来自稳定主流。
- `enhancement_update` 来自 Spike 路由与 Quaternion 条件增强。

这样模型既保留统一主干，也允许在不同数据集上自动退化到不同的增强强度。

---

## 3. 关键数学接口与约束条件

### 3.1 统一符号

记：

- `obs ∈ R^(B×N×D)`：观测节点表示。
- `tg ∈ R^(B×N×D)`：按时间索引 gather 到节点的 temporal hyperedge 表示。
- `vg ∈ R^(B×N×D)`：按变量索引 gather 到节点的 variable hyperedge 表示。
- `H_t ∈ R^(B×L×N)`：temporal incidence matrix。
- `H_v ∈ R^(B×E×N)`：variable incidence matrix。

每层更新都应保持「主干更新 + 条件增强」的形式，而不是让增强替代主干。

### 3.2 Spike Router 接口

输入基于当前节点状态和变量上下文偏差：

```text
var_context = (H_v @ obs) / count
deviation = obs - gather(var_context, variable_idx)
router_input = cat(obs, deviation)
```

输出三元组：

- `retain_gate ∈ (0, 1)`：控制主流节点信息保留强度。
- `event_gate ∈ (0, 1)`：控制事件增强流强度。
- `route_logit ∈ R`：原始事件分数，用于诊断与后续轻量约束。

训练时主路由优先采用连续 gate；如需保留脉冲语义，可额外记录硬二值 firing 作为解释信号。

### 3.3 双通道 `n2h`

定义：

- `obs_base = retain_gate * obs`
- `obs_event = event_gate * EventProj(cat(obs, deviation))`

普通通道沿用现有主干聚合；事件增强通道单独聚合后，以 residual 方式合流：

```text
HE_new = HE_base + λ * HE_event
```

其中 `λ` 初始应为 0，以保证初始行为接近主干。

### 3.4 Hyperedge 事件统计

每个 temporal / variable hyperedge 可以接收轻量事件统计，例如：

- 事件节点质量（event mass）。
- 事件增强流的平均强度。

这些统计不直接大规模扩维，而是先用小线性层映射，再作为 residual bias 加入 hyperedge state。

### 3.5 Quaternion-conditioned `h2n` fusion

保留原始主路径：

```text
base_fusion = Linear(3D, D)(cat(obs_self_updated, tg, vg))
```

增加条件化四元数增强：

```text
quat_residual = QuaternionLinear(base_fusion)
quat_gate = GateNet(cat(base_fusion, event_gate))
fusion_out = base_fusion + quat_gate * quat_residual
```

第一版建议 `quat_gate` 仅为节点级标量，以控制复杂度。

### 3.6 恒等初始化约束

以下部分必须保持精确或近似恒等：

- `retain_gate ≈ 1`
- `event_gate ≈ 0`
- `EventProj / EventAgg` 初始输出接近 0
- hyperedge event residual gate 初始接近 0
- `QuaternionLinear` 继续采用 identity init：`W_r = I`，`W_i = W_j = W_k = 0`
- `quat_gate` 初始接近 0

理想初始行为应退化为近似原始 HyperIMTS 主干。

### 3.7 参数增长约束

第一版需严格控制复杂度：

- Event 投影仅使用小线性层或窄 MLP。
- 尽量复用现有 attention 结构，不新增第二套大模块。
- `quat_gate` 先做节点级标量，不做 feature-wise gate。
- 事件统计仅引入极少维度，不做大规模拼接。

### 3.8 可诊断性约束

每层应至少记录：

- **Spike：** `mean(retain_gate)`、`mean(event_gate)`、`hard_fire_rate`
- **Event flow：** `||event_msg|| / ||base_msg||`
- **Quaternion：** `mean(quat_gate)`、`||quat_residual|| / ||base_fusion||`

这些日志是后续判断结构是否真正被使用的基础。

---

## 4. 训练与泛化策略设计

### 4.1 总体训练哲学：先学主干，再学增强

新版 QSH-Net 的训练应遵循：

- **阶段一：主干优先**
  - 模型主要依赖 HyperIMTS 主干收敛。
  - Spike 事件流和 Quaternion 增强保持很弱。
- **阶段二：增强解冻**
  - 主干表征稳定后，增强分支逐步参与。
- **阶段三：自适应平衡**
  - 模型根据样本和数据复杂度，自主决定增强是否需要开启。

第一版不建议显式多阶段脚本，而应通过初始化和 gate 设计自然实现渐进激活。

### 4.2 泛化逻辑

统一结构之所以能跨数据集工作，不是因为增强永远很强，而是因为：

- 主干提供统一稳定基座。
- Spike 在事件性强的数据上更活跃，在平滑数据上自然退弱。
- Quaternion 在高变量、高交互复杂度数据上更强，在简单数据上自然更弱。

模型的通用性来自「默认安全退化 + 按需增强」。

### 4.3 Spike 的训练目标

Spike 不应被训练成尽可能多 fire 或尽可能少 fire，而应被训练成：

- 在关键节点上激活增强。
- 在普通节点上尽量不引入额外噪声。

建议使用轻量、非强制性的稀疏与稳定性约束，避免塌成全 0 或全 1。

### 4.4 Quaternion 的训练目标

Quaternion 不追求「处处参与」，而追求在以下情况下增强：

- 多源信息冲突明显。
- 普通线性融合不足。
- 节点事件性较高。

旧版 `alpha` 长期接近 0 的原因，很可能是 gate 自由度太低，而不是 Quaternion 本身没有价值。

### 4.5 渐进激活机制

第一版采用隐式渐进激活：

- 初始化时 `event_gate ≈ 0`、`quat_gate ≈ 0`。
- 随训练进行，gate 网络基于任务收益自主调节增强强度。

这种设计比硬切换更连续，更符合主干优先的稳定性要求。

### 4.6 泛化优先的评估重点

真正应优先关注：

- 5-run mean
- 5-run median
- std / spread
- 好运行率

不能再只盯单次最好结果。

### 4.7 失败模式预判

- **Spike 全局塌缩：** `event_gate ≈ 0`，模型退化为纯主干。
- **Spike 全局过强：** 所有节点都走增强流，USHCN / P12 先受伤。
- **Quaternion gate 长期接近 0：** 说明条件化策略仍不足。
- **Quaternion gate 过强：** 训练早期扰动主干，带来不稳定。

---

## 5. 实现优先级与最小落地版本

### 5.1 实现原则

每一轮新增结构只回答一个核心问题，避免大改动后难以归因。

### 5.2 最小落地版本 M1

M1 只做 3 个必要改动：

1. **Spike 从缩放器升级为连续路由器**
   - 输出 `retain_gate`、`event_gate`、`route_logit`。
2. **`n2h` 加入事件增强残差流**
   - 使用零初始化 residual gate 合流。
3. **Quaternion gate 改为条件化节点级 gate**
   - 替换当前全局 `alpha`。

### 5.3 第一版明确不做的内容

- 不重写 observation encoder。
- 不立即把事件统计深度写入 hyperedge state。
- 不引入显式多阶段训练脚本。
- 不一开始加入复杂辅助 loss。

### 5.4 实现顺序建议

- **Step 1：Spike Router 接口化**
- **Step 2：双通道 `n2h`**
- **Step 3：条件化 Quaternion gate**

### 5.5 第一轮验证范围

第一轮只测：

- **MIMIC_III**：高变量、当前四元数收益最明显。
- **USHCN**：低变量、最容易暴露过增强问题。

### 5.6 第一版成功判据

- 新增分支确实被调用，而不是始终静默。
- 初始化时行为接近旧版主干。
- MIMIC_III 不低于旧版太多，最好出现正收益趋势。
- USHCN 不出现系统性明显退化。

### 5.7 第二版 M2 再考虑的内容

仅当 M1 有希望时，再考虑：

- 事件统计写入 hyperedge state。
- 更丰富的 observation event features。
- 更细粒度的 gate。
- 轻量辅助 regularization。

---

## 6. 实验计划与决策门槛

### 6.1 总体策略

优先验证两个核心机制：

- Spike Router + 双通道 `n2h` 是否比旧版前置缩放更有效。
- 条件化 Quaternion gate 是否比全局 `alpha` 更能发挥四元数作用。

### 6.2 阶段 A：仅验证 Spike Router

改动：

- 将 `SpikeSelection` 改为 `SpikeRouter`。
- 暂不接入双通道 `n2h`。

必测数据：

- USHCN
- MIMIC_III

Go 条件（满足任意 2 条即可继续）：

- `event_gate` 不是全 0 / 全 1。
- USHCN 没有明显比旧版更差。
- MIMIC_III 上 `event_gate` 分布比 USHCN 更活跃。
- 训练曲线稳定。

No-Go 条件：

- `event_gate` 完全塌缩。
- 两个数据集都无区分度。
- 两个数据集都系统性恶化。

### 6.3 阶段 B：接入双通道 `n2h`

改动：

- 引入 `obs_event = event_gate * EventProj(...)`
- 引入 `HE_new = HE_base + λ * HE_event`

Go 条件：

- 事件流从近 0 上升到非零，但不过强。
- MIMIC_III 不退步，最好有收益趋势。
- USHCN 没有显著全面劣化。

No-Go 条件：

- 事件流始终为 0。
- 事件流压过 base flow，训练不稳。
- 两个数据集都明显变差。

### 6.4 阶段 C：条件化 Quaternion gate

改动：

- 用节点级条件 gate 替换当前全局 `alpha`。

Go 条件：

- `quat_gate` 不再长期贴近 0。
- 高 `event_gate` 节点上 `quat_gate` 更高。
- MIMIC_III 至少不差于旧版太多，最好提升。
- USHCN 不出现系统性过增强退化。

No-Go 条件：

- `quat_gate` 仍完全不动。
- `quat_gate` 全局过大。
- 高变量和低变量数据都退步。

### 6.5 指标体系

除 MSE / MAE 外，还需记录：

- mean / median / std / spread
- 好运行率
- `mean(event_gate)`、`mean(quat_gate)`
- `||event_msg|| / ||base_msg||`
- `||quat_residual|| / ||base_fusion||`

### 6.6 数据集优先级与实验预算

- **第一轮：** USHCN、MIMIC_III
- **第二轮：** MIMIC_IV
- **第三轮：** HumanActivity、P12

每阶段优先使用 `itr=3` 判断机制是否健康，确认方向后再扩展到 `itr=5`。

### 6.7 止损标准

应当砍掉当前子设计的情况包括：

- Spike / Quaternion 在多阶段下仍几乎不工作。
- 机制工作了，但所有数据集平均都更差。
- 只在极少数 seed 上偶然变好，均值和中位数不支持。
- 新结构需要大量训练技巧才能勉强稳定。

---

## 7. 最终设计收束与论文叙事

### 7.1 一句话定义

> QSH-Net 是一个以 HyperIMTS 为结构骨架、以 Spike 进行事件驱动路由、以 Quaternion 进行条件化结构交互增强的分层自适应 IMTS 模型。

### 7.2 三个主创新点

1. **事件驱动的超图内部路由**
   - Spike 从前处理信号升级为超图内部消息路由机制。
2. **条件化的四元数结构交互**
   - Quaternion 从静态 residual 升级为条件化的多源交互建模器。
3. **可安全退化的分层增强框架**
   - 模型默认接近 HyperIMTS，并能按需增强，在简单数据上安全退化，在复杂数据上增强表达。

### 7.3 相比 HyperIMTS 的真实变化

保留：

- 基本超图拓扑。
- 主消息传递主干。
- 原始可直达路径。

重构：

- 信息选择机制。
- `n2h` 的内部路由结构。
- `h2n` 的融合交互方式。
- 增强强度的条件化与自适应机制。

### 7.4 建议的核心研究问题

建议将研究问题表述为：

> 事件驱动路由与结构化交互增强，能否在保留超图主干稳定性的同时，提高 IMTS 模型对不同数据复杂度的自适应能力？

### 7.5 最终设计总结

新版 QSH-Net 的核心不是替代 HyperIMTS，而是在其稳定超图骨架上建立一套「先判断是否需要增强，再决定如何做结构化增强」的分层自适应机制。

三个组件的最终职责是：

- **Hypergraph：** 负责稳定结构传播。
- **Spike：** 负责判断何时需要额外增强。
- **Quaternion：** 负责在需要时提供结构化交互能力。
