# QSH-Net EQHO 设计规格

> 日期：2026-04-17
> 状态：已评审通过，可据此进入实现计划阶段
> 主题：以 `event-conditioned quaternion hyperedge operator` 强化 QSH-Net 的机制原创性，并保住“超图 + 四元数 + 脉冲”三元素统一框架叙事

## 1. 设计动机

当前 QSH-Net 的现实问题不是效果完全不可用，而是论文层面的机制原创性不足。

现状更接近：

- 以 HyperIMTS 风格的超图结构作为主干；
- 以 quaternion residual 作为独立增强分支；
- 以 spike / event 作为辅助注入支路。

这种结构可以讲出“三元素统一框架”的故事，但三者之间仍然偏并列，容易被审稿人理解为：

- 超图来自已有 backbone；
- 四元数来自已有增强模块；
- 脉冲 / event 来自已有 event routing 思想；
- 最终只是一个集成型拼装方法。

因此，本轮架构改造的目标不是继续做常规调参，也不是继续微调现有上界常量，而是引入一个明确属于 QSH-Net 新版本的核心机制：

> 让 event 不再只决定“注入多少”，而是进一步决定 quaternion 在 hyperedge 层“如何耦合更新”。

这会把三元素关系从“并列挂接”升级为“机制闭环”：

- 超图：结构容器与主干传播骨架；
- 脉冲 / event：由上下文偏离得到的条件信号；
- 四元数：在条件信号驱动下执行 hyperedge refinement 的耦合算子。

## 2. 总体目标

本轮设计有两个目标，优先级如下：

1. 第一优先级：提升机制原创性。
2. 第二优先级：保住“超图 + 四元数 + 脉冲”三元素统一框架叙事。

因此，本轮不再优先追求：

- 更保守的稳定性微调；
- 更优的局部指标；
- 更复杂的超参数扫描。

本轮优先追求：

- 一个可以明确定义、明确定义输入输出、明确定义消融链的新机制；
- 一个能够解释三元素为何必须共同出现的结构设计。

## 3. 核心机制定义

### 3.1 名称

本轮新机制命名为：

- `Event-Conditioned Quaternion Hyperedge Operator`
- 简称：`EQHO`

### 3.2 核心思想

当前模型中，event 主要作用在：

- 生成 `event_delta`
- 注入 temporal / variable hyperedge 更新

quaternion 主要作用在：

- 对主干输出做 residual refinement

新设计将其改写为：

- event 在 hyperedge 层先被压缩成条件摘要 `event_summary`
- `event_summary` 再生成 quaternion 四分量 mixing 参数与总强度
- quaternion 不再是静态 residual branch，而是一个由 event 驱动的 hyperedge operator
- refined hyperedge 再参与后续 node update

因此，event 的角色从：

- “注入量支路”

升级为：

- “耦合调制支路”

quaternion 的角色从：

- “独立增强残差”

升级为：

- “事件条件化的 hyperedge 耦合算子”

## 4. 新增模块

本轮新增 3 个核心模块，避免继续扩张模块数量。

### 4.1 `HyperedgeEventSummarizer`

职责：

- 将 hyperedge 主状态与 event delta 压缩成一个低维条件向量。

输入：

- `hyperedge_main`: `(B, E, D)`
- `event_delta`: `(B, E, D)`
- `event_gate_summary`: `(B, E, 1)` 或 `None`

输出：

- `event_summary`: `(B, E, D_cond)`

建议：

- `D_cond = max(16, D // 8)`
- 使用轻量 MLP，而不是直接把 `event_delta` 高维送入 mixer

设计理由：

- 让 event 从 node-level 扰动升级为 hyperedge-level condition；
- 避免控制信号过强；
- 便于诊断和论文表达。

### 4.2 `EventConditionedQuaternionMixer`

职责：

- 根据 `event_summary` 输出 quaternion refinement 的 mixing 参数与总强度。

输入：

- `event_summary`: `(B, E, D_cond)`

输出：

- `mix_coef`: `(B, E, 4)`
- `mix_gain`: `(B, E, 1)`

其中：

- `mix_coef` 对应 `r/i/j/k` 四个分量的相对权重；
- `mix_gain` 控制整体 refinement 强度。

约束：

- `mix_coef = softmax(...)`
- `mix_gain = gain_max * sigmoid(...)`

设计理由：

- event 不只控制“强度”，还控制“耦合模式”；
- 这是从 `A2` 到 `A3` 的关键跃迁。

### 4.3 `EventConditionedQuaternionHyperedgeOperator`

职责：

- 在 hyperedge 层执行条件化 quaternion refinement。

输入：

- `hyperedge_main`: `(B, E, D)`
- `mix_coef`: `(B, E, 4)`
- `mix_gain`: `(B, E, 1)`

输出：

- `hyperedge_refined`: `(B, E, D)`
- `diag_dict`

推荐内部形式：

- 显式构造 `q_r / q_i / q_j / q_k`
- 再做条件化组合：
  - `q_mix = coef_r * q_r + coef_i * q_i + coef_j * q_j + coef_k * q_k`
- 再做受控 residual 注入：
  - `out = hyperedge_main + bounded(mix_gain * q_mix)`

设计理由：

- 保持机制可解释；
- 便于做 `A1 / A2 / A3` 消融；
- 避免过早把新机制藏进旧 `QuaternionLinear` 黑箱。

## 5. 对现有模块的改动

### 5.1 `HypergraphLearner.__init__`

需新增：

- `event_cond_dim`
- `temporal_event_summarizer`
- `variable_event_summarizer`
- `temporal_eqho_mixer`
- `variable_eqho_mixer`
- `temporal_eqho_operator`
- `variable_eqho_operator`

建议：

- temporal / variable 两条分支第一版不共享参数；
- 后续如有必要，再做共享版本消融。

需新增的超参常量：

- `event_cond_dim = max(16, d_model // 8)`
- `quat_hyperedge_gain_max`
- `quat_hyperedge_ratio_max`

### 5.2 `HypergraphLearner.forward`

需新增以下中间量：

- `temporal_event_gate_summary`
- `variable_event_gate_summary`
- `temporal_event_summary`
- `variable_event_summary`
- `temporal_mix_coef`
- `temporal_mix_gain`
- `variable_mix_coef`
- `variable_mix_gain`
- `temporal_refined`
- `variable_refined`

执行顺序：

1. `SpikeRouter` 产生 `obs_base / obs_event / route_state`
2. 聚合得到 `temporal_event_delta / variable_event_delta`
3. 主干正常得到 `temporal_main / variable_main`
4. 构造 `event_gate_summary`
5. 构造 `event_summary`
6. Mixer 产生 `mix_coef / mix_gain`
7. EQHO 执行 hyperedge refinement
8. 后续 node update 使用 refined hyperedge
9. 原 `event injection` 保留为辅助项，不删除

### 5.3 `QSHDiag`

需扩展记录：

- `temporal_event_summary_norm`
- `variable_event_summary_norm`
- `temporal_mix_r_mean`
- `temporal_mix_i_mean`
- `temporal_mix_j_mean`
- `temporal_mix_k_mean`
- `variable_mix_r_mean`
- `variable_mix_i_mean`
- `variable_mix_j_mean`
- `variable_mix_k_mean`
- `temporal_mix_gain_mean`
- `variable_mix_gain_mean`
- `temporal_eqho_residual_norm`
- `variable_eqho_residual_norm`

目的：

- 解释 event 是否真的在调 quaternion；
- 判断 temporal / variable 哪个分支更激进；
- 为论文实验分析提供证据。

## 6. 初始化与稳定性约束

由于该方案属于高风险强创新版，初始化和上界控制必须严于当前母体。

### 6.1 `Summarizer`

目标：

- 初始输出接近零；
- 不让 event 一开始就强行主导 quaternion operator。

建议：

- 线性层使用小尺度初始化；
- `bias = 0`

### 6.2 `Mixer`

#### `mix_coef`

目标：

- 初始时 `r` 分量主导；
- `i/j/k` 分量仅为弱扰动。

建议：

- `coef_head.bias = [2.5, -1.0, -1.0, -1.0]`

#### `mix_gain`

目标：

- 初始 refinement 很弱；
- 退化到接近当前 baseline。

建议：

- `gain_head.weight = 0`
- `gain_head.bias` 设为使 `sigmoid(bias) * gain_max ≈ 0.02`

### 6.3 `EQHO`

目标：

- 初始接近“实值主导 + 弱虚部扰动”的安全区。

建议：

- `real_proj` 近恒等初始化；
- `i/j/k_proj` 近零初始化。

### 6.4 双保险约束

必须保留两层约束：

1. `mix_gain` 上界；
2. residual 相对 `hyperedge_main` 范数上界。

原则：

- 新机制必须是 refinement，不是替代 backbone。

## 7. 分阶段开发策略

虽然最终目标是 `A3 = Full EQHO`，实现顺序必须按可控路径推进。

### 7.1 `Patch 1`：只接 `event_summary`

只做：

- 新增 `HyperedgeEventSummarizer`
- 在 forward 中生成 summary
- 不接入 quaternion operator

目标：

- 确认 hyperedge-level event condition 可生成、可诊断、可稳定存在。

### 7.2 `Patch 2`：实现 `A1`

只做：

- quaternion refinement 迁移到 hyperedge 层
- 不使用 event 条件化

目标：

- 先验证“换到 hyperedge 层”本身是否成立。

### 7.3 `Patch 3`：实现 `A2`

只做：

- `event_summary -> mix_gain`
- `mix_coef` 固定

目标：

- 先验证 event 是否能稳定控制 quaternion refinement 强度。

### 7.4 `Patch 4`：实现 `A3`

新增：

- `event_summary -> mix_coef`
- full `r/i/j/k` 动态 mixing

目标：

- 完成完整 `EQHO`。

### 7.5 `Patch 5`：补诊断与日志

新增：

- 全部 `EQHO` 相关诊断量

目标：

- 让新机制可解释、可分析、可写论文。

### 7.6 `Patch 6`：文档与消融同步

新增：

- `A1 / A2 / A3` 消融记录
- 架构文档与计划文档同步

## 8. 消融链定义

为了防止新方法再次落回“看起来很新，实际上只是换壳”的状态，必须预先设计好消融链。

### `A0` Baseline

- 当前 `eventscalecap_main`

### `A1` Hyperedge Quaternion Only

- quaternion refinement 迁移到 hyperedge 层
- 不用 event 条件化

意义：

- 证明不是“换位置”就够了。

### `A2` Event-Gated Hyperedge Quaternion

- event 只控制 scalar gain
- 不控制 `r/i/j/k`

意义：

- 证明简单条件 gate 不足以代表完整新机制。

### `A3` Full EQHO

- event 控制 `mix_coef + mix_gain`

意义：

- 证明真正新增的贡献来自“事件驱动的 quaternion 耦合模式控制”。

## 9. 验证顺序

### 第一层：结构验证

每一 patch 首先验证：

- shape 正确
- forward 跑通
- 无 NaN / Inf
- 初始化安全

### 第二层：最小训练验证

优先数据集：

- `HumanActivity itr=3`
- `USHCN itr=5`

原则：

- 不先上大规模扫描；
- 不先上 `USHCN itr=10`；
- 先看新机制能不能活。

### 第三层：论文价值验证

确认：

- `A3` 是否相对 `A2` 学出更有解释力的 mixing pattern；
- temporal / variable 两分支是否具有差异；
- 新机制是否真的让三元素关系从并列变成闭环。

## 10. 停机条件

以下情况发生时，应暂停推进并复盘，而不是继续往下叠。

### 10.1 在 `A1` 就明显炸掉

说明：

- quaternion 搬到 hyperedge 层本身就不成立；
- 不应继续推进 `A2 / A3`。

### 10.2 在 `A2` 中 `mix_gain` 长期塌成常数

说明：

- `event_summary` 没有真正形成有效控制信号；
- 不应继续推进 full mixing。

### 10.3 在 `A3` 中 `mix_coef` 完全塌缩为固定模板

说明：

- 所谓“条件化 mixing”没有学出来；
- 机制原创性会很虚。

### 10.4 `USHCN` 短跑持续远差于当前基线

说明：

- 强创新版可能不具备工程可行性；
- 应回退到 `A2` 或中风险版。

## 11. 第一版实现边界

为了避免一次性做得过重，第一版明确不做：

- 动态重构 hypergraph adjacency / incidence；
- event memory；
- 多头 quaternion operator；
- 删除原有 event injection；
- 同时改 temporal / variable 聚合规则。

第一版只做：

- hyperedge-level event summarization；
- event-conditioned quaternion hyperedge refinement；
- 明确诊断与消融接口。

## 12. 当前结论

如果按本规格推进，新版本 QSH-Net 的核心方法贡献将从：

- “在超图主干上叠加 quaternion 与 event 支路”

升级为：

- “在 hypergraph 主干上引入一个由 event 驱动的 quaternion hyperedge operator，使事件信号不仅决定注入量，还决定高阶关系如何被耦合 refinement”

这会让论文创新点从“模块拼装”前进到“机制设计”。

## 13. 下一步

本设计规格已经完成，可进入：

- 实现计划编写；
- 按 `Patch 1 -> Patch 2 -> Patch 3 -> Patch 4` 顺序分阶段开发。
