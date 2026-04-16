# QSH-Net 当前模型架构状态总览

> **最后更新：** 2026-04-16  
> **用途：** 作为后续继续做结构试验时的唯一基准文档。优先级高于口头结论与零散实验记录。

## 1. 当前真实动力学

当前 QSH-Net 的真实训练动力学，不应再表述为：

- `Spike + Event + Quaternion` 已经稳定协同

更准确的表述是：

- `HyperIMTS + retain 调制 + quaternion 残差 + 已被唤醒且受控的 event 注入`

其中：

- 超图主干仍然承担主要结构建模
- `retain` 与 quaternion 仍然是持续活跃的主增强路径
- `event` 已经不再是死分支
- 但 `event` 仍然具有明显的数据集依赖尾部风险

## 2. 已经确认的基础事实

### 2.1 旧 M1 的真实问题

旧 M1 已通过本地参数诊断确认：

- `event_proj.weight norm = 0.0`
- `event_residual_scale = 0.0`
- 在 `USHCN` 与 `HumanActivity` 上都成立

这说明：

- 旧 M1 的 `event residual` 分支最初是死分支
- 当时真正持续学习的是：
  - `retain_log_scale`
  - `membrane_proj.weight`
  - `quat_gate`
  - `quat_h2n`

### 2.2 USHCN 的核心难点

`USHCN` 的坏轮更接近：

- 真实活跃分支在高方差数据上被放大

而不是：

- 单纯的 `event gate` 没调好
- 常规学习率 / batch size 问题

## 3. 当前代码里真正保留的结构基线

当前应当视为“活跃基线”的结构包括：

1. `retain` 温和约束
- `retain_strength_max = 0.1`

2. quaternion 残差保留
- 节点级 quaternion gate
- quaternion residual ratio bound

3. `event` 支路已被激活
- `SpikeRouter.event_log_scale = -4.5`
- `event_residual_scale` 采用 `sigmoid` 语义，而不是旧的近零残差语义

4. `event` 双支路独立归一化
- `temporal_event_norm`
- `variable_event_norm`

5. `event` 总注入量温和上界
- `event_scale = clamp(sigmoid(event_residual_scale), max=0.12)`

## 4. 当前主线候选版本

### 4.1 最稳的保守版本

- `retaincap_main`

适用定位：

- 如果目标只是追求短期最稳的工程结果

### 4.2 当前统一实验母体

- `eventnorm + mild event_scale cap`
- 对应实验标识：
  - `eventscalecap_main`
  - `eventscalecap_itr10`

当前最准确的判断：

- 它优于单纯的 `eventnorm_itr10`
- 它是目前冻结后的统一实验母体
- 后续所有跨数据集归因实验默认都相对 `eventscalecap_main` / `eventscalecap_itr10` 比较

## 5. 从上次文档更新后到现在的完整试验链

### 5.1 成功保留的方向

1. `eventnorm_main`
- 说明 `event` 活化后，独立归一化是关键控制手段

2. `eventscalecap_main`
- 说明限制 `event` 总注入量是安全方向

3. `eventscalecap_itr10`
- 说明温和上界对长重复稳定性有真实帮助

### 5.2 已明确否决的方向

1. `eventgain_main`
- 额外主干条件化 gain
- 结论：失败

2. `event_temporal_only`
- 只保留 temporal 注入
- 结论：失败

3. `eventnorm_clip`
- 直接裁剪归一化后的 `event_delta`
- 结论：失败

这些失败方向共同说明：

- 不能靠增加额外控制器来“驯服 event”
- 不能靠简单删支路来“缩小传播范围”
- 不能靠粗暴裁剪 `event_delta` 形状来消除坏轮

真正更有效的方向是：

- 保留双支路表达
- 保留独立归一化
- 对总注入量做平滑而轻量的控制

## 6. 当前数值结论

### 6.1 `USHCN itr=5`

- `eventnorm_main`: `0.1653 ± 0.0011`
- `eventscalecap_main`: `0.1663 ± 0.0027`

解读：

- `event_scale cap` 在短重复下没有显著超过 `eventnorm_main`
- 但它是安全改动，没有破坏当前框架

### 6.2 `USHCN itr=10`

- `eventnorm_itr10`: `0.1829 ± 0.0279`
- `eventscalecap_itr10`: `0.1728 ± 0.0222`

解读：

- `event_scale cap` 对长重复稳定性有真实改善
- 但最差轮仍然达到 `0.2355`
- 因此它属于“明显改善”，不是“最终解决”

### 6.3 冻结对照表

| 基线 | 数据集 | 轮数 | MSE 均值 ± std | 备注 |
|------|--------|------|----------------|------|
| `eventscalecap_main` | HumanActivity | 3 | `0.0430 ± 0.0013` | 当前统一实验母体的一部分 |
| `eventscalecap_main` | USHCN | 5 | `0.1663 ± 0.0027` | 当前统一实验母体的一部分 |
| `eventscalecap_itr10` | USHCN | 10 | `0.1728 ± 0.0222` | 长重复默认对照 |

## 7. 当前统一表述

后续继续讨论模型时，应统一使用下面这句作为基准：

> QSH-Net 当前已经形成一个由超图主干、四元数增强与事件感知注入组成的统一框架，其中事件支路已从静默模块变成真实可训练、可控，并可通过轻量总量约束改善长重复稳定性的结构组成部分；但在 `USHCN` 这类高方差数据上，长重复下仍然存在尾部坏轮，因此当前版本应被视为“改进后的主线候选”，而不是“最终稳定定版”。

## 8. 后续试验约束

后续若继续做结构试验，默认遵守以下规则：

1. 每次只改一个核心因素
2. 优先使用：
- `USHCN`
- `HumanActivity`

3. 结论标准：
- `HumanActivity` 可用 `itr=3`
- `USHCN` 至少 `itr=5`
- 更稳妥的最终判断需要 `itr=10`

4. 不再优先做：
- 常规超参数扫描
- 无差别增强 `event` 强度
- 回到已经明确否决的 `eventgain` / `temporal_only` / `eventnorm_clip`

5. 后续所有跨数据集归因实验默认相对 `eventscalecap_main` / `eventscalecap_itr10` 比较。

## 9. 相关文档

- [QSHNet 演化记录](./QSHNet_evolution.md)
- [超参数与诊断计划](./hyperparameter_tuning_plan.md)
- [论文叙事草稿](./QSHNet_paper_narrative.md)
