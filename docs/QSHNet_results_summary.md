# QSH-Net 当前结果汇总

> **最后更新：** 2026-04-18
> **用途：** 单独汇总当前需要频繁引用的实验结果，避免在演化记录、调参计划和结构状态文档之间来回查找。

## 1. 使用原则

这份文档只汇总当前阶段最常用的结果表，不承担完整实验日志职责。

使用时默认遵守：

- 当前统一主线母体：`eventscalecap_main / eventscalecap_itr10`
- 当前工作区保留候选：`eventdensvar_main`
- 已明确失败的近邻版本：`eventrescap_main / eventdenscap_main`

更完整的上下文说明见：

- [QSHNet_naming_conventions.md](/opt/Codes/PyOmniTS/docs/QSHNet_naming_conventions.md)
- [QSHNet_overview.md](/opt/Codes/PyOmniTS/docs/QSHNet_overview.md)
- [QSHNet_server_validation.md](/opt/Codes/PyOmniTS/docs/QSHNet_server_validation.md)
- [QSHNet_architecture_status.md](/opt/Codes/PyOmniTS/docs/QSHNet_architecture_status.md)
- [hyperparameter_tuning_plan.md](/opt/Codes/PyOmniTS/docs/hyperparameter_tuning_plan.md)
- [QSHNet_evolution.md](/opt/Codes/PyOmniTS/docs/QSHNet_evolution.md)

## 2. 当前最常引用的版本

| 版本 | 定位 | 说明 |
|------|------|------|
| `retaincap_main` | 最保守稳定版本 | 如果只追求短期最稳工程结果，优先参考它 |
| `eventscalecap_main` | 当前统一主线母体 | 当前默认短重复对照版本 |
| `eventscalecap_itr10` | 当前统一主线长重复母体 | 当前默认长重复对照版本 |
| `eventdensvar_main` | 当前可接受保留候选 | 当前工作区保留版本，适合作为 density-aware 方向的直接起点 |

## 3. 主线与保留候选结果

### 3.1 HumanActivity

| 版本 | 轮数 | MSE 均值 ± std | 结论 |
|------|------|----------------|------|
| `eventnorm_main` | 3 | `0.0430 ± 0.0013` | 三元素统一框架的早期有效版本 |
| `eventscalecap_main` | 3 | `0.0430 ± 0.0013` | 当前统一主线母体 |
| `eventdensvar_main` | 3 | `0.04181 ± 0.00011` | 当前保留候选，在 HumanActivity 上继续改善 |

### 3.2 USHCN

| 版本 | 轮数 | MSE 均值 ± std | 结论 |
|------|------|----------------|------|
| `retaincap_main` | 5 | `0.1673 ± 0.0033` | 最保守稳定版本 |
| `eventnorm_main` | 5 | `0.1653 ± 0.0011` | 短重复下数值最佳的三元素版本之一 |
| `eventscalecap_main` | 5 | `0.1663 ± 0.0027` | 当前统一主线母体 |
| `eventdensvar_main` | 5 | `0.1703 ± 0.0058` | 略弱于主线，但已达到可接受保留水平 |
| `eventnorm_itr10` | 10 | `0.1829 ± 0.0279` | 长重复下尾部风险明显 |
| `eventscalecap_itr10` | 10 | `0.1728 ± 0.0222` | 当前长重复主线母体，优于 `eventnorm_itr10` |

## 4. 本轮 `event density` 试验链结果

### 4.1 HumanActivity

| 版本 | 轮数 | MSE 均值 ± std | 结论 |
|------|------|----------------|------|
| `eventrescap_main` | 3 | `0.04185 ± 0.00009` | HumanActivity 改善，但不足以证明方向成立 |
| `eventdenscap_main` | 3 | `0.04181 ± 0.00011` | HumanActivity 继续保持改善 |
| `eventdensvar_main` | 3 | `0.04181 ± 0.00011` | 最终保留候选 |

### 4.2 USHCN

| 版本 | 轮数 | MSE 均值 ± std | 结论 |
|------|------|----------------|------|
| `eventrescap_main` | 5 | `0.1834 ± 0.0273` | 全局 residual 比例硬约束失败 |
| `eventdenscap_main` | 5 | `0.1891 ± 0.0347` | temporal + variable 全路径 density 抑制失败 |
| `eventdensvar_main` | 5 | `0.1703 ± 0.0058` | 只保留 variable 路径 density 控制后，结果收敛到可接受水平 |

## 5. EQHO 支线结果

### 5.1 HumanActivity

| 版本 | 轮数 | MSE 均值 ± std | 结论 |
|------|------|----------------|------|
| `QSHNet_EQHO_A3` | 3 | `0.0418 ± 0.0002` | 可训练，可运行 |
| `QSHNet_EQHO_A25` | 3 | `0.0418 ± 0.0002` | 受限动态化不伤简单数据 |
| `QSHNet_EQHO_A26` | 3 | `0.0418 ± 0.0003` | gain hard cap 不伤简单数据 |
| `QSHNet_EQHO_S1` | 3 | `0.0418 ± 0.0003` | 输出归一化不伤简单数据 |
| `QSHNet_EQHO_S2` | 3 | `0.0422 ± 0.0002` | 已开始伤简单数据 |
| `QSHNet_EQHO_S3` | 3 | `0.0418 ± 0.0004` | residual-style summarizer 回到安全线附近 |

### 5.2 USHCN

| 版本 | 轮数 | MSE 均值 ± std | 结论 |
|------|------|----------------|------|
| `QSHNet_EQHO_A3` | 5 | `0.1979 ± 0.0310` | full dynamic mix_coef 明显失稳 |
| `QSHNet_EQHO_A25` | 5 | `0.2062 ± 0.0264` | 压住 mix_coef 后仍明显退化 |
| `QSHNet_EQHO_A26` | 5 | `0.1846 ± 0.0297` | 优于 A25/A3，但仍明显弱于主线 |
| `QSHNet_EQHO_S1` | 5 | `0.2072 ± 0.0461` | 只压输出尺度无效 |
| `QSHNet_EQHO_S2` | 5 | `0.2218 ± 0.0336` | 分路静态聚合失败 |
| `QSHNet_EQHO_S3` | 5 | `0.2173 ± 0.0340` | residual-style summarizer 仍失败 |

## 6. 当前推荐引用口径

如果后续需要快速引用“当前模型效果”，默认优先使用下面四条：

1. `eventscalecap_main`
   - HumanActivity `0.0430 ± 0.0013`
   - USHCN `0.1663 ± 0.0027`

2. `eventscalecap_itr10`
   - USHCN `0.1728 ± 0.0222`

3. `eventdensvar_main`
   - HumanActivity `0.04181 ± 0.00011`
   - USHCN `0.1703 ± 0.0058`

4. `retaincap_main`
   - USHCN `0.1673 ± 0.0033`

## 7. 当前结果层级结论

1. 如果要讲“统一主线”，引用 `eventscalecap_main / eventscalecap_itr10`。
2. 如果要讲“当前工作区保留的可接受候选”，引用 `eventdensvar_main`。
3. 如果要讲“最稳保守版本”，引用 `retaincap_main`。
4. 如果要讲“为什么不继续走全局 density / residual 收缩”，引用 `eventrescap_main / eventdenscap_main`。
5. 如果要讲“为什么 EQHO 目前不能替代主线”，引用 `QSHNet_EQHO_A26` 及其后续 `S1/S2/S3`。
