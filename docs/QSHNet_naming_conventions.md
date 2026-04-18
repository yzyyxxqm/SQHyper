# QSH-Net 标准命名与引用规范

> **最后更新：** 2026-04-18
> **用途：** 固定当前阶段的标准叫法，避免不同文档中对同一版本使用不同称呼。

## 1. 使用原则

后续在文档、汇报、实验记录和口头讨论中，优先使用本页定义的标准名称。

目标只有一个：

- 同一个版本只保留一组主称呼
- 同一个结构层级只保留一组标准术语

## 2. 版本层级标准术语

### 2.1 统一主线母体

标准称呼：

- `统一主线母体`

对应版本：

- `eventscalecap_main`

适用场景：

- 当前默认短重复对照
- 所有跨数据集归因实验的主要参考点

不推荐再混用的叫法：

- 当前主线
- 当前统一版本
- 当前默认版本

这些说法太宽泛，后续容易和 `eventdensvar_main` 或 `retaincap_main` 混淆。

### 2.2 长重复主线母体

标准称呼：

- `长重复主线母体`

对应版本：

- `eventscalecap_itr10`

适用场景：

- 需要强调 `USHCN itr=10` 长重复判断时
- 需要引用当前长重复稳定性结论时

### 2.3 工作区保留候选

标准称呼：

- `工作区保留候选`

对应版本：

- `eventdensvar_main`

适用场景：

- 说明当前代码工作区实际保留的是哪一版
- 说明如果继续沿 `event density` 方向推进，应从哪里接着做

可接受的补充叫法：

- `可接受保留候选`

但默认优先写：

- `工作区保留候选`

### 2.4 最保守稳定版本

标准称呼：

- `最保守稳定版本`

对应版本：

- `retaincap_main`

适用场景：

- 如果讨论目标是“短期最稳工程结果”
- 如果需要一个相对不冒险的稳定参考

### 2.5 失败近邻版本

标准称呼：

- `失败近邻版本`

当前对应版本：

- `eventrescap_main`
- `eventdenscap_main`

适用场景：

- 说明为什么不继续沿全局 residual/density 收缩方向推进
- 给 `eventdensvar_main` 做最近邻对照

### 2.6 EQHO 探索支线

标准称呼：

- `EQHO 探索支线`

当前对应范围：

- `QSHNet_EQHO_A3`
- `QSHNet_EQHO_A25`
- `QSHNet_EQHO_A26`
- `QSHNet_EQHO_S1`
- `QSHNet_EQHO_S2`
- `QSHNet_EQHO_S3`

适用场景：

- 讨论 hyperedge-level event-conditioned quaternion refinement
- 说明为什么 EQHO 目前不能替代主线

## 3. 推荐引用模板

### 3.1 讲当前整体框架时

推荐写法：

- 当前统一主线母体是 `eventscalecap_main`，长重复主线母体是 `eventscalecap_itr10`。

### 3.2 讲当前工作区代码时

推荐写法：

- 当前工作区保留候选是 `eventdensvar_main`，但它不替代统一主线母体。

### 3.3 讲保守稳定参考时

推荐写法：

- 如果只追求短期最稳工程结果，当前最保守稳定版本仍然是 `retaincap_main`。

### 3.4 讲为什么不继续某条方向时

推荐写法：

- `eventrescap_main` 和 `eventdenscap_main` 属于失败近邻版本，不再继续细调。

### 3.5 讲 EQHO 时

推荐写法：

- `EQHO` 当前仍属于探索支线，不能替代统一主线母体。

## 4. 当前不建议再混用的说法

以下说法在当前阶段尽量少用，因为容易造成歧义：

- `当前主线`
- `当前默认版本`
- `当前保留版本`
- `现在最好的版本`
- `当前稳定版本`

原因：

- 它们没有明确说明是在说短重复、长重复、工作区代码，还是保守稳定参考。

## 5. 推荐文档分工

如果后续需要写文档或汇报，默认按下面分工引用：

- 先看 [QSHNet_overview.md](/opt/Codes/PyOmniTS/docs/QSHNet_overview.md)
- 查数值看 [QSHNet_results_summary.md](/opt/Codes/PyOmniTS/docs/QSHNet_results_summary.md)
- 查结构状态看 [QSHNet_architecture_status.md](/opt/Codes/PyOmniTS/docs/QSHNet_architecture_status.md)
- 查实验判断与约束看 [hyperparameter_tuning_plan.md](/opt/Codes/PyOmniTS/docs/hyperparameter_tuning_plan.md)
- 查完整演化过程看 [QSHNet_evolution.md](/opt/Codes/PyOmniTS/docs/QSHNet_evolution.md)
