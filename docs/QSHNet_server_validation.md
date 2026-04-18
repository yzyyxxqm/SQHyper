# QSH-Net 服务器验证说明

> **最后更新：** 2026-04-18
> **用途：** 说明当前推荐的服务器验证对象、运行顺序、脚本入口与结果回收方式。

## 1. 当前推荐验证对象

当前推荐放到服务器上做全数据集验证的版本是：

- `eventdensvar_main`

原因：

- 它是当前工作区保留候选
- `HumanActivity` 上已经明显优于统一主线母体
- `USHCN` 上虽然略弱于统一主线母体，但已经收敛到可接受水平
- 它是后续如果继续沿 `event density` 方向推进的直接起点

如果只是要做当前默认对照，则仍使用：

- `eventscalecap_main`
- `eventscalecap_itr10`

## 2. 当前推荐验证范围

服务器验证默认覆盖 4 个数据集：

| 数据集 | 推荐 `itr` | 目的 |
|------|------------|------|
| `HumanActivity` | 5 | 用和其他主数据集一致的重复次数确认改善是否稳固 |
| `USHCN` | 10 | 重点确认高方差数据集的长重复表现 |
| `P12` | 5 | 验证医疗类中等规模数据上的迁移效果 |
| `MIMIC_III` | 5 | 验证大医疗数据集上的泛化效果 |

当前默认不纳入：

| 数据集 | 原因 |
|------|------|
| `MIMIC_IV` | 本地与服务器单次运行耗时都较高，当前阶段先不作为首轮验证对象 |

## 3. 脚本入口

### 3.1 全数据集验证脚本

直接运行：

```bash
bash scripts/QSHNet/server_validate_eventdensvar_all.sh
```

这个脚本默认：

- `HumanActivity`: `itr=5`
- `USHCN`: `itr=10`
- `P12`: `itr=5`
- `MIMIC_III`: `itr=5`
- 四个数据集并行启动

### 3.2 可选环境变量

如果服务器资源或时间窗口不同，可以临时覆盖：

```bash
USE_MULTI_GPU=1 ITR_USHCN=5 ITR_HUMAN=3 ITR_MIMIC_III=3 bash scripts/QSHNet/server_validate_eventdensvar_all.sh
```

支持的环境变量：

- `USE_MULTI_GPU`
- `ITR_USHCN`
- `ITR_HUMAN`
- `ITR_P12`
- `ITR_MIMIC_III`

### 3.3 服务器启动训练前的维度自检

在正式启动多数据集训练前，建议先执行一次：

```bash
conda run -n pyomnits python scripts/QSHNet/check_eventdensvar_dims.py
```

这个脚本会直接对当前 `eventdensvar_main` 做真实 batch 前向检查，覆盖：

- `HumanActivity`
- `USHCN`
- `P12`
- `MIMIC_III`

通过标准不是简单比较配置文件中的 `pred_len`，而是检查：

- 输入变量维度是否匹配 `enc_in`
- 目标变量维度是否匹配 `c_out`
- `x / y` 与对应 `mask` 是否同形
- `x_mark / y_mark` 是否与序列长度一致
- 模型输出 `pred` 是否与真实目标 `y` 同形

如果该脚本返回 `ALL_OK`，就说明当前服务器环境下至少不存在这四个数据集的基础维度错配问题。

## 4. 结果回收

### 4.1 训练日志

脚本会自动创建日志目录：

```bash
storage/logs/eventdensvar_main_server_validate_YYYYmmdd_HHMMSS/
```

其中包括：

- 每个数据集一个独立日志文件
- `summary.log`
- `summary.json`

### 4.2 汇总脚本

如果中途单独跑了某几个数据集，也可以手动汇总：

```bash
python scripts/QSHNet/summarize_variant_results.py \
  --model_name QSHNet \
  --model_id eventdensvar_main \
  --datasets HumanActivity,USHCN,P12,MIMIC_III
```

如需落盘 JSON：

```bash
python scripts/QSHNet/summarize_variant_results.py \
  --model_name QSHNet \
  --model_id eventdensvar_main \
  --datasets HumanActivity,USHCN,P12,MIMIC_III \
  --output_json storage/logs/eventdensvar_summary.json
```

## 5. 当前参考基线

服务器结果回来后，默认与下面几组值对比：

### 5.1 统一主线母体

- `eventscalecap_main`
  - HumanActivity: `0.0430 ± 0.0013`
  - USHCN: `0.1663 ± 0.0027`

- `eventscalecap_itr10`
  - USHCN: `0.1728 ± 0.0222`

### 5.2 当前工作区保留候选

- `eventdensvar_main`
  - HumanActivity: `0.04181 ± 0.00011`
  - USHCN: `0.1703 ± 0.0058`

### 5.3 最保守稳定版本

- `retaincap_main`
  - USHCN: `0.1673 ± 0.0033`

## 6. 当前推荐判读方式

1. 先看 `HumanActivity` 是否继续保持在 `0.0418` 量级附近。
2. 再看 `USHCN itr=10` 是否仍维持比 `eventnorm_itr10` 更稳的长重复表现。
3. 然后看 `P12 / MIMIC_III` 是否出现明显退化。
4. 最后再决定：
   - `eventdensvar_main` 是否值得升级成新的跨数据集候选
   - 或者仍只保留为 `event density` 方向的工作区保留候选

## 7. 相关文档

- [QSHNet_overview.md](/opt/Codes/PyOmniTS/docs/QSHNet_overview.md)
- [QSHNet_naming_conventions.md](/opt/Codes/PyOmniTS/docs/QSHNet_naming_conventions.md)
- [QSHNet_results_summary.md](/opt/Codes/PyOmniTS/docs/QSHNet_results_summary.md)
- [QSHNet_data_audit.md](/opt/Codes/PyOmniTS/docs/QSHNet_data_audit.md)
- [QSHNet_architecture_status.md](/opt/Codes/PyOmniTS/docs/QSHNet_architecture_status.md)
