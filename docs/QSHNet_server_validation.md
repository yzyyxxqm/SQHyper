# QSH-Net 服务器验证说明

> **最后更新：** 2026-04-18
> **用途：** 说明当前推荐的服务器验证对象、运行顺序、脚本入口与结果回收方式。

## 1. 当前推荐验证对象

当前推荐放到服务器上做全数据集验证的版本是：

- `spikeselectprop_res005_itr10`

原因：

- 它是当前最新的结构候选；
- `USHCN itr=10 = 0.16988 ± 0.00937`，max `0.19176`，优于 `eventgateconst`；
- `HumanActivity itr=5 = 0.04175 ± 0.00018`，不伤简单数据；
- 它比纯 `eventgateconst` 更适合讲「spike-driven residual propagation selection」。

上一轮服务器验证对象：

- `eventdensvar_main`

已经确认：

- `HumanActivity` 与 `P12` 可用；
- `USHCN` 坏轮明显；
- 因此不再作为当前默认服务器验证对象。

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

如果需要单独验证 `MIMIC_IV`，使用独立脚本，不和四数据集并行脚本混跑：

```bash
bash scripts/QSHNet/server_validate_spikeselectprop_res005_mimic_iv.sh
```

建议先用 1 轮 smoke test 确认服务器数据缓存和维度无误：

```bash
ITR_MIMIC_IV=1 bash scripts/QSHNet/server_validate_spikeselectprop_res005_mimic_iv.sh
```

如果显存紧张，可以降低 batch size：

```bash
BATCH_SIZE_MIMIC_IV=16 ITR_MIMIC_IV=1 bash scripts/QSHNet/server_validate_spikeselectprop_res005_mimic_iv.sh
```

## 3. 脚本入口

### 3.1 全数据集验证脚本

直接运行：

```bash
bash scripts/QSHNet/server_validate_spikeselectprop_res005_all.sh
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
USE_MULTI_GPU=1 ITR_USHCN=5 ITR_HUMAN=3 ITR_MIMIC_III=3 bash scripts/QSHNet/server_validate_spikeselectprop_res005_all.sh
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

### 4.2 `QSHDiag` 轨迹解析

当前 `QSHNet` 的 epoch 级诊断已经扩展到 route、fused residual、quaternion residual 与核心梯度范数。

如果手动运行训练，建议始终用 `tee` 保存日志：

```bash
mkdir -p storage/logs/qshdiag
conda run -n pyomnits python main.py \
  --is_training 1 \
  --collate_fn collate_fn \
  --loss MSE \
  --d_model 256 \
  --n_layers 1 \
  --n_heads 1 \
  --use_multi_gpu 0 \
  --dataset_root_path storage/datasets/USHCN \
  --model_id coupledctxadapt_main \
  --model_name QSHNet \
  --dataset_name USHCN \
  --dataset_id USHCN \
  --features M \
  --seq_len 150 \
  --pred_len 3 \
  --enc_in 5 \
  --dec_in 5 \
  --c_out 5 \
  --train_epochs 300 \
  --patience 10 \
  --val_interval 1 \
  --itr 10 \
  --batch_size 16 \
  --learning_rate 1e-3 \
  2>&1 | tee storage/logs/qshdiag/ushcn_coupledctxadapt_itr10.log
```

训练完成后解析 `QSHDiag`：

```bash
python scripts/QSHNet/extract_qshdiag.py \
  storage/logs/qshdiag/ushcn_coupledctxadapt_itr10.log \
  -o storage/logs/qshdiag/ushcn_coupledctxadapt_itr10_qshdiag.csv
```

判读重点：

1. 先按最终 `metric.json` 将 `itr` 分成好轮与坏轮。
2. 比较早期 epoch 的 `vali_loss` 是否已经分叉。
3. 再看 `retain_min / route_logit_std / fused_clip / quat_bound_ratio_max / quat_clip` 是否在坏轮中提前异常。
4. 最后看梯度范数，尤其是 `L0_membrane_w_grad`、`L0_event_proj_w_grad`、`L0_quat_gate_w_grad` 与 `L0_quat_*_grad`。

### 4.3 汇总脚本

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

## 5. 本轮服务器验证结果

本轮实际得到的 `eventdensvar_main` 服务器结果如下：

| 数据集 | 轮数 | MSE 均值 ± std | 结论 |
|------|------|----------------|------|
| `HumanActivity` | 5 | `0.04174 ± 0.00019` | 改善可复现，且稳定 |
| `USHCN` | 10 | `0.1886 ± 0.0324` | 坏轮明显，未通过统一主线门槛 |
| `P12` | 5 | `0.30092 ± 0.00062` | 结果稳定，通过 |
| `MIMIC_III` | 5 | `0.39791 ± 0.01530` | 均值可接受，但仍有单轮失稳 |

本轮结论：

1. `eventdensvar_main` 不能升级为新的统一主线。
2. 它仍可保留为 `event density` 方向的工作区候选。
3. 如果后续继续优化，重点应只放在压制 `USHCN` 坏轮，且不能破坏 `HumanActivity / P12`。

## 6. 当前参考基线

服务器结果回来后，默认与下面几组值对比：

### 6.1 统一主线母体

- `eventscalecap_main`
  - HumanActivity: `0.0430 ± 0.0013`
  - USHCN: `0.1663 ± 0.0027`

- `eventscalecap_itr10`
  - USHCN: `0.1728 ± 0.0222`

### 6.2 当前工作区保留候选

- `eventdensvar_main`
  - HumanActivity: `0.04181 ± 0.00011`
  - USHCN: `0.1703 ± 0.0058`

### 6.3 最保守稳定版本

- `retaincap_main`
  - USHCN: `0.1673 ± 0.0033`

## 7. 当前推荐判读方式

1. 先看 `HumanActivity` 是否继续保持在 `0.0418` 量级附近。
2. 再看 `USHCN itr=10` 是否仍维持比 `eventnorm_itr10` 更稳的长重复表现。
3. 然后看 `P12 / MIMIC_III` 是否出现明显退化。
4. 最后再决定：
   - `eventdensvar_main` 是否值得升级成新的跨数据集候选
   - 或者仍只保留为 `event density` 方向的工作区保留候选

按本轮实际结果，应使用更明确的判读口径：

1. `HumanActivity` 已通过。
2. `P12` 已通过。
3. `MIMIC_III` 基本可接受，但还不是完全稳定版本。
4. `USHCN` 未通过，因此当前版本不能升级为统一主线。

## 8. 相关文档

- [QSHNet_overview.md](/opt/Codes/PyOmniTS/docs/QSHNet_overview.md)
- [QSHNet_naming_conventions.md](/opt/Codes/PyOmniTS/docs/QSHNet_naming_conventions.md)
- [QSHNet_results_summary.md](/opt/Codes/PyOmniTS/docs/QSHNet_results_summary.md)
- [QSHNet_data_audit.md](/opt/Codes/PyOmniTS/docs/QSHNet_data_audit.md)
- [QSHNet_architecture_status.md](/opt/Codes/PyOmniTS/docs/QSHNet_architecture_status.md)
