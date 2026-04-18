# QSH-Net 数据路径、结构与维度核对

> **最后更新：** 2026-04-18
> **用途：** 记录当前本机数据目录、原始数据结构、代码读取逻辑以及真实 batch 维度与当前模型的匹配情况，避免服务器环境配置错误或运行时张量维度不一致。

## 1. 结论摘要

当前已确认两件事：

1. 数据读取逻辑分为两类。
2. 当前 `eventdensvar_main` 在 `HumanActivity / USHCN / P12 / MIMIC_III` 上已经通过真实 batch 前向维度检查，没有发现输入输出维度不匹配问题。

其中数据读取逻辑分为两类：

1. `USHCN / P12 / MIMIC_III / MIMIC_IV`
- 实际走 `tsdm` 数据管线
- 真正依赖的是：
  - `~/.tsdm/rawdata/...`
  - `~/.tsdm/datasets/...`
- **不是** `~/.tsdm/rawdatas/...`
- 也不是主要依赖 `storage/datasets/...`

2. `HumanActivity`
- 不走 `tsdm` 原始数据目录
- 直接读取 `storage/datasets/HumanActivity/raw/` 和 `processed/`

## 2. 当前本机实际目录

### 2.1 `~/.tsdm`

当前本机存在：

- `/home/wgx/.tsdm/rawdata`
- `/home/wgx/.tsdm/datasets`

当前本机不存在：

- `/home/wgx/.tsdm/rawdatas`

所以如果服务器上也按当前代码跑，目录名必须是：

- `~/.tsdm/rawdata`

而不是：

- `~/.tsdm/rawdatas`

## 3. 各数据集核对结果

### 3.1 USHCN

原始文件：

- `/home/wgx/.tsdm/rawdata/USHCN_DeBrouwer2019/small_chunked_sporadic.csv`

文件头部结构：

- `ID,Time,Value_0..Value_4,Mask_0..Mask_4`

代码入口：

- [USHCN.py](/opt/Codes/PyOmniTS/data/data_provider/datasets/USHCN.py)

判断：

- 原始文件存在
- 表头结构与 `ushcn_debrouwer2019` 任务格式相符
- 当前没有发现路径或字段层面的明显问题

### 3.2 P12

原始文件：

- `/home/wgx/.tsdm/rawdata/Physionet2012/set-a.tar.gz`
- `/home/wgx/.tsdm/rawdata/Physionet2012/set-b.tar.gz`
- `/home/wgx/.tsdm/rawdata/Physionet2012/set-c.tar.gz`

缓存目录：

- `/home/wgx/.tsdm/datasets/Physionet2012/`

代码入口：

- [P12.py](/opt/Codes/PyOmniTS/data/data_provider/datasets/P12.py)

判断：

- 当前 rawdata 和 datasets 都存在
- `P12` 通过 `Physionet2012` 任务读取，不依赖 `storage/datasets/P12` 的原始文件布局
- 当前没有看到明显路径问题

### 3.3 MIMIC_III

原始文件：

- `/home/wgx/.tsdm/rawdata/MIMIC_III_DeBrouwer2019/complete_tensor.csv`

文件头部结构：

- `UNIQUE_ID, LABEL_CODE, TIME_STAMP, VALUENUM, MEAN, STD, VALUENORM`

缓存目录：

- `/home/wgx/.tsdm/datasets/MIMIC_III_DeBrouwer2019/`
- 已存在：
  - `metadata.parquet`
  - `timeseries.parquet`

代码入口：

- [MIMIC_III.py](/opt/Codes/PyOmniTS/data/data_provider/datasets/MIMIC_III.py)

判断：

- 原始文件与处理后缓存都存在
- 当前本机环境下，`MIMIC_III` 数据侧状态完整

### 3.4 MIMIC_IV

原始文件：

- `/home/wgx/.tsdm/rawdata/MIMIC_IV_Bilos2021/full_dataset.csv`

文件头部结构：

- `hadm_id,time_stamp,Value_label_x,Mask_label_x,...`

代码入口：

- [MIMIC_IV.py](/opt/Codes/PyOmniTS/data/data_provider/datasets/MIMIC_IV.py)

当前额外观察：

- `/home/wgx/.tsdm/datasets/MIMIC_IV_Bilos2021/` 目录存在
- 但当前没有看到像 `MIMIC_III` 那样已经生成好的 parquet 缓存文件

判断：

- 原始文件存在，字段结构看起来合理
- 但当前本机处理缓存状态不如 `MIMIC_III` 明确
- 这也是当前先不把 `MIMIC_IV` 纳入首轮服务器验证的一个合理原因

### 3.5 HumanActivity

当前使用目录：

- `storage/datasets/HumanActivity/raw/ConfLongDemo_JSI.txt`
- `storage/datasets/HumanActivity/processed/data.pt`

原始文件头部结构：

- `record_id, tag_id, time, date, val1, val2, val3, label`

代码入口：

- [HumanActivity.py](/opt/Codes/PyOmniTS/data/data_provider/datasets/HumanActivity.py)
- [HumanActivity dependency](/opt/Codes/PyOmniTS/data/dependencies/HumanActivity/HumanActivity.py)

判断：

- `HumanActivity` 完全不依赖 `~/.tsdm/rawdata`
- 直接使用 `storage/datasets/HumanActivity`
- 当前 raw 与 processed 都存在，没有看到明显目录问题

## 4. 代码层面的关键判断

### 4.1 `dataset_root_path` 对 tsdm 数据集不是决定性路径

对应代码：

- [tsdmDataset.py](/opt/Codes/PyOmniTS/data/dependencies/tsdm/PyOmniTS/tsdmDataset.py)

关键事实：

- `tsdmDataset` 保存了 `configs.dataset_root_path`
- 但 `_preprocess_base(task)` 实际取的是 `task.get_dataset(...)`
- 底层真正依赖的是 `tsdm` 的 `RAWDATA_DIR` / `DATASET_DIR`

这意味着：

- 对 `USHCN / P12 / MIMIC_III / MIMIC_IV`，`--dataset_root_path` 更多是配置层兼容参数
- 真正要保证的是服务器上的 `~/.tsdm/rawdata` 和 `~/.tsdm/datasets` 完整

### 4.2 `HumanActivity` 是单独路径体系

对应代码：

- [HumanActivity.py](/opt/Codes/PyOmniTS/data/data_provider/datasets/HumanActivity.py)

关键事实：

- 它直接实例化 `HumanActivity(root=self.configs.dataset_root_path)`
- 所以它确实依赖 `storage/datasets/HumanActivity`

## 5. 当前模型可用维度核对

### 5.1 核对方法

本轮检查不是只看配置文件中的 `seq_len / pred_len / enc_in / c_out`，而是直接：

1. 用当前配置真实构造训练集 dataloader。
2. 取一个 batch。
3. 直接把 batch 喂给当前 `QSHNet(eventdensvar_main)` 前向。
4. 验证输入、mask、time mark 和模型输出是否一致。

对应脚本：

- [check_eventdensvar_dims.py](/opt/Codes/PyOmniTS/scripts/QSHNet/check_eventdensvar_dims.py)

检查项为：

- `x.shape[-1] == enc_in`
- `x_mask.shape == x.shape`
- `y.shape[-1] == c_out`
- `y_mask.shape == y.shape`
- `x_mark.shape[:2] == x.shape[:2]`
- `y_mark.shape[:2] == y.shape[:2]`
- `pred.shape == true.shape == y.shape`

### 5.2 实际核对结果

#### HumanActivity

- `x`: `(2, 98, 12)`
- `x_mask`: `(2, 98, 12)`
- `x_mark`: `(2, 98, 1)`
- `y`: `(2, 11, 12)`
- `y_mask`: `(2, 11, 12)`
- `y_mark`: `(2, 11, 1)`
- `pred`: `(2, 11, 12)`

判断：

- 变量维度与 `enc_in=c_out=12` 一致
- mask 与时间标记长度一致
- 模型输出与目标张量完全对齐

#### USHCN

- `x`: `(2, 287, 5)`
- `x_mask`: `(2, 287, 5)`
- `x_mark`: `(2, 287, 1)`
- `y`: `(2, 3, 5)`
- `y_mask`: `(2, 3, 5)`
- `y_mark`: `(2, 3, 1)`
- `pred`: `(2, 3, 5)`

判断：

- 变量维度与 `enc_in=c_out=5` 一致
- 当前模型可以正常前向

#### P12

- `x`: `(2, 35, 36)`
- `x_mask`: `(2, 35, 36)`
- `x_mark`: `(2, 35, 1)`
- `y`: `(2, 3, 36)`
- `y_mask`: `(2, 3, 36)`
- `y_mark`: `(2, 3, 1)`
- `pred`: `(2, 3, 36)`

判断：

- 变量维度与 `enc_in=c_out=36` 一致
- 当前模型可以正常前向

#### MIMIC_III

- `x`: `(2, 72, 96)`
- `x_mask`: `(2, 72, 96)`
- `x_mark`: `(2, 72, 1)`
- `y`: `(2, 3, 96)`
- `y_mask`: `(2, 3, 96)`
- `y_mark`: `(2, 3, 1)`
- `pred`: `(2, 3, 96)`

判断：

- 变量维度与 `enc_in=c_out=96` 一致
- 当前模型可以正常前向

### 5.3 对 irregular 数据必须特别注意的点

不能把配置文件中的 `pred_len` 机械理解为 batch 里的真实目标时间长度。

例如：

- `HumanActivity` 配置中使用 `pred_len=300`
- 但当前真实 batch 中 `y.shape[1] = 11`

这是当前 irregular `collate_fn` 的正常行为，不代表代码有 bug，也不代表模型维度不匹配。

因此，当前项目里判断“数据是否能喂给模型”，应优先看：

1. 特征维度是否与 `enc_in / c_out` 一致。
2. `x / y` 与各自 `mask` 是否完全同形。
3. `x_mark / y_mark` 的前两维是否与对应序列一致。
4. 模型输出 `pred` 是否与目标 `y` 同形。

## 6. 当前未发现的明显问题

目前没有发现下面这些问题：

- `USHCN` 原始表头和任务读取格式不匹配
- `MIMIC_III` 原始文件字段名不匹配
- `HumanActivity` 原始文本字段数不匹配
- `P12` 缺少 raw tar 文件
- `HumanActivity / USHCN / P12 / MIMIC_III` 与当前 `QSHNet` 前向存在维度不匹配

## 7. 当前确实需要注意的点

1. 服务器路径名必须是 `~/.tsdm/rawdata`，不是 `~/.tsdm/rawdatas`。
2. `USHCN / P12 / MIMIC_III / MIMIC_IV` 的真正数据来源是 `tsdm` 目录，不是 `storage/datasets/...`。
3. `HumanActivity` 则相反，真正依赖 `storage/datasets/HumanActivity`。
4. `MIMIC_IV` 当前本机缓存状态不完整，因此本轮先不纳入服务器首轮验证是合理的。
5. irregular 数据集的真实 batch 时间长度可能小于配置中的 `seq_len / pred_len`，不能因此误判为维度错误。

## 8. 建议的服务器上线前检查

在服务器上至少检查下面三件事：

```bash
find ~/.tsdm/rawdata -maxdepth 2 -mindepth 1 | sort
find ~/.tsdm/datasets -maxdepth 2 -mindepth 1 | sort
find storage/datasets/HumanActivity -maxdepth 2 -mindepth 1 | sort
```

然后再补一条模型侧快速检查：

```bash
conda run -n pyomnits python scripts/QSHNet/check_eventdensvar_dims.py
```

只要这几项都通过，当前这批验证脚本的数据侧和张量维度侧前提就基本成立。
