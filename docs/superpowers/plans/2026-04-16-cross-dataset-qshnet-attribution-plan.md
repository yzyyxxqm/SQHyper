# QSH-Net 跨数据集结构归因与平均表现优化计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 以当前 `eventscalecap` 主线为统一母体，完成 `event / quaternion / retain` 三条增强路径的跨数据集单因素归因，优先提升 `USHCN / HumanActivity / P12 / MIMIC_III` 四数据集的平均表现。

**架构：** 固定当前基线为 `eventnorm + mild event_scale cap`，不再频繁更换拓扑。每次只改一个核心因素，先在本地用 `USHCN + HumanActivity` 做快速筛选，再把通过门槛的方案送到服务器跑 `P12 + MIMIC_III`，最后根据四数据集平均表现决定后续主攻方向。

**技术栈：** PyTorch、PyOmniTS 训练入口 `main.py`、`QSHNet` 模型、Markdown 实验文档。

---

## 文件结构

- 修改：`models/QSHNet.py`
  - 负责承载单因素结构改动，只允许围绕 `event / quaternion / retain` 其中一条路径做最小改动。
- 修改：`tests/models/test_QSHNet.py`
  - 为每个结构改动补最小单元测试，确保语义没有漂移。
- 修改：`docs/QSHNet_architecture_status.md`
  - 负责沉淀当前统一基线和阶段性结构结论。
- 修改：`docs/QSHNet_evolution.md`
  - 记录每一轮实验的结果、失败方向和保留方向。
- 修改：`docs/hyperparameter_tuning_plan.md`
  - 维护后续实验的执行边界、主线候选和下一步判断。
- 参考：`scripts/QSHNet/run_all_itr5.sh`
  - 提供四数据集标准命令模板。

## 总体执行规则

- [ ] **规则 1：固定统一基线**

基线版本固定为：

- `eventnorm + mild event_scale cap`
- `model_id` 基线命名建议：`QSHNet_eventscalecap_main`
- 长重复比较基线：`QSHNet_eventscalecap_itr10`

- [ ] **规则 2：每轮只改一个核心因素**

允许的单因素方向仅限：

- `E-minus`：弱化 `event`
- `Q-minus`：弱化 quaternion
- `R-minus`：弱化 `retain`

禁止：

- 同一轮同时改两条以上增强路径
- 再次引入已否决的 `eventgain` / `temporal_only` / `eventnorm_clip`
- 将问题重新退化为常规学习率 / batch size 大扫描

- [ ] **规则 3：分层执行**

本地快速筛选：

- `USHCN`
- `HumanActivity`

服务器完整验证：

- `P12`
- `MIMIC_III`

只有在本地两数据集都未明显退化时，才允许送服务器。

- [ ] **规则 4：统一决策指标**

每轮实验只看下面 3 件事：

1. 四数据集平均表现是否提升。
2. 是否有某个关键数据集被明显打坏。
3. 改动是否保持结构简单、可解释、可保留。

建议的拒绝门槛：

- 本地任一数据集相对基线出现明显退化，即停止该方向。
- 服务器任一医疗数据集出现明显退化，即不作为新的平均表现主线候选。

## 任务 1：冻结当前统一基线并建立对照表

**文件：**
- 修改：`docs/QSHNet_architecture_status.md`
- 修改：`docs/QSHNet_evolution.md`
- 修改：`docs/hyperparameter_tuning_plan.md`

- [ ] **步骤 1：确认基线定义只保留一版**

基线应统一写成：

```markdown
当前统一实验母体：`eventnorm + mild event_scale cap`

- `event_scale = clamp(sigmoid(event_residual_scale), max=0.12)`
- 保留双支路 `event` 注入
- 保留 `temporal_event_norm` / `variable_event_norm`
- 保留 `retain_strength_max = 0.1`
- 保留 quaternion residual ratio bound
```

- [ ] **步骤 2：列出当前基线数值**

文档中必须固定以下结果：

```text
HumanActivity itr=3, eventscalecap_main: 0.0430 ± 0.0013
USHCN itr=5, eventscalecap_main: 0.1663 ± 0.0027
USHCN itr=10, eventscalecap_itr10: 0.1728 ± 0.0222
```

- [ ] **步骤 3：写入“后续全部实验相对该基线比较”的约束**

文档新增或保留如下表述：

```markdown
后续所有跨数据集归因实验，默认相对 `eventscalecap_main / eventscalecap_itr10` 比较；
若未特别说明，不再将 `retaincap_main` 或 `eventnorm_main` 作为新的统一母体。
```

- [ ] **步骤 4：人工检查三份文档口径一致**

运行：
```bash
rg -n "eventscalecap_main|eventscalecap_itr10|统一实验母体|当前统一实验母体" \
  docs/QSHNet_architecture_status.md \
  docs/QSHNet_evolution.md \
  docs/hyperparameter_tuning_plan.md
```

预期：三份文档都明确指向同一基线，没有冲突表述。

## 任务 2：E-minus 归因实验

**文件：**
- 修改：`models/QSHNet.py`
- 测试：`tests/models/test_QSHNet.py`
- 修改：`docs/QSHNet_evolution.md`
- 修改：`docs/hyperparameter_tuning_plan.md`

**设计原则：**
- 保留 `event` 支路存在
- 不删除 `event` 拓扑
- 只显著减弱 `event` 总体影响

推荐实现优先级：

```python
self.event_scale_max = 0.06
```

或其他等价的“仅弱化 event 总量、不改拓扑”的做法。

- [ ] **步骤 1：编写失败测试，约束 E-minus 的唯一变化点**

```python
def test_event_minus_reduces_event_scale_cap_without_removing_event_branch():
    learner = HypergraphLearner(n_layers=1, d_model=8, n_heads=1, time_length=10)
    learner.event_scale_max = 0.06
    scale = learner.compute_event_scale(0)
    assert float(scale) <= 0.06 + 1e-6
```

- [ ] **步骤 2：运行测试验证失败或缺少实现约束**

运行：
```bash
conda run -n pyomnits python -m unittest tests.models.test_QSHNet
```

预期：如果测试先写入，应先红灯或因实现未同步而失败。

- [ ] **步骤 3：实现最小改动并补命名**

修改建议：

```python
# models/QSHNet.py
self.event_scale_max = 0.06
```

`model_id` 命名建议：

```text
QSHNet_eventminus_main
```

- [ ] **步骤 4：运行测试验证通过**

运行：
```bash
conda run -n pyomnits python -m unittest tests.models.test_QSHNet
```

预期：全部通过。

- [ ] **步骤 5：本地快速筛选（USHCN + HumanActivity）**

运行：
```bash
python main.py --is_training 1 --collate_fn collate_fn --loss MSE \
  --d_model 256 --n_layers 1 --n_heads 1 \
  --dataset_root_path storage/datasets/USHCN \
  --model_id QSHNet_eventminus_main --model_name QSHNet \
  --dataset_name USHCN --dataset_id USHCN \
  --features M --seq_len 150 --pred_len 3 \
  --enc_in 5 --dec_in 5 --c_out 5 \
  --train_epochs 300 --patience 10 --val_interval 1 \
  --itr 5 --batch_size 16 --learning_rate 1e-3

python main.py --is_training 1 --collate_fn collate_fn --loss MSE \
  --d_model 128 --n_layers 3 --n_heads 1 \
  --dataset_root_path storage/datasets/HumanActivity \
  --model_id QSHNet_eventminus_main --model_name QSHNet \
  --dataset_name HumanActivity --dataset_id HumanActivity \
  --features M --seq_len 3000 --pred_len 300 \
  --enc_in 12 --dec_in 12 --c_out 12 \
  --train_epochs 300 --patience 10 --val_interval 1 \
  --itr 3 --batch_size 32 --learning_rate 1e-3
```

- [ ] **步骤 6：判断是否送服务器**

通过门槛：

```text
USHCN 和 HumanActivity 都没有明显退化，且至少一者出现可解释收益。
```

- [ ] **步骤 7：服务器完整验证（P12 + MIMIC_III）**

运行：
```bash
python main.py --is_training 1 --collate_fn collate_fn --loss MSE \
  --d_model 256 --n_layers 2 --n_heads 8 \
  --dataset_root_path storage/datasets/P12 \
  --model_id QSHNet_eventminus_main --model_name QSHNet \
  --dataset_name P12 --dataset_id P12 \
  --features M --seq_len 36 --pred_len 3 \
  --enc_in 36 --dec_in 36 --c_out 36 \
  --train_epochs 300 --patience 10 --val_interval 1 \
  --itr 5 --batch_size 32 --learning_rate 1e-3

python main.py --is_training 1 --collate_fn collate_fn --loss MSE \
  --d_model 256 --n_layers 2 --n_heads 4 \
  --dataset_root_path storage/datasets/MIMIC_III \
  --model_id QSHNet_eventminus_main --model_name QSHNet \
  --dataset_name MIMIC_III --dataset_id MIMIC_III \
  --features M --seq_len 72 --pred_len 3 \
  --enc_in 96 --dec_in 96 --c_out 96 \
  --train_epochs 300 --patience 10 --val_interval 1 \
  --itr 5 --batch_size 32 --learning_rate 1e-3
```

- [ ] **步骤 8：更新文档并计算四数据集平均值**

文档中新增如下对照表：

```text
baseline_mean = mean([USHCN, HumanActivity, P12, MIMIC_III])
E-minus_mean = mean([USHCN, HumanActivity, P12, MIMIC_III])
```

- [ ] **步骤 9：Commit**

```bash
git add models/QSHNet.py tests/models/test_QSHNet.py docs/QSHNet_evolution.md docs/hyperparameter_tuning_plan.md docs/QSHNet_architecture_status.md
git commit -m "plan: run E-minus cross-dataset attribution"
```

## 任务 3：Q-minus 归因实验

**文件：**
- 修改：`models/QSHNet.py`
- 测试：`tests/models/test_QSHNet.py`
- 修改：`docs/QSHNet_evolution.md`
- 修改：`docs/hyperparameter_tuning_plan.md`

**设计原则：**
- 保留 quaternion 分支存在
- 不删除 `quat_h2n`
- 只显著减弱 quaternion residual 影响

推荐实现优先级：

```python
self.quat_residual_ratio_max = 0.10
```

- [ ] **步骤 1：编写失败测试，约束 quaternion 弱化只改比例上界**

```python
def test_q_minus_reduces_quaternion_residual_ratio_bound():
    learner = HypergraphLearner(n_layers=1, d_model=8, n_heads=1, time_length=10)
    learner.quat_residual_ratio_max = 0.10
    assert learner.quat_residual_ratio_max == 0.10
```

- [ ] **步骤 2：运行测试验证失败或约束缺失**

运行：
```bash
conda run -n pyomnits python -m unittest tests.models.test_QSHNet
```

预期：先红灯或缺少对应测试逻辑。

- [ ] **步骤 3：实现最小改动并补命名**

修改建议：

```python
# models/QSHNet.py
self.quat_residual_ratio_max = 0.10
```

`model_id` 命名建议：

```text
QSHNet_qminus_main
```

- [ ] **步骤 4：运行测试验证通过**

运行：
```bash
conda run -n pyomnits python -m unittest tests.models.test_QSHNet
```

预期：全部通过。

- [ ] **步骤 5：本地快速筛选（USHCN + HumanActivity）**

命令同任务 2，只把 `model_id` 改为 `QSHNet_qminus_main`。

- [ ] **步骤 6：若本地通过，则送服务器（P12 + MIMIC_III）**

命令同任务 2，只把 `model_id` 改为 `QSHNet_qminus_main`。

- [ ] **步骤 7：更新文档并计算四数据集平均值**

文档必须新增：

```text
Q-minus_mean = mean([USHCN, HumanActivity, P12, MIMIC_III])
```

并与 baseline / E-minus 并列。

- [ ] **步骤 8：Commit**

```bash
git add models/QSHNet.py tests/models/test_QSHNet.py docs/QSHNet_evolution.md docs/hyperparameter_tuning_plan.md docs/QSHNet_architecture_status.md
git commit -m "plan: run Q-minus cross-dataset attribution"
```

## 任务 4：R-minus 归因实验

**文件：**
- 修改：`models/QSHNet.py`
- 测试：`tests/models/test_QSHNet.py`
- 修改：`docs/QSHNet_evolution.md`
- 修改：`docs/hyperparameter_tuning_plan.md`

**设计原则：**
- 不删除 `retain`
- 只弱化 `retain` 约束强度

推荐实现优先级：

```python
self.retain_strength_max = 0.05
```

- [ ] **步骤 1：编写失败测试，约束 retain 弱化只改强度上界**

```python
def test_r_minus_reduces_retain_strength_max():
    router = SpikeRouter(d_model=8)
    router.retain_strength_max = 0.05
    strength = router.compute_retain_strength()
    assert float(strength) <= 0.05 + 1e-6
```

- [ ] **步骤 2：运行测试验证失败或约束缺失**

运行：
```bash
conda run -n pyomnits python -m unittest tests.models.test_QSHNet
```

预期：先红灯或缺少对应测试。

- [ ] **步骤 3：实现最小改动并补命名**

修改建议：

```python
# models/QSHNet.py
self.retain_strength_max = 0.05
```

`model_id` 命名建议：

```text
QSHNet_rminus_main
```

- [ ] **步骤 4：运行测试验证通过**

运行：
```bash
conda run -n pyomnits python -m unittest tests.models.test_QSHNet
```

预期：全部通过。

- [ ] **步骤 5：本地快速筛选（USHCN + HumanActivity）**

命令同任务 2，只把 `model_id` 改为 `QSHNet_rminus_main`。

- [ ] **步骤 6：若本地通过，则送服务器（P12 + MIMIC_III）**

命令同任务 2，只把 `model_id` 改为 `QSHNet_rminus_main`。

- [ ] **步骤 7：更新文档并计算四数据集平均值**

文档必须新增：

```text
R-minus_mean = mean([USHCN, HumanActivity, P12, MIMIC_III])
```

并与 baseline / E-minus / Q-minus 并列。

- [ ] **步骤 8：Commit**

```bash
git add models/QSHNet.py tests/models/test_QSHNet.py docs/QSHNet_evolution.md docs/hyperparameter_tuning_plan.md docs/QSHNet_architecture_status.md
git commit -m "plan: run R-minus cross-dataset attribution"
```

## 任务 5：统一汇总与后续主攻方向决策

**文件：**
- 修改：`docs/QSHNet_architecture_status.md`
- 修改：`docs/QSHNet_evolution.md`
- 修改：`docs/hyperparameter_tuning_plan.md`

- [ ] **步骤 1：汇总四数据集平均表现表**

文档中必须新增类似表格：

```markdown
| 版本 | USHCN | HumanActivity | P12 | MIMIC_III | 四数据集平均表现 | 结论 |
|------|-------|---------------|-----|-----------|------------------|------|
| baseline | ... | ... | ... | ... | ... | 当前统一母体 |
| E-minus | ... | ... | ... | ... | ... | ... |
| Q-minus | ... | ... | ... | ... | ... | ... |
| R-minus | ... | ... | ... | ... | ... | ... |
```

- [ ] **步骤 2：根据平均表现给出后续优先级**

决策规则：

```text
如果某条路径弱化后平均表现变好，说明该路径当前可能过强或职责错配；
如果某条路径弱化后平均表现显著变差，说明该路径值得继续加大投入；
如果某条路径弱化后结果基本不变，说明它当前不是主要限制项。
```

- [ ] **步骤 3：更新总览文档中的“下一阶段主攻方向”**

文档必须明确写出：

```markdown
下一阶段不再盲目修补单个数据集，而是围绕平均表现最敏感的增强路径继续优化。
```

- [ ] **步骤 4：验证文档已写全**

运行：
```bash
rg -n "四数据集平均表现|E-minus|Q-minus|R-minus|下一阶段主攻方向" \
  docs/QSHNet_architecture_status.md \
  docs/QSHNet_evolution.md \
  docs/hyperparameter_tuning_plan.md
```

预期：三份文档都包含最新归因结论。

- [ ] **步骤 5：Commit**

```bash
git add docs/QSHNet_architecture_status.md docs/QSHNet_evolution.md docs/hyperparameter_tuning_plan.md
git commit -m "docs: summarize cross-dataset attribution conclusions"
```
