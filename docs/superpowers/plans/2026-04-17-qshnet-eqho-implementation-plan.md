# QSH-Net EQHO 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 在当前 `eventscalecap` 母体上分阶段实现 `Event-Conditioned Quaternion Hyperedge Operator (EQHO)`，把 event 从辅助注入支路升级为 quaternion hyperedge refinement 的条件控制信号。

**架构：** 先在 hyperedge 层引入 `event_summary`，再分阶段实现 hyperedge-level quaternion refinement、event-conditioned scalar gain，最后实现 full `r/i/j/k` mixing。整个实现严格保持 “主干 backbone + 受控 refinement” 结构，不允许新机制直接替代现有 hypergraph backbone。所有实验先经过 `HumanActivity itr=3` 和 `USHCN itr=5` 本地筛查，再考虑更长重复。

**技术栈：** Python、PyTorch、`unittest`、现有 `QSHNet` / `Exp_Main` 训练框架

---

## 文件结构

### 本轮会修改的文件

- 修改：`models/QSHNet.py`
  - 职责：新增 `HyperedgeEventSummarizer`、`EventConditionedQuaternionMixer`、`EventConditionedQuaternionHyperedgeOperator`，并接入 `HypergraphLearner`
- 修改：`tests/models/test_QSHNet.py`
  - 职责：为新模块、初始化安全性、残差上界和完整 forward 增加单测
- 修改：`exp/exp_main.py`
  - 职责：在现有 `QSHDiag` 中补充 EQHO 相关诊断统计
- 修改：`docs/QSHNet_architecture_status.md`
  - 职责：记录 EQHO 主线的目标、实施状态与阶段性结论
- 修改：`docs/QSHNet_evolution.md`
  - 职责：追加 `A1 / A2 / A3` 的实验记录
- 修改：`docs/hyperparameter_tuning_plan.md`
  - 职责：记录 EQHO 开发阶段的实验路线与停机条件

### 本轮会新增的文件

- 创建：`docs/superpowers/specs/2026-04-17-qshnet-eqho-design.md`
  - 职责：本轮 EQHO 方案的正式规格
- 创建：`docs/superpowers/plans/2026-04-17-qshnet-eqho-implementation-plan.md`
  - 职责：本实现计划

### 不做的事情

- 不修改数据集 loader、collate 逻辑
- 不引入新的 optimizer 或 scheduler
- 不重构现有超图 incidence / adjacency
- 不删除原有 `event injection`
- 不同时推进 event memory 或动态结构重构

---

### 任务 1：冻结当前基线并补齐基线约束测试

**文件：**
- 修改：`/opt/Codes/PyOmniTS/models/QSHNet.py`
- 修改：`/opt/Codes/PyOmniTS/tests/models/test_QSHNet.py`

- [ ] **步骤 1：确认当前基线常量的失败测试已经存在或补齐**

```python
def test_quaternion_residual_respects_the_baseline_025_ratio_upper_bound(self):
    learner = HypergraphLearner(n_layers=1, d_model=8, n_heads=1, time_length=4)
    linear_out = torch.ones(2, 3, 8)
    quat_out = torch.full((2, 3, 8), 20.0)
    alpha = torch.ones(2, 3, 1)

    bounded_residual = learner.bound_quaternion_residual(
        linear_out=linear_out,
        quat_out=quat_out,
        alpha=alpha,
    )

    residual_norm = bounded_residual.norm(dim=-1)
    linear_norm = linear_out.norm(dim=-1)
    self.assertTrue(torch.all(residual_norm <= 0.25 * linear_norm + 1e-6))
```

- [ ] **步骤 2：运行现有基线单测**

运行：

```bash
conda run -n pyomnits python -m unittest tests.models.test_QSHNet
```

预期：

- PASS
- 输出 `Ran ... tests ... OK`

- [ ] **步骤 3：检查基线常量仍是统一母体**

应保留：

```python
self.retain_strength_max = 0.10
self.event_scale_max = 0.12
self.quat_residual_ratio_max = 0.25
```

- [ ] **步骤 4：再次运行完整测试确认基线冻结**

运行：

```bash
conda run -n pyomnits python -m unittest tests.models.test_QSHNet
```

预期：

- PASS

- [ ] **步骤 5：Commit**

```bash
git add models/QSHNet.py tests/models/test_QSHNet.py
git commit -m "test: freeze qshnet baseline before eqho"
```

---

### 任务 2：新增 `HyperedgeEventSummarizer` 并只接入 `event_summary`

**文件：**
- 修改：`/opt/Codes/PyOmniTS/models/QSHNet.py`
- 修改：`/opt/Codes/PyOmniTS/tests/models/test_QSHNet.py`

- [ ] **步骤 1：先写失败测试，覆盖 summarizer 的 shape 与数值稳定性**

```python
def test_hyperedge_event_summarizer_preserves_expected_shape(self):
    summarizer = HyperedgeEventSummarizer(d_model=8, cond_dim=4)
    hyperedge_main = torch.randn(2, 5, 8)
    event_delta = torch.randn(2, 5, 8)
    event_gate_summary = torch.rand(2, 5, 1)

    event_summary = summarizer(
        hyperedge_main=hyperedge_main,
        event_delta=event_delta,
        event_gate_summary=event_gate_summary,
    )

    self.assertEqual(event_summary.shape, torch.Size((2, 5, 4)))
    self.assertFalse(torch.isnan(event_summary).any())
    self.assertFalse(torch.isinf(event_summary).any())
```

- [ ] **步骤 2：运行新测试确认失败**

运行：

```bash
conda run -n pyomnits python -m unittest tests.models.test_QSHNet.TestQSHNet.test_hyperedge_event_summarizer_preserves_expected_shape
```

预期：

- FAIL
- 报错 `NameError: name 'HyperedgeEventSummarizer' is not defined`

- [ ] **步骤 3：新增 `HyperedgeEventSummarizer` 类**

```python
class HyperedgeEventSummarizer(nn.Module):
    def __init__(self, d_model, cond_dim, use_gate_summary=True):
        super().__init__()
        self.use_gate_summary = use_gate_summary
        self.main_proj = nn.Linear(d_model, cond_dim)
        self.event_proj = nn.Linear(d_model, cond_dim)
        self.gate_proj = nn.Linear(1, cond_dim) if use_gate_summary else None
        in_dim = cond_dim * (3 if use_gate_summary else 2)
        self.fuse = nn.Linear(in_dim, cond_dim)
        self.activation = nn.SiLU()

        for layer in [self.main_proj, self.event_proj, self.fuse]:
            nn.init.normal_(layer.weight, mean=0.0, std=1e-2)
            nn.init.zeros_(layer.bias)
        if self.gate_proj is not None:
            nn.init.normal_(self.gate_proj.weight, mean=0.0, std=1e-2)
            nn.init.zeros_(self.gate_proj.bias)

    def forward(self, hyperedge_main, event_delta, event_gate_summary=None):
        main_feat = self.main_proj(hyperedge_main)
        event_feat = self.event_proj(event_delta)
        feats = [main_feat, event_feat]
        if self.use_gate_summary and event_gate_summary is not None:
            feats.append(self.gate_proj(event_gate_summary))
        summary = self.fuse(torch.cat(feats, dim=-1))
        return self.activation(summary)
```

- [ ] **步骤 4：在 `HypergraphLearner.__init__` 中声明 `event_cond_dim`、`temporal_event_summarizer`、`variable_event_summarizer`**

```python
self.event_cond_dim = max(16, d_model // 8)
self.temporal_event_summarizer = HyperedgeEventSummarizer(d_model, self.event_cond_dim)
self.variable_event_summarizer = HyperedgeEventSummarizer(d_model, self.event_cond_dim)
```

- [ ] **步骤 5：在 `HypergraphLearner.forward` 中只生成但不消费 `event_summary`**

```python
temporal_event_gate_summary = (
    temporal_incidence_matrix @ route_state["event_gate"].unsqueeze(-1)
) / temporal_incidence_matrix.sum(-1, keepdim=True).clamp(min=1)
variable_event_gate_summary = (
    variable_incidence_matrix @ route_state["event_gate"].unsqueeze(-1)
) / variable_incidence_matrix.sum(-1, keepdim=True).clamp(min=1)

temporal_event_summary = self.temporal_event_summarizer(
    temporal_main, temporal_event_delta, temporal_event_gate_summary
)
variable_event_summary = self.variable_event_summarizer(
    variable_main, variable_event_delta, variable_event_gate_summary
)
```

- [ ] **步骤 6：运行模块级测试和完整测试**

运行：

```bash
conda run -n pyomnits python -m unittest tests.models.test_QSHNet.TestQSHNet.test_hyperedge_event_summarizer_preserves_expected_shape
conda run -n pyomnits python -m unittest tests.models.test_QSHNet
```

预期：

- PASS

- [ ] **步骤 7：Commit**

```bash
git add models/QSHNet.py tests/models/test_QSHNet.py
git commit -m "feat: add qshnet hyperedge event summarizer"
```

---

### 任务 3：实现 `A1`，把 quaternion refinement 迁移到 hyperedge 层

**文件：**
- 修改：`/opt/Codes/PyOmniTS/models/QSHNet.py`
- 修改：`/opt/Codes/PyOmniTS/tests/models/test_QSHNet.py`

- [ ] **步骤 1：先写失败测试，约束 hyperedge operator 的输出与上界**

```python
def test_eqho_operator_bounds_residual_relative_to_main_norm(self):
    operator = EventConditionedQuaternionHyperedgeOperator(d_model=8, residual_ratio_max=0.2)
    hyperedge_main = torch.ones(2, 4, 8)
    mix_coef = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]).expand(2, 4, 4)
    mix_gain = torch.ones(2, 4, 1)

    refined, diag = operator(
        hyperedge_main=hyperedge_main,
        mix_coef=mix_coef,
        mix_gain=mix_gain,
    )

    residual = refined - hyperedge_main
    residual_norm = residual.norm(dim=-1)
    main_norm = hyperedge_main.norm(dim=-1)
    self.assertTrue(torch.all(residual_norm <= 0.2 * main_norm + 1e-6))
    self.assertIn("residual_norm_mean", diag)
```

- [ ] **步骤 2：运行新测试确认失败**

运行：

```bash
conda run -n pyomnits python -m unittest tests.models.test_QSHNet.TestQSHNet.test_eqho_operator_bounds_residual_relative_to_main_norm
```

预期：

- FAIL
- 报错 `NameError: name 'EventConditionedQuaternionHyperedgeOperator' is not defined`

- [ ] **步骤 3：新增 `EventConditionedQuaternionHyperedgeOperator` 的 `A1` 版本**

```python
class EventConditionedQuaternionHyperedgeOperator(nn.Module):
    def __init__(self, d_model, residual_ratio_max):
        super().__init__()
        self.residual_ratio_max = residual_ratio_max
        self.real_proj = nn.Linear(d_model, d_model)
        self.i_proj = nn.Linear(d_model, d_model)
        self.j_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        nn.init.eye_(self.real_proj.weight)
        nn.init.zeros_(self.real_proj.bias)
        for proj in [self.i_proj, self.j_proj, self.k_proj]:
            nn.init.normal_(proj.weight, mean=0.0, std=1e-3)
            nn.init.zeros_(proj.bias)

    def forward(self, hyperedge_main, mix_coef, mix_gain):
        q_r = self.real_proj(hyperedge_main)
        q_i = self.i_proj(hyperedge_main)
        q_j = self.j_proj(hyperedge_main)
        q_k = self.k_proj(hyperedge_main)
        coef_r = mix_coef[..., 0:1]
        coef_i = mix_coef[..., 1:2]
        coef_j = mix_coef[..., 2:3]
        coef_k = mix_coef[..., 3:4]
        q_mix = coef_r * q_r + coef_i * q_i + coef_j * q_j + coef_k * q_k
        residual = mix_gain * q_mix
        main_norm = hyperedge_main.norm(dim=-1, keepdim=True)
        residual_norm = residual.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        scale = torch.clamp(self.residual_ratio_max * main_norm / residual_norm, max=1.0)
        residual = residual * scale
        refined = hyperedge_main + residual
        return refined, {"residual_norm_mean": residual.norm(dim=-1).mean()}
```

- [ ] **步骤 4：在 `HypergraphLearner.__init__` 中声明 temporal / variable operator，并使用固定 `mix_coef / mix_gain`**

```python
self.temporal_eqho_operator = EventConditionedQuaternionHyperedgeOperator(d_model, residual_ratio_max=0.20)
self.variable_eqho_operator = EventConditionedQuaternionHyperedgeOperator(d_model, residual_ratio_max=0.20)
```

固定控制信号：

```python
temporal_mix_coef = temporal_main.new_tensor([1.0, 0.0, 0.0, 0.0]).view(1, 1, 4).expand(B, temporal_main.shape[1], 4)
variable_mix_coef = variable_main.new_tensor([1.0, 0.0, 0.0, 0.0]).view(1, 1, 4).expand(B, variable_main.shape[1], 4)
temporal_mix_gain = temporal_main.new_full((B, temporal_main.shape[1], 1), 0.02)
variable_mix_gain = variable_main.new_full((B, variable_main.shape[1], 1), 0.02)
```

- [ ] **步骤 5：用 refined hyperedge 替换后续 node update 使用的状态**

```python
temporal_refined, _ = self.temporal_eqho_operator(temporal_main, temporal_mix_coef, temporal_mix_gain)
variable_refined, _ = self.variable_eqho_operator(variable_main, variable_mix_coef, variable_mix_gain)
```

- [ ] **步骤 6：运行完整测试**

运行：

```bash
conda run -n pyomnits python -m unittest tests.models.test_QSHNet
```

预期：

- PASS

- [ ] **步骤 7：Commit**

```bash
git add models/QSHNet.py tests/models/test_QSHNet.py
git commit -m "feat: add qshnet hyperedge quaternion operator"
```

---

### 任务 4：实现 `A2`，让 event 控制 quaternion refinement 的 scalar gain

**文件：**
- 修改：`/opt/Codes/PyOmniTS/models/QSHNet.py`
- 修改：`/opt/Codes/PyOmniTS/tests/models/test_QSHNet.py`

- [ ] **步骤 1：先写失败测试，约束 mixer 输出与 gain 上界**

```python
def test_eqho_mixer_outputs_normalized_gain_with_upper_bound(self):
    mixer = EventConditionedQuaternionMixer(cond_dim=4, gain_max=0.15)
    event_summary = torch.randn(2, 5, 4)
    mix_coef, mix_gain = mixer(event_summary)
    self.assertEqual(mix_coef.shape, torch.Size((2, 5, 4)))
    self.assertEqual(mix_gain.shape, torch.Size((2, 5, 1)))
    self.assertTrue(torch.allclose(mix_coef.sum(dim=-1), torch.ones(2, 5), atol=1e-6))
    self.assertTrue(torch.all(mix_gain <= 0.15 + 1e-6))
```

- [ ] **步骤 2：运行新测试确认失败**

运行：

```bash
conda run -n pyomnits python -m unittest tests.models.test_QSHNet.TestQSHNet.test_eqho_mixer_outputs_normalized_gain_with_upper_bound
```

预期：

- FAIL
- 报错 `NameError: name 'EventConditionedQuaternionMixer' is not defined`

- [ ] **步骤 3：新增 `EventConditionedQuaternionMixer` 类，但第一版只实际使用 `mix_gain`**

```python
class EventConditionedQuaternionMixer(nn.Module):
    def __init__(self, cond_dim, gain_max):
        super().__init__()
        self.gain_max = gain_max
        self.coef_head = nn.Linear(cond_dim, 4)
        self.gain_head = nn.Linear(cond_dim, 1)
        nn.init.zeros_(self.coef_head.weight)
        with torch.no_grad():
            self.coef_head.bias.copy_(torch.tensor([2.5, -1.0, -1.0, -1.0]))
        nn.init.zeros_(self.gain_head.weight)
        nn.init.constant_(self.gain_head.bias, -1.9)

    def forward(self, event_summary):
        mix_coef = torch.softmax(self.coef_head(event_summary), dim=-1)
        mix_gain = self.gain_max * torch.sigmoid(self.gain_head(event_summary))
        return mix_coef, mix_gain
```

- [ ] **步骤 4：在 `HypergraphLearner.__init__` 中声明 temporal / variable mixer**

```python
self.quat_hyperedge_gain_max = 0.15
self.temporal_eqho_mixer = EventConditionedQuaternionMixer(self.event_cond_dim, self.quat_hyperedge_gain_max)
self.variable_eqho_mixer = EventConditionedQuaternionMixer(self.event_cond_dim, self.quat_hyperedge_gain_max)
```

- [ ] **步骤 5：在 `forward` 中改用 `event_summary -> mix_gain`，但保持 `mix_coef` 固定为安全模板**

```python
_, temporal_mix_gain = self.temporal_eqho_mixer(temporal_event_summary)
_, variable_mix_gain = self.variable_eqho_mixer(variable_event_summary)
temporal_mix_coef = temporal_main.new_tensor([1.0, 0.0, 0.0, 0.0]).view(1, 1, 4).expand(B, temporal_main.shape[1], 4)
variable_mix_coef = variable_main.new_tensor([1.0, 0.0, 0.0, 0.0]).view(1, 1, 4).expand(B, variable_main.shape[1], 4)
```

- [ ] **步骤 6：运行完整测试**

运行：

```bash
conda run -n pyomnits python -m unittest tests.models.test_QSHNet
```

预期：

- PASS

- [ ] **步骤 7：Commit**

```bash
git add models/QSHNet.py tests/models/test_QSHNet.py
git commit -m "feat: add event-conditioned gain for eqho"
```

---

### 任务 5：实现 `A3`，接入 full `r/i/j/k` mixing

**文件：**
- 修改：`/opt/Codes/PyOmniTS/models/QSHNet.py`
- 修改：`/opt/Codes/PyOmniTS/tests/models/test_QSHNet.py`

- [ ] **步骤 1：先写失败测试，约束 `mix_coef` 初始 real-dominant 且可学习**

```python
def test_eqho_mixer_starts_real_component_dominant(self):
    mixer = EventConditionedQuaternionMixer(cond_dim=4, gain_max=0.15)
    event_summary = torch.zeros(2, 5, 4)
    mix_coef, mix_gain = mixer(event_summary)
    self.assertTrue(torch.all(mix_coef[..., 0] > mix_coef[..., 1]))
    self.assertTrue(torch.all(mix_coef[..., 0] > mix_coef[..., 2]))
    self.assertTrue(torch.all(mix_coef[..., 0] > mix_coef[..., 3]))
```

- [ ] **步骤 2：运行新测试验证当前实现是否已满足，不满足则先修正初始化**

运行：

```bash
conda run -n pyomnits python -m unittest tests.models.test_QSHNet.TestQSHNet.test_eqho_mixer_starts_real_component_dominant
```

预期：

- PASS；如果 FAIL，先调整 `coef_head.bias`

- [ ] **步骤 3：在 `forward` 中正式使用 `mix_coef`**

```python
temporal_mix_coef, temporal_mix_gain = self.temporal_eqho_mixer(temporal_event_summary)
variable_mix_coef, variable_mix_gain = self.variable_eqho_mixer(variable_event_summary)
```

- [ ] **步骤 4：运行完整测试**

运行：

```bash
conda run -n pyomnits python -m unittest tests.models.test_QSHNet
```

预期：

- PASS

- [ ] **步骤 5：Commit**

```bash
git add models/QSHNet.py tests/models/test_QSHNet.py
git commit -m "feat: enable full event-conditioned quaternion mixing"
```

---

### 任务 6：把 EQHO 诊断接入 `QSHDiag`

**文件：**
- 修改：`/opt/Codes/PyOmniTS/models/QSHNet.py`
- 修改：`/opt/Codes/PyOmniTS/exp/exp_main.py`

- [ ] **步骤 1：在 `HypergraphLearner.forward` 中缓存每层 EQHO 诊断字段**

```python
eqho_diag = {
    "temporal_event_summary_norm": temporal_event_summary.norm(dim=-1).mean().item(),
    "variable_event_summary_norm": variable_event_summary.norm(dim=-1).mean().item(),
    "temporal_mix_coef_mean": temporal_mix_coef.mean(dim=(0, 1)).detach().cpu(),
    "variable_mix_coef_mean": variable_mix_coef.mean(dim=(0, 1)).detach().cpu(),
    "temporal_mix_gain_mean": temporal_mix_gain.mean().item(),
    "variable_mix_gain_mean": variable_mix_gain.mean().item(),
}
```

- [x] **步骤 2：在 `exp_main.py` 中扩展 epoch 级 `QSHDiag` 输出**

运行时输出格式应包含：

```text
[EQHO L0] t_summary=... v_summary=... t_mix=[...] v_mix=[...] t_gain=... v_gain=...
```

- [x] **步骤 3：运行完整测试**

运行：

```bash
conda run -n pyomnits python -m unittest tests.models.test_QSHNet
```

预期：

- PASS

- [ ] **步骤 4：Commit**

```bash
git add models/QSHNet.py exp/exp_main.py
git commit -m "feat: add eqho diagnostics"
```

---

### 任务 7：本地实验与停机判断

**文件：**
- 修改：`/opt/Codes/PyOmniTS/docs/QSHNet_evolution.md`
- 修改：`/opt/Codes/PyOmniTS/docs/QSHNet_architecture_status.md`
- 修改：`/opt/Codes/PyOmniTS/docs/hyperparameter_tuning_plan.md`

- [x] **步骤 1：跑 `A1` 本地 smoke**

运行：

```bash
python main.py --is_training 1 --collate_fn collate_fn --loss MSE \
  --d_model 128 --n_layers 3 --n_heads 1 \
  --dataset_root_path storage/datasets/HumanActivity \
  --model_id QSHNet_EQHO_A1 --model_name QSHNet \
  --dataset_name HumanActivity --dataset_id HumanActivity \
  --features M --seq_len 3000 --pred_len 300 \
  --enc_in 12 --dec_in 12 --c_out 12 \
  --train_epochs 300 --patience 10 --val_interval 1 \
  --itr 3 --batch_size 32 --learning_rate 1e-3
```

预期：

- 能正常训练结束
- 不出现 NaN

- [x] **步骤 2：跑 `A1` 的 `USHCN itr=5`**

运行：

```bash
python main.py --is_training 1 --collate_fn collate_fn --loss MSE \
  --d_model 256 --n_layers 1 --n_heads 1 \
  --dataset_root_path storage/datasets/USHCN \
  --model_id QSHNet_EQHO_A1 --model_name QSHNet \
  --dataset_name USHCN --dataset_id USHCN \
  --features M --seq_len 150 --pred_len 3 \
  --enc_in 5 --dec_in 5 --c_out 5 \
  --train_epochs 300 --patience 10 --val_interval 1 \
  --itr 5 --batch_size 16 --learning_rate 1e-3
```

停机标准：

- 如果 `A1` 明显炸掉或远差于当前基线，则停止，不进入 `A2 / A3`

- [x] **步骤 3：若 `A1` 通过，再跑 `A2` 的 `HumanActivity itr=3` 和 `USHCN itr=5`**

判断重点：

- `mix_gain` 是否脱离常数初始化
- 是否出现立即打满上界的现象

- [x] **步骤 4：若 `A2` 通过，再跑 `A3` 的 `HumanActivity itr=3` 和 `USHCN itr=5`**

判断重点：

- `mix_coef` 是否学出非平凡模式
- temporal / variable 是否不同
- 是否有论文价值

- [x] **步骤 5：将结果同步回文档**

文档中至少记录：

- 实验版本
- 唯一改动
- `HumanActivity itr=3`
- `USHCN itr=5`
- 是否继续推进

- [ ] **步骤 6：Commit**

```bash
git add docs/QSHNet_evolution.md docs/QSHNet_architecture_status.md docs/hyperparameter_tuning_plan.md
git commit -m "docs: record eqho local screening results"
```

---

## 自检

## 当前执行状态补记（2026-04-17）

- `A3` 已完成本地筛查：
  - `HumanActivity itr=3`：`0.0418 ± 0.0002`
  - `USHCN itr=5`：`0.1979 ± 0.0310`
- 结论：
  - `EQHO` 具备真实可训练性；
  - 但 `A3 full dynamic mix_coef` 在 `USHCN` 上明显失稳；
  - 当前已满足停机条件，不再将 `A3` 作为主线继续推进。
- 下一步若继续做 `EQHO`，应优先转向受限动态化版本（`A2.5`），而不是继续修补 `A3`。

- 本计划覆盖了从 `event_summary`、`A1`、`A2`、`A3` 到 `QSHDiag`、本地实验和文档回填的所有规格要求。
- 本计划未包含占位符类步骤，每个任务都给出明确文件、代码片段、命令和预期结果。
- 类型与命名保持一致，统一使用：
  - `HyperedgeEventSummarizer`
  - `EventConditionedQuaternionMixer`
  - `EventConditionedQuaternionHyperedgeOperator`
  - `EQHO`
  - `mix_coef`
  - `mix_gain`

## 执行交接

计划已完成并保存到 `docs/superpowers/plans/2026-04-17-qshnet-eqho-implementation-plan.md`。两种执行方式：

**1. 子代理驱动（推荐）** - 每个任务调度一个新的子代理，任务间进行审查，快速迭代

**2. 内联执行** - 在当前会话中使用 executing-plans 执行任务，批量执行并设有检查点

选哪种方式？
