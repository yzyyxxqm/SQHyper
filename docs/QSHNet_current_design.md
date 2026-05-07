# QSHNet 当前模型设计文档

> **最后更新：** 2026-04-22
> **对应代码：** [models/QSHNet.py](/opt/Codes/PyOmniTS/models/QSHNet.py:1)
> **适用范围：** 解释当前工作区里实际在运行的 QSHNet 架构，而不是历史实验快照

## 1. 文档目的

这份文档只回答一个问题：

**当前代码里的 QSHNet，到底是怎么工作的。**

这里的「当前模型」以 [`models/QSHNet.py`](/opt/Codes/PyOmniTS/models/QSHNet.py:1) 为唯一准绳，因此它和一些旧文档中的实验叙述可能不同。尤其需要明确的是：

- 当前 QSHNet 的主体仍然是 `HyperIMTS` 风格的超图消息传递骨架；
- 当前增强项主要有两类：
  - **SpikeRouter 驱动的事件选择与事件注入**
  - **QuaternionLinear 驱动的四元数残差细化**
- 当前代码里**没有** residual-correction 输出头这类后期试验结构；
- 当前训练入口的优化器也已经统一回 `Adam`，不再是旧文档里写过的 `AdamW`。

## 2. 一句话概括

当前 QSHNet 可以概括为：

**以 HyperIMTS 的 observation node / temporal hyperedge / variable hyperedge 三层超图结构为主干，在每一层消息传递前用 SpikeRouter 做上下文感知的观测分流，在 hyperedge-to-node 融合时再用四元数残差做有界细化。**

对应的数据流可以写成：

```text
IMTS 输入
  -> 序列拼接与 flatten
  -> HypergraphEncoder
  -> 多层 HypergraphLearner
       1) SpikeRouter：obs -> obs_base + obs_event
       2) node -> temporal / variable hyperedge
       3) event 注入到 hyperedge
       4) hyperedge -> node 融合
       5) quaternion residual refinement
       6) 最后一层 variable-to-variable hyperedge 更新
  -> HypergraphDecoder
  -> 预测输出
```

## 3. 总体结构

当前模型由 3 个主模块组成：

1. **`HypergraphEncoder`**
   把输入序列编码成：
   - observation nodes
   - temporal hyperedges
   - variable hyperedges
   - temporal / variable incidence matrix

2. **`HypergraphLearner`**
   执行 `n_layers` 层超图消息传递，是 QSHNet 的核心。每层在 HyperIMTS 主干上插入：
   - `SpikeRouter`
   - event injection
   - quaternion refinement

3. **`HypergraphDecoder`**
   把最终 observation / temporal / variable 三路表征重新拼起来，映射到标量预测值。

## 4. 输入与预处理流程

### 4.1 输入张量

模型前向接口是：

```python
forward(
    x, x_mark=None, x_mask=None,
    y=None, y_mark=None, y_mask=None,
    ...
)
```

主要输入含义如下：

- `x`: `(B, SEQ_LEN, ENC_IN)`，历史观测值
- `x_mark`: `(B, SEQ_LEN, 1)`，历史时间标记
- `x_mask`: `(B, SEQ_LEN, ENC_IN)`，历史观测掩码
- `y`: `(B, PRED_LEN, ENC_IN)`，预测目标
- `y_mark`: `(B, PRED_LEN, 1)`，未来时间标记
- `y_mask`: `(B, PRED_LEN, ENC_IN)`，预测部分掩码

如果部分输入为空，模型会在前向里补默认值。

### 4.2 历史段与预测段拼接

对 forecasting 任务，当前实现会构造 4 组拼接量：

- `x_L = cat(x, zeros_like(y))`
- `x_y_mask = cat(x_mask, y_mask)`
- `y_L = cat(zeros_like(x), y)`
- `y_mask_L = cat(zeros_like(x), y_mask)`

同时把时间标记拼成：

- `x_y_mark = cat(x_mark, y_mark)`

这样长度统一变成：

- `L = SEQ_LEN + PRED_LEN`

这个设计的意义是：模型始终在一张统一的超图上看「历史观测位置 + 预测目标位置」。

### 4.3 flatten 与 pad

IMTS 的关键问题是每个样本的有效观测数不同。当前实现有两种路径：

- **规则场景**：如果所有位置都有观测，直接把 `(B, L, ENC_IN)` reshape 成 `(B, L * ENC_IN)`
- **不规则场景**：通过 `pad_and_flatten()` 只保留有效观测，再 pad 到 batch 内最大观测数

同时也会生成两个扁平索引：

- `time_indices_flattened`
- `variable_indices_flattened`

它们分别记录每个 observation node 属于哪个时间步、哪个变量。

## 5. HypergraphEncoder

`HypergraphEncoder` 基本保持了 HyperIMTS 的编码方式。

### 5.1 observation node 初始化

每个有效观测点会先被压成一个二维输入：

```text
[观测值, 预测指示器]
```

然后过一层线性层：

```python
observation_nodes = ReLU(Linear(2, d_model)(...)) * mask
```

其中第二维不是时间嵌入，而是「这个位置是否属于预测段」的指示信息。

### 5.2 temporal hyperedge 初始化

时间超边来自时间标记 `x_y_mark`：

```python
temporal_hyperedges = sin(Linear(1, d_model)(x_y_mark))
```

也就是说，每个时间步都有一个对应的 temporal hyperedge 表征。

### 5.3 variable hyperedge 初始化

变量超边由可学习参数直接初始化：

```python
variable_hyperedges = ReLU(variable_hyperedge_weights)
```

每个变量有一条 variable hyperedge。

### 5.4 incidence matrix

编码器会同时构造两张关联矩阵：

- `temporal_incidence_matrix: (B, L, N)`
- `variable_incidence_matrix: (B, E, N)`

作用分别是：

- 指出某个 observation node 属于哪个时间超边
- 指出某个 observation node 属于哪个变量超边

这两张矩阵是后续 SpikeRouter 和消息传递的结构基础。

## 6. HypergraphLearner 的层内流程

`HypergraphLearner` 是当前模型最关键的部分。每一层都按下面的顺序执行。

### 6.1 第一步：SpikeRouter 做观测分流

每层先调用：

```python
obs_base, obs_event, route_state = self.spike_select[i](...)
```

其中 `SpikeRouter` 的输入包括：

- 当前 observation node `obs`
- 扩展后的 mask `mask_d`
- `variable_incidence_matrix`
- `variable_indices_flattened`

#### 6.1.1 SpikeRouter 的核心思想

它不是简单地看单个观测值，而是先利用变量超图结构算出**变量上下文**：

```python
var_context = (variable_incidence_matrix @ obs) / var_count
obs_var_ctx = gather(var_context, variable_indices_flattened)
deviation = obs - obs_var_ctx
```

所以它判断的不是「这个值大不大」，而是：

**这个观测相对于同变量上下文，是否偏离得足够明显。**

#### 6.1.2 产生的 4 个核心量

随后它会生成：

- `route_logit`
- `retain_gate`
- `selection_weight`
- `event_gate`

以及一条事件特征支路：

```python
event_features = event_proj(cat(obs, deviation))
```

最终输出：

- `obs_base = obs * retain_gate * mask`
- `obs_event = event_features * event_gate * mask`

可以把它理解成：

- `obs_base` 负责保留稳定主路径信息
- `obs_event` 负责把“偏离上下文的事件性信息”单独抽出来

#### 6.1.3 当前实现不是硬脉冲

虽然代码里还保留了 `SpikeFunction`、`threshold`、`gamma` 这些脉冲语义相关组件，但**当前实际主路径并没有使用硬二值 spike**。现在真正生效的是连续 gate：

- `retain_gate`
- `event_gate`
- `selection_weight`

所以当前 QSHNet 更准确地说是：

**带脉冲语义命名的连续事件选择器**，而不是严格 SNN 式硬脉冲网络。

### 6.2 第二步：事件注入前的传播强度调制

在 `obs_base` 形成后，当前实现还会额外计算一个传播选择因子：

```python
propagation_selection_factor = 1.0 + strength * (selection_weight - 0.5)
obs_selected = obs_base * propagation_selection_factor
```

这里的默认强度是：

- `propagation_selection_strength = 0.05`

含义是：

- 路由分数高的 observation node，进入超边聚合时会略微放大
- 路由分数低的节点会略微减弱
- 但整体仍然是温和调制，而不是激进裁剪

### 6.3 第三步：node -> temporal hyperedge

这一段仍然沿用 HyperIMTS 的 attention 结构：

```python
temporal_hyperedges_base = node2temporal_hyperedge(
    temporal_hyperedges,
    cat(variable_hyperedges_gathered, obs_selected),
    temporal_incidence_matrix
)
```

也就是说，时间超边的更新同时依赖：

- 节点所属变量超边的信息
- 当前被 SpikeRouter 调制过的 observation node

### 6.4 第四步：node -> variable hyperedge

变量超边更新与上面对称：

```python
variable_hyperedges_base = node2variable_hyperedge(
    variable_hyperedges,
    cat(temporal_hyperedges_gathered, obs_selected),
    variable_incidence_matrix
)
```

此时 temporal / variable 两种超边都先得到一条「主干更新结果」。

### 6.5 第五步：event 注入到 hyperedge

`obs_event` 不会直接替代主干，而是先按 incidence matrix 聚合成两类事件增量：

```python
temporal_event_delta = (temporal_incidence_matrix @ obs_event) / temporal_count
variable_event_delta = (variable_incidence_matrix @ obs_event) / variable_count
```

再经过各自的 LayerNorm：

- `temporal_event_norm[i]`
- `variable_event_norm[i]`

然后通过 `apply_event_injection()` 注入到主干超边状态里：

```python
temporal_hyperedges_updated = temporal_hyperedges_base + event_scale * temporal_event_delta
variable_hyperedges_updated = variable_hyperedges_base + event_scale * variable_event_delta
```

#### 6.5.1 event scale 的当前实现

当前 event 注入强度不是无界的，而是：

```python
event_scale = clamp(sigmoid(event_residual_scale), max=0.12)
```

并且初始化时：

- `event_residual_scale = log(0.1 / 0.9)`

所以初始 `sigmoid(event_residual_scale) = 0.1`，再经过 `max=0.12` 的上界裁剪，意味着 event 支路从一开始就是**弱注入**而不是零注入。

这和一些旧文档里写的「事件支路初始完全静默」已经不一致。

#### 6.5.2 event density 调制

当前代码还会根据 route density 调节 event scale：

- `temporal_route_density`
- `variable_route_density`

其中默认配置是：

- `temporal_event_density_penalty_max = 0.0`
- `variable_event_density_penalty_max = 0.5`

因此：

- temporal event 注入默认不因密度而衰减
- variable event 注入默认会在高密度区域受到额外抑制

### 6.6 第六步：hyperedge -> node 回流

更新完 temporal / variable hyperedge 后，每个 observation node 会重新 gather 自己对应的两路超边上下文：

- `tg`: temporal context
- `vg`: variable context

当前代码还会对 fused route density 做一次稳定化处理：

1. 先取 temporal / variable 路由密度的逐点最大值
2. 对 variable context 的耦合残差做 `bound_coupled_residual()`
3. 把风险较大的 residual 比例压到允许范围内

这一段的关键作用是：

**防止 hyperedge 回流到 node 时，事件耦合残差在高风险状态下过大。**

### 6.7 第七步：node self update 与 OOM fallback

在正常情况下，模型会先做一次 node 级自更新：

```python
obs_for_h2n = node_self_update(
    observation_nodes,
    cat(tg, vg, observation_nodes),
    pairwise_mask
)
```

再构造：

```python
h2n_input = cat(obs_for_h2n, tg, vg)
```

但这一段有 `O(N^2)` 的节点间注意力开销，所以代码里保留了 OOM fallback：

- 如果触发显存溢出，`self.oom_flag = True`
- 后续层直接跳过 `node_self_update`
- 改成：

```python
obs_for_h2n = observation_nodes
h2n_input = cat(obs_for_h2n, tg, vg)
```

也就是说，当前模型是带**显存自降级机制**的。

### 6.8 第八步：四元数残差细化

这一段是 QSHNet 相对 HyperIMTS 的第二个核心增强。

先走一条普通线性路径：

```python
linear_out = hyperedge2node[i](h2n_input)
```

再走四元数路径：

```python
quat_out = quat_h2n[i](linear_out)
alpha = compute_quaternion_gate(i, linear_out, route_state["event_gate"])
quat_residual = bound_quaternion_residual(linear_out, quat_out, alpha)
h2n_out = linear_out + quat_residual
```

#### 6.8.1 QuaternionLinear 的作用

`QuaternionLinear` 把特征维按 4 组组织，用 Hamilton product 构造块矩阵线性变换。它的目标不是替代主干线性层，而是：

**在 `linear_out` 的基础上补充跨特征组耦合。**

#### 6.8.2 为什么 `d_model` 必须是 4 的倍数

因为四元数层会把通道均分成 `r / i / j / k` 四组，所以模型初始化时会做：

```python
self.d_model = (configs.d_model // 4) * 4
```

如果外部传入的 `d_model` 不是 4 的倍数，代码会自动向下取整并打日志警告。

#### 6.8.3 四元数 gate 的输入

当前 `alpha` 不是全局常数，而是 node-wise 的：

```python
gate_input = cat(linear_out, event_gate)
alpha = sigmoid(Linear(d_model + 1, 1)(gate_input))
```

所以它同时看：

- 节点当前线性状态 `linear_out`
- 节点当前事件强度 `event_gate`

这意味着四元数残差强度是**条件化、自适应、逐节点**的。

#### 6.8.4 四元数残差上界

当前代码不会无条件把 `alpha * quat_out` 加回去，而是会通过：

```python
bound_quaternion_residual()
```

强行限制：

- 四元数残差范数不超过主线性输出范数的 `25%`

即：

- `quat_residual_ratio_max = 0.25`

因此当前实现里的 quaternion branch 是一个**有上界的补充残差**，不是主导路径。

### 6.9 第九步：更新 observation node

得到 `h2n_out` 后，节点更新为：

```python
observation_nodes = ReLU((observation_nodes + h2n_out) * mask)
```

这里本质上仍是 residual update：

- 保留旧节点状态
- 加上本层融合结果
- 再经过激活函数

### 6.10 第十步：最后一层 variable hyperedge-to-hyperedge

仅在最后一层执行：

```python
variable_hyperedges = variable_hyperedges + variable_hyperedge2variable_hyperedge(...)
```

这一部分与 HyperIMTS 一致，用于学习变量超边之间的高阶依赖。

## 7. Decoder 与输出形式

编码和多层学习结束后，decoder 会把三部分状态重新拼接：

- 最终 observation node
- 对应的 temporal hyperedge
- 对应的 variable hyperedge

然后过一层：

```python
pred_flattened = Linear(3 * d_model, 1)(...)
```

### 7.1 训练阶段输出

在 `train / val` 阶段，模型直接返回 flatten 后的结果：

- `pred`
- `true`
- `mask`

这样做是为了方便 loss 直接在 flatten 视图上计算。

### 7.2 测试阶段输出

在 `test` 阶段，会通过 `unpad_and_reshape()` 把 flatten 预测恢复回原始时空布局，再裁出最后 `PRED_LEN` 段返回。

## 8. 当前模型的关键设计约束

当前代码里有几个非常重要的结构约束，文档里必须明确写清楚。

### 8.1 主干仍是 HyperIMTS

QSHNet 不是另起炉灶的全新骨架，而是在 HyperIMTS 上做增量增强：

- 编码方式基本不变
- 主消息传递骨架基本不变
- decoder 基本不变

真正新增的是层内的事件分流和四元数残差。

### 8.2 event 支路是弱注入、受控注入

当前 event 支路不是死分支，但也不是无约束强分支。它受到：

- `event_scale_max = 0.12`
- route density penalty
- LayerNorm

这几层共同约束。

### 8.3 quaternion 支路是残差细化，不是平行主干

四元数部分的定位非常明确：

- 输入来自 `linear_out`
- 输出只作为 residual
- residual 还有范数上界

所以它更像「高阶耦合细化器」，而不是第二条完整主网络。

### 8.4 当前实现高度依赖诊断量

代码里保留了大量 `latest_event_diagnostics` 和周期性日志输出，用来跟踪：

- `retain_gate`
- `event_gate`
- `route_logit`
- route density
- event residual ratio
- quaternion residual ratio

这说明当前模型的设计不仅是前向结构，还内含一套专门为不稳定性分析准备的诊断接口。

## 9. 当前模型和旧文档最容易混淆的地方

为了避免后续再写错，这里单独列出来。

1. **当前代码没有 residual-correction 输出头。**
   旧文档里提过的很多 residual-correction 变体，不属于现在这份 `models/QSHNet.py`。

2. **当前 event 支路不是“初始化完全为 0”。**
   现在是弱注入起步，而不是严格静默。

3. **当前 quaternion gate 是逐节点条件化 gate。**
   不是早期实验里那种全局单标量门控。

4. **当前优化器默认已经统一成 `Adam`。**
   不能再沿用旧文档里的 `AdamW` 表述。

5. **当前项目运行时主要依赖命令行参数。**
   yaml 可以作为参考，但不是运行时的唯一真实来源。

## 10. 推荐阅读顺序

如果是第一次看当前 QSHNet，建议按下面顺序读：

1. 先看本文件，理解当前模型结构和前向流程
2. 再看 [QSHNet_overview.md](/opt/Codes/PyOmniTS/docs/QSHNet_overview.md)
3. 再看 [QSHNet_architecture_status.md](/opt/Codes/PyOmniTS/docs/QSHNet_architecture_status.md)
4. 如果要追溯历史实验，再看 [QSHNet_evolution.md](/opt/Codes/PyOmniTS/docs/QSHNet_evolution.md)

## 11. 当前模型的最简总结

如果必须用最短的话描述当前 QSHNet，可以写成：

> 当前 QSHNet 是一个以 HyperIMTS 为主干的 IMTS 预测模型。它先用基于变量上下文的 SpikeRouter 把 observation node 分成基础流和事件流，再把事件流以弱注入方式汇入 temporal / variable hyperedge，最后在 hyperedge-to-node 融合处加入带范数上界的四元数残差细化。
