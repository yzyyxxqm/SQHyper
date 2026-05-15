# SQEH 完整上下文文档（用于新对话）

## 一、项目基本信息
- 路径：`/opt/Codes/PyOmniTS`
- 环境：conda `pyomnits`，Python `/home/wgx/miniconda3/envs/pyomnits/bin/python`
- 框架：PyOmniTS（不规则多变量时序预测）
- GPU：3.6GB
- 模型文件：`/opt/Codes/PyOmniTS/models/SQEH.py`
- 优化上下文：`/opt/Codes/PyOmniTS/docs/SQEH_optimization_context.md`
- 状态报告：`/opt/Codes/PyOmniTS/docs/SQEH_status_report.md`
- 论文草稿：`/opt/Codes/PyOmniTS/docs/my_paper/draft_1.md`

## 二、用户核心要求

### 性能目标
- HumanActivity: MSE 严格低于 0.04（当前 0.0418±0.0006，HyperIMTS 0.0414±0.0002）
- P12: MSE 低于 0.29（当前 0.312，HyperIMTS 0.2991）
- USHCN: 已超越 HyperIMTS 9%（0.2195 vs 0.2411）
- 所有数据集都要明显优于 HyperIMTS

### 模型要求
- 必须是新模型，有 2-3 个强创新点
- 不过度参考 HyperIMTS，关注模型自身特点
- 核心消息传递（QSA）必须原创
- 配置必须与 HyperIMTS 一致（公平对比）

### 论文要求
- 目标：一区或二区期刊（TNNLS/TKDE/Information Fusion/Neural Networks 等）
- Related Work 中介绍了 HyperIMTS，所以实验必须包含它作为 baseline
- 需要 4-5 个数据集的实验（HumanActivity, USHCN, P12, MIMIC-III, 可选 MIMIC-IV/ILI）
- 需要消融实验、参数敏感性、可视化

### 训练约束
- 不改训练方式（不用 SWA、R-Drop 等训练 trick），因为 baseline 都是标准训练
- GPU 3.6GB，batch_size=32 已接近极限
- HumanActivity 约 1000 样本，极易过拟合

### 监控要求
- 实验运行时需要监视参数和 loss 的变化轨迹
- 每个 epoch 记录一次关键参数（threshold、alpha、event density 等）
- 参数最好保存到 log 文件

## 三、当前模型架构

### 创新点
1. **QSA（Quaternion Spike Attention）**：QLinear Q/K + 四元数分量点积相似度 + 脉冲调制温度 softmax
2. **四元数语义编码**：R=观测值, I=时间相位(sin/cos), J=变量嵌入, K=指示器(mask+target)
3. **动态 Event 超边**：当前用 softmax 归一化（之前用脉冲阈值但会死）

### 架构流程
1. 编码器：value_proj + time_freq + var_emb + indicator_proj → QLinear mixer → QNorm
2. 三种超边初始化：temporal(sin编码), variable(learnable), event(learnable prototypes)
3. 每层 QSMPBlock：
   - Node→Temporal HE: QSA + variable cross context
   - Node→Variable HE: QSA + temporal cross context
   - Node→Event HE: mean pooling（去掉了 QSA 加速）
   - HE→Node: Node Self-Attention + ReLU(q + h2n + 0.1*e_gathered)
   - FFN: QLinear(D,4D) → GELU → QLinear(4D,D)
4. 超图传播平滑：1步 temporal+variable 拉普拉斯传播（可学习α）
5. 解码器：四元数分量感知（4×Linear(Q*3,1) + softmax gate）
6. 辅助损失：Event diversity loss (λ=0.05)

### 当前代码状态
- Event 超边改用 softmax 归一化（不会死）
- Event HE 更新用 mean pooling（去掉了 qsa_e，速度翻倍）
- Event context 以 0.1 系数加入 HE→Node 残差

## 四、实验结果汇总

### HumanActivity（正确代码，itr=10）
- MSE = 0.0418 ± 0.0006（最优 0.04184）
- 配置：d_model=128, n_layers=3, n_events=32, spike_slope=5.0

### P12（改进版，itr=3）
- MSE = 0.3118-0.3124
- 配置：d_model=256, n_layers=1, n_events=32

### USHCN（之前版本）
- MSE = 0.2195

## 五、已验证无效的方向（不要再尝试）

- 增大 d_model (192/256) → 过拟合
- 增加层数 (5层) → 过拟合
- 多头 QSA → 过拟合
- 增大 dropout (0.1) → 欠拟合
- 加深解码器 → 过拟合
- Gated Residual 替代 ReLU → 更差
- Stochastic Depth → 更差
- FFN 4x→2x → 欠拟合
- CosineAnnealingLR → 更差
- weight_decay → 无改善
- 旋转残差/脉冲解码/共轭损失 → 过拟合
- Event context 加入解码器 → 过拟合
- 超边空间预测（三路分解）→ 更差
- 简化解码器 Linear(3D,1) → 更差（四元数分量感知是必要的）
- Variable 超边自注意力 → 过拟合
- 自适应对级脉冲阈值 → 无改善
- Event Router (QSHNet移植) → 参数不动，无效
- n_events=16 → 无改善
- spike_slope=3.0 → 无改善

## 六、关键发现

1. **HumanActivity 上所有参数不动**：threshold=0.31, α=0.574 从不变化
2. **P12 上参数会动**：variable QSA threshold 0.31→0.27, α 0.574→0.585
3. **Event 超边在旧版本中会死**（density→0），改用 softmax 后活了但对性能无帮助
4. **模型有效部分**：主要是 temporal + variable 超边的 QSA 消息传递
5. **HyperIMTS 的优势**：8头注意力、variable-to-variable 交互（IrregularityAwareAttention）
6. **训练速度**：去掉 qsa_e 后 P12 从 3.5→6.5 it/s

## 七、HyperIMTS 架构对比

HyperIMTS 有但 SQEH 没有的：
- 多头注意力（8头 vs 我们的单头 QSA）
- Variable-to-variable 超边交互（IrregularityAwareAttention）
- 更简单的解码器（Linear(3D,1)）

SQEH 有但 HyperIMTS 没有的：
- 四元数结构（QLinear, 四元数编码）
- 脉冲调制温度
- 动态 Event 超边
- 四元数分量感知解码

## 八、运行命令

### HumanActivity
```bash
cd /opt/Codes/PyOmniTS && /home/wgx/miniconda3/envs/pyomnits/bin/python main.py \
    --is_training 1 --collate_fn "collate_fn" --loss "MSE_aux" \
    --d_model 128 --n_layers 3 --n_heads 1 --dropout 0.05 \
    --dataset_root_path "storage/datasets/HumanActivity" \
    --model_id SQEH --model_name SQEH \
    --dataset_name HumanActivity --dataset_id HumanActivity \
    --features M --seq_len 3000 --pred_len 300 \
    --enc_in 12 --dec_in 12 --c_out 12 \
    --train_epochs 300 --patience 10 --val_interval 1 \
    --itr 5 --batch_size 32 --learning_rate 1e-3 \
    --sqeh_n_events 32 --sqeh_spike_slope 5.0 --sqeh_diag_interval 30
```

### P12
```bash
cd /opt/Codes/PyOmniTS && /home/wgx/miniconda3/envs/pyomnits/bin/python main.py \
    --is_training 1 --collate_fn "collate_fn" --loss "MSE_aux" \
    --d_model 256 --n_layers 1 --n_heads 8 --dropout 0.05 \
    --dataset_root_path "storage/datasets/P12" \
    --model_id SQEH --model_name SQEH \
    --dataset_name P12 --dataset_id P12 \
    --features M --seq_len 36 --pred_len 3 \
    --enc_in 36 --dec_in 36 --c_out 36 \
    --train_epochs 300 --patience 10 --val_interval 1 \
    --itr 3 --batch_size 32 --learning_rate 1e-3 \
    --sqeh_n_events 32 --sqeh_diag_interval 304
```

## 九、下一步方向建议

1. **让 Event 超边真正有用**：当前即使活着也不提供有用信息。可能需要完全重新设计 event 超边的语义（如基于时间窗口的局部事件检测，而非全局原型匹配）
2. **增强 temporal/variable 超边的表达力**：这是模型真正有效的部分，可以考虑多头 QSA 或更深的超边更新
3. **P12 需要不同的配置**：n_layers=1 可能不够，但 n_layers=2 + d_model=256 太慢。可以试 n_layers=2 + d_model=128
4. **考虑去掉 Event 超边**：如果它确实无用，去掉可以减少参数和计算，可能反而提升性能
