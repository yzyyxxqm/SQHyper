# SQEH: Spiking Quaternion Event Hypergraph
import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from torch import Tensor
from utils.ExpConfigs import ExpConfigs
from utils.globals import logger

# =============================================================================
# Quaternion primitives
# =============================================================================
def qsplit(q: Tensor):
    Q = q.shape[-1] // 4
    return q[..., :Q], q[..., Q:2*Q], q[..., 2*Q:3*Q], q[..., 3*Q:]

def qcat(r, i, j, k):
    return torch.cat([r, i, j, k], dim=-1)

def hamilton(p: Tensor, q: Tensor) -> Tensor:
    pr, pi, pj, pk = qsplit(p)
    qr, qi, qj, qk = qsplit(q)
    return qcat(
        pr*qr - pi*qi - pj*qj - pk*qk,
        pr*qi + pi*qr + pj*qk - pk*qj,
        pr*qj - pi*qk + pj*qr + pk*qi,
        pr*qk + pi*qj - pj*qi + pk*qr,
    )

class QNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.norm = nn.LayerNorm(d // 4)
    def forward(self, x):
        r, i, j, k = qsplit(x)
        return qcat(self.norm(r), self.norm(i), self.norm(j), self.norm(k))

class QLinear(nn.Module):
    def __init__(self, d_in, d_out, bias=True):
        super().__init__()
        Qi, Qo = d_in // 4, d_out // 4
        self.r = nn.Parameter(torch.empty(Qo, Qi))
        self.i = nn.Parameter(torch.empty(Qo, Qi))
        self.j = nn.Parameter(torch.empty(Qo, Qi))
        self.k = nn.Parameter(torch.empty(Qo, Qi))
        self.bias = nn.Parameter(torch.zeros(d_out)) if bias else None
        for p in (self.r, self.i, self.j, self.k):
            nn.init.xavier_uniform_(p)
    def forward(self, x):
        r, i, j, k = self.r, self.i, self.j, self.k
        W = torch.cat([
            torch.cat([r, -i, -j, -k], 1),
            torch.cat([i,  r, -k,  j], 1),
            torch.cat([j,  k,  r, -i], 1),
            torch.cat([k, -j,  i,  r], 1),
        ], 0)
        return F.linear(x, W, self.bias)


# =============================================================================
# QSA - Quaternion Spike Attention
# =============================================================================
class QuaternionSpikeAttention(nn.Module):
    def __init__(self, d_model, n_heads=1, spike_slope=5.0, cross_ctx=False, dropout=0.1):
        super().__init__()
        assert d_model % 4 == 0
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert self.head_dim % 4 == 0, "head_dim must be divisible by 4 for quaternion"
        self.Q = self.head_dim // 4
        self.spike_slope = spike_slope
        self.cross_ctx = cross_ctx
        self.q_proj = QLinear(d_model, d_model)
        self.k_proj = QLinear(d_model, d_model)
        v_in = 2 * d_model if cross_ctx else d_model
        self.v_proj = nn.Linear(v_in, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.threshold = nn.Parameter(torch.tensor(-1.0))
        self.density_mod = nn.Sequential(nn.Linear(3, 16), nn.GELU(), nn.Linear(16, 1))
        nn.init.zeros_(self.density_mod[-1].weight)
        nn.init.zeros_(self.density_mod[-1].bias)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask, density_feat=None, cross_context=None):
        q = self.q_proj(query)
        k = self.k_proj(key)
        if self.cross_ctx and cross_context is not None:
            v = self.v_proj(torch.cat([value, cross_context], dim=-1))
            k = k + self.k_proj(cross_context)
        else:
            v = self.v_proj(value)

        H = self.n_heads
        hd = self.head_dim
        Q = self.Q

        # 多头：split 沿最后一维，concat 沿 batch 维（与 HyperIMTS 相同策略）
        # [B, M, D] -> H x [B, M, hd] -> cat -> [B*H, M, hd]
        q_h = torch.cat(q.split(hd, dim=-1), dim=0)  # [B*H, M, hd]
        k_h = torch.cat(k.split(hd, dim=-1), dim=0)  # [B*H, N, hd]
        v_h = torch.cat(v.split(hd, dim=-1), dim=0)  # [B*H, N, hd]

        # 四元数点积（每头独立）
        qr, qi, qj, qk_c = qsplit(q_h)
        kr, ki, kj, kk_c = qsplit(k_h)
        scores = (torch.einsum("bmd,bnd->bmn", qr, kr)
                 + torch.einsum("bmd,bnd->bmn", qi, ki)
                 + torch.einsum("bmd,bnd->bmn", qj, kj)
                 + torch.einsum("bmd,bnd->bmn", qk_c, kk_c)) / math.sqrt(Q)
        scores = scores.clamp(-6, 6)

        # 密度自适应脉冲阈值
        base_threshold = F.softplus(self.threshold)
        if density_feat is not None:
            d_shift = self.density_mod(density_feat).squeeze(-1)  # [B, N]
            # repeat H 次对齐 batch 维
            d_shift_h = d_shift.repeat(H, 1)  # [B*H, N]
            threshold = base_threshold + d_shift_h.unsqueeze(1) * 0.1
        else:
            threshold = base_threshold

        spike_gate = torch.sigmoid(self.spike_slope * (scores - threshold))
        modulated_scores = scores * (1.0 + spike_gate * 2.0)

        # mask repeat H 次
        mask_h = mask.repeat(H, 1, 1) if mask.dim() == 3 else mask.repeat(H, 1)
        if mask_h.dim() == 2:
            mask_h = mask_h.unsqueeze(1) * mask_h.unsqueeze(2)
        modulated_scores = modulated_scores.masked_fill(mask_h == 0, -1e4)
        weights = torch.softmax(modulated_scores.clamp(-30, 30), dim=-1)
        weights = self.attn_dropout(weights)
        out_h = torch.bmm(weights, v_h)  # [B*H, M, hd]

        # 合并多头：[B*H, M, hd] -> H x [B, M, hd] -> cat -> [B, M, D]
        B = query.shape[0]
        out = torch.cat(out_h.split(B, dim=0), dim=-1)  # [B, M, D]
        return self.out_proj(out)

# =============================================================================
# Encoder
# =============================================================================
class Encoder(nn.Module):
    def __init__(self, d_model, n_vars):
        super().__init__()
        Q = d_model // 4
        self.Q = Q
        self.value_proj = nn.Linear(2, Q)
        freqs = torch.exp(torch.linspace(0.0, math.log(10.0), Q))
        self.register_buffer("time_freq", freqs)
        self.var_emb = nn.Embedding(n_vars, Q)
        nn.init.normal_(self.var_emb.weight, std=0.1)
        self.indicator_proj = nn.Linear(2, Q)
        self.mixer = QLinear(d_model, d_model)
        self.norm = QNorm(d_model)

    def forward(self, value, time_norm, var_id, mask, target_mask):
        Q = self.Q
        r = self.value_proj(torch.stack([value * mask, mask], dim=-1))
        t_phase = time_norm.unsqueeze(-1) * self.time_freq
        half = Q // 2
        i_part = torch.cat([torch.sin(t_phase[..., :half]), torch.cos(t_phase[..., half:])], dim=-1)[..., :Q]
        j = self.var_emb(var_id.clamp(0, self.var_emb.num_embeddings - 1))
        k = self.indicator_proj(torch.stack([mask, target_mask], dim=-1))
        q = qcat(r, i_part, j, k)
        return self.norm(self.mixer(q)) * mask.unsqueeze(-1)


# =============================================================================
# Temporal Event Convolution
# =============================================================================
class TemporalEventConvolution(nn.Module):
    """时间局部事件卷积检测器 - 替代全局原型匹配 Event 超边。

    通过在时间排序的节点序列上滑动窗口，计算每个节点与事件滤波器的
    四元数点积相似度，生成事件上下文信息。
    """

    def __init__(self, d_model: int, n_events: int, window_size: int = 5):
        super().__init__()
        assert d_model % 4 == 0, "d_model must be divisible by 4 (quaternion constraint)"
        assert window_size % 2 == 1, "window_size must be odd"
        self.d_model = d_model
        self.n_events = n_events
        self.window_size = window_size
        self.Q = d_model // 4

        # 事件滤波器：每个 event 是一个时间模式检测器 [K, W, D]
        self.event_filters = nn.Parameter(
            torch.randn(n_events, window_size, d_model) * 0.02
        )
        # 事件中心向量：用于加权聚合输出 [K, D]
        self.event_centers = nn.Parameter(
            torch.randn(n_events, d_model) * 0.02
        )
        # 输出投影
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, q: Tensor, time_idx: Tensor, mask: Tensor) -> Tensor:
        """
        Args:
            q: 节点表示 [B, N, D]
            time_idx: 时间索引 [B, N] (long型)
            mask: 有效节点掩码 [B, N]
        Returns:
            event_context: 事件上下文信息 [B, N, D]
        """
        B, N, D = q.shape
        K = self.n_events
        W = self.window_size
        half_w = W // 2
        Q = self.Q

        # Step 1: 按时间排序节点
        sorted_idx = time_idx.argsort(dim=1)  # [B, N]
        q_sorted = q.gather(1, sorted_idx.unsqueeze(-1).expand(-1, -1, D))  # [B, N, D]
        mask_sorted = mask.gather(1, sorted_idx)  # [B, N]

        # Step 2: 零填充两端，构建滑动窗口
        # q_sorted: [B, N, D] -> q_padded: [B, N + W - 1, D]
        q_padded = F.pad(q_sorted, (0, 0, half_w, half_w), mode='constant', value=0.0)
        # unfold 在时间维度(dim=1)上提取窗口: [B, N, D, W]
        q_windows = q_padded.unfold(1, W, 1)  # [B, N, D, W]
        # 调整为 [B, N, W, D]
        q_windows = q_windows.permute(0, 1, 3, 2)  # [B, N, W, D]

        # Step 3: 四元数点积激活度计算
        # 将 q_windows 和 event_filters 拆分为 R/I/J/K 分量
        # q_windows: [B, N, W, D] -> 4 x [B, N, W, Q]
        qw_r = q_windows[..., :Q]
        qw_i = q_windows[..., Q:2*Q]
        qw_j = q_windows[..., 2*Q:3*Q]
        qw_k = q_windows[..., 3*Q:]

        # event_filters: [K, W, D] -> 4 x [K, W, Q]
        ef_r = self.event_filters[..., :Q]
        ef_i = self.event_filters[..., Q:2*Q]
        ef_j = self.event_filters[..., 2*Q:3*Q]
        ef_k = self.event_filters[..., 3*Q:]

        # 四元数点积：对 W 和 Q 维度求和，得到 [B, N, K]
        # einsum: 'bnwq,kwq->bnk' 对每个分量
        activation = (
            torch.einsum('bnwq,kwq->bnk', qw_r, ef_r)
            + torch.einsum('bnwq,kwq->bnk', qw_i, ef_i)
            + torch.einsum('bnwq,kwq->bnk', qw_j, ef_j)
            + torch.einsum('bnwq,kwq->bnk', qw_k, ef_k)
        )  # [B, N, K]

        # 缩放
        activation = activation / math.sqrt(Q)

        # Step 4: mask 处理 - 无效节点的激活度设为 -1e4
        activation = activation.masked_fill(mask_sorted.unsqueeze(-1) == 0, -1e4)

        # softmax 归一化（在 K 维度上）
        activation = F.softmax(activation, dim=-1)  # [B, N, K]

        # Step 5: 加权聚合事件滤波器中心向量
        # event_centers: [K, D], activation: [B, N, K]
        event_context = torch.einsum('bnk,kd->bnd', activation, self.event_centers)  # [B, N, D]

        # 输出投影 + LayerNorm
        event_context = self.norm(self.out_proj(event_context))

        # 确保无效节点输出为零
        event_context = event_context * mask_sorted.unsqueeze(-1)

        # Step 6: 恢复原始顺序
        inv_idx = sorted_idx.argsort(dim=1)  # [B, N]
        event_context = event_context.gather(1, inv_idx.unsqueeze(-1).expand(-1, -1, D))

        # 最终确保无效节点为零向量
        event_context = event_context * mask.unsqueeze(-1)

        return event_context


# =============================================================================
# Variable Interaction
# =============================================================================
class VariableInteraction(nn.Module):
    """变量间四元数交互 - 密度调制注意力"""

    def __init__(self, d_model: int, n_vars: int, dropout: float = 0.05):
        super().__init__()
        assert d_model % 4 == 0, "d_model must be divisible by 4 (quaternion constraint)"
        self.d_model = d_model
        self.Q = d_model // 4

        # 变量级别的四元数注意力
        self.q_proj = QLinear(d_model, d_model)
        self.k_proj = QLinear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # 密度调制
        self.density_gate = nn.Sequential(
            nn.Linear(3, 16), nn.GELU(), nn.Linear(16, 1), nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, variable_he: Tensor, density_feat: Tensor,
                var_idx: Tensor, mask: Tensor) -> Tensor:
        """
        Args:
            variable_he: 变量超边表示 [B, V, D]
            density_feat: 密度特征 [B, N, 3]
            var_idx: 变量索引 [B, N] (long型)
            mask: 有效节点掩码 [B, N]
        Returns:
            updated_variable_he: 更新后的变量超边 [B, V, D]
        """
        B, V, D = variable_he.shape
        N = density_feat.shape[1]
        Q = self.Q

        # Step 1: 计算每个变量的平均密度（不使用 torch_scatter）
        var_density = torch.zeros(B, V, 3, device=density_feat.device, dtype=density_feat.dtype)
        var_count = torch.zeros(B, V, 1, device=density_feat.device, dtype=density_feat.dtype)
        var_idx_exp = var_idx.unsqueeze(-1)  # [B, N, 1]
        var_density.scatter_add_(1, var_idx_exp.expand(-1, -1, 3), density_feat * mask.unsqueeze(-1))
        var_count.scatter_add_(1, var_idx_exp, mask.unsqueeze(-1))
        var_density = var_density / var_count.clamp_min(1.0)

        # Step 2: 四元数 Q/K 投影
        q = self.q_proj(variable_he)  # [B, V, D]
        k = self.k_proj(variable_he)  # [B, V, D]
        v = self.v_proj(variable_he)  # [B, V, D]

        # Step 3: 四元数点积注意力分数（R/I/J/K 分量分别 einsum 后求和）
        qr, qi, qj, qk_c = qsplit(q)
        kr, ki, kj, kk_c = qsplit(k)
        scores = (
            torch.einsum('bvd,bwd->bvw', qr, kr)
            + torch.einsum('bvd,bwd->bvw', qi, ki)
            + torch.einsum('bvd,bwd->bvw', qj, kj)
            + torch.einsum('bvd,bwd->bvw', qk_c, kk_c)
        ) / math.sqrt(Q)

        # Step 4: 密度调制
        density_weight = self.density_gate(var_density)  # [B, V, 1]
        scores = scores * (1.0 + density_weight.transpose(1, 2) * 0.5)

        # Step 5: softmax 归一化
        weights = torch.softmax(scores, dim=-1)  # [B, V, V]

        # Step 6: 加权聚合 V
        output = torch.bmm(weights, v)  # [B, V, D]

        # Step 7: 输出投影 + dropout
        output = self.dropout(self.out_proj(output))

        # Step 8: 残差连接 + LayerNorm
        variable_he = self.norm(variable_he + output)

        return variable_he


# =============================================================================
# QSMP Block
# =============================================================================
class QSMPBlock(nn.Module):
    def __init__(self, d_model, n_events, n_vars: int = 1, n_heads: int = 1,
                 spike_slope=5.0, dropout=0.0, no_spike=False, window_size=5,
                 use_var_interaction: bool = True):
        super().__init__()
        self.d_model = d_model
        self.use_var_interaction = use_var_interaction
        self.qsa_t = QuaternionSpikeAttention(d_model, n_heads=n_heads, spike_slope=spike_slope, cross_ctx=True, dropout=0.05)
        self.qsa_v = QuaternionSpikeAttention(d_model, n_heads=n_heads, spike_slope=spike_slope, cross_ctx=True, dropout=0.05)
        self.he_norm_t = nn.LayerNorm(d_model)
        self.he_norm_v = nn.LayerNorm(d_model)
        self.event_conv = TemporalEventConvolution(d_model, n_events, window_size)
        # Node self-attention for HE→Node aggregation
        self.node_self_q = nn.Linear(d_model, d_model)
        self.node_self_k = nn.Linear(3 * d_model, d_model)
        self.node_self_v = nn.Linear(3 * d_model, d_model)
        self.h2n_proj = nn.Linear(3 * d_model, d_model)
        # 可学习的 event context 缩放因子
        self.event_scale = nn.Parameter(torch.tensor(0.0))  # sigmoid(0)=0.5
        self.ffn = nn.Sequential(QLinear(d_model, d_model * 4), nn.GELU(), QLinear(d_model * 4, d_model))
        self.ffn_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self._last_diag = {}

        # Variable-to-Variable interaction (Phase 2)
        if use_var_interaction:
            self.var_interaction = VariableInteraction(d_model, n_vars, dropout)

    def forward(self, q, temporal_he, variable_he, H_temporal, H_variable,
                time_idx, var_idx, mask, density_feat, first_mask=None):
        B, N, D = q.shape
        if first_mask is not None:
            agg_mask_t = H_temporal * first_mask.unsqueeze(1)
            agg_mask_v = H_variable * first_mask.unsqueeze(1)
        else:
            agg_mask_t = H_temporal * mask.unsqueeze(1)
            agg_mask_v = H_variable * mask.unsqueeze(1)

        v_at_node = variable_he.gather(1, var_idx.unsqueeze(-1).expand(-1, -1, D))
        t_at_node = temporal_he.gather(1, time_idx.unsqueeze(-1).expand(-1, -1, D))

        t_msg = self.qsa_t(temporal_he, q, q, agg_mask_t, density_feat, cross_context=v_at_node)
        temporal_he = self.he_norm_t(temporal_he + self.dropout(t_msg))
        v_msg = self.qsa_v(variable_he, q, q, agg_mask_v, density_feat, cross_context=t_at_node)
        variable_he = self.he_norm_v(variable_he + self.dropout(v_msg))

        # Event context via TemporalEventConvolution
        event_context = self.event_conv(q, time_idx, mask)

        # HE -> Node with self-attention
        t_gathered = temporal_he.gather(1, time_idx.unsqueeze(-1).expand(-1, -1, D))
        v_gathered = variable_he.gather(1, var_idx.unsqueeze(-1).expand(-1, -1, D))

        node_kv = torch.cat([t_gathered, v_gathered, q], dim=-1)
        ns_q = self.node_self_q(q)
        ns_k = self.node_self_k(node_kv)
        ns_v = self.node_self_v(node_kv)
        ns_scores = torch.bmm(ns_q, ns_k.transpose(1, 2)) / math.sqrt(D)
        ns_scores = ns_scores.clamp(-6, 6)
        ns_mask = mask.unsqueeze(1) * mask.unsqueeze(2)
        ns_scores = ns_scores.masked_fill(ns_mask == 0, -1e4)
        ns_weights = torch.softmax(ns_scores.clamp(-30, 30), dim=-1)
        node_attn_out = torch.bmm(ns_weights, ns_v)

        h2n = self.h2n_proj(torch.cat([node_attn_out, t_gathered, v_gathered], dim=-1))
        e_scale = torch.sigmoid(self.event_scale)
        q = F.relu(q + self.dropout(h2n) + e_scale * event_context) * mask.unsqueeze(-1)

        # Variable-to-Variable interaction (Phase 2)
        if self.use_var_interaction:
            variable_he = self.var_interaction(variable_he, density_feat, var_idx, mask)

        q = self.ffn_norm(q + self.dropout(self.ffn(q))) * mask.unsqueeze(-1)

        self._last_diag = {"event_scale": float(e_scale.detach().cpu())}
        return q, temporal_he, variable_he


# =============================================================================
# Density features + Main Model
# =============================================================================
def compute_density_features(time_norm, mask, dw=0.08):
    B, N = time_norm.shape
    vt = time_norm.masked_fill(mask == 0, 2.0)
    st, order = vt.sort(dim=1)
    sm = torch.gather(mask, 1, order)
    pg = torch.zeros_like(st); pg[:, 1:] = (st[:, 1:] - st[:, :-1]).clamp_min(0)
    ng = torch.zeros_like(st); ng[:, :-1] = (st[:, 1:] - st[:, :-1]).clamp_min(0)
    pg = pg.masked_fill(sm == 0, 1.0); ng = ng.masked_fill(sm == 0, 1.0)
    inv = torch.empty_like(order)
    inv.scatter_(1, order, torch.arange(N, device=mask.device).unsqueeze(0).expand(B, N))
    pg = torch.gather(pg, 1, inv).clamp(0, 1)
    ng = torch.gather(ng, 1, inv).clamp(0, 1)
    ld = 1.0 / (pg + ng + dw)
    ld = ld / ld.amax(dim=1, keepdim=True).clamp_min(1.0)
    return torch.stack([ld, pg, ng], dim=-1) * mask.unsqueeze(-1)


class Model(nn.Module):
    def __init__(self, configs: ExpConfigs):
        super().__init__()
        self.configs = configs
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.n_layers = configs.n_layers
        n_events = int(getattr(configs, "sqeh_n_events", 32))
        spike_slope = float(getattr(configs, "sqeh_spike_slope", 5.0))
        self.density_window = float(getattr(configs, "sqeh_density_window", 0.08))
        no_spike = bool(int(getattr(configs, "sqeh_no_spike", 0)))
        self.diag_interval = int(getattr(configs, "sqeh_diag_interval", 0))
        dropout = float(getattr(configs, "dropout", 0.05))
        seq_len = getattr(configs, "seq_len_max_irr", None) or configs.seq_len
        pred_len = getattr(configs, "pred_len_max_irr", None) or configs.pred_len
        self.time_length = seq_len + pred_len
        window_size = int(getattr(configs, "sqeh_window_size", 5))
        use_var_interaction = bool(int(getattr(configs, "sqeh_var_interaction", 1)))

        self.encoder = Encoder(self.d_model, self.enc_in)
        self.temporal_he_enc = nn.Linear(1, self.d_model)
        self.variable_he_w = nn.Parameter(torch.randn(self.enc_in, self.d_model) * 0.1)
        self.n_events = n_events
        self.blocks = nn.ModuleList([
            QSMPBlock(self.d_model, n_events, self.enc_in, configs.n_heads, spike_slope, dropout, no_spike, window_size, use_var_interaction)
            for _ in range(self.n_layers)
        ])

        Q = self.d_model // 4
        # 解码器：4 分量感知（R/I/J/K 各自独立解码 + softmax gate）
        self.dec_r = nn.Linear(Q * 3, 1)
        self.dec_i = nn.Linear(Q * 3, 1)
        self.dec_j = nn.Linear(Q * 3, 1)
        self.dec_k = nn.Linear(Q * 3, 1)
        self.dec_gate = nn.Linear(self.d_model * 3, 4)
        self.smooth_alpha = nn.Parameter(torch.tensor(0.3))
        self.register_buffer("_diag_step", torch.zeros(1, dtype=torch.long), persistent=False)

    @staticmethod
    def pad_and_flatten(tensor, mask, max_len):
        B = tensor.shape[0]
        tf, mf = tensor.view(B, -1), mask.view(B, -1)
        dest = torch.cumsum(mf, 1) - 1
        keep = (mf == 1) & (dest < max_len)
        result = torch.zeros(B, max_len, dtype=tensor.dtype, device=tensor.device)
        rows = torch.arange(B, device=tensor.device).unsqueeze(1).expand_as(mf)
        result[rows[keep], dest[keep].long()] = tf[keep]
        return result

    @staticmethod
    def unpad_and_reshape(tf, mask, shape):
        mask = mask.bool()
        result = torch.zeros(shape, dtype=tf.dtype, device=tf.device)
        B, M = tf.shape[:2]
        counts = mask.sum(dim=tuple(range(1, mask.dim())))
        steps = torch.arange(M, device=tf.device).expand(B, M)
        result[mask] = tf[steps < counts.unsqueeze(-1)]
        return result

    def _log_diag(self):
        if self.diag_interval <= 0 or not self.training: return
        step = int(self._diag_step.item())
        if step % self.diag_interval == 0:
            parts = []
            for i, b in enumerate(self.blocks):
                for nm, qsa in [("t", b.qsa_t), ("v", b.qsa_v)]:
                    parts.append(f"L{i}.{nm}_thr={F.softplus(qsa.threshold).item():.3f}")
                es = b._last_diag.get("event_scale", 0)
                parts.append(f"L{i}.e_s={es:.3f}")
            parts.append(f"alpha={torch.sigmoid(self.smooth_alpha).item():.4f}")
            # 解码器 gate 分布（最近一次 forward 的平均）
            if hasattr(self, '_last_gate_dist'):
                parts.append(f"gate={self._last_gate_dist}")
            logger.info(f"[SQEH][s={step}] {' | '.join(parts)}")
        self._diag_step += 1

    def forward(self, x, x_mark=None, x_mask=None, y=None, y_mark=None, y_mask=None,
                exp_stage="train", **kwargs):
        B, SL, EI = x.shape
        YL = self.configs.pred_len if self.configs.pred_len != 0 else SL
        if x_mark is None: x_mark = repeat(torch.arange(SL, dtype=x.dtype, device=x.device)/SL, "L->B L 1", B=B)
        if x_mask is None: x_mask = torch.ones_like(x)
        if y is None: y = torch.zeros((B, YL, EI), dtype=x.dtype, device=x.device)
        if y_mark is None: y_mark = repeat(torch.arange(y.shape[1], dtype=y.dtype, device=y.device)/y.shape[1], "L->B L 1", B=B)
        if y_mask is None: y_mask = torch.ones_like(y)

        PL = y.shape[1]; L = SL + PL
        xym = torch.cat([x_mark[:,:,:1], y_mark[:,:,:1]], 1)
        xL = torch.cat([x, torch.zeros_like(y)], 1)
        xymask = torch.cat([x_mask, y_mask], 1)
        yL = torch.cat([torch.zeros_like(x), y], 1)
        ymL = torch.cat([torch.zeros_like(x_mask), y_mask], 1)

        ti = torch.cumsum(torch.ones_like(xL).long(), 1) - 1
        vi = torch.cumsum(torch.ones_like(xL).long(), -1) - 1
        mb = xymask.bool()
        NMAX = int(xymask.sum((1,2)).max())
        reg = (NMAX == L * EI)

        if reg:
            xf=xL.reshape(B,L*EI); mf=xymask.reshape(B,L*EI)
            yf=yL.reshape(B,L*EI); ymf=ymL.reshape(B,L*EI)
            tf=ti.reshape(B,L*EI); vf=vi.reshape(B,L*EI)
        else:
            xf=self.pad_and_flatten(xL,mb,NMAX); mf=self.pad_and_flatten(xymask,mb,NMAX)
            yf=self.pad_and_flatten(yL,mb,NMAX); ymf=self.pad_and_flatten(ymL,mb,NMAX)
            tf=self.pad_and_flatten(ti,mb,NMAX); vf=self.pad_and_flatten(vi,mb,NMAX)

        N=mf.shape[1]; tfl=tf.long(); vfl=vf.long()
        tff=tf.float()
        tmin=tff.masked_fill(mf==0,float("inf")).min(1,keepdim=True).values
        tmax=tff.masked_fill(mf==0,float("-inf")).max(1,keepdim=True).values
        tn=(tff-tmin)/(tmax-tmin).clamp_min(1.0)

        q = self.encoder(xf, tn, vfl, mf, ymf)

        bi=torch.arange(B,device=x.device).view(-1,1).expand(B,N)
        ni=torch.arange(N,device=x.device).view(1,-1).expand(B,N)
        Ht=torch.zeros(B,L,N,device=x.device,dtype=x.dtype); Ht[bi,tfl,ni]=mf
        Hv=torch.zeros(B,EI,N,device=x.device,dtype=x.dtype); Hv[bi,vfl,ni]=mf

        the=torch.sin(self.temporal_he_enc(xym))
        vhe=F.relu(self.variable_he_w.unsqueeze(0).expand(B,-1,-1))

        df=compute_density_features(tn,mf,self.density_window)
        obs_mask=mf*(1.0-ymf)
        obs_mask=torch.where(obs_mask.sum(1,keepdim=True)>0,obs_mask,mf)

        for i, block in enumerate(self.blocks):
            q, the, vhe = block(q, the, vhe, Ht, Hv, tfl, vfl, mf, df,
                                first_mask=obs_mask if i==0 else None)

        tg=the.gather(1,tfl.unsqueeze(-1).expand(-1,-1,self.d_model))
        vg=vhe.gather(1,vfl.unsqueeze(-1).expand(-1,-1,self.d_model))

        alpha = torch.sigmoid(self.smooth_alpha)
        Ht_norm = Ht / Ht.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        t_prop = torch.bmm(Ht_norm.transpose(1,2), torch.bmm(Ht_norm, q))
        Hv_norm = Hv / Hv.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        v_prop = torch.bmm(Hv_norm.transpose(1,2), torch.bmm(Hv_norm, q))
        q = ((1 - alpha) * q + alpha * 0.5 * (t_prop + v_prop)) * mf.unsqueeze(-1)

        concat_dec = torch.cat([q, tg, vg], -1)
        q_r, q_i, q_j, q_k = qsplit(q)
        t_r, t_i, t_j, t_k = qsplit(tg)
        v_r, v_i, v_j, v_k = qsplit(vg)
        pred_r = self.dec_r(torch.cat([q_r, t_r, v_r], -1)).squeeze(-1)
        pred_i = self.dec_i(torch.cat([q_i, t_i, v_i], -1)).squeeze(-1)
        pred_j = self.dec_j(torch.cat([q_j, t_j, v_j], -1)).squeeze(-1)
        pred_k = self.dec_k(torch.cat([q_k, t_k, v_k], -1)).squeeze(-1)
        gate = torch.softmax(self.dec_gate(concat_dec), dim=-1)
        pred = (gate[..., 0] * pred_r + gate[..., 1] * pred_i
              + gate[..., 2] * pred_j + gate[..., 3] * pred_k) * mf

        # 记录 gate 分布用于诊断
        if self.training:
            with torch.no_grad():
                gm = gate[mf > 0].mean(0)
                self._last_gate_dist = f"[R={gm[0]:.2f},I={gm[1]:.2f},J={gm[2]:.2f},K={gm[3]:.2f}]"

        self._log_diag()

        if exp_stage in ("train", "val"):
            return {"pred": pred, "true": yf, "mask": ymf, "aux_loss": torch.tensor(0.0, device=pred.device)}

        p = self.unpad_and_reshape(pred, torch.cat([x_mask, y_mask], 1), (B, SL+PL, EI))
        fd = -1 if self.configs.features == "MS" else 0
        return {"pred": p[:, -PL:, fd:], "true": y[:, :, fd:], "mask": y_mask[:, :, fd:]}
