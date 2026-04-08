# QSH-Net v8: Quaternion-Spiking-Hypergraph Network
#
# v8: Deep fusion of Quaternion, Spiking, and Hypergraph within message passing:
#
# 1. Quaternion Hamilton product (Kronecker block matrix) replaces Linear in
#    hyperedge→node fusion (h2n), so the cross-variable information merging
#    directly leverages quaternion's inter-component coupling.
#
# 2. Time-Aware Spiking Gate uses irregular time intervals Δt to modulate
#    membrane potential via LIF-style exponential decay: membrane *= exp(-Δt/τ).
#    This makes the spike mechanism aware of observation density.
#
# 3. Causal temporal masking: node→temporal_hyperedge attention is masked so
#    each temporal hyperedge only aggregates from nodes at current or past
#    timesteps, enforcing strict causality (time's arrow).

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat, rearrange
from torch import Tensor
from utils.ExpConfigs import ExpConfigs
from utils.globals import logger


# ============================================================================
# Quaternion (our addition)
# ============================================================================

class QuaternionLinear(nn.Module):
    def __init__(self, in_f, out_f, bias=True):
        super().__init__()
        assert in_f % 4 == 0 and out_f % 4 == 0
        qi, qo = in_f // 4, out_f // 4
        self.r = nn.Parameter(torch.empty(qo, qi))
        self.i = nn.Parameter(torch.empty(qo, qi))
        self.j = nn.Parameter(torch.empty(qo, qi))
        self.k = nn.Parameter(torch.empty(qo, qi))
        self.bias = nn.Parameter(torch.zeros(out_f)) if bias else None
        s = (2.0 * qi) ** -0.5
        for w in [self.r, self.i, self.j, self.k]:
            nn.init.normal_(w, 0, s)

    def forward(self, x):
        r, i, j, k = self.r, self.i, self.j, self.k
        W = torch.cat([torch.cat([r,-i,-j,-k],1), torch.cat([i,r,-k,j],1),
                        torch.cat([j,k,r,-i],1), torch.cat([k,-j,i,r],1)], 0)
        return F.linear(x, W, self.bias)


class QuaternionBlock(nn.Module):
    def __init__(self, d, dropout=0.1):
        super().__init__()
        self.fc = QuaternionLinear(d, d)
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        q = x.shape[-1] // 4
        h = self.fc(x)
        h = torch.cat([F.gelu(h[..., c*q:(c+1)*q]) for c in range(4)], -1)
        return self.norm(x + self.drop(h))


# ============================================================================
# Spiking (our addition)
# ============================================================================

class SpikeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, threshold, gamma):
        ctx.save_for_backward(x)
        ctx.threshold = threshold
        ctx.gamma = gamma
        return (x >= threshold).float()

    @staticmethod
    def backward(ctx, grad):
        x, = ctx.saved_tensors
        s = torch.sigmoid(ctx.gamma * (x - ctx.threshold))
        return grad * ctx.gamma * s * (1 - s), None, None


class TimeAwareSpikingGate(nn.Module):
    """
    Time-aware spiking gate: membrane potential decays with exp(-Δt/τ),
    modeling the LIF neuron's leaky integration over irregular time intervals.
    
    When Δt is large (long gap since last observation), membrane decays → spike
    unlikely → information from temporally isolated nodes is attenuated.
    When Δt is small (dense observations), membrane stays high → spike fires →
    information from dense regions is amplified.
    
    This is the core SNN contribution: event-driven sparse activation that
    respects the irregular temporal structure of IMTS data.
    """
    def __init__(self, d):
        super().__init__()
        self.proj = nn.Linear(d, d)
        self.threshold = nn.Parameter(torch.tensor(0.5))
        self.gamma = nn.Parameter(torch.tensor(5.0))
        # Learnable time constant τ (log-parameterized for positivity)
        self.log_tau = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, mask_d, dt):
        """
        x: (B, N, D) node features
        mask_d: (B, N, D) observation mask expanded to D
        dt: (B, N) time intervals between consecutive observations
        """
        # LIF-style membrane potential with temporal decay
        tau = torch.exp(self.log_tau) + 1e-4  # ensure τ > 0
        decay = torch.exp(-dt.unsqueeze(-1) / tau)  # (B, N, 1), decays for large Δt
        membrane = self.proj(x) * decay
        spike = SpikeFunction.apply(membrane, self.threshold, self.gamma.abs() + 0.1)
        return x * (1.0 + spike) * mask_d


# ============================================================================
# MultiHeadAttentionBlock (identical to HyperIMTS)
# ============================================================================

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, n_dim, num_heads, ln=False):
        super().__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.n_dim = n_dim
        self.fc_q = nn.Linear(dim_Q, n_dim)
        self.fc_k = nn.Linear(dim_K, n_dim)
        self.fc_v = nn.Linear(dim_K, n_dim)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(n_dim, n_dim)

    def forward(self, Q, K, mask=None):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)
        ds = self.n_dim // self.num_heads
        Q_ = torch.cat(Q.split(ds, 2), 0)
        K = torch.cat(K.split(ds, 2), 0)
        V = torch.cat(V.split(ds, 2), 0)
        A = Q_.bmm(K.transpose(1, 2)) / math.sqrt(self.n_dim)
        if mask is not None:
            A = A.masked_fill(mask.repeat(self.num_heads, 1, 1) == 0, -1e9)
        A = torch.softmax(A, 2)
        O = torch.cat((Q_ + A.bmm(V)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O


# ============================================================================
# IrregularityAwareAttention (identical to HyperIMTS)
# ============================================================================

class IrregularityAwareAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.scale = d_model ** 0.5
        self.threshold = nn.Parameter(torch.tensor(0.5), requires_grad=True)

    def forward(self, x, query_aux=None, key_aux=None, adjacency_mask=None, merge_coefficients=None):
        batch_size, n_variables, hidden_dim = x.shape
        query = self.query_proj(x)
        key = self.key_proj(x)
        value = self.value_proj(x)
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale
        mask_value = torch.finfo(attention_scores.dtype).min
        if query_aux is not None and key_aux is not None:
            attention_scores_aux = torch.matmul(query_aux, key_aux.transpose(-2, -1)) / (key_aux.shape[-1] ** 0.5)
            non_zero_mask = (attention_scores_aux != 0)
            positive_mask = (attention_scores > self.threshold)
            mask = positive_mask & non_zero_mask
            attention_scores[mask] = ((1 - merge_coefficients) * attention_scores + merge_coefficients * attention_scores_aux)[mask]
        if adjacency_mask is not None:
            attention_scores = attention_scores.masked_fill(adjacency_mask == 0, mask_value)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        return torch.matmul(attention_weights, value)


# ============================================================================
# HypergraphEncoder (identical to HyperIMTS)
# ============================================================================

class HypergraphEncoder(nn.Module):
    def __init__(self, enc_in, time_length, d_model):
        super().__init__()
        self.enc_in = enc_in
        self.d_model = d_model
        self.variable_hyperedge_weights = nn.Parameter(torch.randn(enc_in, d_model), requires_grad=True)
        self.relu = nn.ReLU()
        self.observation_node_encoder = nn.Linear(2, d_model)
        self.temporal_hyperedge_encoder = nn.Linear(1, d_model)

    def forward(self, x_L_flattened, x_y_mask_flattened, y_mask_L_flattened,
                x_y_mark, variable_indices_flattened, time_indices_flattened, N_OBSERVATIONS_MAX):
        B = x_L_flattened.shape[0]
        E, L, D = self.enc_in, x_y_mark.shape[1], self.d_model
        N = N_OBSERVATIONS_MAX

        x_L_flattened = torch.stack([x_L_flattened, 1 - x_y_mask_flattened + y_mask_L_flattened], dim=-1)

        temporal_incidence_matrix = repeat(time_indices_flattened, "B N -> B L N", L=L)
        temporal_incidence_matrix = (temporal_incidence_matrix == repeat(
            torch.ones(B, L, device=x_L_flattened.device).cumsum(dim=1),
            "B L -> B L N", N=N) - 1).float()
        temporal_incidence_matrix = temporal_incidence_matrix * repeat(x_y_mask_flattened, "B N -> B L N", L=L)

        variable_incidence_matrix = repeat(
            torch.ones(B, E, device=x_L_flattened.device).cumsum(dim=1) - 1,
            "B E -> B E N", N=N)
        variable_incidence_matrix = (variable_incidence_matrix == repeat(
            variable_indices_flattened, "B N -> B E N", E=E)).float()
        variable_incidence_matrix = variable_incidence_matrix * repeat(x_y_mask_flattened, "B N -> B E N", E=E)

        observation_nodes = self.relu(self.observation_node_encoder(x_L_flattened)) * repeat(
            x_y_mask_flattened, "B N -> B N D", D=D)
        temporal_hyperedges = torch.sin(self.temporal_hyperedge_encoder(x_y_mark))
        variable_hyperedges = self.relu(repeat(
            self.variable_hyperedge_weights, "E D -> B E D", B=B))

        return (observation_nodes, temporal_hyperedges, variable_hyperedges,
                temporal_incidence_matrix, variable_incidence_matrix)


# ============================================================================
# HypergraphLearner (HyperIMTS structure + Quaternion/Spike enhancements)
# ============================================================================

class HypergraphLearner(nn.Module):
    def __init__(self, n_layers, d_model, n_heads, time_length,
                 use_quaternion_h2n=True, use_spiking=True, use_causal_mask=True):
        super().__init__()
        self.n_layers = n_layers
        self.d_model = d_model
        self.activation = nn.ReLU()
        self.use_quaternion_h2n = use_quaternion_h2n
        self.use_spiking = use_spiking
        self.use_causal_mask = use_causal_mask

        # Identical to HyperIMTS: node→hyperedge attention
        self.node2temporal_hyperedge = nn.ModuleList([
            MultiHeadAttentionBlock(d_model, 2*d_model, 2*d_model, d_model, n_heads)
            for _ in range(n_layers)])
        self.node2variable_hyperedge = nn.ModuleList([
            MultiHeadAttentionBlock(d_model, 2*d_model, 2*d_model, d_model, n_heads)
            for _ in range(n_layers)])
        self.node_self_update = nn.ModuleList([
            MultiHeadAttentionBlock(d_model, 3*d_model, 3*d_model, d_model, n_heads)
            for _ in range(n_layers)])

        # h2n: QuaternionLinear or plain Linear depending on ablation flag
        if use_quaternion_h2n:
            self.hyperedge2node = nn.ModuleList([
                QuaternionLinear(3*d_model, d_model) for _ in range(n_layers)])
        else:
            self.hyperedge2node = nn.ModuleList([
                nn.Linear(3*d_model, d_model) for _ in range(n_layers)])

        self.variable_hyperedge2variable_hyperedge = IrregularityAwareAttention(d_model)
        self.hyperedge2hyperedge_layers = [n_layers - 1]
        self.scale = 1 / time_length
        self.oom_flag = False

        # Spiking gate (only created if enabled)
        if use_spiking:
            self.spikes = nn.ModuleList([TimeAwareSpikingGate(d_model) for _ in range(n_layers)])

    def get_fine_grained_embedding(self, tensor_flattened, target_shape):
        """Identical to HyperIMTS."""
        B, L, E = target_shape.shape
        D = tensor_flattened.shape[-1]
        nd = max(1, int(D * self.scale))
        tf = tensor_flattened[:, :, :nd]
        mask = (target_shape > 0).unsqueeze(-1).expand(-1, -1, -1, nd)
        result = torch.zeros(B, L, E, nd, dtype=tf.dtype, device=tf.device)
        result.masked_scatter_(mask, tf)
        return rearrange(result, "B L E D -> B E (L D)")

    def forward(self, observation_nodes, temporal_hyperedges, variable_hyperedges,
                time_indices_flattened, variable_indices_flattened,
                temporal_incidence_matrix, variable_incidence_matrix,
                x_y_mask_flattened, x_y_mask, y_mask_L_flattened,
                dt_flattened):
        D = self.d_model
        L = temporal_incidence_matrix.shape[1]

        # Causal temporal mask (only if enabled)
        if self.use_causal_mask:
            node_times = time_indices_flattened.unsqueeze(1).float()
            he_times = torch.arange(L, device=node_times.device).view(1, L, 1).float()
            causal_t_mask = (node_times <= he_times).float()
            causal_t_inc = temporal_incidence_matrix * causal_t_mask
        else:
            causal_t_inc = temporal_incidence_matrix

        for i in range(self.n_layers):
            if i == 0:
                mask_temp = 1 - repeat(y_mask_L_flattened, "B N -> B L N", L=L)
                mask_temp[mask_temp == 0] = 1e-8

            inc_t = causal_t_inc if i != 0 else causal_t_inc * mask_temp
            temporal_hyperedges_updated = self.node2temporal_hyperedge[i](
                temporal_hyperedges,
                torch.cat([variable_hyperedges.gather(1, repeat(variable_indices_flattened, "B N -> B N D", D=D)),
                           observation_nodes], -1),
                inc_t)

            if i == 0:
                mask_temp = 1 - repeat(y_mask_L_flattened, "B N -> B E N", E=variable_incidence_matrix.shape[1])
                mask_temp[mask_temp == 0] = 1e-8

            variable_hyperedges_updated = self.node2variable_hyperedge[i](
                variable_hyperedges,
                torch.cat([temporal_hyperedges.gather(1, repeat(time_indices_flattened, "B N -> B N D", D=D)),
                           observation_nodes], -1),
                variable_incidence_matrix if i != 0 else variable_incidence_matrix * mask_temp)

            variable_hyperedges = variable_hyperedges_updated
            temporal_hyperedges = temporal_hyperedges_updated

            tg = temporal_hyperedges.gather(1, repeat(time_indices_flattened, "B N -> B N D", D=D))
            vg = variable_hyperedges.gather(1, repeat(variable_indices_flattened, "B N -> B N D", D=D))
            md = repeat(x_y_mask_flattened, "B N -> B N D", D=D)

            if not self.oom_flag:
                try:
                    obs_updated = self.node_self_update[i](
                        observation_nodes,
                        torch.cat([tg, vg, observation_nodes], -1),
                        x_y_mask_flattened.unsqueeze(2) * x_y_mask_flattened.unsqueeze(1))
                    observation_nodes = self.activation(
                        (observation_nodes + self.hyperedge2node[i](
                            torch.cat([obs_updated, tg, vg], -1))) * md)
                except:
                    self.oom_flag = True
                    logger.warning("QSH-Net: CUDA OOM, switching to fallback path.")

            if self.oom_flag:
                observation_nodes = self.activation(
                    (observation_nodes + self.hyperedge2node[i](
                        torch.cat([observation_nodes, tg, vg], -1))) * md)

            # Spiking gate (only if enabled)
            if self.use_spiking:
                observation_nodes = self.spikes[i](observation_nodes, md, dt_flattened)

            if i in self.hyperedge2hyperedge_layers:
                sync_mask = x_y_mask
                qk = self.get_fine_grained_embedding(observation_nodes, sync_mask)
                mc = sync_mask.transpose(-1, -2) @ sync_mask
                nopv = mc.diagonal(0, -2, -1)
                mc[nopv != 0] = (mc / repeat(nopv, "B E -> B E E2", E2=sync_mask.shape[-1]))[nopv != 0]
                variable_hyperedges = variable_hyperedges + self.variable_hyperedge2variable_hyperedge(
                    x=variable_hyperedges, query_aux=qk, key_aux=qk, merge_coefficients=mc)

        return observation_nodes, temporal_hyperedges, variable_hyperedges


# ============================================================================
# Main Model
# ============================================================================

class Model(nn.Module):
    """
    QSH-Net with ablation support.
    
    Ablation flags (set via model_id string parsing for script compatibility):
    - use_quaternion_block: QuaternionBlock after encoder (default: True)
    - use_quaternion_h2n: QuaternionLinear in h2n fusion (default: True)
    - use_spiking: TimeAwareSpikingGate per layer (default: True)
    - use_causal_mask: Causal temporal incidence masking (default: True)
    
    model_id format: "QSHNet" or "QSHNet_noQB_noQH_noSP_noCM" etc.
    """
    def __init__(self, configs: ExpConfigs):
        super().__init__()
        self.configs = configs
        self.enc_in = configs.enc_in
        self.d_model = (configs.d_model // 4) * 4
        if self.d_model != configs.d_model:
            logger.warning(f"QSH-Net: d_model {configs.d_model}->{self.d_model} (×4)")
        D = self.d_model
        sl = configs.seq_len_max_irr or configs.seq_len
        pl = configs.pred_len_max_irr or configs.pred_len
        tl = sl + pl

        # Parse ablation flags from model_id
        mid = configs.model_id if configs.model_id else ""
        self.use_quaternion_block = "noQB" not in mid
        use_quaternion_h2n = "noQH" not in mid
        use_spiking = "noSP" not in mid
        use_causal_mask = "noCM" not in mid

        # HyperIMTS core
        self.hypergraph_encoder = HypergraphEncoder(self.enc_in, tl, D)
        self.hypergraph_learner = HypergraphLearner(
            configs.n_layers, D, configs.n_heads, tl,
            use_quaternion_h2n=use_quaternion_h2n,
            use_spiking=use_spiking,
            use_causal_mask=use_causal_mask)
        self.hypergraph_decoder = nn.Linear(3 * D, 1)

        # QSH-Net: QuaternionBlock (only if enabled)
        if self.use_quaternion_block:
            self.quat = QuaternionBlock(D, configs.dropout)

    # Pad/flatten utilities (identical to HyperIMTS v2)
    def pad_and_flatten(self, tensor, mask, max_len):
        B = tensor.shape[0]
        tf = tensor.view(B, -1)
        mf = mask.view(B, -1)
        d = torch.cumsum(mf, 1) - 1
        k = (mf == 1) & (d < max_len)
        r = torch.zeros(B, max_len, dtype=tensor.dtype, device=tensor.device)
        rows = torch.arange(B, device=tensor.device).unsqueeze(1).expand_as(mf)
        r[rows[k], d[k]] = tf[k]
        return r

    def unpad_and_reshape(self, tensor_flattened, original_mask, original_shape):
        original_mask = original_mask.bool()
        device = tensor_flattened.device
        result = torch.zeros(original_shape, dtype=tensor_flattened.dtype, device=device)
        counts = original_mask.sum(dim=tuple(range(1, original_mask.dim())))
        batch_size, max_len = tensor_flattened.shape[:2]
        steps = torch.arange(max_len, device=device).expand(batch_size, max_len)
        src_mask = steps < counts.unsqueeze(-1)
        result[original_mask] = tensor_flattened[src_mask]
        return result

    def forward(self, x, x_mark=None, x_mask=None, y=None, y_mark=None, y_mask=None,
                x_L_flattened=None, x_y_mask_flattened=None,
                y_L_flattened=None, y_mask_L_flattened=None,
                exp_stage="train", **kwargs):
        # === Adaptor (identical to HyperIMTS) ===
        BATCH_SIZE, SEQ_LEN, ENC_IN = x.shape
        Y_LEN = self.configs.pred_len if self.configs.pred_len != 0 else SEQ_LEN
        if x_mark is None:
            x_mark = repeat(torch.arange(end=x.shape[1], dtype=x.dtype, device=x.device) / x.shape[1], "L -> B L 1", B=x.shape[0])
        if x_mask is None:
            x_mask = torch.ones_like(x, device=x.device, dtype=x.dtype)
        if y is None:
            y = torch.ones((BATCH_SIZE, Y_LEN, ENC_IN), dtype=x.dtype, device=x.device)
        if y_mark is None:
            y_mark = repeat(torch.arange(end=y.shape[1], dtype=y.dtype, device=y.device) / y.shape[1], "L -> B L 1", B=y.shape[0])
        if y_mask is None:
            y_mask = torch.ones_like(y, device=y.device, dtype=y.dtype)

        BATCH_SIZE, SEQ_LEN, ENC_IN = x.shape
        _, PRED_LEN, _ = y.shape
        L = SEQ_LEN + PRED_LEN
        x_mark = x_mark[:, :, :1]
        y_mark = y_mark[:, :, :1]

        if self.configs.task_name in ["short_term_forecast", "long_term_forecast", "classification", "representation_learning"]:
            x_zeros = torch.zeros_like(y, dtype=x.dtype, device=x.device)
            y_zeros = torch.zeros_like(x, dtype=y.dtype, device=y.device)
            x_y_mark = torch.cat([x_mark, y_mark], dim=1)
            x_L = torch.cat([x, x_zeros], dim=1)
            x_y_mask = torch.cat([x_mask, y_mask], dim=1)
            y_L = torch.cat([y_zeros, y], dim=1)
            y_mask_L = torch.cat([y_zeros, y_mask], dim=1)
        elif self.configs.task_name in ["imputation"]:
            x_y_mark = x_mark; x_L = x; x_y_mask = x_mask + y_mask; y_L = y; y_mask_L = y_mask
        else:
            raise NotImplementedError()

        time_indices = torch.cumsum(torch.ones_like(x_L).to(torch.int64), dim=1) - 1
        variable_indices = torch.cumsum(torch.ones_like(x_L).to(torch.int64), dim=-1) - 1
        x_y_mask_bool = x_y_mask.to(torch.bool)

        N_OBSERVATIONS_MAX = torch.max(x_y_mask.sum((1, 2))).to(torch.int64)
        N_OBSERVATIONS_MIN = torch.min(x_y_mask.sum((1, 2))).to(torch.int64)
        is_regular = (N_OBSERVATIONS_MAX == N_OBSERVATIONS_MIN == L * ENC_IN)

        if (x_L_flattened or x_y_mask_flattened or y_L_flattened or y_mask_L_flattened) is None:
            if is_regular:
                x_L_flattened = x_L.reshape(BATCH_SIZE, L * ENC_IN)
                x_y_mask_flattened = x_y_mask.reshape(BATCH_SIZE, L * ENC_IN)
                y_L_flattened = y_L.reshape(BATCH_SIZE, L * ENC_IN)
                y_mask_L_flattened = y_mask_L.reshape(BATCH_SIZE, L * ENC_IN)
            else:
                x_L_flattened = self.pad_and_flatten(x_L, x_y_mask_bool, N_OBSERVATIONS_MAX)
                x_y_mask_flattened = self.pad_and_flatten(x_y_mask, x_y_mask_bool, N_OBSERVATIONS_MAX)
                y_L_flattened = self.pad_and_flatten(y_L, x_y_mask_bool, N_OBSERVATIONS_MAX)
                y_mask_L_flattened = self.pad_and_flatten(y_mask_L, x_y_mask_bool, N_OBSERVATIONS_MAX)

        if is_regular:
            time_indices_flattened = time_indices.reshape(BATCH_SIZE, L * ENC_IN)
            variable_indices_flattened = variable_indices.reshape(BATCH_SIZE, L * ENC_IN)
        else:
            time_indices_flattened = self.pad_and_flatten(time_indices, x_y_mask_bool, N_OBSERVATIONS_MAX)
            variable_indices_flattened = self.pad_and_flatten(variable_indices, x_y_mask_bool, N_OBSERVATIONS_MAX)

        # === Encode ===
        (observation_nodes, temporal_hyperedges, variable_hyperedges,
         temporal_incidence_matrix, variable_incidence_matrix
        ) = self.hypergraph_encoder(
            x_L_flattened, x_y_mask_flattened, y_mask_L_flattened,
            x_y_mark, variable_indices_flattened, time_indices_flattened, N_OBSERVATIONS_MAX)

        # === QSH-Net addition: Quaternion enrichment of node features (if enabled) ===
        if self.use_quaternion_block:
            observation_nodes = self.quat(observation_nodes)

        # === Compute Δt for time-aware spiking gates ===
        # Gather per-observation timestamps, then compute intervals
        timestamps_flat = x_y_mark.squeeze(-1)  # (B, L)
        t_per_obs = timestamps_flat.gather(1, time_indices_flattened.clamp(0, L-1).long())  # (B, N)
        dt_flattened = torch.zeros_like(t_per_obs)
        dt_flattened[:, 1:] = (t_per_obs[:, 1:] - t_per_obs[:, :-1]).clamp(min=0)

        # === Hypergraph learning (with quaternion h2n, time-aware spikes, causal masking) ===
        (observation_nodes, temporal_hyperedges, variable_hyperedges
        ) = self.hypergraph_learner(
            observation_nodes, temporal_hyperedges, variable_hyperedges,
            time_indices_flattened, variable_indices_flattened,
            temporal_incidence_matrix, variable_incidence_matrix,
            x_y_mask_flattened, x_y_mask, y_mask_L_flattened,
            dt_flattened)

        # === Decode ===
        if self.configs.task_name in ['long_term_forecast', 'short_term_forecast', 'imputation']:
            D = self.d_model
            pred_flattened = self.hypergraph_decoder(torch.cat([
                observation_nodes,
                temporal_hyperedges.gather(1, repeat(time_indices_flattened, "B N -> B N D", D=D)),
                variable_hyperedges.gather(1, repeat(variable_indices_flattened, "B N -> B N D", D=D)),
            ], dim=-1)).squeeze(-1)

            if exp_stage in ["train", "val"]:
                return {"pred": pred_flattened, "true": y_L_flattened, "mask": y_mask_L_flattened}
            else:
                pred = self.unpad_and_reshape(
                    pred_flattened, torch.cat([x_mask, y_mask], dim=1),
                    (BATCH_SIZE, SEQ_LEN + PRED_LEN, ENC_IN))
                f_dim = -1 if self.configs.features == 'MS' else 0
                return {"pred": pred[:, -PRED_LEN:, f_dim:],
                        "true": y[:, :, f_dim:],
                        "mask": y_mask[:, :, f_dim:]}
        else:
            raise NotImplementedError()
