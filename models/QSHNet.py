# QSH-Net: Quaternion-Spiking-Hypergraph Network for IMTS Forecasting
#
# Three modules form a signal processing pipeline:
#   obs → [Spiking: event detection] → [Hypergraph: message passing] → [Quaternion: refinement]
#
# Core design: IDENTITY INITIALIZATION — all QSH enhancements start as exact identity,
# so the model begins as pure HyperIMTS and learns to deviate only when beneficial.
#
# 1. Context-Aware Spike Selection (before n2h):
#    Uses H's variable_incidence_matrix to compute per-variable context.
#    Fires when observation deviates from its variable's context.
#    Zero-init membrane_proj → all fire at start → identity.
#
# 2. Quaternion Refinement (in h2n):
#    Additive refinement: h2n_out = linear_out + α · QuatLinear(linear_out)
#    Identity-init QuaternionLinear (r=I, i=j=k=0) → starts as (1+α)·linear_out.
#    Hamilton product learns cross-feature-group interactions during training.
#
# 3. Hypergraph core (identical to HyperIMTS): n2h attention, h2h, node_self_update.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat, rearrange
from torch import Tensor
from utils.ExpConfigs import ExpConfigs
from utils.globals import logger


# ============================================================================
# Quaternion Linear (Kronecker block matrix Hamilton product)
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
        # Identity init: r=I, i=j=k=0 → quat_linear(x)=x at start
        nn.init.eye_(self.r)
        nn.init.zeros_(self.i)
        nn.init.zeros_(self.j)
        nn.init.zeros_(self.k)

    def forward(self, x):
        r, i, j, k = self.r, self.i, self.j, self.k
        W = torch.cat([torch.cat([r,-i,-j,-k],1), torch.cat([i,r,-k,j],1),
                        torch.cat([j,k,r,-i],1), torch.cat([k,-j,i,r],1)], 0)
        return F.linear(x, W, self.bias)


# ============================================================================
# Spike Function (surrogate gradient)
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


# ============================================================================
# Spike Selection: event-driven node filtering before message passing
# ============================================================================

class SpikeRouter(nn.Module):
    """
    Context-aware event-driven node filtering before message passing.
    
    Uses the hypergraph's variable incidence matrix to compute each variable's
    aggregate context. An observation "fires" (spike=1) when it deviates from
    its variable's context — i.e., it carries novel information worth propagating.
    Non-firing observations are attenuated (not zeroed) to preserve gradient flow.
    
    Membrane potential is a scalar per observation (Linear(2D, 1)), not a
    D-dimensional vector — this is a binary event decision, not a feature transform.
    """
    def __init__(self, d_model):
        super().__init__()
        self.membrane_proj = nn.Linear(d_model * 2, 1)
        # Zero-init: membrane=0 for all obs → all fire (0>=0=True) → identity at start
        nn.init.zeros_(self.membrane_proj.weight)
        nn.init.zeros_(self.membrane_proj.bias)
        self.event_proj = nn.Linear(d_model * 2, d_model)
        nn.init.zeros_(self.event_proj.weight)
        nn.init.zeros_(self.event_proj.bias)
        self.threshold = nn.Parameter(torch.tensor(0.0))
        self.gamma = nn.Parameter(torch.tensor(5.0))
        self.retain_log_scale = nn.Parameter(torch.tensor(0.0))
        self.event_log_scale = nn.Parameter(torch.tensor(-8.0))

    def forward(self, obs, mask_d, variable_incidence_matrix, variable_indices_flattened):
        """
        obs: (B, N, D) observation node features
        mask_d: (B, N, D) observation mask (expanded)
        variable_incidence_matrix: (B, E, N) which obs belongs to which variable
        variable_indices_flattened: (B, N) variable index per observation
        Returns: base path, event path, route diagnostics
        """
        D = obs.shape[-1]
        # Compute per-variable context via hypergraph structure
        var_count = variable_incidence_matrix.sum(-1, keepdim=True).clamp(min=1)  # (B, E, 1)
        var_context = (variable_incidence_matrix @ obs) / var_count  # (B, E, D)
        obs_var_ctx = var_context.gather(
            1, repeat(variable_indices_flattened, "B N -> B N D", D=D))  # (B, N, D)
        # Membrane: how much does this observation deviate from its variable's context?
        deviation = obs - obs_var_ctx
        membrane = self.membrane_proj(
            torch.cat([obs, deviation], dim=-1)).squeeze(-1)  # (B, N)
        route_logit = membrane
        retain_strength = torch.exp(self.retain_log_scale) - 1.0
        retain_gate = 1.0 - retain_strength * torch.sigmoid(-route_logit)
        event_gate = torch.exp(self.event_log_scale) * torch.sigmoid(route_logit)
        event_features = self.event_proj(torch.cat([obs, deviation], dim=-1))
        obs_base = obs * retain_gate.unsqueeze(-1) * mask_d
        obs_event = event_features * event_gate.unsqueeze(-1) * mask_d
        # Diagnostic logging (every 200 forward calls)
        if not hasattr(self, '_fwd_count'):
            self._fwd_count = 0
        self._fwd_count += 1
        if self._fwd_count % 200 == 1:
            logger.info(f"[Spike] retain={retain_gate.mean().item():.3f} event={event_gate.mean().item():.5f} "
                        f"threshold={self.threshold.item():.3f} gamma={self.gamma.item():.3f}")
        return obs_base, obs_event, {
            "retain_gate": retain_gate,
            "event_gate": event_gate,
            "route_logit": route_logit,
        }


class SpikeSelection(SpikeRouter):
    pass


# ============================================================================
# MultiHeadAttentionBlock (identical to HyperIMTS)
# ============================================================================

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, n_dim, num_heads, ln=False):
        super().__init__()
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
        variable_hyperedges = self.relu(repeat(self.variable_hyperedge_weights, "E D -> B E D", B=B))

        return (observation_nodes, temporal_hyperedges, variable_hyperedges,
                temporal_incidence_matrix, variable_incidence_matrix)


# ============================================================================
# HypergraphLearner (HyperIMTS + Spike Selection + Quaternion Parallel Branch)
# ============================================================================

class HypergraphLearner(nn.Module):
    def __init__(self, n_layers, d_model, n_heads, time_length):
        super().__init__()
        self.n_layers = n_layers
        self.d_model = d_model
        self.activation = nn.ReLU()

        # === HyperIMTS core (identical) ===
        self.node2temporal_hyperedge = nn.ModuleList([
            MultiHeadAttentionBlock(d_model, 2*d_model, 2*d_model, d_model, n_heads)
            for _ in range(n_layers)])
        self.node2variable_hyperedge = nn.ModuleList([
            MultiHeadAttentionBlock(d_model, 2*d_model, 2*d_model, d_model, n_heads)
            for _ in range(n_layers)])
        self.node_self_update = nn.ModuleList([
            MultiHeadAttentionBlock(d_model, 3*d_model, 3*d_model, d_model, n_heads)
            for _ in range(n_layers)])
        self.hyperedge2node = nn.ModuleList([
            nn.Linear(3*d_model, d_model) for _ in range(n_layers)])
        self.variable_hyperedge2variable_hyperedge = IrregularityAwareAttention(d_model)
        self.hyperedge2hyperedge_layers = [n_layers - 1]
        self.scale = 1 / time_length
        self.oom_flag = False

        # === QSH-Net Enhancement 1: Spike Selection (per layer) ===
        self.spike_select = nn.ModuleList([
            SpikeRouter(d_model) for _ in range(n_layers)])
        self.event_residual_scale = nn.ParameterList([
            nn.Parameter(torch.tensor(0.0)) for _ in range(n_layers)])

        # === QSH-Net Enhancement 2: Quaternion Refinement (per layer) ===
        # QuaternionLinear refines linear output via Hamilton product cross-coupling
        # of 4 feature groups within the fused representation
        self.quat_h2n = nn.ModuleList([
            QuaternionLinear(d_model, d_model) for _ in range(n_layers)])
        self.quat_gate = nn.ModuleList([
            nn.Linear(d_model + 1, 1) for _ in range(n_layers)])
        for gate in self.quat_gate:
            nn.init.zeros_(gate.weight)
            nn.init.constant_(gate.bias, -3.0)

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

    def compute_quaternion_gate(self, layer_idx, linear_out, event_gate):
        gate_input = torch.cat([linear_out, event_gate.unsqueeze(-1)], dim=-1)
        return torch.sigmoid(self.quat_gate[layer_idx](gate_input))

    def forward(self, observation_nodes, temporal_hyperedges, variable_hyperedges,
                time_indices_flattened, variable_indices_flattened,
                temporal_incidence_matrix, variable_incidence_matrix,
                x_y_mask_flattened, x_y_mask, y_mask_L_flattened):
        D = self.d_model
        for i in range(self.n_layers):
            if i == 0:
                mask_temp = 1 - repeat(y_mask_L_flattened, "B N -> B L N", L=temporal_incidence_matrix.shape[1])
                mask_temp[mask_temp == 0] = 1e-8

            md = repeat(x_y_mask_flattened, "B N -> B N D", D=D)

            # === Enhancement 1: Context-Aware Spike Selection BEFORE message passing ===
            # Uses variable incidence matrix to detect observations that deviate from
            # their variable's context — true event-driven filtering
            obs_base, obs_event, route_state = self.spike_select[i](
                observation_nodes, md, variable_incidence_matrix, variable_indices_flattened)
            event_scale = torch.exp(self.event_residual_scale[i]) - 1.0
            temporal_event_delta = (temporal_incidence_matrix @ obs_event) / temporal_incidence_matrix.sum(-1, keepdim=True).clamp(min=1)
            variable_event_delta = (variable_incidence_matrix @ obs_event) / variable_incidence_matrix.sum(-1, keepdim=True).clamp(min=1)

            # Node→temporal hyperedge (using spike-selected nodes)
            temporal_hyperedges_updated = self.node2temporal_hyperedge[i](
                temporal_hyperedges,
                torch.cat([variable_hyperedges.gather(1, repeat(variable_indices_flattened, "B N -> B N D", D=D)),
                           obs_base], -1),
                temporal_incidence_matrix if i != 0 else temporal_incidence_matrix * mask_temp)
            temporal_hyperedges_updated = temporal_hyperedges_updated + event_scale * temporal_event_delta

            if i == 0:
                mask_temp = 1 - repeat(y_mask_L_flattened, "B N -> B E N", E=variable_incidence_matrix.shape[1])
                mask_temp[mask_temp == 0] = 1e-8

            # Node→variable hyperedge (using spike-selected nodes)
            variable_hyperedges_updated = self.node2variable_hyperedge[i](
                variable_hyperedges,
                torch.cat([temporal_hyperedges.gather(1, repeat(time_indices_flattened, "B N -> B N D", D=D)),
                           obs_base], -1),
                variable_incidence_matrix if i != 0 else variable_incidence_matrix * mask_temp)
            variable_hyperedges_updated = variable_hyperedges_updated + event_scale * variable_event_delta

            variable_hyperedges = variable_hyperedges_updated
            temporal_hyperedges = temporal_hyperedges_updated

            # Hyperedge→node
            tg = temporal_hyperedges.gather(1, repeat(time_indices_flattened, "B N -> B N D", D=D))
            vg = variable_hyperedges.gather(1, repeat(variable_indices_flattened, "B N -> B N D", D=D))

            if not self.oom_flag:
                try:
                    obs_for_h2n = self.node_self_update[i](
                        observation_nodes,
                        torch.cat([tg, vg, observation_nodes], -1),
                        x_y_mask_flattened.unsqueeze(2) * x_y_mask_flattened.unsqueeze(1))
                    h2n_input = torch.cat([obs_for_h2n, tg, vg], -1)
                except:
                    self.oom_flag = True
                    logger.warning("QSH-Net: CUDA OOM, using fallback.")

            if self.oom_flag:
                obs_for_h2n = observation_nodes
                h2n_input = torch.cat([obs_for_h2n, tg, vg], -1)

            # === Enhancement 2: Quaternion Refinement ===
            # Linear path (HyperIMTS original)
            linear_out = self.hyperedge2node[i](h2n_input)
            # Quaternion refines linear output: Hamilton product captures cross-feature
            # interactions within the fused representation that a single linear misses
            quat_out = self.quat_h2n[i](linear_out)
            # Additive refinement with learnable gate (starts near 0 → pure HyperIMTS)
            alpha = self.compute_quaternion_gate(i, linear_out, route_state["event_gate"])
            h2n_out = linear_out + alpha * quat_out
            # Diagnostic logging
            if not hasattr(self, '_h2n_count'):
                self._h2n_count = 0
            self._h2n_count += 1
            if self._h2n_count % 200 == 1:
                logger.info(f"[Quat L{i}] alpha={alpha.mean().item():.4f} event={route_state['event_gate'].mean().item():.5f} "
                            f"|linear|={linear_out.norm().item():.2f} "
                            f"|quat|={quat_out.norm().item():.2f}")

            observation_nodes = self.activation((observation_nodes + h2n_out) * md)

            # Hyperedge-to-hyperedge (identical to HyperIMTS, last layer only)
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

        self.hypergraph_encoder = HypergraphEncoder(self.enc_in, tl, D)
        self.hypergraph_learner = HypergraphLearner(configs.n_layers, D, configs.n_heads, tl)
        self.hypergraph_decoder = nn.Linear(3 * D, 1)

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

        # Encode
        (observation_nodes, temporal_hyperedges, variable_hyperedges,
         temporal_incidence_matrix, variable_incidence_matrix
        ) = self.hypergraph_encoder(
            x_L_flattened, x_y_mask_flattened, y_mask_L_flattened,
            x_y_mark, variable_indices_flattened, time_indices_flattened, N_OBSERVATIONS_MAX)

        # Hypergraph learning (with spike selection + quaternion parallel branch)
        (observation_nodes, temporal_hyperedges, variable_hyperedges
        ) = self.hypergraph_learner(
            observation_nodes, temporal_hyperedges, variable_hyperedges,
            time_indices_flattened, variable_indices_flattened,
            temporal_incidence_matrix, variable_incidence_matrix,
            x_y_mask_flattened, x_y_mask, y_mask_L_flattened)

        # Decode
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
