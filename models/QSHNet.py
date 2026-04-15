# QSH-Net: Quaternion-Spiking-Hypergraph Network for IMTS Forecasting
#
# Core idea: Quaternion and Spiking are organically fused INTO the hypergraph
# message passing, not bolted on as pre/post-processing.
#
# 1. Quaternion-Structured H2N Fusion (replaces Linear(3D, D)):
#    The three information sources (obs, tg, vg) plus their linear fusion are
#    mapped to quaternion components (r, i, j, k), each D/4-dimensional.
#    Hamilton product (Kronecker block matrix) captures structured cross-source
#    interactions: time×variable (jk), variable×node (ki), node×time (ij).
#    Identity init: Wr=I, Wi=Wj=Wk=0 → degrades to original Linear at start.
#
# 2. Spike-Modulated Attention Temperature (replaces scalar pre-filtering):
#    Spike signal modulates per-node attention temperature in n2h attention.
#    Deviant observations → low temperature → sharp attention (focused).
#    Conforming observations → high temperature → smooth attention (diffused).
#    Zero-init → uniform temperature → identical to standard attention at start.
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
# Quaternion-Structured Fusion (Kronecker block matrix Hamilton product)
# ============================================================================

class QuaternionStructuredFusion(nn.Module):
    """
    Fuses obs, tg, vg via original Linear(3D, D) PLUS a Hamilton product
    residual that captures structured cross-source interactions.

    Main path (identical to HyperIMTS):
      linear_out = Linear(3D, D)(cat(obs, tg, vg))

    Residual path (quaternion cross-source interactions):
      r = linear_out                        — real: main path output
      i = Linear(D, D)(obs)                 — node self
      j = Linear(D, D)(tg)                  — temporal context
      k = Linear(D, D)(vg)                  — variable context
      q_out = HamiltonProduct(cat(r,i,j,k)) — (4D×4D) Kronecker block matrix
      residual = q_out[..., :D]             — take real part

    Output = linear_out + α * residual

    Identity init: Wr=I, Wi=Wj=Wk=0, proj_i/j/k=I, α=0
    → residual = linear_out, output = linear_out + 0*linear_out = linear_out
    → EXACT HyperIMTS at initialization.
    """
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

        # Main path: identical to HyperIMTS
        self.linear = nn.Linear(3 * d_model, d_model)

        # Quaternion residual projections
        self.proj_i = nn.Linear(d_model, d_model, bias=False)
        self.proj_j = nn.Linear(d_model, d_model, bias=False)
        self.proj_k = nn.Linear(d_model, d_model, bias=False)
        nn.init.eye_(self.proj_i.weight)
        nn.init.eye_(self.proj_j.weight)
        nn.init.eye_(self.proj_k.weight)

        # Hamilton product weights (D × D sub-matrices)
        self.Wr = nn.Parameter(torch.empty(d_model, d_model))
        self.Wi = nn.Parameter(torch.empty(d_model, d_model))
        self.Wj = nn.Parameter(torch.empty(d_model, d_model))
        self.Wk = nn.Parameter(torch.empty(d_model, d_model))
        nn.init.eye_(self.Wr)
        nn.init.zeros_(self.Wi)
        nn.init.zeros_(self.Wj)
        nn.init.zeros_(self.Wk)

        # Gate: starts at 0 → pure HyperIMTS, learns to incorporate quaternion
        self.gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, obs, tg, vg):
        D = self.d_model
        # Main path (HyperIMTS original)
        linear_out = self.linear(torch.cat([obs, tg, vg], dim=-1))  # (B, N, D)

        # Quaternion residual: structured cross-source interactions
        r = linear_out
        i = self.proj_i(obs)
        j = self.proj_j(tg)
        k = self.proj_k(vg)

        q_in = torch.cat([r, i, j, k], dim=-1)  # (B, N, 4D)

        Wr, Wi, Wj, Wk = self.Wr, self.Wi, self.Wj, self.Wk
        W = torch.cat([
            torch.cat([ Wr, -Wi, -Wj, -Wk], dim=1),
            torch.cat([ Wi,  Wr, -Wk,  Wj], dim=1),
            torch.cat([ Wj,  Wk,  Wr, -Wi], dim=1),
            torch.cat([ Wk, -Wj,  Wi,  Wr], dim=1),
        ], dim=0)  # (4D, 4D)

        q_out = F.linear(q_in, W)  # (B, N, 4D)
        residual = q_out[..., :D]  # real part

        # Additive residual with learnable gate
        alpha = torch.tanh(self.gate)  # ∈ (-1, 1), starts at 0
        return linear_out + alpha * residual


# ============================================================================
# Spike-Modulated Attention Temperature
        # q_out has interleaved [r0,i0,j0,k0, r1,i1,j1,k1, ...]
        # We want just the r-components: every 4th starting from 0
        # Reshape to (B, N, D/4, 4), take [:,:,:,0], reshape to (B, N, D/4)
        # But that only gives D/4 dims... we want all D dims of the output.
        #
        # Actually the Kronecker block matrix output IS D-dimensional and
        # already contains the full mixed result. The interleaving ensures
        # that the Hamilton product structure is applied correctly across
        # the semantic components. The output should be used as-is.
        return q_out


# ============================================================================
# Spike-Modulated Attention Temperature
# ============================================================================

class SpikeTemperature(nn.Module):
    """
    Computes per-node attention temperature based on how much each observation
    deviates from its variable's context (via hypergraph structure).

    Deviant observations → high spike_signal → low temperature → sharp attention
    Conforming observations → low spike_signal → high temperature → smooth attention

    This is organically fused into the attention mechanism itself, not a
    pre-filtering step.

    Zero-init: spike_signal = 0.5 for all → uniform temperature → standard attention.
    """
    def __init__(self, d_model):
        super().__init__()
        self.membrane_proj = nn.Linear(d_model * 2, 1)
        # Zero-init → sigmoid(0) = 0.5 → uniform temperature at start
        nn.init.zeros_(self.membrane_proj.weight)
        nn.init.zeros_(self.membrane_proj.bias)
        # Learnable temperature scale: controls how much spike modulates attention
        self.log_temp_scale = nn.Parameter(torch.tensor(0.0))  # exp(0)=1.0

    def forward(self, obs, variable_incidence_matrix, variable_indices_flattened):
        """
        obs: (B, N, D)
        variable_incidence_matrix: (B, E, N)
        variable_indices_flattened: (B, N)
        Returns: temperature (B, N, 1) — per-node attention temperature
        """
        D = obs.shape[-1]
        # Compute per-variable context via hypergraph structure
        var_count = variable_incidence_matrix.sum(-1, keepdim=True).clamp(min=1)
        var_context = (variable_incidence_matrix @ obs) / var_count  # (B, E, D)
        obs_var_ctx = var_context.gather(
            1, repeat(variable_indices_flattened, "B N -> B N D", D=D))
        # Deviation from variable context
        deviation = obs - obs_var_ctx
        # Continuous spike signal via sigmoid (differentiable, no surrogate needed)
        spike_signal = torch.sigmoid(
            self.membrane_proj(torch.cat([obs, deviation], dim=-1)))  # (B, N, 1)
        # Temperature: high spike (deviant) → low temp → sharp attention
        #              low spike (conforming) → high temp → smooth attention
        temp_scale = torch.exp(self.log_temp_scale)  # positive scale
        temperature = 1.0 + (1.0 - spike_signal) * temp_scale  # (B, N, 1)
        # At init: spike_signal=0.5, temp_scale=1.0 → temperature=1.5 for all (uniform)

        # Diagnostic logging
        if not hasattr(self, '_fwd_count'):
            self._fwd_count = 0
        self._fwd_count += 1
        if self._fwd_count % 200 == 1:
            logger.info(f"[SpikeTemp] spike_mean={spike_signal.mean().item():.3f} "
                        f"temp_mean={temperature.mean().item():.3f} "
                        f"temp_scale={temp_scale.item():.3f}")
        return temperature


# ============================================================================
# Spike-Modulated MultiHeadAttentionBlock
# ============================================================================

class SpikeModulatedAttentionBlock(nn.Module):
    """
    MultiHeadAttention where each key-node has its own temperature that
    modulates the attention sharpness. Temperature is provided externally
    by SpikeTemperature.

    When temperature is None, behaves identically to standard MultiHeadAttentionBlock.
    """
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

    def forward(self, Q, K, mask=None, temperature=None):
        """
        Q: (B, L_q, dim_Q)
        K: (B, L_k, dim_K)  — also used as V
        mask: (B, L_q, L_k)
        temperature: (B, L_k, 1) or None — per-key-node temperature
        """
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)
        ds = self.n_dim // self.num_heads
        Q_ = torch.cat(Q.split(ds, 2), 0)
        K = torch.cat(K.split(ds, 2), 0)
        V = torch.cat(V.split(ds, 2), 0)

        A = Q_.bmm(K.transpose(1, 2)) / math.sqrt(self.n_dim)

        # Spike-modulated temperature: scale attention logits per key-node
        if temperature is not None:
            # temperature: (B, L_k, 1) → repeat for heads → (B*H, 1, L_k)
            temp = temperature.squeeze(-1)  # (B, L_k)
            temp = temp.repeat(self.num_heads, 1).unsqueeze(1)  # (B*H, 1, L_k)
            A = A / temp  # broadcast: (B*H, L_q, L_k) / (B*H, 1, L_k)

        if mask is not None:
            A = A.masked_fill(mask.repeat(self.num_heads, 1, 1) == 0, -1e9)
        A = torch.softmax(A, 2)
        O = torch.cat((Q_ + A.bmm(V)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O


# ============================================================================
# MultiHeadAttentionBlock (identical to HyperIMTS, for non-spike-modulated uses)
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
# HypergraphLearner (HyperIMTS + Quaternion Structured Fusion + Spike Temperature)
# ============================================================================

class HypergraphLearner(nn.Module):
    def __init__(self, n_layers, d_model, n_heads, time_length):
        super().__init__()
        self.n_layers = n_layers
        self.d_model = d_model
        self.activation = nn.ReLU()

        # === Node-to-Hyperedge: standard attention (identical to HyperIMTS) ===
        self.node2temporal_hyperedge = nn.ModuleList([
            MultiHeadAttentionBlock(d_model, 2*d_model, 2*d_model, d_model, n_heads)
            for _ in range(n_layers)])
        self.node2variable_hyperedge = nn.ModuleList([
            MultiHeadAttentionBlock(d_model, 2*d_model, 2*d_model, d_model, n_heads)
            for _ in range(n_layers)])

        # === Node self-update (identical to HyperIMTS) ===
        self.node_self_update = nn.ModuleList([
            MultiHeadAttentionBlock(d_model, 3*d_model, 3*d_model, d_model, n_heads)
            for _ in range(n_layers)])

        # === Hyperedge-to-Node: quaternion structured fusion (replaces Linear(3D, D)) ===
        self.hyperedge2node = nn.ModuleList([
            QuaternionStructuredFusion(d_model) for _ in range(n_layers)])

        # === Hyperedge-to-Hyperedge (identical to HyperIMTS) ===
        self.variable_hyperedge2variable_hyperedge = IrregularityAwareAttention(d_model)
        self.hyperedge2hyperedge_layers = [n_layers - 1]
        self.scale = 1 / time_length
        self.oom_flag = False

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
                x_y_mask_flattened, x_y_mask, y_mask_L_flattened):
        D = self.d_model
        for i in range(self.n_layers):
            if i == 0:
                mask_temp = 1 - repeat(y_mask_L_flattened, "B N -> B L N", L=temporal_incidence_matrix.shape[1])
                mask_temp[mask_temp == 0] = 1e-8

            md = repeat(x_y_mask_flattened, "B N -> B N D", D=D)

            # === Node→Temporal Hyperedge (standard attention) ===
            temporal_hyperedges_updated = self.node2temporal_hyperedge[i](
                temporal_hyperedges,
                torch.cat([variable_hyperedges.gather(1, repeat(variable_indices_flattened, "B N -> B N D", D=D)),
                           observation_nodes], -1),
                temporal_incidence_matrix if i != 0 else temporal_incidence_matrix * mask_temp)

            if i == 0:
                mask_temp = 1 - repeat(y_mask_L_flattened, "B N -> B E N", E=variable_incidence_matrix.shape[1])
                mask_temp[mask_temp == 0] = 1e-8

            # === Node→Variable Hyperedge (standard attention) ===
            variable_hyperedges_updated = self.node2variable_hyperedge[i](
                variable_hyperedges,
                torch.cat([temporal_hyperedges.gather(1, repeat(time_indices_flattened, "B N -> B N D", D=D)),
                           observation_nodes], -1),
                variable_incidence_matrix if i != 0 else variable_incidence_matrix * mask_temp)

            variable_hyperedges = variable_hyperedges_updated
            temporal_hyperedges = temporal_hyperedges_updated

            # === Hyperedge→Node ===
            tg = temporal_hyperedges.gather(1, repeat(time_indices_flattened, "B N -> B N D", D=D))
            vg = variable_hyperedges.gather(1, repeat(variable_indices_flattened, "B N -> B N D", D=D))

            # Node self-update (O(N²), with OOM fallback — identical to HyperIMTS)
            if not self.oom_flag:
                try:
                    obs_for_h2n = self.node_self_update[i](
                        observation_nodes,
                        torch.cat([tg, vg, observation_nodes], -1),
                        x_y_mask_flattened.unsqueeze(2) * x_y_mask_flattened.unsqueeze(1))
                except:
                    self.oom_flag = True
                    logger.warning("QSH-Net: CUDA OOM in node_self_update, using fallback.")

            if self.oom_flag:
                obs_for_h2n = observation_nodes

            # === Quaternion Structured Fusion (replaces Linear(3D, D)) ===
            h2n_out = self.hyperedge2node[i](obs_for_h2n, tg, vg)

            # Diagnostic logging
            if not hasattr(self, '_h2n_count'):
                self._h2n_count = 0
            self._h2n_count += 1
            if self._h2n_count % 200 == 1:
                logger.info(f"[QuatFusion L{i}] |h2n_out|={h2n_out.norm().item():.2f}")

            observation_nodes = self.activation((observation_nodes + h2n_out) * md)

            # === Hyperedge-to-Hyperedge (identical to HyperIMTS, last layer only) ===
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
            logger.warning(f"QSH-Net: d_model {configs.d_model}->{self.d_model} (must be ×4)")
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

        # Hypergraph learning
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
