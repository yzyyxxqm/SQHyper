# SQHyper: Spike-Quaternion Hypergraph Network for IMTS Forecasting
#
# Two tightly-coupled mechanisms address structural shortcomings of HyperIMTS:
#
# 1. Spike-Gated Incidence (SGI):
#    Computes per-node membrane potential from context deviation.
#    Produces gate g_n ∈ (0,1] that modulates K/V in n2h attention,
#    making high-information observations contribute more strongly.
#    Also extracts event features as quaternion K-component.
#
# 2. Quaternion Multi-Source Fusion (QMF):
#    Replaces Linear(3D,D) h2n with semantically-aligned quaternion transform.
#    Four sources → four quaternion components (R=self, I=temporal, J=variable, K=event).
#    Hamilton product naturally captures all cross-source interactions.
#
# Ablation design:
#   no_sgi: g_n=1, e_n=0 → uniform weighting, zero event features
#   no_qmf: replace QuatLinear with standard Linear → flat fusion

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat, rearrange
from torch import Tensor
from utils.ExpConfigs import ExpConfigs
from utils.globals import logger


# ============================================================================
# Quaternion Linear (Hamilton product structured transform)
# ============================================================================

class QuaternionLinear(nn.Module):
    """Quaternion-algebra linear layer with structured cross-group interactions."""
    def __init__(self, in_f, out_f, bias=True):
        super().__init__()
        assert in_f % 4 == 0 and out_f % 4 == 0, \
            f"QuaternionLinear requires in_f and out_f divisible by 4, got {in_f}, {out_f}"
        qi, qo = in_f // 4, out_f // 4
        self.r = nn.Parameter(torch.empty(qo, qi))
        self.i = nn.Parameter(torch.empty(qo, qi))
        self.j = nn.Parameter(torch.empty(qo, qi))
        self.k = nn.Parameter(torch.empty(qo, qi))
        self.bias = nn.Parameter(torch.zeros(out_f)) if bias else None

    def init_identity(self):
        """Initialize as identity (only works for square qi==qo)."""
        nn.init.eye_(self.r)
        nn.init.zeros_(self.i)
        nn.init.zeros_(self.j)
        nn.init.zeros_(self.k)

    def init_xavier(self):
        """Initialize with Xavier uniform for non-square layers."""
        for p in [self.r, self.i, self.j, self.k]:
            nn.init.xavier_uniform_(p)

    def forward(self, x):
        r, i, j, k = self.r, self.i, self.j, self.k
        # Hamilton product block matrix: W ∈ R^{4qo × 4qi}
        W = torch.cat([
            torch.cat([r, -i, -j, -k], 1),
            torch.cat([i,  r, -k,  j], 1),
            torch.cat([j,  k,  r, -i], 1),
            torch.cat([k, -j,  i,  r], 1),
        ], 0)
        return F.linear(x, W, self.bias)


# ============================================================================
# Spike-Gated Incidence (SGI)
# ============================================================================

class SpikeGatedIncidence(nn.Module):
    """
    Computes per-node spike gate g_n and event features e_n.

    g_n modulates the node's contribution as key/value in n2h attention.
    e_n provides the quaternion K-component (event signal) for QMF.

    Initialization:
      membrane_proj.bias = +3 → g_n ≈ σ(3) ≈ 0.95 (near-uniform)
      event_proj weights = 0 → e_n ≈ 0 (no event signal initially)
    """
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        # Membrane potential: how event-like is this observation?
        self.membrane_proj = nn.Linear(d_model * 2, 1)
        nn.init.zeros_(self.membrane_proj.weight)
        nn.init.constant_(self.membrane_proj.bias, 3.0)  # g_n ≈ 0.95 at init
        # Event feature extraction → quaternion K component (D/4 dim)
        self.event_proj = nn.Linear(d_model * 2, d_model // 4)
        nn.init.zeros_(self.event_proj.weight)
        nn.init.zeros_(self.event_proj.bias)

    def forward(self, obs, mask_flat, variable_incidence_matrix, variable_indices_flattened):
        """
        Args:
            obs: (B, N, D) observation node embeddings
            mask_flat: (B, N) observation mask
            variable_incidence_matrix: (B, E, N)
            variable_indices_flattened: (B, N) variable index per obs
        Returns:
            g_n: (B, N) spike gate in (0, 1]
            e_n: (B, N, D/4) event features (gated by g_n)
        """
        D = obs.shape[-1]
        # Per-variable context via hypergraph structure
        var_count = variable_incidence_matrix.sum(-1, keepdim=True).clamp(min=1)  # (B, E, 1)
        var_context = (variable_incidence_matrix @ obs) / var_count  # (B, E, D)
        obs_var_ctx = var_context.gather(
            1, repeat(variable_indices_flattened, "B N -> B N D", D=D))  # (B, N, D)
        # Deviation from variable context
        deviation = obs - obs_var_ctx  # (B, N, D)
        # Membrane potential → spike gate
        membrane_input = torch.cat([obs, deviation], dim=-1)  # (B, N, 2D)
        membrane = self.membrane_proj(membrane_input).squeeze(-1)  # (B, N)
        g_n = torch.sigmoid(membrane) * mask_flat  # (B, N), masked
        # Event features → quaternion K component
        e_n = self.event_proj(membrane_input)  # (B, N, D/4)
        e_n = e_n * g_n.unsqueeze(-1)  # gate event features by spike gate
        e_n = e_n * mask_flat.unsqueeze(-1)
        return g_n, e_n


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
# HypergraphLearner (HyperIMTS core + SGI + QMF)
# ============================================================================

class HypergraphLearner(nn.Module):
    def __init__(self, n_layers, d_model, n_heads, time_length, no_sgi=False, no_qmf=False):
        super().__init__()
        self.n_layers = n_layers
        self.d_model = d_model
        self.activation = nn.ReLU()
        self.no_sgi = bool(no_sgi)
        self.no_qmf = bool(no_qmf)
        D = d_model
        Q = D // 4  # quaternion component dimension

        # === HyperIMTS core: n2h attention ===
        self.node2temporal_hyperedge = nn.ModuleList([
            MultiHeadAttentionBlock(D, 2 * D, 2 * D, D, n_heads)
            for _ in range(n_layers)])
        self.node2variable_hyperedge = nn.ModuleList([
            MultiHeadAttentionBlock(D, 2 * D, 2 * D, D, n_heads)
            for _ in range(n_layers)])
        # === HyperIMTS core: self-attention for node update ===
        self.node_self_update = nn.ModuleList([
            MultiHeadAttentionBlock(D, 3 * D, 3 * D, D, n_heads)
            for _ in range(n_layers)])
        # === HyperIMTS core: h2h (last layer only) ===
        self.variable_hyperedge2variable_hyperedge = IrregularityAwareAttention(D)
        self.hyperedge2hyperedge_layers = [n_layers - 1]
        self.scale = 1 / time_length
        self.oom_flag = False

        # === SGI: Spike-Gated Incidence (per layer) ===
        self.sgi = nn.ModuleList([
            SpikeGatedIncidence(D) for _ in range(n_layers)])

        # === QMF: Quaternion Multi-Source Fusion (per layer) ===
        # Project each source from D to D/4 (quaternion component)
        self.proj_R = nn.ModuleList([nn.Linear(D, Q) for _ in range(n_layers)])  # self → real
        self.proj_I = nn.ModuleList([nn.Linear(D, Q) for _ in range(n_layers)])  # temporal → i
        self.proj_J = nn.ModuleList([nn.Linear(D, Q) for _ in range(n_layers)])  # variable → j
        # K component comes from SGI event features (already Q dim)
        # QuaternionLinear: D → D with identity init
        self.quat_h2n = nn.ModuleList()
        for _ in range(n_layers):
            ql = QuaternionLinear(D, D)
            ql.init_identity()
            self.quat_h2n.append(ql)
        # Fallback linear for no_qmf ablation
        self.linear_h2n = nn.ModuleList([nn.Linear(D, D) for _ in range(n_layers)])

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
        Q = D // 4

        for i in range(self.n_layers):
            if i == 0:
                mask_temp = 1 - repeat(y_mask_L_flattened, "B N -> B L N", L=temporal_incidence_matrix.shape[1])
                mask_temp[mask_temp == 0] = 1e-8

            # ============================================================
            # Step 1: SGI — Spike-Gated Incidence
            # ============================================================
            if not self.no_sgi:
                g_n, e_n = self.sgi[i](
                    observation_nodes, x_y_mask_flattened,
                    variable_incidence_matrix, variable_indices_flattened)
            else:
                # Ablation: uniform gate, zero event features
                g_n = x_y_mask_flattened  # 1 for observed, 0 for padding
                e_n = torch.zeros(
                    observation_nodes.shape[0], observation_nodes.shape[1], Q,
                    device=observation_nodes.device, dtype=observation_nodes.dtype)

            # Spike-gated observation for K/V in n2h attention
            obs_gated = observation_nodes * g_n.unsqueeze(-1)

            # ============================================================
            # Step 2: Node → Hyperedge (with spike-gated K/V)
            # ============================================================
            # Node → temporal hyperedge
            temporal_hyperedges = self.node2temporal_hyperedge[i](
                temporal_hyperedges,
                torch.cat([
                    variable_hyperedges.gather(1, repeat(variable_indices_flattened, "B N -> B N D", D=D)),
                    obs_gated
                ], -1),
                temporal_incidence_matrix if i != 0 else temporal_incidence_matrix * mask_temp)

            if i == 0:
                mask_temp = 1 - repeat(y_mask_L_flattened, "B N -> B E N", E=variable_incidence_matrix.shape[1])
                mask_temp[mask_temp == 0] = 1e-8

            # Node → variable hyperedge
            variable_hyperedges = self.node2variable_hyperedge[i](
                variable_hyperedges,
                torch.cat([
                    temporal_hyperedges.gather(1, repeat(time_indices_flattened, "B N -> B N D", D=D)),
                    obs_gated
                ], -1),
                variable_incidence_matrix if i != 0 else variable_incidence_matrix * mask_temp)

            # ============================================================
            # Step 3: Hyperedge → Node (QMF or fallback)
            # ============================================================
            tg = temporal_hyperedges.gather(1, repeat(time_indices_flattened, "B N -> B N D", D=D))
            vg = variable_hyperedges.gather(1, repeat(variable_indices_flattened, "B N -> B N D", D=D))

            if not self.oom_flag:
                try:
                    obs_for_h2n = self.node_self_update[i](
                        observation_nodes,
                        torch.cat([tg, vg, observation_nodes], -1),
                        x_y_mask_flattened.unsqueeze(2) * x_y_mask_flattened.unsqueeze(1))
                except:
                    self.oom_flag = True
                    logger.warning("SQHyper: CUDA OOM in self-attention, using fallback.")

            if self.oom_flag:
                obs_for_h2n = observation_nodes

            # Quaternion Multi-Source Fusion
            q_R = self.proj_R[i](obs_for_h2n)  # D → D/4: self (real part)
            q_I = self.proj_I[i](tg)            # D → D/4: temporal (i part)
            q_J = self.proj_J[i](vg)            # D → D/4: variable (j part)
            q_K = e_n                            # D/4: event (k part, from SGI)

            q = torch.cat([q_R, q_I, q_J, q_K], dim=-1)  # (B, N, D)

            if not self.no_qmf:
                h2n_out = self.quat_h2n[i](q)
            else:
                # Ablation: flat linear fusion instead of quaternion
                h2n_out = self.linear_h2n[i](q)

            md = repeat(x_y_mask_flattened, "B N -> B N D", D=D)
            observation_nodes = self.activation((observation_nodes + h2n_out) * md)

            # ============================================================
            # Step 4: Hyperedge → Hyperedge (last layer only, identical to HyperIMTS)
            # ============================================================
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
            logger.warning(f"SQHyper: d_model {configs.d_model}->{self.d_model} (must be ×4)")
        D = self.d_model
        sl = configs.seq_len_max_irr or configs.seq_len
        pl = configs.pred_len_max_irr or configs.pred_len
        tl = sl + pl

        self.hypergraph_encoder = HypergraphEncoder(self.enc_in, tl, D)
        self.hypergraph_learner = HypergraphLearner(
            configs.n_layers, D, configs.n_heads, tl,
            no_sgi=getattr(configs, "sqhyper_no_sgi", 0),
            no_qmf=getattr(configs, "sqhyper_no_qmf", 0),
        )
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

        # Hypergraph learning (SGI + QMF)
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
