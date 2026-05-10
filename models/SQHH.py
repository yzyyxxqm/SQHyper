# SQHH v2: Spike-Quaternion Hypergraph (SQHyper backbone + SRI + SQC)
#
# Design history:
#   - SQHH v0 (paradigm reset: L0 primary / L1 spike-event / L2 anchor
#     hypergraph layers) was architected, implemented and deployed on 4
#     IMTS datasets. Empirical result: 12x regression on HumanActivity
#     vs SQHyper baseline (0.016 -> 0.20+) because the decoder lost the
#     hyperedge-gather mechanism. Archived in models/SQHH_v0_archive.py.
#   - SQHH v1 (scatter-mean time/var summaries at decode) partially fixed
#     P12 and MIMIC_III but HumanActivity remained ~13x above baseline,
#     showing static aggregation can't match attention-based learnable
#     hyperedges on long sequences.
#   - SQHH v2 (this file) accepts the evidence: keep SQHyper's learnable
#     hyperedge backbone and inject two focused, testable SQHH
#     contributions on top.
#
# Two contributions over SQHyper:
#
#   1. SRI (Spike-Refractory Incidence): SGI detects spikes purely via
#      membrane potential (context deviation). SRI adds a per-variable
#      refractory time-constant tau_v: after a spike at time t_m on
#      variable v, subsequent spikes on v within tau_v are inhibited by
#      (1 - tanh(alpha_v * sum_m spike_m * exp(-(t_n - t_m)/tau_v))).
#      Interpretation: physical sensors / biological events have
#      characteristic cooldown windows; bursty re-spikes are usually noise
#      or duplication of the same underlying event.
#
#   2. SQC (Spike-Quaternion Coupling): before each layer's node2hyperedge
#      attention, apply a spike-driven quaternion rotation to the cell
#      embeddings. Rotation angle theta_n = theta_max * spike[n] around
#      the K-axis. SGI gives spike scalar gating (magnitude), SQC gives
#      spike orthogonal geometric action (direction).
#
# Ablations:
#   --sqhh_no_sri    : use SQHyper's SGI instead of SRI (no refractory)
#   --sqhh_no_sqc    : skip quaternion rotation (back to scalar gating)
#   --sqhh_no_qmf    : flat linear fusion (inherited from SQHyper)
#
# Note on parameter names: we keep the historical `sqhh_*` flag prefix so
# existing scripts under scripts/SQHH/*.sh continue to work. The flags
# sqhh_k_a, sqhh_k_e, sqhh_no_layer0/1/2 from SQHH v0 are accepted but
# ignored (no L0/L1/L2 layers in v2).

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat, rearrange
from torch import Tensor
from utils.ExpConfigs import ExpConfigs
from utils.globals import logger


# ============================================================================
# Quaternion Linear (identical to SQHyper)
# ============================================================================

class QuaternionLinear(nn.Module):
    """Quaternion-algebra linear layer with structured cross-group interactions."""

    def __init__(self, in_f, out_f, bias=True):
        super().__init__()
        assert in_f % 4 == 0 and out_f % 4 == 0, (
            f"QuaternionLinear requires in_f and out_f divisible by 4, "
            f"got {in_f}, {out_f}"
        )
        qi, qo = in_f // 4, out_f // 4
        self.r = nn.Parameter(torch.empty(qo, qi))
        self.i = nn.Parameter(torch.empty(qo, qi))
        self.j = nn.Parameter(torch.empty(qo, qi))
        self.k = nn.Parameter(torch.empty(qo, qi))
        self.bias = nn.Parameter(torch.zeros(out_f)) if bias else None

    def init_identity(self):
        nn.init.eye_(self.r)
        nn.init.zeros_(self.i)
        nn.init.zeros_(self.j)
        nn.init.zeros_(self.k)

    def init_xavier(self):
        for p in [self.r, self.i, self.j, self.k]:
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


# ============================================================================
# Quaternion utilities (for SQC rotation)
# ============================================================================

def quaternion_split(q: Tensor):
    Q = q.shape[-1] // 4
    return q[..., :Q], q[..., Q:2 * Q], q[..., 2 * Q:3 * Q], q[..., 3 * Q:]


def quaternion_concat(qr, qi, qj, qk):
    return torch.cat([qr, qi, qj, qk], dim=-1)


def hamilton_product(p: Tensor, q: Tensor) -> Tensor:
    pr, pi, pj, pk = quaternion_split(p)
    qr, qi, qj, qk = quaternion_split(q)
    # Hamilton product (p * q)
    r = pr * qr - pi * qi - pj * qj - pk * qk
    i = pr * qi + pi * qr + pj * qk - pk * qj
    j = pr * qj - pi * qk + pj * qr + pk * qi
    k = pr * qk + pi * qj - pj * qi + pk * qr
    return quaternion_concat(r, i, j, k)


# ============================================================================
# Spike-Refractory Incidence (SRI) — SGI + per-variable refractory dynamics
# ============================================================================

class SpikeRefractoryIncidence(nn.Module):
    """Drop-in replacement for SGI with per-variable refractory inhibition.

    Forward pass mirrors SGI:
        g_n: (B, N) gate in (0, 1], masked
        e_n: (B, N, D/4) event features, gated by g_n

    Refractory extension: given detected raw spike s_n in (0, 1], for each
    cell n we compute an inhibition from all earlier cells of the same
    variable v_n:
        inh_n = sum_{m: v_m == v_n, t_m < t_n} s_m * exp(-(t_n - t_m) / tau[v_n])
        g_refr_n = 1 - tanh(alpha[v_n] * inh_n) in (0, 1]
        g_n = g_raw_n * g_refr_n

    Interpretation: tau[v_n] is the characteristic cooldown of variable v_n;
    alpha[v_n] is the strength of inhibition. Both are learned (per-variable).

    Initialization is tuned so that at init:
        membrane_proj.bias = +3  ->  g_raw approx sigmoid(3) ~= 0.95
        event_proj weights = 0   ->  e_n approx 0
        tau_init = 0.1           ->  1/e decay over 10% of normalized time
        alpha_init = 0.0         ->  inhibition off at init (g_refr = 1)
    This means SRI degrades smoothly to SGI at init; refractory activates
    only as alpha learns to grow.
    """

    def __init__(self, d_model: int, n_vars: int,
                 tau_init: float = 0.1, alpha_init: float = 0.0,
                 no_refractory: bool = False):
        super().__init__()
        self.d_model = d_model
        self.n_vars = n_vars
        self.no_refractory = bool(no_refractory)

        # Identical to SGI
        self.membrane_proj = nn.Linear(d_model * 2, 1)
        nn.init.zeros_(self.membrane_proj.weight)
        nn.init.constant_(self.membrane_proj.bias, 3.0)
        self.event_proj = nn.Linear(d_model * 2, d_model // 4)
        nn.init.zeros_(self.event_proj.weight)
        nn.init.zeros_(self.event_proj.bias)

        # Refractory parameters per variable (stored as log for positivity).
        # alpha starts at log(very small) so tanh(alpha * inh) ~= 0 at init,
        # i.e., SRI initially behaves as SGI. Model can grow alpha if needed.
        self.log_tau = nn.Parameter(
            torch.full((n_vars,), math.log(max(tau_init, 1e-4))))
        # alpha_init is used as the multiplier AT init. If alpha_init=0 we
        # actually want log_alpha to give exp(log_alpha) = 0 which is
        # impossible, so we use a very small positive value.
        self.log_alpha = nn.Parameter(
            torch.full((n_vars,),
                       math.log(max(alpha_init, 1e-4)) if alpha_init > 0
                       else math.log(1e-4)))

    def forward(self, obs: Tensor, mask_flat: Tensor,
                variable_incidence_matrix: Tensor,
                variable_indices_flattened: Tensor,
                time_norm: Tensor):
        """
        Args:
            obs: (B, N, D) observation node embeddings
            mask_flat: (B, N) observation mask (1 observed, 0 padding)
            variable_incidence_matrix: (B, E, N)
            variable_indices_flattened: (B, N) long
            time_norm: (B, N) in [0, 1]
        Returns:
            g_n: (B, N) spike gate in (0, 1], masked
            e_n: (B, N, D/4) event features (gated by g_n), masked
        """
        D = obs.shape[-1]
        # Per-variable context via hypergraph structure (same as SGI)
        var_count = variable_incidence_matrix.sum(-1, keepdim=True).clamp(min=1)
        var_context = (variable_incidence_matrix @ obs) / var_count  # (B, E, D)
        obs_var_ctx = var_context.gather(
            1, repeat(variable_indices_flattened, "B N -> B N D", D=D))
        deviation = obs - obs_var_ctx
        membrane_input = torch.cat([obs, deviation], dim=-1)
        membrane = self.membrane_proj(membrane_input).squeeze(-1)
        g_raw = torch.sigmoid(membrane) * mask_flat  # (B, N)
        e_n = self.event_proj(membrane_input) * g_raw.unsqueeze(-1)
        e_n = e_n * mask_flat.unsqueeze(-1)

        if self.no_refractory:
            return g_raw, e_n

        # ----- Refractory inhibition (core SRI contribution) -----
        # Per-cell tau and alpha looked up by variable id.
        tau = torch.exp(self.log_tau)[variable_indices_flattened]
        alpha = torch.exp(self.log_alpha)[variable_indices_flattened]  # (B, N)

        # Pair masks: same variable, earlier time, both observed.
        same_var = (
            variable_indices_flattened.unsqueeze(2)
            == variable_indices_flattened.unsqueeze(1)
        )
        time_diff = time_norm.unsqueeze(2) - time_norm.unsqueeze(1)  # t_n - t_m
        earlier = (time_diff > 0).to(obs.dtype)  # m before n
        valid_pair = (
            mask_flat.unsqueeze(2) * mask_flat.unsqueeze(1))  # both observed
        pair_mask = same_var.float() * earlier * valid_pair  # (B, N, N)

        # Decay kernel exp(-(t_n - t_m) / tau[v_n]). tau[v_n] is per
        # receiving cell (row n).
        decay = torch.exp(
            -time_diff.clamp_min(0) / tau.unsqueeze(2).clamp_min(1e-6))
        # Sum over m: inhibition input uses raw spike s_m
        inh = (pair_mask * decay * g_raw.unsqueeze(1)).sum(dim=-1)  # (B, N)
        g_refr = 1.0 - torch.tanh(alpha * inh)  # in (0, 1]
        g_n = g_raw * g_refr  # masked by g_raw via g_raw * mask

        # Also dampen event features by refractory factor
        e_n = e_n * g_refr.unsqueeze(-1)
        return g_n, e_n


# ============================================================================
# Spike-Quaternion Coupling (SQC) rotation
# ============================================================================

class SpikeQuaternionRotation(nn.Module):
    """Rotate quaternion embedding around K-axis by angle theta_n = theta_max * spike[n].

    rot_n = (cos(theta_n/2), 0, 0, sin(theta_n/2))  (unit quaternion)
    out   = rot_n * q  (Hamilton product, per cell)

    Geometric effect: preserves norm, acts orthogonally. When spike = 0 the
    rotation is identity. theta_max is a learnable scalar initialized small
    so the init behavior is near-identity (small perturbation).
    """

    def __init__(self, theta_max_init: float = math.pi / 8):
        super().__init__()
        # Learnable theta_max, initialized at pi/8 (small rotation).
        # Stored raw (can go negative).
        self.theta_max = nn.Parameter(torch.tensor(float(theta_max_init)))

    def forward(self, q: Tensor, spike: Tensor) -> Tensor:
        """
        Args:
            q:     (B, N, D) quaternion-valued embedding, D % 4 == 0
            spike: (B, N) in [0, 1]
        Returns:
            q_rot: (B, N, D) same shape, rotated
        """
        D = q.shape[-1]
        Q = D // 4
        assert D % 4 == 0, f"SQC requires D % 4 == 0, got {D}"

        theta = (self.theta_max * spike).unsqueeze(-1)  # (B, N, 1)
        c = torch.cos(theta * 0.5)  # (B, N, 1)
        s = torch.sin(theta * 0.5)  # (B, N, 1)

        # Build rotation quaternion (cos, 0, 0, sin) broadcast to (B, N, D)
        zeros = torch.zeros_like(c).expand(-1, -1, Q)
        rot = torch.cat([
            c.expand(-1, -1, Q),
            zeros,
            zeros,
            s.expand(-1, -1, Q),
        ], dim=-1)
        return hamilton_product(rot, q)


# ============================================================================
# MultiHeadAttentionBlock (identical to SQHyper / HyperIMTS)
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
# IrregularityAwareAttention (identical to SQHyper / HyperIMTS)
# ============================================================================

class IrregularityAwareAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.scale = d_model ** 0.5
        self.threshold = nn.Parameter(torch.tensor(0.5), requires_grad=True)

    def forward(self, x, query_aux=None, key_aux=None,
                adjacency_mask=None, merge_coefficients=None):
        query = self.query_proj(x)
        key = self.key_proj(x)
        value = self.value_proj(x)
        attention_scores = torch.matmul(
            query, key.transpose(-2, -1)) / self.scale
        mask_value = torch.finfo(attention_scores.dtype).min
        if query_aux is not None and key_aux is not None:
            attention_scores_aux = torch.matmul(
                query_aux, key_aux.transpose(-2, -1)) / (
                    key_aux.shape[-1] ** 0.5)
            non_zero_mask = (attention_scores_aux != 0)
            positive_mask = (attention_scores > self.threshold)
            mask = positive_mask & non_zero_mask
            attention_scores[mask] = (
                (1 - merge_coefficients) * attention_scores
                + merge_coefficients * attention_scores_aux)[mask]
        if adjacency_mask is not None:
            attention_scores = attention_scores.masked_fill(
                adjacency_mask == 0, mask_value)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        return torch.matmul(attention_weights, value)


# ============================================================================
# HypergraphEncoder (identical to SQHyper / HyperIMTS)
# ============================================================================

class HypergraphEncoder(nn.Module):
    def __init__(self, enc_in, time_length, d_model):
        super().__init__()
        self.enc_in = enc_in
        self.d_model = d_model
        self.variable_hyperedge_weights = nn.Parameter(
            torch.randn(enc_in, d_model), requires_grad=True)
        self.relu = nn.ReLU()
        self.observation_node_encoder = nn.Linear(2, d_model)
        self.temporal_hyperedge_encoder = nn.Linear(1, d_model)

    def forward(self, x_L_flattened, x_y_mask_flattened, y_mask_L_flattened,
                x_y_mark, variable_indices_flattened,
                time_indices_flattened, N_OBSERVATIONS_MAX):
        B = x_L_flattened.shape[0]
        E, L, D = self.enc_in, x_y_mark.shape[1], self.d_model
        N = N_OBSERVATIONS_MAX

        x_L_flattened = torch.stack([
            x_L_flattened, 1 - x_y_mask_flattened + y_mask_L_flattened
        ], dim=-1)

        temporal_incidence_matrix = repeat(
            time_indices_flattened, "B N -> B L N", L=L)
        temporal_incidence_matrix = (temporal_incidence_matrix == repeat(
            torch.ones(B, L, device=x_L_flattened.device).cumsum(dim=1),
            "B L -> B L N", N=N) - 1).float()
        temporal_incidence_matrix = temporal_incidence_matrix * repeat(
            x_y_mask_flattened, "B N -> B L N", L=L)

        variable_incidence_matrix = repeat(
            torch.ones(B, E, device=x_L_flattened.device).cumsum(dim=1) - 1,
            "B E -> B E N", N=N)
        variable_incidence_matrix = (variable_incidence_matrix == repeat(
            variable_indices_flattened, "B N -> B E N", E=E)).float()
        variable_incidence_matrix = variable_incidence_matrix * repeat(
            x_y_mask_flattened, "B N -> B E N", E=E)

        observation_nodes = self.relu(
            self.observation_node_encoder(x_L_flattened)) * repeat(
            x_y_mask_flattened, "B N -> B N D", D=D)
        temporal_hyperedges = torch.sin(
            self.temporal_hyperedge_encoder(x_y_mark))
        variable_hyperedges = self.relu(repeat(
            self.variable_hyperedge_weights, "E D -> B E D", B=B))

        return (observation_nodes, temporal_hyperedges, variable_hyperedges,
                temporal_incidence_matrix, variable_incidence_matrix)


# ============================================================================
# HypergraphLearner (SQHyper backbone + SRI + SQC)
# ============================================================================

class HypergraphLearner(nn.Module):
    """SQHyper's HypergraphLearner with two SQHH substitutions:
      - SGI  -> SRI  (refractory inhibition per variable)
      - SQC  inserted before each layer's node2hyperedge attention
    """

    def __init__(self, n_layers, d_model, n_heads, time_length, n_vars,
                 no_sri=False, no_sqc=False, no_qmf=False):
        super().__init__()
        self.n_layers = n_layers
        self.d_model = d_model
        self.activation = nn.ReLU()
        self.no_sri = bool(no_sri)
        self.no_sqc = bool(no_sqc)
        self.no_qmf = bool(no_qmf)
        D = d_model
        Q = D // 4

        # === HyperIMTS/SQHyper core attention blocks ===
        self.node2temporal_hyperedge = nn.ModuleList([
            MultiHeadAttentionBlock(D, 2 * D, 2 * D, D, n_heads)
            for _ in range(n_layers)])
        self.node2variable_hyperedge = nn.ModuleList([
            MultiHeadAttentionBlock(D, 2 * D, 2 * D, D, n_heads)
            for _ in range(n_layers)])
        self.node_self_update = nn.ModuleList([
            MultiHeadAttentionBlock(D, 3 * D, 3 * D, D, n_heads)
            for _ in range(n_layers)])
        self.variable_hyperedge2variable_hyperedge = IrregularityAwareAttention(D)
        self.hyperedge2hyperedge_layers = [n_layers - 1]
        self.scale = 1 / time_length
        self.oom_flag = False

        # === SRI (replaces SGI) — per layer ===
        # If no_sri is True we still use SpikeRefractoryIncidence but disable
        # its refractory term, making it identical to SGI.
        self.sri = nn.ModuleList([
            SpikeRefractoryIncidence(D, n_vars, no_refractory=self.no_sri)
            for _ in range(n_layers)])
        # Per-layer gate scale (same residual-gating trick as SQHyper):
        # gating = mask + gate_scale * (g_n - mask). gate_scale = 0 at init.
        self.gate_scale = nn.ParameterList([
            nn.Parameter(torch.zeros(1)) for _ in range(n_layers)])

        # === SQC — one rotation per layer ===
        if not self.no_sqc:
            self.sqc = nn.ModuleList([
                SpikeQuaternionRotation() for _ in range(n_layers)])

        # === QMF (Quaternion Multi-Source Fusion) — identical to SQHyper ===
        self.proj_R = nn.ModuleList([
            nn.Linear(D, Q) for _ in range(n_layers)])
        self.proj_I = nn.ModuleList([
            nn.Linear(D, Q) for _ in range(n_layers)])
        self.proj_J = nn.ModuleList([
            nn.Linear(D, Q) for _ in range(n_layers)])
        self.quat_h2n = nn.ModuleList()
        for _ in range(n_layers):
            ql = QuaternionLinear(D, D)
            ql.init_identity()
            self.quat_h2n.append(ql)
        self.linear_h2n = nn.ModuleList([
            nn.Linear(D, D) for _ in range(n_layers)])

    def get_fine_grained_embedding(self, tensor_flattened, target_shape):
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
                time_norm):
        D = self.d_model
        Q = D // 4

        for i in range(self.n_layers):
            if i == 0:
                mask_temp = 1 - repeat(
                    y_mask_L_flattened, "B N -> B L N",
                    L=temporal_incidence_matrix.shape[1])
                mask_temp[mask_temp == 0] = 1e-8

            # --------- Step 1: SRI — Spike-Refractory Incidence ---------
            g_n, e_n = self.sri[i](
                observation_nodes, x_y_mask_flattened,
                variable_incidence_matrix, variable_indices_flattened,
                time_norm)

            # --------- Step 1.5: SQC — spike-driven rotation ---------
            # Rotate observation_nodes by theta proportional to spike before
            # computing gated K/V for n2h attention. Geometric effect.
            if not self.no_sqc:
                obs_rotated = self.sqc[i](observation_nodes, g_n)
            else:
                obs_rotated = observation_nodes

            # --------- Step 2: residual spike gating (mask + scale*(g-mask)) ---------
            gs = self.gate_scale[i]
            mask_2d = x_y_mask_flattened.unsqueeze(-1)
            gating = mask_2d + gs * (g_n.unsqueeze(-1) - mask_2d)
            obs_gated = obs_rotated * gating

            # --------- Step 3: Node -> Temporal hyperedge ---------
            temporal_hyperedges = self.node2temporal_hyperedge[i](
                temporal_hyperedges,
                torch.cat([
                    variable_hyperedges.gather(
                        1, repeat(variable_indices_flattened,
                                  "B N -> B N D", D=D)),
                    obs_gated,
                ], -1),
                temporal_incidence_matrix
                if i != 0 else temporal_incidence_matrix * mask_temp)

            if i == 0:
                mask_temp = 1 - repeat(
                    y_mask_L_flattened, "B N -> B E N",
                    E=variable_incidence_matrix.shape[1])
                mask_temp[mask_temp == 0] = 1e-8

            # --------- Step 4: Node -> Variable hyperedge ---------
            variable_hyperedges = self.node2variable_hyperedge[i](
                variable_hyperedges,
                torch.cat([
                    temporal_hyperedges.gather(
                        1, repeat(time_indices_flattened,
                                  "B N -> B N D", D=D)),
                    obs_gated,
                ], -1),
                variable_incidence_matrix
                if i != 0 else variable_incidence_matrix * mask_temp)

            # --------- Step 5: Hyperedge -> Node via QMF ---------
            tg = temporal_hyperedges.gather(
                1, repeat(time_indices_flattened, "B N -> B N D", D=D))
            vg = variable_hyperedges.gather(
                1, repeat(variable_indices_flattened, "B N -> B N D", D=D))

            if not self.oom_flag:
                try:
                    obs_for_h2n = self.node_self_update[i](
                        observation_nodes,
                        torch.cat([tg, vg, observation_nodes], -1),
                        x_y_mask_flattened.unsqueeze(2)
                        * x_y_mask_flattened.unsqueeze(1))
                except:
                    self.oom_flag = True
                    logger.warning(
                        "SQHH: CUDA OOM in self-attention, using fallback.")

            if self.oom_flag:
                obs_for_h2n = observation_nodes

            q_R = self.proj_R[i](obs_for_h2n)
            q_I = self.proj_I[i](tg)
            q_J = self.proj_J[i](vg)
            q_K = e_n
            q = torch.cat([q_R, q_I, q_J, q_K], dim=-1)

            if not self.no_qmf:
                h2n_out = self.quat_h2n[i](q)
            else:
                h2n_out = self.linear_h2n[i](q)

            md = repeat(x_y_mask_flattened, "B N -> B N D", D=D)
            observation_nodes = self.activation((observation_nodes + h2n_out) * md)

            # --------- Step 6: Hyperedge <-> Hyperedge (last layer only) ---------
            if i in self.hyperedge2hyperedge_layers:
                sync_mask = x_y_mask
                qk = self.get_fine_grained_embedding(observation_nodes, sync_mask)
                mc = sync_mask.transpose(-1, -2) @ sync_mask
                nopv = mc.diagonal(0, -2, -1)
                mc[nopv != 0] = (mc / repeat(
                    nopv, "B E -> B E E2", E2=sync_mask.shape[-1]))[nopv != 0]
                variable_hyperedges = (
                    variable_hyperedges
                    + self.variable_hyperedge2variable_hyperedge(
                        x=variable_hyperedges,
                        query_aux=qk, key_aux=qk,
                        merge_coefficients=mc))

        return observation_nodes, temporal_hyperedges, variable_hyperedges


# ============================================================================
# Main Model
# ============================================================================

class Model(nn.Module):
    """SQHH v2: SQHyper backbone + SRI + SQC."""

    def __init__(self, configs: ExpConfigs):
        super().__init__()
        self.configs = configs
        self.enc_in = configs.enc_in
        D = (configs.d_model // 4) * 4
        if D != configs.d_model:
            logger.warning(
                f"SQHH: d_model {configs.d_model} -> {D} (must be ×4)")
        self.d_model = D

        sl = configs.seq_len_max_irr or configs.seq_len
        pl = configs.pred_len_max_irr or configs.pred_len
        tl = sl + pl

        # Ablation flags (accept both new sqhh_* names and legacy ones)
        no_sri = bool(int(getattr(configs, "sqhh_no_sri", 0)))
        no_sqc = bool(int(getattr(configs, "sqhh_no_sqc", 0)))
        no_qmf = bool(int(getattr(configs, "sqhh_no_qmf",
                                   getattr(configs, "sqhyper_no_qmf", 0))))

        self.hypergraph_encoder = HypergraphEncoder(
            self.enc_in, tl, D)
        self.hypergraph_learner = HypergraphLearner(
            configs.n_layers, D, configs.n_heads, tl, self.enc_in,
            no_sri=no_sri, no_sqc=no_sqc, no_qmf=no_qmf)
        self.hypergraph_decoder = nn.Linear(3 * D, 1)

        # Diagnostic: track spike sparsity and refractory activity
        self.diag_interval = int(getattr(configs, "sqhh_diag_interval", 0))
        self.register_buffer(
            "_diag_step", torch.zeros(1, dtype=torch.long), persistent=False)

    def pad_and_flatten(self, tensor, mask, max_len):
        B = tensor.shape[0]
        tf = tensor.view(B, -1)
        mf = mask.view(B, -1)
        d = torch.cumsum(mf, 1) - 1
        k = (mf == 1) & (d < max_len)
        r = torch.zeros(B, max_len, dtype=tensor.dtype, device=tensor.device)
        rows = torch.arange(B, device=tensor.device).unsqueeze(1).expand_as(mf)
        r[rows[k], d[k].long()] = tf[k]
        return r

    def unpad_and_reshape(self, tensor_flattened, original_mask, original_shape):
        original_mask = original_mask.bool()
        device = tensor_flattened.device
        result = torch.zeros(original_shape,
                              dtype=tensor_flattened.dtype, device=device)
        counts = original_mask.sum(
            dim=tuple(range(1, original_mask.dim())))
        batch_size, max_len = tensor_flattened.shape[:2]
        steps = torch.arange(max_len, device=device).expand(
            batch_size, max_len)
        src_mask = steps < counts.unsqueeze(-1)
        result[original_mask] = tensor_flattened[src_mask]
        return result

    def forward(self, x, x_mark=None, x_mask=None, y=None, y_mark=None,
                y_mask=None, x_L_flattened=None, x_y_mask_flattened=None,
                y_L_flattened=None, y_mask_L_flattened=None,
                exp_stage="train", **kwargs):
        BATCH_SIZE, SEQ_LEN, ENC_IN = x.shape
        Y_LEN = self.configs.pred_len if self.configs.pred_len != 0 else SEQ_LEN
        if x_mark is None:
            x_mark = repeat(
                torch.arange(end=x.shape[1], dtype=x.dtype, device=x.device)
                / x.shape[1], "L -> B L 1", B=x.shape[0])
        if x_mask is None:
            x_mask = torch.ones_like(x, device=x.device, dtype=x.dtype)
        if y is None:
            y = torch.ones((BATCH_SIZE, Y_LEN, ENC_IN),
                           dtype=x.dtype, device=x.device)
        if y_mark is None:
            y_mark = repeat(
                torch.arange(end=y.shape[1], dtype=y.dtype, device=y.device)
                / y.shape[1], "L -> B L 1", B=y.shape[0])
        if y_mask is None:
            y_mask = torch.ones_like(y, device=y.device, dtype=y.dtype)

        _, PRED_LEN, _ = y.shape
        L = SEQ_LEN + PRED_LEN
        x_mark = x_mark[:, :, :1]
        y_mark = y_mark[:, :, :1]

        if self.configs.task_name in [
            "short_term_forecast", "long_term_forecast",
            "classification", "representation_learning",
        ]:
            x_zeros = torch.zeros_like(y, dtype=x.dtype, device=x.device)
            y_zeros = torch.zeros_like(x, dtype=y.dtype, device=y.device)
            x_y_mark = torch.cat([x_mark, y_mark], dim=1)
            x_L = torch.cat([x, x_zeros], dim=1)
            x_y_mask = torch.cat([x_mask, y_mask], dim=1)
            y_L = torch.cat([y_zeros, y], dim=1)
            y_mask_L = torch.cat([y_zeros, y_mask], dim=1)
        elif self.configs.task_name in ["imputation"]:
            x_y_mark = x_mark
            x_L = x
            x_y_mask = x_mask + y_mask
            y_L = y
            y_mask_L = y_mask
        else:
            raise NotImplementedError()

        time_indices = torch.cumsum(
            torch.ones_like(x_L).to(torch.int64), dim=1) - 1
        variable_indices = torch.cumsum(
            torch.ones_like(x_L).to(torch.int64), dim=-1) - 1
        x_y_mask_bool = x_y_mask.to(torch.bool)
        N_OBSERVATIONS_MAX = torch.max(x_y_mask.sum((1, 2))).to(torch.int64)
        N_OBSERVATIONS_MIN = torch.min(x_y_mask.sum((1, 2))).to(torch.int64)
        is_regular = (
            N_OBSERVATIONS_MAX == N_OBSERVATIONS_MIN == L * ENC_IN)

        if (x_L_flattened or x_y_mask_flattened
                or y_L_flattened or y_mask_L_flattened) is None:
            if is_regular:
                x_L_flattened = x_L.reshape(BATCH_SIZE, L * ENC_IN)
                x_y_mask_flattened = x_y_mask.reshape(BATCH_SIZE, L * ENC_IN)
                y_L_flattened = y_L.reshape(BATCH_SIZE, L * ENC_IN)
                y_mask_L_flattened = y_mask_L.reshape(BATCH_SIZE, L * ENC_IN)
            else:
                x_L_flattened = self.pad_and_flatten(
                    x_L, x_y_mask_bool, N_OBSERVATIONS_MAX)
                x_y_mask_flattened = self.pad_and_flatten(
                    x_y_mask, x_y_mask_bool, N_OBSERVATIONS_MAX)
                y_L_flattened = self.pad_and_flatten(
                    y_L, x_y_mask_bool, N_OBSERVATIONS_MAX)
                y_mask_L_flattened = self.pad_and_flatten(
                    y_mask_L, x_y_mask_bool, N_OBSERVATIONS_MAX)

        if is_regular:
            time_indices_flattened = time_indices.reshape(
                BATCH_SIZE, L * ENC_IN)
            variable_indices_flattened = variable_indices.reshape(
                BATCH_SIZE, L * ENC_IN)
        else:
            time_indices_flattened = self.pad_and_flatten(
                time_indices, x_y_mask_bool, N_OBSERVATIONS_MAX)
            variable_indices_flattened = self.pad_and_flatten(
                variable_indices, x_y_mask_bool, N_OBSERVATIONS_MAX)

        # Normalized time for SRI refractory kernel
        time_norm = time_indices_flattened.float() / max(1, L - 1)

        # Encode
        (observation_nodes, temporal_hyperedges, variable_hyperedges,
         temporal_incidence_matrix, variable_incidence_matrix
         ) = self.hypergraph_encoder(
            x_L_flattened, x_y_mask_flattened, y_mask_L_flattened,
            x_y_mark, variable_indices_flattened,
            time_indices_flattened, N_OBSERVATIONS_MAX)

        # Hypergraph learning (SRI + SQC + QMF)
        (observation_nodes, temporal_hyperedges, variable_hyperedges
         ) = self.hypergraph_learner(
            observation_nodes, temporal_hyperedges, variable_hyperedges,
            time_indices_flattened, variable_indices_flattened,
            temporal_incidence_matrix, variable_incidence_matrix,
            x_y_mask_flattened, x_y_mask, y_mask_L_flattened,
            time_norm)

        # Decode (same as SQHyper)
        if self.configs.task_name in [
            "long_term_forecast", "short_term_forecast", "imputation"
        ]:
            D = self.d_model
            pred_flattened = self.hypergraph_decoder(torch.cat([
                observation_nodes,
                temporal_hyperedges.gather(
                    1, repeat(time_indices_flattened, "B N -> B N D", D=D)),
                variable_hyperedges.gather(
                    1, repeat(variable_indices_flattened, "B N -> B N D", D=D)),
            ], dim=-1)).squeeze(-1)

            # Diagnostic (optional)
            if self.training and self.diag_interval > 0:
                self._diag_step += 1
                if int(self._diag_step.item()) % self.diag_interval == 0:
                    with torch.no_grad():
                        # Report spike stats from the last layer's SRI (by
                        # reusing the existing gate stats is complex, so we
                        # re-run a quick pass on the first layer).
                        # Simple proxy: reuse x_y_mask_flattened density.
                        obs_density = float(
                            x_y_mask_flattened.float().mean().item())
                        logger.debug(
                            f"[SQHH diag step={int(self._diag_step.item())}] "
                            f"obs_density={obs_density:.3f} "
                            f"n_layers={self.hypergraph_learner.n_layers} "
                            f"no_sri={self.hypergraph_learner.no_sri} "
                            f"no_sqc={self.hypergraph_learner.no_sqc}"
                        )

            if exp_stage in ["train", "val"]:
                return {
                    "pred": pred_flattened,
                    "true": y_L_flattened,
                    "mask": y_mask_L_flattened,
                }
            else:
                pred = self.unpad_and_reshape(
                    pred_flattened,
                    torch.cat([x_mask, y_mask], dim=1),
                    (BATCH_SIZE, SEQ_LEN + PRED_LEN, ENC_IN))
                f_dim = -1 if self.configs.features == 'MS' else 0
                return {
                    "pred": pred[:, -PRED_LEN:, f_dim:],
                    "true": y[:, :, f_dim:],
                    "mask": y_mask[:, :, f_dim:],
                }
        else:
            raise NotImplementedError()
