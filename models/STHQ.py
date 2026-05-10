# STHQ v7: Spike-Triggered Hypergraph with Quaternion fusion (SQHyper backbone)
#
# Diagnostic finding from v5/v6:
#   STHQ (v1-v6) replaced HyperIMTS's HARD per-timestep temporal hyperedges
#   with SOFT learnable K_t Gaussian anchors. On long irregular series this
#   caused 5x regression vs SQHyper (HA: 0.092 vs 0.0173). Per-timestep
#   coupling between cells was destroyed, and there was no cell-to-cell
#   attention to compensate.
#
# v7 commitment:
#   1. RECOVER the SQHyper backbone:
#        - L hard temporal hyperedges (one per timestep)
#        - enc_in hard variable hyperedges (one per variable)
#        - Hard 0/1 incidence matrices (NOT Gaussian kernels)
#        - node_self_update: full cell-to-cell attention (3D MHA)
#        - QMF: Quaternion Multi-Source Fusion for hyperedge -> node
#        - IrregularityAwareAttention for variable-hyperedge co-update
#
#   2. STHQ-distinct contribution: stronger spike encoder
#        - Multi-feature input: (value, time, mask, var_emb)  vs SGI's
#          (obs, deviation). Captures temporal phase + variable identity
#          without relying on the hypergraph structure to bootstrap.
#        - Spike floor: every observed cell contributes >= floor mass to
#          K/V gating. Prevents "spike starvation" on sparse medical data
#          (P12, MIMIC) where the encoder otherwise collapses to ~0.
#        - Produces both:
#            g_n: scalar gate (B, N) for K/V gating in n2h attention
#            e_n: D/4 event features for the K-component of QMF
#
#   3. Removed (proven not to help):
#        - K_t/K_v soft hyperedges (replaced by L/enc_in hard)
#        - Hamilton/Linear hybrid (alpha) — both paths equivalent
#        - STEA dynamic event anchors (beta) — redundant with L hyperedges
#        - Spike-modulated bandwidth (gamma) — no kernel to modulate
#        - tau-repulsion / var-entropy aux losses (no anchors to regularize)
#
# Key STHQ-distinct hyperparameters:
#   --sthq_spike_floor      : minimum K/V gating weight on observed cells
#   --sthq_event_layer_warmup: number of layers where e_n is zeroed (warmup)
#   --sthq_no_quaternion    : ablation flag (replace QMF with linear h2n)
#   --sthq_no_spike         : ablation flag (replace spike encoder with mask)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat, rearrange
from torch import Tensor
from utils.ExpConfigs import ExpConfigs
from utils.globals import logger


# ============================================================================
# Quaternion Linear (Hamilton-product structured transform)
# ============================================================================

class QuaternionLinear(nn.Module):
    """Hamilton-product structured linear layer.

    For input/output dim 4Q, parameter count is 4 * Q * Q (1/4 of flat
    Linear). Implements W in block form with cross-component coupling that
    matches the algebra of quaternion multiplication.
    """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        assert in_features % 4 == 0 and out_features % 4 == 0, (
            f"QuaternionLinear requires dims divisible by 4, "
            f"got {in_features}, {out_features}"
        )
        qi, qo = in_features // 4, out_features // 4
        self.r = nn.Parameter(torch.empty(qo, qi))
        self.i = nn.Parameter(torch.empty(qo, qi))
        self.j = nn.Parameter(torch.empty(qo, qi))
        self.k = nn.Parameter(torch.empty(qo, qi))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def init_identity(self):
        nn.init.eye_(self.r)
        nn.init.zeros_(self.i)
        nn.init.zeros_(self.j)
        nn.init.zeros_(self.k)

    def init_xavier(self):
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


# ============================================================================
# Stronger Spike Encoder (STHQ-distinct)
#
# Replaces SQHyper's SGI module. Two key improvements:
#   1. Multi-feature input: (value, time_norm, mask, var_emb) — captures
#      temporal phase + variable identity intrinsically rather than via
#      hypergraph context (which is unreliable at layer 0).
#   2. Spike floor: g_n = floor + (1-floor) * sigmoid(MLP). Guarantees
#      every observed cell contributes >= floor mass to attention K/V.
#      Critical on sparse medical data where MLP otherwise outputs ~0.
# ============================================================================

class SpikeEncoder(nn.Module):
    def __init__(self, n_vars, d_model, hidden=None, floor=0.0):
        super().__init__()
        H = hidden if hidden is not None else max(16, d_model // 4)
        self.var_emb = nn.Embedding(n_vars, H)
        nn.init.normal_(self.var_emb.weight, std=0.1)
        # Encoder body: shared between gate head and event head
        self.body = nn.Sequential(
            nn.Linear(3 + H, H),
            nn.GELU(),
            nn.Linear(H, H),
        )
        # Gate head -> scalar in [floor, 1]
        self.gate_head = nn.Linear(H, 1)
        # Initialize gate near 1.0 (sigmoid(2) ~ 0.88) so layer 0 starts
        # close to mask-only gating, then learns to specialize.
        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(self.gate_head.bias, 2.0)
        # Event head -> D/4 features (used as q_K in QMF)
        # Init to zero so event signal starts off and is learned in.
        Q = d_model // 4
        self.event_head = nn.Linear(H, Q)
        nn.init.zeros_(self.event_head.weight)
        nn.init.zeros_(self.event_head.bias)
        self.floor = float(floor)

    def forward(self, value, time_norm, mask, var_id):
        """
        value     : [B, N]
        time_norm : [B, N]   in [0, 1]
        mask      : [B, N]   1 for observed (incl. forecast queries)
        var_id    : [B, N]
        Returns:
            g_n : [B, N]      gate in [floor, 1] on observed cells, 0 elsewhere
            e_n : [B, N, D/4] event features (gated by g_n and mask)
        """
        ve = self.var_emb(var_id)
        feats = torch.cat([
            (value * mask).unsqueeze(-1),
            time_norm.unsqueeze(-1),
            mask.unsqueeze(-1),
            ve,
        ], dim=-1)
        h = self.body(feats)                              # [B, N, H]
        g_raw = torch.sigmoid(self.gate_head(h).squeeze(-1))
        g_n = (self.floor + (1.0 - self.floor) * g_raw) * mask
        e_n = self.event_head(h) * g_n.unsqueeze(-1) * mask.unsqueeze(-1)
        return g_n, e_n


# ============================================================================
# Multi-Head Attention Block (from HyperIMTS / SQHyper, identical)
# ============================================================================

class MultiHeadAttentionBlock(nn.Module):
    """Standard masked multi-head attention with residual + ReLU FFN.

    Used for n2h (node -> hyperedge) and h2n self-update across cells.
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

    def forward(self, Q, K, mask=None):
        Q = self.fc_q(Q)
        K_, V = self.fc_k(K), self.fc_v(K)
        ds = self.n_dim // self.num_heads
        Q_ = torch.cat(Q.split(ds, 2), 0)
        K_ = torch.cat(K_.split(ds, 2), 0)
        V = torch.cat(V.split(ds, 2), 0)
        A = Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.n_dim)
        if mask is not None:
            A = A.masked_fill(mask.repeat(self.num_heads, 1, 1) == 0, -1e9)
        A = torch.softmax(A, 2)
        O = torch.cat((Q_ + A.bmm(V)).split(Q.size(0), 0), 2)
        O = O if getattr(self, "ln0", None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, "ln1", None) is None else self.ln1(O)
        return O


# ============================================================================
# Irregularity-Aware Attention (from HyperIMTS, used in last layer h2h)
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
        B, V, D = x.shape
        query = self.query_proj(x)
        key = self.key_proj(x)
        value = self.value_proj(x)
        att = torch.matmul(query, key.transpose(-2, -1)) / self.scale
        mask_val = torch.finfo(att.dtype).min
        if query_aux is not None and key_aux is not None:
            att_aux = torch.matmul(query_aux, key_aux.transpose(-2, -1)) / (
                key_aux.shape[-1] ** 0.5)
            non_zero = (att_aux != 0)
            positive = (att > self.threshold)
            mask = positive & non_zero
            att[mask] = (
                (1 - merge_coefficients) * att
                + merge_coefficients * att_aux
            )[mask]
        if adjacency_mask is not None:
            att = att.masked_fill(adjacency_mask == 0, mask_val)
        attn = torch.softmax(att, dim=-1)
        return torch.matmul(attn, value)


# ============================================================================
# Hypergraph Encoder: cells -> nodes + hard hyperedges with binary incidence
# (Identical to HyperIMTS / SQHyper structure)
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
                x_y_mark, variable_indices_flattened, time_indices_flattened,
                N_OBSERVATIONS_MAX):
        B = x_L_flattened.shape[0]
        E, L, D = self.enc_in, x_y_mark.shape[1], self.d_model
        N = N_OBSERVATIONS_MAX

        # Concat value with forecast-target indicator (1 for query, 0 for obs)
        x_L_flattened = torch.stack(
            [x_L_flattened, 1 - x_y_mask_flattened + y_mask_L_flattened],
            dim=-1)

        # Hard temporal incidence: time_step t connected to cell n iff
        # time_indices_flattened[b, n] == t.
        temporal_incidence_matrix = repeat(
            time_indices_flattened, "B N -> B L N", L=L)
        temporal_incidence_matrix = (temporal_incidence_matrix == repeat(
            torch.ones(B, L, device=x_L_flattened.device).cumsum(dim=1),
            "B L -> B L N", N=N) - 1).float()
        temporal_incidence_matrix = (
            temporal_incidence_matrix
            * repeat(x_y_mask_flattened, "B N -> B L N", L=L)
        )

        # Hard variable incidence: variable v connected to cell n iff
        # variable_indices_flattened[b, n] == v.
        variable_incidence_matrix = repeat(
            torch.ones(B, E, device=x_L_flattened.device).cumsum(dim=1) - 1,
            "B E -> B E N", N=N)
        variable_incidence_matrix = (variable_incidence_matrix == repeat(
            variable_indices_flattened, "B N -> B E N", E=E)).float()
        variable_incidence_matrix = (
            variable_incidence_matrix
            * repeat(x_y_mask_flattened, "B N -> B E N", E=E)
        )

        # Node / hyperedge initializations
        observation_nodes = self.relu(
            self.observation_node_encoder(x_L_flattened)
        ) * repeat(x_y_mask_flattened, "B N -> B N D", D=D)
        temporal_hyperedges = torch.sin(
            self.temporal_hyperedge_encoder(x_y_mark))
        variable_hyperedges = self.relu(
            repeat(self.variable_hyperedge_weights, "E D -> B E D", B=B))

        return (observation_nodes, temporal_hyperedges, variable_hyperedges,
                temporal_incidence_matrix, variable_incidence_matrix)


# ============================================================================
# Hypergraph Learner: SQHyper backbone + STHQ SpikeEncoder + QMF
# ============================================================================

class HypergraphLearner(nn.Module):
    """Per-layer message passing.

    Step 1: SpikeEncoder -> (g_n, e_n)
    Step 2: Node -> Temporal hyperedge      (K/V gated by g_n)
    Step 3: Node -> Variable hyperedge      (K/V gated by g_n)
    Step 4: Cell-to-cell self-attention     (node_self_update; full 3D MHA)
    Step 5: Quaternion Multi-Source Fusion  (q_R=self, q_I=temp, q_J=var, q_K=event)
    Step 6: (last layer only) Hyperedge -> Hyperedge via IrregularityAware attn
    """

    def __init__(self, n_layers, d_model, n_heads, time_length, n_vars,
                 spike_floor=0.0, no_spike=False, no_quaternion=False,
                 event_layer_warmup=0):
        super().__init__()
        self.n_layers = n_layers
        self.d_model = d_model
        self.n_vars = n_vars
        self.no_spike = bool(no_spike)
        self.no_quaternion = bool(no_quaternion)
        self.event_layer_warmup = int(event_layer_warmup)
        self.activation = nn.ReLU()
        D = d_model
        Q = D // 4

        # n2h attention (per layer)
        self.node2temporal_hyperedge = nn.ModuleList([
            MultiHeadAttentionBlock(D, 2 * D, 2 * D, D, n_heads)
            for _ in range(n_layers)
        ])
        self.node2variable_hyperedge = nn.ModuleList([
            MultiHeadAttentionBlock(D, 2 * D, 2 * D, D, n_heads)
            for _ in range(n_layers)
        ])
        # Cross-cell self-attention (per layer, OOM-safe with try/except)
        self.node_self_update = nn.ModuleList([
            MultiHeadAttentionBlock(D, 3 * D, 3 * D, D, n_heads)
            for _ in range(n_layers)
        ])
        # Variable-hyperedge co-update (last layer only)
        self.variable_hyperedge2variable_hyperedge = IrregularityAwareAttention(D)
        self.hyperedge2hyperedge_layers = [n_layers - 1]
        self.scale = 1 / time_length
        self.oom_flag = False

        # STHQ-distinct: per-layer stronger SpikeEncoder
        self.spike = nn.ModuleList([
            SpikeEncoder(n_vars=n_vars, d_model=D, floor=spike_floor)
            for _ in range(n_layers)
        ])
        # Per-layer learnable K/V gating strength (init 0 = HyperIMTS-equivalent)
        # Lets the model learn per-layer per-dataset gating intensity.
        self.gate_scale = nn.ParameterList([
            nn.Parameter(torch.zeros(1)) for _ in range(n_layers)
        ])

        # QMF projections (per layer): D -> D/4 for each non-spike component
        self.proj_R = nn.ModuleList([nn.Linear(D, Q) for _ in range(n_layers)])
        self.proj_I = nn.ModuleList([nn.Linear(D, Q) for _ in range(n_layers)])
        self.proj_J = nn.ModuleList([nn.Linear(D, Q) for _ in range(n_layers)])
        # Quaternion fusion layer
        self.quat_h2n = nn.ModuleList()
        for _ in range(n_layers):
            ql = QuaternionLinear(D, D)
            ql.init_identity()
            self.quat_h2n.append(ql)
        # Linear fallback (used when --sthq_no_quaternion)
        self.linear_h2n = nn.ModuleList(
            [nn.Linear(D, D) for _ in range(n_layers)])

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
                value_flattened, time_norm_flattened):
        D = self.d_model
        Q = D // 4
        diag_records = []

        for i in range(self.n_layers):
            # Layer-0 contribute mask: cells that are forecast targets must
            # not contribute their (zero) values to any hyperedge they form.
            if i == 0:
                mask_temp_t = 1 - repeat(
                    y_mask_L_flattened, "B N -> B L N",
                    L=temporal_incidence_matrix.shape[1])
                mask_temp_t[mask_temp_t == 0] = 1e-8
                mask_temp_v = 1 - repeat(
                    y_mask_L_flattened, "B N -> B E N",
                    E=variable_incidence_matrix.shape[1])
                mask_temp_v[mask_temp_v == 0] = 1e-8

            # ----------------------------------------------------------------
            # Step 1: STHQ SpikeEncoder -> (g_n, e_n)
            # ----------------------------------------------------------------
            if not self.no_spike:
                g_n, e_n = self.spike[i](
                    value=value_flattened,
                    time_norm=time_norm_flattened,
                    mask=x_y_mask_flattened,
                    var_id=variable_indices_flattened,
                )
                # K/V gating: residual blend between mask-only and full spike
                # gating, controlled by per-layer learnable gate_scale.
                gs = self.gate_scale[i]
                mask_2d = x_y_mask_flattened.unsqueeze(-1)
                gating = mask_2d + gs * (g_n.unsqueeze(-1) - mask_2d)
                obs_gated = observation_nodes * gating
            else:
                # Ablation: no spike at all (mask-only gating, zero events)
                g_n = x_y_mask_flattened
                e_n = torch.zeros(
                    observation_nodes.shape[0], observation_nodes.shape[1], Q,
                    device=observation_nodes.device,
                    dtype=observation_nodes.dtype)
                obs_gated = observation_nodes

            # Event warmup: zero e_n for the first `warmup` layers so QMF
            # learns spatio-temporal fusion before integrating event signal.
            if i < self.event_layer_warmup:
                e_n_used = torch.zeros_like(e_n)
            else:
                e_n_used = e_n

            # ----------------------------------------------------------------
            # Step 2: Node -> Temporal hyperedge
            # ----------------------------------------------------------------
            temporal_hyperedges = self.node2temporal_hyperedge[i](
                temporal_hyperedges,
                torch.cat([
                    variable_hyperedges.gather(
                        1,
                        repeat(variable_indices_flattened,
                               "B N -> B N D", D=D)
                    ),
                    obs_gated,
                ], -1),
                (temporal_incidence_matrix * mask_temp_t) if i == 0
                else temporal_incidence_matrix,
            )

            # ----------------------------------------------------------------
            # Step 3: Node -> Variable hyperedge
            # ----------------------------------------------------------------
            variable_hyperedges = self.node2variable_hyperedge[i](
                variable_hyperedges,
                torch.cat([
                    temporal_hyperedges.gather(
                        1,
                        repeat(time_indices_flattened, "B N -> B N D", D=D)
                    ),
                    obs_gated,
                ], -1),
                (variable_incidence_matrix * mask_temp_v) if i == 0
                else variable_incidence_matrix,
            )

            # ----------------------------------------------------------------
            # Step 4: Cell-to-cell self-attention (with OOM fallback)
            # ----------------------------------------------------------------
            tg = temporal_hyperedges.gather(
                1, repeat(time_indices_flattened, "B N -> B N D", D=D))
            vg = variable_hyperedges.gather(
                1, repeat(variable_indices_flattened, "B N -> B N D", D=D))

            obs_for_h2n = observation_nodes
            if not self.oom_flag:
                try:
                    obs_for_h2n = self.node_self_update[i](
                        observation_nodes,
                        torch.cat([tg, vg, observation_nodes], -1),
                        x_y_mask_flattened.unsqueeze(2)
                        * x_y_mask_flattened.unsqueeze(1),
                    )
                except RuntimeError:
                    self.oom_flag = True
                    logger.warning(
                        "STHQ: cross-cell self-attention OOM, falling back "
                        "to skip on subsequent layers")
                    obs_for_h2n = observation_nodes

            # ----------------------------------------------------------------
            # Step 5: Quaternion Multi-Source Fusion (or linear fallback)
            # ----------------------------------------------------------------
            q_R = self.proj_R[i](obs_for_h2n)   # self
            q_I = self.proj_I[i](tg)             # temporal hyperedge
            q_J = self.proj_J[i](vg)             # variable hyperedge
            q_K = e_n_used                        # event features
            q = torch.cat([q_R, q_I, q_J, q_K], dim=-1)

            if not self.no_quaternion:
                h2n_out = self.quat_h2n[i](q)
            else:
                h2n_out = self.linear_h2n[i](q)

            md = repeat(x_y_mask_flattened, "B N -> B N D", D=D)
            observation_nodes = self.activation(
                (observation_nodes + h2n_out) * md)

            # Diagnostic snapshot (detach to avoid grad-graph retention)
            with torch.no_grad():
                diag_records.append({
                    "layer": i,
                    "g_mean": float(
                        (g_n * x_y_mask_flattened).sum().item()
                        / x_y_mask_flattened.sum().clamp_min(1).item()
                    ),
                    "e_norm": float(e_n.norm(dim=-1).mean().item()),
                    "gate_scale": float(self.gate_scale[i].detach().item()),
                })

            # ----------------------------------------------------------------
            # Step 6: (last layer) Hyperedge -> Hyperedge via Irregularity attn
            # ----------------------------------------------------------------
            if i in self.hyperedge2hyperedge_layers:
                sync_mask = x_y_mask
                qk = self.get_fine_grained_embedding(
                    observation_nodes, sync_mask)
                mc = sync_mask.transpose(-1, -2) @ sync_mask
                nopv = mc.diagonal(0, -2, -1)
                mc[nopv != 0] = (mc / repeat(
                    nopv, "B E -> B E E2", E2=sync_mask.shape[-1]
                ))[nopv != 0]
                variable_hyperedges = (
                    variable_hyperedges
                    + self.variable_hyperedge2variable_hyperedge(
                        x=variable_hyperedges,
                        query_aux=qk, key_aux=qk,
                        merge_coefficients=mc,
                    )
                )

        return observation_nodes, temporal_hyperedges, variable_hyperedges, diag_records


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
            logger.warning(
                f"STHQ: d_model {configs.d_model} -> {self.d_model} "
                f"(must be ×4)")
        D = self.d_model
        sl = configs.seq_len_max_irr or configs.seq_len
        pl = configs.pred_len_max_irr or configs.pred_len
        tl = sl + pl

        self.hypergraph_encoder = HypergraphEncoder(self.enc_in, tl, D)
        self.hypergraph_learner = HypergraphLearner(
            n_layers=configs.n_layers,
            d_model=D,
            n_heads=configs.n_heads,
            time_length=tl,
            n_vars=self.enc_in,
            spike_floor=float(getattr(configs, "sthq_spike_floor", 0.0)),
            no_spike=int(getattr(configs, "sthq_no_spike", 0)),
            no_quaternion=int(getattr(configs, "sthq_no_quaternion", 0)),
            event_layer_warmup=int(
                getattr(configs, "sthq_event_layer_warmup", 0)),
        )
        self.hypergraph_decoder = nn.Linear(3 * D, 1)

        self.diag_interval = int(getattr(configs, "sthq_diag_interval", 0))
        self.register_buffer(
            "_diag_step", torch.zeros(1, dtype=torch.long), persistent=False)

    # --- helpers (shared with SQHyper) -----------------------------------

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
        result = torch.zeros(original_shape, dtype=tensor_flattened.dtype,
                             device=device)
        counts = original_mask.sum(
            dim=tuple(range(1, original_mask.dim())))
        batch_size, max_len = tensor_flattened.shape[:2]
        steps = torch.arange(max_len, device=device).expand(batch_size, max_len)
        src_mask = steps < counts.unsqueeze(-1)
        result[original_mask] = tensor_flattened[src_mask]
        return result

    # --- forward ----------------------------------------------------------

    def forward(self, x, x_mark=None, x_mask=None, y=None, y_mark=None,
                y_mask=None,
                x_L_flattened=None, x_y_mask_flattened=None,
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
            y = torch.ones((BATCH_SIZE, Y_LEN, ENC_IN), dtype=x.dtype,
                           device=x.device)
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

        time_norm_flattened = (
            time_indices_flattened.float() / max(1, L - 1))

        # Encode
        (observation_nodes, temporal_hyperedges, variable_hyperedges,
         temporal_incidence_matrix, variable_incidence_matrix
         ) = self.hypergraph_encoder(
            x_L_flattened, x_y_mask_flattened, y_mask_L_flattened,
            x_y_mark, variable_indices_flattened, time_indices_flattened,
            N_OBSERVATIONS_MAX,
        )

        # Hypergraph learning
        (observation_nodes, temporal_hyperedges, variable_hyperedges,
         diag_records) = self.hypergraph_learner(
            observation_nodes, temporal_hyperedges, variable_hyperedges,
            time_indices_flattened, variable_indices_flattened,
            temporal_incidence_matrix, variable_incidence_matrix,
            x_y_mask_flattened, x_y_mask, y_mask_L_flattened,
            value_flattened=x_L_flattened,
            time_norm_flattened=time_norm_flattened,
        )

        # Diagnostics
        if self.training and self.diag_interval > 0:
            self._diag_step += 1
            if int(self._diag_step.item()) % self.diag_interval == 0:
                stats = " | ".join([
                    f"L{r['layer']}: g={r['g_mean']:.3f} "
                    f"|e|={r['e_norm']:.3f} gs={r['gate_scale']:+.3f}"
                    for r in diag_records
                ])
                logger.debug(
                    f"[STHQ diag step={int(self._diag_step.item())}] {stats}"
                )

        # Decode
        if self.configs.task_name in [
            "long_term_forecast", "short_term_forecast", "imputation"
        ]:
            D = self.d_model
            pred_flattened = self.hypergraph_decoder(torch.cat([
                observation_nodes,
                temporal_hyperedges.gather(
                    1, repeat(time_indices_flattened, "B N -> B N D", D=D)),
                variable_hyperedges.gather(
                    1,
                    repeat(variable_indices_flattened,
                           "B N -> B N D", D=D)),
            ], dim=-1)).squeeze(-1)

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
                    (BATCH_SIZE, SEQ_LEN + PRED_LEN, ENC_IN),
                )
                f_dim = -1 if self.configs.features == "MS" else 0
                return {
                    "pred": pred[:, -PRED_LEN:, f_dim:],
                    "true": y[:, :, f_dim:],
                    "mask": y_mask[:, :, f_dim:],
                }
        else:
            raise NotImplementedError()
