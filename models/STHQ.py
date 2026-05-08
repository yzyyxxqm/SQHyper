# STHQ-Net: Spike-Triggered Hyperedge with Quaternion States
#
# A native hypergraph model for IMTS forecasting where:
#   - SPIKE intensity directly multiplies hyperedge membership weights
#     (cells with low spike intensity contribute proportionally less)
#   - QUATERNION represents typed cell state with 4 components:
#       q_r = value channel
#       q_i = temporal channel
#       q_j = variable channel
#       q_k = spike-driven hidden channel
#     Hamilton product (Kronecker-form) composes hyperedge with cell state
#   - HYPERGRAPH has two emergent edge types:
#       Temporal hyperedges: K_t soft Gaussian time-windows (τ_k, ω_k learned)
#       Variable hyperedges: V × K_v learnable affinity matrix
#
# Key differences from PE-RQH/SC-PERQH/HyperIMTS:
#   - No codebook bottleneck: cells -> hyperedges -> cells through compositional
#     Hamilton product, not lossy code-routing
#   - Hyperedges have analytic membership (Gaussian + softmax), not VQ
#   - Each layer uses a different ω_k range (multi-scale temporal hyperedges)
#   - Spike is the multiplicative core of membership weights, not a side-gate
#
# Hyperparameters:
#   d_model           : 4Q (must be divisible by 4)
#   n_layers          : number of STHQ layers (default 3)
#   sthq_k_t          : K_t per layer (number of temporal hyperedges)
#   sthq_k_v          : K_v per layer (number of variable hyperedges)
#   sthq_omega_min    : min temporal bandwidth (relative to seq_len)
#   sthq_omega_max    : max temporal bandwidth
#   sthq_use_he_attn_from_layer : layer index from which to apply hyperedge
#                                  self-attention (set to n_layers to disable)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from utils.ExpConfigs import ExpConfigs
from utils.globals import logger


# ============================================================================
# Quaternion primitives (Kronecker formulation)
# ============================================================================

# Basis matrices for the 4x4 Hamilton structure constants.
# These constants implement the Hamilton product map M(p):
#   for p = (p_r, p_i, p_j, p_k), M(p) acts on q = (q_r, q_i, q_j, q_k)^T as
#   p ⊗ q.
# Equivalently, the QuaternionLinear weight matrix W of size (4Q, 4Q) is
#   W = A_1 ⊗ R + A_i ⊗ I + A_j ⊗ J + A_k ⊗ K
# where R, I, J, K are Q×Q learnable matrices.
_BASIS_A1 = torch.tensor([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
])
_BASIS_AI = torch.tensor([
    [0.0, -1.0, 0.0, 0.0],
    [1.0,  0.0, 0.0, 0.0],
    [0.0,  0.0, 0.0, -1.0],
    [0.0,  0.0, 1.0, 0.0],
])
_BASIS_AJ = torch.tensor([
    [0.0, 0.0, -1.0, 0.0],
    [0.0, 0.0,  0.0, 1.0],
    [1.0, 0.0,  0.0, 0.0],
    [0.0, -1.0, 0.0, 0.0],
])
_BASIS_AK = torch.tensor([
    [0.0,  0.0, 0.0, -1.0],
    [0.0,  0.0, -1.0, 0.0],
    [0.0,  1.0, 0.0,  0.0],
    [1.0,  0.0, 0.0,  0.0],
])


class QuaternionLinear(nn.Module):
    """Learnable quaternion-structured linear layer using Kronecker product.

    For input/output dim 4Q, parameter count is 4 * Q * Q (1/4 of flat Linear).
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        assert in_features % 4 == 0 and out_features % 4 == 0
        self.q_in = in_features // 4
        self.q_out = out_features // 4
        self.R = nn.Parameter(torch.empty(self.q_out, self.q_in))
        self.I = nn.Parameter(torch.empty(self.q_out, self.q_in))
        self.J = nn.Parameter(torch.empty(self.q_out, self.q_in))
        self.K = nn.Parameter(torch.empty(self.q_out, self.q_in))
        for p in (self.R, self.I, self.J, self.K):
            nn.init.xavier_uniform_(p, gain=0.5)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        # register basis as buffers (so .to(device) works automatically)
        self.register_buffer("_A1", _BASIS_A1.clone(), persistent=False)
        self.register_buffer("_AI", _BASIS_AI.clone(), persistent=False)
        self.register_buffer("_AJ", _BASIS_AJ.clone(), persistent=False)
        self.register_buffer("_AK", _BASIS_AK.clone(), persistent=False)

    def forward(self, x):
        W = (
            torch.kron(self._A1, self.R)
            + torch.kron(self._AI, self.I)
            + torch.kron(self._AJ, self.J)
            + torch.kron(self._AK, self.K)
        )
        return F.linear(x, W, self.bias)


def hamilton_product(p, q):
    """Hamilton product of two quaternion-structured tensors.

    p, q: shape [..., 4Q]
    Returns: shape [..., 4Q]

    Implemented as cross-component combinations of the 4 split parts
    (mathematically equivalent to (M(p) @ q.unsqueeze(-1)).squeeze(-1) on
    the per-Q-block components).
    """
    Q = p.shape[-1] // 4
    pr, pi, pj, pk = p[..., :Q], p[..., Q:2*Q], p[..., 2*Q:3*Q], p[..., 3*Q:]
    qr, qi, qj, qk = q[..., :Q], q[..., Q:2*Q], q[..., 2*Q:3*Q], q[..., 3*Q:]
    out_r = pr*qr - pi*qi - pj*qj - pk*qk
    out_i = pr*qi + pi*qr + pj*qk - pk*qj
    out_j = pr*qj - pi*qk + pj*qr + pk*qi
    out_k = pr*qk + pi*qj - pj*qi + pk*qr
    return torch.cat([out_r, out_i, out_j, out_k], dim=-1)


class QuaternionLayerNorm(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        assert d_model % 4 == 0
        self.Q = d_model // 4
        self.ln_r = nn.LayerNorm(self.Q)
        self.ln_i = nn.LayerNorm(self.Q)
        self.ln_j = nn.LayerNorm(self.Q)
        self.ln_k = nn.LayerNorm(self.Q)

    def forward(self, x):
        Q = self.Q
        return torch.cat([
            self.ln_r(x[..., :Q]),
            self.ln_i(x[..., Q:2*Q]),
            self.ln_j(x[..., 2*Q:3*Q]),
            self.ln_k(x[..., 3*Q:]),
        ], dim=-1)


# ============================================================================
# Spike Encoder: produces salience intensity ∈ [0, 1] per cell
# ============================================================================

class SpikeEncoder(nn.Module):
    """1-layer MLP + sigmoid; gated by mask so missing cells have spike=0."""
    def __init__(self, n_vars, hidden):
        super().__init__()
        self.var_emb = nn.Embedding(n_vars, hidden)
        nn.init.normal_(self.var_emb.weight, std=0.1)
        self.proj = nn.Linear(3 + hidden, 1)
        nn.init.zeros_(self.proj.bias)

    def forward(self, value, time_norm, mask, var_id):
        ve = self.var_emb(var_id)                            # [B,N,hidden]
        feats = torch.cat([
            (value * mask).unsqueeze(-1),
            time_norm.unsqueeze(-1),
            mask.unsqueeze(-1),
            ve,
        ], dim=-1)
        s = torch.sigmoid(self.proj(feats).squeeze(-1))
        return s * mask                                       # gate by mask


# ============================================================================
# Quaternion State Encoder: 4 typed components per cell
# ============================================================================

class CellEncoder(nn.Module):
    def __init__(self, d_model, n_vars):
        super().__init__()
        assert d_model % 4 == 0
        self.Q = d_model // 4
        self.value_proj = nn.Linear(2, self.Q)               # (value*mask, mask) -> r
        freqs = torch.exp(torch.linspace(0.0, math.log(50.0), self.Q))
        self.time_freq = nn.Parameter(freqs)                  # learnable freqs
        self.var_emb_j = nn.Embedding(n_vars, self.Q)        # j-component
        nn.init.normal_(self.var_emb_j.weight, std=0.1)
        self.spike_proj = nn.Linear(1, self.Q)                # spike-driven k
        self.mix = QuaternionLinear(d_model, d_model)

    def forward(self, value, time_norm, var_id, spike_intensity, mask):
        v_input = torch.stack([value * mask, mask], dim=-1)
        q_r = self.value_proj(v_input)
        t_phase = time_norm.unsqueeze(-1) * self.time_freq
        q_i = torch.sin(t_phase)
        q_j = self.var_emb_j(var_id)
        q_k = self.spike_proj(spike_intensity.unsqueeze(-1))
        q = torch.cat([q_r, q_i, q_j, q_k], dim=-1)
        return self.mix(q) * mask.unsqueeze(-1)


# ============================================================================
# STHQ Layer: cells -> hyperedges -> cells via spike-gated membership and
# Hamilton-product composition.
# ============================================================================

class STHQLayer(nn.Module):
    def __init__(self, d_model, n_vars, k_t, k_v,
                 omega_min, omega_max, layer_idx, n_layers,
                 use_he_attn=False):
        super().__init__()
        self.d_model = d_model
        self.k_t = k_t
        self.k_v = k_v
        self.layer_idx = layer_idx
        self.use_he_attn = use_he_attn

        # Layer-specific bandwidth range (multi-scale across layers)
        if n_layers > 1:
            log_min = math.log(omega_min)
            log_max = math.log(omega_max)
            t_layer = layer_idx / (n_layers - 1)
            this_log_center = log_min + t_layer * (log_max - log_min)
        else:
            this_log_center = 0.5 * (math.log(omega_min) + math.log(omega_max))
        # τ initialized uniformly in [0, 1]; τ are learnable
        tau_init = torch.linspace(0.0, 1.0, k_t)
        self.tau = nn.Parameter(tau_init)
        # ω initialized to layer-specific value, learnable
        omega_init = torch.full((k_t,), math.exp(this_log_center))
        self.omega_log = nn.Parameter(torch.log(omega_init))

        # Variable affinity: V × K_v matrix (logits, softmax over K_v)
        self.var_affinity = nn.Parameter(torch.zeros(n_vars, k_v))
        # initialize so that var v has slight preference for code v % k_v
        with torch.no_grad():
            for v in range(n_vars):
                self.var_affinity[v, v % k_v] = 1.0

        # Per-edge-type projection of hyperedge state (quaternion-aware)
        self.edge_proj_t = QuaternionLinear(d_model, d_model)
        self.edge_proj_v = QuaternionLinear(d_model, d_model)

        # Hybrid Hamilton + Linear message paths.
        #   msg_path_h : Hamilton-composed messages (typed-quaternion product)
        #   msg_path_l : direct linear messages (no quaternion structure)
        # Mix via learnable scalar α ∈ [0,1] per-layer.
        self.msg_proj_h = QuaternionLinear(d_model, d_model)
        self.msg_proj_l = nn.Linear(d_model, d_model)
        self.alpha_logit = nn.Parameter(torch.tensor(0.0))  # sigmoid(0)=0.5
        self.norm = QuaternionLayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

        # Optional hyperedge self-attention (deep layers)
        if use_he_attn:
            self.he_attn = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
            self.he_norm = QuaternionLayerNorm(d_model)

    def forward(self, q, spike_intensity, time_norm, var_id, mask):
        """
        q              : [B, N, 4Q]  cell state
        spike_intensity: [B, N]      salience in [0,1] (zero on query positions)
        time_norm      : [B, N]      time normalized to [0,1]
        var_id         : [B, N] long
        mask           : [B, N]      valid-position mask {0,1} (1 for both
                                     observations AND forecast queries)
        Returns: q' [B, N, 4Q]

        Membership matrices (separated to allow query cells to RECEIVE without
        contributing to hyperedge formation):
          - m_aggr (with spike): cell -> hyperedge (only observed cells with
            non-zero spike contribute to forming hyperedges)
          - m_dist (without spike): hyperedge -> cell (queries also receive
            messages based purely on their (time, var) coordinates)
        """
        B, N, D = q.shape
        # ---- kernels (position-only, no spike) ------------------------------
        omega = torch.exp(self.omega_log).clamp(min=1e-3)         # [K_t]
        diff = time_norm.unsqueeze(-1) - self.tau.view(1, 1, -1)   # [B, N, K_t]
        kernel_temp = torch.exp(-0.5 * (diff / omega.view(1, 1, -1)) ** 2)
        var_logits = self.var_affinity[var_id]                     # [B, N, K_v]
        kernel_var = F.softmax(var_logits, dim=-1)

        # ---- aggregation membership (cell -> hyperedge): spike-gated --------
        m_aggr_t = spike_intensity.unsqueeze(-1) * kernel_temp * mask.unsqueeze(-1)
        m_aggr_v = spike_intensity.unsqueeze(-1) * kernel_var  * mask.unsqueeze(-1)

        # ---- distribution membership (hyperedge -> cell): position-only -----
        # No spike factor — query positions receive messages too.
        m_dist_t = kernel_temp * mask.unsqueeze(-1)
        m_dist_v = kernel_var  * mask.unsqueeze(-1)

        # ---- aggregate cells -> hyperedges (B, K, D) ------------------------
        denom_t = m_aggr_t.sum(dim=1, keepdim=True).clamp_min(1e-6)
        h_temp = (m_aggr_t.transpose(1, 2) @ q) / denom_t.transpose(1, 2)
        denom_v = m_aggr_v.sum(dim=1, keepdim=True).clamp_min(1e-6)
        h_var = (m_aggr_v.transpose(1, 2) @ q) / denom_v.transpose(1, 2)

        # Per-edge-type quaternion projection
        h_temp = self.edge_proj_t(h_temp)
        h_var = self.edge_proj_v(h_var)

        # ---- optional hyperedge self-attention -------------------------------
        if self.use_he_attn:
            H = torch.cat([h_temp, h_var], dim=1)                 # [B, K_t+K_v, D]
            H_attn, _ = self.he_attn(H, H, H, need_weights=False)
            H_new = self.he_norm(H + H_attn)
            h_temp = H_new[:, :self.k_t]
            h_var = H_new[:, self.k_t:]

        # ---- distribute hyperedges -> cells via Hamilton product ------------
        # Cell-side normalization of distribution weights (rows sum to 1).
        # Hamilton bilinearity: Σ_k m[i,k] · (h_k ⊗ q[i]) = (Σ_k m[i,k] · h_k) ⊗ q[i]
        m_dist_t_norm = m_dist_t / m_dist_t.sum(dim=2, keepdim=True).clamp_min(1e-6)
        m_dist_v_norm = m_dist_v / m_dist_v.sum(dim=2, keepdim=True).clamp_min(1e-6)
        h_temp_per_cell = m_dist_t_norm @ h_temp                  # [B, N, D]
        h_var_per_cell = m_dist_v_norm @ h_var

        # Hybrid message: blend Hamilton-product path with linear path.
        h_per_cell = h_temp_per_cell + h_var_per_cell
        msg_h = hamilton_product(h_per_cell, q)        # [B, N, D] Hamilton path
        msg_l = h_per_cell                             # [B, N, D] linear path
        alpha = torch.sigmoid(self.alpha_logit)
        msg = alpha * self.msg_proj_h(msg_h) + (1 - alpha) * self.msg_proj_l(msg_l)
        msg = self.dropout(msg)
        q_new = self.norm(q + msg)
        return q_new * mask.unsqueeze(-1)


# ============================================================================
# Main Model
# ============================================================================

class Model(nn.Module):
    def __init__(self, configs: ExpConfigs):
        super().__init__()
        self.configs = configs
        D = (configs.d_model // 4) * 4
        if D != configs.d_model:
            logger.warning(f"STHQ: rounding d_model {configs.d_model} -> {D}")
        self.d_model = D
        self.n_layers = configs.n_layers
        self.enc_in = configs.enc_in

        self.k_t = int(getattr(configs, "sthq_k_t", 32))
        self.k_v = int(getattr(configs, "sthq_k_v", min(configs.enc_in, 32)))
        self.omega_min = float(getattr(configs, "sthq_omega_min", 0.02))
        self.omega_max = float(getattr(configs, "sthq_omega_max", 0.5))
        self.use_he_attn_from = int(getattr(configs, "sthq_use_he_attn_from_layer",
                                            max(1, self.n_layers // 2)))

        # ---- multi-scale per-layer K_t schedule ------------------------------
        # Comma-separated layer-specific K_t (e.g. "192,64,16" for n_layers=3).
        # Empty string → all layers share the global k_t.
        # Layer 0 sees the finest scale (most anchors, narrowest ω); deeper
        # layers progressively coarsen, providing global aggregation.
        kt_sched_str = str(getattr(configs, "sthq_k_t_per_layer", "")).strip()
        if kt_sched_str:
            kt_per_layer = [int(s) for s in kt_sched_str.split(",")]
            assert len(kt_per_layer) == self.n_layers, (
                f"sthq_k_t_per_layer has {len(kt_per_layer)} entries but "
                f"n_layers={self.n_layers}")
        else:
            kt_per_layer = [self.k_t] * self.n_layers
        self.kt_per_layer = kt_per_layer

        Q = D // 4
        self.spike_encoder = SpikeEncoder(configs.enc_in, hidden=Q)
        self.cell_encoder = CellEncoder(D, configs.enc_in)

        self.layers = nn.ModuleList()
        for l in range(self.n_layers):
            use_he_attn = (l >= self.use_he_attn_from)
            self.layers.append(STHQLayer(
                d_model=D,
                n_vars=configs.enc_in,
                k_t=kt_per_layer[l],
                k_v=self.k_v,
                omega_min=self.omega_min,
                omega_max=self.omega_max,
                layer_idx=l,
                n_layers=self.n_layers,
                use_he_attn=use_he_attn,
            ))

        # Time-aware decoder: explicitly sees query (time, var) embedding
        # plus skip connection from initial cell encoding.
        D_time_emb = 32
        self.dec_time_freqs = nn.Parameter(
            torch.exp(torch.linspace(0.0, math.log(50.0), D_time_emb // 2)))
        self.dec_var_emb = nn.Embedding(configs.enc_in, D_time_emb)
        nn.init.normal_(self.dec_var_emb.weight, std=0.1)
        dec_in_dim = D + D + D_time_emb * 2  # q_final + q_initial + [time_emb, var_emb]
        self.decoder = nn.Sequential(
            nn.Linear(dec_in_dim, D),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(D, D // 2),
            nn.GELU(),
            nn.Linear(D // 2, 1),
        )

        # Anti-collapse regularization weights
        self.lambda_tau = float(getattr(configs, "sthq_lambda_tau", 0.01))
        self.lambda_var = float(getattr(configs, "sthq_lambda_var", 0.005))

        # ---- diagnostic logging ----------------------------------------------
        # Print spike / α / τ / hyperedge usage every diag_interval forward
        # passes during training. Set to 0 to disable.
        self.diag_interval = int(getattr(configs, "sthq_diag_interval", 0))
        self.register_buffer("_diag_step", torch.zeros(1, dtype=torch.long),
                             persistent=False)

    # ---- diagnostics -----------------------------------------------------
    def _log_diagnostics(self, spike_intensity, contribute_mask):
        """Print spike / α / τ / ω / var-affinity entropy every diag_interval
        forward passes during training. Cheap (tensor reductions only)."""
        if self.diag_interval <= 0 or not self.training:
            return
        # increment in-place to keep on-device tracking
        self._diag_step += 1
        if (self._diag_step.item() - 1) % self.diag_interval != 0:
            return
        with torch.no_grad():
            active = contribute_mask.bool()
            if active.any():
                vs = spike_intensity[active]
                sm = vs.mean().item()
                ss = vs.std().item()
                sa = (vs > 0.5).float().mean().item()
            else:
                sm = ss = sa = 0.0
            layer_stats = []
            for l, layer in enumerate(self.layers):
                alpha = torch.sigmoid(layer.alpha_logit).item()
                tau_min = layer.tau.min().item()
                tau_max = layer.tau.max().item()
                tau_std = layer.tau.std().item()
                om = torch.exp(layer.omega_log).mean().item()
                va_sm = F.softmax(layer.var_affinity, dim=-1)
                ent = (-(va_sm * va_sm.clamp_min(1e-8).log()).sum(-1).mean()).item()
                layer_stats.append(
                    f"L{l}(K_t={layer.k_t}): α={alpha:.2f} "
                    f"τ∈[{tau_min:.2f},{tau_max:.2f}] τ_std={tau_std:.3f} "
                    f"ω̄={om:.3f} var_ent={ent:.2f}"
                )
        logger.debug(
            f"[STHQ diag step={self._diag_step.item()}] "
            f"spike: mean={sm:.3f} std={ss:.3f} active={sa:.3f}; "
            f"{' | '.join(layer_stats)}"
        )

    # ---- helpers (shared with PE-RQH structure) --------------------------
    def pad_and_flatten(self, tensor, mask, max_len):
        B = tensor.shape[0]
        tf = tensor.view(B, -1)
        mf = mask.view(B, -1).to(tensor.dtype)
        d = torch.cumsum(mf, 1) - 1
        k = (mf == 1) & (d < max_len)
        r = torch.zeros(B, max_len, dtype=tensor.dtype, device=tensor.device)
        rows = torch.arange(B, device=tensor.device).unsqueeze(1).expand_as(mf)
        r[rows[k], d[k].long()] = tf[k]
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
            x_mark = repeat(
                torch.arange(end=x.shape[1], dtype=x.dtype, device=x.device) / x.shape[1],
                "L -> B L 1", B=x.shape[0])
        if x_mask is None:
            x_mask = torch.ones_like(x, device=x.device, dtype=x.dtype)
        if y is None:
            y = torch.ones((BATCH_SIZE, Y_LEN, ENC_IN), dtype=x.dtype, device=x.device)
        if y_mark is None:
            y_mark = repeat(
                torch.arange(end=y.shape[1], dtype=y.dtype, device=y.device) / y.shape[1],
                "L -> B L 1", B=y.shape[0])
        if y_mask is None:
            y_mask = torch.ones_like(y, device=y.device, dtype=y.dtype)

        _, PRED_LEN, _ = y.shape
        L = SEQ_LEN + PRED_LEN
        x_mark = x_mark[:, :, :1]
        y_mark = y_mark[:, :, :1]

        if self.configs.task_name in [
            "short_term_forecast", "long_term_forecast",
            "classification", "representation_learning"
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

        times_flat = time_indices_flattened.float() / max(1, L - 1)
        var_ids_flat = variable_indices_flattened
        valid_mask = x_y_mask_flattened
        # contribute_mask excludes future targets (their values are unknown at input)
        contribute_mask = valid_mask * (1.0 - y_mask_L_flattened)

        # Spike intensity is computed only on contribute cells (mask gates)
        spike_intensity = self.spike_encoder(
            x_L_flattened, times_flat, contribute_mask, var_ids_flat)
        # Periodic diagnostics on spike / hyperedge state
        self._log_diagnostics(spike_intensity, contribute_mask)

        # Encode to quaternion state (uses contribute_mask for value gating)
        q_initial = self.cell_encoder(
            value=x_L_flattened,
            time_norm=times_flat,
            var_id=var_ids_flat,
            spike_intensity=spike_intensity,
            mask=valid_mask,
        )

        # STHQ layers — query positions still get to receive messages but
        # don't contribute to forming hyperedges (spike=0 for them).
        q = q_initial
        for layer in self.layers:
            q = layer(
                q=q,
                spike_intensity=spike_intensity,
                time_norm=times_flat,
                var_id=var_ids_flat,
                mask=valid_mask,
            )

        # Time-aware decoder: [q_final | q_initial | sin/cos(time) | var_emb]
        t_phase = times_flat.unsqueeze(-1) * self.dec_time_freqs  # [B, N, T_emb/2]
        time_emb = torch.cat([torch.sin(t_phase), torch.cos(t_phase)], dim=-1)
        var_emb = self.dec_var_emb(var_ids_flat)
        decoder_input = torch.cat([q, q_initial, time_emb, var_emb], dim=-1)
        pred_flattened = self.decoder(decoder_input).squeeze(-1)

        # ---- anti-collapse auxiliary loss ------------------------------------
        # τ repulsion: encourage temporal anchors to spread out within [0, 1]
        # Penalize pairs of τ's that are too close (< 0.5 * ω_mean).
        aux_loss = pred_flattened.new_zeros(())
        if self.training:
            for layer in self.layers:
                tau = layer.tau                                # [K_t]
                omega = torch.exp(layer.omega_log)             # [K_t]
                # Pairwise distances scaled by mean ω
                d_tau = (tau.unsqueeze(0) - tau.unsqueeze(1)).abs()
                mask_pair = ~torch.eye(tau.numel(), dtype=torch.bool, device=tau.device)
                min_spacing = 0.5 * omega.mean()
                repulse = F.relu(min_spacing - d_tau) ** 2
                aux_loss = aux_loss + self.lambda_tau * repulse[mask_pair].mean()
                # Variable affinity: penalize near-collapse of softmax (entropy bonus)
                var_softmax = F.softmax(layer.var_affinity, dim=-1)
                ent = -(var_softmax * (var_softmax.clamp_min(1e-8)).log()).sum(-1).mean()
                aux_loss = aux_loss - self.lambda_var * ent
        result = {
            "pred": pred_flattened,
            "true": y_L_flattened,
            "mask": y_mask_L_flattened,
            "aux_loss": aux_loss,
        }

        if self.configs.task_name in ['long_term_forecast', 'short_term_forecast', 'imputation']:
            if exp_stage in ["train", "val"]:
                return result
            else:
                pred_full = self.unpad_and_reshape(
                    pred_flattened, torch.cat([x_mask, y_mask], dim=1),
                    (BATCH_SIZE, SEQ_LEN + PRED_LEN, ENC_IN),
                )
                f_dim = -1 if self.configs.features == 'MS' else 0
                return {
                    "pred": pred_full[:, -PRED_LEN:, f_dim:],
                    "true": y[:, :, f_dim:],
                    "mask": y_mask[:, :, f_dim:],
                }
        else:
            raise NotImplementedError()
