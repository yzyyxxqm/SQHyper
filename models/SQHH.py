# SQHH: Spike-Quaternion Heterogeneous Hypergraph
#
# A novel hypergraph paradigm for IMTS forecasting where:
#   - Cells are quaternion-typed natively (R=value, I=time, J=variable, K=spike)
#   - Three heterogeneous hyperedge layers coexist:
#       L0 (Primary):  per-observation self-loop, guarantees no info loss
#       L1 (Event):    spike-triggered dynamic edges with refractory inhibition
#       L2 (Anchor):   K_a quaternion anchors with quaternion-distance incidence
#   - Spike-Refractory Activation (SRA) inspired by SNN refractory periods
#   - Spike-Quaternion Coupling (SQC): spike rotates cell quaternion towards
#     the K-axis ("event subspace") before message passing
#   - Quaternion Hypergraph Attention (QHA): Hamilton-product structured
#     Q/K/V projections, preserving cross-component coupling
#
# HyperIMTS / SQHyper are recovered as degenerate special cases:
#   - Single layer (L1+L2 disabled), hard incidence (no quat anchors), no SRA,
#     no SQC, dot-product attention with real-valued Linear projections.
#
# This file builds bottom-up: primitives -> SRA -> SQC -> Layer 0 -> Layer 2
# -> Layer 1 -> main Model. Each module carries a docstring with the math
# from docs/my_paper/SQHH_design.md.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat, rearrange
from torch import Tensor

from utils.ExpConfigs import ExpConfigs
from utils.globals import logger


# ============================================================================
# Quaternion primitives
# ============================================================================


class QuaternionLinear(nn.Module):
    """Hamilton-product-structured linear layer.

    For input/output dim 4Q, parameter count is 4 * Q * Q (1/4 of flat
    Linear). The block matrix W enforces that input components R/I/J/K
    interact through Hamilton multiplication rules:

        W = [[ R, -I, -J, -K],
             [ I,  R, -K,  J],
             [ J,  K,  R, -I],
             [ K, -J,  I,  R]]
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
        self.init_xavier()

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


def quaternion_split(q: Tensor):
    """Split a [..., 4Q] tensor into 4 tensors of [..., Q]."""
    Q = q.shape[-1] // 4
    return q[..., :Q], q[..., Q:2*Q], q[..., 2*Q:3*Q], q[..., 3*Q:]


def quaternion_concat(qr, qi, qj, qk):
    """Concatenate 4 [..., Q] tensors along the last dim into [..., 4Q]."""
    return torch.cat([qr, qi, qj, qk], dim=-1)


def hamilton_product(p: Tensor, q: Tensor) -> Tensor:
    """Element-wise Hamilton product. p, q: [..., 4Q] -> [..., 4Q]."""
    pr, pi, pj, pk = quaternion_split(p)
    qr, qi, qj, qk = quaternion_split(q)
    out_r = pr * qr - pi * qi - pj * qj - pk * qk
    out_i = pr * qi + pi * qr + pj * qk - pk * qj
    out_j = pr * qj - pi * qk + pj * qr + pk * qi
    out_k = pr * qk + pi * qj - pj * qi + pk * qr
    return quaternion_concat(out_r, out_i, out_j, out_k)


def quaternion_conjugate(q: Tensor) -> Tensor:
    """Negate i, j, k components: q* = (qr, -qi, -qj, -qk)."""
    qr, qi, qj, qk = quaternion_split(q)
    return quaternion_concat(qr, -qi, -qj, -qk)


def quaternion_norm_sq(q: Tensor) -> Tensor:
    """Squared norm summed over the 4Q dimensions: returns [..., 1]."""
    return q.pow(2).sum(dim=-1, keepdim=True)


def quaternion_distance(p: Tensor, q: Tensor) -> Tensor:
    """||p ⊗ conj(q)||² which captures both magnitude and rotation.

    For unit quaternions this is 2 - 2*<p, q> (twice the chordal distance).
    Returns [..., 1].
    """
    return quaternion_norm_sq(hamilton_product(p, quaternion_conjugate(q)))


def quaternion_exp_K(theta: Tensor, Q: int) -> Tensor:
    """exp(theta * ê_K) for the unit K-axis quaternion ê_K = (0, 0, 0, 1).

    Closed form:
        exp(theta * (0,0,0,1)) = cos(theta) + sin(theta) * (0,0,0,1)
                               = (cos(theta), 0, 0, sin(theta))

    Args:
        theta: [..., 1] or scalar — rotation angle
        Q: per-component dimension

    Returns: [..., 4Q] quaternion. The R-component is filled with cos(theta)
    repeated Q times, K-component with sin(theta) repeated Q times.
    """
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    zero = torch.zeros_like(cos_t)
    # Broadcast each component to Q dim
    cos_Q = cos_t.expand(*cos_t.shape[:-1], Q)
    sin_Q = sin_t.expand(*sin_t.shape[:-1], Q)
    zero_Q = zero.expand(*zero.shape[:-1], Q)
    return quaternion_concat(cos_Q, zero_Q, zero_Q, sin_Q)


class QuaternionLayerNorm(nn.Module):
    """Per-component LayerNorm (R, I, J, K each normalized independently)."""

    def __init__(self, d_model):
        super().__init__()
        assert d_model % 4 == 0
        self.Q = d_model // 4
        self.ln_r = nn.LayerNorm(self.Q)
        self.ln_i = nn.LayerNorm(self.Q)
        self.ln_j = nn.LayerNorm(self.Q)
        self.ln_k = nn.LayerNorm(self.Q)

    def forward(self, x):
        qr, qi, qj, qk = quaternion_split(x)
        return quaternion_concat(
            self.ln_r(qr), self.ln_i(qi), self.ln_j(qj), self.ln_k(qk)
        )


# ============================================================================
# Quaternion Hypergraph Attention (QHA)
# ============================================================================


class QuaternionMHA(nn.Module):
    """Quaternion-structured multi-head attention.

    All projections are QuaternionLinear (1/4 params of vanilla, preserves
    cross-component Hamilton coupling). Score is computed as the real part
    of the Hamilton product, which factors as:

        Re(Q_i ⊗ conj(K_j)) = Q_i^R·K_j^R + Q_i^I·K_j^I
                              + Q_i^J·K_j^J + Q_i^K·K_j^K

    Numerically equal to dot product, but the projections preserving Hamilton
    structure ensure that the four channels carry distinct semantics
    (value/time/variable/spike) throughout. The output projection is
    quaternion-typed.

    Optional `rotate_v` flag enables Hamilton-product correction on V before
    aggregation (rotates V[j] by the imaginary part of Q[i]⊗conj(K[j]) for
    each query-key pair). This is the QHA-distinct mechanism over standard
    quaternion attention. Defaults off because it's O(N²·D) memory.
    """

    def __init__(self, d_model: int, num_heads: int = 1, dropout: float = 0.0,
                 dot_product_only: bool = False):
        super().__init__()
        assert d_model % 4 == 0, "d_model must be ×4"
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        # Per-head dim must also be ×4 so each head is quaternion-typed
        head_dim = d_model // num_heads
        assert head_dim % 4 == 0, (
            f"per-head dim must be ×4, got d_model={d_model}, "
            f"num_heads={num_heads}, head_dim={head_dim}"
        )
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dot_product_only = bool(dot_product_only)

        self.fc_q = QuaternionLinear(d_model, d_model)
        self.fc_k = QuaternionLinear(d_model, d_model)
        self.fc_v = QuaternionLinear(d_model, d_model)
        self.fc_o = QuaternionLinear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(head_dim)

    def forward(self, query: Tensor, key: Tensor, value: Tensor,
                mask: Tensor | None = None):
        """
        query : [B, Nq, D]
        key   : [B, Nk, D]
        value : [B, Nk, D]
        mask  : [B, Nq, Nk] (1 for valid, 0 for masked) or None
        """
        B, Nq, D = query.shape
        Nk = key.shape[1]
        H, hd = self.num_heads, self.head_dim

        Q = self.fc_q(query).view(B, Nq, H, hd).transpose(1, 2)  # [B,H,Nq,hd]
        K = self.fc_k(key).view(B, Nk, H, hd).transpose(1, 2)
        V = self.fc_v(value).view(B, Nk, H, hd).transpose(1, 2)

        # Score = Re(Q ⊗ conj(K)) = standard dot product (per quaternion).
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B,H,Nq,Nk]
        if mask is not None:
            # mask: [B, Nq, Nk] -> [B, 1, Nq, Nk]
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)                                # [B,H,Nq,hd]
        out = out.transpose(1, 2).reshape(B, Nq, D)                # [B,Nq,D]
        return self.fc_o(out)


# ============================================================================
# Spike-Refractory Activation (SRA)
# ============================================================================


class SpikeRefractoryEncoder(nn.Module):
    """Spike encoder with SNN-inspired refractory inhibition.

    Forward pass:
        spike_raw[n] = sigmoid(SpikeMLP(value, time, mask, var_emb))
        refractory[n] = Σ_{m: var_m == var_n, t_m < t_n}
                        spike_eff[m] · exp(-(t_n - t_m) / tau_r)
        spike_eff[n] = max(floor, spike_raw[n] · (1 - tanh(alpha · refractory[n])))

    Implementation:
        - We approximate refractory by computing the per-variable decayed sum
          using a single matrix multiplication per variable id, masked to
          earlier time only.
        - This is O(N²) per batch but with small constants since N is the
          flattened observation count.
    """

    def __init__(self, n_vars: int, d_model: int, hidden: int | None = None,
                 floor: float = 0.0, tau_r_init: float = 0.1,
                 alpha_init: float = 1.0, no_refractory: bool = False):
        super().__init__()
        H = hidden if hidden is not None else max(16, d_model // 4)
        self.var_emb = nn.Embedding(n_vars, H)
        nn.init.normal_(self.var_emb.weight, std=0.1)
        self.body = nn.Sequential(
            nn.Linear(3 + H, H), nn.GELU(), nn.Linear(H, H),
        )
        self.gate_head = nn.Linear(H, 1)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(self.gate_head.bias, 2.0)  # sigmoid(2) ≈ 0.88
        # Per-variable refractory time-constant tau_r and inhibition alpha.
        # Stored as log so they remain positive after exp().
        self.log_tau_r = nn.Parameter(
            torch.full((n_vars,), math.log(tau_r_init)))
        self.log_alpha = nn.Parameter(
            torch.full((n_vars,), math.log(alpha_init)))
        self.floor = float(floor)
        self.no_refractory = bool(no_refractory)

    def forward(self, value: Tensor, time_norm: Tensor, mask: Tensor,
                var_id: Tensor):
        """
        value     : [B, N]
        time_norm : [B, N] in [0, 1]
        mask      : [B, N] (1 for observed, 0 for padding)
        var_id    : [B, N] long
        Returns:  spike_eff [B, N] in [floor, 1] for observed cells, 0 padding
        """
        B, N = value.shape
        ve = self.var_emb(var_id)
        feats = torch.cat([
            (value * mask).unsqueeze(-1),
            time_norm.unsqueeze(-1),
            mask.unsqueeze(-1),
            ve,
        ], dim=-1)
        h = self.body(feats)
        spike_raw = torch.sigmoid(self.gate_head(h).squeeze(-1)) * mask

        if self.no_refractory:
            spike_eff = self.floor + (1.0 - self.floor) * spike_raw
            return spike_eff * mask

        # Per-cell tau_r and alpha looked up by variable id
        tau_r = torch.exp(self.log_tau_r)[var_id]      # [B, N]
        alpha = torch.exp(self.log_alpha)[var_id]      # [B, N]

        # Build pair masks: same-variable, earlier time
        var_eq = (var_id.unsqueeze(2) == var_id.unsqueeze(1))    # [B, N, N]
        time_diff = time_norm.unsqueeze(2) - time_norm.unsqueeze(1)  # [B,N,N]
        earlier = (time_diff > 0).to(value.dtype)              # m before n
        valid_pair = (mask.unsqueeze(2) * mask.unsqueeze(1))   # both observed
        pair_mask = var_eq.float() * earlier * valid_pair      # [B, N, N]

        # Decay kernel: exp(-(t_n - t_m) / tau_r[var_n])
        # tau_r per RECEIVING cell (row n) — broadcast over m
        decay = torch.exp(
            -time_diff.clamp_min(0) / tau_r.unsqueeze(2).clamp_min(1e-6)
        )

        # Refractory sum at cell n is over m
        # spike_raw_m: [B, N] -> [B, 1, N]
        refractory_input = spike_raw.unsqueeze(1)              # [B, 1, N]
        refractory = (pair_mask * decay * refractory_input).sum(dim=-1)  # [B,N]

        inhibit = torch.tanh(alpha * refractory)               # [B, N] in [0,1]
        spike_modulated = spike_raw * (1.0 - inhibit)
        spike_eff = self.floor + (1.0 - self.floor) * spike_modulated
        return spike_eff * mask


# ============================================================================
# Spike-Quaternion Coupling (SQC) rotation
# ============================================================================


class SpikeQuaternionRotation(nn.Module):
    """Rotate cell quaternion towards the K-axis by a spike-driven angle.

        rot[n]   = exp(theta_n · ê_K) = (cos θ_n, 0, 0, sin θ_n)
        q_n_rot  = rot[n] ⊗ q_n
        theta_n  = theta_max · spike[n]   (theta_max learnable in [0, π/2])

    High-spike cells are rotated up to theta_max towards the K-axis,
    bringing them into an "event subspace" that downstream layers can attend
    to differentially.
    """

    def __init__(self, theta_max_init: float = math.pi / 4):
        super().__init__()
        # Parameterize theta_max via sigmoid so it stays in (0, π/2)
        # raw = 0 -> sigmoid(0)=0.5 -> theta_max = π/4
        init_logit = math.log(
            (2.0 * theta_max_init / math.pi)
            / (1.0 - 2.0 * theta_max_init / math.pi)
        )
        self.theta_max_logit = nn.Parameter(torch.tensor(init_logit))

    def forward(self, q: Tensor, spike: Tensor):
        """
        q     : [B, N, 4Q]
        spike : [B, N]
        """
        Q = q.shape[-1] // 4
        theta_max = (math.pi / 2.0) * torch.sigmoid(self.theta_max_logit)
        theta = theta_max * spike.unsqueeze(-1)                    # [B, N, 1]
        rot = quaternion_exp_K(theta, Q)                           # [B, N, 4Q]
        return hamilton_product(rot, q)


# ============================================================================
# Cell Encoder: produces quaternion cells with R/I/J/K = value/time/var/spike
# ============================================================================


class QuaternionCellEncoder(nn.Module):
    """Initialize each cell's quaternion state with semantic component split.

    R-channel (value):     MLP on (value, mask) → ℝ^Q
    I-channel (time):      sin(t · ω) ⊕ cos(t · ω), Q learnable freqs
    J-channel (variable):  Variable embedding ℝ^Q
    K-channel (spike):     MLP on spike intensity ℝ^Q

    Then a single QuaternionLinear mixes all four (preserves typing).
    """

    def __init__(self, d_model: int, n_vars: int):
        super().__init__()
        assert d_model % 4 == 0
        self.Q = d_model // 4
        Q = self.Q
        self.value_proj = nn.Linear(2, Q)
        # Time frequencies log-spaced from 1 to ~50
        freqs = torch.exp(torch.linspace(0.0, math.log(50.0), Q // 2 if Q > 1 else 1))
        if Q % 2 == 1:
            freqs = torch.cat([freqs, freqs[:1]])
        self.time_freq = nn.Parameter(freqs[:Q])
        self.var_emb = nn.Embedding(n_vars, Q)
        nn.init.normal_(self.var_emb.weight, std=0.1)
        self.spike_proj = nn.Linear(1, Q)
        self.mixer = QuaternionLinear(d_model, d_model)
        self.mixer.init_identity()  # start as identity, learn to mix

    def forward(self, value: Tensor, time_norm: Tensor, var_id: Tensor,
                spike: Tensor, mask: Tensor):
        # R: value
        r = self.value_proj(torch.stack([value * mask, mask], dim=-1))
        # I: time phases — half sin / half cos to fill Q dim
        t_phase = time_norm.unsqueeze(-1) * self.time_freq        # [B,N,Q]
        Q = self.Q
        half = Q // 2
        if half > 0:
            i_part = torch.cat(
                [torch.sin(t_phase[..., :half]),
                 torch.cos(t_phase[..., half:half * 2])],
                dim=-1,
            )
            if Q % 2 == 1:
                i_part = torch.cat(
                    [i_part, torch.sin(t_phase[..., -1:])], dim=-1)
        else:
            i_part = torch.sin(t_phase)
        # Pad/truncate to Q
        if i_part.shape[-1] != Q:
            i_part = i_part[..., :Q] if i_part.shape[-1] > Q \
                else F.pad(i_part, (0, Q - i_part.shape[-1]))
        # J: variable
        j = self.var_emb(var_id)
        # K: spike
        k = self.spike_proj(spike.unsqueeze(-1))
        q = quaternion_concat(r, i_part, j, k)
        return self.mixer(q) * mask.unsqueeze(-1)


# ============================================================================
# Layer 0 — Primary Edges (per-observation self-loop)
# ============================================================================


class PrimaryLayer(nn.Module):
    """Per-cell quaternion residual transform.

    Each cell n has its own self-loop edge e_n = {n}. The "message" from
    this edge is simply a learned quaternion transform of q_n itself.
    Guarantees no per-observation information is lost regardless of how
    Layer 1/2 cluster cells.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.proj = QuaternionLinear(d_model, d_model)

    def forward(self, q_rot: Tensor) -> Tensor:
        return self.proj(q_rot)


# ============================================================================
# Layer 2 — Quaternion Anchor Edges (global pattern)
# ============================================================================


class QuaternionAnchorLayer(nn.Module):
    """K_a learnable quaternion anchors with quaternion-distance incidence.

    Forward:
        d[k, n]   = ||q_n_rot ⊗ conj(Q_k)||²              # quaternion distance
        I[k, n]   = exp(-d[k, n] / σ_k²) · mask[n]
        h_k       = QLin_aggr( Σ_n I[k, n]·q_n_rot / Σ I + ε )
        msg_n     = QHA(query=q_n_rot, key=h_k, value=h_k)  # Hamilton attention
    """

    def __init__(self, d_model: int, k_a: int, num_heads: int = 1):
        super().__init__()
        self.d_model = d_model
        self.k_a = k_a
        # Anchors: random init, will learn
        self.anchors = nn.Parameter(torch.empty(k_a, d_model))
        nn.init.normal_(self.anchors, std=0.5)
        self.log_sigma = nn.Parameter(torch.zeros(k_a))  # init σ=1
        self.aggr_proj = QuaternionLinear(d_model, d_model)
        self.attn = QuaternionMHA(d_model, num_heads=num_heads)

    def forward(self, q_rot: Tensor, mask: Tensor) -> Tensor:
        """
        q_rot : [B, N, D] quaternion cell states (post-SQC)
        mask  : [B, N]
        Returns: msg [B, N, D]
        """
        B, N, D = q_rot.shape
        K_a = self.k_a
        # Broadcast anchors over batch: [B, K_a, D]
        anchors_b = self.anchors.unsqueeze(0).expand(B, K_a, D)

        # Compute pairwise quaternion distance: [B, K_a, N]
        # ||q_n ⊗ conj(Q_k)||² = ||q_n||² + ||Q_k||² - 2·Re(q_n·conj(Q_k))
        # = ||q||² + ||Q||² - 2·dot(q, Q) (since Re of Hamilton equals dot)
        # We'll use the explicit formula via Hamilton product to keep
        # quaternion semantics in code.
        # Memory: O(B * K_a * N * D). For HA N=12k, K_a=16, D=128 -> 25M floats. OK.
        q_exp = q_rot.unsqueeze(1).expand(B, K_a, N, D)
        a_exp = anchors_b.unsqueeze(2).expand(B, K_a, N, D)
        prod = hamilton_product(q_exp, quaternion_conjugate(a_exp))
        d = prod.pow(2).sum(dim=-1)                                 # [B,K_a,N]

        sigma_sq = torch.exp(self.log_sigma).pow(2).clamp_min(1e-6)  # [K_a]
        incid = torch.exp(-d / sigma_sq.view(1, K_a, 1))             # [B,K_a,N]
        incid = incid * mask.unsqueeze(1)

        # Aggregate cells -> anchors (weighted mean of quaternion states)
        denom = incid.sum(dim=-1, keepdim=True).clamp_min(1e-6)      # [B,K_a,1]
        h = torch.bmm(incid, q_rot) / denom                          # [B,K_a,D]
        h = self.aggr_proj(h)

        # Hamilton-product attention back to cells
        msg = self.attn(query=q_rot, key=h, value=h)                 # [B,N,D]
        return msg


# ============================================================================
# Layer 1 — Spike-Triggered Event Edges (dynamic)
# ============================================================================


class SpikeTriggeredEventLayer(nn.Module):
    """Top-K spike-triggered dynamic hyperedges with windowed incidence.

    Per forward pass:
        1. trigger_idx = topk(spike_eff·mask, K_e)
        2. event edge h_k seeded from QLin(q_{trigger_k}_rot)
        3. aggregate from cells in (t_n - t_k) ∈ [0, Δt_aggr] window,
           weighted by spike[n] · exp(-(t_n-t_k)/τ_r)
        4. distribute back to cells in (|t_n - t_k|) ≤ Δt_dist window
           via Hamilton-product attention
    """

    def __init__(self, d_model: int, k_e: int = 32,
                 dt_aggr: float = 0.05, dt_dist: float = 0.1,
                 num_heads: int = 1):
        super().__init__()
        self.d_model = d_model
        self.k_e = k_e
        # Window sizes (relative to normalized time in [0,1]); learnable
        self.log_dt_aggr = nn.Parameter(torch.tensor(math.log(dt_aggr)))
        self.log_dt_dist = nn.Parameter(torch.tensor(math.log(dt_dist)))
        self.seed_proj = QuaternionLinear(d_model, d_model)
        self.aggr_proj = QuaternionLinear(d_model, d_model)
        self.attn = QuaternionMHA(d_model, num_heads=num_heads)

    def forward(self, q_rot: Tensor, spike: Tensor, time_norm: Tensor,
                var_id: Tensor, mask: Tensor) -> Tensor:
        """
        q_rot     : [B, N, D]
        spike     : [B, N]
        time_norm : [B, N]
        var_id    : [B, N]
        mask      : [B, N]
        Returns:  msg [B, N, D]
        """
        B, N, D = q_rot.shape
        if N == 0 or self.k_e == 0:
            return torch.zeros_like(q_rot)
        K_e = min(self.k_e, N)

        # --- 1. Trigger selection: top-K_e cells by spike (excluding padding)
        spike_for_topk = spike * mask                          # padding gets 0
        _, idx = torch.topk(spike_for_topk, K_e, dim=1)        # [B, K_e]

        # Gather trigger info
        gather_idx_d = idx.unsqueeze(-1).expand(B, K_e, D)
        q_seed = torch.gather(q_rot, 1, gather_idx_d)          # [B, K_e, D]
        t_seed = torch.gather(time_norm, 1, idx)               # [B, K_e]
        var_seed = torch.gather(var_id, 1, idx)                # [B, K_e]

        # Initial event-edge state
        h_event = self.seed_proj(q_seed)                       # [B, K_e, D]

        # --- 2. Aggregation incidence: cells -> events
        # Window: t_n - t_k ∈ [-dt_aggr/2, +dt_aggr/2]
        # (We use absolute time difference so events grab both directions.)
        dt_aggr = torch.exp(self.log_dt_aggr).clamp(min=1e-3, max=1.0)
        t_n = time_norm.unsqueeze(1)                           # [B, 1, N]
        t_k = t_seed.unsqueeze(2)                              # [B, K_e, 1]
        delta = (t_n - t_k).abs()                              # [B, K_e, N]
        in_window_aggr = (delta <= dt_aggr).float()
        # Decay kernel; tau ≈ dt_aggr / 2 makes weight small at window edge
        decay_aggr = torch.exp(-2.0 * delta / dt_aggr.clamp_min(1e-3))
        # Variable affinity: same-variable boost (1.0); cross-variable gets 0.5
        same_var = (var_id.unsqueeze(1) == var_seed.unsqueeze(2)).float()
        var_aff = 0.5 + 0.5 * same_var                          # [B, K_e, N]
        # Spike-weighted incidence
        incid_aggr = (
            in_window_aggr * decay_aggr * var_aff
            * spike.unsqueeze(1) * mask.unsqueeze(1)
        )                                                        # [B, K_e, N]

        denom = incid_aggr.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        h_event = h_event + self.aggr_proj(
            torch.bmm(incid_aggr, q_rot) / denom
        )

        # --- 3. Distribution back to cells via Hamilton-product attention
        dt_dist = torch.exp(self.log_dt_dist).clamp(min=1e-3, max=1.0)
        in_window_dist = (delta <= dt_dist).float()              # [B,K_e,N]
        # Distribution mask must match attention mask shape [B, Nq, Nk]
        # Here Nq = N (cells), Nk = K_e (events)
        attn_mask = in_window_dist.transpose(1, 2)               # [B, N, K_e]
        attn_mask = attn_mask * mask.unsqueeze(-1)               # mask out padding

        msg = self.attn(query=q_rot, key=h_event, value=h_event,
                        mask=attn_mask)
        return msg


# ============================================================================
# SQHH Block: 3-layer message passing per "round" of hypergraph reasoning
# ============================================================================


class SQHHBlock(nn.Module):
    """One SQHH block: SQC rotation -> L0 + L1 + L2 -> mix -> norm + residual."""

    def __init__(self, d_model: int, n_vars: int, k_a: int, k_e: int,
                 num_heads: int = 1, no_layer0: bool = False,
                 no_layer1: bool = False, no_layer2: bool = False,
                 no_sqc: bool = False):
        super().__init__()
        self.no_layer0 = no_layer0
        self.no_layer1 = no_layer1
        self.no_layer2 = no_layer2
        self.no_sqc = no_sqc

        if not no_sqc:
            self.sqc = SpikeQuaternionRotation()
        if not no_layer0:
            self.layer0 = PrimaryLayer(d_model)
        if not no_layer1:
            self.layer1 = SpikeTriggeredEventLayer(d_model, k_e=k_e,
                                                    num_heads=num_heads)
        if not no_layer2:
            self.layer2 = QuaternionAnchorLayer(d_model, k_a=k_a,
                                                 num_heads=num_heads)

        # Per-layer mixing weights (softplus to stay positive)
        self.w0 = nn.Parameter(torch.tensor(0.5))
        self.w1 = nn.Parameter(torch.tensor(0.5))
        self.w2 = nn.Parameter(torch.tensor(0.5))
        self.norm = QuaternionLayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, q: Tensor, spike: Tensor, time_norm: Tensor,
                var_id: Tensor, mask: Tensor) -> Tensor:
        # SQC rotation (or identity if disabled)
        q_rot = self.sqc(q, spike) if not self.no_sqc else q

        # Per-layer messages
        msgs = []
        weights = []
        if not self.no_layer0:
            msgs.append(self.layer0(q_rot))
            weights.append(F.softplus(self.w0))
        if not self.no_layer1:
            msgs.append(self.layer1(q_rot, spike, time_norm, var_id, mask))
            weights.append(F.softplus(self.w1))
        if not self.no_layer2:
            msgs.append(self.layer2(q_rot, mask))
            weights.append(F.softplus(self.w2))

        if not msgs:
            # All layers ablated; just return q (degenerate identity block)
            return q

        # Weighted sum
        msg_total = sum(w * m for w, m in zip(weights, msgs))
        msg_total = self.dropout(msg_total)
        return self.norm(q + msg_total) * mask.unsqueeze(-1)


# ============================================================================
# Main SQHH Model
# ============================================================================


class Model(nn.Module):
    """SQHH-Net: Spike-Quaternion Heterogeneous Hypergraph for IMTS."""

    def __init__(self, configs: ExpConfigs):
        super().__init__()
        self.configs = configs
        D = (configs.d_model // 4) * 4
        if D != configs.d_model:
            logger.warning(f"SQHH: d_model {configs.d_model} -> {D} (×4)")
        self.d_model = D
        self.enc_in = configs.enc_in
        self.n_layers = configs.n_layers

        sl = configs.seq_len_max_irr or configs.seq_len
        pl = configs.pred_len_max_irr or configs.pred_len
        self.L_total = sl + pl

        # Hyperparameters
        self.k_a = int(getattr(configs, "sqhh_k_a", 16))
        self.k_e = int(getattr(configs, "sqhh_k_e", 32))
        self.no_layer0 = bool(int(getattr(configs, "sqhh_no_layer0", 0)))
        self.no_layer1 = bool(int(getattr(configs, "sqhh_no_layer1", 0)))
        self.no_layer2 = bool(int(getattr(configs, "sqhh_no_layer2", 0)))
        self.no_sra = bool(int(getattr(configs, "sqhh_no_sra", 0)))
        self.no_sqc = bool(int(getattr(configs, "sqhh_no_sqc", 0)))
        spike_floor = float(getattr(configs, "sqhh_spike_floor", 0.0))

        # Encoder
        self.spike_encoder = SpikeRefractoryEncoder(
            n_vars=self.enc_in, d_model=D, floor=spike_floor,
            no_refractory=self.no_sra,
        )
        self.cell_encoder = QuaternionCellEncoder(
            d_model=D, n_vars=self.enc_in)

        # Stack of SQHH blocks
        self.blocks = nn.ModuleList([
            SQHHBlock(
                d_model=D, n_vars=self.enc_in,
                k_a=self.k_a, k_e=self.k_e,
                num_heads=configs.n_heads,
                no_layer0=self.no_layer0,
                no_layer1=self.no_layer1,
                no_layer2=self.no_layer2,
                no_sqc=self.no_sqc,
            )
            for _ in range(self.n_layers)
        ])

        # Time-aware decoder (re-uses the STHQ v6 design — vanilla downstream)
        D_time_emb = 32
        self.dec_time_freqs = nn.Parameter(
            torch.exp(torch.linspace(0.0, math.log(50.0), D_time_emb // 2)))
        self.dec_var_emb = nn.Embedding(self.enc_in, D_time_emb)
        nn.init.normal_(self.dec_var_emb.weight, std=0.1)
        dec_in_dim = D + D + D_time_emb * 2
        self.decoder = nn.Sequential(
            nn.Linear(dec_in_dim, D), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(D, D // 2), nn.GELU(),
            nn.Linear(D // 2, 1),
        )

        # Diagnostic
        self.diag_interval = int(getattr(configs, "sqhh_diag_interval", 0))
        self.register_buffer(
            "_diag_step", torch.zeros(1, dtype=torch.long), persistent=False)

    # ---- helpers (shared with HyperIMTS / SQHyper / STHQ) ---------------

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
        counts = original_mask.sum(dim=tuple(range(1, original_mask.dim())))
        batch_size, max_len = tensor_flattened.shape[:2]
        steps = torch.arange(max_len, device=device).expand(batch_size, max_len)
        src_mask = steps < counts.unsqueeze(-1)
        result[original_mask] = tensor_flattened[src_mask]
        return result

    # ---- forward --------------------------------------------------------

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
            x_L = torch.cat([x, x_zeros], dim=1)
            x_y_mask = torch.cat([x_mask, y_mask], dim=1)
            y_L = torch.cat([y_zeros, y], dim=1)
            y_mask_L = torch.cat([y_zeros, y_mask], dim=1)
        elif self.configs.task_name in ["imputation"]:
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

        time_norm = time_indices_flattened.float() / max(1, L - 1)
        var_id = variable_indices_flattened
        mask = x_y_mask_flattened
        # Contribute mask: query positions don't form events (their value = 0)
        contribute_mask = mask * (1.0 - y_mask_L_flattened)

        # Compute spike (with SRA refractory) only on contributing cells
        spike = self.spike_encoder(
            value=x_L_flattened, time_norm=time_norm,
            mask=contribute_mask, var_id=var_id)
        # But for the K-component encoding, queries get 0 spike naturally

        # Encode quaternion cells
        q_init = self.cell_encoder(
            value=x_L_flattened, time_norm=time_norm, var_id=var_id,
            spike=spike, mask=mask)

        # SQHH blocks
        q = q_init
        for blk in self.blocks:
            q = blk(q, spike, time_norm, var_id, mask)

        # Time-aware decoder
        t_phase = time_norm.unsqueeze(-1) * self.dec_time_freqs
        time_emb = torch.cat([torch.sin(t_phase), torch.cos(t_phase)], dim=-1)
        var_emb = self.dec_var_emb(var_id)
        decoder_input = torch.cat([q, q_init, time_emb, var_emb], dim=-1)
        pred_flattened = self.decoder(decoder_input).squeeze(-1)

        # Optional diagnostic
        if self.training and self.diag_interval > 0:
            self._diag_step += 1
            if int(self._diag_step.item()) % self.diag_interval == 0:
                with torch.no_grad():
                    sm = float(spike[mask > 0].mean().item()) if mask.sum() > 0 else 0.0
                    ss = float(spike[mask > 0].std().item()) if mask.sum() > 0 else 0.0
                    sa = float((spike > 0.5).float().mean().item())
                    logger.debug(
                        f"[SQHH diag step={int(self._diag_step.item())}] "
                        f"spike: mean={sm:.3f} std={ss:.3f} active={sa:.3f}; "
                        f"K_a={self.k_a} K_e={self.k_e}"
                    )

        if self.configs.task_name in [
            "long_term_forecast", "short_term_forecast", "imputation"
        ]:
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
