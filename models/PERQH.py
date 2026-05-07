# PE-RQH: Pure Event-Routed Quaternion Hypergraph
#
# A genuinely new IMTS forecasting model, designed independently of HyperIMTS.
#
# Key characteristics (vs HyperIMTS):
#   - No predefined time/variable hyperedges. Hyperedges are entirely emergent
#     via a learned VQ codebook.
#   - Quaternion-native node states throughout (4 components: value, time-i,
#     time-j, variable). All updates use Hamilton products.
#   - Pure routed message passing: each obs joins top-m codes; each code's
#     prototype is a weighted quaternion mean of its members; node state is
#     updated by Hamilton-product routing q ⊗ prototype.
#   - Query-driven decode: predicted cells participate in routing via their
#     time/variable encoding (no value), then quaternion → real via MLP.
#
# Hyperparameters (all overridable via config / cli):
#   d_model       : must be divisible by 4
#   n_layers      : number of message-passing layers
#   perqh_n_codes : codebook size K (default 64)
#   perqh_top_m   : top-m soft assignment per obs (default 3)
#   perqh_tau     : Gumbel-softmax temperature (annealed during training)
#   perqh_lambda_div    : diversity loss weight
#   perqh_lambda_commit : commitment loss weight

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat, rearrange
from utils.ExpConfigs import ExpConfigs
from utils.globals import logger


# ============================================================================
# Quaternion primitives
# ============================================================================

def hamilton_product(p, q):
    """Element-wise Hamilton product on stacked quaternions.

    p, q: [..., D] where D = 4 * Q. Components ordered as (r, i, j, k).
    Returns: [..., D]
    """
    Q = p.shape[-1] // 4
    pr, pi, pj, pk = p[..., :Q], p[..., Q:2*Q], p[..., 2*Q:3*Q], p[..., 3*Q:]
    qr, qi, qj, qk = q[..., :Q], q[..., Q:2*Q], q[..., 2*Q:3*Q], q[..., 3*Q:]
    out_r = pr*qr - pi*qi - pj*qj - pk*qk
    out_i = pr*qi + pi*qr + pj*qk - pk*qj
    out_j = pr*qj - pi*qk + pj*qr + pk*qi
    out_k = pr*qk + pi*qj - pj*qi + pk*qr
    return torch.cat([out_r, out_i, out_j, out_k], dim=-1)


class QuaternionLinear(nn.Module):
    """Quaternion linear layer: weight matrix is structured as Hamilton block."""
    def __init__(self, in_f, out_f, bias=True):
        super().__init__()
        assert in_f % 4 == 0 and out_f % 4 == 0
        qi, qo = in_f // 4, out_f // 4
        self.r = nn.Parameter(torch.empty(qo, qi))
        self.i = nn.Parameter(torch.empty(qo, qi))
        self.j = nn.Parameter(torch.empty(qo, qi))
        self.k = nn.Parameter(torch.empty(qo, qi))
        self.bias = nn.Parameter(torch.zeros(out_f)) if bias else None
        for p in [self.r, self.i, self.j, self.k]:
            nn.init.xavier_uniform_(p, gain=0.5)

    def forward(self, x):
        r, i, j, k = self.r, self.i, self.j, self.k
        W = torch.cat([
            torch.cat([r, -i, -j, -k], 1),
            torch.cat([i,  r, -k,  j], 1),
            torch.cat([j,  k,  r, -i], 1),
            torch.cat([k, -j,  i,  r], 1),
        ], 0)
        return F.linear(x, W, self.bias)


class QuaternionLayerNorm(nn.Module):
    """LayerNorm applied independently on each quaternion component."""
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
        r = self.ln_r(x[..., :Q])
        i = self.ln_i(x[..., Q:2*Q])
        j = self.ln_j(x[..., 2*Q:3*Q])
        k = self.ln_k(x[..., 3*Q:])
        return torch.cat([r, i, j, k], dim=-1)


# ============================================================================
# Quaternion Encoder: (value, time, var_id, mask) → quaternion
# ============================================================================

class QuaternionEncoder(nn.Module):
    """Encode each observation/query cell as a quaternion node.

    Component layout (initially):
      r: value channel  (Linear(2 → Q) on [value*mask, mask])
      i: sin(t · ω)     (learnable frequencies)
      j: cos(t · ω)
      k: variable embedding
    Then mixed via QuaternionLinear so the model can rotate among components.
    """
    def __init__(self, d_model, n_vars):
        super().__init__()
        assert d_model % 4 == 0
        Q = d_model // 4
        self.Q = Q
        self.value_proj = nn.Linear(2, Q)
        # Learnable frequencies for time encoding
        # Initialize with log-spaced frequencies covering [1, 100]
        freqs = torch.exp(torch.linspace(0.0, math.log(100.0), Q))
        self.time_freq = nn.Parameter(freqs)
        self.var_emb = nn.Embedding(n_vars, Q)
        nn.init.normal_(self.var_emb.weight, std=0.1)
        self.q_mix = QuaternionLinear(d_model, d_model)

    def forward(self, values, times, var_ids, mask):
        """
        values: [B, N] float
        times:  [B, N] float in [0, 1] typically
        var_ids: [B, N] long
        mask:   [B, N] float (1 valid, 0 padding-or-future)

        Returns: q [B, N, D]
        """
        # value channel: gated value plus mask flag
        v_input = torch.stack([values * mask, mask], dim=-1)  # [B, N, 2]
        q_r = self.value_proj(v_input)                          # [B, N, Q]
        # time channel
        t_phase = times.unsqueeze(-1) * self.time_freq          # [B, N, Q]
        q_i = torch.sin(t_phase)
        q_j = torch.cos(t_phase)
        # variable channel
        q_k = self.var_emb(var_ids)                             # [B, N, Q]
        q_seed = torch.cat([q_r, q_i, q_j, q_k], dim=-1)        # [B, N, D]
        return self.q_mix(q_seed)


# ============================================================================
# VQ Codebook: top-m soft assignment with Gumbel-softmax
# ============================================================================

class VQCodebook(nn.Module):
    """A learned codebook of quaternion event prototypes."""
    def __init__(self, n_codes, d_model):
        super().__init__()
        self.n_codes = n_codes
        self.d_model = d_model
        # Init codebook with small random values
        self.codes = nn.Parameter(torch.randn(n_codes, d_model) / math.sqrt(d_model))

    def assign(self, q, mask, top_m=3, tau=1.0, training=True):
        """Assign each obs to top-m codes.

        q: [B, N, D]
        mask: [B, N]   (1 valid, 0 padding)

        Returns:
          idx: [B, N, m]  long
          w:   [B, N, m]  float (sum to 1 over m, zero where mask=0)
          sim_full: [B, N, K]  full similarity matrix (for diversity loss)
        """
        B, N, D = q.shape
        K = self.n_codes
        sim = q @ self.codes.t()                                  # [B, N, K]
        sim_masked = sim.masked_fill(mask.unsqueeze(-1) == 0, -1e4)
        # Top-m selection
        top_vals, top_idx = sim_masked.topk(top_m, dim=-1)         # [B, N, m]
        # Soft weights via softmax (or Gumbel during training)
        if training:
            # Gumbel noise on top-m logits for exploration.
            # Formula: g = -log(-log(u)) for u ~ U(0,1).
            # Use parentheses + explicit clamps to keep arguments strictly positive.
            u = torch.rand_like(top_vals).clamp(min=1e-7, max=1.0 - 1e-7)
            inner = (-torch.log(u)).clamp_min(1e-7)
            gumbel = -torch.log(inner)
            top_logits = (top_vals + gumbel) / max(tau, 1e-3)
        else:
            top_logits = top_vals / max(tau, 1e-3)
        weights = F.softmax(top_logits, dim=-1)
        weights = weights * mask.unsqueeze(-1)                     # zero on padding
        return top_idx, weights, sim


# ============================================================================
# ERQH Layer: aggregate prototypes, route via Hamilton product, residual update
# ============================================================================

class ERQHLayer(nn.Module):
    def __init__(self, d_model, n_codes):
        super().__init__()
        self.n_codes = n_codes
        self.d_model = d_model
        self.proto_proj = QuaternionLinear(d_model, d_model)
        self.update_proj = QuaternionLinear(d_model, d_model)
        self.norm = QuaternionLayerNorm(d_model)

    def forward(self, q, assign_idx, assign_w, contribute_mask):
        """
        q:               [B, N, D]
        assign_idx:      [B, N, m]   long
        assign_w:        [B, N, m]   float
        contribute_mask: [B, N]      float (1 if cell contributes to prototype;
                                            0 for future-query cells)

        Returns:
          q_new:        [B, N, D]
          prototypes:   [B, K, D]
        """
        B, N, D = q.shape
        K = self.n_codes
        m = assign_idx.shape[-1]

        # ---- 1. Aggregate prototypes from observed cells only ----
        # For prototype aggregation, weight by (assign_w * contribute_mask)
        agg_w = assign_w * contribute_mask.unsqueeze(-1)           # [B, N, m]
        # Flatten N×m
        flat_idx = assign_idx.reshape(B, N * m)                    # [B, N*m]
        flat_w = agg_w.reshape(B, N * m, 1)                        # [B, N*m, 1]
        flat_q = q.unsqueeze(2).expand(-1, -1, m, -1).reshape(B, N * m, D)
        weighted_q = flat_q * flat_w                               # [B, N*m, D]
        proto = torch.zeros(B, K, D, device=q.device, dtype=q.dtype)
        weight_sum = torch.zeros(B, K, 1, device=q.device, dtype=q.dtype)
        proto.scatter_add_(1, flat_idx.unsqueeze(-1).expand(-1, -1, D), weighted_q)
        weight_sum.scatter_add_(1, flat_idx.unsqueeze(-1), flat_w)
        proto = proto / (weight_sum + 1e-6)
        proto = self.proto_proj(proto)                             # [B, K, D]

        # ---- 2. Route messages: each obs receives Hamilton-routed prototype mix ----
        idx_exp = assign_idx.unsqueeze(-1).expand(-1, -1, -1, D)   # [B, N, m, D]
        proto_per_obs = torch.gather(
            proto.unsqueeze(1).expand(-1, N, -1, -1), 2, idx_exp
        )                                                          # [B, N, m, D]
        q_exp = q.unsqueeze(2).expand(-1, -1, m, -1)               # [B, N, m, D]
        routed = hamilton_product(q_exp, proto_per_obs)            # [B, N, m, D]
        msg = (routed * assign_w.unsqueeze(-1)).sum(dim=2)         # [B, N, D]
        msg = self.update_proj(msg)

        # ---- 3. Quaternion residual + LN ----
        q_new = q + msg
        q_new = self.norm(q_new)
        return q_new, proto


# ============================================================================
# Main Model
# ============================================================================

class Model(nn.Module):
    """PE-RQH main model. Compatible with PyOmniTS forward signature."""
    def __init__(self, configs: ExpConfigs):
        super().__init__()
        self.configs = configs
        D = (configs.d_model // 4) * 4
        if D != configs.d_model:
            logger.warning(f"PE-RQH: d_model {configs.d_model}->{D} (must be ×4)")
        self.d_model = D
        self.enc_in = configs.enc_in
        self.n_layers = configs.n_layers

        self.n_codes = int(getattr(configs, "perqh_n_codes", 64))
        self.top_m = int(getattr(configs, "perqh_top_m", 3))
        self.tau_init = float(getattr(configs, "perqh_tau_init", 1.0))
        self.tau_min = float(getattr(configs, "perqh_tau_min", 0.5))
        self.tau_decay = float(getattr(configs, "perqh_tau_decay", 0.999))
        self.lambda_div = float(getattr(configs, "perqh_lambda_div", 0.1))
        self.lambda_commit = float(getattr(configs, "perqh_lambda_commit", 0.01))

        # Step counter for tau schedule (registered as buffer, persistent across saves)
        self.register_buffer("global_step", torch.zeros(1, dtype=torch.long))

        self.encoder = QuaternionEncoder(D, configs.enc_in)
        self.codebook = VQCodebook(self.n_codes, D)
        self.layers = nn.ModuleList([
            ERQHLayer(D, self.n_codes) for _ in range(self.n_layers)
        ])
        # Decoder: quaternion → scalar
        self.decoder = nn.Sequential(
            nn.Linear(D, D),
            nn.GELU(),
            nn.Linear(D, 1),
        )

    # --- helpers ported from SQHyper for irregular padding ---
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

    @property
    def current_tau(self):
        step = self.global_step.item()
        tau = max(self.tau_min, self.tau_init * (self.tau_decay ** step))
        return tau

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

        # Concatenate observed and future cells in one flat representation
        if self.configs.task_name in [
            "short_term_forecast", "long_term_forecast",
            "classification", "representation_learning"
        ]:
            x_zeros = torch.zeros_like(y, dtype=x.dtype, device=x.device)
            y_zeros = torch.zeros_like(x, dtype=y.dtype, device=y.device)
            x_y_mark = torch.cat([x_mark, y_mark], dim=1)              # [B, L, 1]
            x_L = torch.cat([x, x_zeros], dim=1)                       # [B, L, V]
            x_y_mask = torch.cat([x_mask, y_mask], dim=1)              # [B, L, V]
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

        # ---- Build per-cell metadata ----
        # times in [0, 1] from time_indices / L
        times_flat = time_indices_flattened.float() / max(1, L - 1)  # [B, N]
        var_ids_flat = variable_indices_flattened                    # [B, N] long
        # Cell-level masks:
        #   contribute_mask: 1 if observed (has real value, contributes to prototype)
        #                    0 if future-query or padding
        #   valid_mask:      1 if either observed or future-query (participates in routing)
        #                    0 if padding
        valid_mask = x_y_mask_flattened                              # 0 for padding
        contribute_mask = valid_mask * (1.0 - y_mask_L_flattened)    # observed only

        # ---- Encode each cell as quaternion ----
        # value mask = contribute_mask: only observed cells get the actual value
        q = self.encoder(
            values=x_L_flattened,
            times=times_flat,
            var_ids=var_ids_flat,
            mask=contribute_mask,
        )                                                            # [B, N, D]
        # Zero out padded cells entirely
        q = q * valid_mask.unsqueeze(-1)

        # ---- VQ codebook assignment ----
        idx, w, sim_full = self.codebook.assign(
            q, valid_mask, top_m=self.top_m,
            tau=self.current_tau,
            training=self.training,
        )

        # ---- L layers of message passing ----
        last_proto = None
        for layer in self.layers:
            q, last_proto = layer(q, idx, w, contribute_mask)
            q = q * valid_mask.unsqueeze(-1)

        # ---- Decode each cell to scalar prediction ----
        pred_flattened = self.decoder(q).squeeze(-1)                 # [B, N]

        # ---- Auxiliary losses (only during training) ----
        aux_loss = self._auxiliary_losses(q, idx, w, sim_full, valid_mask, contribute_mask)
        if self.training:
            self.global_step += 1

        result = {
            "pred": pred_flattened,
            "true": y_L_flattened,
            "mask": y_mask_L_flattened,
        }
        if aux_loss is not None and self.training:
            result["aux_loss"] = aux_loss

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

    def _auxiliary_losses(self, q, idx, w, sim_full, valid_mask, contribute_mask):
        """Codebook diversity + commitment losses.

        Diversity: encourage uniform usage across codes (negative entropy of
        average code usage, scaled).
        Commitment: encourage q_obs to be close to assigned codes (stop-grad
        on q so codebook moves toward q, not vice versa here we let both move).
        """
        B, N, D = q.shape
        K = self.n_codes
        m = idx.shape[-1]

        # ---- Usage distribution per code ----
        # Flat assignments [B, N*m] with weights, only counting observed cells
        ag_w = (w * contribute_mask.unsqueeze(-1)).reshape(B, N * m)
        ag_idx = idx.reshape(B, N * m)
        usage = torch.zeros(B, K, device=q.device, dtype=q.dtype)
        usage.scatter_add_(1, ag_idx, ag_w)
        # Normalize to probability per batch
        usage = usage / (usage.sum(dim=-1, keepdim=True) + 1e-6)
        # Entropy averaged across batch
        ent = -(usage * torch.log(usage + 1e-8)).sum(dim=-1).mean()
        loss_div = math.log(K) - ent                              # 0 when uniform

        # ---- Commitment: each obs's q close to its assigned codes ----
        # Use codebook weights directly (not stop-grad, so both can move)
        codes = self.codebook.codes                               # [K, D]
        # gather assigned codes per obs: [B, N, m, D]
        sel = codes[idx]
        diff = q.unsqueeze(2) - sel                               # [B, N, m, D]
        mse = (diff ** 2).sum(dim=-1)                             # [B, N, m]
        wmse = (mse * w * contribute_mask.unsqueeze(-1)).sum() / (
            contribute_mask.sum() + 1e-6
        )

        return self.lambda_div * loss_div + self.lambda_commit * wmse
