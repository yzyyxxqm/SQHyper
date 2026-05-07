# SC-PERQH: Structured-Codebook Pure Event-Routed Quaternion Hypergraph
#
# Builds on PE-RQH but with structured (typed) codebook:
#   - K_time time-codes  (initialized with Fourier-basis quaternion templates)
#   - K_var  variable-codes (initialized as orthogonal per-variable templates)
#   - K_event event-codes (random init, learned event prototypes)
#
# Each cell q_obs receives messages from each group via separate top-k routing.
# The three messages are summed and projected through a QuaternionLinear, then
# applied as residual update with quaternion LayerNorm.
#
# Key design choices:
#   - Group-wise top-k (k_t, k_v, k_e) prevents starvation of any group.
#   - Initialization differentiation introduces inductive bias *without*
#     hard-coding HyperIMTS time/variable hyperedges.
#   - Codes remain learnable, so the model can refine the priors during training.
#
# Hyperparameters:
#   d_model            : must be divisible by 4
#   n_layers           : message-passing layers
#   scperqh_k_time     : number of time-codes
#   scperqh_k_var      : number of var-codes (>= enc_in recommended)
#   scperqh_k_event    : number of event-codes
#   scperqh_top_t      : top-k per cell from time-codes (default 2)
#   scperqh_top_v      : top-k from var-codes (default 2)
#   scperqh_top_e      : top-k from event-codes (default 3)
#   scperqh_tau_init   : Gumbel-softmax temperature start
#   scperqh_tau_min    : minimum temperature
#   scperqh_tau_decay  : temperature decay per step
#   scperqh_lambda_div : per-group diversity loss weight
#   scperqh_lambda_commit : per-group commitment loss weight

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat, rearrange
from utils.ExpConfigs import ExpConfigs
from utils.globals import logger


# ============================================================================
# Quaternion primitives (shared with PE-RQH)
# ============================================================================

def hamilton_product(p, q):
    Q = p.shape[-1] // 4
    pr, pi, pj, pk = p[..., :Q], p[..., Q:2*Q], p[..., 2*Q:3*Q], p[..., 3*Q:]
    qr, qi, qj, qk = q[..., :Q], q[..., Q:2*Q], q[..., 2*Q:3*Q], q[..., 3*Q:]
    out_r = pr*qr - pi*qi - pj*qj - pk*qk
    out_i = pr*qi + pi*qr + pj*qk - pk*qj
    out_j = pr*qj - pi*qk + pj*qr + pk*qi
    out_k = pr*qk + pi*qj - pj*qi + pk*qr
    return torch.cat([out_r, out_i, out_j, out_k], dim=-1)


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
# Quaternion Encoder (shared with PE-RQH)
# ============================================================================

class QuaternionEncoder(nn.Module):
    def __init__(self, d_model, n_vars):
        super().__init__()
        assert d_model % 4 == 0
        Q = d_model // 4
        self.Q = Q
        self.value_proj = nn.Linear(2, Q)
        freqs = torch.exp(torch.linspace(0.0, math.log(100.0), Q))
        self.time_freq = nn.Parameter(freqs)
        self.var_emb = nn.Embedding(n_vars, Q)
        nn.init.normal_(self.var_emb.weight, std=0.1)
        self.q_mix = QuaternionLinear(d_model, d_model)

    def forward(self, values, times, var_ids, mask):
        v_input = torch.stack([values * mask, mask], dim=-1)
        q_r = self.value_proj(v_input)
        t_phase = times.unsqueeze(-1) * self.time_freq
        q_i = torch.sin(t_phase)
        q_j = torch.cos(t_phase)
        q_k = self.var_emb(var_ids)
        q_seed = torch.cat([q_r, q_i, q_j, q_k], dim=-1)
        return self.q_mix(q_seed)


# ============================================================================
# Structured Codebook with three typed groups
# ============================================================================

class StructuredCodebook(nn.Module):
    """Codebook split into three typed groups with structural initialization.

    Initialization conventions (D = 4*Q):
      time_codes [K_time, D]:
        - Each code k has frequency ω_k = 2π · k / K_time
        - Quaternion components seeded as (cos shift, sin(t·ω) basis, ...)
        - Effectively encodes "this code corresponds to a time scale ω_k"

      var_codes [K_var, D]:
        - Orthogonal random initialization (so each var-code points to a
          distinct direction in quaternion space)
        - Optionally aligned with var_emb if K_var == n_vars

      event_codes [K_event, D]:
        - Standard small-magnitude random init (purely learned)

    All three groups are nn.Parameter and remain learnable.
    """
    def __init__(self, k_time, k_var, k_event, d_model, n_vars):
        super().__init__()
        assert d_model % 4 == 0
        self.k_time = k_time
        self.k_var = k_var
        self.k_event = k_event
        self.d_model = d_model
        self.n_vars = n_vars
        Q = d_model // 4

        # ---- time-codes: structured Fourier basis ----
        # For code k (k=0..K_time-1), we use frequency ω_k linearly spaced.
        # Components:
        #   r-part: cos(0) = 1 placeholder (will be modulated by time_freq)
        #   i-part: sin(2π k / K_time · ω_basis)
        #   j-part: cos(2π k / K_time · ω_basis)
        #   k-part: small random
        time_init = torch.zeros(k_time, d_model)
        # frequency pattern across Q dimensions
        ω_basis = torch.exp(torch.linspace(0.0, math.log(50.0), Q))  # [Q]
        for kk in range(k_time):
            phase = 2 * math.pi * kk / max(k_time, 1)
            time_init[kk, 0:Q]       = math.cos(phase)            # r: constant
            time_init[kk, Q:2*Q]     = torch.sin(phase * ω_basis) # i
            time_init[kk, 2*Q:3*Q]   = torch.cos(phase * ω_basis) # j
            time_init[kk, 3*Q:4*Q]   = 0.05 * torch.randn(Q)      # k: small noise
        self.time_codes = nn.Parameter(time_init)

        # ---- var-codes: orthogonal init ----
        var_init = torch.randn(max(k_var, 1), d_model)
        # Make rows approximately orthogonal via QR (only meaningful if k_var <= d_model)
        if k_var <= d_model:
            var_init, _ = torch.linalg.qr(var_init.t())
            var_init = var_init.t()[:k_var]
            var_init = var_init * (d_model ** 0.5) * 0.5  # rescale to similar magnitude
        else:
            var_init = var_init / math.sqrt(d_model)
        self.var_codes = nn.Parameter(var_init)

        # ---- event-codes: random small ----
        event_init = torch.randn(k_event, d_model) / math.sqrt(d_model)
        self.event_codes = nn.Parameter(event_init)

    def assign_group(self, q, codes, mask, top_k, tau, training):
        """Top-k soft assignment to a single code group.

        q: [B, N, D], codes: [Kg, D], mask: [B, N]
        Returns:
          idx: [B, N, top_k] long
          w:   [B, N, top_k] float (sums to 1, zero on mask)
        """
        sim = q @ codes.t()                                # [B, N, Kg]
        sim_masked = sim.masked_fill(mask.unsqueeze(-1) == 0, -1e4)
        Kg = codes.shape[0]
        top_k = min(top_k, Kg)
        top_vals, top_idx = sim_masked.topk(top_k, dim=-1)
        if training:
            u = torch.rand_like(top_vals).clamp(min=1e-7, max=1.0 - 1e-7)
            inner = (-torch.log(u)).clamp_min(1e-7)
            gumbel = -torch.log(inner)
            top_logits = (top_vals + gumbel) / max(tau, 1e-3)
        else:
            top_logits = top_vals / max(tau, 1e-3)
        weights = F.softmax(top_logits, dim=-1)
        weights = weights * mask.unsqueeze(-1)
        return top_idx, weights, sim


# ============================================================================
# SC-ERQH Layer: per-group prototype aggregation + Hamilton routing
# ============================================================================

class SCLayer(nn.Module):
    def __init__(self, d_model, k_time, k_var, k_event):
        super().__init__()
        self.d_model = d_model
        self.k_time = k_time
        self.k_var = k_var
        self.k_event = k_event
        # Each group has its own prototype/update projection
        self.proto_t = QuaternionLinear(d_model, d_model)
        self.proto_v = QuaternionLinear(d_model, d_model)
        self.proto_e = QuaternionLinear(d_model, d_model)
        # Combined update projection
        self.update_proj = QuaternionLinear(d_model, d_model)
        self.norm = QuaternionLayerNorm(d_model)

    def _aggregate_and_route(self, q, idx, w, codes, contribute_mask, proto_proj):
        """Compute prototypes for one group then route Hamilton message back.

        Returns: msg [B, N, D]
        """
        B, N, D = q.shape
        Kg = codes.shape[0]
        m = idx.shape[-1]
        # Aggregate prototypes (batch-local)
        agg_w = w * contribute_mask.unsqueeze(-1)             # [B, N, m]
        flat_idx = idx.reshape(B, N * m)                       # [B, N*m]
        flat_w = agg_w.reshape(B, N * m, 1)
        flat_q = q.unsqueeze(2).expand(-1, -1, m, -1).reshape(B, N * m, D)
        weighted_q = flat_q * flat_w
        proto = torch.zeros(B, Kg, D, device=q.device, dtype=q.dtype)
        weight_sum = torch.zeros(B, Kg, 1, device=q.device, dtype=q.dtype)
        proto.scatter_add_(1, flat_idx.unsqueeze(-1).expand(-1, -1, D), weighted_q)
        weight_sum.scatter_add_(1, flat_idx.unsqueeze(-1), flat_w)
        # Mix learned codebook prior with batch-local prototype
        # When weight_sum is small, fall back to global codes
        weight_sum_clamp = weight_sum.clamp_min(1e-3)
        proto_local = proto / weight_sum_clamp
        # Blend with global codes (batch-broadcast)
        codes_broadcast = codes.unsqueeze(0).expand(B, -1, -1)
        # blending factor: low when weight_sum is small (use global), high when many cells routed
        blend = (weight_sum / (weight_sum + 0.5)).clamp(0, 1)  # [B, Kg, 1]
        proto_blended = blend * proto_local + (1 - blend) * codes_broadcast
        proto_blended = proto_proj(proto_blended)              # [B, Kg, D]

        # Route message: gather assigned prototypes, Hamilton with q, weighted sum
        idx_exp = idx.unsqueeze(-1).expand(-1, -1, -1, D)      # [B, N, m, D]
        proto_per_cell = torch.gather(
            proto_blended.unsqueeze(1).expand(-1, N, -1, -1), 2, idx_exp
        )                                                      # [B, N, m, D]
        q_exp = q.unsqueeze(2).expand(-1, -1, m, -1)
        routed = hamilton_product(q_exp, proto_per_cell)
        msg = (routed * w.unsqueeze(-1)).sum(dim=2)            # [B, N, D]
        return msg, proto_blended

    def forward(self, q, idx_t, w_t, idx_v, w_v, idx_e, w_e,
                codebook, contribute_mask):
        """Single SC-ERQH layer.

        Returns:
          q_new:        [B, N, D]
          protos:       dict of group prototypes for inspection
        """
        msg_t, proto_t = self._aggregate_and_route(
            q, idx_t, w_t, codebook.time_codes, contribute_mask, self.proto_t)
        msg_v, proto_v = self._aggregate_and_route(
            q, idx_v, w_v, codebook.var_codes, contribute_mask, self.proto_v)
        msg_e, proto_e = self._aggregate_and_route(
            q, idx_e, w_e, codebook.event_codes, contribute_mask, self.proto_e)

        # Combine messages via QuaternionLinear after sum
        msg_total = self.update_proj(msg_t + msg_v + msg_e)
        q_new = q + msg_total
        q_new = self.norm(q_new)
        return q_new, {"t": proto_t, "v": proto_v, "e": proto_e}


# ============================================================================
# Main Model
# ============================================================================

class Model(nn.Module):
    def __init__(self, configs: ExpConfigs):
        super().__init__()
        self.configs = configs
        D = (configs.d_model // 4) * 4
        if D != configs.d_model:
            logger.warning(f"SC-PERQH: d_model {configs.d_model}->{D}")
        self.d_model = D
        self.n_layers = configs.n_layers
        self.enc_in = configs.enc_in

        self.k_time = int(getattr(configs, "scperqh_k_time", 32))
        self.k_var = int(getattr(configs, "scperqh_k_var", max(8, configs.enc_in)))
        self.k_event = int(getattr(configs, "scperqh_k_event", 32))
        self.top_t = int(getattr(configs, "scperqh_top_t", 2))
        self.top_v = int(getattr(configs, "scperqh_top_v", 2))
        self.top_e = int(getattr(configs, "scperqh_top_e", 3))
        self.tau_init = float(getattr(configs, "scperqh_tau_init", 1.0))
        self.tau_min = float(getattr(configs, "scperqh_tau_min", 0.5))
        self.tau_decay = float(getattr(configs, "scperqh_tau_decay", 0.999))
        self.lambda_div = float(getattr(configs, "scperqh_lambda_div", 0.05))
        self.lambda_commit = float(getattr(configs, "scperqh_lambda_commit", 0.005))

        self.register_buffer("global_step", torch.zeros(1, dtype=torch.long))

        self.encoder = QuaternionEncoder(D, configs.enc_in)
        self.codebook = StructuredCodebook(
            self.k_time, self.k_var, self.k_event, D, configs.enc_in)
        self.layers = nn.ModuleList([
            SCLayer(D, self.k_time, self.k_var, self.k_event)
            for _ in range(self.n_layers)
        ])
        self.decoder = nn.Sequential(
            nn.Linear(D, D),
            nn.GELU(),
            nn.Linear(D, 1),
        )

    # ---- helpers (same as PE-RQH) ----
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
        return max(self.tau_min, self.tau_init * (self.tau_decay ** step))

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
        contribute_mask = valid_mask * (1.0 - y_mask_L_flattened)

        # Encode each cell as quaternion
        q = self.encoder(
            values=x_L_flattened,
            times=times_flat,
            var_ids=var_ids_flat,
            mask=contribute_mask,
        )
        q = q * valid_mask.unsqueeze(-1)

        # Per-group structured codebook assignment
        tau = self.current_tau
        idx_t, w_t, _ = self.codebook.assign_group(
            q, self.codebook.time_codes, valid_mask, self.top_t, tau, self.training)
        idx_v, w_v, _ = self.codebook.assign_group(
            q, self.codebook.var_codes, valid_mask, self.top_v, tau, self.training)
        idx_e, w_e, _ = self.codebook.assign_group(
            q, self.codebook.event_codes, valid_mask, self.top_e, tau, self.training)

        # SC-ERQH layers
        for layer in self.layers:
            q, _ = layer(q, idx_t, w_t, idx_v, w_v, idx_e, w_e,
                         self.codebook, contribute_mask)
            q = q * valid_mask.unsqueeze(-1)

        # Decode
        pred_flattened = self.decoder(q).squeeze(-1)

        # Aux loss (per-group diversity + commitment)
        aux_loss = None
        if self.training:
            aux_loss = self._auxiliary_losses(
                q, idx_t, w_t, idx_v, w_v, idx_e, w_e, contribute_mask)
            self.global_step += 1

        result = {
            "pred": pred_flattened,
            "true": y_L_flattened,
            "mask": y_mask_L_flattened,
        }
        if aux_loss is not None:
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

    def _auxiliary_losses(self, q, idx_t, w_t, idx_v, w_v, idx_e, w_e, contribute_mask):
        loss = torch.tensor(0.0, device=q.device, dtype=q.dtype)
        for idx, w, codes, K in [
            (idx_t, w_t, self.codebook.time_codes, self.k_time),
            (idx_v, w_v, self.codebook.var_codes, self.k_var),
            (idx_e, w_e, self.codebook.event_codes, self.k_event),
        ]:
            B, N, m = idx.shape
            ag_w = (w * contribute_mask.unsqueeze(-1)).reshape(B, N * m)
            ag_idx = idx.reshape(B, N * m)
            usage = torch.zeros(B, K, device=q.device, dtype=q.dtype)
            usage.scatter_add_(1, ag_idx, ag_w)
            usage = usage / (usage.sum(dim=-1, keepdim=True) + 1e-6)
            ent = -(usage * torch.log(usage + 1e-8)).sum(dim=-1).mean()
            loss_div = math.log(K) - ent
            sel = codes[idx]
            diff = q.unsqueeze(2) - sel
            mse = (diff ** 2).sum(dim=-1)
            wmse = (mse * w * contribute_mask.unsqueeze(-1)).sum() / (
                contribute_mask.sum() * m + 1e-6
            )
            loss = loss + self.lambda_div * loss_div + self.lambda_commit * wmse
        return loss
