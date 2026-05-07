# PE-RQH: Pure Event-Routed Quaternion Hypergraph

A genuinely new IMTS forecasting architecture, designed independently of HyperIMTS.

## 0. Design Philosophy

| Principle | Decision |
|-----------|----------|
| Hyperedges | Fully emergent from data (VQ codebook), no predefined time/variable edges |
| Node state | Quaternion-native (4 components throughout) |
| Message passing | Pure Hamilton-product algebra, no softmax(QK^T)V |
| Time/variable info | Encoded into quaternion components, NOT structural |
| Decode | Query-driven: future point's quaternion routes through codebook |

## 1. Notation

- B: batch size
- N: number of observations per sample (variable across batch, padded)
- V: number of variables
- T: max time index
- D: model dimension (must be divisible by 4)
- K: codebook size (default 64)
- m: top-m soft assignment per observation (default 3)
- L: number of message passing layers (default 2)

A quaternion vector q ∈ ℝ^D is split as q = (q_r, q_i, q_j, q_k) where each component ∈ ℝ^(D/4).

## 2. Quaternion Algebra Primitives

### 2.1 Hamilton product (element-wise on D/4 stacked quaternions)

Given p = (p_r, p_i, p_j, p_k), q = (q_r, q_i, q_j, q_k):

```
(p ⊗ q)_r = p_r·q_r - p_i·q_i - p_j·q_j - p_k·q_k
(p ⊗ q)_i = p_r·q_i + p_i·q_r + p_j·q_k - p_k·q_j
(p ⊗ q)_j = p_r·q_j - p_i·q_k + p_j·q_r + p_k·q_i
(p ⊗ q)_k = p_r·q_k + p_i·q_j - p_j·q_i + p_k·q_r
```

All operations are element-wise on D/4 dim.

### 2.2 Conjugate

```
q* = (q_r, -q_i, -q_j, -q_k)
```

### 2.3 Quaternion inner product (real-valued similarity)

```
sim(p, q) = Re(p ⊗ q*) = p_r·q_r + p_i·q_i + p_j·q_j + p_k·q_k
```

This is just the standard real inner product on the concatenated 4D vector. Used for VQ assignment.

### 2.4 Quaternion linear layer

A QLinear from D to D' has 4 separate real-valued matrices W_r, W_i, W_j, W_k each of shape (D/4, D'/4):

```
QLinear(q) = (W_r, W_i, W_j, W_k) ⊗ q
           with W treated as a quaternion-valued matrix
```

Equivalent expansion:
```
out_r = q_r W_r - q_i W_i - q_j W_j - q_k W_k
out_i = q_r W_i + q_i W_r + q_j W_k - q_k W_j
out_j = q_r W_j - q_i W_k + q_j W_r + q_k W_i
out_k = q_r W_k + q_i W_j - q_j W_i + q_k W_r
```

Parameter count: 4·(D/4)·(D'/4) = D·D'/4 (one quarter of standard linear).

## 3. Architecture

### 3.1 Quaternion Encoder (per observation)

Input per obs: value v ∈ ℝ, time t ∈ ℝ, var_id ∈ {0,...,V-1}, mask ∈ {0,1}.

Stage 1 — raw quaternion seed (D=64, broadcast to D dim):
```
q_seed = [
    Linear_v(v),                            # value channel
    sin(t · ω) where ω ∈ ℝ^(D/4),           # phase i
    cos(t · ω),                             # phase j
    Embed(var_id) ∈ ℝ^(D/4),                # variable k
]   ∈ ℝ^D
```

Stage 2 — learnable quaternion mixing:
```
q_obs = QLinear_enc(q_seed)
```

This gives a node embedding where time and variable info live initially in i, j, k components, but the QLinear may rotate them.

For masked observations, q_obs is set to zero (won't participate in codebook).

### 3.2 VQ Codebook

A learnable codebook C ∈ ℝ^(K × D) initialized with N(0, 1/√D).

For each obs i, compute assignment to all codes:
```
sim_ik = sim(q_obs_i, C_k)              # real-valued
g_ik = sim_ik / temperature
A_ik = top_m_softmax(g_i)               # keep top-m, renormalize, others = 0
```

During training, use Gumbel-softmax with hard=True for the top-1 component, while keeping soft top-m for the others (mixed approach for stable gradients):

```
A_ik = GumbelSoftmax_topm(g_i, m=3, tau=τ_schedule(step))
```

Temperature schedule: τ = max(0.5, 1.0 · 0.99^(step/100)).

For masked obs, set A_ik = 0 ∀k.

### 3.3 Hyperedge construction

For each code k, the set of obs assigned (with weight) forms hyperedge E_k:
```
E_k = {(i, A_ik) : A_ik > 0}
```

This is the only hyperedge type. K hyperedges total.

### 3.4 Layer-l message passing

Given current node states q^(l) ∈ ℝ^(N × D), assignment matrix A ∈ ℝ^(N × K):

Step 1 — compute event prototype per code:
```
prototype_k = (Σ_i A_ik · q^(l)_i) / (Σ_i A_ik + ε)
prototype_k = QLinear_proto^(l)(prototype_k)
```

Step 2 — node update via routed Hamilton product:
```
For each node i:
    msg_i = Σ_k A_ik · (q^(l)_i ⊗ prototype_k)
    q^(l+1)_i = q^(l)_i + QLinear_update^(l)(msg_i)
```

Note: q ⊗ prototype is the structure-preserving update — modulating node state by event direction.

Step 3 — quaternion layer norm:
```
q^(l+1) = QLayerNorm(q^(l+1))
```

QLayerNorm: Norm computed on each component independently, scale/shift learned.

### 3.5 Query-driven decoder

For each prediction target (var_id, future_time):

Stage 1 — query quaternion:
```
q_query = QEncode(value=0, time=future_time, var_id, mask=1)
q_query = QLinear_query(q_query)
```

Stage 2 — find best codes for query:
```
sim_qk = sim(q_query, C_k)
A_q = top_m_softmax(sim_qk, m=3)
```

Stage 3 — assemble prediction quaternion via routing:
```
pred_q = Σ_k A_q_k · (q_query ⊗ prototype_k^(L))   # use last-layer prototypes
```

Stage 4 — quaternion → real:
```
pred = MLP(Concat(pred_q.r, pred_q.i, pred_q.j, pred_q.k))
```

The MLP is a 2-layer real-valued network with output dim 1 per query.

## 4. Losses

```
L_total = L_pred + λ_div · L_diversity + λ_commit · L_commit
```

### 4.1 Prediction loss
```
L_pred = MSE(pred, target)
```

### 4.2 Diversity loss (prevent codebook collapse)

Average usage per code over a batch:
```
u_k = (1/N) Σ_i A_ik
H = -Σ_k u_k · log(u_k + ε)        # entropy
L_diversity = log(K) - H            # 0 when uniform, increases when collapse
```

### 4.3 Commitment loss (encourage q_obs near assigned codes)
```
L_commit = (1/N) Σ_i Σ_k A_ik · ||stop_grad(q_obs_i) - C_k||^2
```

Default: λ_div = 0.1, λ_commit = 0.01.

## 5. Hyperparameters Summary

| Param | Default | Notes |
|-------|---------|-------|
| D (d_model) | 256 | Must be %4 |
| K (codes) | 64 | per-dataset adjustable |
| m (top-m) | 3 | sparse routing |
| L (layers) | 2 | |
| τ_init / τ_min | 1.0 / 0.5 | Gumbel anneal |
| λ_div | 0.1 | diversity reg |
| λ_commit | 0.01 | commitment |
| Optimizer | Adam | |
| LR | 1e-3 | |

## 6. Memory and Compute

### Memory (worst case: HA, B=32, N=36000)
- q states: 32 · 36000 · 256 · 4 byte = 1.18 GB per layer (fp32)
- A matrix: 32 · 36000 · 64 · 4 byte = 295 MB (sparse via top-m → 3 entries per row, ~14 MB)
- Codebook: 64 · 256 · 4 byte = 65 KB
- **Use top-m sparse representation** for A → A as (indices, values) pairs of shape (B·N, m).

### Compute per layer per sample (asymptotic)
- QLinear: O(N · D²/4)
- Codebook similarity: O(N · K · D)
- Aggregation: O(N · m · D)  [sparse]
- Total ≈ O(N · D · max(D/4, K))

For N=36000, D=256, K=64: ~600M FLOPs per layer per sample. Comparable to HyperIMTS.

## 7. Implementation Pseudocode

```python
class PERQH(nn.Module):
    def __init__(self, d_model, n_codes, n_layers, n_vars):
        self.encoder = QuaternionEncoder(d_model, n_vars)
        self.codebook = nn.Parameter(torch.randn(n_codes, d_model) / sqrt(d_model))
        self.layers = nn.ModuleList([
            ERQHLayer(d_model) for _ in range(n_layers)
        ])
        self.decoder = QuaternionDecoder(d_model)

    def forward(self, x):
        # x: dict with observed_data, observed_tp, observed_mask, predicted_tp, ...
        q_obs = self.encoder.encode_obs(x.values, x.times, x.var_ids, x.masks)
        # [B, N, D] quaternion

        A = vq_assign(q_obs, self.codebook, m=3, tau=current_tau)
        # [B, N, K] sparse top-m

        for layer in self.layers:
            q_obs, prototypes = layer(q_obs, A, self.codebook)

        # Query phase
        q_query = self.encoder.encode_query(x.predicted_tp, x.predicted_var_ids)
        A_query = vq_assign(q_query, self.codebook, m=3, tau=current_tau)
        pred = self.decoder(q_query, prototypes, A_query)

        loss_div = diversity_loss(A)
        loss_commit = commitment_loss(q_obs, self.codebook, A)
        return pred, loss_div, loss_commit
```

## 8. Out-of-scope (deliberate exclusions)

- ❌ Time hyperedges (any obs sharing time bin)
- ❌ Variable hyperedges (any obs sharing var_id)
- ❌ Standard softmax attention
- ❌ Real-valued state representation
- ❌ Bipartite obs-edge bidirectional message passing

These are HyperIMTS's structural choices and are explicitly avoided.
