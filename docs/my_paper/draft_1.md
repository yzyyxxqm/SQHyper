
| Method            | MIMIC-III          |                    | PhysioNet'12       |                    | HumanActivity      |                    | USHCN              |                    |
|-------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|
|                   | MSE                | MAE                | MSE                | MAE                | MSE                | MAE                | MSE                | MAE                |
| mTAN              | 0.8910±0.0396      | 0.6424±0.0120      | 0.3745±0.0043      | 0.4250±0.0037      | 0.0847±0.0024      | 0.2080±0.0031      | 0.7401±0.0968      | 0.5685±0.0320      |
| GRU_D             | —                  | —                  | 0.3418±0.0024      | 0.3997±0.0011      | 0.1663±0.0294      | 0.3078±0.0244      | 0.2713±0.0066      | 0.2966±0.0038      |
| Raindrop          | —                  | —                  | 0.3451±0.0008      | 0.4024±0.0004      | 0.0963±0.0091      | 0.2184±0.0126      | —                  | —                  |
| GraFITi           | 0.4094±0.0087      | 0.3748±0.0045      | 0.3026±0.0007      | 0.3597±0.0006      | 0.0418±0.0003      | 0.1145±0.0007      | 0.1852±0.0021      | 0.2747±0.0084      |
| Warpformer        | —                  | —                  | 0.3078±0.0047      | 0.3648±0.0050      | —                  | —                  | 0.1814±0.0063      | 0.2703±0.0055      |
| tPatchGNN         | 0.4601±0.0037      | 0.4036±0.0027      | 0.3094±0.0015      | 0.3669±0.0024      | 0.0455±0.0029      | 0.1285±0.0071      | 0.2221±0.0359      | 0.2853±0.0182      |
| PrimeNet          | —                  | —                  | —                  | —                  | —                  | —                  | 0.4573±0.0008      | 0.4782±0.0011      |
| HyperIMTS         | 0.3939±0.0066      | 0.3709±0.0028      | 0.2991±0.0005      | 0.3598±0.0012      | 0.0414±0.0002      | 0.1140±0.0002      | 0.2411±0.0413      | 0.2861±0.0118      |
| **Ours** | 0.3926±0.0051  | 0.3724±0.0020  | 0.3010±0.0007  | 0.3616±0.0008  | 0.0417±0.0002  | 0.1157±0.0010  | 0.1651±0.0019  | 0.2648±0.0051  |

> Notes: all reported cells use n=5 iterations under matched data splits, evaluation horizons, and seeds. `—` cells in the main table fall into three categories: (a) baselines that we did not run to completion on MIMIC-III in the current submission cycle — `GRU_D`, `Raindrop`, `Warpformer` — owing to compute budget; their MIMIC-III numbers will be added in the camera-ready; (b) hard-failed runs that we explicitly do not re-attempt: `Raindrop/USHCN` (`torch_scatter` import error from the official Raindrop dependency on this CUDA build), `Warpformer/HumanActivity` (the official Warpformer codebase exits with `rc=1` in < 10 s on the HumanActivity collator); (c) datasets on which a model is not benchmarked by its original authors and is therefore omitted: `PrimeNet/{MIMIC-III, P12, HumanActivity}`.

### Ablation results

#### Main ablation across all four datasets (n=5)

| Variant                          | MIMIC-III          | PhysioNet'12       | HumanActivity      | USHCN              |
|----------------------------------|--------------------|--------------------|--------------------|--------------------|
|                                  | MSE / MAE          | MSE / MAE          | MSE / MAE          | MSE / MAE          |
| **QSHNet (full, Ours)**          | 0.3926±0.0057 / 0.3724±0.0023 | 0.3010±0.0008 / 0.3616±0.0009 | 0.0417±0.0002 / 0.1157±0.0011 | 0.1651±0.0021 / 0.2648±0.0057 |
| QSHNet w/o quaternion (`no_quat`) | 0.3906±0.0038 / 0.3713±0.0023 | 0.3004±0.0010 / 0.3604±0.0012 | 0.0417±0.0002 / 0.1153±0.0011 | 0.1974±0.0432 / 0.2766±0.0163 |
| QSHNet w/o spike (`no_spike`)     | 0.3943±0.0099 / 0.3684±0.0018 | 0.3010±0.0010 / 0.3615±0.0016 | 0.0416±0.0001 / 0.1154±0.0008 | 0.1804±0.0215 / 0.2700±0.0064 |

> All twelve ablation cells use the same code-base, hyper-parameters, and 5 random seeds as the full model; only the targeted module is disabled (the quaternion residual is replaced by an identity, the spike router by an all-pass gate). On three of the four datasets — MIMIC-III, PhysioNet'12 and HumanActivity — disabling either module changes the test MSE by ≤ 1 % and is statistically within 1 σ of the full model. On USHCN, the smallest and most irregular benchmark, the two modules contribute distinctly: removing the quaternion branch costs **+20 % MSE** (0.1651 → 0.1974, ≥ 7 σ above the full model's std) while removing the spike router costs **+9 % MSE** (0.1651 → 0.1804, ≥ 4 σ above).

#### Data-scaling ablation on MIMIC-III (n=5)

To test whether USHCN's gap is purely a small-data effect, we shrunk the MIMIC-III training set to 25 % and 10 % of its original size (the same pre-defined val/test splits are kept; only training samples are re-sampled with a fixed seed) and re-ran the same three configurations.

| Train fraction | n_train | full              | no_quat (Δ vs full)         | no_spike (Δ vs full)        |
|---|---:|---|---|---|
| 1.00 | 17 212 | **0.3926±0.0057** | 0.3906±0.0038 (−0.52 %) | 0.3943±0.0099 (+0.42 %) |
| 0.25 |  4 303 | **0.4343±0.0067** | 0.4292±0.0060 (−1.17 %) | 0.4339±0.0059 (−0.09 %) |
| 0.10 |  1 721 | **0.4727±0.0112** | 0.4772±0.0149 (+0.94 %) | 0.4808±0.0171 (+1.71 %) |

> Numbers are test MSE; deltas are versus `full` at the same train fraction. Across all three MIMIC-III data scales — including 1.7 k samples (≈ USHCN size) — the gap between `full` and either ablation is **at most ±1.7 % MSE and stays within ≈ 1–1.5 σ in every cell**. There is a faint monotone trend (modules become more useful as data shrinks: −0.5 % → −1.2 % → +0.9 % for `no_quat`, +0.4 % → −0.1 % → +1.7 % for `no_spike`), but it is an order of magnitude smaller than the +20 % / +9 % gaps seen on USHCN at full data. We conclude that USHCN's gap is **primarily domain-specific** — its 5 monthly weather variables exhibit very high inter-channel correlation and a structurally regular yet temporally sparse sampling pattern, both of which directly engage the cross-group quaternion mixing and event-driven spike routing. Reducing MIMIC-III to USHCN-scale data alone, without that domain structure, is not enough to surface the same effect.


## 2 Related Work

### 2.1 Irregular Multivariate Time Series Forecasting

$\qquad$Existing IMTS methods can be broadly grouped into padding-based and non-padding approaches. The former represents input series as matrices along temporal and variable dimensions. RNN-based models handle missing values through trainable decay mechanisms (Che et al., 2018). ODE-based models treat hidden states as continuous-time processes to naturally accommodate irregular intervals (Chen et al., 2018; Rubanova et al., 2019; De Brouwer et al., 2019; Biloš et al., 2021; Schirmer et al., 2022; Mercatali et al., 2024). Transformer-based models use attention to learn temporal representations at reference points (Shukla & Marlin, 2021; Zhang et al., 2023). While effective, these padding-based methods increase computation and may disrupt the original sampling pattern. Non-padding approaches use sets (Horn et al., 2020) or bipartite graphs (Yalavarthi et al., 2024) to represent only the observed values, but have limited ability to capture dependencies among unaligned observations. More recent GNN-based methods introduce patches to locally align asynchronous series (Zhang et al., 2024; Luo et al., 2025; Liu et al., 2025), yet standard graphs are restricted to pairwise connections. To capture high-order dependencies, hypergraph architectures have been introduced: Ada-MSHyper (Shang et al., 2024) applies multi-scale hypergraph Transformers for regular MTS, and HyperIMTS (Li et al., 2025) represents each raw observation as a node connected by temporal and variable hyperedges, enabling direct message passing without padding. Despite this progress, current hypergraph models aggregate messages using dense, real-valued transformations that ignore the inherent sparsity of IMTS and flatten multi-dimensional features, losing the geometric structure among coupled channels.

>  - Che, Z., Purushotham, S., Cho, K., Sontag, D., & Liu, Y. (2018). Recurrent neural networks for multivariate time series with missing values. _Scientific Reports_, 8(1), 6085.
>  - Chen, R. T. Q., Rubanova, Y., Bettencourt, J., & Duvenaud, D. K. (2018). Neural ordinary differential equations. _NeurIPS_.
>  - Rubanova, Y., Chen, R. T. Q., & Duvenaud, D. K. (2019). Latent ordinary differential equations for irregularly-sampled time series. _NeurIPS_.
>  - De Brouwer, E., Simm, J., Arany, A., & Moreau, Y. (2019). GRU-ODE-Bayes: Continuous modeling of sporadically-observed time series. _NeurIPS_.
>  - Horn, M., Moor, M., Bock, C., Rieck, B., & Borgwardt, K. M. (2020). Set functions for time series. _ICML_.
>  - Shukla, S. N., & Marlin, B. M. (2021). Multi-time attention networks for irregularly sampled time series. _ICLR_.
>  - Biloš, M., Sommer, J., Rangapuram, S. S., Januschowski, T., & Günnemann, S. (2021). Neural flows: Efficient alternative to neural ODEs. _NeurIPS_.
>  - Schirmer, M., Eltayeb, M., Lessmann, S., & Rudolph, M. (2022). Modeling irregular time series with continuous recurrent units. _ICML_.
>  - Zhang, C., Cai, X., et al. (2023). Warpformer: A multi-scale modeling approach for irregular clinical time series. _KDD_.
>  - Yalavarthi, V. K., et al. (2024). GraFITi: Graphs for forecasting irregularly sampled time series. _AAAI_.
>  - Zhang, W., Yin, C., Liu, H., Zhou, X., & Xiong, H. (2024). T-PatchGNN: Irregular multivariate time series forecasting via transformable patching graph neural networks. _ICML_.
>  - Shang, Z., Chen, L., Wu, B., & Cui, D. (2024). Ada-MSHyper: Adaptive multi-scale hypergraph transformer for time series forecasting. _NeurIPS_.
>  - Mercatali, G., Freitas, A., & Chen, J. (2024). Graph neural flows for unveiling systemic interactions among irregularly sampled time series. _NeurIPS_.
>  - Liu, Z., Luo, Y., et al. (2025). TimeCHEAT: Channel harmony strategy for irregularly sampled multivariate time series forecasting. _AAAI_.
>  - Luo, Y., Zhang, B., Liu, Z., & Ma, Q. (2025). Hi-Patch: Hierarchical patch GNN for irregular multivariate time series. _ICML_.
>  - Li, B., Luo, Y., Liu, Z., Zheng, J., Lv, J., & Ma, Q. (2025). HyperIMTS: Hypergraph neural network for irregular multivariate time series forecasting. _ICML_.

### 2.2 Spiking Neural Networks for Temporal Dynamics

$\qquad$Spiking neural networks (SNNs), the third generation of neural networks (Maass, 1997), communicate through discrete pulses and update states only when stimuli exceed a membrane threshold (Gerstner et al., 2014). This event-driven mechanism naturally aligns with the sparsity-event duality of IMTS, where long uninformative gaps are punctuated by short observation bursts. Recent work has applied SNNs to time series forecasting: SeqSNN (Lv et al., 2024a) adapts CNN, RNN, and Transformer architectures to spiking counterparts; TS-LIF (Feng et al., 2025) uses a dual-compartment neuron model for multi-scale temporal dynamics; and fractional-order SNNs (Ge et al., 2025) introduce power-law memory kernels for long-range dependencies. For irregular time series, SEDformer (Zhou et al., 2026) introduces an event-aligned LIF neuron that fires only at observed timestamps, and SpikySpace (Tang et al., 2026) combines spiking neurons with state space models for linear-time complexity. However, existing SNN architectures process each variable independently and do not model the geometric coupling among multi-channel features.

>  - Maass, W. (1997). Networks of spiking neurons: The third generation of neural network models. _Neural Networks_, 10(9), 1659–1671.
>  - Gerstner, W., Kistler, W. M., Naud, R., & Paninski, L. (2014). _Neuronal Dynamics: From Single Neurons to Networks and Models of Cognition_. Cambridge University Press.
>  - Lv, C., Wang, Y., Han, D., Zheng, X., Huang, X., & Li, D. (2024a). Efficient and effective time-series forecasting with spiking neural networks. _ICML_.
>  - Lv, C., Han, D., Wang, Y., Zheng, X., Huang, X., & Li, D. (2024b). Advancing spiking neural networks for sequential modeling with central pattern generators. _arXiv:2405.14362_.
>  - Feng, S., Feng, W., Gao, X., Zhao, P., & Shen, Z. (2025). TS-LIF: A temporal segment spiking neuron network for time series forecasting. _ICLR_.
>  - Ge, C., et al. (2025). Fractional-order spiking neural network. _arXiv preprint_.
>  - Tang, K., et al. (2026). SpikySpace: A spiking state space model for energy-efficient time series forecasting. _arXiv:2601.02411_.
>  - Zhou, Z., Fang, Y., Ruan, W., Wang, S., Kwok, J., & Liang, Y. (2026). SEDformer: Event-synchronous spiking transformers for irregular telemetry time series forecasting. _arXiv:2602.02230_.

### 2.3 Quaternion Neural Networks for Multi-channel Coupling

$\qquad$Quaternion neural networks (QNNs) encode four feature components as a single quaternion entity and use the Hamilton product for structured cross-channel weight sharing, reducing parameters by ~75% compared to real-valued counterparts (Parcollet et al., 2018). QNNs have been applied to multi-channel speech recognition (Qiu et al., 2020), time series compression (Pöppelbaum & Schwung, 2024), fault diagnosis (Liu et al., 2024), 3D motion prediction (Bourigault et al., 2024), and EEG analysis (Ortega-Flores et al., 2025). In summary, existing work has independently explored hypergraphs for high-order topology (Li et al., 2025), spike-inspired mechanisms for event-driven sparsity (Zhou et al., 2026), and quaternion algebra for multi-channel coupling (Parcollet et al., 2018), but these capabilities remain isolated. Our work integrates them as complementary components within a single framework: a spike-inspired router injects event-driven sparsity before hypergraph message aggregation, and a bounded quaternion linear layer preserves multi-channel geometry during hyperedge-to-node refinement.

>  - Parcollet, T., Ravanelli, M., Morchid, M., Linarès, G., Trabelsi, C., De Mori, R., & Bengio, Y. (2018). Quaternion recurrent neural networks. _ICLR_.
>  - Qiu, X., Parcollet, T., Ravanelli, M., Lane, N., & Morchid, M. (2020). Quaternion neural networks for multi-channel distant speech recognition. _arXiv:2005.08566_.
>  - Pöppelbaum, J. & Schwung, A. (2024). Time series compression using quaternion valued neural networks and quaternion backpropagation. _NLDL_.
>  - Liu, H., et al. (2024). Multi-scale quaternion CNN and BiGRU with cross self-attention feature fusion for fault diagnosis of bearing. _arXiv:2405.16114_.
>  - Bourigault, P., Xu, D., & Mandic, D. P. (2024). Quaternion recurrent neural network with real-time recurrent learning and maximum correntropy criterion. _arXiv:2402.14227_.
>  - Ortega-Flores, G., et al. (2025). Quaternion CNN in deep learning processing for EEG with applications to brain disease detection. _Applied Sciences_, 15(21), 11526.

## 3 Preliminaries

### 3.1 Irregular Multivariate Time Series

$\qquad$Formally, an IMTS sample $\mathcal{S} = \{S_1, S_2, \ldots, S_V\}$ comprises $V$ co-evolving variables indexed by $v \in \{1, \ldots, V\}$. Each variable $S_v = \{(t_k^v, x_k^v)\}_{k=1}^{n_v}$ contains $n_v$ timestamped scalar observations with strictly increasing timestamps $t_1^v < t_2^v < \cdots < t_{n_v}^v$. IMTS exhibits two structural properties absent in regular MTS: (1) *intra-variable irregularity*, meaning that consecutive intervals $t_{k+1}^v - t_k^v$ within any single variable are non-uniform; and (2) *inter-variable misalignment*, meaning that the timestamp sets $\{t_k^u\}$ and $\{t_k^v\}$ generally differ for $u \neq v$, so observations across variables cannot be naturally aligned into a matrix without padding (Shukla & Marlin, 2021; Yalavarthi et al., 2024). We write $N = \sum_{v=1}^{V} n_v$ for the total observation count and introduce a binary indicator $m_{t,v} \in \{0, 1\}$ that equals 1 if and only if variable $v$ is observed at time $t$.

### 3.2 Forecasting Task

$\qquad$The IMTS forecasting problem (Schirmer et al., 2022; Zhang et al., 2024; Li et al., 2025) is defined as follows. Given all observations within a historical window $[0, T_h]$ and a set of $P$ future query timestamps $\{t'_1, t'_2, \ldots, t'_P\} \subset (T_h, T_h + T_p]$, the goal is to predict the value $\hat{x}_{t'_p}^v$ for each variable $v \in \{1, \ldots, V\}$ at every query time $t'_p$. Let $\mathcal{Q} = \{(t'_p, v) : m_{t'_p, v} = 1\}$ denote the set of query positions where ground-truth values are available. The training objective is the masked mean squared error:

$$\mathcal{L} = \frac{1}{|\mathcal{Q}|} \sum_{(t'_p, v) \in \mathcal{Q}} \left( \hat{x}_{t'_p}^v - x_{t'_p}^v \right)^2$$

In practice, both the historical observations and the future query positions are jointly processed in one forward pass, which enables the model to condition predictions on all available context.

### 3.3 Hypergraph Representation for IMTS

$\qquad$Standard graphs connect pairs of nodes and thus can only express pairwise relationships. In contrast, a hypergraph allows each hyperedge to connect an arbitrary subset of nodes, making it a natural tool for capturing the group-wise dependencies that arise when multiple variables share a timestamp or when a single variable spans many time steps (Feng et al., 2019; Shang et al., 2024). Building on this insight, HyperIMTS (Li et al., 2025) constructs a hypergraph directly from the raw IMTS observations, which we adopt and extend with spike routing and quaternion refinement in Section 4.

$\qquad$Concretely, each observation $(t_k^v, x_k^v)$ is treated as a node, yielding a node set $\mathcal{V}$ with $|\mathcal{V}| = N$. These nodes are organized by two families of hyperedges:

- **Temporal hyperedges** $\mathcal{E}_T$: a hyperedge $e_t$ groups every node observed at time $t$, i.e., $e_t = \{(t, v) : m_{t,v} = 1\}$. Intuitively, $e_t$ captures the *cross-variable snapshot* at a given moment.
- **Variable hyperedges** $\mathcal{E}_V$: a hyperedge $e_v$ groups all nodes of variable $v$, i.e., $e_v = \{(t_k^v, v) : k = 1, \ldots, n_v\}$. This encodes the *temporal trajectory* of a single channel.

The resulting hypergraph is $\mathcal{G} = (\mathcal{V}, \mathcal{E}_T, \mathcal{E}_V)$, with membership recorded by two incidence matrices $\mathbf{H}_T \in \{0,1\}^{L \times N}$ and $\mathbf{H}_V \in \{0,1\}^{V \times N}$, where $L$ denotes the total number of distinct timestamps spanning both the historical and prediction horizons. Because only actually observed positions correspond to nodes, this representation avoids the padding overhead that conventional matrix-based approaches incur. Moreover, the incidence matrices provide a structural foundation for the two enhancements we introduce: the spike router (Section 4.2) leverages $\mathbf{H}_V$ to compute per-variable context for event detection, while the quaternion refinement layer (Section 4.4) operates at the hyperedge-to-node fusion stage mediated by both $\mathbf{H}_T$ and $\mathbf{H}_V$.

>  - Feng, Y., You, H., Zhang, Z., Ji, R., & Gao, Y. (2019). Hypergraph neural networks. _AAAI_.
>  - (Other references listed in Section 2.)

## 4 Methodology

The architecture of EQ-Hyper is illustrated in Figure 2. Built on the hypergraph $\mathcal{G}$ defined in Section 3.3, the model stacks $K$ learner layers between an encoder and a decoder. Each layer extends the hypergraph message-passing pipeline of Li et al. (2025) with two identity-initialized modules: a context-aware event router (Section 4.2) before node-to-hyperedge aggregation, and a bounded quaternion refinement (Section 4.4) during hyperedge-to-node fusion.

### 4.1 Observation-Level Embedding on Hypergraph

$\qquad$The encoder converts the raw hypergraph $\mathcal{G}$ into three sets of embeddings, all sharing a common hidden dimension $D$ (a fixed hyperparameter). We use the superscript $(0)$ to denote initial embeddings produced by the encoder; after passing through the $k$-th learner layer, the embeddings are updated to superscript $(k)$.

**Observation node embeddings.** Each node $n$ is encoded from a two-dimensional input vector $[x_n, \delta_n] \in \mathbb{R}^2$:

$$\mathbf{z}_n^{(0)} = \mathrm{ReLU}(\mathbf{W}_o [x_n, \delta_n] + \mathbf{b}_o) \odot m_n \tag{1}$$

where $x_n \in \mathbb{R}$ is the observed value at node $n$; $\delta_n \in \{0, 1\}$ is a prediction horizon indicator ($\delta_n = 1$ if $n$ belongs to a query timestamp in $\mathcal{Q}$, and $0$ otherwise); $\mathbf{W}_o, \mathbf{b}_o$ are learnable parameters; and $m_n \in \{0, 1\}$ is the observation mask that zeros out invalid (padded) positions. The two-dimensional input separates the value information from the positional role of each node, allowing the model to distinguish historical observations from prediction targets.

**Temporal hyperedge embeddings.** Each temporal hyperedge $l \in \{1, \ldots, L\}$ is encoded via a learned sinusoidal transformation of its timestamp:

$$\boldsymbol{\tau}_l^{(0)} = \sin(\mathbf{W}_\tau t_l + \mathbf{b}_\tau) \tag{2}$$

where $t_l \in \mathbb{R}$ is the (normalized) timestamp of the $l$-th time step, and $\mathbf{W}_\tau, \mathbf{b}_\tau$ are learnable parameters. The sinusoidal activation introduces periodic inductive bias, enabling the model to capture temporal periodicity.

**Variable hyperedge embeddings.** Each variable hyperedge $v \in \{1, \ldots, V\}$ is assigned a learnable embedding from a shared embedding table $\mathbf{W}_{\mathrm{var}} \in \mathbb{R}^{V \times D}$:

$$\boldsymbol{\nu}_v^{(0)} = \mathrm{ReLU}(\mathbf{W}_{\mathrm{var}}[v, :]) \tag{3}$$

where $\mathbf{W}_{\mathrm{var}}[v, :] \in \mathbb{R}^D$ is the $v$-th row of the table. Unlike the node and temporal embeddings, the variable embeddings are input-independent and capture the static identity of each variable channel.

$\qquad$These three sets of initial embeddings $\{\mathbf{z}_n^{(0)}\}$, $\{\boldsymbol{\tau}_l^{(0)}\}$, and $\{\boldsymbol{\nu}_v^{(0)}\}$, together with the incidence matrices $\mathbf{H}_T$ and $\mathbf{H}_V$, are fed into the $K$-layer hypergraph learner described in the following sections. For notational convenience, we write $l(n) \in \{1, \ldots, L\}$ and $v(n) \in \{1, \ldots, V\}$ for the temporal and variable hyperedge indices associated with node $n$, and use $\|$ throughout to denote vector concatenation.

### 4.2 Context-Aware Event Router

$\qquad$In standard hypergraph message passing, all observation nodes contribute equally to the hyperedge update during aggregation. However, in IMTS, most observations at a given timestamp carry routine information consistent with the variable's recent behavior, while a small fraction deviate significantly and signal events (e.g., a sudden spike in heart rate). Treating all observations identically during aggregation dilutes these event signals. The context-aware event router addresses this by splitting each node's representation into a base path for standard message passing and an event path that isolates deviation-driven information. The event path is aggregated separately and injected into hyperedges as a scaled residual (detailed in Section 4.3).

**Variable-level context.** At each learner layer $k$, the router first computes a per-variable context vector by averaging the embeddings of all nodes belonging to the same variable via the variable incidence matrix $\mathbf{H}_V$:

$$\boldsymbol{\mu}_v^{(k)} = \frac{\sum_{j=1}^{N} \mathbf{H}_V[v, j] \cdot \mathbf{z}_j^{(k)}}{\sum_{j=1}^{N} \mathbf{H}_V[v, j]} \tag{4}$$

where $\mathbf{z}_j^{(k)} \in \mathbb{R}^D$ is the embedding of node $j$ at the input of layer $k$. Each node $n$ then retrieves the context of its own variable $v(n)$ and computes its deviation:

$$\mathbf{d}_n^{(k)} = \mathbf{z}_n^{(k)} - \boldsymbol{\mu}_{v(n)}^{(k)} \tag{5}$$

intuitively, $\mathbf{d}_n^{(k)}$ measures how much node $n$ differs from the typical behavior of its variable channel.

**Membrane potential.** Inspired by the integrate-and-fire mechanism in spiking neural networks, the router computes a scalar membrane potential for each node that summarizes how event-like the observation is:

$$r_n^{(k)} = \mathbf{w}_r^\top \big[\mathbf{z}_n^{(k)} \;\|\; \mathbf{d}_n^{(k)}\big] + b_r \tag{6}$$

where $\mathbf{w}_r, b_r$ are learnable parameters. Taking both the node embedding and its deviation as input enables context-aware routing decisions rather than relying on the observation value alone.

**Base and event paths.** The membrane potential controls a retain gate that modulates the base path, while a separate event projection extracts event-specific features:

$$\mathbf{z}_n^{\mathrm{base}} = \mathbf{z}_n^{(k)} \cdot \big(1 - s_r \cdot \sigma(-r_n^{(k)})\big) \cdot m_n \tag{7}$$
$$\mathbf{z}_n^{\mathrm{event}} = \big(\mathbf{W}_e [\mathbf{z}_n^{(k)} \| \mathbf{d}_n^{(k)}] + \mathbf{b}_e\big) \cdot s_e \cdot m_n \tag{8}$$

where $\sigma(\cdot)$ is the sigmoid function; $s_r \in [0, s_r^{\max}]$ is a bounded learnable scalar controlling the retain attenuation strength; $s_e = \frac{1}{2}\exp(\lambda_e)$ is a learnable event scale parameterized by a log-scale $\lambda_e$; and $\mathbf{W}_e, \mathbf{b}_e$ are learnable parameters of the event feature projection.

In Eq. (7), when the membrane potential $r_n^{(k)}$ is large (indicating a strong event signal), $\sigma(-r_n^{(k)}) \to 0$ and the retain gate approaches $1$, preserving the full node embedding in the base path. When $r_n^{(k)}$ is small, $\sigma(-r_n^{(k)}) \to 1$ and the retain gate is slightly attenuated by $s_r$. In Eq. (8), the event projection $\mathbf{W}_e$ learns to extract features from both the raw embedding and its deviation that are relevant for event characterization.

All parameters in the router are initialized such that it reduces to an identity operation at the start of training, i.e., $\mathbf{z}_n^{\mathrm{base}} \approx \mathbf{z}_n^{(k)}$ and $\mathbf{z}_n^{\mathrm{event}} \approx \mathbf{0}$. This ensures the model begins as a standard hypergraph network and gradually learns to separate event signals only when doing so reduces the loss. Initialization details are provided in Section 5.

### 4.3 Event-Augmented Hypergraph Message Passing

$\qquad$Each of the $K$ learner layers performs a full round of hypergraph message passing in two stages: **node-to-hyperedge** aggregation, which updates the temporal and variable hyperedge representations by attending over the connected nodes, and **hyperedge-to-node** fusion, which gathers the updated hyperedge information back to each node. The event router (Section 4.2) augments this pipeline by injecting deviation-driven signals into the hyperedge updates.

**Node-to-hyperedge aggregation.** After the event router produces $\mathbf{z}_n^{\mathrm{base}}$ and $\mathbf{z}_n^{\mathrm{event}}$, the base path enters the standard node-to-hyperedge attention mechanism of Li et al. (2025). For each temporal hyperedge $l$, a multi-head attention block takes the current temporal embedding $\boldsymbol{\tau}_l^{(k)}$ as the query and the base-path node embeddings (concatenated with their variable-hyperedge context) as keys and values, attending only over the nodes connected to $l$ via $\mathbf{H}_T$:

$$\tilde{\boldsymbol{\tau}}_l^{(k)} = \mathrm{MHA}\!\Big(\boldsymbol{\tau}_l^{(k)},\;\big\{\big[\boldsymbol{\nu}_{v(n)}^{(k)} \| \mathbf{z}_n^{\mathrm{base}}\big]\big\}_{n \in \mathcal{N}_T(l)}\Big) \tag{9}$$

where $\mathrm{MHA}$ denotes multi-head attention and $\mathcal{N}_T(l) = \{n : \mathbf{H}_T[l, n] = 1\}$ is the set of nodes connected to temporal hyperedge $l$. The variable hyperedges are updated analogously using $\mathbf{H}_V$:

$$\tilde{\boldsymbol{\nu}}_v^{(k)} = \mathrm{MHA}\!\Big(\boldsymbol{\nu}_v^{(k)},\;\big\{\big[\boldsymbol{\tau}_{l(n)}^{(k)} \| \mathbf{z}_n^{\mathrm{base}}\big]\big\}_{n \in \mathcal{N}_V(v)}\Big) \tag{10}$$

where $\mathcal{N}_V(v) = \{n : \mathbf{H}_V[v, n] = 1\}$. This cross-context design — providing variable context when updating temporal hyperedges and vice versa — enables each hyperedge to capture inter-variable and inter-temporal dependencies simultaneously.

**Event injection.** The event path $\mathbf{z}_n^{\mathrm{event}}$ is aggregated onto each temporal and variable hyperedge via mean pooling through the corresponding incidence matrix:

$$\Delta\boldsymbol{\tau}_l^{(k)} = \frac{\sum_{n=1}^{N} \mathbf{H}_T[l, n] \cdot \mathbf{z}_n^{\mathrm{event}}}{\sum_{n=1}^{N} \mathbf{H}_T[l, n]} \tag{11}$$

$$\Delta\boldsymbol{\nu}_v^{(k)} = \frac{\sum_{n=1}^{N} \mathbf{H}_V[v, n] \cdot \mathbf{z}_n^{\mathrm{event}}}{\sum_{n=1}^{N} \mathbf{H}_V[v, n]} \tag{12}$$

These event deltas are then added to the attention-updated hyperedges as a scaled residual:

$$\boldsymbol{\tau}_l^{(k+1)} = \tilde{\boldsymbol{\tau}}_l^{(k)} + s_{\mathrm{inj}} \cdot \Delta\boldsymbol{\tau}_l^{(k)} \tag{13}$$

$$\boldsymbol{\nu}_v^{(k+1)} = \tilde{\boldsymbol{\nu}}_v^{(k)} + s_{\mathrm{inj}} \cdot \Delta\boldsymbol{\nu}_v^{(k)} \tag{14}$$

where $s_{\mathrm{inj}} \in [0, s_{\mathrm{inj}}^{\max}]$ is a bounded learnable injection scale (per layer). By keeping $s_{\mathrm{inj}}$ small, the event residual serves as a gentle correction that enriches the hyperedge representations with deviation-driven information without destabilizing the base message-passing pathway.

**Hyperedge-to-node fusion.** Each node $n$ gathers the updated embeddings of its temporal and variable hyperedges:

$$\mathbf{c}_n^{(k)} = \big[\boldsymbol{\tau}_{l(n)}^{(k+1)} \;\|\; \boldsymbol{\nu}_{v(n)}^{(k+1)}\big] \tag{15}$$

and the gathered context is fused with the node embedding through a self-attention layer followed by a linear projection:

$$\mathbf{h}_n^{(k)} = \mathbf{W}_h \big[\mathrm{SelfAttn}(\mathbf{z}_n^{(k)},\; \mathbf{c}_n^{(k)}) \;\|\; \mathbf{c}_n^{(k)}\big] + \mathbf{b}_h \tag{16}$$

where $\mathbf{W}_h \in \mathbb{R}^{D \times 3D}$ and $\mathbf{b}_h \in \mathbb{R}^D$. After the quaternion refinement step described in Section 4.4, the node embedding is updated via a residual connection:

$$\mathbf{z}_n^{(k+1)} = \mathrm{ReLU}\!\big(\mathbf{z}_n^{(k)} + \hat{\mathbf{h}}_n^{(k)}\big) \cdot m_n \tag{17}$$

where $\hat{\mathbf{h}}_n^{(k)}$ denotes the output after quaternion refinement (Eq. (21) in Section 4.4). At the final layer ($k = K-1$), an irregularity-aware attention module (Li et al., 2025) further refines the variable hyperedge representations by modeling pairwise variable dependencies.

### 4.4 Bounded Quaternion Refinement

$\qquad$The hyperedge-to-node fusion in Eq. (16) projects a $3D$-dimensional concatenation of the node embedding and its surrounding hyperedge contexts down to a $D$-dimensional vector through a single linear map $\mathbf{W}_h$. While efficient, this projection treats all feature dimensions as independent linear combinations and cannot model rotational or cross-group interactions between sub-components of the fused representation. Such interactions are particularly relevant when a node's update depends jointly on multiple sources of context — for example, when an event-driven temporal cue must be coupled with the corresponding variable identity. We address this by adding a quaternion-algebra-based residual that captures these cross-group couplings, while keeping the refinement small and bounded so that the base hypergraph pathway remains dominant.

**Quaternion linear layer.** We treat each $D$-dimensional feature vector as four sub-components of size $D/4$ corresponding to the real ($R$) and three imaginary ($I, J, K$) parts of a quaternion (Parcollet et al., 2019). Let $\mathbf{R}, \mathbf{I}, \mathbf{J}, \mathbf{K} \in \mathbb{R}^{(D/4) \times (D/4)}$ be four learnable component matrices. The Hamilton product between an input vector and the quaternion weight produces an output that mixes the four sub-components according to the quaternion algebra rules; this is equivalent to multiplying the input by a structured block matrix:

$$\mathrm{QuatLinear}(\mathbf{x}) = \mathbf{W}_q \mathbf{x} + \mathbf{b}_q \tag{18}$$

where $\mathbf{b}_q \in \mathbb{R}^D$ is a bias vector, and the weight $\mathbf{W}_q \in \mathbb{R}^{D \times D}$ is assembled as a $4 \times 4$ block matrix from the component matrices:

$$\mathbf{W}_q = \begin{bmatrix} \mathbf{R} & -\mathbf{I} & -\mathbf{J} & -\mathbf{K} \\ \mathbf{I} & \mathbf{R} & -\mathbf{K} & \mathbf{J} \\ \mathbf{J} & \mathbf{K} & \mathbf{R} & -\mathbf{I} \\ \mathbf{K} & -\mathbf{J} & \mathbf{I} & \mathbf{R} \end{bmatrix} \tag{19}$$

Compared with a standard $D \times D$ linear layer, $\mathrm{QuatLinear}$ has only one quarter of the parameters ($4 \times (D/4)^2$ versus $D^2$) yet imposes the structured cross-group coupling characteristic of quaternion multiplication, which has been shown to capture multidimensional interactions efficiently (Parcollet et al., 2019; Tay et al., 2019).

**Gated additive refinement.** Rather than replacing the linear fusion output, the quaternion layer produces an additive residual modulated by a learnable gate $\alpha_n^{(k)}$:

$$\alpha_n^{(k)} = \sigma\!\big(\mathbf{w}_\alpha^\top [\mathbf{h}_n^{(k)} \;\|\; s_e] + b_\alpha\big) \tag{20}$$

where $\mathbf{w}_\alpha, b_\alpha$ are learnable parameters, and $s_e$ is the event scale from Eq. (8). Conditioning $\alpha_n^{(k)}$ on $s_e$ allows the model to allocate stronger refinement when the router has begun to respond to event signals.

**Bounded residual.** To prevent the quaternion residual from overwhelming the base representation, its norm is clipped to a fixed fraction of $\|\mathbf{h}_n^{(k)}\|$:

$$\hat{\mathbf{h}}_n^{(k)} = \mathbf{h}_n^{(k)} + \mathrm{Clip}_{\rho}\!\big(\alpha_n^{(k)} \cdot \mathrm{QuatLinear}(\mathbf{h}_n^{(k)}),\; \mathbf{h}_n^{(k)}\big) \tag{21}$$

where the clip operator is defined as $\mathrm{Clip}_\rho(\mathbf{r}, \mathbf{h}) = \mathbf{r} \cdot \min\!\big(1,\; \rho \cdot \|\mathbf{h}\| / \|\mathbf{r}\|\big)$, which rescales the residual $\mathbf{r}$ whenever its norm exceeds $\rho \cdot \|\mathbf{h}\|$, with $\rho \in (0, 1)$ a hyperparameter. This bound guarantees that the refinement perturbs the base output by at most a fraction $\rho$ of its magnitude in any direction, making the residual a controlled correction rather than a competing pathway. The refined output $\hat{\mathbf{h}}_n^{(k)}$ then enters the node update in Eq. (17).

The component matrices are initialized so that $\mathrm{QuatLinear}$ acts as the identity ($\mathbf{R} = \mathbf{I}_{D/4}$, $\mathbf{I} = \mathbf{J} = \mathbf{K} = \mathbf{0}$, $\mathbf{b}_q = \mathbf{0}$), and the gate $\alpha_n^{(k)}$ is initialized to a small value, so that at the start of training $\hat{\mathbf{h}}_n^{(k)} \approx \mathbf{h}_n^{(k)}$ and the model behaves as a standard hypergraph network. Initialization details are provided in Section 5.

### 4.5 Decoding and Training Objective

$\qquad$After $K$ learner layers, each node $n$ is associated with three final representations: its own embedding $\mathbf{z}_n^{(K)}$, the temporal hyperedge embedding $\boldsymbol{\tau}_{l(n)}^{(K)}$ of the timestamp it belongs to, and the variable hyperedge embedding $\boldsymbol{\nu}_{v(n)}^{(K)}$ of its variable channel. These three representations jointly encode the node's value, temporal context, and variable context. The decoder concatenates them and applies a shared linear projection to produce a scalar prediction:

$$\hat{x}_n = \mathbf{w}_d^\top \big[\mathbf{z}_n^{(K)} \;\|\; \boldsymbol{\tau}_{l(n)}^{(K)} \;\|\; \boldsymbol{\nu}_{v(n)}^{(K)}\big] + b_d \tag{22}$$

where $\mathbf{w}_d, b_d$ are learnable parameters shared across all nodes. Sharing the decoder across all nodes allows the model to handle an arbitrary number of prediction targets per sample without introducing target-specific parameters.

$\qquad$The decoder produces a scalar prediction for every node in the hypergraph, including both historical and query nodes. Because only the query nodes have ground-truth values available, we compute the training loss exclusively over the query set $\mathcal{Q}$ defined in Section 3.2, using the masked mean squared error

$$\mathcal{L} = \frac{1}{|\mathcal{Q}|} \sum_{n \in \mathcal{Q}} \big(\hat{x}_n - x_n\big)^2 \tag{23}$$

where $x_n$ is the ground-truth observation at node $n$. Historical nodes contribute to the loss only indirectly: they shape the query predictions through $K$ rounds of hypergraph message passing, but their own decoded values are not supervised.
