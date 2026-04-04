# graphTRIP Model — Complete Explanation

This document explains the **graphTRIP** architecture from the paper *"Accurate and Interpretable Prediction of Antidepressant Treatment Response from Receptor-informed Neuroimaging"* (Tolle et al., 2026, bioRxiv), and the corresponding implementation in `src/models/graphTrip/`.

---

## 1. The Big Picture

**The problem**: Can we predict how depressed a patient will respond to antidepressant treatment *before* they take it, using only brain imaging data taken at baseline?

**The solution**: graphTRIP is a **geometric deep learning** model that:
1. Represents the brain as a **graph** (regions = nodes, functional connectivity = edges)
2. Learns a compressed, informative **latent representation** of that brain graph via a Variational Graph Autoencoder (VGAE)
3. Uses that latent representation + clinical data to **predict post-treatment depression severity** (QIDS score) via a multi-layer perceptron (MLP)
4. Enables **interpretability** by analyzing gradients in latent space

The model achieved $r = 0.72$ on a primary dataset of 42 MDD patients (psilocybin vs. escitalopram trial).

---

## 2. The Input Data

### What is the brain graph?

Each patient yields **one brain graph** built from resting-state fMRI data:

- **Nodes** = brain regions defined by the Schaefer 100-parcel atlas (100 nodes per graph)
- **Edges** = functional connectivity (FC): Pearson correlation between BOLD time series of each pair of regions. Only the **top 20% strongest** absolute correlations are kept (fixed density $\rho = 0.2$). This gives a sparse but consistent topology across patients.

### Node features (what information is stored per node)

Each node carries **two types** of feature:

1. **Learned node features** (`batch.x`, shape: `[N, 3]`): the **REACT values** for 3 serotonin-related molecular targets:
   - `R5-HT1A` — serotonin receptor 1A density
   - `R5-HT2A` — serotonin receptor 2A density
   - `R5-HTT` — serotonin transporter density

   REACT (Receptor-Enriched Analysis of functional Connectivity by Targets) estimates how much each brain region's activity is functionally coupled to a given receptor system, computed from PET maps. These are **patient-specific**.

2. **Conditional node features** (`batch.xc`, shape: `[N, 3]`): the **3D MNI coordinates** of each brain region's centroid. These are *not* patient-specific — they encode anatomical identity. The encoder uses them to understand *which region* a node represents, enabling permutation-invariant processing without losing spatial context.

### Edge features

Each edge carries the **raw (non-thresholded) FC correlation value** as a scalar, i.e. `edge_attr` has shape `[num_edges, 1]`. The threshold is used only to determine *which* edges exist; their weights are preserved as attributes.

---

## 3. Overall Architecture

```
Brain Graph (per patient)
   ├─ nodes: REACT [N, 3]  +  MNI coords [N, 3]
   └─ edges: FC correlations [E, 1]
         │
         ▼
   ┌─────────────────────────────┐
   │  VGAE Encoder               │
   │  (NodeEmbeddingGraphormer)  │
   │  3 Graphormer layers        │
   │  → node embeddings [N, D]   │
   │  → (μ_i, logσ²_i) per node │
   │  → sample z_i               │
   └─────────────┬───────────────┘
                 │
         ┌───── ┤ ─────┐
         │             │
         ▼             ▼
   VGAE Decoder   Readout Layer
   (reconstruct   (GraphTransformerPooling)
   graph)         → graph vector z [D]
                        │
                        ▼
              ┌──────────────────────┐
              │  MLP Prediction Head  │
              │  input: z + clinical  │
              │  → predicted QIDS     │
              └──────────────────────┘
```

The VGAE and MLP are trained **jointly** end-to-end.

---

## 4. VGAE Encoder — The Graphormer (`NodeEmbeddingGraphormer`)

### What is a Graphormer?

A Graphormer adapts the Transformer architecture (normally used on sequences) to work on graphs. Standard attention treats every token equally. In a graph, we want attention to be informed by **graph structure** — which nodes are close together, which are strongly connected. Graphormer achieves this by adding **structural bias terms** to the attention scores.

### Step-by-step forward pass

**Step 1 — Input projection**

Node features are concatenated with conditional (MNI coordinate) features and projected into the working embedding space:

$$h_i^{(0)} = W_{\text{in}} \begin{bmatrix} x_i \\ c_i \end{bmatrix}$$

where $x_i \in \mathbb{R}^3$ are the REACT values, $c_i \in \mathbb{R}^3$ are the MNI coordinates, and $W_{\text{in}}$ is a learned linear projection. The output $h_i^{(0)} \in \mathbb{R}^{d_\text{hidden}}$ with $d_\text{hidden} = 16$.

**Step 2 — Compute attention biases (from graph structure)**

For each pair of nodes $(i, j)$, two bias terms are added to the attention scores:

- **Shortest-path distance bias** $b_{ij}^\text{spd}$: The shortest-path distance between nodes $i$ and $j$ is discretised and looked up in a learned embedding table:

$$b_{ij}^\text{spd} = \text{DistEmbed}\!\left(\text{SPD}(i,j)\right) \in \mathbb{R}^{n_\text{heads}}$$

- **Edge feature bias** $b_{ij}^\text{edge}$: For edges that *exist* in the graph, the FC correlation value is passed through a small 2-layer MLP to produce a per-head scalar. Non-edges have zero bias:

$$b_{ij}^\text{edge} = \text{MLP}_\text{edge}(e_{ij}) \in \mathbb{R}^{n_\text{heads}}$$

Both biases are summed: $b_{ij} = b_{ij}^\text{spd} + b_{ij}^\text{edge}$ and added to the attention scores below.

**Step 3 — 3 Graphormer layers**

Each layer (class `GraphormerLayer`) updates node embeddings as follows:

*Multi-head self-attention with biased scores:*

$$e_{ij}^{(\ell,h)} = \frac{\left(Q_i^{(\ell,h)}\right)^\top K_j^{(\ell,h)}}{\sqrt{d_h}} + b_{ij}^{\text{spd},h} + b_{ij}^{\text{edge},h}$$

$$\alpha_{ij}^{(\ell,h)} = \frac{\exp\!\left(e_{ij}^{(\ell,h)}\right)}{\sum_k \exp\!\left(e_{ik}^{(\ell,h)}\right)}$$

$$\hat{h}_i^{(\ell)} = W_\text{attn} \sum_h \sum_j \alpha_{ij}^{(\ell,h)} V_j^{(\ell,h)}$$

where $Q^{(\ell,h)}, K^{(\ell,h)}, V^{(\ell,h)}$ are learned linear projections and $d_h$ is the per-head dimension.

*Residual connections and feedforward network:*

$$u_i^{(\ell)} = \text{LN}\!\left(h_i^{(\ell)} + \text{Dropout}\!\left(\hat{h}_i^{(\ell)}\right)\right)$$

$$h_i^{(\ell+1)} = \text{LN}\!\left(u_i^{(\ell)} + \text{Dropout}\!\left(\Phi_\text{ff}\!\left(u_i^{(\ell)}\right)\right)\right)$$

where $\Phi_\text{ff}$ is a 2-layer MLP with ReLU activation. After 3 such layers, each node embedding integrates information from the full graph, weighted by both feature similarity and structural proximity.

**Step 4 — Output projection**

$$h_i = W_\text{out}\, h_i^{(L)} \in \mathbb{R}^{d_\text{node}}, \quad d_\text{node} = 32$$

**Step 5 — Variational encoding (reparameterization)**

A final linear layer maps each node embedding to the parameters of a Gaussian:

$$\left(\mu_i,\, \log \sigma_i^2\right) = W_\text{enc}\, h_i$$

Latent node representations are then sampled via the reparameterization trick:

$$z_i = \mu_i + \sigma_i \odot \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, I)$$

where $\sigma_i = \exp(0.5 \cdot \log \sigma_i^2)$. This allows gradients to flow back through $z_i$ to the encoder during training. Each node now has a stochastic latent representation $z_i \in \mathbb{R}^{32}$.

---

## 5. Readout Layer — `GraphTransformerPooling`

The encoder produces *node-level* latent vectors $\{z_i\}$. We need a single *graph-level* vector $\mathbf{z}$ to feed to the MLP. The readout layer:

1. **Self-attention** (without structural biases) across all nodes — a standard multi-head attention block followed by LayerNorm + Dropout, letting each node attend to all others once more.

2. **Feed-forward network** + residual + LayerNorm to produce contextualised embeddings $\tilde{z}_i$.

3. **Mean pooling**:

$$\mathbf{z} = \frac{1}{N} \sum_{i=1}^{N} \tilde{z}_i \in \mathbb{R}^{32}$$

This single vector represents the **entire brain graph** for one patient.

---

## 6. VGAE Decoder — Reconstructing the Graph

The decoder takes node-level latent samples $\{z_i\}$ and tries to reconstruct the original graph. It has 3 independent sub-modules:

### Node decoder (`MLPNodeDecoder`)
- Architecture: $32 \to 32 \to 32 \to 3$ with LeakyReLU + Dropout
- Reconstructs the **REACT node features** $\hat{x}_i \in \mathbb{R}^3$
- MNI coordinates are *not* reconstructed (they are conditional inputs)

### Edge index decoder
- Architecture: $64 \to 32 \to 32 \to 1$ with LeakyReLU + Dropout + **Sigmoid**
- Input for each node pair $(i,j)$: concatenation $[z_i \| z_j] \in \mathbb{R}^{64}$
- Output: $\hat{p}_{ij} \in [0, 1]$ — probability that edge $(i,j)$ exists

### Edge attribute decoder (`MLPEdgeDecoder`)
- Architecture: $64 \to 32 \to 32 \to 1$ with LeakyReLU + Dropout + **Tanh**
- Same input format as edge index decoder
- Output: $\hat{e}_{ij} \in [-1, 1]$ — reconstructed FC correlation value

The **reconstructed FC matrix** is the element-wise product $\hat{p}_{ij} \cdot \hat{e}_{ij}$.

---

## 7. MLP Prediction Head (`RegressionMLP`)

The MLP takes as input the concatenation:

$$\text{input} = \left[\mathbf{z} \;\|\; \text{QIDS}_\text{base} \;\|\; \text{BDI}_\text{base} \;\|\; \text{SSRI} \;\|\; d\right] \in \mathbb{R}^{36}$$

where $d \in \{-1, +1\}$ is the drug condition (escitalopram / psilocybin).

Architecture: $36 \to 64 \to 64 \to 1$ with LeakyReLU (slope 0.01) and Dropout(0.25) between layers.

Output: a single continuous value $\hat{y}$ = **predicted post-treatment QIDS score**.

---

## 8. Training Objective

The VGAE and MLP are trained **jointly** using a weighted sum of two losses:

$$\mathcal{L}_\text{graphTRIP} = \alpha\, \mathcal{L}_\text{VGAE} + (1 - \alpha)\, \mathcal{L}_\text{MLP}, \quad \alpha = 0.5$$

### VGAE loss

$$\mathcal{L}_\text{VGAE} = \mathcal{L}_\text{recon} + \mathcal{L}_\text{KL} + \mathcal{L}_\text{reg}$$

$$\mathcal{L}_\text{recon} = \sum_{i=1}^{N} \|x_i - \hat{x}_i\|_2^2 \;+\; \sum_{(i,j)\in\mathcal{E}} \|e_{ij} - \hat{e}_{ij}\|_2^2$$

$$\mathcal{L}_\text{KL} = \mathbb{E}_{q(z|X)}\!\left[\text{KL}\!\left(q(z|X) \;\|\; p(z)\right)\right], \quad p(z) = \mathcal{N}(0, I)$$

$$\mathcal{L}_\text{reg} = \lambda \sum_k W_{\text{VGAE},k}^2, \quad \lambda = 0.01$$

The KL term regularises the latent space towards a standard Gaussian, preventing the encoder from memorising the training data.

### MLP loss

$$\mathcal{L}_\text{MLP} = \sum_{i=1}^{n} \|y_i - \hat{y}_i\|_2^2 \;+\; \lambda \sum_k W_{\text{MLP},k}^2$$

### Training protocol
- Adam optimiser, $\text{lr} = 0.001$
- 300 epochs
- 7-fold cross-validation, stratified by treatment condition
- Batch size = 9
- Repeated 10 times with different random seeds; results averaged

---

## 9. Medusa-graphTRIP (Causal Extension)

This is an extension to estimate **Individual Treatment Effects (ITEs)** — which drug would benefit *this specific patient* more?

Key differences from graphTRIP:

1. **Shared VGAE**: same architecture as graphTRIP, trained on patients from *both* treatment arms simultaneously.

2. **Different pooling**: uses `MeanStdPooling`. The graph-level vector is the concatenation of mean and standard deviation of node latent vectors:
$$\mathbf{z} = \left[\frac{1}{N}\sum_i z_i \;\bigg\|\; \sqrt{\frac{1}{N}\sum_i (z_i - \bar{z})^2}\right] \in \mathbb{R}^{64}$$

3. **Two MLP heads** (`CFRHead`): $\text{MLP}_0$ for escitalopram and $\text{MLP}_1$ for psilocybin. Each is trained only on patients from its treatment arm. The ITE is estimated as:

$$\tilde{D}(x_i) = \mu_1(x_i) - \mu_0(x_i)$$

A negative ITE indicates the patient is predicted to respond better to psilocybin.

4. **MMD penalty**: Maximum Mean Discrepancy between the latent representations of the two treatment groups, encouraging the shared VGAE to produce treatment-invariant representations (a necessary condition for counterfactual inference).

5. **Extended loss**:

$$\mathcal{L}_\text{Medusa} = \alpha\, \mathcal{L}_\text{VGAE} + (1-\alpha)\, \mathcal{L}_\text{MLP} + \beta\, \mathcal{L}_\text{MMD}, \quad \beta = 1.0$$

---

## 10. Interpretability: GRAIL

**GRAIL** (Gradient Alignment for Interpreting Latent-variable models) estimates learned associations between candidate brain-graph biomarkers and model predictions, *through* the latent space.

### How it works

For a biomarker $b$ (e.g., mean FC within the visual network) computed from the reconstructed graph, GRAIL computes the cosine similarity between two gradients in latent space:

$$\text{Alignment}(b) = \frac{\nabla_{\mathbf{z}}\,\hat{y} \;\cdot\; \nabla_{\mathbf{z}}\,b}{\|\nabla_{\mathbf{z}}\,\hat{y}\|\;\|\nabla_{\mathbf{z}}\,b\|} \;\in\; [-1,\, 1]$$

- $\nabla_{\mathbf{z}}\,\hat{y}$: direction in latent space that increases the predicted QIDS
- $\nabla_{\mathbf{z}}\,b$: direction in latent space that increases the biomarker

**Interpretation**:
- $+1$: biomarker and prediction increase together → biomarker predicts **worse** outcome (resistance)
- $-1$: biomarker and prediction point in opposite directions → biomarker predicts **better** outcome (responsiveness)
- $0$: no learned association

To account for latent-space stochasticity, alignment is computed over 25 samples from each patient's $\mathcal{N}(\mu_i, \sigma_i^2)$ latent distribution and then averaged. Statistical significance is assessed against spatially-informed null models (spin permutations).

### Regional Attribution Analysis

Gradient-based attribution decomposes the model's sensitivity into per-brain-region contributions:

1. Compute the full gradient $\nabla_{\mathbf{z}}\,\hat{y}$ and normalise to unit Euclidean norm.
2. For each brain region $i$, compute the node-wise partial gradient $\nabla_{z_i}\,\hat{y}$.
3. Regional attribution score for region $i$:

$$a_i = \frac{\|\nabla_{z_i}\,\hat{y}\|^2}{\|\nabla_{\mathbf{z}}\,\hat{y}\|^2} \times 100\%$$

Scores sum to 100%, indicating each region's percentage contribution to the total gradient energy.

---

## 11. Code Structure (`src/models/graphTrip/`)

| File | Contents |
|------|----------|
| `models.py` | All model classes |
| `model_configs.json` | Default hyperparameter configs per model class |
| `utils.py` | `init_model()`, `get_model_configs()`, freeze/unfreeze helpers |
| `__init__.py` | Package exports |

### Key classes in `models.py`

| Class | Role |
|-------|------|
| `NodeEmbeddingGraphormer` | Main encoder: 3 Graphormer layers with SPD + edge biases |
| `GraphormerLayer` | Single Graphormer layer (attention + FFN + residual) |
| `EdgeAwareMultiheadAttention` | Multi-head attention with additive edge/SPD bias |
| `GraphTransformerPooling` | Readout: self-attention + mean pooling → graph vector |
| `MeanStdPooling` | Simpler readout for Medusa: concat(mean, std) of node latents |
| `MLPNodeDecoder` | Reconstructs REACT node features from $z_i$ |
| `MLPEdgeDecoder` | Reconstructs FC edge weights from $[z_i \| z_j]$ |
| `RegressionMLP` | Prediction head: $\mathbf{z}$ + clinical → QIDS |
| `CFRHead` | Two-head prediction for Medusa (one per treatment arm) |
| `NodeEmbeddingGATv2Conv` | Alternative encoder using Graph Attention Network v2 |
| `StandardMLP` | Base MLP building block (shared by all MLP-based models) |
| `NodeEmbeddingMLP` | Simpler baseline encoder (no message passing) |

### Hyperparameter summary (from Table II of paper)

| Component | Setting |
|-----------|---------|
| Latent dimension | 32 |
| Encoder hidden dim | 16 |
| Encoder layers | 3 Graphormer layers |
| Encoder FFN dim | 32 |
| Attention heads | 4 |
| Dropout | 0.25 |
| MLP architecture | $[36 \to 64 \to 64 \to 1]$ |
| Regularisation $\lambda$ | 0.01 |
| Learning rate | 0.001 |
| Epochs | 300 |
| Batch size | 9 |
| CV folds | 7 |
| $\alpha$ (loss weight) | 0.5 |

---

## 12. Key Findings from the Paper

- **Most important predictor**: the latent brain graph $\mathbf{z}$ (more important than baseline QIDS, BDI, or drug condition)
- **Shared resistance biomarker**: stronger FC in Visual (VIS) and Sensory-Motor (SMN) networks → worse outcomes under both drugs
- **Differential biomarker**: FPN serotonergic coupling (R5-HT1A, R5-HT2A) → resistance to psilocybin but responsiveness to escitalopram
- **Cortical hierarchy**: unimodal regions (VIS, SMN) predict overall response; transmodal regions (DMN, FPN) predict treatment-specific effects
- **Transfer learning**: VGAEs trained on Schaefer 100 generalise to Schaefer 200 and AAL atlases without retraining

---

## 13. What Would Need to Change for EEG + DMT Blood Prediction

*(Discussed in detail separately — this section is a placeholder for planning)*

The key differences in your use case:
- **Input modality**: EEG instead of fMRI → different graph construction (EEG channels as nodes, coherence/correlation as edges)
- **Node features**: EEG spectral features (band power, etc.) instead of REACT maps (no PET data available)
- **Prediction target**: DMT blood concentration (continuous, pharmacokinetic) instead of depression scores
- **No treatment condition**: single-arm, so no Medusa extension needed initially
- **Temporal structure**: EEG is dynamic → time-series structure may need to be considered

The core VGAE + MLP architecture is directly re-usable with these modifications.
