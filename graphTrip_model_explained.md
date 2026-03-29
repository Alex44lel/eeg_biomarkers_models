# graphTRIP Model — Explained

This document walks through every class in `src/models/graphTrip/model_cleaned.py` in the recommended reading order (simplest to most complex). The goal is to build up a mental model of how all pieces fit together.

---

## Big Picture

graphTRIP is a **Node-level Variational Graph Autoencoder (VGAE)** that takes a brain connectivity graph as input and predicts a clinical outcome (QIDS depression score).

The full pipeline, end-to-end:

```
Brain graph (nodes=REACT values, edges=FC correlations)
    │
    ▼
NodeEmbeddingGraphormer   ← Graphormer-style transformer encoder
    │  [N, node_emb_dim=32]
    ▼
DenseOneLayerEncoder      ← maps embeddings to (mu, logvar)
    │  reparameterize → z_i  [N, latent_dim=32]
    ▼
├─ MLPNodeDecoder          → reconstructed REACT values   (reconstruction loss)
└─ MLPEdgeDecoder (×2)     → reconstructed FC values      (reconstruction loss)
    │
    ▼
GraphTransformerPooling    ← node latents → one graph vector [B, 32]
    │  concat with clinical features
    ▼
RegressionMLP              → predicted QIDS score
```

The VGAE is trained jointly: it must **reconstruct the brain graph** *and* **predict the outcome**. The latent space is forced to be Gaussian via KL divergence.

---

## Data Container

### `Outputs` (dataclass)

A simple named container that `NodeLevelVGAE.forward()` returns. Holds everything produced in one forward pass:

| Field | Shape | Description |
|---|---|---|
| `x` | `[N, 3]` | Original node features (REACT receptor values) |
| `rcn_x` | `[N, 3]` | Reconstructed node features |
| `edges` | `[E_triu, 1]` | Original upper-triangular FC values |
| `rcn_edges` | `[E_triu, 1]` | Reconstructed edge attributes |
| `z` | `[N, latent_dim]` | Sampled latent vectors |
| `mu` | `[N, latent_dim]` | Encoder mean |
| `logvar` | `[N, latent_dim]` | Encoder log-variance |

---

## Helper Functions

### `get_device()`
Returns a CUDA device if available, otherwise CPU. Used to move tensors to the right device.

### `L2_reg(model)`
Computes L2 regularisation loss over all weight tensors of a model. Used by every module's `.penalty()` method.

### `get_num_triu_edges(num_nodes)`
Given `N` nodes in a fully-connected graph, returns `N*(N-1)/2` — the number of unique edges (upper triangle only, excluding self-loops).

### `batch_spd(batch, max_spd_dist)` and `get_batched_spd(batch, max_spd_dist)`
These two functions prepare the **Shortest Path Distance (SPD)** matrix for batched processing.

- Graphs may have different numbers of nodes across a batch, so the SPD tensor must be padded to shape `[B, N_max, N_max]`.
- `batch_spd` pads each graph individually and produces a validity mask `[B, N_max]`.
- `get_batched_spd` is the entry point: if all graphs are the same size it does a fast reshape; otherwise it falls back to `batch_spd`.

---

## Step 1 — `StandardMLP`

**What it is:** the base MLP building block used by every other MLP class.

**Architecture:** `Linear → LeakyReLU → Dropout → ... → Linear` (last layer has no activation — subclasses add their own).

```
layer_dims = [in_dim, h1, h2, ..., out_dim]
```

Optional `layernorm=True` applies LayerNorm to the *input* before the first layer.

All other MLP classes (`RegressionMLP`, `MLPNodeDecoder`, `MLPEdgeDecoder`) inherit from this and call `super().forward(x)` to run the stack of layers.

---

## Step 2 — `RegressionMLP`

**What it is:** the prediction head. Takes a graph-level embedding concatenated with clinical features and outputs a scalar QIDS score.

**Input:** `[z_graph (32) || QIDS_base (1) || BDI_base (1) || SSRI (1) || drug (1)]` = 36 dims
**Output:** scalar prediction

Adds two extras on top of `StandardMLP`:
- `.penalty()` — L2 regularisation weighted by `reg_strength`
- `.loss(ypred, ytrue)` — MSE loss + L2 penalty

---

## Step 3 — `EdgeAwareMultiheadAttention`

**What it is:** standard multi-head self-attention with an **additive structural bias** per attention head.

The key insight of the Graphormer architecture: instead of relying on message passing to propagate graph structure, the graph topology is injected *directly into the attention scores*:

```
attn_scores[b, h, i, j] += edge_bias[b, h, i, j]
```

The `edge_bias` tensor (shape `[B, H, N, N]`) encodes both:
- **SPD bias** — how far apart nodes `i` and `j` are in the graph
- **FC edge bias** — the actual connectivity strength between them

**Internals:**
1. Project `x` into Q, K, V: `[B, H, N, d_head]`
2. Compute raw scores: `QK^T / sqrt(d_head)`
3. Add `edge_bias`
4. Apply padding mask (fill `-inf` for invalid nodes)
5. Softmax → dropout → multiply by V → output projection

---

## Step 4 — `GraphormerLayer`

**What it is:** one complete transformer layer wrapping `EdgeAwareMultiheadAttention`.

```
x → EdgeAwareMultiheadAttention → residual + LayerNorm
  → 2-layer FFN (Linear-ReLU-Linear) → residual + LayerNorm
→ x_out
```

The FFN expands to `2 × embed_dim` internally before projecting back. Three of these are stacked inside the encoder.

---

## Step 5 — `NodeEmbeddingGraphormer`

**What it is:** the main graph encoder. Processes the brain connectivity graph and outputs one embedding vector per node.

**Inputs (from `batch`):**
- `batch.x` — REACT receptor values `[N, 3]`
- `batch.xc` — MNI spatial coordinates (conditional, not reconstructed) `[N, 3]`
- `batch.edge_attr` — FC correlation values `[E, 1]`
- `batch.spd` — precomputed shortest-path distances `[N, N]`

**Output:** node embeddings `[N, node_emb_dim=32]`

**Pipeline in detail:**

```
1. concat(x_react, x_mni) → Linear → [N, hidden_dim]
2. SPD distances → Embedding lookup → spd_bias [B, H, N, N]
3. FC edge attrs → 2-layer MLP → edge_bias [B, H, N, N]
4. attn_bias = spd_bias + edge_bias
5. 3 × GraphormerLayer(x, mask, attn_bias)
6. Linear → [N, node_emb_dim=32]
```

Key design choices:
- **SPD bias** uses `nn.Embedding` to learn a different bias per distance level per attention head. Distances beyond `max_spd_dist` are clamped and treated as padding.
- **Edge bias** is computed per edge and scattered into a `[B, H, N, N]` dense tensor.
- The combined `attn_bias` lets every attention head "see" both graph structure and edge weights.
- MNI coordinates are used as **conditional** features (appended to input, but the decoder doesn't reconstruct them).

---

## Step 6 — `DenseOneLayerEncoder`

**What it is:** the variational part — maps node embeddings to Gaussian parameters.

```
h [N, node_emb_dim] → Linear(node_emb_dim → 2 * latent_dim) → split → (mu, logvar)
```

Intentionally minimal: single linear layer, no nonlinearity, no dropout. The reparameterization trick is applied in `NodeLevelVGAE`:

```
z = mu + eps * exp(0.5 * logvar),   eps ~ N(0, I)
```

This makes the latent space differentiable and forces it toward a unit Gaussian via KL divergence in the loss.

---

## Step 7 — `GraphTransformerPooling`

**What it is:** the readout layer — aggregates per-node latent vectors into a single graph-level vector.

**Input:** `z` `[N_total, latent_dim]` (all nodes across the batch) + `batch_index`
**Output:** `[B, pooling_dim=32]`

**Pipeline:**
```
1. to_dense_batch(z) → [B, N_max, D]  (pad variable-size graphs)
2. Multi-head self-attention (no structural bias — pure transformer)
3. Residual + LayerNorm
4. FFN (Linear-ReLU-Linear) + Residual + LayerNorm
5. Mean pooling over valid nodes → [B, D]
```

The extra `.get_attention_weights()` method returns **inbound attention** per node — useful for interpretability (which brain regions the model attends to most when forming the graph summary).

> Note: `HANDLES_CONTEXT = False` means this pooling layer does not support concatenating additional context features internally; context is concatenated outside before passing to `RegressionMLP`.

---

## Step 8 — Decoders

### `MLPNodeDecoder`

Reconstructs REACT receptor values from latent node vectors.

```
z_i [N, latent_dim] → MLP → x̂_i [N, 3]
```

Inherits `StandardMLP`. Uses `act='identity'` by default (linear output). The same MLP is applied independently to every node (shared weights).

### `MLPEdgeDecoder`

Reconstructs edge attributes from *pairs* of node latent vectors.

```
[z_i || z_j] [E, 2*latent_dim] → MLP → ê_{ij} [E, 1]
```

Used **twice** with different activations:
- `act='tanh'` → reconstructed FC correlation value ∈ `[-1, 1]`
- `act='sigmoid'` → edge existence probability ∈ `[0, 1]`

When both decoders are active, the final reconstructed edge = `sigmoid_out × tanh_out`. This factorises "does the edge exist?" from "what is its weight?".

---

## Step 9 — `NodeLevelVGAE` (the full model)

**What it is:** the top-level class that assembles everything above.

### `__init__` / `_build_modules`

Reads config dicts for each submodule and instantiates them, ensuring parameter consistency (e.g., all modules agree on `latent_dim`, `num_node_attr`, etc.).

### `forward(batch) → Outputs`

The full VGAE forward pass:

```python
node_embeddings = node_emb_model(batch)       # Graphormer encoder
mu, logvar = encoder(node_embeddings)         # variational params
z = reparameterize(mu, logvar)                # sample latent

rcn_x = node_decoder(z)                      # reconstruct REACT
rcn_edges = edge_decoder(z, triu_idx)        # reconstruct FC values
if decode_edge_idx:
    rcn_edge_idx = edge_idx_decoder(z, triu_idx)
    rcn_edges = rcn_edge_idx * rcn_edges      # gate by existence prob
```

### `readout(z, context, batch_idx) → z_graph`

Called *after* `forward()` to produce the graph-level embedding used for prediction:

```python
z_graph = pooling(cat([z, context], dim=1), batch_idx)  # [B, 32]
# Then externally: RegressionMLP(cat([z_graph, clinical_features]))
```

### `loss(out) → scalar`

Three-part loss:

| Term | Formula | Purpose |
|---|---|---|
| Reconstruction | `MSE(rcn_x, x) + MSE(rcn_edges, edges)` | Faithfully encode the brain graph |
| KL divergence | `-0.5 * sum(1 + logvar - mu² - exp(logvar))` | Regularise latent space toward N(0,1) |
| L2 regularisation | `reg_strength * sum(weight²)` per module | Prevent overfitting |

### Utility methods

| Method | Purpose |
|---|---|
| `decode(z, triu_idx)` | Decode from a given `z` without a full forward pass |
| `freeze(exclude_pooling)` | Freeze all parameters (e.g., for fine-tuning only the readout) |
| `unfreeze()` | Unfreeze all parameters |
| `reinit_pooling(cfg)` | Replace the pooling layer with a new one (used in transfer learning) |
| `penalty()` | Sum L2 penalties from all submodules |

---

## Summary: How it all connects

```
batch
 ├── .x        [N, 3]   REACT
 ├── .xc       [N, 3]   MNI coords
 ├── .edge_attr [E,1]   FC values
 ├── .edge_index [2,E]
 └── .spd      [N, N]   shortest-path distances

         NodeEmbeddingGraphormer
         ┌──────────────────────────────────┐
         │ concat(x, xc) → input_proj       │
         │ SPD → dist_encoder → spd_bias    │
         │ FC  → edge_encoder → edge_bias   │
         │ attn_bias = spd_bias + edge_bias │
         │ 3 × GraphormerLayer              │
         │ output_proj                      │
         └──────────────┬───────────────────┘
                        │ [N, 32]
                DenseOneLayerEncoder
                        │ (mu, logvar) → reparameterize → z [N, 32]
               ┌────────┴──────────┐
       MLPNodeDecoder        MLPEdgeDecoder(×2)
       rcn_x [N,3]           rcn_edges [E_triu, 1]
               └────────┬──────────┘
                    VGAE loss (MSE + KL + L2)

                GraphTransformerPooling
                        │ [B, 32]
                    concat clinical
                        │ [B, 36]
                   RegressionMLP
                        │
                  QIDS prediction
```
