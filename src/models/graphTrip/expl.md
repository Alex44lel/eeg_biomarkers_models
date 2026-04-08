# EEG Graph Model — How It Works

## 1. The Data

We have EEG recordings from 10 subjects during a DMT experiment. Each recording is cut into
**3-second trials** (3000 samples at 1000 Hz). Each trial has **36 channels**, of which we keep
**32 EEG channels** (excluding ECG, VEOG, EMGfront, EMGtemp).

Each trial has a label:
- **0** = pre-injection (baseline)
- **1** = post-injection (DMT effect)

The dataset is balanced per subject: the number of pre-injection trials is trimmed to match
the number of post-injection trials.

| Property | Value |
|----------|-------|
| Total trials | 1152 |
| Channels kept | 32 (standard 10-10 EEG electrodes) |
| Samples per trial | 3000 (3 seconds at 1000 Hz) |
| Subjects | 10 (S01, S02, S04, S05, S06, S07, S08, S10, S12, S13) |
| Labels | 0 (pre, ~559 trials) vs 1 (post, ~593 trials) |

---

## 2. From EEG Trial to Brain Graph

Each 3-second trial is converted into a **graph** where:

```
Nodes = EEG channels (32 nodes)
Node features = spectral band power (5 values per node)
Edges = all channel pairs (fully connected, 32x31 = 992 directed edges)
Edge features = amplitude envelope correlation (1 value per edge)
```

### 2.1 Node Features: Spectral Band Power

For each of the 32 channels, we compute how much power the signal has in each standard
EEG frequency band:

| Band | Frequency range | What it captures |
|------|----------------|------------------|
| Delta | 1 - 4 Hz | Deep sleep, unconscious processes |
| Theta | 4 - 8 Hz | Drowsiness, meditation, memory |
| Alpha | 8 - 13 Hz | Relaxed wakefulness, eyes closed |
| Beta | 13 - 30 Hz | Active thinking, focus, alertness |
| Gamma | 30 - 45 Hz | Higher cognitive functions, binding |

**How we compute it:**

1. Apply **Welch's method** to the 3-second signal of each channel. This estimates the
   Power Spectral Density (PSD) — how much power exists at each frequency.
2. For each band, **integrate the PSD** over that frequency range (trapezoidal rule).
   This gives the total power in that band.
3. Apply **log10 transform** to the power values. EEG power spans several orders of
   magnitude, so the log transform compresses the range and makes it more suitable
   for neural network inputs.

Result: each node (channel) gets a feature vector of **5 values** (one per band).

```
Channel Fp1 → [log_delta, log_theta, log_alpha, log_beta, log_gamma]
Channel Fp2 → [log_delta, log_theta, log_alpha, log_beta, log_gamma]
...
```

### 2.2 Edge Features: Amplitude Envelope Correlation (AEC)

AEC measures how similarly two channels' signal amplitudes fluctuate over time.
It captures **functional connectivity** — which brain regions are "talking" to each other.

**How we compute it:**

1. Apply the **Hilbert transform** to each channel's 3-second signal. This produces an
   analytic signal (complex-valued), from which we can extract the instantaneous amplitude.

2. Take the **absolute value** of the analytic signal to get the **amplitude envelope**.
   The envelope traces the slow fluctuations in signal strength, ignoring the fast
   oscillations.

   ```
   Raw signal:    ~~~\/\/\/\/~~~\/\/\/\/\/~~~
   Envelope:      ___/‾‾‾‾‾\___/‾‾‾‾‾‾‾‾\___
   ```

3. Compute the **Pearson correlation** between every pair of channel envelopes. If two
   channels have envelopes that rise and fall together, their AEC is close to +1. If
   they are unrelated, AEC is near 0.

Result: a **32x32 symmetric correlation matrix**. Each edge in the graph gets the AEC
value between its two endpoint channels.

### 2.3 Conditional Features: Electrode Coordinates (Optional)

With the `--use_coords` flag, each node also gets **3 additional features**: the (x, y, z)
coordinates of that electrode on the scalp, based on the standard 10-10 system.

- x = left(-) to right(+)
- y = posterior(-) to anterior(+)
- z = inferior(-) to superior(+)

These tell the model *where* each channel is physically located, which can help it learn
spatial patterns. Without coords, the model only sees the spectral/connectivity features
and must infer spatial relationships from the data alone.

In the model, these are called **conditional attributes** (`xc`). They are fed into the
encoder but the decoder does NOT try to reconstruct them (unlike node features). This is
because electrode positions are fixed and known — there's nothing to learn about them.

### 2.4 Shortest Path Distance (SPD)

The Graphormer encoder uses SPD as a structural bias in its attention mechanism. Since our
graph is **fully connected** (every channel connects to every other channel), the shortest
path between any two different nodes is always **1**. The SPD matrix is:

```
SPD[i, j] = 0 if i == j
SPD[i, j] = 1 otherwise
```

This means the structural bias is uniform across all pairs — the model relies on the edge
features (AEC) and node features (band powers) to distinguish between channel relationships,
rather than graph topology.

---

## 3. The Model: VGAE + Classifier

The model has two objectives:
1. **Reconstruct** the input graph (VGAE — learn a good representation)
2. **Classify** pre vs post injection (the actual task)

### 3.1 Architecture Overview

```
Input graph (32 nodes, 992 edges)
        │
        ▼
┌─────────────────────────┐
│  NodeEmbeddingGraphormer │  Graphormer encoder
│  (3 transformer layers)  │  Processes node features + edge features + SPD
│                          │  Uses edge-aware multi-head self-attention
└──────────┬───────────────┘
           │ node embeddings [32, node_emb_dim]
           ▼
┌─────────────────────────┐
│  DenseOneLayerEncoder    │  Variational encoder
│  Linear → split          │  Produces (mu, logvar) for each node
└──────────┬───────────────┘
           │ z = reparameterize(mu, logvar)  [32, latent_dim]
           │
     ┌─────┼──────────┐
     │     │          │
     ▼     │          ▼
┌─────────┐│  ┌──────────────┐
│  Node   ││  │    Edge      │
│ Decoder ││  │   Decoder    │   Reconstruction branch (VGAE loss)
│  → x̂   ││  │ → ê (tanh)   │
└─────────┘│  │ × p̂ (sigmoid)│
           │  └──────────────┘
           ▼
┌─────────────────────────┐
│ GraphTransformerPooling  │  Readout layer
│ Self-attention + mean    │  Aggregates 32 node vectors into 1 graph vector
└──────────┬───────────────┘
           │ graph_features [1, latent_dim]
           ▼
┌─────────────────────────┐
│   Classification Head    │  Linear → ReLU → Dropout → Linear → 2 classes
│   → logits [1, 2]       │
└──────────────────────────┘
```

### 3.2 The Graphormer Encoder (NodeEmbeddingGraphormer)

This is the core of the model. It's a transformer adapted for graphs.

**Standard transformer attention:**
```
attention(i, j) = softmax(Q_i · K_j / √d)
```

**Graphormer attention adds two bias terms:**
```
attention(i, j) = softmax(Q_i · K_j / √d  +  spd_bias(i,j)  +  edge_bias(i,j))
```

- **SPD bias**: learned embedding of the shortest path distance between nodes i and j.
  In our case this is always 1 (fully connected), so it acts as a uniform offset.

- **Edge bias**: the AEC value between channels i and j is passed through a small MLP
  to produce a per-head bias. This is how the model "sees" connectivity — strongly
  correlated channels get different attention patterns than weakly correlated ones.

The encoder stacks 3 of these transformer layers, each with:
1. Edge-aware multi-head self-attention
2. Residual connection + LayerNorm
3. Feedforward MLP (2-layer, ReLU)
4. Residual connection + LayerNorm

### 3.3 Variational Encoder (DenseOneLayerEncoder)

A single linear layer maps each node embedding to **(mu, logvar)** — the parameters of a
Gaussian distribution in latent space. During training, we sample from this distribution
(reparameterization trick). During inference, we just use mu.

The **KL divergence** loss pushes these distributions toward N(0,1), which regularizes the
latent space and prevents the model from encoding arbitrary noise.

### 3.4 Decoders (Reconstruction Branch)

Two decoders try to reconstruct the original graph from the latent vectors:

- **Node decoder** (MLP): takes z_i → reconstructs the 5 band powers for node i
- **Edge decoder** (MLP, tanh): takes [z_i || z_j] → reconstructs the AEC value for edge (i,j)
- **Edge index decoder** (MLP, sigmoid): takes [z_i || z_j] → predicts if the edge exists

The reconstruction loss (MSE) forces the latent representation to preserve the information
in the original graph. This is the "autoencoder" part of the VGAE.

### 3.5 Graph Transformer Pooling

After encoding, we have 32 latent vectors (one per channel). We need a single vector to
classify the whole trial. The pooling layer:

1. Runs self-attention across all 32 node vectors (letting nodes exchange information)
2. Residual + LayerNorm
3. Feedforward MLP + Residual + LayerNorm
4. **Mean pools** over all nodes → one vector of size `latent_dim`

### 3.6 Classification Head

A small MLP takes the graph-level vector and outputs 2 logits (pre vs post). Trained with
**cross-entropy loss**.

---

## 4. Training

### Loss Function

```
total_loss = vgae_loss + cls_weight * classification_loss
```

Where:
- `vgae_loss` = node reconstruction MSE + edge reconstruction MSE + KL divergence
- `classification_loss` = CrossEntropyLoss(logits, label)
- `cls_weight` controls the balance (default 1.0)

### Subject-Level Splits

A subject that appears in training NEVER appears in validation. This prevents the model
from memorizing subject-specific patterns and forces it to generalize across individuals.

### Early Stopping

Training stops when the total validation loss hasn't improved for `patience` epochs.
The model from the best epoch is restored for final evaluation.

### Metrics

- **Accuracy**: overall correctness
- **Precision**: of all predicted "post", how many are truly post
- **Recall (sensitivity)**: of all truly post, how many did we catch
- **Specificity**: of all truly pre, how many did we correctly identify as pre
- **NPV (negative predictive value)**: of all predicted "pre", how many are truly pre
- **Confusion matrix**: TP, TN, FP, FN counts

### Latent Space Visualization

After training, we extract the graph-level features (the `latent_dim`-dimensional vectors
from the pooling layer) for all trials, project them to 2D with t-SNE, and plot them
colored by class and by split. This shows whether the model has learned to separate
pre-injection from post-injection in its internal representation.
