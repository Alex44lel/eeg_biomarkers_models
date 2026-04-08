"""
graphTRIP model — cleaned version containing only the classes used in graphTRIP.

Removed: all Medusa-graphTRIP classes (CFRHead, MeanStdPooling), all alternative
encoders (GATv2Conv, NodeEmbeddingMLP), GraphLevelVGAE, and all unused utilities.

Reading order (bottom-up, simplest first):
  1. Helpers
  2. StandardMLP
  3. RegressionMLP
  4. EdgeAwareMultiheadAttention
  5. GraphormerLayer
  6. NodeEmbeddingGraphormer
  7. DenseOneLayerEncoder
  8. GraphTransformerPooling
  9. MLPNodeDecoder  +  MLPEdgeDecoder
  10. NodeLevelVGAE  (assembles everything above)

License: BSD 3-Clause
Authors: Hanna M. Tolle, Lewis J. Ng
"""
from models.utils import get_model_configs, init_model
from dataclasses import dataclass
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_dense_adj, to_dense_batch
import torch
from typing import Optional
import sys
sys.path.append('../')


# ==============================================================================
# Data container
# ==============================================================================

@dataclass
class Outputs:
    """Container for all outputs produced by NodeLevelVGAE.forward()."""
    x: torch.Tensor         # original node features
    rcn_x: torch.Tensor     # reconstructed node features
    edges: torch.Tensor     # original edge attributes (upper triangular)
    rcn_edges: torch.Tensor  # reconstructed edge attributes
    z: torch.Tensor         # sampled latent node vectors  [N, latent_dim]
    mu: torch.Tensor        # encoder mean                 [N, latent_dim]
    logvar: torch.Tensor    # encoder log-variance         [N, latent_dim]


# ==============================================================================
# Helpers
# ==============================================================================

def get_device():
    """Return CUDA device if available, else CPU."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def L2_reg(model):
    """L2 regularization over all weight tensors of a model."""
    l2_loss = 0.
    for name, param in model.named_parameters():
        if 'weight' in name:
            l2_loss += torch.sum(torch.square(param))
    return l2_loss


def get_num_triu_edges(num_nodes: int):
    """Number of upper-triangular edges in a fully connected graph."""
    return num_nodes * (num_nodes - 1) // 2


def batch_spd(batch: Batch, max_spd_dist: int):
    """
    Build a dense SPD tensor [B, N_max, N_max] and a validity mask [B, N_max].
    Needed only when graphs in the same batch have different numbers of nodes.
    """
    batch_vec = batch.batch
    num_nodes_per_graph = torch.bincount(batch_vec)
    N_max = num_nodes_per_graph.max().item()

    spd_list, mask_list = [], []
    start = 0
    for n in num_nodes_per_graph:
        n = n.item()
        spd = getattr(batch, 'spd')[start:start + n, start:start + n] \
            if hasattr(batch, 'spd') else None
        if spd is None:
            raise ValueError("batch.spd not found.")
        pad_size = (0, N_max - n, 0, N_max - n)
        spd_padded = torch.nn.functional.pad(spd, pad_size, value=max_spd_dist + 1)
        spd_list.append(spd_padded)
        mask = torch.zeros(N_max, dtype=torch.bool)
        mask[:n] = True
        mask_list.append(mask)
        start += n

    spd_dense = torch.stack(spd_list, dim=0)   # [B, N_max, N_max]
    mask = torch.stack(mask_list, dim=0)        # [B, N_max]
    return spd_dense, mask


def get_batched_spd(batch, max_spd_dist):
    """Return SPD tensor [B, N, N] and mask [B, N], handling variable-size graphs."""
    B = batch.num_graphs
    num_nodes = batch.num_nodes // B
    spd = batch.spd
    if spd.shape[0] == B * num_nodes and spd.shape[1] == num_nodes:
        spd_dense = spd.view(B, num_nodes, num_nodes)
        mask = torch.ones(B, num_nodes, dtype=torch.bool, device=spd.device)
    else:
        spd_dense, mask = batch_spd(batch, max_spd_dist)
    return spd_dense, mask


# ==============================================================================
# Step 1 — StandardMLP
# Base building block. Every other MLP-based class inherits from this.
# ==============================================================================

class StandardMLP(torch.nn.Module):
    """
    Multi-Layer Perceptron.

    Builds: Linear → LeakyReLU → Dropout → ... → Linear
    The final layer has NO activation (subclasses add their own).

    Parameters
    ----------
    layer_dims : list of ints, e.g. [36, 64, 64, 1]
    dropout    : dropout probability between hidden layers (0 = disabled)
    layernorm  : if True, applies LayerNorm to the input before the first layer
    """

    def __init__(self, layer_dims: list[int],
                 dropout: float = 0.0,
                 layernorm: bool = False):
        super().__init__()

        # Create linear layers
        num_layers = len(layer_dims) - 1
        self.layers = torch.nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(torch.nn.Linear(layer_dims[i], layer_dims[i + 1]))

            # Add activation and dropout
            if i != num_layers - 1:
                self.layers.append(torch.nn.LeakyReLU())
                if dropout > 0:
                    self.layers.append(torch.nn.Dropout(p=dropout))

        # Add layer normalization
        self.layernorm = torch.nn.LayerNorm(layer_dims[0]) if layernorm else None

    def forward(self, x):
        if self.layernorm is not None:
            x = self.layernorm(x)
        for layer in self.layers:
            x = layer(x)
        return x

# ==============================================================================
# Step 2 — RegressionMLP
# The prediction head. Takes [z_graph || clinical_data] → predicted QIDS score.
# ==============================================================================


class RegressionMLP(StandardMLP):
    """
    Regression MLP used as the graphTRIP prediction head.

    Architecture (paper defaults): [36 → 64 → 64 → 1]
      Input 36 = latent_dim(32) + QIDS_base(1) + BDI_base(1) + SSRI(1) + drug(1)

    Parameters
    ----------
    input_dim     : total input size (graph latent + clinical features)
    hidden_dim    : width of hidden layers
    output_dim    : 1 for scalar regression
    num_layers    : total number of linear layers (including output)
    dropout       : dropout between hidden layers
    reg_strength  : L2 weight-decay coefficient
    mse_reduction : 'sum' or 'mean' for the MSE loss
    """

    def __init__(self, input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 num_layers: int = 4,
                 dropout: float = 0.25,
                 layernorm: bool = False,
                 reg_strength: float = 0.01,
                 mse_reduction: str = 'sum'):
        layer_dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        super().__init__(layer_dims, dropout=dropout, layernorm=layernorm)
        self.reg_strength = reg_strength
        self.mse_reduction = mse_reduction

    def penalty(self):
        return self.reg_strength * L2_reg(self)

    def loss(self, ypred, ytrue):
        return torch.nn.functional.mse_loss(ypred, ytrue,
                                            reduction=self.mse_reduction) + self.penalty()


# ==============================================================================
# Step 3 — EdgeAwareMultiheadAttention
# Standard multi-head attention + an additive structural bias per attention head.
# This is the lowest-level building block of the Graphormer encoder.
# ==============================================================================

class EdgeAwareMultiheadAttention(torch.nn.Module):
    """
    Multi-head self-attention with an additive per-head bias term.

    The bias (edge_bias) encodes graph structure: each entry [b, h, i, j]
    is added to the raw attention score between nodes i and j for head h.
    This is how the Graphormer "sees" the graph topology inside attention.

    ----------
    embed_dim  : total embedding dimension (must be divisible by num_heads)
    num_heads  : number of attention heads
    dropout    : atteexplanation_graphTrip_paper.mdntion weight dropout probability
    """

    def __init__(self, embed_dim: int,
                 num_heads: int,
                 dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = torch.nn.Linear(embed_dim, embed_dim)
        self.k_proj = torch.nn.Linear(embed_dim, embed_dim)
        self.v_proj = torch.nn.Linear(embed_dim, embed_dim)
        self.out_proj = torch.nn.Linear(embed_dim, embed_dim)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, mask=None, edge_bias=None):
        B, N, D = x.shape
        H, d_h = self.num_heads, self.head_dim

        # Compute queries, keys, and values for x
        Q = self.q_proj(x).view(B, N, H, d_h).transpose(1, 2)  # [B,H,N,d_h]
        K = self.k_proj(x).view(B, N, H, d_h).transpose(1, 2)
        V = self.v_proj(x).view(B, N, H, d_h).transpose(1, 2)

        # Compute raw attention scores for each node pair
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B,H,N,N]

        # Add edge bias to attention scores, if provided
        if edge_bias is not None:
            attn_scores = attn_scores + edge_bias  # [B,H,N,N]

        # Mask out attention scores for padded nodes
        if mask is not None:
            attn_scores = attn_scores.masked_fill(~mask[:, None, None, :], float('-inf'))

        # Compute attention weights
        attn_weights = torch.softmax(attn_scores, dim=-1)  # [B,H,N,N]
        attn_weights = self.dropout(attn_weights)

        # Compute output
        out = torch.matmul(attn_weights, V)  # [B,H,N,d_h]
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        out = self.out_proj(out)  # mix information across heads
        return out, attn_weights

# ==============================================================================
# Step 4 — GraphormerLayer
# One full transformer layer: attention → residual+LN → FFN → residual+LN.
# Three of these are stacked inside NodeEmbeddingGraphormer.
# ==============================================================================


class GraphormerLayer(torch.nn.Module):
    """
    Single Graphormer layer.

    forward(x, mask, attn_bias):
      1. EdgeAwareMultiheadAttention(x)  — with structural bias
      2. Residual + LayerNorm
      3. 2-layer feedforward MLP (ReLU hidden)
      4. Residual + LayerNorm
    """

    def __init__(self, embed_dim: int,
                 num_heads: int,
                 dropout: float = 0.1,
                 layernorm: bool = True):
        super().__init__()
        self.attn = EdgeAwareMultiheadAttention(embed_dim, num_heads, dropout)
        self.ff = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, 2 * embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * embed_dim, embed_dim)
        )
        self.norm1 = torch.nn.LayerNorm(embed_dim) if layernorm else torch.nn.Identity()
        self.norm2 = torch.nn.LayerNorm(embed_dim) if layernorm else torch.nn.Identity()
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, mask, attn_bias):
        attn_out, _ = self.attn(x, mask=mask, edge_bias=attn_bias)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x


# ==============================================================================
# Step 5 — NodeEmbeddingGraphormer
# The main encoder. Processes the brain graph and returns one embedding per node.
#
# Input:  batch.x   [N, 3]   REACT values (R5-HT1A, R5-HT2A, R5-HTT)
#         batch.xc  [N, 3]   MNI coordinates (conditional, not reconstructed)
#         batch.edge_attr [E, 1]  FC correlation values
#         batch.spd [N, N]   precomputed shortest-path distances
# Output: node embeddings [N, node_emb_dim=32]
# ==============================================================================

class NodeEmbeddingGraphormer(torch.nn.Module):
    """
    Graphormer-style node encoder.

    Pipeline:
      1. concat(x_react, x_mni) → input_proj → [N, hidden_dim]
      2. Compute SPD bias:  dist_encoder(SPD[i,j]) → [B, H, N, N]
      3. Compute edge bias: edge_encoder(FC[i,j])  → [B, H, N, N]
      4. attn_bias = spd_bias + edge_bias
      5. 3 × GraphormerLayer(x, mask, attn_bias)
      6. output_proj → [N, node_emb_dim]

    Parameters
    ----------
    num_node_attr  : number of learned node features (3 for REACT)
    num_cond_attrs : number of conditional node features (3 for MNI coords) - this are called conditional because they are feed into the encoder but the decoder does not need to reconstruct them
    num_edge_attr  : number of edge features (1 for FC correlation)
    hidden_dim     : internal transformer dimension (must be divisible by num_heads)
    node_emb_dim   : output embedding dimension per node
    max_spd_dist   : maximum SPD value to embed (distances beyond this are clamped)
    num_layers     : number of stacked GraphormerLayers
    num_heads      : number of attention heads
    """

    def __init__(self,
                 num_node_attr: int,
                 num_cond_attrs: int,
                 num_edge_attr: int,
                 hidden_dim: int,   # usually higher than input_dim (this is the internal dimension of the transformer)
                 node_emb_dim: int,  # final output dimension that gets pass to the variational encoder
                 max_spd_dist: int = 10,
                 num_layers: int = 3,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 reg_strength: float = 0.01,
                 layernorm: bool = True):
        super().__init__()

        # Validate hidden_dim is compatible with num_heads
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})")

        # Parameters
        self.hidden_dim = hidden_dim
        self.node_emb_dim = node_emb_dim
        self.reg_strength = reg_strength
        self.input_dim = num_node_attr + num_cond_attrs
        self.max_spd_dist = max_spd_dist

        # Embeddings
        self.input_proj = torch.nn.Linear(self.input_dim, hidden_dim)  # [N, 6] → [N, hidden_dim]
        self.edge_encoder = torch.nn.Sequential(
            torch.nn.Linear(num_edge_attr, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, num_heads)
        )
        # each attention head gets its own distance
        self.dist_encoder = torch.nn.Embedding(max_spd_dist + 2, num_heads)  # +2 for padding nodes/inf

        # Transformer layers
        self.layers = torch.nn.ModuleList([
            GraphormerLayer(hidden_dim, num_heads, dropout, layernorm)
            for _ in range(num_layers)
        ])

        self.output_proj = torch.nn.Linear(hidden_dim, node_emb_dim)

    # Notes Ale: PyG stores all graphs concatenated with global node indices
    # Notes Ale: how does batch look like
    # batch.x          [N_total, 3]      REACT values for all nodes in all graphs
    # batch.xc         [N_total, 3]      MNI coordinates for all nodes
    # batch.edge_index [2, E_total]      edge connections for all graphs (stores edges in both directions)
    # batch.edge_attr  [E_total, 1]      FC correlation for all edges
    # batch.spd        [N_total, N]      shortest path distances
    # batch.batch      [N_total]         which graph each node belongs to
    #                                 e.g. [0,0,0,...,1,1,1,...,2,2,2,...]

    def forward(self, batch):
        x, edge_index = batch.x, batch.edge_index
        edge_attr = getattr(batch, 'edge_attr', None)
        x = torch.cat([x, batch.xc], dim=1)  # add conditional attributes
        x = self.input_proj(x)  # [N, hidden_dim], ex: [320, hidden_dim]

        # Dense batching
        x_dense, x_mask = to_dense_batch(x, batch.batch)  # [B, N_graph, hidden_dim], ex:  [4, 80, hidden_dim]
        B, N, D = x_dense.shape

        # SPD embeddings
        spd = getattr(batch, 'spd', None)
        assert spd is not None, "batch.spd must be provided for Graphormer"
        spd_dense, spd_mask = get_batched_spd(batch, self.max_spd_dist)  # [B, N_max, N_max] # ex [4,80,80]
        spd_dense = spd_dense.clamp(max=self.dist_encoder.num_embeddings - 1)
        spd_bias = self.dist_encoder(spd_dense.long())  # [B, N_max, N_max, num_heads]
        spd_bias = spd_bias.permute(0, 3, 1, 2)         # [B, num_heads, N_max, N_max]

        # Make sure masks are equal
        if not torch.equal(x_mask, spd_mask):
            mask = x_mask & spd_mask
        else:
            mask = x_mask

        # Edge feature bias
        edge_bias = torch.zeros_like(spd_bias)  # [B, num_heads, N_max, N_max]
        if edge_attr is not None:
            edge_bias_values = self.edge_encoder(edge_attr)  # [E, num_heads]

            # Convert global edge_index to per-graph local indices
            # Compute per-graph node ranges so we can map to [0..N_g-1]
            node_counts = torch.bincount(batch.batch)  # ex: [80,80,80,80]
            starts = torch.cumsum(torch.cat([torch.tensor([0], device=node_counts.device),
                                             node_counts[:-1]]), dim=0)  # [B] , ex: [0, 80, 160, 240]

            for g in range(B):
                g_mask = (batch.batch[edge_index[0]] == g)  # g_mask selects edjes that belong to graph G
                # global indices, # e.g. for graph 1: [[80, 81, 82,...], [81, 80, 83,...]]
                e_idx = edge_index[:, g_mask]
                start = starts[g]
                e_idx_local = e_idx - start  # map to [0..N_g-1], converts to local indices
                e_attr = edge_bias_values[g_mask]  # [E_g, num_heads]
                # scatter into the [B,num_heads,N,N] bias tensor
                edge_bias[g, :, e_idx_local[0], e_idx_local[1]] = e_attr.T  # [num_heads, E_g]

        # Combine attention biases
        attn_bias = spd_bias + edge_bias  # [B, num_heads, N_max, N_max]

        # Transformer layers
        for layer in self.layers:
            x_dense = layer(x_dense, mask, attn_bias)

        # Output projection
        # ex x_dense[4,80,128] , mask [4,80] all true then x_out= [320,128]
        x_out = x_dense[mask]
        x_out = self.output_proj(x_out)
        return x_out

    def penalty(self):
        return self.reg_strength * L2_reg(self)


# ==============================================================================
# Step 6 — DenseOneLayerEncoder
# Maps node embeddings to (mu, logvar) for the variational reparameterization.
# Intentionally minimal: single linear layer, no hidden layers, no dropout.
# ==============================================================================

class DenseOneLayerEncoder(torch.nn.Module):
    """
    Variational encoder: maps node embeddings to Gaussian parameters.

    forward(h) → (mu, logvar)
      where h: [N, node_emb_dim],  mu/logvar: [N, latent_dim]

    A single Linear(node_emb_dim → 2*latent_dim) split into two halves.
    """

    def __init__(self, input_dim: int,
                 latent_dim: int,
                 reg_strength: float = 0.01):
        super().__init__()
        self.layer = torch.nn.Linear(input_dim, latent_dim * 2)
        self.reg_strength = reg_strength
        self.latent_dim = latent_dim

    def forward(self, embeddings):
        return torch.chunk(self.layer(embeddings), 2, dim=1)   # (mu, logvar)

    def penalty(self):
        return self.reg_strength * L2_reg(self)


# ==============================================================================
# Step 7 — GraphTransformerPooling
# Aggregates node-level latent vectors z_i into one graph-level vector z.
# Used as the "readout layer" between the VGAE and the MLP prediction head.
# ==============================================================================

class GraphTransformerPooling(torch.nn.Module):
    """
    Readout layer: node latents → graph-level vector.

    Pipeline:
      1. Multi-head self-attention across all nodes (no structural bias)
      2. Residual + LayerNorm
      3. Feedforward MLP + Residual + LayerNorm
      4. Mean pooling over valid nodes

    Output dim = pooling_dim (same as latent_dim = 32).

    Parameters
    ----------
    pooling_dim   : embedding dimension (= latent_dim from VGAE)
    num_heads     : attention heads
    ff_hidden_dim : feedforward hidden size (defaults to 2 * pooling_dim)
    dropout       : dropout probability
    reduce        : 'mean' or 'sum' pooling after attention
    """
    HANDLES_CONTEXT = False  # NOTES_ALE: clinical features are concatenated after pooling

    @classmethod
    def can_handle_context(cls):
        return cls.HANDLES_CONTEXT

    def __init__(self, pooling_dim: int,
                 num_heads: int = 4,
                 ff_hidden_dim: int = None,
                 dropout: float = 0.1,
                 reg_strength: float = 0.01,
                 reduce: str = 'mean'):
        super().__init__()
        self.pooling_dim = pooling_dim
        self.num_heads = num_heads
        self.reduce = reduce
        self.output_dim = pooling_dim
        self.dropout = torch.nn.Dropout(dropout)

        # Multi-head self-attention
        self.attn = torch.nn.MultiheadAttention(pooling_dim,
                                                num_heads, dropout=dropout, batch_first=True)

        # Feedforward network (position-wise)
        ff_hidden_dim = ff_hidden_dim or 2 * pooling_dim
        self.ff = torch.nn.Sequential(
            torch.nn.Linear(pooling_dim, ff_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(ff_hidden_dim, pooling_dim),
        )
        self.reg_strength = reg_strength
        self.norm1 = torch.nn.LayerNorm(pooling_dim)
        self.norm2 = torch.nn.LayerNorm(pooling_dim)

    def forward(self, z, batch_index):
        # Convert node features to dense batch form: [B, N_max, D]
        z_dense, mask = to_dense_batch(z, batch_index)
        # z_dense: [batch_size, num_nodes, embed_dim]
        # mask: [batch_size, num_nodes]

        # Self-attention: contextualise node embeddings within each graph
        attn_out, _ = self.attn(z_dense, z_dense, z_dense,
                                key_padding_mask=~mask)  # mask=False -> attend only to valid nodes
        z_dense = self.norm1(z_dense + self.dropout(attn_out))

        # Feed-forward transformation
        ff_out = self.ff(z_dense)
        z_dense = self.norm2(z_dense + self.dropout(ff_out))

        # Pooling: mean or attention over valid nodes
        if self.reduce == 'mean':
            pooled = (z_dense * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True)
        elif self.reduce == 'sum':
            pooled = (z_dense * mask.unsqueeze(-1)).sum(dim=1)
        else:
            raise ValueError("reduce must be 'mean' or 'sum'")

        return pooled

    def get_attention_weights(self, z, batch_index):
        """
        Returns the self-attention weights (averaged over heads)
        for each graph in the batch.

        Parameters
        ----------
        z : torch.Tensor
            Node embeddings [num_nodes, pooling_dim]
        batch_index : torch.Tensor
            Batch vector mapping each node to its graph [num_nodes]

        Returns
        -------
        attn_avg : torch.Tensor
            Attention weight matrices averaged over heads [batch_size, num_nodes, num_nodes],
            where N = max number of nodes in the batch.
        mask : torch.Tensor
            Boolean mask indicating valid nodes [batch_size, num_nodes]
        """

        # Multi-head self-attention
        z_dense, mask = to_dense_batch(z, batch_index)
        _, attn_weights = self.attn(
            z_dense, z_dense, z_dense,
            key_padding_mask=~mask,
            need_weights=True,
            average_attn_weights=False  # keep per-head attention
        )  # [batch_size, num_heads, num_nodes, num_nodes]

        # Average over heads
        attn_avg = attn_weights.mean(dim=1)  # [batch_size, num_nodes, num_nodes]

        # Mask out invalid nodes (padding)
        attn_avg = attn_avg * mask[:, None, :].float() * mask[:, :, None].float()

        # Compute inbound attention weights
        inbound = attn_avg.sum(dim=-2)
        inbound = inbound * mask.float() / mask.sum(dim=-2, keepdim=True).float()
        # outbound = attn_avg.sum(dim=-1)
        # outbound = outbound * mask.float() / mask.sum(dim=-1, keepdim=True).float()

        return inbound

    def penalty(self):
        return self.reg_strength * L2_reg(self)


# ==============================================================================
# Step 8 — Decoders
# Two decoders reconstruct the brain graph from node latent vectors z_i.
# They are used INSIDE NodeLevelVGAE to compute the reconstruction loss.
# ==============================================================================

class MLPNodeDecoder(StandardMLP):
    """
    Reconstructs node features (REACT values) from latent node vectors.

    forward(z_i) → x̂_i
      Input:  z_i  [N_total, latent_dim]
      Output: x̂_i [N_total, num_node_attr=3]

    Weights are shared across all nodes (same MLP applied node-by-node).
    """

    def __init__(self,
                 latent_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 num_layers: int = 3,
                 dropout: float = 0.25,
                 reg_strength: float = 0.01,
                 act: str = 'identity'):
        layer_dims = [latent_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        super().__init__(layer_dims, dropout=dropout, layernorm=False)
        self.reg_strength = reg_strength
        self.act = (lambda x: x) if act == 'identity' else activation_resolver(act)

    def forward(self, z):
        # z shape: [nodes_in_batch, latent_dim + extra_dim]
        return self.act(super().forward(z))

    def penalty(self):
        return self.reg_strength * L2_reg(self)


class MLPEdgeDecoder(StandardMLP):
    """
    Reconstructs edge attributes (FC values or edge existence) from pairs of
    node latent vectors.

    forward([z_i || z_j]) → ê_{ij}
      Input:  concatenation of node pairs  [E, 2*latent_dim]
      Output: scalar per edge              [E, 1]

    Used twice in graphTRIP:
      - Edge attribute decoder: act='tanh'    → FC value ∈ [-1, 1]
      - Edge index decoder:     act='sigmoid' → existence probability ∈ [0, 1]

    Reconstructed FC matrix = sigmoid_output × tanh_output (element-wise).
    """

    def __init__(self, latent_dim: int,
                 num_layers: int = 2,
                 hidden_dim: int = 32,
                 reg_strength: float = 0.01,
                 dropout: float = 0.25,
                 act: str = 'tanh'):
        layer_dims = [latent_dim * 2] + [hidden_dim] * (num_layers - 1) + [1]
        super().__init__(layer_dims, dropout=dropout, layernorm=False)
        self.reg_strength = reg_strength
        self.act = activation_resolver(act)

    def forward(self, z, edge_idx):
        node_pairs = torch.cat([z[edge_idx[0]], z[edge_idx[1]]], dim=1)
        return self.act(super().forward(node_pairs))

    def penalty(self):
        return self.reg_strength * L2_reg(self)


# ==============================================================================
# Step 9 — NodeLevelVGAE
# The full VGAE. Assembles all components above into one forward pass.
#
# forward(batch) → Outputs
#   1. node_emb_model  (NodeEmbeddingGraphormer) → node embeddings [N, 32]
#   2. encoder         (DenseOneLayerEncoder)    → (mu, logvar)    [N, 32]
#   3. reparameterize                            → z_i             [N, 32]
#   4. node_decoder    (MLPNodeDecoder)          → reconstructed REACT
#   5. edge_decoder    (MLPEdgeDecoder, tanh)    → reconstructed FC values
#   6. edge_idx_decoder(MLPEdgeDecoder, sigmoid) → edge existence probs
#
# readout(z, context, batch_idx) → z_graph [B, 32]
#   7. pooling (GraphTransformerPooling)         → graph-level vector
#
# The graph-level vector z_graph is then concatenated with clinical features
# and passed to RegressionMLP to predict the QIDS score.
# ==============================================================================

class NodeLevelVGAE(torch.nn.Module):
    """
    Node-level Variational Graph Autoencoder — the graphTRIP VGAE.

    Each node has its own latent Gaussian. The VGAE is trained jointly with
    the downstream MLP to optimise both graph reconstruction and outcome
    prediction.

    Parameters (all passed as dicts with 'model_type' and 'params' keys)
    ----------
    params              : shared graph parameters (num_nodes, num_node_attr, etc.)
    node_emb_model_cfg  : config for NodeEmbeddingGraphormer
    encoder_cfg         : config for DenseOneLayerEncoder
    pooling_cfg         : config for GraphTransformerPooling
    node_decoder_cfg    : config for MLPNodeDecoder   (optional)
    edge_decoder_cfg    : config for MLPEdgeDecoder   (tanh, optional)
    edge_idx_decoder_cfg: config for MLPEdgeDecoder   (sigmoid, optional)
    """
    # NOTES_ALE NodeLevelVGAE owns the required params
    # NOTES_ALE Node_features for the paper usa case are scalars

    def __init__(self, params: dict,
                 node_emb_model_cfg: dict,
                 encoder_cfg: dict,
                 pooling_cfg: dict,
                 node_decoder_cfg: Optional[dict] = None,
                 edge_decoder_cfg: Optional[dict] = None,
                 edge_idx_decoder_cfg: Optional[dict] = None):
        super().__init__()
        # Get shared parameters
        self.params = get_model_configs(self.__class__.__name__, **params)
        self.decode_edge_idx = False
        if edge_idx_decoder_cfg:
            self.decode_edge_idx = True

        # Make sure that submodule configs match params
        self._build_modules(node_emb_model_cfg,
                            pooling_cfg,
                            encoder_cfg,
                            edge_decoder_cfg,
                            edge_idx_decoder_cfg,
                            node_decoder_cfg)

    def _build_modules(self, node_emb_model_cfg,
                       pooling_cfg,
                       encoder_cfg,
                       edge_decoder_cfg,
                       edge_idx_decoder_cfg,
                       node_decoder_cfg):
        # Node embedding model
        updated_params = self._get_module_params(node_emb_model_cfg)
        self.node_emb_model = init_model(node_emb_model_cfg['model_type'], updated_params)

        # Encoder
        updated_params = self._get_module_params(encoder_cfg)
        updated_params['input_dim'] = self.params['node_emb_dim']
        self.encoder = init_model(encoder_cfg['model_type'], updated_params)

        # Pooling/readout layer
        updated_params = self._get_module_params(pooling_cfg)
        updated_params['pooling_dim'] = self.params['latent_dim']

        # Add num_context_attrs if pooling layer handles itx
        pooling_class = globals()[pooling_cfg['model_type']]  # pooling_cfg['model_type'] = "GraphTransformerPooling"
        if pooling_class.can_handle_context():
            updated_params['num_context_attrs'] = self.params['num_context_attrs']
        elif self.params['num_context_attrs'] > 0:
            raise ValueError(f"Pooling layer '{pooling_cfg['model_type']}' cannot handle context attributes. "
                             f"Set 'num_context_attrs' to 0 or change pooling layer.")
        self.pooling = init_model(pooling_cfg['model_type'], updated_params)

        # Decoders
        self.edge_decoder, self.edge_idx_decoder, self.node_decoder = None, None, None
        if edge_decoder_cfg:
            if not edge_decoder_cfg['model_type'].endswith('EdgeDecoder'):
                raise ValueError(f"Edge decoder must be an 'EdgeDecoder', got '{edge_decoder_cfg['model_type']}'")
            updated_params = self._get_module_params(edge_decoder_cfg)
            self.edge_decoder = init_model(edge_decoder_cfg['model_type'], updated_params)

        if edge_idx_decoder_cfg:
            assert edge_decoder_cfg, "Edge decoder must be specified if edge index decoder is specified"
            updated_params = self._get_module_params(edge_idx_decoder_cfg)
            self.edge_idx_decoder = init_model(edge_idx_decoder_cfg['model_type'], updated_params)

        if node_decoder_cfg:
            updated_params = self._get_module_params(node_decoder_cfg)
            updated_params['output_dim'] = self.params['num_node_attr']
            self.node_decoder = init_model(node_decoder_cfg['model_type'], updated_params)

        self.modules = [self.node_emb_model,
                        self.encoder,
                        self.pooling,
                        self.edge_decoder,
                        self.edge_idx_decoder,
                        self.node_decoder]
        self.readout_dim = self.pooling.output_dim  # stores the output size of pooling layer

    def _get_module_params(self, submodule_cfg):
        '''
        Get all required and optional parameters for a submodule.
        '''
        # Get all required and optional parameters
        all_params = get_model_configs(submodule_cfg['model_type'], **self.params)
        # These parameters must be consistent across all submodules
        must_be_consistent = ['num_nodes',
                              'num_edge_attr',
                              'num_node_attr',
                              'num_graph_attr',
                              'latent_dim',
                              'node_emb_dim',
                              'num_cond_attrs']
        # Override default optional parameters with user inputs
        for k, v in submodule_cfg['params'].items():
            if k in all_params:
                if k not in must_be_consistent and v is not None:
                    all_params[k] = v
            else:
                raise ValueError(f"Unknown parameter '{k}' for model type '{submodule_cfg['model_type']}'")
        return all_params

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, batch: Data):
        '''
        Parameters:
        ----------
        batch (Data): Batch of data objects from PyTorch Geometric.
        '''
        # Save a copy of the original inputs for the decoder, triu_idx are the upper triangular edges
        x, edges, triu_idx = self._get_decoder_labels(batch)

        # Encode node features
        node_embeddings = self.node_emb_model(batch)       # [nodes_in_batch, node_emb_dim]
        mu, logvar = self.encoder(node_embeddings)         # 2 x [nodes_in_batch, latent_dim]
        z = self.reparameterize(mu, logvar)                # [nodes_in_batch, latent_dim]

        # Decode node features
        rcn_x = None
        if self.node_decoder is not None:
            rcn_x = self.node_decoder(z)                   # [nodes_in_batch, num_node_attr]

        # Decode edge features
        rcn_edges = None
        if self.edge_decoder is not None:
            rcn_edges = self.edge_decoder(z, triu_idx)     # [num_triu_edges, num_edge_attr]

            # Decode edge indices (decodes if an edge exitsts or not)
            if self.decode_edge_idx:
                rcn_edge_idx = self.edge_idx_decoder(z, triu_idx)  # [num_triu_edges, 1] between 0 and 1
                rcn_edges = rcn_edge_idx * rcn_edges

        # Wrap outputs
        out = Outputs(x=x,
                      rcn_x=rcn_x,
                      edges=edges,
                      rcn_edges=rcn_edges,
                      z=z, mu=mu, logvar=logvar)
        return out

    def decode(self, z, triu_idx):
        '''
        Decodes the latent sample (num_nodes x latent_dim) 
        into node features and edge features.
        '''
        rcn_x, rcn_edges = None, None
        if self.node_decoder is not None:
            rcn_x = self.node_decoder(z)
        if self.edge_decoder is not None:
            rcn_edges = self.edge_decoder(z, triu_idx)
            if self.decode_edge_idx:
                rcn_edge_idx = self.edge_idx_decoder(z, triu_idx)
                rcn_edges = rcn_edge_idx * rcn_edges
        return rcn_x, rcn_edges

    def readout(self, z, context, batch_idx):
        '''Pools across nodes and returns a graph embeddings.'''
        z_with_context = torch.cat([z, context], dim=1)
        return self.pooling(z_with_context, batch_idx)

    def _get_decoder_labels(self, batch: Data):
        # Node features
        x = batch.x.clone()                          # [nodes_in_batch, num_node_attr]

        # Edge features, if all graphs have the same edge indices
        if not self.decode_edge_idx:
            edges = batch.edge_attr.clone()          # [edges_in_batch, num_edge_attr]
            edge_idx = batch.edge_index              # [2, edges_in_batch]
            triu_mask = edge_idx[0] < edge_idx[1]    # [edges_in_batch]
            triu_edges = edges[triu_mask]            # [num_triu_edges, num_edge_attr]
            triu_idx = edge_idx[:, triu_mask]        # [2, num_triu_edges]

        # Edges features, if graphs have different edge indices
        else:
            triu_edges, triu_idx = self._get_edge_decoder_labels(batch)

        return x, triu_edges, triu_idx

    def _get_edge_decoder_labels(self, batch: Data):
        '''
        Get zero-filled triu edges and indices for fully connected graphs,
        if graphs have different adjacency matrices (but the same number of nodes).
        '''
        # Get edge features
        edges = batch.edge_attr.clone()  # [edges_in_batch, num_edge_attr]

        # Convert to dense adjacency (zero-fills missing edges)
        # [batch_size, num_nodes, num_nodes, num_edge_attr]
        dense_edges = to_dense_adj(batch.edge_index, batch.batch, edges)

        # Get triu indices for each graph in batch
        num_nodes = dense_edges.size(1)
        triu_idx = torch.triu_indices(num_nodes, num_nodes, offset=1)

        # Extract upper triangular edges from each graph and concatenate
        # [batch_size * num_triu_edges, num_edge_attr]
        triu_edges = dense_edges[:, triu_idx[0], triu_idx[1]].reshape(-1, edges.shape[1])

        # Get triu indices for graphs inside batch
        offset = 0
        triu_indices = []
        for i in range(batch.num_graphs):
            idx = torch.triu_indices(num_nodes, num_nodes, offset=1)
            triu_indices.append(idx + offset)
            offset += num_nodes
        triu_idx = torch.cat(triu_indices, dim=1).to(edges.device)

        return triu_edges, triu_idx

    def freeze(self, exclude_pooling: bool = False):
        '''Freeze all modules except for the readout layer.'''
        for module in self.modules:
            if module is not None and (not exclude_pooling or module != self.pooling):
                for param in module.parameters():
                    param.requires_grad = False

    def unfreeze(self):
        '''Unfreeze all modules.'''
        for module in self.modules:
            if module is not None:
                for param in module.parameters():
                    param.requires_grad = True

    def reinit_pooling(self, new_pooling_cfg: dict = None):
        ''' Reinitialise the pooling, possibly with new config.'''
        if new_pooling_cfg is not None:
            # Check if new configs are valid
            updated_params = self._get_module_params(new_pooling_cfg)
            updated_params['pooling_dim'] = self.params['latent_dim']
            pooling_class = globals()[new_pooling_cfg['model_type']]
            if pooling_class.can_handle_context():
                updated_params['num_context_attrs'] = self.params['num_context_attrs']
            elif self.params['num_context_attrs'] > 0:
                raise ValueError(f"Pooling layer '{new_pooling_cfg['model_type']}' \
                                   cannot handle context attributes. "
                                 f"Set 'num_context_attrs' to 0 or change pooling layer.")
            self.pooling = init_model(new_pooling_cfg['model_type'], new_pooling_cfg['params'])
        else:
            # If no new config, we don't need to check them again
            self.pooling = init_model(self.pooling_cfg['model_type'], self.pooling_cfg['params'])
        # Update readout dimension and modules list
        self.readout_dim = self.pooling.output_dim
        self.modules[2] = self.pooling

    def penalty(self):
        reg_loss = sum(m.penalty() for m in self.modules if m is not None)
        return reg_loss

    def loss(self, out):
        # Reconstruction loss
        rcn_loss = 0.
        if out.rcn_x is not None:
            rcn_loss += torch.nn.functional.mse_loss(out.rcn_x, out.x, reduction='sum')
        if out.rcn_edges is not None:
            rcn_loss += torch.nn.functional.mse_loss(out.rcn_edges, out.edges, reduction='sum')

        # KL divergence
        # KL divergence between VAE's learned distribution N(mu, var) and prior N(0,1)
        # Formula: KL(N(mu,var) || N(0,1)) = -0.5 * sum(1 + log(var) - mu^2 - var)
        # out.logvar is log(var), so var = exp(logvar)
        kl_loss = torch.mean(-0.5 * torch.sum(1 + out.logvar - out.mu.pow(2) - out.logvar.exp(), dim=1))

        # Regularization
        reg_loss = self.penalty()

        return rcn_loss + kl_loss + reg_loss
