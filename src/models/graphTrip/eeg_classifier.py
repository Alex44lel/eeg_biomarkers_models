"""
EEG Graph Classifier — wraps graphTrip VGAE components for binary classification.

Bypasses the NodeLevelVGAE config system and directly instantiates the components
from model.py. This avoids the broken `models.utils` import chain.

Architecture:
  1. NodeEmbeddingGraphormer  → node embeddings
  2. DenseOneLayerEncoder     → (mu, logvar) → z (reparameterized)
  3. MLPNodeDecoder           → reconstructed node features
  4. MLPEdgeDecoder (tanh)    → reconstructed edge values
  5. MLPEdgeDecoder (sigmoid) → edge existence probability
  6. GraphTransformerPooling  → graph-level vector
  7. Classification head      → 2-class logits

Loss = VGAE loss (reconstruction + KL) + cls_weight * CrossEntropy
"""

import sys
import types
from pathlib import Path

# Mock the `models.utils` import so model.py can be loaded
# We never call init_model/get_model_configs — we instantiate classes directly
_mock_utils = types.ModuleType("models.utils")
_mock_utils.get_model_configs = lambda *a, **kw: {}
_mock_utils.init_model = lambda *a, **kw: None
_mock_pkg = types.ModuleType("models")
_mock_pkg.utils = _mock_utils
sys.modules.setdefault("models", _mock_pkg)
sys.modules.setdefault("models.utils", _mock_utils)

import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch

from model import (
    NodeEmbeddingGraphormer,
    DenseOneLayerEncoder,
    GraphTransformerPooling,
    MLPNodeDecoder,
    MLPEdgeDecoder,
)


class EEGGraphClassifier(nn.Module):
    """
    VGAE + classification head for EEG graph data.

    Parameters
    ----------
    num_node_attr  : int, number of node features (5 = spectral bands)
    num_cond_attrs : int, 0 (no coords) or 3 (electrode x,y,z)
    num_edge_attr  : int, edge feature dim (1 = AEC)
    hidden_dim     : int, internal Graphormer dimension
    node_emb_dim   : int, node embedding output dimension
    latent_dim     : int, VAE latent dimension
    num_layers     : int, number of Graphormer layers
    num_heads      : int, number of attention heads
    dropout        : float
    """

    def __init__(
        self,
        num_node_attr=5,
        num_cond_attrs=0,
        num_edge_attr=1,
        hidden_dim=32,
        node_emb_dim=32,
        latent_dim=32,
        num_layers=3,
        num_heads=4,
        dropout=0.1,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        # Graphormer encoder: graph → node embeddings
        self.node_emb_model = NodeEmbeddingGraphormer(
            num_node_attr=num_node_attr,
            num_cond_attrs=num_cond_attrs,
            num_edge_attr=num_edge_attr,
            hidden_dim=hidden_dim,
            node_emb_dim=node_emb_dim,
            max_spd_dist=2,  # fully connected graph: max SPD = 1
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Variational encoder: node embeddings → (mu, logvar)
        self.var_encoder = DenseOneLayerEncoder(
            input_dim=node_emb_dim,
            latent_dim=latent_dim,
        )

        # Decoders (for VGAE reconstruction loss)
        self.node_decoder = MLPNodeDecoder(
            latent_dim=latent_dim,
            hidden_dim=32,
            output_dim=num_node_attr,
            num_layers=3,
            dropout=dropout,
        )
        self.edge_decoder = MLPEdgeDecoder(
            latent_dim=latent_dim,
            hidden_dim=32,
            num_layers=2,
            dropout=dropout,
            act="tanh",
        )
        self.edge_idx_decoder = MLPEdgeDecoder(
            latent_dim=latent_dim,
            hidden_dim=32,
            num_layers=2,
            dropout=dropout,
            act="sigmoid",
        )

        # Pooling: node latents → graph-level vector
        self.pooling = GraphTransformerPooling(
            pooling_dim=latent_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, 2),
        )

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, batch):
        """
        Returns
        -------
        logits : (B, 2) classification logits
        vgae_data : dict with keys needed for VGAE loss
        graph_feats : (B, latent_dim) graph-level features (for t-SNE)
        """
        # Save originals for reconstruction loss
        x_orig = batch.x.clone()
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr.clone()
        triu_mask = edge_index[0] < edge_index[1]
        triu_edges = edge_attr[triu_mask]
        triu_idx = edge_index[:, triu_mask]

        # Encode
        node_emb = self.node_emb_model(batch)       # (N_total, node_emb_dim)
        mu, logvar = self.var_encoder(node_emb)      # 2x (N_total, latent_dim)
        z = self.reparameterize(mu, logvar)           # (N_total, latent_dim)

        # Decode (for reconstruction loss)
        rcn_x = self.node_decoder(z)                  # (N_total, num_node_attr)
        rcn_edges = self.edge_decoder(z, triu_idx)    # (E_triu, 1)
        rcn_edge_idx = self.edge_idx_decoder(z, triu_idx)
        rcn_edges = rcn_edge_idx * rcn_edges

        # Pool → graph-level vector
        graph_feats = self.pooling(z, batch.batch)    # (B, latent_dim)

        # Classify
        logits = self.classifier(graph_feats)         # (B, 2)

        vgae_data = {
            "x": x_orig,
            "rcn_x": rcn_x,
            "edges": triu_edges,
            "rcn_edges": rcn_edges,
            "mu": mu,
            "logvar": logvar,
        }

        return logits, vgae_data, graph_feats

    def vgae_loss(self, vgae_data):
        """Reconstruction + KL divergence loss."""
        rcn_loss = 0.0
        if vgae_data["rcn_x"] is not None:
            rcn_loss += nn.functional.mse_loss(vgae_data["rcn_x"], vgae_data["x"], reduction="sum")
        if vgae_data["rcn_edges"] is not None:
            rcn_loss += nn.functional.mse_loss(vgae_data["rcn_edges"], vgae_data["edges"], reduction="sum")

        # KL divergence: KL(N(mu, var) || N(0, 1))
        mu, logvar = vgae_data["mu"], vgae_data["logvar"]
        kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

        return rcn_loss + kl_loss
