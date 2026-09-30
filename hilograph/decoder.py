"""Top-down GCN-FiLM decoders (Sec. 2.2, "Top-down GCN-FiLM decoders").

Coarse-to-fine reconstruction from the latent code z:

1. ROI level: z is broadcast to all 148 ROI nodes and concatenated with
   learnable node-identity embeddings; a GCN whose hidden states are
   FiLM-modulated by z produces X_ROI_hat in R^{148 x 5}.
2. 3HG level: each 3HG node receives its parent ROI's decoded features,
   P = M X_ROI_hat in R^{N x 5}. P is concatenated with the broadcast z (and a
   node-identity embedding) and decoded by a second GCN-FiLM into
   X_3HG_hat in R^{N x 5}.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class FiLM(nn.Module):
    """Feature-wise linear modulation of node states by z (Perez et al., 2018).

    Parameterised as (1 + gamma(z)) * h + beta(z) so that it starts near identity.
    """

    def __init__(self, z_dim: int, h_dim: int):
        super().__init__()
        self.gamma = nn.Linear(z_dim, h_dim)
        self.beta = nn.Linear(z_dim, h_dim)

    def forward(self, h, z_node):
        return (1.0 + self.gamma(z_node)) * h + self.beta(z_node)


class GCNFiLM(nn.Module):
    """Input projection -> ``n_layers`` x [GCN -> BatchNorm -> ReLU -> FiLM(z)] -> linear read-out."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, z_dim: int,
                 n_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.inp = nn.Linear(in_dim, hidden_dim)
        self.convs = nn.ModuleList(GCNConv(hidden_dim, hidden_dim) for _ in range(n_layers))
        self.norms = nn.ModuleList(nn.BatchNorm1d(hidden_dim) for _ in range(n_layers))
        self.films = nn.ModuleList(FiLM(z_dim, hidden_dim) for _ in range(n_layers))
        self.out = nn.Linear(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight, z_node):
        h = self.inp(x)
        for conv, norm, film in zip(self.convs, self.norms, self.films):
            h = F.relu(norm(conv(h, edge_index, edge_weight)))
            h = film(h, z_node)
            h = F.dropout(h, self.dropout, self.training)
        return self.out(h)


def local_node_index(batch_vec: torch.Tensor, num_graphs: int) -> torch.Tensor:
    """Position of every node inside its own graph (0, 1, ... per graph)."""
    counts = torch.bincount(batch_vec, minlength=num_graphs)
    starts = torch.cumsum(counts, 0) - counts
    return torch.arange(batch_vec.numel(), device=batch_vec.device) - starts[batch_vec]


class TopDownDecoder(nn.Module):
    def __init__(self, latent_dim: int = 64, hidden_dim: int = 64, feat_dim: int = 5,
                 n_roi: int = 148, max_3hg_nodes: int = 512, node_embed_dim: int = 32,
                 n_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.n_roi = n_roi
        self.max_3hg_nodes = max_3hg_nodes
        self.roi_embed = nn.Embedding(n_roi, node_embed_dim)
        self.hg_embed = nn.Embedding(max_3hg_nodes, node_embed_dim)
        self.roi_dec = GCNFiLM(latent_dim + node_embed_dim, hidden_dim, feat_dim,
                               latent_dim, n_layers, dropout)
        self.hg_dec = GCNFiLM(latent_dim + feat_dim + node_embed_dim, hidden_dim, feat_dim,
                              latent_dim, n_layers, dropout)

    def forward(self, z: torch.Tensor, batch):
        num_graphs = z.size(0)

        # ---- ROI level ----
        z_roi = z[batch.x_roi_batch]
        roi_idx = local_node_index(batch.x_roi_batch, num_graphs)
        x_roi_in = torch.cat([z_roi, self.roi_embed(roi_idx)], dim=-1)
        x_roi_hat = self.roi_dec(x_roi_in, batch.edge_index_roi, batch.edge_weight_roi, z_roi)

        # ---- 3HG level: parent-ROI prior P = M X_ROI_hat ----
        roi_id = batch.roi_id
        assigned = (roi_id >= 0).unsqueeze(-1).to(x_roi_hat.dtype)
        parent = batch.x_3hg_batch * self.n_roi + roi_id.clamp_min(0)
        prior = x_roi_hat[parent] * assigned
        z_hg = z[batch.x_3hg_batch]
        hg_idx = local_node_index(batch.x_3hg_batch, num_graphs)
        if int(hg_idx.max()) >= self.max_3hg_nodes:
            raise ValueError(f"A scan has more than {self.max_3hg_nodes} 3HG nodes; "
                             "increase model.max_3hg_nodes")
        x_hg_in = torch.cat([z_hg, prior, self.hg_embed(hg_idx)], dim=-1)
        x_3hg_hat = self.hg_dec(x_hg_in, batch.edge_index_3hg, batch.edge_weight_3hg, z_hg)
        return x_roi_hat, x_3hg_hat
