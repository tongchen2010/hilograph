"""Hierarchical graph encoder (Sec. 2.2, "Hierarchical graph encoder").

    H_3HG  = GCN_3HG(X_3HG, A_3HG)                       3 layers, R^{N x d}
    H_pool = D^{-1} M^T H_3HG                            Eq. (1), R^{148 x d}
    H_ROI  = GCN_ROI(X_ROI, A_ROI)                       2 layers, R^{148 x d}
    h_bar  = 1/148 sum_i [H_pool,i ; H_ROI,i]
    mu, log sigma^2 = MLP(h_bar)                         Eq. (2)

The ``branches`` argument implements the single-scale ablations:
``("3hg",)`` removes the ROI encoder (3HG-only) and ``("roi",)`` removes the
3HG encoder (ROI-only).
"""

from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    """Stack of GCN layers (BatchNorm + ReLU + dropout between layers)."""

    def __init__(self, in_dim: int, hidden_dim: int, n_layers: int, dropout: float = 0.1):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * n_layers
        self.convs = nn.ModuleList(GCNConv(dims[i], dims[i + 1]) for i in range(n_layers))
        self.norms = nn.ModuleList(nn.BatchNorm1d(hidden_dim) for _ in range(n_layers - 1))
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_weight)
            if i < len(self.convs) - 1:
                x = F.relu(self.norms[i](x))
                x = F.dropout(x, self.dropout, self.training)
        return x


def pool_to_roi(h_3hg: torch.Tensor, roi_id: torch.Tensor, batch_3hg: torch.Tensor,
                num_graphs: int, n_roi: int) -> torch.Tensor:
    """Eq. (1): H_pool = D^{-1} M^T H_3HG, computed for a whole mini-batch.

    ``roi_id[n]`` is the ROI containing 3HG node ``n`` (-1 if unassigned), i.e.
    the non-zero column of row ``n`` of M. ROIs without any 3HG node get a
    zero embedding. Returns a ``(num_graphs * n_roi, d)`` tensor ordered like
    the batched ROI graph.
    """
    valid = roi_id >= 0
    target = batch_3hg[valid] * n_roi + roi_id[valid]
    out = h_3hg.new_zeros(num_graphs * n_roi, h_3hg.size(1))
    out.index_add_(0, target, h_3hg[valid])
    count = h_3hg.new_zeros(num_graphs * n_roi)
    count.index_add_(0, target, h_3hg.new_ones(target.numel()))
    return out / count.clamp_min(1.0).unsqueeze(1)


class HierarchicalEncoder(nn.Module):
    def __init__(self, in_dim: int = 5, hidden_dim: int = 64, latent_dim: int = 64,
                 n_layers_3hg: int = 3, n_layers_roi: int = 2, n_roi: int = 148,
                 dropout: float = 0.1, branches: Sequence[str] = ("3hg", "roi")):
        super().__init__()
        branches = tuple(branches)
        if not branches or set(branches) - {"3hg", "roi"}:
            raise ValueError(f"branches must be a non-empty subset of ('3hg', 'roi'), got {branches}")
        self.branches = branches
        self.n_roi = n_roi
        if "3hg" in branches:
            self.gcn_3hg = GCN(in_dim, hidden_dim, n_layers_3hg, dropout)
        if "roi" in branches:
            self.gcn_roi = GCN(in_dim, hidden_dim, n_layers_roi, dropout)
        fused_dim = hidden_dim * len(branches)
        self.mlp = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.ReLU(),
            nn.Linear(fused_dim, 2 * latent_dim),
        )

    def forward(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        num_graphs = batch.num_graphs
        parts = []
        if "3hg" in self.branches:
            h_3hg = self.gcn_3hg(batch.x_3hg, batch.edge_index_3hg, batch.edge_weight_3hg)
            parts.append(pool_to_roi(h_3hg, batch.roi_id, batch.x_3hg_batch, num_graphs, self.n_roi))
        if "roi" in self.branches:
            parts.append(self.gcn_roi(batch.x_roi, batch.edge_index_roi, batch.edge_weight_roi))
        h = torch.cat(parts, dim=-1).view(num_graphs, self.n_roi, -1)
        h_bar = h.mean(dim=1)                                   # mean over the 148 ROIs
        mu, logvar = self.mlp(h_bar).chunk(2, dim=-1)
        return mu, logvar.clamp(-10.0, 10.0)
