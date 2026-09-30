"""HiLoGraph: hierarchical graph encoder -> VAE latent z -> top-down decoders,
plus the auxiliary age regressor f_age (Fig. 1b)."""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from .decoder import TopDownDecoder
from .encoder import HierarchicalEncoder


class HiLoGraph(nn.Module):
    def __init__(self, in_dim: int = 5, hidden_dim: int = 64, latent_dim: int = 64,
                 n_layers_3hg: int = 3, n_layers_roi: int = 2, decoder_layers: int = 3,
                 node_embed_dim: int = 32, n_roi: int = 148, max_3hg_nodes: int = 512,
                 age_hidden_dim: int = 32, dropout: float = 0.1,
                 branches=("3hg", "roi")):
        super().__init__()
        self.encoder = HierarchicalEncoder(in_dim, hidden_dim, latent_dim, n_layers_3hg,
                                           n_layers_roi, n_roi, dropout, branches)
        self.decoder = TopDownDecoder(latent_dim, hidden_dim, in_dim, n_roi, max_3hg_nodes,
                                      node_embed_dim, decoder_layers, dropout)
        # f_age: two-layer MLP predicting chronological age (years) from z. Its output is
        # de-standardised with the training-set age mean/std (see set_age_stats), so the
        # prediction starts at the mean age instead of 0.
        self.age_head = nn.Sequential(
            nn.Linear(latent_dim, age_hidden_dim), nn.ReLU(), nn.Linear(age_hidden_dim, 1))
        self.register_buffer("age_mean", torch.tensor(0.0))
        self.register_buffer("age_std", torch.tensor(1.0))

    @classmethod
    def from_config(cls, model_cfg: Dict[str, Any], n_roi: int = 148) -> "HiLoGraph":
        return cls(n_roi=n_roi, **model_cfg)

    def set_age_stats(self, mean: float, std: float) -> None:
        self.age_mean.fill_(float(mean))
        self.age_std.fill_(max(float(std), 1e-6))

    def predict_age(self, z: torch.Tensor) -> torch.Tensor:
        return self.age_mean + self.age_std * self.age_head(z).squeeze(-1)

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        mu, logvar = self.encoder(batch)
        if self.training:  # reparameterisation trick, Eq. (2)
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        else:
            z = mu
        x_roi_hat, x_3hg_hat = self.decoder(z, batch)
        return {
            "mu": mu, "logvar": logvar, "z": z,
            "x_roi_hat": x_roi_hat, "x_3hg_hat": x_3hg_hat,
            "age_hat": self.predict_age(z),
        }

    @torch.no_grad()
    def embed(self, batch) -> torch.Tensor:
        """Latent code used downstream: the posterior mean of the frozen encoder."""
        mu, _ = self.encoder(batch)
        return mu
