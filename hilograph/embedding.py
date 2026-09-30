"""Checkpoint loading and latent-code extraction with a frozen encoder."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from .data import ScanDataset, make_loader
from .model import HiLoGraph


def stats_to_lists(stats: Dict[str, np.ndarray]) -> Dict[str, list]:
    return {k: np.asarray(v).tolist() for k, v in stats.items()}


def stats_from_lists(stats: Dict[str, list]) -> Dict[str, np.ndarray]:
    return {k: np.asarray(v, dtype=np.float32) for k, v in stats.items()}


def load_checkpoint(path: str | Path, device: str = "cpu") -> Tuple[HiLoGraph, dict]:
    ckpt = torch.load(path, map_location=device)
    cfg = ckpt["config"]
    model = HiLoGraph.from_config(cfg["model"], n_roi=cfg["data"]["n_roi"])
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), ckpt


@torch.no_grad()
def embed_scans(model: HiLoGraph, dataset: ScanDataset, indices: Sequence[int],
                batch_size: int = 32, device: str = "cpu") -> np.ndarray:
    """Posterior-mean latent codes for ``indices``, returned in the same order."""
    model.eval()
    codes = {}
    for batch in make_loader(dataset, indices, batch_size, shuffle=False):
        batch = batch.to(device)
        mu = model.embed(batch).cpu().numpy()
        for idx, m in zip(batch.scan_idx.tolist(), mu):
            codes[idx] = m
    return np.stack([codes[int(i)] for i in indices])


def embeddings_table(dataset: ScanDataset, indices: Sequence[int], mu: np.ndarray,
                     split: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Scan metadata (without file paths) + one ``z<k>`` column per latent dimension."""
    meta = dataset.table.iloc[list(indices)].drop(columns=["path"]).reset_index(drop=True)
    if split is not None:
        meta.insert(2, "split", list(split))
    z = pd.DataFrame(mu, columns=[f"z{k}" for k in range(mu.shape[1])])
    return pd.concat([meta, z], axis=1)
