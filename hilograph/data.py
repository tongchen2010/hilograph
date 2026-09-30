"""Graph construction and data loading (Sec. 2.2, "Graph construction").

Every scan is represented by two graphs that share five cortical morphometric
node features (thickness, average convexity, mean curvature, surface area,
volume):

* ``G_3HG = (X_3HG, A_3HG)`` -- ~400 3-hinge gyri (3HG) landmarks; ``A_3HG`` is
  structural connectivity (SC) with log-normalised fiber-count edge weights.
* ``G_ROI = (X_ROI, A_ROI)`` -- 148 Destrieux parcels with ROI-level SC.

The binary membership matrix ``M in {0,1}^{N x 148}`` is stored compactly as one
ROI index per 3HG node (the column of the single non-zero entry in each row).

Expected on-disk layout (file names are configurable, see ``data.files``)::

    <root>/<subject_id>/<visit>/features/X_3hg.npy                        (N, 5)
    <root>/<subject_id>/<visit>/features/X_roi_des.npy                    (148, 5)
    <root>/<subject_id>/<visit>/networks/SC_3hg_ring5_count_mat_raw.npy   (N, N)
    <root>/<subject_id>/<visit>/networks/SC_roi_des_count_mat_raw.npy     (148, 148)
    <root>/<subject_id>/<visit>/networks/roi_id_3hg_des.npy               (N,)

Single-timepoint cohorts may omit the ``<visit>`` level (leave the metadata
``visit`` column empty or drop it).
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, Sampler, Subset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

FEATURE_NAMES = ("thickness", "convexity", "curvature", "area", "volume")
FOLLOW_BATCH = ["x_3hg", "x_roi"]

DEFAULT_FILES = {
    "x_3hg": "features/X_3hg.npy",
    "x_roi": "features/X_roi_des.npy",
    "sc_3hg": "networks/SC_3hg_ring5_count_mat_raw.npy",
    "sc_roi": "networks/SC_roi_des_count_mat_raw.npy",
    "roi_id": "networks/roi_id_3hg_des.npy",
}


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def parse_sex(values: pd.Series) -> pd.Series:
    """Map sex strings to 1.0 (male) / 0.0 (female); anything else -> NaN."""

    def _one(v):
        if isinstance(v, str):
            v = v.strip().lower()
            if v.startswith("m"):
                return 1.0
            if v.startswith("f"):
                return 0.0
        return np.nan

    return values.map(_one).astype(float)


def read_metadata(csv_path: str | Path, root: str | Path,
                  files: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """Read a scan table and resolve each scan's directory.

    Required columns: ``subject_id``, ``age``. Optional: ``visit``, ``sex``
    (M/F), plus any extra columns (e.g. ``diagnosis``), which are carried
    through to the saved embeddings. Rows whose graph files are missing are
    dropped with a warning.
    """
    files = files or DEFAULT_FILES
    df = pd.read_csv(csv_path, dtype={"subject_id": str, "visit": str})
    for col in ("subject_id", "age"):
        if col not in df.columns:
            raise ValueError(f"{csv_path}: missing required column '{col}'")
    if "visit" not in df.columns:
        df["visit"] = ""
    df["subject_id"] = df["subject_id"].str.strip()
    df["visit"] = df["visit"].fillna("").astype(str).str.strip()
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["sex"] = parse_sex(df["sex"]) if "sex" in df.columns else np.nan

    root = Path(root)
    df["path"] = [str(root / s / v) if v else str(root / s)
                  for s, v in zip(df["subject_id"], df["visit"])]

    ok = np.array([all((Path(p) / f).exists() for f in files.values()) for p in df["path"]])
    ok &= df["age"].notna().to_numpy()
    if (~ok).any():
        warnings.warn(f"{csv_path}: dropping {(~ok).sum()} of {len(df)} rows "
                      "with missing graph files or age")
    return df[ok].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def sc_to_edges(counts: np.ndarray):
    """Symmetric fiber-count matrix -> (edge_index, edge_weight).

    Edge weights are log-normalised fiber counts: ``log(1 + c) / max log(1 + c)``,
    so every graph's weights lie in (0, 1]. Self-connections are removed.
    """
    a = np.asarray(counts, dtype=np.float64)
    a = 0.5 * (a + a.T)
    np.fill_diagonal(a, 0.0)
    a[a < 0] = 0.0
    w = np.log1p(a)
    if w.max() > 0:
        w /= w.max()
    row, col = np.nonzero(w)
    edge_index = torch.from_numpy(np.stack([row, col])).long()
    edge_weight = torch.from_numpy(w[row, col]).float()
    return edge_index, edge_weight


class BrainGraphs(Data):
    """One scan: the 3HG graph, the ROI graph and the 3HG->ROI membership.

    Batching with ``follow_batch=["x_3hg", "x_roi"]`` produces ``x_3hg_batch``
    and ``x_roi_batch`` graph-assignment vectors for the two scales.
    """

    def __inc__(self, key, value, *args, **kwargs):
        if key == "edge_index_3hg":
            return self.x_3hg.size(0)
        if key == "edge_index_roi":
            return self.x_roi.size(0)
        return super().__inc__(key, value, *args, **kwargs)


class ScanDataset(Dataset):
    """Loads the two-scale graphs of each scan listed in a metadata table."""

    def __init__(self, table: pd.DataFrame, files: Optional[Dict[str, str]] = None,
                 n_roi: int = 148, cache: bool = True):
        self.table = table.reset_index(drop=True)
        self.files = files or DEFAULT_FILES
        self.n_roi = n_roi
        self.cache = cache
        self._cache: Dict[int, Dict[str, torch.Tensor]] = {}
        self.stats: Optional[Dict[str, np.ndarray]] = None

    def __len__(self) -> int:
        return len(self.table)

    @property
    def subject_ids(self) -> List[str]:
        return self.table["subject_id"].tolist()

    def raw(self, i: int) -> Dict[str, torch.Tensor]:
        """Un-normalised tensors of scan ``i`` (cached in memory if enabled)."""
        if i in self._cache:
            return self._cache[i]
        d = Path(self.table.at[i, "path"])
        x_3hg = np.load(d / self.files["x_3hg"]).astype(np.float32)
        x_roi = np.load(d / self.files["x_roi"]).astype(np.float32)
        roi_id = np.load(d / self.files["roi_id"]).astype(np.int64).reshape(-1)
        if x_roi.shape[0] != self.n_roi:
            raise ValueError(f"{d}: expected {self.n_roi} ROIs, found {x_roi.shape[0]}")
        if roi_id.shape[0] != x_3hg.shape[0]:
            raise ValueError(f"{d}: roi_id has {roi_id.shape[0]} entries for {x_3hg.shape[0]} 3HG nodes")
        outside = (roi_id < 0) | (roi_id >= self.n_roi)
        if outside.mean() > 0.1:
            warnings.warn(f"{d}: {outside.mean():.0%} of 3HG nodes have no ROI in [0, {self.n_roi}); "
                          "roi_id must be 0-based")
        roi_id[outside] = -1  # unassigned landmarks
        sc_3hg = np.load(d / self.files["sc_3hg"])
        sc_roi = np.load(d / self.files["sc_roi"])
        n = x_3hg.shape[0]
        if sc_3hg.shape != (n, n) or sc_roi.shape != (self.n_roi, self.n_roi):
            raise ValueError(f"{d}: SC shapes {sc_3hg.shape} / {sc_roi.shape} do not match "
                             f"{n} 3HG nodes / {self.n_roi} ROIs")
        for name, arr in (("x_3hg", x_3hg), ("x_roi", x_roi), ("sc_3hg", sc_3hg), ("sc_roi", sc_roi)):
            if not np.isfinite(arr).all():
                raise ValueError(f"{d}: non-finite values in {name}")
        ei_3hg, ew_3hg = sc_to_edges(sc_3hg)
        ei_roi, ew_roi = sc_to_edges(sc_roi)
        item = {
            "x_3hg": torch.from_numpy(x_3hg), "x_roi": torch.from_numpy(x_roi),
            "edge_index_3hg": ei_3hg, "edge_weight_3hg": ew_3hg,
            "edge_index_roi": ei_roi, "edge_weight_roi": ew_roi,
            "roi_id": torch.from_numpy(roi_id),
        }
        if self.cache:
            self._cache[i] = item
        return item

    def set_feature_stats(self, stats: Dict[str, np.ndarray]) -> None:
        self.stats = stats

    def __getitem__(self, i: int) -> BrainGraphs:
        if self.stats is None:
            raise RuntimeError("Call set_feature_stats() before indexing the dataset")
        r = self.raw(i)
        s = self.stats
        row = self.table.iloc[i]
        g = BrainGraphs(
            x_3hg=(r["x_3hg"] - torch.as_tensor(s["x_3hg_mean"])) / torch.as_tensor(s["x_3hg_std"]),
            x_roi=(r["x_roi"] - torch.as_tensor(s["x_roi_mean"])) / torch.as_tensor(s["x_roi_std"]),
            edge_index_3hg=r["edge_index_3hg"], edge_weight_3hg=r["edge_weight_3hg"],
            edge_index_roi=r["edge_index_roi"], edge_weight_roi=r["edge_weight_roi"],
            roi_id=r["roi_id"],
            age=torch.tensor([float(row["age"])], dtype=torch.float32),
            sex=torch.tensor([float(row["sex"])], dtype=torch.float32),
            scan_idx=torch.tensor([i]),
            subject_id=str(row["subject_id"]),
        )
        g.num_nodes = r["x_3hg"].size(0)
        return g


def compute_feature_stats(dataset: ScanDataset, indices: Sequence[int]) -> Dict[str, np.ndarray]:
    """Per-feature mean/std of each scale, computed on training scans only."""
    stats = {}
    for key in ("x_3hg", "x_roi"):
        x = torch.cat([dataset.raw(int(i))[key] for i in indices]).double()
        stats[f"{key}_mean"] = x.mean(0).float().numpy()
        stats[f"{key}_std"] = x.std(0).clamp_min(1e-6).float().numpy()
    return stats


# ---------------------------------------------------------------------------
# Splits and batching
# ---------------------------------------------------------------------------

def subject_folds(subject_ids: Sequence[str], n_folds: int = 5, val_fraction: float = 0.2,
                  seed: int = 42) -> List[Dict[str, np.ndarray]]:
    """Subject-level K-fold CV. Each fold's training subjects are further split
    into train/validation (for early stopping); all visits of a subject stay in
    the same split."""
    sid = np.asarray(subject_ids)
    subjects = np.array(sorted(set(sid)))
    folds = []
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for k, (tr, te) in enumerate(kf.split(subjects)):
        tr_subj = subjects[tr].copy()
        np.random.default_rng(seed + k).shuffle(tr_subj)
        n_val = int(round(len(tr_subj) * val_fraction))
        folds.append({
            "train": np.where(np.isin(sid, tr_subj[n_val:]))[0],
            "val": np.where(np.isin(sid, tr_subj[:n_val]))[0],
            "test": np.where(np.isin(sid, subjects[te]))[0],
        })
    return folds


class SubjectBatchSampler(Sampler[List[int]]):
    """Packs all visits of a subject into the same mini-batch, so that the
    longitudinal losses (Eqs. 3-4) see every within-subject visit pair."""

    def __init__(self, subject_ids: Sequence[str], batch_size: int,
                 shuffle: bool = True, seed: int = 0):
        groups = defaultdict(list)
        for i, s in enumerate(subject_ids):
            groups[s].append(i)
        self.groups = list(groups.values())
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)

    def _pack(self, order) -> Iterator[List[int]]:
        batch: List[int] = []
        for g in order:
            group = self.groups[g]
            if batch and len(batch) + len(group) > self.batch_size:
                yield batch
                batch = []
            batch.extend(group)
        if batch:
            yield batch

    def __iter__(self) -> Iterator[List[int]]:
        n = len(self.groups)
        order = self.rng.permutation(n) if self.shuffle else np.arange(n)
        yield from self._pack(order)

    def __len__(self) -> int:
        # approximate: the packed count can differ by one under shuffling
        return sum(1 for _ in self._pack(np.arange(len(self.groups))))


def make_loader(dataset: ScanDataset, indices: Sequence[int], batch_size: int,
                shuffle: bool = False, seed: int = 0, num_workers: int = 0) -> DataLoader:
    subset = Subset(dataset, [int(i) for i in indices])
    subjects = [dataset.table.at[int(i), "subject_id"] for i in indices]
    sampler = SubjectBatchSampler(subjects, batch_size, shuffle=shuffle, seed=seed)
    return DataLoader(subset, batch_sampler=sampler, follow_batch=FOLLOW_BATCH,
                      num_workers=num_workers)
