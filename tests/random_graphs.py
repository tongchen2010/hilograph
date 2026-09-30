"""Random graphs in the expected on-disk format, used only as unit-test fixtures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from hilograph.data import DEFAULT_FILES


def _sparse_counts(n: int, density: float, rng) -> np.ndarray:
    upper = np.triu(rng.random((n, n)) < density, 1)
    counts = np.where(upper, rng.integers(1, 2000, (n, n)), 0)
    counts = counts + counts.T
    np.fill_diagonal(counts, rng.integers(1000, 5000, n))
    return counts.astype(np.float32)


def write_scan(scan_dir: Path, age: float, rng, n_roi: int = 148,
               n_nodes: int = 80, thickness_shift: float = 0.0) -> None:
    roi_id = rng.integers(0, n_roi, n_nodes).astype(np.int32)
    roi_id[rng.random(n_nodes) < 0.02] = 999                       # a few unassigned landmarks
    thickness = 2.9 - 0.012 * (age - 36) - thickness_shift + 0.2 * rng.standard_normal(n_nodes)
    x_3hg = np.column_stack([
        thickness,
        -6.5 + 2.4 * rng.standard_normal(n_nodes),
        -0.3 + 0.07 * rng.standard_normal(n_nodes),
        0.55 + 0.15 * rng.standard_normal(n_nodes),
        thickness * 0.6 + 0.2 * rng.standard_normal(n_nodes),
    ]).astype(np.float32)
    x_roi = np.column_stack([
        2.5 - 0.01 * (age - 36) - thickness_shift + 0.1 * rng.standard_normal(n_roi),
        0.5 + 0.2 * rng.standard_normal(n_roi),
        -0.02 + 0.01 * rng.standard_normal(n_roi),
        1000 + 200 * rng.standard_normal(n_roi),
        2500 + 500 * rng.standard_normal(n_roi),
    ]).astype(np.float32)
    arrays = {
        "x_3hg": x_3hg, "x_roi": x_roi, "roi_id": roi_id,
        "sc_3hg": _sparse_counts(n_nodes, 0.1, rng),
        "sc_roi": _sparse_counts(n_roi, 0.5, rng),
    }
    for key, rel in DEFAULT_FILES.items():
        path = scan_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, arrays[key])


def make_random_cohorts(out_dir: str | Path, n_healthy: int = 40, n_clinical: int = 45,
                     seed: int = 0):
    """Writes ``healthy/`` (longitudinal, 1-3 visits) and ``clinical/``
    (single visit, CN/AD/LBD labels) plus their metadata CSVs."""
    out_dir = Path(out_dir)
    rng = np.random.default_rng(seed)

    rows = []
    for s in range(n_healthy):
        sid = f"H{s:04d}"
        sex = rng.choice(["M", "F"])
        age = float(rng.uniform(36, 85))
        for v in range(int(rng.integers(1, 4))):
            write_scan(out_dir / "healthy" / sid / f"V{v + 1}", age, rng,
                       n_nodes=int(rng.integers(60, 100)))
            rows.append({"subject_id": sid, "visit": f"V{v + 1}", "age": round(age, 2), "sex": sex})
            age += float(rng.uniform(1.0, 3.0))
    healthy_csv = out_dir / "healthy_scans.csv"
    pd.DataFrame(rows).to_csv(healthy_csv, index=False)

    labels = ["CN", "AD", "LBD"]
    shift = {"CN": 0.0, "AD": 0.35, "LBD": 0.25}
    rows = []
    for s in range(n_clinical):
        sid = f"C{s:04d}"
        dx = labels[s % len(labels)]
        age = float(rng.uniform(55, 85))
        write_scan(out_dir / "clinical" / sid, age, rng, n_nodes=int(rng.integers(60, 100)),
                   thickness_shift=shift[dx])
        rows.append({"subject_id": sid, "age": round(age, 1), "sex": rng.choice(["M", "F"]),
                     "diagnosis": dx})
    clinical_csv = out_dir / "clinical_scans.csv"
    pd.DataFrame(rows).to_csv(clinical_csv, index=False)
    return healthy_csv, clinical_csv
