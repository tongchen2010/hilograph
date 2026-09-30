import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from hilograph.data import (ScanDataset, SubjectBatchSampler, compute_feature_stats, make_loader,
                            read_metadata, sc_to_edges, subject_folds)
from hilograph.encoder import pool_to_roi
from hilograph.losses import contrastive_loss, hilograph_loss, kl_loss, smoothness_loss
from hilograph.model import HiLoGraph
from hilograph.normative import NormativeModel
from random_graphs import make_random_cohorts

REPO = Path(__file__).resolve().parents[1]
LOSS_CFG = {"lambda_roi": 1.0, "lambda_3hg": 0.1, "lambda_kl": 0.01, "lambda_smooth": 0.5,
            "lambda_contrast": 0.1, "lambda_age": 0.1, "temperature": 0.07, "min_age_gap": 0.5}


@pytest.fixture(scope="module")
def cohorts(tmp_path_factory):
    root = tmp_path_factory.mktemp("graphs")
    healthy_csv, clinical_csv = make_random_cohorts(root, n_healthy=16, n_clinical=36)
    return root, healthy_csv, clinical_csv


@pytest.fixture(scope="module")
def healthy_dataset(cohorts):
    root, healthy_csv, _ = cohorts
    ds = ScanDataset(read_metadata(healthy_csv, root / "healthy"))
    ds.set_feature_stats(compute_feature_stats(ds, range(len(ds))))
    return ds


def test_pool_matches_dense_membership():
    torch.manual_seed(0)
    n_roi, sizes = 6, [5, 7]
    h = torch.randn(sum(sizes), 3)
    roi_id = torch.tensor([0, 0, 2, 5, -1, 1, 1, 1, 3, 3, 0, 2])
    batch = torch.tensor([0] * 5 + [1] * 7)
    pooled = pool_to_roi(h, roi_id, batch, 2, n_roi).view(2, n_roi, 3)
    start = 0
    for b, n in enumerate(sizes):
        m = torch.zeros(n, n_roi)
        for r, rid in enumerate(roi_id[start:start + n]):
            if rid >= 0:
                m[r, rid] = 1
        deg = m.sum(0).clamp_min(1)
        expected = (m.t() @ h[start:start + n]) / deg.unsqueeze(1)   # D^-1 M^T H
        assert torch.allclose(pooled[b], expected, atol=1e-6)
        start += n


def test_sc_edges_are_log_normalised_without_self_loops():
    a = np.array([[5, 10, 0], [10, 7, 100], [0, 100, 3]], dtype=float)
    ei, ew = sc_to_edges(a)
    assert (ei[0] != ei[1]).all()
    assert ew.max().item() == pytest.approx(1.0)
    assert ew.min().item() == pytest.approx(np.log1p(10) / np.log1p(100), rel=1e-6)


def test_subject_batches_keep_visits_together(healthy_dataset):
    sids = healthy_dataset.subject_ids
    for batch in SubjectBatchSampler(sids, batch_size=5, shuffle=True, seed=1):
        subjects = {sids[i] for i in batch}
        for s in subjects:
            assert sum(sids[i] == s for i in batch) == sids.count(s)


def test_folds_are_subject_disjoint(healthy_dataset):
    sids = np.array(healthy_dataset.subject_ids)
    folds = subject_folds(sids, n_folds=4, val_fraction=0.25)
    tests = np.concatenate([f["test"] for f in folds])
    assert sorted(tests) == list(range(len(sids)))
    for f in folds:
        sets = [set(sids[f[k]]) for k in ("train", "val", "test")]
        assert not (sets[0] & sets[1]) and not (sets[0] & sets[2]) and not (sets[1] & sets[2])


def test_longitudinal_losses():
    z = torch.randn(4, 8)
    ids = ["a", "a", "b", "c"]
    ages = torch.tensor([60.0, 62.0, 70.0, 71.0])
    assert smoothness_loss(torch.cat([z[:1], z[:1], z[2:]]), ages, ids) == 0
    expected = ((z[1] - z[0]) / 2.0).pow(2).sum()
    assert torch.isclose(smoothness_loss(z, ages, ids), expected)
    assert smoothness_loss(z, ages, ["a", "b", "c", "d"]) == 0      # no repeated visits
    assert torch.isfinite(contrastive_loss(z, ids))
    aligned = torch.cat([z[:1], z[:1], z[2:]])
    assert contrastive_loss(aligned, ids) < contrastive_loss(torch.cat([z[:1], -z[:1], z[2:]]), ids)
    assert kl_loss(torch.zeros(3, 5), torch.zeros(3, 5)) == 0


@pytest.mark.parametrize("branches", [("3hg", "roi"), ("3hg",), ("roi",)])
def test_forward_backward(healthy_dataset, branches):
    loader = make_loader(healthy_dataset, range(len(healthy_dataset)), batch_size=8)
    batch = next(iter(loader))
    model = HiLoGraph(branches=branches)
    out = model(batch)
    assert out["mu"].shape == (batch.num_graphs, 64)
    assert out["x_roi_hat"].shape == batch.x_roi.shape == (batch.num_graphs * 148, 5)
    assert out["x_3hg_hat"].shape == batch.x_3hg.shape
    loss, parts = hilograph_loss(out, batch, LOSS_CFG)
    loss.backward()
    assert np.isfinite(parts["total"])
    assert all(p.grad is not None for p in model.encoder.parameters())


def test_normative_model_recovers_scale():
    rng = np.random.default_rng(0)
    age = rng.uniform(40, 90, 3000)
    sex = rng.integers(0, 2, 3000).astype(float)
    sigma = 0.5 + 0.02 * (age - 40)
    z = (0.001 * (age - 60) ** 2 + 0.3 * sex + sigma * rng.standard_normal(3000))[:, None]
    norm = NormativeModel().fit(z, age, sex)
    d = norm.deviation(z, age, sex)
    assert abs(d.mean()) < 0.05 and abs(d.std() - 1) < 0.1


def test_end_to_end_cli(cohorts, tmp_path):
    root, healthy_csv, clinical_csv = cohorts
    run = tmp_path / "run"
    py = sys.executable
    subprocess.run([py, "train.py", "--config", "configs/hilograph.yaml", "--folds", "0", "1", "--set",
                    f"data.root={root / 'healthy'}", f"data.metadata={healthy_csv}",
                    "train.epochs=2", "train.n_folds=3", "train.batch_size=8",
                    "train.device=cpu", f"train.out_dir={run}"], cwd=REPO, check=True)
    subprocess.run([py, "extract.py", "--run_dir", str(run), "--root", str(root / "clinical"),
                    "--metadata", str(clinical_csv), "--name", "clinical", "--device", "cpu"],
                   cwd=REPO, check=True)
    subprocess.run([py, "evaluate.py", "--run_dir", str(run), "--name", "clinical",
                    "--n_repeats", "1"], cwd=REPO, check=True)
    assert (run / "clinical_classification.json").exists()
