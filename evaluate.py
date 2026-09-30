"""Downstream disease classification from normative deviations (Sec. 3.1).

For every cross-validation fold the frozen encoder's healthy *training* latent
codes define the normative model (age- and sex-conditioned robust quadratic
regression). The external cohort is scored with deviations
d_k = (z_k - mu_k(a, s)) / sigma_k(a, s), which feed a logistic-regression
classifier (balanced class weights, repeated stratified 5-fold CV).
Results are reported as mean +- std over the encoder folds.

    python evaluate.py --run_dir runs/hilograph --name clinical --label_col diagnosis

Features: ``z`` (raw latent codes), ``d`` (deviations), ``zd`` ([z; d]).
The label column must contain CN / AD / LBD; finer-grained labels can be
merged with ``--group SUBTYPE=GROUP`` (repeatable).
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from hilograph.normative import NormativeModel

N_SPLITS = 5

CLASSES = ("CN", "AD", "LBD")
TASKS = {                      # task -> (negative class, ..., positive class)
    "3-class": ("CN", "AD", "LBD"),
    "AD vs CN": ("CN", "AD"),
    "AD vs LBD": ("LBD", "AD"),
    "CN vs LBD": ("CN", "LBD"),
}


def cv_classify(x: np.ndarray, y: np.ndarray, n_repeats: int, seed: int):
    """Repeated stratified 5-fold CV; returns mean AUC (one-vs-rest macro for
    >2 classes) and mean balanced accuracy over repeats."""
    n_classes = len(np.unique(y))
    multiclass = n_classes > 2
    aucs, baccs = [], []
    for r in range(n_repeats):
        cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed + r)
        fold_auc, fold_bacc = [], []
        for tr, te in cv.split(x, y):
            clf = make_pipeline(StandardScaler(),
                                LogisticRegression(class_weight="balanced", max_iter=5000))
            clf.fit(x[tr], y[tr])
            prob = clf.predict_proba(x[te])
            fold_bacc.append(balanced_accuracy_score(y[te], clf.predict(x[te])))
            if len(np.unique(y[te])) < n_classes:
                continue  # AUC undefined when a class is absent from the test fold
            if multiclass:
                fold_auc.append(roc_auc_score(y[te], prob, multi_class="ovr", average="macro"))
            else:
                fold_auc.append(roc_auc_score(y[te], prob[:, 1]))
        aucs.append(np.mean(fold_auc))
        baccs.append(np.mean(fold_bacc))
    return float(np.mean(aucs)), float(np.mean(baccs))


def evaluate_fold(fold_dir: Path, name: str, label_col: str, groups: dict, use_sex: bool,
                  n_repeats: int, seed: int):
    healthy = pd.read_csv(fold_dir / "embeddings.csv")
    healthy = healthy[healthy["split"] == "train"]
    ext = pd.read_csv(fold_dir / f"{name}_embeddings.csv")
    labels = ext[label_col].astype(str).str.strip()
    ext["group"] = labels.map(lambda v: groups.get(v, v))
    unmapped = labels[~ext["group"].isin(CLASSES)].value_counts()
    if len(unmapped):
        warnings.warn(f"{fold_dir.name}: ignoring labels not in {CLASSES}: {unmapped.to_dict()} "
                      "(use --group SUBTYPE=GROUP to merge them)")
    ext = ext[ext["group"].isin(CLASSES)].reset_index(drop=True)
    if use_sex:
        for name, df in (("healthy", healthy), (name, ext)):
            if df["sex"].isna().any():
                warnings.warn(f"{fold_dir.name}: dropping {int(df['sex'].isna().sum())} {name} "
                              "subjects without sex (or pass --no_sex)")
        healthy = healthy[healthy["sex"].notna()]
        ext = ext[ext["sex"].notna()].reset_index(drop=True)
    zcols = [c for c in healthy.columns if c.startswith("z") and c[1:].isdigit()]

    sex_h = healthy["sex"].to_numpy(float) if use_sex else None
    sex_e = ext["sex"].to_numpy(float) if use_sex else None
    norm = NormativeModel(use_sex=use_sex).fit(
        healthy[zcols].to_numpy(), healthy["age"].to_numpy(float), sex_h)
    z = ext[zcols].to_numpy()
    d = norm.deviation(z, ext["age"].to_numpy(float), sex_e)
    features = {"z": z, "d": d, "zd": np.hstack([z, d])}

    results = {}
    for feat, x in features.items():
        for task, classes in TASKS.items():
            keep = ext["group"].isin(classes).to_numpy()
            y = ext.loc[keep, "group"].map({c: i for i, c in enumerate(classes)}).to_numpy()
            if len(np.unique(y)) < len(classes) or np.bincount(y).min() < N_SPLITS:
                results[(feat, task)] = (np.nan, np.nan)   # too few subjects in a class
                continue
            results[(feat, task)] = cv_classify(x[keep], y, n_repeats, seed)
    counts = ext["group"].value_counts().to_dict()
    return results, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--name", default="external", help="prefix used in extract.py")
    parser.add_argument("--label_col", default="diagnosis")
    parser.add_argument("--n_repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--group", nargs="*", default=[], metavar="SUBTYPE=GROUP",
                        help="merge a finer-grained label into CN, AD or LBD")
    parser.add_argument("--no_sex", action="store_true",
                        help="condition the normative model on age only")
    args = parser.parse_args()
    groups = dict(item.split("=", 1) for item in args.group)
    if set(groups.values()) - set(CLASSES):
        raise SystemExit(f"--group targets must be one of {CLASSES}")

    fold_dirs = sorted(p.parent for p in Path(args.run_dir).glob(f"fold*/{args.name}_embeddings.csv"))
    if not fold_dirs:
        raise SystemExit(f"No fold*/{args.name}_embeddings.csv under {args.run_dir}; run extract.py first")

    per_fold = []
    for fold_dir in fold_dirs:
        res, counts = evaluate_fold(fold_dir, args.name, args.label_col, groups, not args.no_sex,
                                    args.n_repeats, args.seed)
        per_fold.append(res)
        print(f"{fold_dir.name}: " + ", ".join(f"{g}={n}" for g, n in sorted(counts.items())))

    summary = {}
    header = f"{'feature':<8}" + "".join(f"{t + ' AUC':>18}" for t in TASKS) + f"{'3-class BAcc':>18}"
    print("\n" + header + "\n" + "-" * len(header))
    for feat in ("z", "d", "zd"):
        row = f"{feat:<8}"
        summary[feat] = {}
        for task in TASKS:
            aucs = np.array([r[(feat, task)][0] for r in per_fold])
            baccs = np.array([r[(feat, task)][1] for r in per_fold])
            if np.isnan(aucs).all():
                summary[feat][task] = None
                row += f"{'n/a':>18}"
                continue
            summary[feat][task] = {"auc_mean": np.nanmean(aucs), "auc_std": np.nanstd(aucs),
                                   "bacc_mean": np.nanmean(baccs), "bacc_std": np.nanstd(baccs)}
            row += f"{np.nanmean(aucs):>11.3f} ± {np.nanstd(aucs):.3f}"
        b = summary[feat]["3-class"]
        row += f"{b['bacc_mean']:>11.3f} ± {b['bacc_std']:.3f}" if b else f"{'n/a':>18}"
        print(row)

    out = Path(args.run_dir) / f"{args.name}_classification.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
