"""Train HiLoGraph on healthy longitudinal scans with subject-level K-fold CV.

    python train.py --config configs/hilograph.yaml
    python train.py --config configs/hilograph.yaml --folds 0 1 --set train.epochs=50

Per fold, ``<out_dir>/fold<k>/`` receives:
    model.pt         best checkpoint (by validation loss) + config + feature statistics
    embeddings.csv   latent codes of every train / val / test scan
    history.json     per-epoch loss components
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from hilograph.config import apply_overrides, load_config, save_config
from hilograph.data import (ScanDataset, compute_feature_stats, make_loader, read_metadata,
                            subject_folds)
from hilograph.embedding import embed_scans, embeddings_table, stats_to_lists
from hilograph.losses import hilograph_loss
from hilograph.model import HiLoGraph


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, loss_cfg, device, optimizer=None, grad_clip=None):
    training = optimizer is not None
    model.train(training)
    sums, n = defaultdict(float), 0
    with torch.set_grad_enabled(training):
        for batch in loader:
            batch = batch.to(device)
            loss, parts = hilograph_loss(model(batch), batch, loss_cfg)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            for k, v in parts.items():
                sums[k] += v * batch.num_graphs
            n += batch.num_graphs
    return {k: v / n for k, v in sums.items()}


def train_fold(k, split, dataset, cfg, device, out_dir: Path):
    tcfg = cfg["train"]
    set_seed(tcfg["seed"] + k)
    stats = compute_feature_stats(dataset, split["train"])
    dataset.set_feature_stats(stats)
    bs, workers = tcfg["batch_size"], tcfg["num_workers"]
    train_loader = make_loader(dataset, split["train"], bs, shuffle=True,
                               seed=tcfg["seed"] + k, num_workers=workers)
    val_loader = make_loader(dataset, split["val"], bs, shuffle=False, num_workers=workers)

    model = HiLoGraph.from_config(cfg["model"], n_roi=cfg["data"]["n_roi"])
    train_ages = dataset.table["age"].to_numpy(float)[split["train"]]
    model.set_age_stats(train_ages.mean(), train_ages.std())
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=tcfg["lr"],
                                 weight_decay=tcfg["weight_decay"])

    best_loss, best_state, best_epoch, wait, history = float("inf"), None, 0, 0, []
    for epoch in range(1, tcfg["epochs"] + 1):
        tr = run_epoch(model, train_loader, cfg["loss"], device, optimizer, tcfg.get("grad_clip"))
        va = run_epoch(model, val_loader, cfg["loss"], device)
        history.append({"epoch": epoch, "train": tr, "val": va})
        print(f"[fold {k}] epoch {epoch:3d}  train {tr['total']:.4f}  val {va['total']:.4f}  "
              + " ".join(f"{n}={v:.3f}" for n, v in tr.items() if n != "total"), flush=True)
        if va["total"] < best_loss:
            best_loss, best_epoch, wait = va["total"], epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= tcfg["patience"]:
                print(f"[fold {k}] early stopping at epoch {epoch} (best {best_epoch})")
                break

    if best_state is None:
        raise RuntimeError(f"fold {k}: validation loss was never finite")
    model.load_state_dict(best_state)
    fold_dir = out_dir / f"fold{k}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": best_state, "config": cfg, "feature_stats": stats_to_lists(stats),
                "fold": k, "best_epoch": best_epoch, "best_val_loss": best_loss},
               fold_dir / "model.pt")

    order = np.concatenate([split["train"], split["val"], split["test"]])
    names = (["train"] * len(split["train"]) + ["val"] * len(split["val"])
             + ["test"] * len(split["test"]))
    mu = embed_scans(model, dataset, order, bs, device)
    embeddings_table(dataset, order, mu, names).to_csv(fold_dir / "embeddings.csv", index=False)
    with open(fold_dir / "history.json", "w") as f:
        json.dump(history, f, indent=1)
    print(f"[fold {k}] best epoch {best_epoch}, val loss {best_loss:.4f} -> {fold_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--folds", type=int, nargs="*", help="folds to run (default: all)")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="override config entries, e.g. train.epochs=10")
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args.set)
    tcfg = cfg["train"]
    device = tcfg["device"] if torch.cuda.is_available() or tcfg["device"] == "cpu" else "cpu"
    out_dir = Path(tcfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out_dir / "config.yaml")

    dcfg = cfg["data"]
    table = read_metadata(dcfg["metadata"], dcfg["root"], dcfg["files"])
    dataset = ScanDataset(table, dcfg["files"], dcfg["n_roi"], dcfg.get("cache", True))
    folds = subject_folds(dataset.subject_ids, tcfg["n_folds"], tcfg["val_fraction"], tcfg["seed"])
    n_subjects = table["subject_id"].nunique()
    print(f"{len(table)} scans from {n_subjects} subjects; {tcfg['n_folds']}-fold CV on {device}")

    for k in (args.folds if args.folds else range(len(folds))):
        train_fold(k, folds[k], dataset, cfg, device, out_dir)


if __name__ == "__main__":
    main()
