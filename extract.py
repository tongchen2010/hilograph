"""Project an external cohort into the learned latent space with the frozen
encoder of every cross-validation fold (no fine-tuning).

    python extract.py --run_dir runs/hilograph \
        --root /path/to/clinical_graphs --metadata /path/to/clinical_scans.csv --name clinical

The metadata needs ``subject_id``, ``age`` and ``sex``; extra columns such as
``diagnosis`` are copied to ``<run_dir>/fold<k>/<name>_embeddings.csv``.
Feature normalisation uses the statistics stored with each checkpoint.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from hilograph.data import ScanDataset, read_metadata
from hilograph.embedding import embed_scans, embeddings_table, load_checkpoint, stats_from_lists


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--root", required=True, help="graph directory of the external cohort")
    parser.add_argument("--metadata", required=True, help="scan table of the external cohort")
    parser.add_argument("--name", default="external", help="output prefix")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    fold_dirs = sorted(Path(args.run_dir).glob("fold*/model.pt"))
    if not fold_dirs:
        raise SystemExit(f"No fold*/model.pt found under {args.run_dir}")

    dataset = None
    for ckpt_path in fold_dirs:
        model, ckpt = load_checkpoint(ckpt_path, device)
        if dataset is None:
            dcfg = ckpt["config"]["data"]
            table = read_metadata(args.metadata, args.root, dcfg["files"])
            dataset = ScanDataset(table, dcfg["files"], dcfg["n_roi"], cache=True)
        dataset.set_feature_stats(stats_from_lists(ckpt["feature_stats"]))
        indices = np.arange(len(dataset))
        mu = embed_scans(model, dataset, indices, args.batch_size, device)
        out = ckpt_path.parent / f"{args.name}_embeddings.csv"
        embeddings_table(dataset, indices, mu).to_csv(out, index=False)
        print(f"fold {ckpt['fold']}: {len(indices)} scans -> {out}")


if __name__ == "__main__":
    main()
