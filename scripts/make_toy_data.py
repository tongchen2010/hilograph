"""Write small synthetic cohorts for a quick end-to-end check of the pipeline.

    python scripts/make_toy_data.py --out toy_data
    python train.py --config configs/hilograph.yaml \
        --set data.root=toy_data/healthy data.metadata=toy_data/healthy_scans.csv \
              train.epochs=3 train.device=cpu train.out_dir=runs/toy
    python extract.py --run_dir runs/toy --root toy_data/clinical \
        --metadata toy_data/clinical_scans.csv --name clinical
    python evaluate.py --run_dir runs/toy --name clinical --n_repeats 2
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hilograph.toy import make_toy_cohorts  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="toy_data")
    parser.add_argument("--n_healthy", type=int, default=40)
    parser.add_argument("--n_clinical", type=int, default=45)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    healthy, clinical = make_toy_cohorts(args.out, args.n_healthy, args.n_clinical, args.seed)
    print(f"wrote {healthy} and {clinical}")
