# HiLoGraph: Hierarchical-Longitudinal Brain Network Representation Learning

Official PyTorch implementation of **HiLoGraph** (MICCAI 2026).

Tong Chen, Minheng Chen, Jing Zhang, Chao Cao, Li Su, Tianming Liu, Dajiang Zhu

HiLoGraph is a self-supervised framework that encodes multi-scale brain networks (3-hinge gyri
and atlas ROIs, with morphological features and structural connectivity) into a shared latent
space, with longitudinal constraints on within-subject visits. It is trained only on healthy
subjects, and deviations from this normative latent space are used for downstream disease
classification (CN / AD / LBD).

## Repository structure

```
hilograph/
  data.py        graph construction, data loading, subject-level folds
  encoder.py     hierarchical 3HG / ROI graph encoder
  decoder.py     top-down GCN-FiLM decoders
  model.py       HiLoGraph model
  losses.py      reconstruction, KL, smoothness, contrastive and age losses
  normative.py   normative model and deviation scores
configs/         full model and ablations (3HG-only, ROI-only, w/o long)
train.py         training with 5-fold subject-level cross-validation
extract.py       embed an external cohort with the frozen encoders
evaluate.py      normative deviation + logistic-regression classification
tests/           unit tests
```

## Installation

```bash
git clone https://github.com/tongchen2010/hilograph.git
cd hilograph
pip install -r requirements.txt   # install PyTorch / PyTorch Geometric builds matching your CUDA
```

Tested with Python 3.9, PyTorch 2.6 and PyTorch Geometric 2.6.

## Data

No imaging data are distributed with this repository. The healthy training set is
[AABC Release 2 / HCP-Aging](https://www.humanconnectome.org/study/hcp-lifespan-aging),
available from the NIMH Data Archive under its data-use terms.

Each scan is a directory with five arrays (file names can be changed under `data.files`):

```
<root>/<subject_id>/<visit>/
    features/X_3hg.npy                        (N, 5)      3HG node features
    features/X_roi_des.npy                    (148, 5)    Destrieux ROI features
    networks/SC_3hg_ring5_count_mat_raw.npy   (N, N)      3HG structural connectivity (fiber counts)
    networks/SC_roi_des_count_mat_raw.npy     (148, 148)  ROI structural connectivity (fiber counts)
    networks/roi_id_3hg_des.npy               (N,)        0-based ROI index of each 3HG node (-1 if none)
```

Node features: cortical thickness, average convexity, mean curvature, surface area, volume.
For single-timepoint cohorts the `<visit>` level can be omitted.

Scans are listed in a CSV file:

| column | description |
|---|---|
| `subject_id` | subject identifier (all visits of a subject share it) |
| `visit` | visit sub-directory (optional) |
| `age` | age in years at the scan; fractional ages are recommended |
| `sex` | `M` / `F` |
| `diagnosis` | `CN`, `AD` or `LBD` (clinical cohort only) |

## Usage

**Training** on healthy longitudinal data. `configs/hilograph.yaml` contains the settings used in
the paper; set `data.root` and `data.metadata` there or on the command line:

```bash
python train.py --config configs/hilograph.yaml \
    --set data.root=/data/aabc_graphs data.metadata=/data/aabc_scans.csv
```

Ablations: `configs/ablation_3hg_only.yaml`, `configs/ablation_roi_only.yaml`,
`configs/ablation_no_long.yaml`.

**Embedding the clinical cohort** with the frozen encoder of each fold:

```bash
python extract.py --run_dir runs/hilograph \
    --root /data/clinical_graphs --metadata /data/clinical_scans.csv --name clinical
```

**Normative deviation and classification** (CN vs AD vs LBD and the three binary tasks):

```bash
python evaluate.py --run_dir runs/hilograph --name clinical
```

## Citation

```bibtex
@inproceedings{chen2026hilograph,
  title     = {HiLoGraph: Hierarchical-Longitudinal Brain Network Representation Learning},
  author    = {Chen, Tong and Chen, Minheng and Zhang, Jing and Cao, Chao and Su, Li and Liu, Tianming and Zhu, Dajiang},
  booktitle = {Medical Image Computing and Computer Assisted Intervention -- MICCAI 2026},
  year      = {2026}
}
```
