# HiLoGraph: Hierarchical-Longitudinal Brain Network Representation Learning

PyTorch implementation of **HiLoGraph** (MICCAI 2026).

Tong Chen, Minheng Chen, Jing Zhang, Chao Cao, Li Su, Tianming Liu, Dajiang Zhu

HiLoGraph is a self-supervised framework that encodes multi-scale brain networks, built from
cortical morphology and structural connectivity (SC), into a shared latent space. Longitudinal
constraints keep each subject's latent trajectory consistent across visits. The model is trained
only on healthy subjects, so the latent space acts as a normative reference. A subject's
deviation from that reference then serves as features for disease detection (CN / AD / LBD),
with no disease-specific fine-tuning.

```
 3HG graph  (N x 5, SC) --GCN x3--> H_3HG --D^-1 M^T--> H_pool --+
                                                                 +-- mean over 148 ROIs --> MLP --> mu, log s^2 --> z
 ROI graph (148 x 5, SC) --GCN x2------------------> H_ROI ------+                                                   |
                                                                                                                    |
 X_ROI_hat <-- GCN-FiLM(z) <-- [z ; ROI id-embedding]  <-------------------------------------------------------------+
 X_3HG_hat <-- GCN-FiLM(z) <-- [z ; P = M X_ROI_hat ; 3HG id-embedding]
 age_hat   <-- f_age(z)
 + L_smooth, L_contrast over within-subject visit pairs
```

## Method to code

| Paper | Code |
|---|---|
| Sec. 2.2 Graph construction (3HG graph, ROI graph, membership M) | `hilograph/data.py` |
| Sec. 2.2 Hierarchical graph encoder, Eqs. (1)-(2) | `hilograph/encoder.py` |
| Sec. 2.2 Top-down GCN-FiLM decoders | `hilograph/decoder.py` |
| Full model + age regressor f_age | `hilograph/model.py` |
| Sec. 2.3 Training objective, Eqs. (3)-(5) | `hilograph/losses.py` |
| Sec. 3.1 Implementation details, 5-fold subject-level CV | `train.py`, `configs/hilograph.yaml` |
| Sec. 3.1 Ablations: 3HG-only, ROI-only, w/o long | `configs/ablation_*.yaml` |
| Sec. 3.1 Normative deviation (age- and sex-conditioned robust quadratic regression) | `hilograph/normative.py` |
| Sec. 3.1 Projection of the clinical cohort with the frozen encoder | `extract.py` |
| Tables 1-2 protocol (logistic regression on z, d, [z; d]) | `evaluate.py` |

## Installation

```bash
git clone https://github.com/tongchen2010/hilograph.git
cd hilograph
pip install -r requirements.txt   # PyTorch and PyTorch Geometric: install the builds matching your CUDA
```

Tested with Python 3.9, PyTorch 2.6 and PyTorch Geometric 2.6.

## Data

No imaging data are distributed with this repository. The normative training set is
[AABC Release 2 / HCP-Aging](https://www.humanconnectome.org/study/hcp-lifespan-aging),
available from the NIMH Data Archive under its data-use terms. The clinical cohort is not public.

Each scan is a directory with five arrays (file names are configurable under `data.files`):

```
<root>/<subject_id>/<visit>/
    features/X_3hg.npy                        (N, 5)      3HG landmark features
    features/X_roi_des.npy                    (148, 5)    Destrieux ROI features
    networks/SC_3hg_ring5_count_mat_raw.npy   (N, N)      3HG-level fiber counts
    networks/SC_roi_des_count_mat_raw.npy     (148, 148)  ROI-level fiber counts
    networks/roi_id_3hg_des.npy               (N,)        ROI index of every 3HG node (-1 if none)
```

Node features, in order: cortical thickness, average convexity, mean curvature, surface area and
volume. 3HG landmarks are extracted on the white-matter surface following [Chen et al., 2017] and
[Cao et al., 2026], with features computed from a 4-ring vertex neighbourhood around each
landmark; ROI features are averaged within each Destrieux parcel. For single-timepoint cohorts
the `<visit>` level can be omitted.

A metadata CSV lists the scans:

| column | required | notes |
|---|---|---|
| `subject_id` | yes | all visits of a subject share it; CV folds split by subject |
| `visit` | no | sub-directory name; leave empty for a flat layout |
| `age` | yes | in years. Use fractional ages (e.g. age at V1 + days since V1 / 365.25); integer ages make short visit gaps look like 0 years |
| `sex` | for the normative model | `M` / `F` |
| `diagnosis` | clinical cohort | `CN`, `AD` or `LBD` |

Scans with missing files or missing age are skipped with a warning.

## Usage

**1. Train on healthy longitudinal data** (5-fold subject-level CV). Set `data.root` and
`data.metadata` in `configs/hilograph.yaml`, or override them on the command line:

```bash
python train.py --config configs/hilograph.yaml \
    --set data.root=/data/aabc_graphs data.metadata=/data/aabc_scans.csv
```

Each fold writes `runs/hilograph/fold<k>/model.pt` (best validation checkpoint),
`embeddings.csv` (latent codes of all train/val/test scans) and `history.json`.

Ablations:

```bash
python train.py --config configs/ablation_3hg_only.yaml   # removes the ROI encoder
python train.py --config configs/ablation_roi_only.yaml   # removes the 3HG encoder
python train.py --config configs/ablation_no_long.yaml    # lambda_s = lambda_t = 0
```

**2. Project the clinical cohort** into the latent space of every fold, with no fine-tuning:

```bash
python extract.py --run_dir runs/hilograph \
    --root /data/clinical_graphs --metadata /data/clinical_scans.csv --name clinical
```

**3. Normative deviation and classification:**

```bash
python evaluate.py --run_dir runs/hilograph --name clinical --label_col diagnosis
```

For each fold, `evaluate.py` fits the normative model on that fold's healthy *training* latent
codes. It then computes deviations d for the clinical cohort and runs logistic regression
(balanced class weights, 10 x repeated stratified 5-fold CV) on raw z, d and [z; d]. The tasks
are 3-class CN / AD / LBD (one-vs-rest macro AUC, balanced accuracy), AD vs CN, AD vs LBD and
CN vs LBD. Results are reported as mean ± std over folds. If the diagnosis column holds
finer-grained labels, merge them into the three groups with `--group SUBTYPE=GROUP`.

**Quick check on synthetic data** (random graphs, for testing the code only):

```bash
python scripts/make_toy_data.py --out toy_data
python train.py --config configs/hilograph.yaml --set data.root=toy_data/healthy \
    data.metadata=toy_data/healthy_scans.csv train.epochs=3 train.device=cpu train.out_dir=runs/toy
python extract.py --run_dir runs/toy --root toy_data/clinical --metadata toy_data/clinical_scans.csv --name clinical
python evaluate.py --run_dir runs/toy --name clinical --n_repeats 2
pytest -q
```

## Hyper-parameters (Sec. 3.1)

| | |
|---|---|
| 3HG encoder / ROI encoder | 3 / 2 GCN layers, hidden d = 64 |
| Latent z | 64 |
| Decoders | 3 GCN-FiLM layers each, 32-d node-identity embeddings |
| Optimiser | Adam, lr 1e-3, weight decay 1e-5, 100 epochs, early stopping patience 20 |
| Loss weights | lambda_r = 1, lambda_f = 0.1, lambda_KL = 0.01, lambda_s = 0.5, lambda_t = 0.1, lambda_a = 0.1 |
| Contrastive temperature | tau = 0.07 |

## Implementation notes

The paper does not fix these details; this is how the code handles them:

- **SC edge weights**: `log(1 + count)`, divided by the per-graph maximum. Self-connections are removed.
- **Feature scaling**: each scale (3HG, ROI) is z-scored per feature. The statistics come from
  the training scans of each fold, are stored in the checkpoint, and are reused for the clinical cohort.
- **Mini-batches** keep all visits of a subject together (`SubjectBatchSampler`), so that every
  within-subject pair enters L_smooth and L_contrast.
- **L_smooth** floors the age gap a_2 - a_1 at 0.5 years (`loss.min_age_gap`).
- **f_age** predicts age in years through a fixed de-standardisation with the training-set age
  mean and std, so its initial prediction is the mean age. L_age is an MSE in years², so it is
  one of the largest terms early in training.
- **Sex** must be `M`/`F`; subjects without it are dropped from the normative analysis with a
  warning (or pass `--no_sex` to condition on age only).
- **Downstream codes** use the posterior mean mu of the frozen encoder.
- **Normative sigma_k** comes from a robust quadratic fit of log|residual|, calibrated so the
  healthy training residuals have unit robust standard deviation.

## Citation

```bibtex
@inproceedings{chen2026hilograph,
  title     = {HiLoGraph: Hierarchical-Longitudinal Brain Network Representation Learning},
  author    = {Chen, Tong and Chen, Minheng and Zhang, Jing and Cao, Chao and Su, Li and Liu, Tianming and Zhu, Dajiang},
  booktitle = {Medical Image Computing and Computer Assisted Intervention -- MICCAI 2026},
  year      = {2026}
}
```

## Acknowledgments

Supported by the National Institutes of Health (R01AG075582, R01NS128534) and funded by the Sir
Jules Thorn Charitable Trust (05/JTA), Wellcome Trust Intermediate Clinical Fellowship
(WT088441MA), Alzheimer's Research UK Senior Research Fellowship (ARUK-SRF2017B-1), The Lewy
body Society (LBS/002/2019) and NIHR Biomedical Research Centres in Sheffield, Cambridge and
Newcastle.
