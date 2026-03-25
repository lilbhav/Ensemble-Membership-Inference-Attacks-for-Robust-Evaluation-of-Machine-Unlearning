# Ensemble-Membership-Inference-Attacks-for-Robust-Evaluation-of-Machine-Unlearning

#  MIA Disparity & Unlearning Evaluation Framework

A lightweight, modular framework for evaluating **machine unlearning algorithms** using **multiple Membership Inference Attacks (MIAs)** and **ensemble-based privacy analysis**.

This project builds on two external repositories:

-  MIA implementations & disparity framework:  
  https://github.com/RPI-DSPlab/mia-disparity

-  Unlearning algorithms (SCRUB, SSD, BadTeacher, Amnesiac):  
  https://github.com/OngWinKent/MachineUnlearning

---

#  Research Goal

Quantify **disparities among MIAs** and evaluate **privacy risks after unlearning**.

We aim to:

- Understand how different MIAs detect **different subsets of vulnerable samples**
- Measure **stability** of attacks across seeds
- Build **ensembles of MIAs** to improve coverage and robustness
- Provide a more **comprehensive privacy evaluation** for unlearning methods

---

#  Key Concepts

### Disparity
- **Coverage**: how many unique training samples are flagged as members  
- **Stability**: how consistent predictions are across seeds/runs  

### Evaluation Targets
- `forget_vs_test` → primary privacy evaluation  
- `retain_vs_test` → utility vs privacy  
- `forget_vs_retain` → diagnostic (not standard MIA)  

### Ensembles
- **OR (Union)** → maximize coverage  
- **k-of-M Voting** → tradeoff between precision and recall  

---

# V1 External-Engine Architecture

This repository now uses a thin adapter architecture:

- The framework owns split generation, orchestration, result collection, disparity, and ensemble evaluation.
- The third-party repositories under `Third_Party_Code/` are treated as external engines.
- No MIA or unlearning algorithm internals are copied into framework modules.

## Folder Layout

- `configs/experiment.yaml` - single source of truth for datasets, seeds, methods, attacks, and paths.
- `adapters/` - subprocess adapters that call bridge scripts and fail loudly on errors.
- `external/` - bridge scripts that import third-party engine modules and execute runs with canonical splits.
- `scripts/` - user-facing pipeline entrypoints.
- `results/` - all generated artifacts.

## Canonical Split Artifacts

For each `(dataset, base_seed)`, split artifacts are written to:

- `results/splits/<dataset>/seed_<seed>.npz` with:
  - `retain_indices`
  - `forget_indices`
  - `test_indices`
  - `aux_indices` (optional but produced in v1)
- `results/splits/<dataset>/seed_<seed>.meta.json`

Both engines consume the same split file.

## Standardized Per-Sample Output Schema

Every MIA run writes CSV rows with:

- `sample_id`
- `true_membership`
- `split_name`
- `model_name`
- `unlearning_method`
- `attack_name`
- `attack_seed`
- `score`
- `prediction`
- `dataset`
- `base_seed`

## Pipeline Scripts

Run from repository root:

1. `python scripts/prepare_splits.py`
2. `python scripts/run_baseline.py`
3. `python scripts/run_unlearning.py`
4. `python scripts/run_mia.py`
5. `python scripts/run_ensemble_eval.py`
6. `python scripts/aggregate_results.py`

Each script supports focused reruns (single dataset/seed/method/attack) via CLI flags.

---

# Colab-First Run (With Google Drive Checkpoints)

Use this flow if you want all outputs/checkpoints to persist in Google Drive.

## 1) Notebook setup

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
%cd /content
!git clone https://github.com/chavab/Ensemble-Membership-Inference-Attacks-for-Robust-Evaluation-of-Machine-Unlearning.git
%cd /content/Ensemble-Membership-Inference-Attacks-for-Robust-Evaluation-of-Machine-Unlearning
!pip install -q -r requirements-colab.txt
```

## 2) Create/update Colab config

```bash
!python scripts/create_colab_config.py \
  --template configs/experiment.yaml \
  --out configs/experiment_colab.yaml \
  --repo-root /content/Ensemble-Membership-Inference-Attacks-for-Robust-Evaluation-of-Machine-Unlearning \
  --drive-root /content/drive/MyDrive/unlearning_runs \
  --dataset Cifar10
```

This writes:

- Data to `/content/drive/MyDrive/unlearning_runs/data`
- Models/checkpoints/results to `/content/drive/MyDrive/unlearning_runs/results`

## 3) Run your exact basic pipeline

Option A: single command wrapper

```bash
!python scripts/run_basic_colab.py \
  --config configs/experiment_colab.yaml \
  --dataset Cifar10 \
  --seed 0 \
  --method scrub \
  --attack yeom \
  --target forget_vs_test \
  --attack-seed 0
```

Option B: run each step manually

```bash
%cd /content/Ensemble-Membership-Inference-Attacks-for-Robust-Evaluation-of-Machine-Unlearning/scripts
!python prepare_splits.py --config ../configs/experiment_colab.yaml --dataset Cifar10 --seed 0
!python run_baseline.py --config ../configs/experiment_colab.yaml --dataset Cifar10 --seed 0
!python run_unlearning.py --config ../configs/experiment_colab.yaml --dataset Cifar10 --seed 0 --method scrub
!python run_mia.py --config ../configs/experiment_colab.yaml --dataset Cifar10 --seed 0 --method scrub --attack yeom --target forget_vs_test --attack-seed 0
!python run_ensemble_eval.py --config ../configs/experiment_colab.yaml --dataset Cifar10 --seed 0
!python aggregate_results.py --config ../configs/experiment_colab.yaml
```

## 4) Output locations

- Baseline checkpoint:
  - `/content/drive/MyDrive/unlearning_runs/results/models/Cifar10/seed_0/baseline.pt`
- Unlearned checkpoint:
  - `/content/drive/MyDrive/unlearning_runs/results/models/Cifar10/seed_0/unlearn_scrub.pt`
- MIA outputs:
  - `/content/drive/MyDrive/unlearning_runs/results/mia/...`
- Ensemble and aggregate outputs:
  - `/content/drive/MyDrive/unlearning_runs/results/ensemble/...`
  - `/content/drive/MyDrive/unlearning_runs/results/aggregate/...`