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

The repository now supports a single split protocol only:

- `split_mode = targeted_random`
- choose one `target_class`
- sample forget examples only from that class
- keep all remaining train examples in retain
- keep official test set unchanged

In metadata and logs, `forget_fraction` is the fraction within the selected target class, not the full training set.

Required metadata keys include:

- `split_mode` (must be `targeted_random`)
- `dataset`
- `seed`
- `target_class`
- `forget_count`
- `forget_fraction`

Both engines consume the same split file.

## Standardized Per-Sample Output Schema

Every MIA run writes CSV rows with:

- `sample_id`
- `true_membership`
- `split_name`
- `split_mode` (always `targeted_random`)
- `target_class`
- `forget_count`
- `forget_fraction` (within `target_class`)
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

Utility-only validation for one unlearning method at a time (before MIA runs):

- `python scripts/run_utility_eval.py --dataset Cifar10 --seed 0 --method scrub`
- `python scripts/run_utility_eval.py --dataset Cifar10 --seed 0 --method ssd`
- `python scripts/run_utility_eval.py --dataset Cifar10 --seed 0 --method bad_teacher`
- `python scripts/run_utility_eval.py --dataset Cifar10 --seed 0 --method amnesiac`

This workflow uses the canonical `targeted_random` split, trains or reuses baseline, runs one unlearning method, and writes utility metrics plus drops.
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

## 2) Run your exact basic pipeline

`configs/experiment.yaml` is already set to save outputs in Google Drive:

- Data: `/content/drive/MyDrive/unlearning_runs/data`
- Models/checkpoints/results: `/content/drive/MyDrive/unlearning_runs/results`

Run these commands:

```bash
%cd /content/Ensemble-Membership-Inference-Attacks-for-Robust-Evaluation-of-Machine-Unlearning/scripts
!python prepare_splits.py --dataset Cifar10 --seed 0
!python run_baseline.py --dataset Cifar10 --seed 0
!python run_unlearning.py --dataset Cifar10 --seed 0 --method scrub
!python run_mia.py --dataset Cifar10 --seed 0 --method scrub --attack yeom --target forget_vs_test --attack-seed 0
!python run_ensemble_eval.py --dataset Cifar10 --seed 0
!python aggregate_results.py
```

## 3) Output locations

- Baseline checkpoint:
  - `/content/drive/MyDrive/unlearning_runs/results/models/Cifar10/seed_0/baseline.pt`
- Unlearned checkpoint:
  - `/content/drive/MyDrive/unlearning_runs/results/models/Cifar10/seed_0/unlearn_scrub.pt`
- MIA outputs:
  - `/content/drive/MyDrive/unlearning_runs/results/mia/...`
- Ensemble and aggregate outputs:
  - `/content/drive/MyDrive/unlearning_runs/results/ensemble/...`
  - `/content/drive/MyDrive/unlearning_runs/results/aggregate/...`

## Troubleshooting (Colab)

- Error: `ModuleNotFoundError: No module named 'src'` during `prepare_splits.py`
  - Cause: `Third_Party_Code/MachineUnlearning` is missing or not in expected layout.
  - Fix:
    1. Ensure you cloned this repository fresh in Colab runtime.
    2. Verify path exists: `/content/Ensemble-Membership-Inference-Attacks-for-Robust-Evaluation-of-Machine-Unlearning/Third_Party_Code/MachineUnlearning/src/__init__.py`
    3. Re-run `prepare_splits.py` before baseline/unlearning/MIA.

- Error: missing split file in later stages
  - Cause: `prepare_splits.py` failed earlier, so downstream scripts cannot find `seed_0.npz`.
  - Fix: resolve split-generation error first, then rerun pipeline from `prepare_splits.py` onward.