# Ensemble Membership Inference for Robust Machine Unlearning Evaluation

This repository evaluates machine unlearning quality with multiple membership inference attacks (MIAs) and ensemble voting. The core objective is to measure whether forgotten samples still leak membership signal after unlearning.

## Project Goal

Traditional evaluation with one MIA can be fragile. This project runs a suite of MIAs, compares their disagreement (disparity), and aggregates them with ensemble rules (OR and k-of-M) to produce a stronger privacy assessment.

## Active Experiment Pipeline

Run from repository root.

1. Create canonical splits

```bash
python scripts/prepare_splits.py --config configs/experiment.yaml
```

2. Train baseline model

```bash
python scripts/run_baseline.py --config configs/experiment.yaml
```

3. Run unlearning methods

```bash
python scripts/run_unlearning.py --config configs/experiment.yaml
```

4. Run MIAs

```bash
python scripts/run_mia.py --config configs/experiment.yaml
```

5. Compute ensemble metrics

```bash
python scripts/run_ensemble_eval.py --config configs/experiment.yaml
```

6. Aggregate outputs

```bash
python scripts/aggregate_results.py --config configs/experiment.yaml
```

## Input and Output Flow

- Config source of truth:
  - configs/experiment.yaml
- Split artifacts:
  - results/splits/<dataset>/seed_<seed>.npz
  - results/splits/<dataset>/seed_<seed>.meta.json
- Baseline and unlearned checkpoints:
  - results/models/<dataset>/seed_<seed>/baseline.pt
  - results/models/<dataset>/seed_<seed>/unlearn_<method>.pt
- Per-attack predictions:
  - results/mia/<dataset>/seed_<seed>/<method>/<attack>/<target>_attack_seed_<n>.csv
- Ensemble outputs:
  - results/ensemble/<dataset>/seed_<seed>/<method>/<target>/ensemble_metrics.csv
  - results/ensemble/<dataset>/seed_<seed>/<method>/<target>/coverage_per_attack.csv
  - results/ensemble/<dataset>/seed_<seed>/<method>/<target>/disparity_pairwise_jaccard.csv
- Aggregates:
  - results/aggregate/model_metrics.csv
  - results/aggregate/mia_predictions_all.csv
  - results/aggregate/ensemble_metrics_all.csv

## Folder Structure

- configs/: experiment configuration files
- scripts/: user-facing pipeline scripts
- adapters/: subprocess adapters used by orchestration scripts
- external/: bridge scripts that call third-party engines
- Third_Party_Code/: external implementations (MachineUnlearning and mia-disparity)
- results/: generated artifacts

## Dependencies

This repo currently does not include a pinned dependency file. Use a Python environment with the following packages installed:

- numpy
- pandas
- pyyaml
- torch
- torchvision
- scikit-learn
- matplotlib
- seaborn
- tqdm

Example install command:

```bash
pip install numpy pandas pyyaml torch torchvision scikit-learn matplotlib seaborn tqdm
```

## Reproducibility Notes

- Split mode is targeted_random only.
- Split metadata is validated before baseline/unlearning/MIA stages.
- Both third-party engines consume the same split artifacts.
- Every MIA row includes split metadata fields used by downstream aggregation.
