"""compute_mia_metrics.py — Compute calibrated threshold and attack metrics from per-sample MIA score CSVs.

Reads:  results/mia/<dataset>/seed_<seed>/<method>/<attack>/<target>_attack_seed_<n>.csv
Writes: results/aggregate/mia_score_metrics.csv

Output columns:
    dataset, base_seed, unlearning_method, attack_name, attack_seed, target,
    split_mode, target_class, n_samples, calibrated_threshold, auc,
    balanced_accuracy, tpr_at_1pct_fpr, coverage_fraction, score_direction
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from adapters.io_utils import ensure_dir, load_config, resolve_path

FIELDS = [
    "dataset",
    "base_seed",
    "unlearning_method",
    "attack_name",
    "attack_seed",
    "target",
    "split_mode",
    "target_class",
    "n_samples",
    "calibrated_threshold",
    "auc",
    "balanced_accuracy",
    "tpr_at_1pct_fpr",
    "coverage_fraction",
    "score_direction",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute per-(model, attack) AUC and TPR@FPR thresholds from MIA score CSVs."
    )
    p.add_argument("--config", default="configs/experiment.yaml")
    p.add_argument("--dataset", help="Restrict to one dataset name.")
    p.add_argument("--seed", type=int, help="Restrict to one base seed.")
    p.add_argument(
        "--target",
        default=None,
        help="Evaluation target, e.g. forget_vs_test. Defaults to all targets found on disk.",
    )
    return p.parse_args()


def read_score_csv(path: Path) -> tuple[list[float], list[int], list[int], dict]:
    """Return (scores, labels, predictions, metadata) from a per-sample attack CSV."""
    scores: list[float] = []
    labels: list[int] = []
    predictions: list[int] = []
    meta: dict = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scores.append(float(row["score"]))
            labels.append(int(row["true_membership"]))
            predictions.append(int(row["prediction"]))
            if not meta:
                meta = {
                    k: row.get(k, "")
                    for k in (
                        "split_mode",
                        "target_class",
                        "dataset",
                        "base_seed",
                        "calibrated_threshold",
                        "calibration_target_fpr",
                        "score_direction",
                    )
                }
    return scores, labels, predictions, meta


def tpr_at_fpr_threshold(fpr_arr, tpr_arr, threshold: float) -> float:
    """Max TPR achieved where FPR <= threshold."""
    eligible = [tpr for fpr, tpr in zip(fpr_arr, tpr_arr) if fpr <= threshold]
    return max(eligible) if eligible else 0.0


def compute_metrics(scores: list[float], labels: list[int], predictions: list[int]):
    """Return (auc, balanced_accuracy, tpr_at_1pct, coverage_fraction) using sklearn."""
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve  # type: ignore

    auc = float(roc_auc_score(labels, scores))
    fpr_arr, tpr_arr, _ = roc_curve(labels, scores)
    tpr_1pct = tpr_at_fpr_threshold(fpr_arr, tpr_arr, 0.01)
    balanced_accuracy = float(balanced_accuracy_score(labels, predictions))
    coverage_fraction = float(sum(predictions) / max(1, len(predictions)))
    return auc, balanced_accuracy, tpr_1pct, coverage_fraction


def process_seed(mia_root: Path, dataset: str, seed: int, target_filter: str | None) -> list[dict]:
    seed_dir = mia_root / dataset / f"seed_{seed}"
    if not seed_dir.exists():
        return []

    rows: list[dict] = []
    for method_dir in sorted(seed_dir.iterdir()):
        if not method_dir.is_dir():
            continue
        method = method_dir.name
        for attack_dir in sorted(method_dir.iterdir()):
            if not attack_dir.is_dir():
                continue
            attack_name = attack_dir.name
            # Each CSV is named <target>_attack_seed_<n>.csv
            for score_file in sorted(attack_dir.glob("*_attack_seed_*.csv")):
                stem = score_file.stem  # e.g. "forget_vs_test_attack_seed_0"
                # Extract target: everything before "_attack_seed_"
                if "_attack_seed_" not in stem:
                    continue
                target_name, attack_seed_str = stem.rsplit("_attack_seed_", 1)
                if target_filter and target_name != target_filter:
                    continue

                scores, labels, predictions, meta = read_score_csv(score_file)
                n = len(scores)
                if n == 0:
                    continue
                if len(set(labels)) < 2:
                    print(f"  Skipping {score_file.name}: only one class present.")
                    continue

                try:
                    auc, balanced_accuracy, tpr_1pct, coverage_fraction = compute_metrics(scores, labels, predictions)
                except Exception as exc:
                    print(f"  Warning: metrics failed for {score_file}: {exc}")
                    continue

                rows.append({
                    "dataset": dataset,
                    "base_seed": seed,
                    "unlearning_method": method,
                    "attack_name": attack_name,
                    "attack_seed": attack_seed_str,
                    "target": target_name,
                    "split_mode": meta.get("split_mode", ""),
                    "target_class": meta.get("target_class", ""),
                    "n_samples": n,
                    "calibrated_threshold": meta.get("calibrated_threshold", ""),
                    "auc": round(auc, 6),
                    "balanced_accuracy": round(balanced_accuracy, 6),
                    "tpr_at_1pct_fpr": round(tpr_1pct, 6),
                    "coverage_fraction": round(coverage_fraction, 6),
                    "score_direction": meta.get("score_direction", ""),
                })

    return rows


def main() -> None:
    args = parse_args()

    try:
        import sklearn  # noqa: F401
    except ImportError:
        print("ERROR: scikit-learn is required. Install with: pip install scikit-learn")
        sys.exit(1)

    cfg = load_config(args.config)
    results_root = resolve_path(cfg["paths"]["results_root"])
    mia_root = results_root / "mia"

    datasets = [args.dataset] if args.dataset else cfg["experiment"]["datasets"]
    seeds = [args.seed] if args.seed is not None else cfg["experiment"]["base_seeds"]

    all_rows: list[dict] = []
    for dataset in datasets:
        for seed in seeds:
            print(f"Processing {dataset}/seed_{seed} …")
            rows = process_seed(mia_root, dataset, seed, args.target)
            if not rows:
                print(f"  No score files found (target filter: {args.target!r}).")
            all_rows.extend(rows)

    if not all_rows:
        print("No MIA score files found. Run scripts/run_mia.py first.")
        sys.exit(1)

    out_dir = ensure_dir(results_root / "aggregate")
    out_file = out_dir / "mia_score_metrics.csv"
    with out_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Saved {len(all_rows)} rows → {out_file}")


if __name__ == "__main__":
    main()
