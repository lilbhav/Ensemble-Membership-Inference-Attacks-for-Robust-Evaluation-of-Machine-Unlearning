from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# Add project root to path so imports work regardless of cwd
sys.path.insert(0, str(Path(__file__).parent.parent))

from adapters.io_utils import ensure_dir, load_config, resolve_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute disparity metrics and ensemble voting from standardized MIA rows")
    p.add_argument("--config", default="configs/experiment.yaml")
    p.add_argument("--dataset")
    p.add_argument("--seed", type=int)
    p.add_argument("--method")
    p.add_argument("--target")
    return p.parse_args()


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def jaccard(a: set[int], b: set[int]) -> float:
    # Overlap ratio used to measure disagreement/consistency between attacks
    u = len(a.union(b))
    if u == 0:
        return 0.0
    return len(a.intersection(b)) / u


def metrics_at_target_fpr(labels: np.ndarray, scores: np.ndarray, target_fpr: float) -> tuple[float, float, float, float]:
    """Return (tpr, fpr, acc, threshold) at exact target FPR from ROC curve.

    TPR is linearly interpolated on the ROC curve at target_fpr so every rule
    reports at the same operating point. Accuracy is computed at the closest
    discrete threshold and is provided as a supportive metric.
    """
    from sklearn.metrics import roc_curve  # type: ignore

    if len(np.unique(labels)) < 2:
        return 0.0, float(target_fpr), 0.0, float(np.max(scores) + 1e-12)

    fpr_arr, tpr_arr, thresholds = roc_curve(labels, scores)

    # sklearn ROC is monotonic in FPR; interpolate to the exact target point.
    tpr = float(np.interp(float(target_fpr), fpr_arr, tpr_arr))

    # Choose a nearby discrete threshold for per-sample predictions/accuracy.
    best_idx = int(np.argmin(np.abs(fpr_arr - float(target_fpr))))
    threshold = float(thresholds[best_idx])

    preds = (scores >= threshold).astype(int)
    acc = float((preds == labels).mean()) if len(labels) > 0 else 0.0
    return float(tpr), float(target_fpr), acc, float(threshold)


def main() -> None:
    # 1) Load configuration and optional CLI filters
    args = parse_args()
    cfg = load_config(args.config)
    results_root = resolve_path(cfg["paths"]["results_root"])

    datasets = [args.dataset] if args.dataset else cfg["experiment"]["datasets"]
    seeds = [args.seed] if args.seed is not None else cfg["experiment"]["base_seeds"]
    methods = [args.method] if args.method else ["baseline", *cfg["unlearning"]["methods"]]
    targets = [args.target] if args.target else cfg["mia"]["targets"]
    target_fpr = float(cfg.get("mia", {}).get("target_fpr", 0.05))

    k_values = [int(k) for k in cfg["ensemble"]["voting_rules"].get("k_of_m", [])]

    for dataset in datasets:
        for seed in seeds:
            for method in methods:
                method_dir = results_root / "mia" / dataset / f"seed_{seed}" / method
                if not method_dir.exists():
                    continue

                for target in targets:
                    # Gather all attack outputs produced for this method/target
                    attack_files = sorted(method_dir.glob(f"*/{target}_attack_seed_*.csv"))
                    if not attack_files:
                        continue

                    attack_to_sample_prediction = {}
                    attack_to_positive = {}
                    true_membership = {}

                    for file in attack_files:
                        attack_name = file.parent.name
                        rows = read_rows(file)
                        sample_prediction = {}
                        for r in rows:
                            sid = int(r["sample_id"])
                            pred = int(r["prediction"])
                            sample_prediction[sid] = pred
                            true_membership[sid] = int(r["true_membership"])

                        positives = {sid for sid, pred in sample_prediction.items() if pred == 1}
                        attack_to_sample_prediction[attack_name] = sample_prediction
                        attack_to_positive[attack_name] = positives

                    attacks = sorted(attack_to_sample_prediction.keys())
                    all_sample_ids = sorted(true_membership.keys())
                    m = len(attacks)

                    out_dir = ensure_dir(results_root / "ensemble" / dataset / f"seed_{seed}" / method / target)

                    # 2) Compute pairwise disparity between attack positive sets
                    disparity_rows = []
                    for i, a1 in enumerate(attacks):
                        for j in range(i + 1, len(attacks)):
                            a2 = attacks[j]
                            disparity_rows.append(
                                {
                                    "dataset": dataset,
                                    "base_seed": seed,
                                    "unlearning_method": method,
                                    "target": target,
                                    "attack_a": a1,
                                    "attack_b": a2,
                                    "jaccard": jaccard(attack_to_positive[a1], attack_to_positive[a2]),
                                }
                            )

                    with (out_dir / "disparity_pairwise_jaccard.csv").open("w", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=["dataset", "base_seed", "unlearning_method", "target", "attack_a", "attack_b", "jaccard"])
                        writer.writeheader()
                        writer.writerows(disparity_rows)

                    # 2b) Per-attack coverage: how many samples each attack flags as members
                    union_positives = set().union(*[attack_to_positive[a] for a in attacks]) if attacks else set()
                    coverage_rows = []
                    for a in attacks:
                        positives = attack_to_positive[a]
                        coverage_rows.append({
                            "dataset": dataset,
                            "base_seed": seed,
                            "unlearning_method": method,
                            "target": target,
                            "attack": a,
                            "total_samples": len(all_sample_ids),
                            "positive_count": len(positives),
                            "coverage_fraction": round(len(positives) / max(1, len(all_sample_ids)), 6),
                        })
                    # Synthetic union row: coverage of OR ensemble
                    coverage_rows.append({
                        "dataset": dataset,
                        "base_seed": seed,
                        "unlearning_method": method,
                        "target": target,
                        "attack": "union_or",
                        "total_samples": len(all_sample_ids),
                        "positive_count": len(union_positives),
                        "coverage_fraction": round(len(union_positives) / max(1, len(all_sample_ids)), 6),
                    })

                    with (out_dir / "coverage_per_attack.csv").open("w", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(
                            f,
                            fieldnames=["dataset", "base_seed", "unlearning_method", "target", "attack", "total_samples", "positive_count", "coverage_fraction"],
                        )
                        writer.writeheader()
                        writer.writerows(coverage_rows)

                    ensemble_rows = []
                    metric_rows = []

                    vote_count_by_sid = {
                        sid: sum(attack_to_sample_prediction[a].get(sid, 0) for a in attacks)
                        for sid in all_sample_ids
                    }

                    labels_arr = np.asarray([int(true_membership[sid]) for sid in all_sample_ids], dtype=int)

                    # 3) OR voting: use vote counts as scores and evaluate on ROC at target FPR
                    if cfg["ensemble"]["voting_rules"].get("or", False):
                        or_scores = np.asarray(
                            [float(vote_count_by_sid[sid]) if vote_count_by_sid[sid] >= 1 else 0.0 for sid in all_sample_ids],
                            dtype=float,
                        )
                        tpr, fpr, acc, thr = metrics_at_target_fpr(labels_arr, or_scores, target_fpr)
                        or_predictions = (or_scores >= thr).astype(int)
                        for idx, sid in enumerate(all_sample_ids):
                            ensemble_rows.append(
                                {
                                    "dataset": dataset,
                                    "base_seed": seed,
                                    "unlearning_method": method,
                                    "target": target,
                                    "rule": "or",
                                    "k": 1,
                                    "m": m,
                                    "sample_id": sid,
                                    "prediction": int(or_predictions[idx]),
                                    "true_membership": true_membership[sid],
                                }
                            )
                        metric_rows.append(
                            {
                                "dataset": dataset,
                                "base_seed": seed,
                                "unlearning_method": method,
                                "target": target,
                                "rule": "or",
                                "k": 1,
                                "m": m,
                                "tpr": tpr,
                                "fpr": fpr,
                                "accuracy": acc,
                            }
                        )

                    # 4) k-of-m voting: use vote counts as scores and evaluate on ROC at target FPR
                    for k in k_values:
                        if k > m:
                            continue
                        k_scores = np.asarray(
                            [float(vote_count_by_sid[sid]) if vote_count_by_sid[sid] >= k else 0.0 for sid in all_sample_ids],
                            dtype=float,
                        )
                        tpr, fpr, acc, thr = metrics_at_target_fpr(labels_arr, k_scores, target_fpr)
                        k_predictions = (k_scores >= thr).astype(int)
                        for idx, sid in enumerate(all_sample_ids):
                            ensemble_rows.append(
                                {
                                    "dataset": dataset,
                                    "base_seed": seed,
                                    "unlearning_method": method,
                                    "target": target,
                                    "rule": "k_of_m",
                                    "k": k,
                                    "m": m,
                                    "sample_id": sid,
                                    "prediction": int(k_predictions[idx]),
                                    "true_membership": true_membership[sid],
                                }
                            )
                        metric_rows.append(
                            {
                                "dataset": dataset,
                                "base_seed": seed,
                                "unlearning_method": method,
                                "target": target,
                                "rule": "k_of_m",
                                "k": k,
                                "m": m,
                                "tpr": tpr,
                                "fpr": fpr,
                                "accuracy": acc,
                            }
                        )

                    with (out_dir / "ensemble_per_sample.csv").open("w", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(
                            f,
                            fieldnames=["dataset", "base_seed", "unlearning_method", "target", "rule", "k", "m", "sample_id", "prediction", "true_membership"],
                        )
                        writer.writeheader()
                        writer.writerows(ensemble_rows)

                    # 5) Save per-rule ROC-derived metrics at the configured target FPR.

                    with (out_dir / "ensemble_metrics.csv").open("w", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(
                            f,
                            fieldnames=["dataset", "base_seed", "unlearning_method", "target", "rule", "k", "m", "tpr", "fpr", "accuracy"],
                        )
                        writer.writeheader()
                        writer.writerows(metric_rows)

                    print(f"Saved ensemble evaluation: {out_dir}")


if __name__ == "__main__":
    main()
