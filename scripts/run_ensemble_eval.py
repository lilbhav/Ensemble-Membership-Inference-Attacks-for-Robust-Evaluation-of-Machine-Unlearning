from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

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


def calibrate_predictions_exact_fpr(
    labels_by_sid: dict[int, int],
    scores_by_sid: dict[int, float],
    target_fpr: float,
) -> tuple[dict[int, int], float]:
    """Deterministically enforce an exact non-member FP budget at target FPR."""
    target_fpr = float(max(0.0, min(1.0, target_fpr)))
    predictions = {sid: 0 for sid in labels_by_sid}

    nonmember_ids = [sid for sid, y in labels_by_sid.items() if int(y) == 0]
    member_ids = [sid for sid, y in labels_by_sid.items() if int(y) == 1]
    n_nonmembers = len(nonmember_ids)

    if n_nonmembers == 0:
        return predictions, float("inf")

    fp_budget = int(round(target_fpr * n_nonmembers))
    fp_budget = max(0, min(fp_budget, n_nonmembers))

    ranked_nonmembers = sorted(nonmember_ids, key=lambda sid: (-float(scores_by_sid[sid]), int(sid)))
    selected_nonmembers = set(ranked_nonmembers[:fp_budget])

    if fp_budget > 0:
        cutoff = float(scores_by_sid[ranked_nonmembers[fp_budget - 1]])
    else:
        cutoff = max(float(scores_by_sid[sid]) for sid in scores_by_sid) + 1e-12

    for sid in selected_nonmembers:
        predictions[sid] = 1

    for sid in member_ids:
        if float(scores_by_sid[sid]) >= cutoff:
            predictions[sid] = 1

    return predictions, cutoff


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
                        sample_scores = {}
                        local_labels = {}
                        for r in rows:
                            sid = int(r["sample_id"])
                            sample_scores[sid] = float(r["score"])
                            local_labels[sid] = int(r["true_membership"])

                        calibrated_pred, _ = calibrate_predictions_exact_fpr(local_labels, sample_scores, target_fpr)
                        positives = {sid for sid, pred in calibrated_pred.items() if int(pred) == 1}

                        for sid, label in local_labels.items():
                            true_membership[sid] = label

                        attack_to_sample_prediction[attack_name] = calibrated_pred
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

                    vote_count_by_sid = {
                        sid: sum(attack_to_sample_prediction[a].get(sid, 0) for a in attacks)
                        for sid in all_sample_ids
                    }

                    # 3) OR voting with exact-FPR calibration
                    if cfg["ensemble"]["voting_rules"].get("or", False):
                        or_scores = {sid: 1.0 if vote_count_by_sid[sid] >= 1 else 0.0 for sid in all_sample_ids}
                        or_predictions, _ = calibrate_predictions_exact_fpr(true_membership, or_scores, target_fpr)
                        for sid in all_sample_ids:
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
                                    "prediction": int(or_predictions[sid]),
                                    "true_membership": true_membership[sid],
                                }
                            )

                    # 4) k-of-m voting with exact-FPR calibration
                    for k in k_values:
                        if k > m:
                            continue
                        k_scores = {sid: 1.0 if vote_count_by_sid[sid] >= k else 0.0 for sid in all_sample_ids}
                        k_predictions, _ = calibrate_predictions_exact_fpr(true_membership, k_scores, target_fpr)
                        for sid in all_sample_ids:
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
                                    "prediction": int(k_predictions[sid]),
                                    "true_membership": true_membership[sid],
                                }
                            )

                    with (out_dir / "ensemble_per_sample.csv").open("w", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(
                            f,
                            fieldnames=["dataset", "base_seed", "unlearning_method", "target", "rule", "k", "m", "sample_id", "prediction", "true_membership"],
                        )
                        writer.writeheader()
                        writer.writerows(ensemble_rows)

                    # 5) Summarize ensemble predictions into TPR/FPR/Accuracy
                    summary = defaultdict(lambda: {"tp": 0, "fp": 0, "tn": 0, "fn": 0})
                    for r in ensemble_rows:
                        key = (r["rule"], int(r["k"]), int(r["m"]))
                        y = int(r["true_membership"])
                        p = int(r["prediction"])
                        if y == 1 and p == 1:
                            summary[key]["tp"] += 1
                        elif y == 0 and p == 1:
                            summary[key]["fp"] += 1
                        elif y == 0 and p == 0:
                            summary[key]["tn"] += 1
                        else:
                            summary[key]["fn"] += 1

                    metric_rows = []
                    for (rule, k, m_val), c in summary.items():
                        tpr = c["tp"] / max(1, c["tp"] + c["fn"])
                        fpr = c["fp"] / max(1, c["fp"] + c["tn"])
                        acc = (c["tp"] + c["tn"]) / max(1, c["tp"] + c["tn"] + c["fp"] + c["fn"])
                        metric_rows.append(
                            {
                                "dataset": dataset,
                                "base_seed": seed,
                                "unlearning_method": method,
                                "target": target,
                                "rule": rule,
                                "k": k,
                                "m": m_val,
                                "tpr": tpr,
                                "fpr": fpr,
                                "accuracy": acc,
                            }
                        )

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
