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


def main() -> None:
    # 1) Load configuration and optional CLI filters
    args = parse_args()
    cfg = load_config(args.config)
    results_root = resolve_path(cfg["paths"]["results_root"])

    datasets = [args.dataset] if args.dataset else cfg["experiment"]["datasets"]
    seeds = [args.seed] if args.seed is not None else cfg["experiment"]["base_seeds"]
    methods = [args.method] if args.method else ["baseline", *cfg["unlearning"]["methods"]]
    targets = [args.target] if args.target else cfg["mia"]["targets"]

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
                        positives = set()
                        for r in rows:
                            sid = int(r["sample_id"])
                            pred = int(r["prediction"])
                            sample_prediction[sid] = pred
                            true_membership[sid] = int(r["true_membership"])
                            if pred == 1:
                                positives.add(sid)
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

                    ensemble_rows = []

                    # 3) OR voting: positive if any attack votes positive
                    if cfg["ensemble"]["voting_rules"].get("or", False):
                        for sid in all_sample_ids:
                            votes = sum(attack_to_sample_prediction[a].get(sid, 0) for a in attacks)
                            pred = 1 if votes >= 1 else 0
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
                                    "prediction": pred,
                                    "true_membership": true_membership[sid],
                                }
                            )

                    # 4) k-of-m voting: positive if at least k attacks vote positive
                    for k in k_values:
                        if k > m:
                            continue
                        for sid in all_sample_ids:
                            votes = sum(attack_to_sample_prediction[a].get(sid, 0) for a in attacks)
                            pred = 1 if votes >= k else 0
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
                                    "prediction": pred,
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
