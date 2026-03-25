"""aggregate_utility.py — Collect per-method utility summaries into a single comparison table.

Input:  results/utility/<dataset>/seed_<seed>/<method>/utility_summary.json
Output: results/aggregate/utility_comparison.csv

Columns:
    dataset, base_seed, split_mode, target_class, forget_count, forget_fraction,
    model, train_acc, retain_acc, forget_acc, test_acc,
    utility_drop, retain_drop, forget_drop
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from adapters.io_utils import ensure_dir, load_config, load_json, resolve_path

FIELDS = [
    "dataset",
    "base_seed",
    "split_mode",
    "target_class",
    "forget_count",
    "forget_fraction",
    "model",
    "train_acc",
    "retain_acc",
    "forget_acc",
    "test_acc",
    "utility_drop",
    "retain_drop",
    "forget_drop",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Aggregate per-method utility summaries into a single comparison CSV."
    )
    p.add_argument("--config", default="configs/experiment.yaml")
    p.add_argument("--dataset", help="Restrict to one dataset name.")
    p.add_argument("--seed", type=int, help="Restrict to one base seed.")
    return p.parse_args()


def collect_rows(results_root: Path, dataset: str, seed: int) -> list[dict]:
    utility_dir = results_root / "utility" / dataset / f"seed_{seed}"
    if not utility_dir.exists():
        return []

    baseline_row: dict | None = None
    method_rows: list[dict] = []

    for method_dir in sorted(utility_dir.iterdir()):
        if not method_dir.is_dir():
            continue
        summary_file = method_dir / "utility_summary.json"
        if not summary_file.exists():
            continue

        s = load_json(summary_file)
        common = {
            "dataset": dataset,
            "base_seed": seed,
            "split_mode": s.get("split_mode", "targeted_random"),
            "target_class": s.get("target_class"),
            "forget_count": s.get("forget_count"),
            "forget_fraction": s.get("forget_fraction"),
        }

        # Baseline row — same in every summary; emit once
        if baseline_row is None:
            baseline_row = {
                **common,
                "model": "baseline",
                "train_acc": s.get("baseline_train_acc"),
                "retain_acc": s.get("baseline_retain_acc"),
                "forget_acc": s.get("baseline_forget_acc"),
                "test_acc": s.get("baseline_test_acc"),
                "utility_drop": 0.0,
                "retain_drop": 0.0,
                "forget_drop": 0.0,
            }

        method_rows.append({
            **common,
            "model": s.get("unlearning_method", method_dir.name),
            "train_acc": None,  # not collected for unlearned models
            "retain_acc": s.get("unlearn_retain_acc"),
            "forget_acc": s.get("unlearn_forget_acc"),
            "test_acc": s.get("unlearn_test_acc"),
            "utility_drop": s.get("utility_drop"),
            "retain_drop": s.get("retain_drop"),
            "forget_drop": s.get("forget_drop"),
        })

    rows: list[dict] = []
    if baseline_row:
        rows.append(baseline_row)
    rows.extend(method_rows)
    return rows


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    results_root = resolve_path(cfg["paths"]["results_root"])

    datasets = [args.dataset] if args.dataset else cfg["experiment"]["datasets"]
    seeds = [args.seed] if args.seed is not None else cfg["experiment"]["base_seeds"]

    all_rows: list[dict] = []
    for dataset in datasets:
        for seed in seeds:
            rows = collect_rows(results_root, dataset, seed)
            if not rows:
                print(f"Warning: no utility summaries found for {dataset}/seed_{seed}")
            all_rows.extend(rows)

    if not all_rows:
        print("No utility summaries found. Run scripts/run_utility_eval.py first.")
        sys.exit(1)

    out_dir = ensure_dir(results_root / "aggregate")
    out_file = out_dir / "utility_comparison.csv"
    with out_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Saved {len(all_rows)} rows → {out_file}")


if __name__ == "__main__":
    main()
