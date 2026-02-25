#!/usr/bin/env python3
"""Run a small hyperparameter sweep for SSD and rank runs by forget-vs-retain tradeoff."""

import argparse
import csv
import itertools
import json
import os
from dataclasses import asdict, replace
from collections import defaultdict
from typing import Dict, List

import torch

from mia.experiments import ssd_experiment


def _parse_float_list(text: str) -> List[float]:
    values = []
    for token in text.split(","):
        token = token.strip()
        if token:
            values.append(float(token))
    if not values:
        raise ValueError("Expected at least one float value.")
    return values


def _parse_int_list(text: str) -> List[int]:
    values = []
    for token in text.split(","):
        token = token.strip()
        if token:
            values.append(int(token))
    if not values:
        raise ValueError("Expected at least one integer seed value.")
    return values


def _safe_token(value: float) -> str:
    return str(value).replace("-", "m").replace(".", "p")


def _compute_tradeoff_score(baseline: Dict[str, float], after: Dict[str, float]) -> Dict[str, float]:
    forget_drop = max(0.0, baseline["tf_acc"] - after["tf_acc"]) + max(
        0.0, baseline["vf_acc"] - after["vf_acc"]
    )
    retain_drop = max(0.0, baseline["tr_acc"] - after["tr_acc"]) + max(
        0.0, baseline["vr_acc"] - after["vr_acc"]
    )
    return {
        "forget_drop": forget_drop,
        "retain_drop": retain_drop,
        "score": forget_drop - retain_drop,
    }


def _device_from_args(base_args) -> torch.device:
    if base_args.device:
        return torch.device(base_args.device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: List[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean_value = _mean(values)
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    return variance ** 0.5


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SSD hyperparameter sweep")
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/ssd_experiment.yaml",
        help="Base SSD config file",
    )
    parser.add_argument(
        "--dampening",
        type=str,
        default="1,2,5,10",
        help="Comma-separated dampening_constant values",
    )
    parser.add_argument(
        "--selection",
        type=str,
        default="10,20,50,100",
        help="Comma-separated selection_weighting values",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for per-run checkpoints/results and sweep summary",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated seeds for averaging (defaults to seed from config)",
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Print sweep outputs to console only (do not save checkpoints/results/summary files)",
    )
    args = parser.parse_args()

    dampening_values = _parse_float_list(args.dampening)
    selection_values = _parse_float_list(args.selection)

    base_args = ssd_experiment._load_config(args.config)
    seeds = _parse_int_list(args.seeds) if args.seeds else [int(base_args.seed)]

    if args.output_dir:
        output_dir = args.output_dir
    elif base_args.check_path:
        output_dir = os.path.dirname(base_args.check_path)
    else:
        output_dir = "./results/ssd_sweep"

    if not output_dir:
        output_dir = "./results/ssd_sweep"

    if not args.print_only:
        os.makedirs(output_dir, exist_ok=True)

    device = _device_from_args(base_args)
    per_run_records = []
    baseline_by_seed = {}
    combinations = list(itertools.product(dampening_values, selection_values))

    total_runs = len(combinations) * len(seeds)
    run_index = 0

    for seed in seeds:
        print(f"\nPreparing data split and baseline for seed={seed}...")
        seed_args = replace(base_args, seed=seed)
        loaders = ssd_experiment._create_loaders(seed_args)

        baseline_model = ssd_experiment.load_model(
            dataset=seed_args.dataset,
            checkpoint_path=seed_args.model_path,
            device=device,
        )
        baseline_acc = ssd_experiment.train_validation(
            baseline_model,
            loaders["train_retain_loader"],
            loaders["train_forget_loader"],
            loaders["valid_retain_loader"],
            loaders["valid_forget_loader"],
            device,
        )
        baseline_by_seed[seed] = baseline_acc

        print(
            "Baseline metrics: "
            f"tr={baseline_acc['tr_acc']:.4f}, tf={baseline_acc['tf_acc']:.4f}, "
            f"vr={baseline_acc['vr_acc']:.4f}, vf={baseline_acc['vf_acc']:.4f}"
        )

        for dampening_constant, selection_weighting in combinations:
            run_index += 1
            run_tag = (
                f"ssd_dc_{_safe_token(dampening_constant)}"
                f"_sw_{_safe_token(selection_weighting)}"
                f"_seed_{seed}"
            )
            run_check_path = None if args.print_only else os.path.join(output_dir, f"{run_tag}.pt")
            run_results_path = None if args.print_only else os.path.join(output_dir, f"{run_tag}.txt")

            run_args = replace(
                seed_args,
                dampening_constant=dampening_constant,
                selection_weighting=selection_weighting,
                check_path=run_check_path,
                results_path=run_results_path,
            )

            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))

            print(
                f"[{run_index}/{total_runs}] Running SSD with "
                f"seed={seed}, dampening_constant={dampening_constant}, "
                f"selection_weighting={selection_weighting}"
            )
            _, after_acc = ssd_experiment.ssd(loaders, run_args)

            tradeoff = _compute_tradeoff_score(baseline_acc, after_acc)
            record = {
                "run_tag": run_tag,
                "seed": seed,
                "dampening_constant": dampening_constant,
                "selection_weighting": selection_weighting,
                "check_path": run_check_path,
                "results_path": run_results_path,
                "tr_acc": after_acc["tr_acc"],
                "tf_acc": after_acc["tf_acc"],
                "vr_acc": after_acc["vr_acc"],
                "vf_acc": after_acc["vf_acc"],
                "delta_tr": after_acc["tr_acc"] - baseline_acc["tr_acc"],
                "delta_tf": after_acc["tf_acc"] - baseline_acc["tf_acc"],
                "delta_vr": after_acc["vr_acc"] - baseline_acc["vr_acc"],
                "delta_vf": after_acc["vf_acc"] - baseline_acc["vf_acc"],
                "forget_drop": tradeoff["forget_drop"],
                "retain_drop": tradeoff["retain_drop"],
                "score": tradeoff["score"],
            }
            per_run_records.append(record)

    grouped = defaultdict(list)
    for row in per_run_records:
        grouped[(row["dampening_constant"], row["selection_weighting"])].append(row)

    aggregated_records = []
    for (dampening_constant, selection_weighting), runs in grouped.items():
        aggregated = {
            "dampening_constant": dampening_constant,
            "selection_weighting": selection_weighting,
            "num_runs": len(runs),
            "score_mean": _mean([row["score"] for row in runs]),
            "score_std": _std([row["score"] for row in runs]),
            "forget_drop_mean": _mean([row["forget_drop"] for row in runs]),
            "forget_drop_std": _std([row["forget_drop"] for row in runs]),
            "retain_drop_mean": _mean([row["retain_drop"] for row in runs]),
            "retain_drop_std": _std([row["retain_drop"] for row in runs]),
            "tr_acc_mean": _mean([row["tr_acc"] for row in runs]),
            "tr_acc_std": _std([row["tr_acc"] for row in runs]),
            "tf_acc_mean": _mean([row["tf_acc"] for row in runs]),
            "tf_acc_std": _std([row["tf_acc"] for row in runs]),
            "vr_acc_mean": _mean([row["vr_acc"] for row in runs]),
            "vr_acc_std": _std([row["vr_acc"] for row in runs]),
            "vf_acc_mean": _mean([row["vf_acc"] for row in runs]),
            "vf_acc_std": _std([row["vf_acc"] for row in runs]),
        }
        aggregated_records.append(aggregated)

    ranked = sorted(
        aggregated_records,
        key=lambda row: (row["score_mean"], row["forget_drop_mean"], -row["retain_drop_mean"]),
        reverse=True,
    )

    if not args.print_only:
        summary_json_path = os.path.join(output_dir, "ssd_sweep_summary.json")
        summary_csv_path = os.path.join(output_dir, "ssd_sweep_summary.csv")

        with open(summary_json_path, "w") as f:
            json.dump(
                {
                    "base_config": args.config,
                    "base_args": asdict(base_args),
                    "seeds": seeds,
                    "baseline_by_seed": baseline_by_seed,
                    "grid": {
                        "dampening": dampening_values,
                        "selection": selection_values,
                    },
                    "per_run_results": per_run_records,
                    "ranked_results": ranked,
                },
                f,
                indent=2,
            )

        fieldnames = list(ranked[0].keys()) if ranked else []
        with open(summary_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if fieldnames:
                writer.writeheader()
                writer.writerows(ranked)

    print("\nSweep complete.")
    if not args.print_only:
        print(f"Summary JSON: {summary_json_path}")
        print(f"Summary CSV:  {summary_csv_path}")
    else:
        print("Print-only mode enabled: no files were saved.")

    if ranked:
        best = ranked[0]
        print("\nBest hyperparameter pair by mean score:")
        print(
            f"  dampening_constant={best['dampening_constant']}, "
            f"selection_weighting={best['selection_weighting']} | "
            f"score_mean={best['score_mean']:.4f} (±{best['score_std']:.4f})"
        )
        print(
            f"  forget_drop_mean={best['forget_drop_mean']:.4f}, "
            f"retain_drop_mean={best['retain_drop_mean']:.4f}"
        )
        print(
            f"  tr={best['tr_acc_mean']:.4f}, tf={best['tf_acc_mean']:.4f}, "
            f"vr={best['vr_acc_mean']:.4f}, vf={best['vf_acc_mean']:.4f}"
        )

        print("\nTop 5 hyperparameter pairs:")
        for rank, row in enumerate(ranked[:5], start=1):
            print(
                f"  {rank}) dc={row['dampening_constant']}, sw={row['selection_weighting']} | "
                f"score_mean={row['score_mean']:.4f} (±{row['score_std']:.4f}) | "
                f"forget_drop_mean={row['forget_drop_mean']:.4f} | "
                f"retain_drop_mean={row['retain_drop_mean']:.4f}"
            )


if __name__ == "__main__":
    main()
