from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

# Add project root to path so imports work regardless of cwd
sys.path.insert(0, str(Path(__file__).parent.parent))

from adapters.io_utils import ensure_dir, load_config, load_json, resolve_path
from adapters.machine_unlearning_adapter import MachineUnlearningAdapter


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run one-method utility validation (baseline + unlearning) under targeted_random split"
    )
    p.add_argument("--config", default="configs/experiment.yaml")
    p.add_argument("--dataset")
    p.add_argument("--seed", type=int)
    p.add_argument(
        "--method",
        required=True,
        choices=["scrub", "ssd", "bad_teacher", "amnesiac"],
    )
    p.add_argument(
        "--retrain-baseline",
        action="store_true",
        help="Force baseline retraining even if baseline checkpoint already exists.",
    )
    p.add_argument(
        "--rerun-unlearning",
        action="store_true",
        help="Force unlearning even if unlearned checkpoint and metrics already exist.",
    )
    return p.parse_args()


def validate_targeted_random_split(split_file: Path) -> dict:
    if not split_file.exists():
        raise FileNotFoundError(f"Missing split file: {split_file}. Run scripts/prepare_splits.py first.")

    meta_file = split_file.with_suffix(".meta.json")
    if not meta_file.exists():
        raise FileNotFoundError(f"Missing split metadata file: {meta_file}")

    meta = load_json(meta_file)
    if meta.get("split_mode") != "targeted_random":
        raise ValueError(
            f"Unsupported split_mode in metadata ({meta.get('split_mode')}). "
            "Only targeted_random is supported."
        )
    if "target_class" not in meta:
        raise ValueError(f"Split metadata is missing required key 'target_class': {meta_file}")
    if "forget_count" not in meta:
        raise ValueError(f"Split metadata is missing required key 'forget_count': {meta_file}")
    if "forget_fraction" not in meta:
        raise ValueError(f"Split metadata is missing required key 'forget_fraction': {meta_file}")

    return meta


def read_metrics(metrics_path: Path) -> dict:
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")
    payload = load_json(metrics_path)

    required = ["retain_acc", "forget_acc", "test_acc", "split_mode", "target_class"]
    for key in required:
        if key not in payload:
            raise ValueError(f"Metrics file missing required key '{key}': {metrics_path}")

    if payload["split_mode"] != "targeted_random":
        raise ValueError(f"Unexpected split_mode in metrics: {payload['split_mode']} ({metrics_path})")

    return payload


def summarize_utility(
    method: str,
    split_meta: dict,
    baseline_metrics: dict,
    unlearn_metrics: dict,
) -> dict:
    baseline_retain = float(baseline_metrics["retain_acc"])
    baseline_forget = float(baseline_metrics["forget_acc"])
    baseline_test = float(baseline_metrics["test_acc"])

    unlearn_retain = float(unlearn_metrics["retain_acc"])
    unlearn_forget = float(unlearn_metrics["forget_acc"])
    unlearn_test = float(unlearn_metrics["test_acc"])

    return {
        "method": method,
        "target_class": int(split_meta["target_class"]),
        "forget_count": int(split_meta["forget_count"]),
        "baseline_retain_acc": baseline_retain,
        "baseline_forget_acc": baseline_forget,
        "baseline_test_acc": baseline_test,
        "unlearned_retain_acc": unlearn_retain,
        "unlearned_forget_acc": unlearn_forget,
        "unlearned_test_acc": unlearn_test,
        "retain_drop": baseline_retain - unlearn_retain,
        "forget_drop": baseline_forget - unlearn_forget,
        "test_drop": baseline_test - unlearn_test,
    }


def write_summary(summary_path: Path, row: dict) -> None:
    ensure_dir(summary_path.parent)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(row, f, indent=2)

    csv_path = summary_path.with_suffix(".csv")
    fields = list(row.keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    root = Path(__file__).resolve().parents[1]
    adapter = MachineUnlearningAdapter(root, root / "external" / "machine_unlearning_bridge.py")

    engine_repo = resolve_path(cfg["paths"]["machine_unlearning_repo"])
    data_root = resolve_path(cfg["paths"]["data_root"])
    results_root = resolve_path(cfg["paths"]["results_root"])

    datasets = [args.dataset] if args.dataset else cfg["experiment"]["datasets"]
    seeds = [args.seed] if args.seed is not None else cfg["experiment"]["base_seeds"]

    if len(datasets) != 1 or len(seeds) != 1:
        raise ValueError(
            "Utility validation is intentionally one-method, one-dataset, one-seed at a time. "
            "Pass --dataset and --seed explicitly."
        )

    dataset = datasets[0]
    seed = seeds[0]
    method = args.method

    split_file = results_root / "splits" / dataset / f"seed_{seed}.npz"
    split_meta = validate_targeted_random_split(split_file)

    model_dir = ensure_dir(results_root / "models" / dataset / f"seed_{seed}")
    baseline_model = model_dir / "baseline.pt"
    baseline_metrics_path = baseline_model.with_suffix(".metrics.json")

    if args.retrain_baseline or not baseline_model.exists() or not baseline_metrics_path.exists():
        adapter.run_baseline(
            dataset=dataset,
            seed=seed,
            split_file=split_file,
            model_out=baseline_model,
            data_root=data_root,
            engine_repo=engine_repo,
            training_cfg=cfg["training"]["baseline"],
            device=cfg["experiment"].get("device", "cuda"),
        )

    unlearn_model = model_dir / f"unlearn_{method}.pt"
    unlearn_metrics_path = unlearn_model.with_suffix(".metrics.json")
    if args.rerun_unlearning or not unlearn_model.exists() or not unlearn_metrics_path.exists():
        adapter.run_unlearning(
            dataset=dataset,
            seed=seed,
            unlearning_method=method,
            split_file=split_file,
            baseline_model_path=baseline_model,
            model_out=unlearn_model,
            data_root=data_root,
            engine_repo=engine_repo,
            training_cfg=cfg["training"]["unlearning"],
            device=cfg["experiment"].get("device", "cuda"),
        )
    else:
        print(f"Reusing existing unlearned model: {unlearn_model}")

    baseline_metrics = read_metrics(baseline_metrics_path)
    unlearn_metrics = read_metrics(unlearn_model.with_suffix(".metrics.json"))

    summary = summarize_utility(
        method=method,
        split_meta=split_meta,
        baseline_metrics=baseline_metrics,
        unlearn_metrics=unlearn_metrics,
    )

    summary_path = (
        ensure_dir(results_root / "utility" / dataset / f"seed_{seed}" / method)
        / "utility_summary.json"
    )
    write_summary(summary_path, summary)

    print(
        "Saved utility summary: "
        f"{summary_path} (target_class={summary['target_class']}, "
        f"forget_count={summary['forget_count']})"
    )


if __name__ == "__main__":
    main()
