from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from adapters.io_utils import ensure_dir, load_config, load_json, resolve_path
from adapters.machine_unlearning_adapter import MachineUnlearningAdapter


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a small SCRUB hyperparameter sweep and save a comparison CSV.")
    p.add_argument("--config", default="configs/experiment.yaml")
    p.add_argument("--dataset", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument(
        "--retrain-baseline",
        action="store_true",
        help="Force baseline retraining even if the baseline checkpoint already exists.",
    )
    p.add_argument(
        "--rerun",
        action="store_true",
        help="Force rerunning all SCRUB sweep entries even if their checkpoints already exist.",
    )
    return p.parse_args()


def merge_scrub_config(base_cfg: dict, override_cfg: dict) -> dict:
    merged = dict(base_cfg)
    for key, value in override_cfg.items():
        if key in {"name", "method"}:
            continue
        merged[key] = value
    return merged


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    root = Path(__file__).resolve().parents[1]
    adapter = MachineUnlearningAdapter(root, root / "external" / "machine_unlearning_bridge.py")

    engine_repo = resolve_path(cfg["paths"]["machine_unlearning_repo"])
    data_root = resolve_path(cfg["paths"]["data_root"])
    results_root = resolve_path(cfg["paths"]["results_root"])

    split_file = results_root / "splits" / args.dataset / f"seed_{args.seed}.npz"
    if not split_file.exists():
        raise FileNotFoundError(f"Missing split file: {split_file}. Run scripts/prepare_splits.py first.")

    model_dir = ensure_dir(results_root / "models" / args.dataset / f"seed_{args.seed}")
    baseline_model = model_dir / "baseline.pt"
    baseline_metrics_path = baseline_model.with_suffix(".metrics.json")

    if args.retrain_baseline or not baseline_model.exists() or not baseline_metrics_path.exists():
        adapter.run_baseline(
            dataset=args.dataset,
            seed=args.seed,
            split_file=split_file,
            model_out=baseline_model,
            data_root=data_root,
            engine_repo=engine_repo,
            training_cfg=cfg["training"]["baseline"],
            device=cfg["experiment"].get("device", "cuda"),
        )

    baseline_metrics = load_json(baseline_metrics_path)
    scrub_base_cfg = cfg["training"]["unlearning"].get("scrub", {})
    sweep_entries = cfg["training"]["unlearning"].get("scrub_sweep", [])
    if not sweep_entries:
        raise ValueError("No SCRUB sweep entries configured under training.unlearning.scrub_sweep")

    rows = []
    for entry in sweep_entries:
        run_name = entry["name"]
        method = entry.get("method", "scrub_teacher_loaded")
        scrub_cfg = merge_scrub_config(scrub_base_cfg, entry)

        model_out = model_dir / f"unlearn_{run_name}.pt"
        metrics_path = model_out.with_suffix(".metrics.json")

        if args.rerun or not model_out.exists() or not metrics_path.exists():
            training_cfg = dict(cfg["training"]["unlearning"])
            training_cfg["scrub"] = scrub_cfg
            adapter.run_unlearning(
                dataset=args.dataset,
                seed=args.seed,
                unlearning_method=method,
                split_file=split_file,
                baseline_model_path=baseline_model,
                model_out=model_out,
                data_root=data_root,
                engine_repo=engine_repo,
                training_cfg=training_cfg,
                device=cfg["experiment"].get("device", "cuda"),
                run_name=run_name,
            )
        else:
            print(f"Reusing existing SCRUB sweep run: {model_out}")

        metrics = load_json(metrics_path)
        rows.append(
            {
                "dataset": args.dataset,
                "seed": args.seed,
                "run_name": run_name,
                "method": method,
                "scrub_mode": metrics.get("scrub_mode", scrub_cfg.get("mode", "")),
                "epochs": scrub_cfg.get("epochs"),
                "lr": scrub_cfg.get("lr"),
                "distill_weight": scrub_cfg.get("distill_weight"),
                "forget_loss_weight": scrub_cfg.get("forget_loss_weight"),
                "maximize_epochs": scrub_cfg.get("maximize_epochs"),
                "maximize_steps": scrub_cfg.get("maximize_steps"),
                "minimize_steps": scrub_cfg.get("minimize_steps"),
                "kd_temperature": scrub_cfg.get("kd_temperature"),
                "retain_acc": metrics.get("retain_acc"),
                "forget_acc": metrics.get("forget_acc"),
                "test_acc": metrics.get("test_acc"),
                "train_acc": metrics.get("train_acc"),
                "retain_drop_vs_baseline": float(baseline_metrics["retain_acc"]) - float(metrics["retain_acc"]),
                "forget_drop_vs_baseline": float(baseline_metrics["forget_acc"]) - float(metrics["forget_acc"]),
                "test_drop_vs_baseline": float(baseline_metrics["test_acc"]) - float(metrics["test_acc"]),
                "model_path": str(model_out),
                "history_csv": metrics.get("scrub_history_csv", ""),
            }
        )

    rows.sort(key=lambda row: (row["forget_acc"], row["retain_drop_vs_baseline"], row["test_drop_vs_baseline"]))
    out_path = model_dir / "scrub_sweep_comparison.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved SCRUB sweep comparison: {out_path}")


if __name__ == "__main__":
    main()