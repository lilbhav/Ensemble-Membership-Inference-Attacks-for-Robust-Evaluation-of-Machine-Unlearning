from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

# Add project root to path so imports work regardless of cwd
sys.path.insert(0, str(Path(__file__).parent.parent))

from adapters.io_utils import ensure_dir, load_config, resolve_path
from adapters.machine_unlearning_adapter import MachineUnlearningAdapter


# Bounded 4-run sweep: designed to finish quickly and avoid open-ended tuning.
AMNESIAC_GRID = [
    {"name": "amnesiac_e4_adam_bs64", "epochs": 4, "optimizer": "adam", "batch_size": 64},
    {"name": "amnesiac_e6_adam_bs64", "epochs": 6, "optimizer": "adam", "batch_size": 64},
    {"name": "amnesiac_e8_adam_bs64", "epochs": 8, "optimizer": "adam", "batch_size": 64},
    {"name": "amnesiac_e6_adam_bs128", "epochs": 6, "optimizer": "adam", "batch_size": 128},
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a bounded 4-run amnesiac sweep")
    p.add_argument("--config", default="configs/experiment_cifar100.yaml")
    p.add_argument("--dataset", default="Cifar100")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def score_row(forget_acc: float, test_acc: float, baseline_test_acc: float) -> float:
    # Lower forget_acc is better. Keep a mild utility preference.
    utility_drop = baseline_test_acc - test_acc
    return forget_acc + max(0.0, utility_drop - 5.0) * 2.0


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

    baseline_model = results_root / "models" / args.dataset / f"seed_{args.seed}" / "baseline.pt"
    if not baseline_model.exists():
        raise FileNotFoundError(f"Missing baseline model: {baseline_model}. Run scripts/run_baseline.py first.")

    baseline_metrics_path = results_root / "models" / args.dataset / f"seed_{args.seed}" / "baseline.metrics.json"
    if not baseline_metrics_path.exists():
        raise FileNotFoundError(f"Missing baseline metrics: {baseline_metrics_path}")

    baseline_metrics = json.loads(baseline_metrics_path.read_text(encoding="utf-8"))
    baseline_test_acc = float(baseline_metrics.get("test_acc", 0.0))

    out_dir = ensure_dir(results_root / "models" / args.dataset / f"seed_{args.seed}")

    rows: list[dict[str, float | str]] = []

    for preset in AMNESIAC_GRID:
        run_name = str(preset["name"])
        print(f"\\n=== Running {run_name} ===")

        training_cfg = copy.deepcopy(cfg["training"]["unlearning"])
        method_cfg = copy.deepcopy(training_cfg["methods"]["amnesiac"])
        method_cfg.update(
            {
                "epochs": int(preset["epochs"]),
                "optimizer": str(preset["optimizer"]),
                "batch_size": int(preset["batch_size"]),
            }
        )
        training_cfg["methods"]["amnesiac"] = method_cfg

        model_out = out_dir / f"unlearn_{run_name}.pt"

        adapter.run_unlearning(
            dataset=args.dataset,
            seed=args.seed,
            unlearning_method="amnesiac",
            split_file=split_file,
            baseline_model_path=baseline_model,
            model_out=model_out,
            data_root=data_root,
            engine_repo=engine_repo,
            training_cfg=training_cfg,
            device=cfg["experiment"].get("device", "cuda"),
            run_name=run_name,
        )

        metrics_path = model_out.with_suffix(".metrics.json")
        if not metrics_path.exists():
            raise FileNotFoundError(f"Expected metrics not found: {metrics_path}")

        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        forget_acc = float(metrics.get("forget_acc", 0.0))
        test_acc = float(metrics.get("test_acc", 0.0))
        retain_acc = float(metrics.get("retain_acc", 0.0))
        utility_drop = baseline_test_acc - test_acc

        rows.append(
            {
                "run_name": run_name,
                "forget_acc": forget_acc,
                "retain_acc": retain_acc,
                "test_acc": test_acc,
                "utility_drop": utility_drop,
                "score": score_row(forget_acc=forget_acc, test_acc=test_acc, baseline_test_acc=baseline_test_acc),
            }
        )

    rows_sorted = sorted(rows, key=lambda r: float(r["score"]))

    print("\n=== Amnesiac Sweep Summary ===")
    print(f"Baseline test_acc: {baseline_test_acc:.4f}")
    for r in rows_sorted:
        print(
            f"{r['run_name']}: forget_acc={float(r['forget_acc']):.4f}, "
            f"test_acc={float(r['test_acc']):.4f}, retain_acc={float(r['retain_acc']):.4f}, "
            f"utility_drop={float(r['utility_drop']):.4f}, score={float(r['score']):.4f}"
        )

    best = rows_sorted[0]
    print(
        "\nRecommended run: "
        f"{best['run_name']} (lowest score balancing forgetting and utility)."
    )


if __name__ == "__main__":
    main()
