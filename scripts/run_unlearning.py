from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path so imports work regardless of cwd
sys.path.insert(0, str(Path(__file__).parent.parent))

from adapters.io_utils import ensure_dir, load_config, resolve_path
from adapters.machine_unlearning_adapter import MachineUnlearningAdapter


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run unlearning methods via MachineUnlearning engine")
    p.add_argument("--config", default="configs/experiment.yaml")
    p.add_argument("--dataset")
    p.add_argument("--seed", type=int)
    p.add_argument("--method")
    return p.parse_args()


def main() -> None:
    # 1) Read CLI + config
    args = parse_args()
    cfg = load_config(args.config)

    # 2) Adapter delegates unlearning execution to external bridge code
    root = Path(__file__).resolve().parents[1]
    adapter = MachineUnlearningAdapter(root, root / "external" / "machine_unlearning_bridge.py")

    engine_repo = resolve_path(cfg["paths"]["machine_unlearning_repo"])
    data_root = resolve_path(cfg["paths"]["data_root"])
    results_root = resolve_path(cfg["paths"]["results_root"])

    # 3) Optional filters let you run one dataset/seed/method quickly
    datasets = [args.dataset] if args.dataset else cfg["experiment"]["datasets"]
    seeds = [args.seed] if args.seed is not None else cfg["experiment"]["base_seeds"]
    methods = [args.method] if args.method else cfg["unlearning"]["methods"]

    for dataset in datasets:
        for seed in seeds:
            split_file = results_root / "splits" / dataset / f"seed_{seed}.npz"
            if not split_file.exists():
                raise FileNotFoundError(f"Missing split file: {split_file}. Run scripts/prepare_splits.py first.")

            # Unlearning always starts from the previously trained baseline checkpoint
            baseline_model = results_root / "models" / dataset / f"seed_{seed}" / "baseline.pt"
            if not baseline_model.exists():
                raise FileNotFoundError(f"Missing baseline model: {baseline_model}. Run scripts/run_baseline.py first.")

            for method in methods:
                # Save each unlearned model next to baseline model for easy discovery
                out_dir = ensure_dir(results_root / "models" / dataset / f"seed_{seed}")
                model_out = out_dir / f"unlearn_{method}.pt"
                adapter.run_unlearning(
                    dataset=dataset,
                    seed=seed,
                    unlearning_method=method,
                    split_file=split_file,
                    baseline_model_path=baseline_model,
                    model_out=model_out,
                    data_root=data_root,
                    engine_repo=engine_repo,
                    training_cfg=cfg["training"]["unlearning"],
                    device=cfg["experiment"].get("device", "cuda"),
                )
                print(f"Saved unlearned model: {model_out}")


if __name__ == "__main__":
    main()
