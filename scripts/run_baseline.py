from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path so imports work regardless of cwd
sys.path.insert(0, str(Path(__file__).parent.parent))

from adapters.io_utils import ensure_dir, load_config, resolve_path
from adapters.machine_unlearning_adapter import MachineUnlearningAdapter


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run baseline model training via MachineUnlearning engine")
    p.add_argument("--config", default="configs/experiment.yaml")
    p.add_argument("--dataset")
    p.add_argument("--seed", type=int)
    return p.parse_args()


def main() -> None:
    # 1) Read CLI + experiment config
    args = parse_args()
    cfg = load_config(args.config)

    # 2) Initialize adapter that wraps the external engine bridge
    root = Path(__file__).resolve().parents[1]
    adapter = MachineUnlearningAdapter(root, root / "external" / "machine_unlearning_bridge.py")

    # 3) Resolve configured paths once
    engine_repo = resolve_path(cfg["paths"]["machine_unlearning_repo"])
    data_root = resolve_path(cfg["paths"]["data_root"])
    results_root = resolve_path(cfg["paths"]["results_root"])

    # 4) Allow focused reruns via optional CLI filters
    datasets = [args.dataset] if args.dataset else cfg["experiment"]["datasets"]
    seeds = [args.seed] if args.seed is not None else cfg["experiment"]["base_seeds"]

    for dataset in datasets:
        for seed in seeds:
            # Split is mandatory because both baseline and unlearning must use the same partition
            split_file = results_root / "splits" / dataset / f"seed_{seed}.npz"
            if not split_file.exists():
                raise FileNotFoundError(f"Missing split file: {split_file}. Run scripts/prepare_splits.py first.")

            # Save one baseline checkpoint per (dataset, seed)
            model_out = ensure_dir(results_root / "models" / dataset / f"seed_{seed}") / "baseline.pt"
            adapter.run_baseline(
                dataset=dataset,
                seed=seed,
                split_file=split_file,
                model_out=model_out,
                data_root=data_root,
                engine_repo=engine_repo,
                training_cfg=cfg["training"]["baseline"],
                device=cfg["experiment"].get("device", "cuda"),
            )
            print(f"Saved baseline model: {model_out}")


if __name__ == "__main__":
    main()
