from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path so imports work regardless of cwd
sys.path.insert(0, str(Path(__file__).parent.parent))

from adapters.io_utils import ensure_dir, load_config, resolve_path
from adapters.mia_disparity_adapter import MiaDisparityAdapter


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run MIA attacks via mia-disparity engine")
    p.add_argument("--config", default="configs/experiment.yaml")
    p.add_argument("--dataset")
    p.add_argument("--seed", type=int)
    p.add_argument("--method", help="baseline, scrub, ssd, bad_teacher, amnesiac")
    p.add_argument("--attack")
    p.add_argument("--target")
    p.add_argument("--attack-seed", type=int)
    return p.parse_args()


def collect_model_specs(results_root: Path, dataset: str, seed: int, method_filter: str | None):
    # Baseline model is always a candidate attack target
    model_specs = [("baseline", "baseline", results_root / "models" / dataset / f"seed_{seed}" / "baseline.pt")]

    for method in ["scrub", "ssd", "bad_teacher", "amnesiac"]:
        if method_filter and method_filter != method:
            continue
        model_specs.append((f"unlearn_{method}", method, results_root / "models" / dataset / f"seed_{seed}" / f"unlearn_{method}.pt"))

    if method_filter == "baseline":
        return [model_specs[0]]
    return model_specs


def main() -> None:
    # 1) Read CLI + config
    args = parse_args()
    cfg = load_config(args.config)

    # 2) Adapter wraps subprocess calls into the attack bridge
    root = Path(__file__).resolve().parents[1]
    adapter = MiaDisparityAdapter(root, root / "external" / "mia_disparity_bridge.py")

    engine_repo = resolve_path(cfg["paths"]["mia_disparity_repo"])
    data_root = resolve_path(cfg["paths"]["data_root"])
    results_root = resolve_path(cfg["paths"]["results_root"])

    # 3) Expand optional filters into iteration lists
    datasets = [args.dataset] if args.dataset else cfg["experiment"]["datasets"]
    seeds = [args.seed] if args.seed is not None else cfg["experiment"]["base_seeds"]
    attacks = [args.attack] if args.attack else cfg["mia"]["attacks"]
    targets = [args.target] if args.target else cfg["mia"]["targets"]
    attack_seeds = [args.attack_seed] if args.attack_seed is not None else cfg["mia"]["attack_seeds"]

    for dataset in datasets:
        for seed in seeds:
            split_file = results_root / "splits" / dataset / f"seed_{seed}.npz"
            if not split_file.exists():
                raise FileNotFoundError(f"Missing split file: {split_file}")

            # Build list of model checkpoints to attack (baseline + unlearned models)
            model_specs = collect_model_specs(results_root, dataset, seed, args.method)

            for model_name, unlearning_method, model_path in model_specs:
                if not model_path.exists():
                    # If user explicitly requested a method, fail loudly; otherwise skip missing runs
                    if args.method:
                        raise FileNotFoundError(f"Missing required model for selected method: {model_path}")
                    continue

                for attack_name in attacks:
                    for target_name in targets:
                        for attack_seed in attack_seeds:
                            out_dir = ensure_dir(
                                results_root
                                / "mia"
                                / dataset
                                / f"seed_{seed}"
                                / unlearning_method
                                / attack_name
                            )
                            out_csv = out_dir / f"{target_name}_attack_seed_{attack_seed}.csv"

                            # One subprocess call per attack/target/model/seed combination
                            adapter.run_attack(
                                dataset=dataset,
                                base_seed=seed,
                                attack_name=attack_name,
                                attack_seed=attack_seed,
                                target_name=target_name,
                                split_file=split_file,
                                model_path=model_path,
                                output_csv=out_csv,
                                data_root=data_root,
                                engine_repo=engine_repo,
                                attack_epochs=int(cfg["mia"]["attack_epochs"]),
                                batch_size=int(cfg["mia"]["batch_size"]),
                                device=cfg["experiment"].get("device", "cuda"),
                                unlearning_method=unlearning_method,
                                model_name=model_name,
                            )
                            print(f"Saved MIA predictions: {out_csv}")


if __name__ == "__main__":
    main()
