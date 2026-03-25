from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Add project root to path so imports work regardless of cwd
sys.path.insert(0, str(Path(__file__).parent.parent))

from adapters.io_utils import ensure_dir, load_config, resolve_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create canonical retain/forget/test/aux split files")
    p.add_argument("--config", default="configs/experiment.yaml")
    p.add_argument("--dataset")
    p.add_argument("--seed", type=int)
    return p.parse_args()


def _stratified_partition(indices: np.ndarray, labels: np.ndarray, ratios: dict[str, float], rng: np.random.Generator):
    # Split each class independently so forget/aux ratios are balanced across labels
    class_ids = np.unique(labels)
    out = {"forget": [], "aux": [], "retain": []}

    for cls in class_ids:
        cls_idxs = indices[labels == cls]
        rng.shuffle(cls_idxs)

        n = len(cls_idxs)
        n_forget = int(round(n * ratios["forget"]))
        n_aux = int(round(n * ratios["aux"]))
        n_forget = min(n_forget, n)
        n_aux = min(n_aux, n - n_forget)

        forget = cls_idxs[:n_forget]
        aux = cls_idxs[n_forget:n_forget + n_aux]
        retain = cls_idxs[n_forget + n_aux:]

        out["forget"].append(forget)
        out["aux"].append(aux)
        out["retain"].append(retain)

    return {k: np.concatenate(v).astype(np.int64) if len(v) > 0 else np.array([], dtype=np.int64) for k, v in out.items()}


def _resolve_machine_unlearning_repo(cfg_repo_path: str | Path) -> Path:
    # First, trust config path resolution.
    candidate = resolve_path(cfg_repo_path)
    if (candidate / "src" / "__init__.py").exists():
        return candidate

    # Fallback to the repository-local default location used in this project.
    fallback = Path(__file__).resolve().parents[1] / "Third_Party_Code" / "MachineUnlearning"
    if (fallback / "src" / "__init__.py").exists():
        return fallback

    raise FileNotFoundError(
        "Could not locate MachineUnlearning repo with a valid 'src' package. "
        f"Checked: {candidate} and {fallback}."
    )


def main() -> None:
    # 1) Read config and resolve roots
    args = parse_args()
    cfg = load_config(args.config)

    mu_repo = _resolve_machine_unlearning_repo(cfg["paths"]["machine_unlearning_repo"])
    data_root = resolve_path(cfg["paths"]["data_root"])
    results_root = resolve_path(cfg["paths"]["results_root"])
    split_root = ensure_dir(results_root / "splits")

    # 2) Import dataset loader from third-party MachineUnlearning repo
    import sys

    sys.path.insert(0, str(mu_repo))
    from src import dataset as mu_dataset  # type: ignore

    # 3) Expand optional filters into list form
    datasets = [args.dataset] if args.dataset else cfg["experiment"]["datasets"]
    seeds = [args.seed] if args.seed is not None else cfg["experiment"]["base_seeds"]

    forget_fraction = float(cfg["split"]["forget_fraction"])
    aux_fraction = float(cfg["split"]["aux_fraction"])

    for dataset_name in datasets:
        train_dataset, test_dataset, _, _ = mu_dataset.get_dataset(dataset_name=dataset_name, root=str(data_root), augment=False)
        train_size = len(train_dataset)
        test_size = len(test_dataset)

        labels = np.array([int(train_dataset[i][1]) for i in range(train_size)], dtype=np.int64)
        train_indices = np.arange(train_size, dtype=np.int64)

        for seed in seeds:
            # Use deterministic RNG so the same seed always recreates the same split
            rng = np.random.default_rng(seed)

            if cfg["split"].get("stratified_by_label", True):
                parts = _stratified_partition(
                    indices=train_indices.copy(),
                    labels=labels,
                    ratios={"forget": forget_fraction, "aux": aux_fraction},
                    rng=rng,
                )
            else:
                shuffled = train_indices.copy()
                rng.shuffle(shuffled)
                n_forget = int(round(train_size * forget_fraction))
                n_aux = int(round(train_size * aux_fraction))
                parts = {
                    "forget": shuffled[:n_forget],
                    "aux": shuffled[n_forget:n_forget + n_aux],
                    "retain": shuffled[n_forget + n_aux:],
                }

            test_indices = np.arange(test_size, dtype=np.int64)
            test_fraction = float(cfg["split"].get("test_fraction", 1.0))
            if test_fraction < 1.0:
                # Optional test downsampling for faster experiments
                rng.shuffle(test_indices)
                test_indices = np.sort(test_indices[: int(round(test_size * test_fraction))])

            split_file = split_root / dataset_name / f"seed_{seed}.npz"
            split_file.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                split_file,
                retain_indices=np.sort(parts["retain"]),
                forget_indices=np.sort(parts["forget"]),
                test_indices=np.sort(test_indices),
                aux_indices=np.sort(parts["aux"]),
            )

            # Save a human-readable companion file with split stats
            meta = {
                "dataset": dataset_name,
                "seed": seed,
                "train_size": train_size,
                "test_size": test_size,
                "counts": {
                    "retain": int(len(parts["retain"])),
                    "forget": int(len(parts["forget"])),
                    "test": int(len(test_indices)),
                    "aux": int(len(parts["aux"])),
                },
                "fractions": {
                    "forget_fraction": forget_fraction,
                    "aux_fraction": aux_fraction,
                    "test_fraction": test_fraction,
                },
            }
            meta_file = split_root / dataset_name / f"seed_{seed}.meta.json"
            with meta_file.open("w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

            print(f"Saved split: {split_file}")


if __name__ == "__main__":
    main()
