from __future__ import annotations

"""Create canonical targeted_random split artifacts for the full pipeline.

This script is the single source of truth for split generation. It writes:
- seed_<seed>.npz with retain/forget/test/aux index arrays
- seed_<seed>.meta.json with split configuration + realized stats
- seed_<seed>.debug.json with lightweight inspection fields

Downstream scripts (baseline, unlearning, MIA) rely on these files and validate
metadata to prevent silent split drift.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Add project root to path so imports work regardless of cwd
sys.path.insert(0, str(Path(__file__).parent.parent))

from adapters.io_utils import ensure_dir, load_config, resolve_path


def parse_args() -> argparse.Namespace:
    """Parse optional dataset/seed filters and overwrite behavior."""
    p = argparse.ArgumentParser(description="Create canonical retain/forget/test/aux split files")
    p.add_argument("--config", default="configs/experiment.yaml")
    p.add_argument("--dataset")
    p.add_argument("--seed", type=int)
    p.add_argument(
        "--force-overwrite",
        action="store_true",
        help="Overwrite existing split artifacts even when metadata does not match current config.",
    )
    return p.parse_args()


def _targeted_random_partition(
    indices: np.ndarray,
    labels: np.ndarray,
    target_class: int,
    forget_fraction: float | None,
    forget_count: int | None,
    rng: np.random.Generator,
) -> tuple[dict[str, np.ndarray], dict[str, float | int]]:
    """Create targeted_random split for one dataset/seed.

    Rules:
    - Forget samples are drawn only from target_class.
    - Exactly one of forget_fraction or forget_count must be set.
    - Retain is all remaining train samples.
    - Aux is currently emitted as empty for compatibility.
    """
    target_mask = labels == target_class
    target_indices = indices[target_mask]

    if target_indices.size == 0:
        raise ValueError(f"No samples found for target_class={target_class}.")

    if (forget_fraction is None) == (forget_count is None):
        raise ValueError("Exactly one of split.forget_fraction or split.forget_count must be provided.")

    if forget_fraction is not None:
        if forget_fraction <= 0 or forget_fraction > 1:
            raise ValueError("split.forget_fraction must be in (0, 1].")
        computed_forget_count = int(round(target_indices.size * forget_fraction))
    else:
        if forget_count is None or forget_count <= 0:
            raise ValueError("split.forget_count must be a positive integer.")
        if forget_count > int(target_indices.size):
            raise ValueError(
                f"split.forget_count={forget_count} exceeds available target-class samples ({target_indices.size})."
            )
        computed_forget_count = int(forget_count)

    if computed_forget_count <= 0:
        raise ValueError(
            "Computed forget_count is zero. Increase split.forget_fraction or set split.forget_count explicitly."
        )

    shuffled_target = target_indices.copy()
    rng.shuffle(shuffled_target)
    forget = np.sort(shuffled_target[:computed_forget_count]).astype(np.int64)
    retain = np.sort(np.setdiff1d(indices, forget, assume_unique=False)).astype(np.int64)
    aux = np.array([], dtype=np.int64)

    realized_forget_fraction = float(len(forget) / target_indices.size)

    metadata = {
        "target_class_size": int(target_indices.size),
        "forget_count": int(len(forget)),
        "forget_fraction": realized_forget_fraction,
    }
    return {"forget": forget, "retain": retain, "aux": aux}, metadata


def _expected_split_metadata(
    dataset_name: str,
    seed: int,
    target_class: int,
    forget_fraction: float | None,
    forget_count: int | None,
) -> dict[str, object]:
    """Build the config fingerprint used to decide split reuse compatibility."""
    forget_spec = "fraction" if forget_fraction is not None else "count"
    return {
        "split_mode": "targeted_random",
        "dataset": dataset_name,
        "seed": int(seed),
        "target_class": int(target_class),
        "forget_spec": forget_spec,
        "requested_forget_fraction": None if forget_fraction is None else float(forget_fraction),
        "requested_forget_count": None if forget_count is None else int(forget_count),
    }


def _label_histogram(labels: np.ndarray) -> dict[str, int]:
    """Return compact class-count map for debug summaries."""
    if labels.size == 0:
        return {}
    values, counts = np.unique(labels, return_counts=True)
    return {str(int(v)): int(c) for v, c in zip(values, counts)}


def _validate_existing_meta(meta_file: Path, expected: dict[str, object]) -> tuple[bool, str]:
    """Check whether an existing split metadata file matches current config.

    Returns:
    - (True, "") if compatible and safe to reuse
    - (False, reason) if missing/incompatible
    """
    if not meta_file.exists():
        return False, f"Missing metadata file: {meta_file}"

    with meta_file.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    required_keys = [
        "split_mode",
        "dataset",
        "seed",
        "target_class",
        "train_size",
        "retain_size",
        "test_size",
        "forget_spec",
        "requested_forget_fraction",
        "requested_forget_count",
        "forget_fraction",
        "forget_count",
    ]
    for key in required_keys:
        if key not in meta:
            return False, f"Split metadata missing required key '{key}' in {meta_file}"

    if meta["split_mode"] != "targeted_random":
        return False, f"Unsupported split_mode in existing metadata: {meta['split_mode']}"

    mismatches = []
    always_checked = ["split_mode", "dataset", "seed", "target_class", "forget_spec"]
    for key in always_checked:
        expected_value = expected[key]
        actual_value = meta.get(key)
        if actual_value != expected_value:
            mismatches.append(f"{key}: expected={expected_value}, actual={actual_value}")

    if expected["forget_spec"] == "fraction":
        expected_value = expected["requested_forget_fraction"]
        actual_value = meta.get("requested_forget_fraction")
        if actual_value != expected_value:
            mismatches.append(f"requested_forget_fraction: expected={expected_value}, actual={actual_value}")
    else:
        expected_value = expected["requested_forget_count"]
        actual_value = meta.get("requested_forget_count")
        if actual_value != expected_value:
            mismatches.append(f"requested_forget_count: expected={expected_value}, actual={actual_value}")

    if mismatches:
        return False, "; ".join(mismatches)
    return True, ""


def _resolve_machine_unlearning_repo(cfg_repo_path: str | Path) -> Path:
    """Resolve path to Third_Party_Code/MachineUnlearning (with fallback)."""
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
    """Generate or reuse split artifacts for each requested dataset/seed."""
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
    from src import dataset as mu_dataset 

    # 3) Expand optional filters into list form
    datasets = [args.dataset] if args.dataset else cfg["experiment"]["datasets"]
    seeds = [args.seed] if args.seed is not None else cfg["experiment"]["base_seeds"]

    split_cfg = cfg["split"]
    if split_cfg.get("split_mode") != "targeted_random":
        raise ValueError("Only split_mode=targeted_random is supported.")

    if "target_class" not in split_cfg:
        raise ValueError("split.target_class is required for targeted_random splits.")

    target_class = int(split_cfg["target_class"])
    forget_fraction_cfg = split_cfg.get("forget_fraction")
    forget_count_cfg = split_cfg.get("forget_count")

    forget_fraction = None if forget_fraction_cfg is None else float(forget_fraction_cfg)
    forget_count = None if forget_count_cfg is None else int(forget_count_cfg)

    for dataset_name in datasets:
        train_dataset, test_dataset, _, _ = mu_dataset.get_dataset(dataset_name=dataset_name, root=str(data_root), augment=False)
        train_size = len(train_dataset)
        test_size = len(test_dataset)

        labels = np.array([int(train_dataset[i][1]) for i in range(train_size)], dtype=np.int64)
        train_indices = np.arange(train_size, dtype=np.int64)

        for seed in seeds:
            # Use deterministic RNG so the same seed always recreates the same split
            rng = np.random.default_rng(seed)

            split_file = split_root / dataset_name / f"seed_{seed}.npz"
            split_file.parent.mkdir(parents=True, exist_ok=True)
            meta_file = split_root / dataset_name / f"seed_{seed}.meta.json"

            expected = _expected_split_metadata(
                dataset_name=dataset_name,
                seed=seed,
                target_class=target_class,
                forget_fraction=forget_fraction,
                forget_count=forget_count,
            )

            if split_file.exists() or meta_file.exists():
                is_compatible, reason = _validate_existing_meta(meta_file, expected)
                if is_compatible and split_file.exists():
                    print(f"Reusing existing targeted_random split: {split_file}")
                    continue
                if not args.force_overwrite:
                    raise RuntimeError(
                        "Existing split artifacts are incompatible with current config. "
                        "Refusing to reuse silently. "
                        f"Reason: {reason}. "
                        "Rerun with --force-overwrite to regenerate artifacts."
                    )

            parts, derived = _targeted_random_partition(
                indices=train_indices.copy(),
                labels=labels,
                target_class=target_class,
                forget_fraction=forget_fraction,
                forget_count=forget_count,
                rng=rng,
            )

            forget_labels = labels[parts["forget"]] if len(parts["forget"]) > 0 else np.array([], dtype=np.int64)
            retain_labels = labels[parts["retain"]] if len(parts["retain"]) > 0 else np.array([], dtype=np.int64)
            if not np.all(forget_labels == target_class):
                raise AssertionError("Found forget samples outside the selected target_class.")
            if np.intersect1d(parts["retain"], parts["forget"]).size > 0:
                raise AssertionError("retain_indices and forget_indices must be disjoint.")
            union_sorted = np.sort(np.concatenate([parts["retain"], parts["forget"]]))
            if not np.array_equal(union_sorted, train_indices):
                raise AssertionError("retain_indices union forget_indices must equal the full training set.")

            if forget_count is not None and len(parts["forget"]) != forget_count:
                raise AssertionError(
                    f"Forget count mismatch: expected {forget_count}, got {len(parts['forget'])}."
                )
            if forget_fraction is not None:
                expected_from_fraction = int(round(derived["target_class_size"] * forget_fraction))
                if len(parts["forget"]) != expected_from_fraction:
                    raise AssertionError(
                        "Forget count mismatch against forget_fraction within target class: "
                        f"expected {expected_from_fraction}, got {len(parts['forget'])}."
                    )

            test_indices = np.arange(test_size, dtype=np.int64)
            if not np.array_equal(test_indices, np.arange(test_size, dtype=np.int64)):
                raise AssertionError("test_indices must remain unchanged from the canonical full test set.")
            np.savez(
                split_file,
                retain_indices=np.sort(parts["retain"]),
                forget_indices=np.sort(parts["forget"]),
                test_indices=np.sort(test_indices),
                aux_indices=np.sort(parts["aux"]),
            )

            # Save a readable companion file with split stats
            meta = {
                "split_mode": "targeted_random",
                "dataset": dataset_name,
                "seed": seed,
                "target_class": target_class,
                "forget_spec": "fraction" if forget_fraction is not None else "count",
                "requested_forget_fraction": forget_fraction,
                "requested_forget_count": forget_count,
                "forget_count": int(derived["forget_count"]),
                "forget_fraction": float(derived["forget_fraction"]),
                "train_size": train_size,
                "retain_size": int(len(parts["retain"])),
                "test_size": test_size,
                "counts": {
                    "retain": int(len(parts["retain"])),
                    "forget": int(len(parts["forget"])),
                    "test": int(len(test_indices)),
                    "aux": int(len(parts["aux"])),
                },
                "fractions": {
                    "forget_fraction_within_target_class": float(derived["forget_fraction"]),
                },
            }
            with meta_file.open("w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

            debug_summary_file = split_root / dataset_name / f"seed_{seed}.debug.json"
            debug_summary = {
                "dataset": dataset_name,
                "seed": int(seed),
                "split_mode": "targeted_random",
                "target_class": int(target_class),
                "forget_count": int(len(parts["forget"])),
                "forget_fraction": float(derived["forget_fraction"]),
                "train_size": int(train_size),
                "retain_size": int(len(parts["retain"])),
                "test_size": int(test_size),
                "first_20_forget_indices": [int(x) for x in parts["forget"][:20]],
                "first_20_retain_indices": [int(x) for x in parts["retain"][:20]],
                "forget_label_histogram": _label_histogram(forget_labels),
                "retain_label_histogram": _label_histogram(retain_labels),
            }
            with debug_summary_file.open("w", encoding="utf-8") as f:
                json.dump(debug_summary, f, indent=2)

            print(
                f"Split debug summary ({dataset_name}/seed_{seed})\n"
                f"  first_20_forget_indices={debug_summary['first_20_forget_indices']}\n"
                f"  first_20_retain_indices={debug_summary['first_20_retain_indices']}\n"
                f"  forget_label_histogram={debug_summary['forget_label_histogram']}\n"
                f"  retain_label_histogram={debug_summary['retain_label_histogram']}"
            )

            print(
                "Saved targeted_random split: "
                f"{split_file} (target_class={target_class}, forget_count={derived['forget_count']}, "
                f"forget_fraction_within_target_class={derived['forget_fraction']:.6f})"
            )


if __name__ == "__main__":
    main()
