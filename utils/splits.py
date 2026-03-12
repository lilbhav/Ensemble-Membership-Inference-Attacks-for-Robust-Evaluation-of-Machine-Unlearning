import os
import json
import numpy as np
import torch
from torch.utils.data import Subset
from typing import List, Optional, Tuple


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def _split_file_paths(split_dir: str) -> Tuple[str, str, str]:
    forget_path = os.path.join(split_dir, "forget_idx.npy")
    retain_path = os.path.join(split_dir, "retain_idx.npy")
    meta_path = os.path.join(split_dir, "split_meta.json")
    return forget_path, retain_path, meta_path


def validate_retain_forget_indices(
    forget_idx: np.ndarray,
    retain_idx: np.ndarray,
    dataset_size: int,
    expected_forget_fraction: Optional[float] = None,
) -> List[str]:
    """Return a list of split integrity issues (empty means valid)."""
    issues: List[str] = []

    forget_idx = np.asarray(forget_idx)
    retain_idx = np.asarray(retain_idx)

    if forget_idx.ndim != 1 or retain_idx.ndim != 1:
        issues.append("Split indices must be 1D arrays.")
        return issues

    if len(forget_idx) + len(retain_idx) != dataset_size:
        issues.append(
            "retain+forget size mismatch: "
            f"retain={len(retain_idx)} forget={len(forget_idx)} total={dataset_size}"
        )

    if len(np.unique(forget_idx)) != len(forget_idx):
        issues.append("Duplicate indices detected in forget split.")
    if len(np.unique(retain_idx)) != len(retain_idx):
        issues.append("Duplicate indices detected in retain split.")

    forget_set = set(int(v) for v in forget_idx.tolist())
    retain_set = set(int(v) for v in retain_idx.tolist())
    overlap = len(forget_set.intersection(retain_set))
    if overlap > 0:
        issues.append(f"retain/forget overlap detected: {overlap} indices.")

    if len(forget_idx) > 0:
        if int(np.min(forget_idx)) < 0 or int(np.max(forget_idx)) >= dataset_size:
            issues.append("Forget indices are out of dataset bounds.")
    if len(retain_idx) > 0:
        if int(np.min(retain_idx)) < 0 or int(np.max(retain_idx)) >= dataset_size:
            issues.append("Retain indices are out of dataset bounds.")

    if expected_forget_fraction is not None:
        expected_forget = int(dataset_size * float(expected_forget_fraction))
        if len(forget_idx) != expected_forget:
            issues.append(
                "Forget split size does not match expected forget_fraction: "
                f"expected={expected_forget}, got={len(forget_idx)}"
            )

    return issues


def _write_split_metadata(
    meta_path: str,
    dataset_size: int,
    forget_fraction: float,
    seed: int,
    forget_count: int,
    retain_count: int,
) -> None:
    payload = {
        "version": 1,
        "dataset_size": int(dataset_size),
        "forget_fraction": float(forget_fraction),
        "seed": int(seed),
        "forget_count": int(forget_count),
        "retain_count": int(retain_count),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def create_retain_forget_split(
    dataset,
    forget_fraction: float = 0.1,
    seed: int = 0,
    save_dir: str = None
):
    """
    Splits dataset into retain and forget subsets.
    Optionally saves indices to disk.
    """
    if not (0.0 < float(forget_fraction) < 1.0):
        raise ValueError(f"forget_fraction must be in (0, 1), got {forget_fraction}")

    set_seed(seed)
    n = len(dataset)
    indices = np.random.permutation(n)
    n_forget = int(n * forget_fraction)

    forget_idx = indices[:n_forget]
    retain_idx = indices[n_forget:]

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        forget_path, retain_path, meta_path = _split_file_paths(save_dir)
        np.save(forget_path, forget_idx)
        np.save(retain_path, retain_idx)
        _write_split_metadata(
            meta_path=meta_path,
            dataset_size=n,
            forget_fraction=float(forget_fraction),
            seed=int(seed),
            forget_count=len(forget_idx),
            retain_count=len(retain_idx),
        )

    forget_set = Subset(dataset, forget_idx)
    retain_set = Subset(dataset, retain_idx)

    return retain_set, forget_set


def create_auxiliary_split(
    dataset,
    aux_fraction: float = 0.5,
    seed: int = 0,
    save_dir: str = None
):
    """
    Creates auxiliary dataset indices for attacker models.
    """
    set_seed(seed)
    n = len(dataset)
    indices = np.random.permutation(n)
    n_aux = int(n * aux_fraction)

    aux_idx = indices[:n_aux]

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, "aux_idx.npy"), aux_idx)

    aux_set = Subset(dataset, aux_idx)
    return aux_set


def load_split(dataset, split_dir: str):
    """
    Loads retain/forget splits from saved indices.
    """
    forget_path, retain_path, _ = _split_file_paths(split_dir)
    forget_idx = np.load(forget_path)
    retain_idx = np.load(retain_path)

    forget_set = Subset(dataset, forget_idx)
    retain_set = Subset(dataset, retain_idx)

    return retain_set, forget_set


def load_split_checked(
    dataset,
    split_dir: str,
    expected_forget_fraction: Optional[float] = None,
    expected_seed: Optional[int] = None,
) -> Tuple[Subset, Subset]:
    """Load split files and validate integrity and metadata consistency."""
    forget_path, retain_path, meta_path = _split_file_paths(split_dir)
    forget_idx = np.load(forget_path)
    retain_idx = np.load(retain_path)

    issues = validate_retain_forget_indices(
        forget_idx=forget_idx,
        retain_idx=retain_idx,
        dataset_size=len(dataset),
        expected_forget_fraction=expected_forget_fraction,
    )

    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if int(meta.get("dataset_size", -1)) != len(dataset):
                issues.append(
                    "Split metadata dataset_size mismatch: "
                    f"meta={meta.get('dataset_size')} current={len(dataset)}"
                )
            if expected_forget_fraction is not None and float(meta.get("forget_fraction", -1.0)) != float(
                expected_forget_fraction
            ):
                issues.append(
                    "Split metadata forget_fraction mismatch: "
                    f"meta={meta.get('forget_fraction')} expected={expected_forget_fraction}"
                )
            if expected_seed is not None and int(meta.get("seed", -1)) != int(expected_seed):
                issues.append(
                    f"Split metadata seed mismatch: meta={meta.get('seed')} expected={expected_seed}"
                )
        except Exception as ex:
            issues.append(f"Failed to parse split metadata: {ex}")
    else:
        if expected_seed is not None or expected_forget_fraction is not None:
            issues.append("split_meta.json is missing for expected split validation.")

    if issues:
        details = "\n - ".join(issues)
        raise ValueError(f"Invalid or stale split in '{split_dir}':\n - {details}")

    forget_set = Subset(dataset, forget_idx)
    retain_set = Subset(dataset, retain_idx)
    return retain_set, forget_set


def ensure_retain_forget_split(
    dataset,
    split_dir: str,
    forget_fraction: float,
    seed: int,
    verbose: bool = True,
) -> Tuple[Subset, Subset, bool]:
    """
    Load a random retain/forget split if valid, otherwise recreate it.

    Returns:
        retain_set, forget_set, recreated
    """
    forget_path, retain_path, _ = _split_file_paths(split_dir)
    has_files = os.path.exists(forget_path) and os.path.exists(retain_path)

    if has_files:
        try:
            retain_set, forget_set = load_split_checked(
                dataset,
                split_dir=split_dir,
                expected_forget_fraction=forget_fraction,
                expected_seed=seed,
            )
            if verbose:
                print("Loading retain/forget splits from disk (validated)...")
            return retain_set, forget_set, False
        except Exception as ex:
            if verbose:
                print(f"Existing split failed validation, recreating split: {ex}")

    if verbose:
        print("Creating retain/forget splits...")
    retain_set, forget_set = create_retain_forget_split(
        dataset,
        forget_fraction=forget_fraction,
        seed=seed,
        save_dir=split_dir,
    )
    return retain_set, forget_set, True


def load_auxiliary(dataset, aux_dir: str):
    """
    Loads auxiliary split from saved indices.
    """
    aux_idx = np.load(os.path.join(aux_dir, "aux_idx.npy"))
    aux_set = Subset(dataset, aux_idx)
    return aux_set
