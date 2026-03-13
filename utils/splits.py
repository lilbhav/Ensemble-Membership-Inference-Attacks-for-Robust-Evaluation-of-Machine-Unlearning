import os
import json
import numpy as np
import torch
from torch.utils.data import Subset
from typing import Dict, List, Optional, Tuple


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


def _targeted_split_file_paths(split_dir: str) -> Tuple[str, str, str, str]:
    forget_path = os.path.join(split_dir, "targeted_forget_idx.npy")
    retain_path = os.path.join(split_dir, "targeted_retain_idx.npy")
    left_out_path = os.path.join(split_dir, "targeted_left_out_idx.npy")
    meta_path = os.path.join(split_dir, "targeted_split_meta.json")
    return forget_path, retain_path, left_out_path, meta_path


def _extract_targets(dataset) -> np.ndarray:
    """Extract per-sample labels from torchvision datasets or nested Subset wrappers."""
    if isinstance(dataset, Subset):
        base_targets = _extract_targets(dataset.dataset)
        subset_indices = np.asarray(dataset.indices, dtype=int)
        return base_targets[subset_indices]

    targets = getattr(dataset, "targets", None)
    if targets is None:
        targets = getattr(dataset, "labels", None)
    if targets is None:
        raise ValueError("Dataset must expose labels via `targets` or `labels` for targeted split creation.")

    return np.asarray(targets, dtype=int)


def _validate_targeted_split(
    retain_idx: np.ndarray,
    forget_idx: np.ndarray,
    left_out_idx: np.ndarray,
    targets: np.ndarray,
    forget_class: int,
    retain_count: int,
    forget_count: int,
    left_out_count: int,
) -> List[str]:
    issues: List[str] = []

    retain_idx = np.asarray(retain_idx, dtype=int)
    forget_idx = np.asarray(forget_idx, dtype=int)
    left_out_idx = np.asarray(left_out_idx, dtype=int)

    if len(retain_idx) != int(retain_count):
        issues.append(f"retain size mismatch: expected={retain_count}, got={len(retain_idx)}")
    if len(forget_idx) != int(forget_count):
        issues.append(f"forget size mismatch: expected={forget_count}, got={len(forget_idx)}")
    if len(left_out_idx) != int(left_out_count):
        issues.append(f"left_out size mismatch: expected={left_out_count}, got={len(left_out_idx)}")

    if len(np.unique(retain_idx)) != len(retain_idx):
        issues.append("Duplicate indices detected in retain split.")
    if len(np.unique(forget_idx)) != len(forget_idx):
        issues.append("Duplicate indices detected in forget split.")
    if len(np.unique(left_out_idx)) != len(left_out_idx):
        issues.append("Duplicate indices detected in left_out split.")

    retain_set = set(int(v) for v in retain_idx.tolist())
    forget_set = set(int(v) for v in forget_idx.tolist())
    left_out_set = set(int(v) for v in left_out_idx.tolist())

    if retain_set.intersection(forget_set):
        issues.append("retain/forget overlap detected.")
    if retain_set.intersection(left_out_set):
        issues.append("retain/left_out overlap detected.")
    if forget_set.intersection(left_out_set):
        issues.append("forget/left_out overlap detected.")

    if len(forget_idx) > 0:
        if not np.all(targets[forget_idx] == int(forget_class)):
            issues.append("Forget split contains samples outside forget_class.")

    if len(retain_idx) > 0 and np.any(targets[retain_idx] == int(forget_class)):
        issues.append("Retain split unexpectedly contains forget_class samples.")

    if len(left_out_idx) > 0 and np.any(targets[left_out_idx] == int(forget_class)):
        issues.append("Left-out split unexpectedly contains forget_class samples.")

    return issues


def create_targeted_random_unlearning_split(
    dataset,
    forget_class: int,
    retain_count: int,
    forget_count: int,
    left_out_count: int,
    seed: int = 0,
    save_dir: Optional[str] = None,
):
    """
    Create random unlearning splits with explicit sample counts.

    Sampling protocol:
    - Forget set: random samples from forget_class only.
    - Retain/left-out sets: random samples from pooled non-forget classes (not per-class quotas).
    """
    targets = _extract_targets(dataset)
    n = len(targets)

    forget_candidates = np.where(targets == int(forget_class))[0]
    non_forget_candidates = np.where(targets != int(forget_class))[0]

    if len(forget_candidates) < int(forget_count):
        raise ValueError(
            f"Requested forget_count={forget_count}, but only {len(forget_candidates)} samples exist for class {forget_class}."
        )

    required_non_forget = int(retain_count) + int(left_out_count)
    if len(non_forget_candidates) < required_non_forget:
        raise ValueError(
            "Not enough non-forget samples for requested retain/left_out sizes: "
            f"required={required_non_forget}, available={len(non_forget_candidates)}"
        )

    rng = np.random.default_rng(int(seed))
    forget_idx = rng.choice(forget_candidates, size=int(forget_count), replace=False)
    non_forget_perm = rng.permutation(non_forget_candidates)
    retain_idx = non_forget_perm[: int(retain_count)]
    left_out_idx = non_forget_perm[int(retain_count): int(retain_count) + int(left_out_count)]

    issues = _validate_targeted_split(
        retain_idx=retain_idx,
        forget_idx=forget_idx,
        left_out_idx=left_out_idx,
        targets=targets,
        forget_class=int(forget_class),
        retain_count=int(retain_count),
        forget_count=int(forget_count),
        left_out_count=int(left_out_count),
    )
    if issues:
        raise ValueError("Invalid targeted split generated:\n - " + "\n - ".join(issues))

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        forget_path, retain_path, left_out_path, meta_path = _targeted_split_file_paths(save_dir)
        np.save(forget_path, forget_idx)
        np.save(retain_path, retain_idx)
        np.save(left_out_path, left_out_idx)

        meta_payload: Dict[str, int] = {
            "version": 1,
            "dataset_size": int(n),
            "seed": int(seed),
            "forget_class": int(forget_class),
            "retain_count": int(retain_count),
            "forget_count": int(forget_count),
            "left_out_count": int(left_out_count),
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_payload, f, indent=2)

    retain_set = Subset(dataset, retain_idx.tolist())
    forget_set = Subset(dataset, forget_idx.tolist())
    left_out_set = Subset(dataset, left_out_idx.tolist())
    return retain_set, forget_set, left_out_set


def ensure_targeted_random_unlearning_split(
    dataset,
    split_dir: str,
    forget_class: int,
    retain_count: int,
    forget_count: int,
    left_out_count: int,
    seed: int,
    verbose: bool = True,
):
    """
    Load targeted random splits if valid, otherwise recreate and save them.

    Returns:
        retain_set, forget_set, left_out_set, recreated
    """
    forget_path, retain_path, left_out_path, meta_path = _targeted_split_file_paths(split_dir)
    has_files = all(os.path.exists(p) for p in [forget_path, retain_path, left_out_path])

    targets = _extract_targets(dataset)

    if has_files:
        try:
            forget_idx = np.load(forget_path)
            retain_idx = np.load(retain_path)
            left_out_idx = np.load(left_out_path)

            issues = _validate_targeted_split(
                retain_idx=retain_idx,
                forget_idx=forget_idx,
                left_out_idx=left_out_idx,
                targets=targets,
                forget_class=int(forget_class),
                retain_count=int(retain_count),
                forget_count=int(forget_count),
                left_out_count=int(left_out_count),
            )

            if os.path.exists(meta_path):
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                if int(meta.get("dataset_size", -1)) != len(dataset):
                    issues.append("targeted split metadata dataset_size mismatch.")
                if int(meta.get("seed", -1)) != int(seed):
                    issues.append("targeted split metadata seed mismatch.")
                if int(meta.get("forget_class", -1)) != int(forget_class):
                    issues.append("targeted split metadata forget_class mismatch.")
                if int(meta.get("retain_count", -1)) != int(retain_count):
                    issues.append("targeted split metadata retain_count mismatch.")
                if int(meta.get("forget_count", -1)) != int(forget_count):
                    issues.append("targeted split metadata forget_count mismatch.")
                if int(meta.get("left_out_count", -1)) != int(left_out_count):
                    issues.append("targeted split metadata left_out_count mismatch.")
            else:
                issues.append("targeted_split_meta.json is missing.")

            if issues:
                raise ValueError("\n - ".join(issues))

            if verbose:
                print("Loading targeted random retain/forget/left-out splits from disk (validated)...")
            return (
                Subset(dataset, retain_idx.tolist()),
                Subset(dataset, forget_idx.tolist()),
                Subset(dataset, left_out_idx.tolist()),
                False,
            )
        except Exception as ex:
            if verbose:
                print(f"Existing targeted split failed validation, recreating split: {ex}")

    if verbose:
        print("Creating targeted random retain/forget/left-out splits...")
    retain_set, forget_set, left_out_set = create_targeted_random_unlearning_split(
        dataset=dataset,
        forget_class=int(forget_class),
        retain_count=int(retain_count),
        forget_count=int(forget_count),
        left_out_count=int(left_out_count),
        seed=int(seed),
        save_dir=split_dir,
    )
    return retain_set, forget_set, left_out_set, True


# ── Fully-random split (no class constraint on forget set) ──────────────────

def _fully_random_split_file_paths(split_dir: str):
    forget_path = os.path.join(split_dir, "fully_random_forget_idx.npy")
    retain_path = os.path.join(split_dir, "fully_random_retain_idx.npy")
    left_out_path = os.path.join(split_dir, "fully_random_left_out_idx.npy")
    meta_path = os.path.join(split_dir, "fully_random_split_meta.json")
    return forget_path, retain_path, left_out_path, meta_path


def _validate_fully_random_split(
    retain_idx,
    forget_idx,
    left_out_idx,
    retain_count: int,
    forget_count: int,
    left_out_count: int,
) -> List[str]:
    issues: List[str] = []
    retain_idx = np.asarray(retain_idx, dtype=int)
    forget_idx = np.asarray(forget_idx, dtype=int)
    left_out_idx = np.asarray(left_out_idx, dtype=int)

    if len(retain_idx) != int(retain_count):
        issues.append(f"retain size mismatch: expected={retain_count}, got={len(retain_idx)}")
    if len(forget_idx) != int(forget_count):
        issues.append(f"forget size mismatch: expected={forget_count}, got={len(forget_idx)}")
    if len(left_out_idx) != int(left_out_count):
        issues.append(f"left_out size mismatch: expected={left_out_count}, got={len(left_out_idx)}")

    if len(np.unique(retain_idx)) != len(retain_idx):
        issues.append("Duplicate indices in retain split.")
    if len(np.unique(forget_idx)) != len(forget_idx):
        issues.append("Duplicate indices in forget split.")
    if len(np.unique(left_out_idx)) != len(left_out_idx):
        issues.append("Duplicate indices in left_out split.")

    retain_set = set(int(v) for v in retain_idx.tolist())
    forget_set_s = set(int(v) for v in forget_idx.tolist())
    left_out_set_s = set(int(v) for v in left_out_idx.tolist())

    if retain_set.intersection(forget_set_s):
        issues.append("retain/forget overlap detected.")
    if retain_set.intersection(left_out_set_s):
        issues.append("retain/left_out overlap detected.")
    if forget_set_s.intersection(left_out_set_s):
        issues.append("forget/left_out overlap detected.")

    return issues


def create_fully_random_unlearning_split(
    dataset,
    retain_count: int,
    forget_count: int,
    left_out_count: int,
    seed: int = 0,
    save_dir: Optional[str] = None,
):
    """
    Create retain/forget/left-out splits by randomly shuffling the entire dataset.

    No class constraints — forget samples can come from any class.
    Protocol: shuffle all indices, first forget_count -> forget,
    next retain_count -> retain, next left_out_count -> left_out.
    """
    n = len(dataset)
    total_needed = int(retain_count) + int(forget_count) + int(left_out_count)
    if n < total_needed:
        raise ValueError(
            f"Dataset too small for requested split sizes: dataset={n}, "
            f"retain={retain_count}+forget={forget_count}+left_out={left_out_count}={total_needed}"
        )

    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(n)
    forget_idx = perm[: int(forget_count)]
    retain_idx = perm[int(forget_count): int(forget_count) + int(retain_count)]
    left_out_idx = perm[
        int(forget_count) + int(retain_count):
        int(forget_count) + int(retain_count) + int(left_out_count)
    ]

    issues = _validate_fully_random_split(
        retain_idx, forget_idx, left_out_idx, retain_count, forget_count, left_out_count
    )
    if issues:
        raise ValueError("Invalid fully-random split generated:\n - " + "\n - ".join(issues))

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        forget_path, retain_path, left_out_path, meta_path = _fully_random_split_file_paths(save_dir)
        np.save(forget_path, forget_idx)
        np.save(retain_path, retain_idx)
        np.save(left_out_path, left_out_idx)
        meta_payload: Dict[str, int] = {
            "version": 1,
            "dataset_size": int(n),
            "seed": int(seed),
            "retain_count": int(retain_count),
            "forget_count": int(forget_count),
            "left_out_count": int(left_out_count),
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_payload, f, indent=2)

    return (
        Subset(dataset, retain_idx.tolist()),
        Subset(dataset, forget_idx.tolist()),
        Subset(dataset, left_out_idx.tolist()),
    )


def ensure_fully_random_unlearning_split(
    dataset,
    split_dir: str,
    retain_count: int,
    forget_count: int,
    left_out_count: int,
    seed: int,
    verbose: bool = True,
):
    """
    Load fully-random splits from disk if valid, otherwise recreate and save them.

    Returns:
        retain_set, forget_set, left_out_set, recreated
    """
    forget_path, retain_path, left_out_path, meta_path = _fully_random_split_file_paths(split_dir)
    has_files = all(os.path.exists(p) for p in [forget_path, retain_path, left_out_path])

    if has_files:
        try:
            forget_idx = np.load(forget_path)
            retain_idx = np.load(retain_path)
            left_out_idx = np.load(left_out_path)

            issues = _validate_fully_random_split(
                retain_idx, forget_idx, left_out_idx, retain_count, forget_count, left_out_count
            )

            if os.path.exists(meta_path):
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                if int(meta.get("dataset_size", -1)) != len(dataset):
                    issues.append("fully-random split metadata dataset_size mismatch.")
                if int(meta.get("seed", -1)) != int(seed):
                    issues.append("fully-random split metadata seed mismatch.")
                if int(meta.get("retain_count", -1)) != int(retain_count):
                    issues.append("fully-random split metadata retain_count mismatch.")
                if int(meta.get("forget_count", -1)) != int(forget_count):
                    issues.append("fully-random split metadata forget_count mismatch.")
                if int(meta.get("left_out_count", -1)) != int(left_out_count):
                    issues.append("fully-random split metadata left_out_count mismatch.")
            else:
                issues.append("fully_random_split_meta.json is missing.")

            if issues:
                raise ValueError("\n - ".join(issues))

            if verbose:
                print("Loading fully-random retain/forget/left-out splits from disk (validated)...")
            return (
                Subset(dataset, retain_idx.tolist()),
                Subset(dataset, forget_idx.tolist()),
                Subset(dataset, left_out_idx.tolist()),
                False,
            )
        except Exception as ex:
            if verbose:
                print(f"Existing fully-random split failed validation, recreating: {ex}")

    if verbose:
        print("Creating fully-random retain/forget/left-out splits...")
    retain_set, forget_set, left_out_set = create_fully_random_unlearning_split(
        dataset=dataset,
        retain_count=int(retain_count),
        forget_count=int(forget_count),
        left_out_count=int(left_out_count),
        seed=int(seed),
        save_dir=split_dir,
    )
    return retain_set, forget_set, left_out_set, True
