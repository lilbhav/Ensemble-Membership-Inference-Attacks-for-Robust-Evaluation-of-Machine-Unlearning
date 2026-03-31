from __future__ import annotations

import argparse
import csv
import json
import sys
from itertools import accumulate
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Subset


def safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    from sklearn.metrics import roc_auc_score  # type: ignore

    return float(roc_auc_score(labels, scores))


def orient_scores_for_membership(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, str, float]:
    auc_original = safe_auc(labels, scores)
    if np.isnan(auc_original):
        return scores, "original", auc_original
    if auc_original < 0.5:
        scores = -scores
        return scores, "negated", safe_auc(labels, scores)
    return scores, "original", auc_original


def compute_threshold_at_target_fpr(labels: np.ndarray, scores: np.ndarray, target_fpr: float) -> float:
    from sklearn.metrics import roc_curve  # type: ignore

    fpr_arr, tpr_arr, thresholds = roc_curve(labels, scores)
    eligible = [
        (float(threshold), float(tpr), float(fpr))
        for fpr, tpr, threshold in zip(fpr_arr, tpr_arr, thresholds)
        if fpr <= target_fpr
    ]
    if eligible:
        threshold, _, _ = max(eligible, key=lambda item: (item[1], -item[2], item[0]))
        return threshold
    return float(np.max(scores) + 1e-12)


def confusion_rates(labels: np.ndarray, predictions: np.ndarray) -> tuple[float, float]:
    member_mask = labels == 1
    nonmember_mask = labels == 0
    tpr = float((predictions[member_mask] == 1).sum() / max(1, int(member_mask.sum())))
    fpr = float((predictions[nonmember_mask] == 1).sum() / max(1, int(nonmember_mask.sum())))
    return tpr, fpr


def configure_torch_pickle_compat() -> None:
    """Make torch.load backward-compatible with pickled non-tensor objects.

    PyTorch 2.6 changed torch.load default to weights_only=True, which breaks
    mia-disparity cached artifact loading. Older mia-disparity code also
    expects torch._utils._accumulate, which was removed in newer PyTorch.
    We restore both behaviors for this trusted local workflow.
    """
    if getattr(torch.load, "_miae_compat_patched", False):
        return

    original_torch_load = torch.load

    def _torch_load_compat(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    if not hasattr(torch._utils, "_accumulate"):
        torch._utils._accumulate = accumulate

    _torch_load_compat._miae_compat_patched = True
    torch.load = _torch_load_compat


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bridge to mia-disparity attacks")
    p.add_argument("--dataset", required=True, choices=["Cifar10", "Cifar100"])
    p.add_argument("--base-seed", type=int, required=True)
    p.add_argument("--attack-name", required=True)
    p.add_argument("--attack-seed", type=int, required=True)
    p.add_argument("--target-name", required=True, choices=["forget_vs_test", "retain_vs_test", "forget_vs_retain"])
    p.add_argument("--split-file", required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--output-csv", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--engine-repo", required=True)
    p.add_argument("--attack-epochs", type=int, required=True)
    p.add_argument("--batch-size", type=int, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--unlearning-method", required=True)
    p.add_argument("--model-name", required=True)
    p.add_argument("--num-shadow-models", type=int, default=10)
    p.add_argument("--target-fpr", type=float, default=0.01)
    return p.parse_args()


def normalize_attack_name(attack_name: str) -> str:
    # Accept common alias names and normalize to internal implementation names
    aliases = {
        "loss": "yeom",
        "yeom/loss": "yeom",
        "class-nn": "shokri",
        "shokri/class-nn": "shokri",
        "loss_trajectory": "losstraj",
        "loss-trajectory": "losstraj",
    }
    return aliases.get(attack_name, attack_name)


def pick_target_subsets(target_name: str, retain_ds, forget_ds, test_ds):
    # Choose which subset is treated as member vs non-member for this attack target
    if target_name == "forget_vs_test":
        return forget_ds, test_ds, "forget", "test"
    if target_name == "retain_vs_test":
        return retain_ds, test_ds, "retain", "test"
    if target_name == "forget_vs_retain":
        return forget_ds, retain_ds, "forget", "retain"
    raise ValueError(f"Unknown target: {target_name}")


def get_target_model_access(attack, target_model, untrained_target_model):
    # Build attack-specific model-access wrapper expected by mia-disparity APIs
    from miae.attacks import losstraj_mia, shokri_mia, lira_mia, yeom_mia, aug_mia, calibration_mia, reference_mia  # type: ignore

    if attack == "losstraj":
        return losstraj_mia.LosstrajModelAccess(target_model, untrained_target_model)
    if attack == "yeom":
        return yeom_mia.YeomModelAccess(target_model, untrained_target_model)
    if attack == "shokri":
        return shokri_mia.ShokriModelAccess(target_model, untrained_target_model)
    if attack in {"lira", "lira_offline"}:
        return lira_mia.LiraModelAccess(target_model, untrained_target_model)
    if attack == "reference":
        return reference_mia.ReferenceModelAccess(target_model, untrained_target_model)
    if attack == "aug":
        return aug_mia.AugModelAccess(target_model, untrained_target_model)
    if attack == "calibration":
        return calibration_mia.CalibrationModelAccess(target_model, untrained_target_model)
    raise ValueError(f"Unsupported attack: {attack}")


def get_aux_info(attack, device, num_classes, args):
    # Assemble shared attack hyperparameters and paths for preparation artifacts
    from miae.attacks import losstraj_mia, shokri_mia, lira_mia, yeom_mia, aug_mia, calibration_mia, reference_mia  # type: ignore

    common = {
        "device": device,
        "seed": args.attack_seed,
        "shadow_seed_base": args.attack_seed,
        "save_path": str(Path(args.output_csv).parent / "prep"),
        "num_classes": num_classes,
        "batch_size": args.batch_size,
        "lr": 0.1,
        "epochs": args.attack_epochs,
        "attack_epochs": args.attack_epochs,
        "log_path": str(Path(args.output_csv).parent),
        "num_shadow_models": args.num_shadow_models,
        "shadow_diff_init": False,
        # Each evaluation target (forget_vs_test, retain_vs_test, forget_vs_retain) constructs a
        # different shadow_target_concat_set with a different length, so keep.npy sizes differ.
        # Using a target-scoped shadow_path avoids cross-target cache collisions that cause
        # IndexError when keep.npy size mismatches the current concat dataset size.
        "shadow_path": str(Path(args.output_csv).parent / f"lira_shadows_{Path(args.output_csv).stem}"),
        "augmentation_query": 18,
    }

    if attack == "losstraj":
        common["distillation_epochs"] = args.attack_epochs
        return losstraj_mia.LosstrajAuxiliaryInfo(common)
    if attack == "yeom":
        return yeom_mia.YeomAuxiliaryInfo(common)
    if attack == "shokri":
        return shokri_mia.ShokriAuxiliaryInfo(common)
    if attack in {"lira", "lira_offline"}:
        common["online"] = attack == "lira"
        return lira_mia.LiraAuxiliaryInfo(common)
    if attack == "reference":
        return reference_mia.ReferenceAuxiliaryInfo(common)
    if attack == "aug":
        return aug_mia.AugAuxiliaryInfo(common)
    if attack == "calibration":
        common["num_shadow_models"] = 1
        return calibration_mia.CalibrationAuxiliaryInfo(common)
    raise ValueError(f"Unsupported attack: {attack}")


def get_attack(attack, aux_info, target_model_access):
    # Instantiate the concrete attack object selected by CLI/config
    from miae.attacks import losstraj_mia, shokri_mia, lira_mia, yeom_mia, aug_mia, calibration_mia, reference_mia  # type: ignore

    if attack == "losstraj":
        return losstraj_mia.LosstrajAttack(target_model_access, aux_info)
    if attack == "yeom":
        return yeom_mia.YeomAttack(target_model_access, aux_info)
    if attack == "shokri":
        return shokri_mia.ShokriAttack(target_model_access, aux_info)
    if attack in {"lira", "lira_offline"}:
        return lira_mia.LiraAttack(target_model_access, aux_info)
    if attack == "reference":
        return reference_mia.ReferenceAttack(target_model_access, aux_info)
    if attack == "aug":
        return aug_mia.AugAttack(target_model_access, aux_info)
    if attack == "calibration":
        return calibration_mia.CalibrationAttack(target_model_access, aux_info)
    raise ValueError(f"Unsupported attack: {attack}")


def ensure_initialize_weights(model: nn.Module) -> None:
    """Attach initialize_weights() when the model class does not define it.

    Some mia-disparity attacks (e.g., losstraj) expect this method to exist.
    """
    model_cls = model.__class__
    if hasattr(model_cls, "initialize_weights"):
        return

    def _initialize_weights(self: nn.Module) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    model_cls.initialize_weights = _initialize_weights


def main() -> None:
    args = parse_args()

    # Normalize attack alias names so config can use human-friendly labels
    attack_name = normalize_attack_name(args.attack_name)

    engine_repo = Path(args.engine_repo).resolve()
    if not engine_repo.exists():
        raise FileNotFoundError(f"mia-disparity repo not found: {engine_repo}")
    if not (engine_repo / "miae" / "__init__.py").exists():
        raise FileNotFoundError(
            f"mia-disparity repo is missing expected package layout at: {engine_repo / 'miae'}"
        )

    sys.path.insert(0, str(engine_repo))
    configure_torch_pickle_compat()

    # Load split indices produced by prepare_splits.py
    split_data = np.load(args.split_file)
    retain_indices = split_data["retain_indices"].tolist()
    forget_indices = split_data["forget_indices"].tolist()
    test_indices = split_data["test_indices"].tolist()
    aux_indices = split_data["aux_indices"].tolist() if "aux_indices" in split_data.files else []

    split_path = Path(args.split_file)
    split_meta_path = split_path.with_suffix(".meta.json")
    if not split_meta_path.exists():
        raise FileNotFoundError(f"Missing split metadata file: {split_meta_path}")

    with split_meta_path.open("r", encoding="utf-8") as f:
        split_meta = json.load(f)

    if split_meta.get("split_mode") != "targeted_random":
        raise ValueError(
            f"Unsupported split_mode in metadata ({split_meta.get('split_mode')}). "
            "Only targeted_random is supported."
        )
    if "target_class" not in split_meta:
        raise ValueError(f"Split metadata is missing required key 'target_class': {split_meta_path}")

    target_class = int(split_meta["target_class"])
    forget_count = int(split_meta.get("forget_count", len(forget_indices)))
    forget_fraction = split_meta.get("forget_fraction")

    # Load dataset/model definitions from external repos
    mu_repo = engine_repo.parent / "MachineUnlearning"
    if not (mu_repo / "src" / "__init__.py").exists():
        raise FileNotFoundError(
            f"Sibling MachineUnlearning repo is missing expected package layout at: {mu_repo / 'src'}"
        )
    sys.path.insert(0, str(mu_repo))

    from src import dataset as mu_dataset  # type: ignore
    from model import models as mu_models  # type: ignore

    train_dataset, test_dataset, num_classes, num_channels = mu_dataset.get_dataset(
        dataset_name=args.dataset,
        root=args.data_root,
        augment=False,
    )

    retain_ds = Subset(train_dataset, retain_indices)
    forget_ds = Subset(train_dataset, forget_indices)
    test_ds = Subset(test_dataset, test_indices)
    aux_ds = Subset(train_dataset, aux_indices) if len(aux_indices) > 0 else retain_ds

    # Build dataset views for selected target pairing
    member_ds, nonmember_ds, member_split_name, nonmember_split_name = pick_target_subsets(
        args.target_name,
        retain_ds,
        forget_ds,
        test_ds,
    )

    # Concatenate member+nonmember samples so attack returns one score per row
    target_ds = ConcatDataset([member_ds, nonmember_ds])
    target_membership = np.concatenate([np.ones(len(member_ds)), np.zeros(len(nonmember_ds))]).astype(int)

    device = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"

    # Load trained target model and create an untrained reference model when required by attack
    model = getattr(mu_models, "ResNet18")(num_classes=num_classes, input_channels=num_channels)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    model.eval()

    untrained_model = getattr(mu_models, "ResNet18")(num_classes=num_classes, input_channels=num_channels).to(device)
    ensure_initialize_weights(model)
    ensure_initialize_weights(untrained_model)

    # Prepare attack-specific artifacts (e.g., shadow models) and run inference
    target_model_access = get_target_model_access(attack_name, model, untrained_model)
    aux_info = get_aux_info(attack_name, device, num_classes, args)
    attack = get_attack(attack_name, aux_info, target_model_access)

    attack.prepare(aux_ds)
    pred_scores = np.asarray(attack.infer(target_ds), dtype=float)
    pred_scores, score_direction, auc_after_flip = orient_scores_for_membership(pred_scores, target_membership)
    calibrated_threshold = compute_threshold_at_target_fpr(target_membership, pred_scores, args.target_fpr)
    calibrated_predictions = (pred_scores >= calibrated_threshold).astype(int)
    calibrated_tpr, calibrated_fpr = confusion_rates(target_membership, calibrated_predictions)
    coverage_fraction = float(np.mean(calibrated_predictions)) if len(calibrated_predictions) > 0 else 0.0

    train_size = len(train_dataset)

    # Build standardized per-sample rows expected by downstream ensemble scripts
    rows = []
    for i, score in enumerate(pred_scores):
        is_member = int(target_membership[i])
        in_member = i < len(member_ds)
        split_name = member_split_name if in_member else nonmember_split_name

        # Keep test sample IDs disjoint from train sample IDs by offsetting with train_size
        if split_name == "test":
            base_idx = test_indices[i - len(member_ds)] if not in_member else test_indices[i]
            sample_id = train_size + int(base_idx)
        elif split_name == "forget":
            base_idx = forget_indices[i] if in_member else forget_indices[i - len(member_ds)]
            sample_id = int(base_idx)
        else:
            base_idx = retain_indices[i] if in_member else retain_indices[i - len(member_ds)]
            sample_id = int(base_idx)

        rows.append(
            {
                "sample_id": sample_id,
                "true_membership": is_member,
                "split_name": split_name,
                "split_mode": "targeted_random",
                "target_class": target_class,
                "forget_count": forget_count,
                "forget_fraction": forget_fraction,
                "model_name": args.model_name,
                "unlearning_method": args.unlearning_method,
                "attack_name": attack_name,
                "attack_seed": args.attack_seed,
                "score": float(score),
                "prediction": int(calibrated_predictions[i]),
                "calibrated_threshold": float(calibrated_threshold),
                "calibration_target_fpr": float(args.target_fpr),
                "score_direction": score_direction,
                "auc_after_flip": float(auc_after_flip),
                "dataset": args.dataset,
                "base_seed": args.base_seed,
            }
        )

    # Write final normalized CSV format
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "true_membership",
        "split_name",
        "split_mode",
        "target_class",
        "forget_count",
        "forget_fraction",
        "model_name",
        "unlearning_method",
        "attack_name",
        "attack_seed",
        "score",
        "prediction",
        "calibrated_threshold",
        "calibration_target_fpr",
        "score_direction",
        "auc_after_flip",
        "dataset",
        "base_seed",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    calibration_summary = {
        "dataset": args.dataset,
        "base_seed": args.base_seed,
        "unlearning_method": args.unlearning_method,
        "attack_name": attack_name,
        "attack_seed": args.attack_seed,
        "target_name": args.target_name,
        "target_fpr": float(args.target_fpr),
        "calibrated_threshold": float(calibrated_threshold),
        "score_direction": score_direction,
        "auc_after_flip": float(auc_after_flip),
        "calibrated_tpr": float(calibrated_tpr),
        "calibrated_fpr": float(calibrated_fpr),
        "coverage_fraction": float(coverage_fraction),
    }
    calibration_path = output_csv.with_name(f"{output_csv.stem}_calibration.json")
    with calibration_path.open("w", encoding="utf-8") as f:
        json.dump(calibration_summary, f, indent=2)


if __name__ == "__main__":
    main()
