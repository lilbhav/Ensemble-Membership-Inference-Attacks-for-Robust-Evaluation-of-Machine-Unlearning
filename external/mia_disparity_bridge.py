from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset


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
        "num_shadow_models": 10,
        "shadow_diff_init": False,
        "shadow_path": str(Path(args.output_csv).parent / "lira_shadows"),
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

    # Load split indices produced by prepare_splits.py
    split_data = np.load(args.split_file)
    retain_indices = split_data["retain_indices"].tolist()
    forget_indices = split_data["forget_indices"].tolist()
    test_indices = split_data["test_indices"].tolist()
    aux_indices = split_data["aux_indices"].tolist() if "aux_indices" in split_data.files else []

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

    # Prepare attack-specific artifacts (e.g., shadow models) and run inference
    target_model_access = get_target_model_access(attack_name, model, untrained_model)
    aux_info = get_aux_info(attack_name, device, num_classes, args)
    attack = get_attack(attack_name, aux_info, target_model_access)

    attack.prepare(aux_ds)
    pred_scores = np.asarray(attack.infer(target_ds), dtype=float)

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
                "model_name": args.model_name,
                "unlearning_method": args.unlearning_method,
                "attack_name": attack_name,
                "attack_seed": args.attack_seed,
                "score": float(score),
                "prediction": int(score >= 0.5),
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
        "model_name",
        "unlearning_method",
        "attack_name",
        "attack_seed",
        "score",
        "prediction",
        "dataset",
        "base_seed",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
