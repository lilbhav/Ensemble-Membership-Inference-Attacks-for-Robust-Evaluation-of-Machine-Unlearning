"""
Selective Synaptic Dampening (SSD) unlearning experiment.
This wrapper keeps split management in-framework and delegates unlearning to Third_Party_Code strategy.
"""

import os
import sys
import argparse
from dataclasses import dataclass, replace
from typing import Dict, Optional

import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
import yaml

# Add repo root and third-party package roots.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

TP_MACHINEUNLEARNING_ROOT = os.path.join(REPO_ROOT, "Third_Party_Code", "MachineUnlearning")
if TP_MACHINEUNLEARNING_ROOT not in sys.path:
    sys.path.insert(0, TP_MACHINEUNLEARNING_ROOT)

from data.loaders import load_dataset, get_num_classes
from utils.splits import ensure_retain_forget_split, ensure_targeted_random_unlearning_split, ensure_fully_random_unlearning_split
from utils.metrics import compute_accuracy, log_accuracies
from utils.transfer_setup import ensure_cifar10_from_cifar100_transfer_checkpoint

# Third-party strategy import (delegate algorithm implementation here)
try:
    from Third_Party_Code.MachineUnlearning.unlearn_strategies import strategies as third_party_strategies
except ModuleNotFoundError:
    from unlearn_strategies import strategies as third_party_strategies  # type: ignore[import-not-found]


@dataclass
class SSDInput:
    dataset: str
    dataroot: str
    forget_fraction: float
    seed: int
    batch_size: int
    num_workers: int
    pin_memory: bool
    model_path: str
    check_path: Optional[str]
    learning_rate: float
    dampening_constant: float
    selection_weighting: float
    eval_every: int
    print_accuracies: bool
    split_dir: str = "./data/splits"
    device: Optional[str] = None
    results_path: Optional[str] = None
    lower_bound: float = 1.0
    exponent: float = 1.0
    forget_threshold: float = 1.0
    min_layer: int = -1
    max_layer: int = -1
    forget_class: int = 0
    retain_count: Optional[int] = None
    forget_count: Optional[int] = None
    left_out_count: Optional[int] = None
    retain_per_class: Optional[int] = None
    left_out_per_class: Optional[int] = None
    source_checkpoint_cifar100: Optional[str] = None
    transfer_finetune_epochs: int = 10
    transfer_finetune_batch_size: int = 128
    transfer_finetune_learning_rate: float = 0.001
    rebuild_transfer_checkpoint: bool = False
    split_protocol: str = "fully_random"
    unlearn_epochs: int = 1


def train_validation(
    model: nn.Module,
    train_retain_loader: DataLoader,
    train_forget_loader: DataLoader,
    valid_retain_loader: DataLoader,
    valid_forget_loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    return {
        "tr_acc": compute_accuracy(model, train_retain_loader, device),
        "tf_acc": compute_accuracy(model, train_forget_loader, device),
        "vr_acc": compute_accuracy(model, valid_retain_loader, device),
        "vf_acc": compute_accuracy(model, valid_forget_loader, device),
    }


def load_model(dataset: str, checkpoint_path: str, device: torch.device) -> nn.Module:
    if dataset.lower() != "cifar10":
        raise ValueError(f"Unsupported dataset for SSD experiment: {dataset}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Model checkpoint not found at {checkpoint_path}. "
            "Please provide model_path or configure source_checkpoint_cifar100 transfer setup."
        )

    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, get_num_classes("cifar10"))

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return model.to(device)


def _infer_forget_class(forget_dataset) -> int:
    label_counts = {}
    for sample in forget_dataset:
        if isinstance(sample, tuple) and len(sample) >= 2:
            y = int(sample[1])
            label_counts[y] = label_counts.get(y, 0) + 1
    if not label_counts:
        return 0
    return max(label_counts.items(), key=lambda kv: kv[1])[0]


def ssd(loaders: Dict[str, DataLoader], args: SSDInput):
    train_loader = loaders["train_loader"]
    train_forget_loader = loaders["train_forget_loader"]
    train_retain_loader = loaders["train_retain_loader"]
    valid_forget_loader = loaders["valid_forget_loader"]
    valid_retain_loader = loaders["valid_retain_loader"]
    test_loader = loaders["test_loader"]

    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    model = load_model(dataset=args.dataset, checkpoint_path=args.model_path, device=device)

    baseline_acc = train_validation(
        model,
        train_retain_loader,
        train_forget_loader,
        valid_retain_loader,
        valid_forget_loader,
        device,
    )
    baseline_test_acc = compute_accuracy(model, test_loader, device)
    print(
        "Baseline retain acc - train: {:.4f}, valid: {:.4f}".format(
            baseline_acc["tr_acc"], baseline_acc["vr_acc"]
        )
    )
    print(f"Baseline test acc: {baseline_test_acc:.4f}")
    if args.print_accuracies:
        line = log_accuracies(args.results_path, "baseline", baseline_acc)
        print(f"   {line}")
        print(f"   baseline | test_acc: {baseline_test_acc:.4f}")

    forget_class = int(getattr(args, "forget_class", _infer_forget_class(train_forget_loader.dataset)))
    num_classes = int(get_num_classes(args.dataset))
    num_channels = int(next(iter(train_retain_loader))[0].shape[1])
    strategy_args = argparse.Namespace()

    runs = int(getattr(args, "unlearn_epochs", 1))
    for epoch in range(1, runs + 1):
        model = third_party_strategies.ssd(
            args=strategy_args,
            model=model,
            unlearning_teacher=model,
            unlearn_class=forget_class,
            unlearn_loader=train_forget_loader,
            retain_loader=train_retain_loader,
            test_loader=test_loader,
            num_classes=num_classes,
            num_channels=num_channels,
            device=device,
        )

        acc_epoch = train_validation(
            model,
            train_retain_loader,
            train_forget_loader,
            valid_retain_loader,
            valid_forget_loader,
            device,
        )
        print(
            "[SSD third-party {}/{}] tr={:.4f} tf={:.4f} vr={:.4f} vf={:.4f}".format(
                epoch,
                runs,
                acc_epoch["tr_acc"],
                acc_epoch["tf_acc"],
                acc_epoch["vr_acc"],
                acc_epoch["vf_acc"],
            )
        )
        if args.print_accuracies:
            line = log_accuracies(args.results_path, f"epoch {epoch}", acc_epoch)
            print(f"   {line}")

    acc_dict = train_validation(
        model,
        train_retain_loader,
        train_forget_loader,
        valid_retain_loader,
        valid_forget_loader,
        device,
    )
    after_test_acc = compute_accuracy(model, test_loader, device)
    acc_dict["test_acc"] = after_test_acc

    if args.print_accuracies:
        line = log_accuracies(args.results_path, "after_ssd", acc_dict)
        print(f"   {line}")
        print(f"   after_ssd | test_acc: {after_test_acc:.4f}")

    if args.check_path is not None:
        check_dir = os.path.dirname(args.check_path)
        if check_dir:
            os.makedirs(check_dir, exist_ok=True)
        torch.save(model.state_dict(), args.check_path)

    return model, acc_dict


def _create_loaders(args: SSDInput):
    dataset = load_dataset(dataset_name=args.dataset, root=args.dataroot, train=True)
    test_dataset = load_dataset(dataset_name=args.dataset, root=args.dataroot, train=False)

    split_dir = args.split_dir
    split_protocol = str(args.split_protocol).strip().lower()
    has_count_keys = args.forget_count is not None and (
        args.retain_count is not None or args.retain_per_class is not None
    ) and (
        args.left_out_count is not None or args.left_out_per_class is not None
    )

    if split_protocol == "fully_random" and has_count_keys:
        retain_count = int(
            args.retain_count
            if args.retain_count is not None
            else int(args.retain_per_class) * (int(get_num_classes(args.dataset)) - 1)
        )
        left_out_count = int(
            args.left_out_count
            if args.left_out_count is not None
            else int(args.left_out_per_class) * (int(get_num_classes(args.dataset)) - 1)
        )
        retain_set, forget_set, left_out_set, _ = ensure_fully_random_unlearning_split(
            dataset=dataset,
            split_dir=split_dir,
            retain_count=retain_count,
            forget_count=int(args.forget_count),
            left_out_count=left_out_count,
            seed=int(args.seed),
            verbose=True,
        )
        print(
            "Using fully-random protocol (forget from any class): "
            f"retain={len(retain_set)}, forget={len(forget_set)}, left_out={len(left_out_set)}"
        )
    elif has_count_keys:
        classes_excluding_forget = int(get_num_classes(args.dataset)) - 1
        retain_count = int(
            args.retain_count
            if args.retain_count is not None
            else int(args.retain_per_class) * classes_excluding_forget
        )
        left_out_count = int(
            args.left_out_count
            if args.left_out_count is not None
            else int(args.left_out_per_class) * classes_excluding_forget
        )
        retain_set, forget_set, left_out_set, _ = ensure_targeted_random_unlearning_split(
            dataset=dataset,
            split_dir=split_dir,
            forget_class=int(args.forget_class),
            retain_count=retain_count,
            forget_count=int(args.forget_count),
            left_out_count=left_out_count,
            seed=int(args.seed),
            verbose=True,
        )
        print(
            "Using targeted-random protocol (not per-class quotas): "
            f"retain={len(retain_set)}, forget={len(forget_set)}, left_out={len(left_out_set)}"
        )
    else:
        retain_set, forget_set, _ = ensure_retain_forget_split(
            dataset,
            split_dir=split_dir,
            forget_fraction=args.forget_fraction,
            seed=args.seed,
            verbose=True,
        )
        split_gen = torch.Generator().manual_seed(int(args.seed))
        retain_train_len = int(0.9 * len(retain_set))
        retain_set, left_out_set = torch.utils.data.random_split(
            retain_set,
            [retain_train_len, len(retain_set) - retain_train_len],
            generator=split_gen,
        )
        print(
            "Using fallback random protocol: "
            f"retain={len(retain_set)}, forget={len(forget_set)}, left_out={len(left_out_set)}"
        )

    pin_memory = args.pin_memory and torch.cuda.is_available()

    loaders = {
        "train_loader": DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        ),
        "train_retain_loader": DataLoader(
            retain_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        ),
        "train_forget_loader": DataLoader(
            forget_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        ),
        "valid_retain_loader": DataLoader(
            left_out_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        ),
        "valid_forget_loader": DataLoader(
            forget_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        ),
        "test_loader": DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        ),
    }

    return loaders


def _load_config(config_path: str) -> SSDInput:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f) or {}

    return SSDInput(**config_dict)


def main():
    parser = argparse.ArgumentParser(description="Run SSD unlearning experiment")
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/ssd_experiment.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--dampening-constant",
        type=float,
        default=None,
        help="Compatibility override; third-party strategy currently uses its own defaults.",
    )
    parser.add_argument(
        "--selection-weighting",
        type=float,
        default=None,
        help="Compatibility override; third-party strategy currently uses its own defaults.",
    )
    parser.add_argument(
        "--lower-bound",
        type=float,
        default=None,
        help="Compatibility override; third-party strategy currently uses its own defaults.",
    )
    parser.add_argument(
        "--exponent",
        type=float,
        default=None,
        help="Compatibility override; third-party strategy currently uses its own defaults.",
    )

    cli_args = parser.parse_args()
    args = _load_config(cli_args.config)

    overrides = {}
    if cli_args.dampening_constant is not None:
        overrides["dampening_constant"] = cli_args.dampening_constant
    if cli_args.selection_weighting is not None:
        overrides["selection_weighting"] = cli_args.selection_weighting
    if cli_args.lower_bound is not None:
        overrides["lower_bound"] = cli_args.lower_bound
    if cli_args.exponent is not None:
        overrides["exponent"] = cli_args.exponent
    if overrides:
        args = replace(args, **overrides)

    if args.source_checkpoint_cifar100:
        target_model_path = ensure_cifar10_from_cifar100_transfer_checkpoint(
            source_checkpoint_cifar100=args.source_checkpoint_cifar100,
            target_checkpoint_cifar10=args.model_path,
            dataroot=args.dataroot,
            finetune_epochs=int(args.transfer_finetune_epochs),
            finetune_batch_size=int(args.transfer_finetune_batch_size),
            finetune_learning_rate=float(args.transfer_finetune_learning_rate),
            seed=int(args.seed),
            num_workers=int(args.num_workers),
            pin_memory=bool(args.pin_memory),
            force_rebuild=bool(args.rebuild_transfer_checkpoint),
        )
        if target_model_path != args.model_path:
            args = replace(args, model_path=target_model_path)

    print("Loading dataset and creating splits...")
    loaders = _create_loaders(args)

    print("Running SSD unlearning...")
    model, acc_dict = ssd(loaders, args)

    if args.print_accuracies:
        print(f"   tr_acc: {acc_dict['tr_acc']:.4f}")
        print(f"   tf_acc: {acc_dict['tf_acc']:.4f}")
        print(f"   vr_acc: {acc_dict['vr_acc']:.4f}")
        print(f"   vf_acc: {acc_dict['vf_acc']:.4f}")
        if "test_acc" in acc_dict:
            print(f"   test_acc: {acc_dict['test_acc']:.4f}")

    return model, acc_dict


if __name__ == "__main__":
    main()
