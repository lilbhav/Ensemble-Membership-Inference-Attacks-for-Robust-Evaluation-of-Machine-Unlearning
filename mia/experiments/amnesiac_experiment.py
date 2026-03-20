"""
Amnesiac unlearning experiment.
This follows the SCRUB/SSD experiment structure and supports the same split protocols.
"""

import os
import sys
import random
import argparse
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

TP_MACHINEUNLEARNING_ROOT = os.path.join(REPO_ROOT, "Third_Party_Code", "MachineUnlearning")
if TP_MACHINEUNLEARNING_ROOT not in sys.path:
    sys.path.insert(0, TP_MACHINEUNLEARNING_ROOT)

from data.loaders import load_dataset, get_num_classes
from utils.metrics import compute_accuracy, log_accuracies
from utils.splits import (
    ensure_retain_forget_split,
    ensure_targeted_random_unlearning_split,
    ensure_fully_random_unlearning_split,
)
from utils.transfer_setup import ensure_cifar10_from_cifar100_transfer_checkpoint
try:
    from Third_Party_Code.MachineUnlearning.unlearn_strategies import strategies as third_party_strategies
except ModuleNotFoundError:
    from unlearn_strategies import strategies as third_party_strategies  # type: ignore[import-not-found]


@dataclass
class AmnesiacInput:
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
    unlearn_epochs: int
    eval_every: int
    print_accuracies: bool
    split_dir: str = "./data/splits"
    device: Optional[str] = None
    results_path: Optional[str] = None
    forget_class: Optional[int] = None
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


def _set_global_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _extract_label(sample) -> int:
    if isinstance(sample, tuple) and len(sample) >= 2:
        return int(sample[1])
    raise ValueError("Dataset sample must be (x, y)")


def _load_model(dataset: str, checkpoint_path: str, device: torch.device) -> nn.Module:
    if dataset.lower() != "cifar10":
        raise ValueError(f"Unsupported dataset for amnesiac experiment: {dataset}")

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


def _evaluate_all(model: nn.Module, loaders: Dict[str, DataLoader], device: torch.device) -> Dict[str, float]:
    return {
        "tr_acc": compute_accuracy(model, loaders["train_retain_loader"], device),
        "tf_acc": compute_accuracy(model, loaders["train_forget_loader"], device),
        "vr_acc": compute_accuracy(model, loaders["valid_retain_loader"], device),
        "vf_acc": compute_accuracy(model, loaders["valid_forget_loader"], device),
    }


def _infer_forget_class(forget_dataset) -> int:
    label_counts: Dict[int, int] = {}
    for sample in forget_dataset:
        label = _extract_label(sample)
        label_counts[label] = label_counts.get(label, 0) + 1
    if not label_counts:
        raise ValueError("Forget dataset is empty; cannot infer forget class")
    return max(label_counts.items(), key=lambda kv: kv[1])[0]


def amnesiac(loaders: Dict[str, DataLoader], args: AmnesiacInput):
    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    model = _load_model(args.dataset, args.model_path, device)
    forget_class = int(args.forget_class) if args.forget_class is not None else _infer_forget_class(
        loaders["train_forget_loader"].dataset
    )

    baseline_acc = _evaluate_all(model, loaders, device)
    if args.print_accuracies:
        line = log_accuracies(args.results_path, "baseline", baseline_acc)
        print(f"   {line}")

    strategy_args = argparse.Namespace()
    num_channels = int(next(iter(loaders["train_retain_loader"]))[0].shape[1])
    num_classes = int(get_num_classes(args.dataset))

    epoch_list = []
    tr_accs = []
    tf_accs = []
    vr_accs = []
    vf_accs = []

    for epoch in range(1, int(args.unlearn_epochs) + 1):
        model = third_party_strategies.amnesiac(
            args=strategy_args,
            model=model,
            unlearning_teacher=_load_model(args.dataset, args.model_path, device),
            unlearn_class=forget_class,
            unlearn_loader=loaders["train_forget_loader"],
            retain_loader=loaders["train_retain_loader"],
            test_loader=loaders["test_loader"],
            num_classes=num_classes,
            num_channels=num_channels,
            device=device,
        )

        model.eval()
        acc_dict = _evaluate_all(model, loaders, device)

        epoch_list.append(epoch)
        tr_accs.append(acc_dict["tr_acc"])
        tf_accs.append(acc_dict["tf_acc"])
        vr_accs.append(acc_dict["vr_acc"])
        vf_accs.append(acc_dict["vf_acc"])

        print(
            "[Amnesiac third-party {}/{}] tr={:.4f} tf={:.4f} vr={:.4f} vf={:.4f}".format(
                epoch,
                int(args.unlearn_epochs),
                acc_dict["tr_acc"],
                acc_dict["tf_acc"],
                acc_dict["vr_acc"],
                acc_dict["vf_acc"],
            )
        )

        if args.print_accuracies:
            line = log_accuracies(args.results_path, f"epoch {epoch}", acc_dict)
            print(f"   {line}")

    test_acc = compute_accuracy(model, loaders["test_loader"], device)

    if args.print_accuracies:
        print(f"   final | test_acc: {test_acc:.4f}")

    if args.check_path is not None:
        check_dir = os.path.dirname(args.check_path)
        if check_dir:
            os.makedirs(check_dir, exist_ok=True)
        torch.save(model.state_dict(), args.check_path)

    history = {
        "epoch_list": epoch_list,
        "tr_accs": tr_accs,
        "tf_accs": tf_accs,
        "vr_accs": vr_accs,
        "vf_accs": vf_accs,
        "test_acc": test_acc,
        "forget_class": forget_class,
    }

    return model, history


def _create_loaders(args: AmnesiacInput):
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
            forget_class=int(args.forget_class if args.forget_class is not None else 0),
            retain_count=retain_count,
            forget_count=int(args.forget_count),
            left_out_count=left_out_count,
            seed=int(args.seed),
            verbose=True,
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

    pin_memory = bool(args.pin_memory) and torch.cuda.is_available()

    loaders = {
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


def _load_config(config_path: str) -> AmnesiacInput:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f) or {}

    return AmnesiacInput(**config_dict)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run Amnesiac unlearning experiment")
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/amnesiac_experiment.yaml",
        help="Path to YAML config file",
    )
    cli_args = parser.parse_args()

    args = _load_config(cli_args.config)
    _set_global_determinism(int(args.seed))

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
            args.model_path = target_model_path

    print("Loading dataset and creating splits...")
    loaders = _create_loaders(args)

    print("Running Amnesiac unlearning...")
    model, history = amnesiac(loaders, args)

    print("\nAmnesiac unlearning completed")
    if history["vr_accs"]:
        print(f"Final valid retain acc: {history['vr_accs'][-1]:.4f}")
    if history["vf_accs"]:
        print(f"Final valid forget acc: {history['vf_accs'][-1]:.4f}")
    print(f"Final test acc: {history['test_acc']:.4f}")

    return model, history


if __name__ == "__main__":
    main()
