"""
Selective Synaptic Dampening (SSD) unlearning experiment.
This mirrors the SCRUB experiment layout but uses SSD to dampen weights.
"""

import os
import sys
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset, random_split
import yaml

# Add repo root to path for internal imports
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from data.loaders import load_dataset, get_num_classes
from utils.splits import create_retain_forget_split

# Add SSD src to path for third-party import (after repo utils import to avoid shadowing)
SSD_SRC_DIR = os.path.join(
    REPO_ROOT, "Third_Party_Code", "SSD", "selective-synaptic-dampening", "src"
)
if SSD_SRC_DIR not in sys.path:
    sys.path.insert(0, SSD_SRC_DIR)

import ssd as ssd_file


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
    device: Optional[str] = None


class IndexedDataset(Dataset):
    """Wrap a dataset to return (x, idx, y) tuples."""

    def __init__(self, base: Dataset) -> None:
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        sample = self.base[idx]
        if isinstance(sample, tuple) and len(sample) >= 2:
            x, y = sample[0], sample[1]
        else:
            raise ValueError("Dataset must return at least (x, y).")
        return x, idx, y


def compute_accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Compute accuracy on a given loader."""
    correct = 0
    total = 0
    model.eval()

    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                inputs, _, targets = batch
            else:
                inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    return correct / total if total > 0 else 0.0


def train_validation(
    model: nn.Module,
    train_retain_loader: DataLoader,
    train_forget_loader: DataLoader,
    valid_retain_loader: DataLoader,
    valid_forget_loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Compute train/validation accuracies on retain/forget splits."""
    return {
        "tr_acc": compute_accuracy(model, train_retain_loader, device),
        "tf_acc": compute_accuracy(model, train_forget_loader, device),
        "vr_acc": compute_accuracy(model, valid_retain_loader, device),
        "vf_acc": compute_accuracy(model, valid_forget_loader, device),
    }


def load_model(dataset: str, checkpoint_path: str, device: torch.device) -> nn.Module:
    """Load a pre-trained model for the given dataset."""
    if dataset.lower() != "cifar10":
        raise ValueError(f"Unsupported dataset for SSD experiment: {dataset}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Model checkpoint not found at {checkpoint_path}. "
            "Please run: python scripts/train_resnet18_cifar10.py"
        )

    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, get_num_classes("cifar10"))

    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    return model.to(device)


def ssd(loaders: Dict[str, DataLoader], args: SSDInput):
    """Run SSD unlearning using the provided loaders and args."""
    train_loader = loaders["train_loader"]
    train_forget_loader = loaders["train_forget_loader"]
    train_retain_loader = loaders["train_retain_loader"]
    valid_forget_loader = loaders["valid_forget_loader"]
    valid_retain_loader = loaders["valid_retain_loader"]

    parameters = {
        "lower_bound": 1,
        "exponent": 1,
        "magnitude_diff": None,
        "min_layer": -1,
        "max_layer": -1,
        "forget_threshold": 1,
        "dampening_constant": args.dampening_constant,
        "selection_weighting": args.selection_weighting,
    }

    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    model = load_model(dataset=args.dataset, checkpoint_path=args.model_path, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    pdr = ssd_file.ParameterPerturber(model, optimizer, device, parameters)

    model.eval()

    sample_importances = pdr.calc_importance(train_forget_loader)
    original_importances = pdr.calc_importance(train_loader)

    pdr.modify_weight(original_importances, sample_importances)

    acc_dict = train_validation(
        model,
        train_retain_loader,
        train_forget_loader,
        valid_retain_loader,
        valid_forget_loader,
        device,
    )

    if args.check_path is not None:
        check_dir = os.path.dirname(args.check_path)
        if check_dir:
            os.makedirs(check_dir, exist_ok=True)
        torch.save(model.state_dict(), args.check_path)

    return model, acc_dict


def _wrap_dataset(dataset: Dataset) -> Dataset:
    return IndexedDataset(dataset)


def _create_loaders(args: SSDInput):
    dataset = load_dataset(dataset_name=args.dataset, root=args.dataroot, train=True)

    retain_set, forget_set = create_retain_forget_split(
        dataset,
        forget_fraction=args.forget_fraction,
        seed=args.seed,
        save_dir="./data/splits",
    )

    retain_len = len(retain_set)
    forget_len = len(forget_set)

    retain_train_len = int(0.9 * retain_len)
    forget_train_len = int(0.9 * forget_len)

    retain_train, retain_val = random_split(
        retain_set, [retain_train_len, retain_len - retain_train_len]
    )
    forget_train, forget_val = random_split(
        forget_set, [forget_train_len, forget_len - forget_train_len]
    )

    pin_memory = args.pin_memory and torch.cuda.is_available()

    loaders = {
        "train_loader": DataLoader(
            _wrap_dataset(dataset),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        ),
        "train_retain_loader": DataLoader(
            _wrap_dataset(retain_train),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        ),
        "train_forget_loader": DataLoader(
            _wrap_dataset(forget_train),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        ),
        "valid_retain_loader": DataLoader(
            _wrap_dataset(retain_val),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        ),
        "valid_forget_loader": DataLoader(
            _wrap_dataset(forget_val),
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
    import argparse

    parser = argparse.ArgumentParser(description="Run SSD unlearning experiment")
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/ssd_experiment.yaml",
        help="Path to YAML config file",
    )

    cli_args = parser.parse_args()
    args = _load_config(cli_args.config)

    print("Loading dataset and creating splits...")
    loaders = _create_loaders(args)

    print("Running SSD unlearning...")
    model, acc_dict = ssd(loaders, args)

    if args.print_accuracies:
        print(f"   tr_acc: {acc_dict['tr_acc']:.4f}")
        print(f"   tf_acc: {acc_dict['tf_acc']:.4f}")
        print(f"   vr_acc: {acc_dict['vr_acc']:.4f}")
        print(f"   vf_acc: {acc_dict['vf_acc']:.4f}")

    return model, acc_dict


if __name__ == "__main__":
    main()
