"""
Selective Synaptic Dampening (SSD) unlearning experiment.
This mirrors the SCRUB experiment layout but uses SSD to dampen weights.
"""

import os
import sys
from dataclasses import dataclass, replace
from typing import Dict, Optional

import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset, Subset
import yaml

# Add repo root to path for internal imports
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from data.loaders import load_dataset, get_num_classes
from utils.metrics import compute_accuracy, log_accuracies
from utils.unlearning_setup import (
    create_classwise_unlearning_splits,
    load_or_create_transfer_model,
)

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
    source_checkpoint_cifar100: Optional[str]
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
    retain_per_class: int = 100
    forget_count: int = 25
    left_out_per_class: int = 25
    transfer_finetune_epochs: int = 10
    transfer_finetune_batch_size: int = 128
    transfer_finetune_learning_rate: float = 0.001
    transfer_finetune_weight_decay: float = 0.0
    transfer_finetune_momentum: float = 0.9


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

    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, get_num_classes("cifar10"))

    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    return model.to(device)


def _infer_layer_id(param_name: str) -> int:
    """Infer a layer id from a parameter name for optional layer-range filtering."""
    for token in param_name.split("."):
        if token.isdigit():
            return int(token)
        if token.startswith("layer") and token[5:].isdigit():
            return int(token[5:])
    return -1


def _apply_ssd_weight_update(
    model: nn.Module,
    original_importances: Dict[str, torch.Tensor],
    forget_importances: Dict[str, torch.Tensor],
    args: SSDInput,
) -> Dict[str, float]:
    """Apply SSD update with threshold and layer-range controls enforced in wrapper."""
    total_params = 0
    selected_params = 0
    touched_tensors = 0

    min_layer = int(args.min_layer)
    max_layer = int(args.max_layer)
    enforce_layer_range = min_layer >= 0 and max_layer >= 0 and max_layer >= min_layer

    with torch.no_grad():
        for name, p in model.named_parameters():
            oimp = original_importances[name]
            fimp = forget_importances[name]

            if enforce_layer_range:
                layer_id = _infer_layer_id(name)
                if layer_id < min_layer or layer_id > max_layer:
                    continue

            total_params += int(p.numel())

            # Selection mask: forget importance must exceed weighted original importance threshold.
            selection_threshold = oimp.mul(args.selection_weighting * args.forget_threshold)
            locations = fimp > selection_threshold
            if not torch.any(locations):
                continue

            touched_tensors += 1
            selected_params += int(locations.sum().item())

            # Dampening factor from SSD equation; clamp denominator for numerical safety.
            denom = torch.clamp(fimp[locations], min=1e-12)
            update = ((oimp[locations].mul(args.dampening_constant)).div(denom)).pow(args.exponent)

            # Bound by lower_bound to prevent parameter magnitudes from increasing.
            update = torch.clamp(update, max=args.lower_bound)
            p[locations] = p[locations].mul(update)

    selected_ratio = (selected_params / total_params) if total_params > 0 else 0.0
    return {
        "selected_params": float(selected_params),
        "total_params": float(total_params),
        "selected_ratio": selected_ratio,
        "touched_tensors": float(touched_tensors),
    }


def ssd(loaders: Dict[str, DataLoader], args: SSDInput):
    """Run SSD unlearning using the provided loaders and args."""
    train_loader = loaders["train_loader"]
    train_forget_loader = loaders["train_forget_loader"]
    train_retain_loader = loaders["train_retain_loader"]
    valid_forget_loader = loaders["valid_forget_loader"]
    valid_retain_loader = loaders["valid_retain_loader"]
    test_loader = loaders["test_loader"]

    parameters = {
        "lower_bound": args.lower_bound,
        "exponent": args.exponent,
        "magnitude_diff": None,
        "min_layer": args.min_layer,
        "max_layer": args.max_layer,
        "forget_threshold": args.forget_threshold,
        "dampening_constant": args.dampening_constant,
        "selection_weighting": args.selection_weighting,
    }

    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    model = load_or_create_transfer_model(
        device=device,
        cifar10_checkpoint_path=args.model_path,
        source_checkpoint_cifar100=args.source_checkpoint_cifar100,
        dataroot=args.dataroot,
        finetune_epochs=int(args.transfer_finetune_epochs),
        finetune_batch_size=int(args.transfer_finetune_batch_size),
        finetune_learning_rate=float(args.transfer_finetune_learning_rate),
        finetune_weight_decay=float(args.transfer_finetune_weight_decay),
        finetune_momentum=float(args.transfer_finetune_momentum),
        num_workers=int(args.num_workers),
        pin_memory=bool(args.pin_memory),
    )
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
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    pdr = ssd_file.ParameterPerturber(model, optimizer, device, parameters)

    model.eval()

    sample_importances = pdr.calc_importance(train_forget_loader)
    original_importances = pdr.calc_importance(train_loader)

    update_stats = _apply_ssd_weight_update(
        model=model,
        original_importances=original_importances,
        forget_importances=sample_importances,
        args=args,
    )
    print(
        "SSD update stats | selected_params: {selected:.0f}/{total:.0f} ({ratio:.4%}), "
        "tensors_touched: {touched:.0f}".format(
            selected=update_stats["selected_params"],
            total=update_stats["total_params"],
            ratio=update_stats["selected_ratio"],
            touched=update_stats["touched_tensors"],
        )
    )

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


def _wrap_dataset(dataset: Dataset) -> Dataset:
    return IndexedDataset(dataset)


def _create_loaders(args: SSDInput):
    dataset = load_dataset(dataset_name=args.dataset, root=args.dataroot, train=True)
    test_dataset = load_dataset(dataset_name=args.dataset, root=args.dataroot, train=False)

    retain_set, forget_set, left_out_set, split_info = create_classwise_unlearning_splits(
        dataset=dataset,
        forget_class=int(args.forget_class),
        retain_per_class=int(args.retain_per_class),
        forget_count=int(args.forget_count),
        left_out_per_class=int(args.left_out_per_class),
        seed=int(args.seed),
    )

    train_indices = split_info["retain_indices"] + split_info["forget_indices"]
    train_subset = Subset(dataset, train_indices)

    retain_train = retain_set
    forget_train = forget_set
    retain_val = left_out_set
    forget_val = forget_set

    print(
        "Classwise protocol split sizes | retain: {retain}, forget: {forget}, left_out: {left}".format(
            retain=len(retain_train),
            forget=len(forget_train),
            left=len(retain_val),
        )
    )

    pin_memory = args.pin_memory and torch.cuda.is_available()

    loaders = {
        "train_loader": DataLoader(
            _wrap_dataset(train_subset),
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
        "test_loader": DataLoader(
            _wrap_dataset(test_dataset),
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
    parser.add_argument(
        "--dampening-constant",
        type=float,
        default=None,
        help="Override dampening_constant from config",
    )
    parser.add_argument(
        "--selection-weighting",
        type=float,
        default=None,
        help="Override selection_weighting from config",
    )
    parser.add_argument(
        "--lower-bound",
        type=float,
        default=None,
        help="Override lower_bound from config",
    )
    parser.add_argument(
        "--exponent",
        type=float,
        default=None,
        help="Override exponent from config",
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
