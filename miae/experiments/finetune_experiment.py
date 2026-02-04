"""
Fine-tuning baseline unlearning using the retain set.
This script mirrors the SCRUB experiment layout but uses simple fine-tuning.
"""

import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader, random_split

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data.loaders import load_dataset, get_num_classes
from utils.splits import create_retain_forget_split


class AverageMeter:
    """Track current and average values."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0


def accuracy(output: torch.Tensor, target: torch.Tensor, topk: Tuple[int, ...] = (1,)):
    """Compute top-k accuracy for the specified values of k."""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def l1_regularization(model: nn.Module) -> torch.Tensor:
    """Compute L1 norm over all parameters."""
    l1_norm = torch.tensor(0.0, device=next(model.parameters()).device)
    for param in model.parameters():
        l1_norm = l1_norm + param.abs().sum()
    return l1_norm


@dataclass
class FineTuneInput:
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
    print_freq: int
    with_l1: bool
    eval_every: int
    print_accuracies: bool
    alpha: float
    no_l1_epochs: int
    
def compute_accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Compute accuracy on a given loader."""
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    return correct / total if total > 0 else 0.0


def load_model(dataset: str, checkpoint_path: str, device: torch.device) -> nn.Module:
    """Load a pre-trained model for the given dataset."""
    if dataset.lower() != "cifar10":
        raise ValueError(f"Unsupported dataset for fine-tune experiment: {dataset}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Model checkpoint not found at {checkpoint_path}. "
            f"Please run: python scripts/train_resnet18_cifar10.py"
        )

    model = models.resnet18(pretrained=False)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, get_num_classes("cifar10"))

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return model.to(device)


def ft_iter(data_loaders: Dict[str, DataLoader], model: nn.Module, args: FineTuneInput, epoch: int):
    """Fine-tuning iteration."""
    train_loader = data_loaders["train_retain_loader"]

    losses = AverageMeter()
    top1 = AverageMeter()

    model.train()
    device = next(model.parameters()).device

    start = time.time()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    with_l1 = args.with_l1

    for i, (image, target) in enumerate(train_loader):
        image = image.to(device)
        target = target.to(device)

        if epoch < args.unlearn_epochs - args.no_l1_epochs:
            current_alpha = args.alpha * (
                1 - epoch / (args.unlearn_epochs - args.no_l1_epochs)
            )
        elif args.unlearn_epochs - args.no_l1_epochs == 0:
            current_alpha = args.alpha
        else:
            current_alpha = 0.0

        output_clean = model(image)
        loss = criterion(output_clean, target)

        if with_l1:
            loss = loss + current_alpha * l1_regularization(model)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output_clean.float()
        loss = loss.float()
        prec1 = accuracy(output.data, target)[0]

        losses.update(loss.item(), image.size(0))
        top1.update(prec1.item(), image.size(0))

        if (i + 1) % args.print_freq == 0:
            end = time.time()
            print(
                f"Epoch: [{epoch}][{i}/{len(train_loader)}]\t"
                f"Loss {losses.val:.4f} ({losses.avg:.4f})\t"
                f"Accuracy {top1.val:.3f} ({top1.avg:.3f})\t"
                f"Time {end - start:.2f}"
            )
            start = time.time()

    print(f"train_accuracy {top1.avg:.3f}")
    return top1.avg


def fine_tune(loaders: Dict[str, DataLoader], args: FineTuneInput):
    """Run fine-tuning unlearning on the retain set."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(dataset=args.dataset, checkpoint_path=args.model_path, device=device)

    tr_accs, tf_accs, vr_accs, vf_accs = [], [], [], []
    epoch_list = []
    for epoch in range(1, args.unlearn_epochs + 1):
        print(f"\n[Fine-tune {epoch}/{args.unlearn_epochs}] Starting epoch...")
        ft_iter(loaders, model, args, epoch)

        acc_dict = None
        if args.eval_every and (epoch % args.eval_every == 0):
            model.eval()
            acc_dict = {
                "tr_acc": compute_accuracy(model, loaders["train_retain_loader"], device),
                "tf_acc": compute_accuracy(model, loaders["train_forget_loader"], device),
                "vr_acc": compute_accuracy(model, loaders["valid_retain_loader"], device),
                "vf_acc": compute_accuracy(model, loaders["valid_forget_loader"], device),
            }

            tr_accs.append(acc_dict["tr_acc"])
            tf_accs.append(acc_dict["tf_acc"])
            vr_accs.append(acc_dict["vr_acc"])
            vf_accs.append(acc_dict["vf_acc"])
        else:
            tr_accs.append(None)
            tf_accs.append(None)
            vr_accs.append(None)
            vf_accs.append(None)

        epoch_list.append(epoch)

        if args.print_accuracies and acc_dict is not None:
            print(f"   tr_acc: {acc_dict['tr_acc']:.4f}")
            print(f"   tf_acc: {acc_dict['tf_acc']:.4f}")
            print(f"   vr_acc: {acc_dict['vr_acc']:.4f}")
            print(f"   vf_acc: {acc_dict['vf_acc']:.4f}")

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
    }

    return model, history


def _load_config(config_path: str) -> FineTuneInput:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    import yaml

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f) or {}

    required_keys = {
        "dataset",
        "dataroot",
        "forget_fraction",
        "seed",
        "batch_size",
        "num_workers",
        "pin_memory",
        "model_path",
        "check_path",
        "learning_rate",
        "unlearn_epochs",
        "print_freq",
        "with_l1",
        "alpha",
        "no_l1_epochs",
        "eval_every",
        "print_accuracies",
    }

    missing_keys = sorted(required_keys - set(config_dict.keys()))
    if missing_keys:
        missing = ", ".join(missing_keys)
        raise ValueError(f"Missing required config keys: {missing}")

    return FineTuneInput(**config_dict)


def main():
    """Example usage of fine-tuning unlearning."""
    import argparse

    parser = argparse.ArgumentParser(description="Run fine-tuning unlearning experiment")
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/finetune_experiment.yaml",
        help="Path to YAML config file",
    )

    cli_args = parser.parse_args()
    args = _load_config(cli_args.config)

    # ========== 1. LOAD DATA ==========
    print("Loading dataset...")
    dataset = load_dataset(
        dataset_name=args.dataset,
        root=args.dataroot,
        train=True
    )

    # ========== 2. CREATE SPLITS ==========
    print("Creating retain/forget splits...")
    retain_set, forget_set = create_retain_forget_split(
        dataset,
        forget_fraction=args.forget_fraction,
        seed=args.seed,
        save_dir="./data/splits"
    )

    print(f"  Retain set size: {len(retain_set)}")
    print(f"  Forget set size: {len(forget_set)}")

    # ========== 3. CREATE TRAIN/VAL SPLITS ==========
    retain_len = len(retain_set)
    forget_len = len(forget_set)

    retain_train_len = int(0.9 * retain_len)
    forget_train_len = int(0.9 * forget_len)

    retain_train, retain_val = random_split(retain_set, [retain_train_len, retain_len - retain_train_len])
    forget_train, forget_val = random_split(forget_set, [forget_train_len, forget_len - forget_train_len])

    print(f"  Retain train: {len(retain_train)}, Retain val: {len(retain_val)}")
    print(f"  Forget train: {len(forget_train)}, Forget val: {len(forget_val)}")

    # ========== 4. CREATE DATA LOADERS ==========
    print("Creating data loaders...")
    pin_memory = args.pin_memory and torch.cuda.is_available()
    loaders = {
        "train_retain_loader": DataLoader(
            retain_train,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        ),
        "train_forget_loader": DataLoader(
            forget_train,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        ),
        "valid_retain_loader": DataLoader(
            retain_val,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        ),
        "valid_forget_loader": DataLoader(
            forget_val,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        ),
    }

    print("Data loaders created. Training will start now...")
    print(
        "Configuration: epochs={epochs}, batch_size={batch}, device={device}".format(
            epochs=args.unlearn_epochs,
            batch=args.batch_size,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
    )

    # ========== 5. RUN FINE-TUNING UNLEARNING ==========
    print("Running fine-tuning unlearning...")
    model, history = fine_tune(loaders, args)

    print("\nFine-tuning unlearning completed!")
    if history["tr_accs"] and history["tr_accs"][-1] is not None:
        print(f"Final train retain acc: {history['tr_accs'][-1]:.4f}")
    if history["vf_accs"] and history["vf_accs"][-1] is not None:
        print(f"Final valid forget acc: {history['vf_accs'][-1]:.4f}")

    return model, history


if __name__ == "__main__":
    main()
