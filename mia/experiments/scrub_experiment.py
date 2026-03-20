# Imports
import copy
import argparse
import sys
import os
import random
import numpy as np
import yaml

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models

# Add repo root and third-party package roots.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

TP_MACHINEUNLEARNING_ROOT = os.path.join(REPO_ROOT, "Third_Party_Code", "MachineUnlearning")
if TP_MACHINEUNLEARNING_ROOT not in sys.path:
    sys.path.insert(0, TP_MACHINEUNLEARNING_ROOT)

# Framework imports
from data.loaders import load_dataset, get_num_classes
from utils.splits import ensure_retain_forget_split, ensure_targeted_random_unlearning_split, ensure_fully_random_unlearning_split
from utils.metrics import compute_accuracy, log_accuracies
from utils.transfer_setup import ensure_cifar10_from_cifar100_transfer_checkpoint

# Third-party strategy import (delegate algorithm implementation here)
from Third_Party_Code.MachineUnlearning.unlearn_strategies import strategies as third_party_strategies


def set_global_determinism(seed: int) -> None:
    """Set deterministic seeds for reproducible dataset splits and training order."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _load_model(dataset: str, checkpoint_path: str, device: torch.device) -> nn.Module:
    if dataset.lower() != "cifar10":
        raise ValueError(f"Unsupported dataset for scrub experiment: {dataset}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Model checkpoint not found at {checkpoint_path}. "
            "Please run: python scripts/train_resnet18_cifar10.py"
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


def scrub(loaders, args):
    """
    Perform SCRUB unlearning via the Third_Party_Code strategy implementation.
    """
    model_checkpoint = getattr(args, "model_path", "./models/pretrained_cifar10.pt")
    if not os.path.exists(model_checkpoint):
        raise FileNotFoundError(
            f"Model checkpoint not found at {model_checkpoint}. "
            "Please run: python scripts/train_resnet18_cifar10.py"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = _load_model(dataset=args.dataset, checkpoint_path=model_checkpoint, device=device)

    train_forget_loader = loaders["train_forget_loader"]
    train_retain_loader = loaders["train_retain_loader"]
    valid_forget_loader = loaders["valid_forget_loader"]
    valid_retain_loader = loaders["valid_retain_loader"]
    test_loader = loaders["test_loader"]
    results_path = getattr(args, "results_path", None)

    baseline_acc = {
        "tr_acc": compute_accuracy(model, train_retain_loader, device),
        "tf_acc": compute_accuracy(model, train_forget_loader, device),
        "vr_acc": compute_accuracy(model, valid_retain_loader, device),
        "vf_acc": compute_accuracy(model, valid_forget_loader, device),
    }
    print(
        "Baseline - tr_acc: {tr:.4f}, tf_acc: {tf:.4f}, vr_acc: {vr:.4f}, vf_acc: {vf:.4f}".format(
            tr=baseline_acc["tr_acc"],
            tf=baseline_acc["tf_acc"],
            vr=baseline_acc["vr_acc"],
            vf=baseline_acc["vf_acc"],
        )
    )
    if args.print_accuracies:
        log_accuracies(results_path, "baseline", baseline_acc)

    forget_class = int(getattr(args, "forget_class", _infer_forget_class(train_forget_loader.dataset)))
    num_classes = int(get_num_classes(args.dataset))
    num_channels = int(next(iter(train_retain_loader))[0].shape[1])

    # Keep algorithm execution in third-party code; wrapper only controls run count and evaluation.
    unlearn_runs = int(getattr(args, "unlearn_epochs", 1))
    strategy_args = argparse.Namespace()
    unlearning_teacher = copy.deepcopy(model)

    tr_accs, tf_accs, vr_accs, vf_accs, epoch_list = [], [], [], [], []

    for epoch in range(1, unlearn_runs + 1):
        model = third_party_strategies.scrub(
            args=strategy_args,
            model=model,
            unlearning_teacher=unlearning_teacher,
            unlearn_class=forget_class,
            unlearn_loader=train_forget_loader,
            retain_loader=train_retain_loader,
            test_loader=test_loader,
            num_classes=num_classes,
            num_channels=num_channels,
            device=device,
        )

        model.eval()
        acc_dict = {
            "tr_acc": compute_accuracy(model, train_retain_loader, device),
            "tf_acc": compute_accuracy(model, train_forget_loader, device),
            "vr_acc": compute_accuracy(model, valid_retain_loader, device),
            "vf_acc": compute_accuracy(model, valid_forget_loader, device),
        }

        tr_accs.append(acc_dict["tr_acc"])
        tf_accs.append(acc_dict["tf_acc"])
        vr_accs.append(acc_dict["vr_acc"])
        vf_accs.append(acc_dict["vf_acc"])
        epoch_list.append(epoch)

        print(
            "[SCRUB third-party {}/{}] tr={:.4f} tf={:.4f} vr={:.4f} vf={:.4f}".format(
                epoch,
                unlearn_runs,
                acc_dict["tr_acc"],
                acc_dict["tf_acc"],
                acc_dict["vr_acc"],
                acc_dict["vf_acc"],
            )
        )

        if args.print_accuracies:
            log_accuracies(results_path, f"epoch {epoch}", acc_dict)

    if getattr(args, "check_path", None) is not None:
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
        "selected_acc": {
            "tr_acc": tr_accs[-1] if tr_accs else baseline_acc["tr_acc"],
            "tf_acc": tf_accs[-1] if tf_accs else baseline_acc["tf_acc"],
            "vr_acc": vr_accs[-1] if vr_accs else baseline_acc["vr_acc"],
            "vf_acc": vf_accs[-1] if vf_accs else baseline_acc["vf_acc"],
        },
    }

    return model, history


def main():
    """Run SCRUB unlearning experiment with framework-managed splits."""
    parser = argparse.ArgumentParser(description="Run SCRUB unlearning experiment")
    parser.add_argument("--config", type=str, default="./configs/scrub_experiment.yaml", help="Path to YAML config file")

    cli_args = parser.parse_args()

    if not os.path.exists(cli_args.config):
        raise FileNotFoundError(f"Config file not found: {cli_args.config}")

    with open(cli_args.config, "r") as f:
        config_dict = yaml.safe_load(f) or {}
    args = SimpleNamespace(**config_dict)
    if not hasattr(args, "split_dir"):
        args.split_dir = "./data/splits"

    set_global_determinism(int(args.seed))

    if getattr(args, "source_checkpoint_cifar100", None):
        args.model_path = ensure_cifar10_from_cifar100_transfer_checkpoint(
            source_checkpoint_cifar100=args.source_checkpoint_cifar100,
            target_checkpoint_cifar10=args.model_path,
            dataroot=getattr(args, "dataroot", "./data/raw"),
            finetune_epochs=int(getattr(args, "transfer_finetune_epochs", 10)),
            finetune_batch_size=int(getattr(args, "transfer_finetune_batch_size", 128)),
            finetune_learning_rate=float(getattr(args, "transfer_finetune_learning_rate", 0.001)),
            seed=int(getattr(args, "seed", 42)),
            num_workers=int(getattr(args, "num_workers", 2)),
            pin_memory=bool(getattr(args, "pin_memory", True)),
            force_rebuild=bool(getattr(args, "rebuild_transfer_checkpoint", False)),
        )

    print("Loading dataset...")
    dataset = load_dataset(dataset_name=args.dataset, root=args.dataroot, train=True)
    test_dataset = load_dataset(dataset_name=args.dataset, root=args.dataroot, train=False)

    split_dir = args.split_dir
    split_protocol = str(getattr(args, "split_protocol", "targeted_random")).strip().lower()

    has_count_keys = (
        getattr(args, "forget_count", None) is not None
        and (
            getattr(args, "retain_count", None) is not None
            or getattr(args, "retain_per_class", None) is not None
        )
        and (
            getattr(args, "left_out_count", None) is not None
            or getattr(args, "left_out_per_class", None) is not None
        )
    )

    if split_protocol == "fully_random" and has_count_keys:
        retain_count = int(args.retain_count) if getattr(args, "retain_count", None) is not None else int(args.retain_per_class) * (int(get_num_classes(args.dataset)) - 1)
        left_out_count = int(args.left_out_count) if getattr(args, "left_out_count", None) is not None else int(args.left_out_per_class) * (int(get_num_classes(args.dataset)) - 1)
        forget_count = int(args.forget_count)

        retain_set, forget_set, left_out_set, _ = ensure_fully_random_unlearning_split(
            dataset=dataset,
            split_dir=split_dir,
            retain_count=retain_count,
            forget_count=forget_count,
            left_out_count=left_out_count,
            seed=int(args.seed),
            verbose=True,
        )
    elif getattr(args, "forget_class", None) is not None and has_count_keys:
        classes_excluding_forget = int(get_num_classes(args.dataset)) - 1
        retain_count = int(args.retain_count) if getattr(args, "retain_count", None) is not None else int(args.retain_per_class) * classes_excluding_forget
        left_out_count = int(args.left_out_count) if getattr(args, "left_out_count", None) is not None else int(args.left_out_per_class) * classes_excluding_forget
        forget_count = int(args.forget_count)

        retain_set, forget_set, left_out_set, _ = ensure_targeted_random_unlearning_split(
            dataset=dataset,
            split_dir=split_dir,
            forget_class=int(args.forget_class),
            retain_count=retain_count,
            forget_count=forget_count,
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
        split_generator = torch.Generator().manual_seed(int(args.seed))
        retain_train_len = int(0.9 * len(retain_set))
        retain_set, left_out_set = torch.utils.data.random_split(
            retain_set,
            [retain_train_len, len(retain_set) - retain_train_len],
            generator=split_generator,
        )

    print("Creating data loaders...")
    pin_memory = bool(getattr(args, "pin_memory", True)) and torch.cuda.is_available()
    num_workers = int(getattr(args, "num_workers", 2))
    base_batch_size = int(getattr(args, "batch_size", 64))
    split_generator = torch.Generator().manual_seed(int(args.seed))

    sgda_batch_size = int(getattr(args, "sgda_batch_size", base_batch_size))
    del_batch_size = int(getattr(args, "del_batch_size", base_batch_size))
    loaders = {
        "train_retain_loader": DataLoader(retain_set, batch_size=sgda_batch_size, shuffle=True, generator=split_generator, num_workers=num_workers, pin_memory=pin_memory),
        "train_forget_loader": DataLoader(forget_set, batch_size=del_batch_size, shuffle=True, generator=split_generator, num_workers=num_workers, pin_memory=pin_memory),
        "valid_retain_loader": DataLoader(left_out_set, batch_size=sgda_batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory),
        "valid_forget_loader": DataLoader(forget_set, batch_size=del_batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory),
        "test_loader": DataLoader(test_dataset, batch_size=sgda_batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory),
    }

    print("Running SCRUB unlearning...")
    model, history = scrub(loaders, args)

    print("\nSCRUB unlearning completed")
    selected_acc = history.get("selected_acc")
    if selected_acc is not None:
        print(f"Selected train retain acc: {selected_acc['tr_acc']:.4f}")
        print(f"Selected valid retain acc: {selected_acc['vr_acc']:.4f}")
        print(f"Selected train forget acc: {selected_acc['tf_acc']:.4f}")
        print(f"Selected valid forget acc: {selected_acc['vf_acc']:.4f}")

    return model, history


if __name__ == "__main__":
    from types import SimpleNamespace
    main()
