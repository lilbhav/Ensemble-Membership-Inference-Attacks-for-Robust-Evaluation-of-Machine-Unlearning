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
from utils.metrics import evaluate_split_metrics, log_accuracies, report_weight_diff
from utils.splits import (
    ensure_retain_forget_split,
    ensure_targeted_random_unlearning_split,
    ensure_fully_random_unlearning_split,
)
from utils.transfer_setup import ensure_cifar10_from_cifar100_transfer_checkpoint
from utils.unlearning_results import (
    build_epoch_record,
    build_unlearning_summary,
    make_run_tag,
    resolve_unlearning_artifact_paths,
    save_unlearning_history_csv,
    save_unlearning_summary,
    to_serializable_dict,
)
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
    summary_path: Optional[str] = None
    history_path: Optional[str] = None
    amnesiac_train_epochs: int = 5
    unlearning_batch_size: int = 128
    optimizer_type: str = "adam"
    selection_delta: float = 0.0


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
    metrics: Dict[str, float] = {}
    metrics.update(evaluate_split_metrics(model, loaders["train_retain_loader"], device, "tr"))
    metrics.update(evaluate_split_metrics(model, loaders["train_forget_loader"], device, "tf"))
    metrics.update(evaluate_split_metrics(model, loaders["valid_retain_loader"], device, "vr"))
    metrics.update(evaluate_split_metrics(model, loaders["valid_forget_loader"], device, "vf"))
    return metrics


def _infer_forget_class(forget_dataset) -> int:
    label_counts: Dict[int, int] = {}
    for sample in forget_dataset:
        label = _extract_label(sample)
        label_counts[label] = label_counts.get(label, 0) + 1
    if not label_counts:
        raise ValueError("Forget dataset is empty; cannot infer forget class")
    return max(label_counts.items(), key=lambda kv: kv[1])[0]


def _run_amnesiac_with_config(
    model: nn.Module,
    forget_loader: DataLoader,
    retain_loader: DataLoader,
    test_loader: DataLoader,
    forget_class: int,
    num_classes: int,
    device: torch.device,
    args: AmnesiacInput,
) -> nn.Module:
    candidate_labels = list(range(num_classes))
    candidate_labels.remove(forget_class)

    unlearning_trainset = []
    for x, _ in forget_loader.dataset:
        unlearning_trainset.append((x, random.choice(candidate_labels)))
    for x, y in retain_loader.dataset:
        unlearning_trainset.append((x, y))

    unlearning_train_loader = DataLoader(
        unlearning_trainset,
        batch_size=int(getattr(args, "unlearning_batch_size", 128)),
        shuffle=True,
        pin_memory=bool(getattr(args, "pin_memory", True)) and torch.cuda.is_available(),
        num_workers=int(getattr(args, "num_workers", 2)),
    )

    trained_model = copy.deepcopy(model)
    optimizer_name = str(getattr(args, "optimizer_type", "adam")).strip().lower()
    learning_rate = float(getattr(args, "learning_rate", 1e-4))
    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(trained_model.parameters(), lr=learning_rate, momentum=0.5)
    else:
        optimizer = torch.optim.Adam(trained_model.parameters(), lr=learning_rate, weight_decay=1e-4)

    criterion = nn.CrossEntropyLoss().to(device)
    train_epochs = int(getattr(args, "amnesiac_train_epochs", 5))

    for epoch in range(1, train_epochs + 1):
        loss_values = []
        trained_model.train()
        for images, labels in unlearning_train_loader:
            images = images.to(device)
            labels = labels.long().to(device)

            optimizer.zero_grad()
            output = trained_model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            loss_values.append(float(loss.item()))

        mean_loss = float(np.mean(np.array(loss_values))) if loss_values else 0.0
        train_acc = _evaluate_all(trained_model, {
            "train_retain_loader": retain_loader,
            "train_forget_loader": forget_loader,
            "valid_retain_loader": retain_loader,
            "valid_forget_loader": forget_loader,
        }, device)["tr_acc"]
        test_acc = evaluate_split_metrics(trained_model, test_loader, device, "test")["test_acc"]
        print(f"Epochs: {epoch} Train Loss: {mean_loss:.4f} Train Acc: {train_acc * 100:.4f} Test acc: {test_acc * 100:.4f}")

    return trained_model


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
    baseline_acc.update(evaluate_split_metrics(model, loaders["test_loader"], device, "test"))
    if args.print_accuracies:
        line = log_accuracies(args.results_path, "baseline", baseline_acc)
        print(f"   {line}")
        print(f"   baseline | test_acc: {baseline_acc['test_acc']:.4f}")

    num_classes = int(get_num_classes(args.dataset))

    epoch_list = []
    tr_accs = []
    tf_accs = []
    vr_accs = []
    vf_accs = []
    epoch_metrics = []
    final_acc = dict(baseline_acc)
    selection_delta = float(getattr(args, "selection_delta", 0.0))
    min_vr_for_selection = float(baseline_acc["vr_acc"]) - selection_delta
    epoch_state_snapshots = []
    _baseline_state = {k: v.clone() for k, v in model.state_dict().items()}

    for epoch in range(1, int(args.unlearn_epochs) + 1):
        model = _run_amnesiac_with_config(
            model=model,
            forget_loader=loaders["train_forget_loader"],
            retain_loader=loaders["train_retain_loader"],
            test_loader=loaders["test_loader"],
            forget_class=forget_class,
            num_classes=num_classes,
            device=device,
            args=args,
        )

        model.eval()
        acc_dict = _evaluate_all(model, loaders, device)

        epoch_list.append(epoch)
        tr_accs.append(acc_dict["tr_acc"])
        tf_accs.append(acc_dict["tf_acc"])
        vr_accs.append(acc_dict["vr_acc"])
        vf_accs.append(acc_dict["vf_acc"])
        epoch_metrics.append(build_epoch_record(epoch, acc_dict))
        epoch_state_snapshots.append(
            {
                "epoch": epoch,
                "acc": dict(acc_dict),
                "state_dict": {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in model.state_dict().items()
                },
            }
        )
        final_acc = dict(acc_dict)

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

    if epoch_state_snapshots:
        feasible_candidates = [
            candidate
            for candidate in epoch_state_snapshots
            if float(candidate["acc"]["vr_acc"]) >= min_vr_for_selection
        ]
        used_constraint = True
        if not feasible_candidates:
            feasible_candidates = list(epoch_state_snapshots)
            used_constraint = False
            print(
                "[Amnesiac] No checkpoint satisfied vr_acc >= {:.4f}; "
                "falling back to lowest vf_acc across all checkpoints.".format(min_vr_for_selection)
            )

        selected_candidate = min(
            feasible_candidates,
            key=lambda candidate: (
                float(candidate["acc"]["vf_acc"]),
                -float(candidate["acc"]["vr_acc"]),
                int(candidate["epoch"]),
            ),
        )
        selected_epoch = int(selected_candidate["epoch"])
        selected_state_dict = selected_candidate["state_dict"]
    else:
        selected_epoch = None
        selected_state_dict = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
        used_constraint = True

    test_metrics = evaluate_split_metrics(model, loaders["test_loader"], device, "test")
    test_acc = float(test_metrics["test_acc"])
    final_acc.update(test_metrics)

    model.load_state_dict(selected_state_dict)
    model = model.to(device)
    model.eval()
    report_weight_diff(_baseline_state, model.state_dict(), "Amnesiac")

    selected_acc = _evaluate_all(model, loaders, device)
    selected_test_metrics = evaluate_split_metrics(model, loaders["test_loader"], device, "test")
    selected_acc.update(selected_test_metrics)
    test_acc = float(selected_acc["test_acc"])

    if args.print_accuracies:
        line = log_accuracies(args.results_path, "final", final_acc)
        print(f"   {line}")
        print(f"   final | test_acc: {float(final_acc['test_acc']):.4f}")
        selected_line = log_accuracies(
            args.results_path,
            f"selected_epoch {selected_epoch if selected_epoch is not None else 'baseline'}",
            selected_acc,
        )
        print(f"   {selected_line}")
        print(f"   selected | test_acc: {test_acc:.4f}")

    if args.check_path is not None:
        check_dir = os.path.dirname(args.check_path)
        if check_dir:
            os.makedirs(check_dir, exist_ok=True)
        torch.save(model.state_dict(), args.check_path)

    summary_path, history_path = resolve_unlearning_artifact_paths(
        method="amnesiac",
        results_path=args.results_path,
        check_path=args.check_path,
        summary_path=args.summary_path,
        history_path=args.history_path,
        run_tag=make_run_tag(seed=int(args.seed)),
    )

    history = {
        "epoch_list": epoch_list,
        "tr_accs": tr_accs,
        "tf_accs": tf_accs,
        "vr_accs": vr_accs,
        "vf_accs": vf_accs,
        "test_acc": test_acc,
        "forget_class": forget_class,
        "baseline_acc": baseline_acc,
        "final_acc": final_acc,
        "selected_acc": selected_acc,
        "best_epoch": selected_epoch,
        "selection_strategy": "min_vf_subject_to_vr_floor",
        "selection_vr_floor": min_vr_for_selection,
        "selection_delta": selection_delta,
        "selection_constraint_satisfied": used_constraint,
        "summary_path": summary_path,
        "history_csv_path": history_path,
    }

    summary = build_unlearning_summary(
        method="amnesiac",
        baseline_metrics=baseline_acc,
        final_metrics=final_acc,
        selected_metrics=selected_acc,
        selected_epoch=history["best_epoch"],
        selection_strategy="min_vf_subject_to_vr_floor",
        history_rows=epoch_metrics,
        loaders=loaders,
        run_config=to_serializable_dict(args),
        artifacts={
            "checkpoint_path": args.check_path,
            "results_path": args.results_path,
            "summary_path": summary_path,
            "history_path": history_path,
        },
        extra={
            "forget_class": forget_class,
            "unlearn_epochs": int(args.unlearn_epochs),
            "amnesiac_train_epochs": int(getattr(args, "amnesiac_train_epochs", 5)),
            "unlearning_batch_size": int(getattr(args, "unlearning_batch_size", 128)),
            "optimizer_type": str(getattr(args, "optimizer_type", "adam")),
            "learning_rate": float(getattr(args, "learning_rate", 1e-4)),
            "selection_delta": selection_delta,
            "selection_vr_floor": min_vr_for_selection,
            "selection_constraint_satisfied": used_constraint,
        },
    )
    save_unlearning_summary(summary_path, summary)
    save_unlearning_history_csv(history_path, epoch_metrics)
    history["summary"] = summary

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
    selected_acc = history.get("selected_acc", {})
    if selected_acc.get("vr_acc") is not None:
        print(f"Final valid retain acc: {selected_acc['vr_acc']:.4f}")
    if selected_acc.get("vf_acc") is not None:
        print(f"Final valid forget acc: {selected_acc['vf_acc']:.4f}")
    print(f"Final test acc: {history['test_acc']:.4f}")

    return model, history


if __name__ == "__main__":
    main()
