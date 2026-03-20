"""
Selective Synaptic Dampening (SSD) unlearning experiment.
This wrapper keeps split management in-framework and delegates unlearning to Third_Party_Code strategy.
"""

import os
import sys
import argparse
from dataclasses import dataclass, replace
from typing import Dict, Optional

import numpy as np
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
from utils.metrics import evaluate_split_metrics, log_accuracies, report_weight_diff
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
    summary_path: Optional[str] = None
    history_path: Optional[str] = None
    selection_delta: float = 0.0


def train_validation(
    model: nn.Module,
    train_retain_loader: DataLoader,
    train_forget_loader: DataLoader,
    valid_retain_loader: DataLoader,
    valid_forget_loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    metrics.update(evaluate_split_metrics(model, train_retain_loader, device, "tr"))
    metrics.update(evaluate_split_metrics(model, train_forget_loader, device, "tf"))
    metrics.update(evaluate_split_metrics(model, valid_retain_loader, device, "vr"))
    metrics.update(evaluate_split_metrics(model, valid_forget_loader, device, "vf"))
    return metrics


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


# ---------------------------------------------------------------------------
# Debug helpers — label-mismatch / logit / argmax diagnostics
# ---------------------------------------------------------------------------

_CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def _debug_forget_label_distribution(forget_dataset, prefix: str = "") -> None:
    """Print the raw label distribution of the forget subset without loading images."""
    from torch.utils.data import Subset as _Subset

    tag = f"[DEBUG{' ' + prefix if prefix else ''}]"

    # Fast path: walk the Subset chain and index into dataset.targets directly.
    base = forget_dataset
    chain = []
    while isinstance(base, _Subset):
        chain.append(np.asarray(base.indices, dtype=int))
        base = base.dataset

    raw_targets = getattr(base, "targets", None) or getattr(base, "labels", None)
    if raw_targets is not None:
        targets_arr = np.asarray(raw_targets, dtype=int)
        for idx_arr in reversed(chain):
            targets_arr = targets_arr[idx_arr]
        labels = targets_arr.tolist()
    else:
        # Fallback: iterate items (slower — loads images).
        labels = []
        for sample in forget_dataset:
            if isinstance(sample, tuple) and len(sample) >= 2:
                labels.append(int(sample[1]))

    label_counts: Dict[int, int] = {}
    for lbl in labels:
        label_counts[lbl] = label_counts.get(lbl, 0) + 1

    print(
        f"{tag} Forget-set raw label distribution ({len(labels)} samples): "
        f"{dict(sorted(label_counts.items()))}"
    )
    if len(label_counts) == 1:
        only = next(iter(label_counts))
        name = _CIFAR10_CLASSES[only] if 0 <= only < len(_CIFAR10_CLASSES) else f"cls{only}"
        print(f"{tag}   → single-class forget set: class {only} ({name})")
    else:
        print(
            f"{tag}   → multi-class forget set "
            f"({len(label_counts)} distinct classes: {sorted(label_counts.keys())})"
        )


def debug_forget_predictions(
    model: nn.Module,
    forget_loader: DataLoader,
    device: torch.device,
    max_samples: int = 20,
    forget_class: Optional[int] = None,
    prefix: str = "",
) -> None:
    """
    Run the first *max_samples* forget-set items through the model and print:
      - per-sample: true label, predicted label, full logit vector, pred-true offset.
      - summary: logit range/std, dominant predicted class, and four targeted warnings:
          1. Dead-output  : logit range ≈ 0 (weights zeroed out).
          2. Pred-collapse: all predictions map to one class that is not the true class.
          3. Constant shift: pred-true offset is constant and non-zero across all samples
                             (label-index off-by-one / modular wrapping / wrong dataset).
          4. Config mismatch: forget-set true label != args.forget_class.
    """
    tag = f"[DEBUG{' ' + prefix if prefix else ''}]"
    print("=" * 72)
    print(f"{tag} Forget-set prediction diagnostics (first {max_samples} samples)")
    if forget_class is not None:
        fc_name = (
            _CIFAR10_CLASSES[forget_class]
            if 0 <= forget_class < len(_CIFAR10_CLASSES)
            else "?"
        )
        print(f"{tag} args.forget_class = {forget_class} ({fc_name})")
    print("=" * 72)

    was_training = model.training
    model.eval()

    samples_shown = 0
    true_labels_seen: set = set()
    pred_labels_seen: set = set()
    all_logits_rows = []
    offsets: list = []

    with torch.no_grad():
        for batch in forget_loader:
            if samples_shown >= max_samples:
                break
            if len(batch) == 3:
                imgs, _, targets = batch
            else:
                imgs, targets = batch
            imgs = imgs.to(device)
            logits_batch = model(imgs).cpu()
            preds_batch = torch.argmax(logits_batch, dim=1)
            targets_cpu = targets.cpu()

            for i in range(len(targets_cpu)):
                if samples_shown >= max_samples:
                    break
                true_lbl = int(targets_cpu[i].item())
                pred_lbl = int(preds_batch[i].item())
                logit_vec = logits_batch[i].tolist()
                all_logits_rows.append(logit_vec)
                offsets.append(pred_lbl - true_lbl)

                true_name = (
                    _CIFAR10_CLASSES[true_lbl]
                    if 0 <= true_lbl < len(_CIFAR10_CLASSES)
                    else f"cls{true_lbl}"
                )
                pred_name = (
                    _CIFAR10_CLASSES[pred_lbl]
                    if 0 <= pred_lbl < len(_CIFAR10_CLASSES)
                    else f"cls{pred_lbl}"
                )
                correct_marker = "OK   " if true_lbl == pred_lbl else "WRONG"
                shift = pred_lbl - true_lbl
                shift_tag = (
                    f"  [shift={shift:+d} → possible label-index offset]"
                    if shift != 0
                    else ""
                )
                logit_str = ", ".join(f"{v:7.3f}" for v in logit_vec)

                print(
                    f"  {samples_shown:02d}: true={true_lbl}({true_name:<10s}) "
                    f"pred={pred_lbl}({pred_name:<10s}) [{correct_marker}]{shift_tag}"
                )
                print(f"       logits=[{logit_str}]")

                true_labels_seen.add(true_lbl)
                pred_labels_seen.add(pred_lbl)
                samples_shown += 1

    if was_training:
        model.train()

    # ── Summary diagnostics ──────────────────────────────────────────────────
    print("-" * 72)
    print(f"{tag} Unique true  labels in sample : {sorted(true_labels_seen)}")
    print(f"{tag} Unique pred  labels in sample : {sorted(pred_labels_seen)}")

    if all_logits_rows:
        arr = np.array(all_logits_rows, dtype=np.float32)   # [N, C]
        logit_range = float(arr.max() - arr.min())
        logit_std   = float(arr.std())
        dom_cls     = int(np.bincount(np.argmax(arr, axis=1)).argmax())
        dom_name    = (
            _CIFAR10_CLASSES[dom_cls]
            if 0 <= dom_cls < len(_CIFAR10_CLASSES)
            else "?"
        )
        print(f"{tag} Logit range (all samples+classes) : {logit_range:.4f}  std: {logit_std:.4f}")
        print(f"{tag} Most-common argmax class           : {dom_cls} ({dom_name})")

        # 1. Dead-output check
        if logit_range < 0.01:
            print(
                f"{tag} [WARN] Logit range ≈ 0 → model output is nearly constant; "
                f"SSD may have zeroed out critical weights."
            )

        # 2. Prediction-collapse check
        if len(pred_labels_seen) == 1 and pred_labels_seen != true_labels_seen:
            col_cls = next(iter(pred_labels_seen))
            col_name = (
                _CIFAR10_CLASSES[col_cls]
                if 0 <= col_cls < len(_CIFAR10_CLASSES)
                else "?"
            )
            print(
                f"{tag} [WARN] All {samples_shown} predictions collapse to class "
                f"{col_cls} ({col_name}) → systematic wrong prediction, NOT random forgetting."
            )

        # 3. Constant-shift check
        unique_offsets = set(offsets)
        if len(unique_offsets) == 1 and 0 not in unique_offsets:
            const_shift = next(iter(unique_offsets))
            print(
                f"{tag} [WARN] Constant pred-true offset = {const_shift:+d} across all samples "
                f"→ likely label-index shift bug (off-by-one, modular wrapping, or wrong dataset)."
            )

        # 4. Config-mismatch check
        if forget_class is not None and len(true_labels_seen) == 1:
            only_true = next(iter(true_labels_seen))
            if only_true != forget_class:
                print(
                    f"{tag} [WARN] Forget-set true label ({only_true}) != args.forget_class "
                    f"({forget_class}) → split was built for the wrong class or config is stale."
                )

    print("=" * 72)


# ---------------------------------------------------------------------------


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
    baseline_test = evaluate_split_metrics(model, test_loader, device, "test")
    baseline_test_acc = float(baseline_test["test_acc"])
    baseline_acc.update(baseline_test)
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

    # ── PRE-unlearning diagnostics ────────────────────────────────────────────
    _debug_forget_label_distribution(train_forget_loader.dataset, prefix="PRE-UNLEARN")
    debug_forget_predictions(
        model, train_forget_loader, device,
        forget_class=forget_class, prefix="PRE-UNLEARN",
    )
    # ─────────────────────────────────────────────────────────────────────────

    def _run_ssd_with_config(current_model: nn.Module) -> nn.Module:
        # Keep third-party algorithm components, but inject wrapper-configured hyperparameters.
        # The vendored convenience function currently hardcodes defaults and ignores caller args.
        parameters = {
            "lower_bound": float(getattr(args, "lower_bound", 1.0)),
            "exponent": float(getattr(args, "exponent", 1.0)),
            "magnitude_diff": None,
            "min_layer": int(getattr(args, "min_layer", -1)),
            "max_layer": int(getattr(args, "max_layer", -1)),
            "forget_threshold": float(getattr(args, "forget_threshold", 1.0)),
            "dampening_constant": float(getattr(args, "dampening_constant", 1.0)),
            "selection_weighting": float(getattr(args, "selection_weighting", 10.0)),
        }

        optimizer = torch.optim.SGD(current_model.parameters(), lr=float(args.learning_rate))
        pdr = third_party_strategies.ParameterPerturber(current_model, optimizer, device, parameters)
        current_model = current_model.eval()

        sample_importances = pdr.calc_importance(train_forget_loader)
        original_importances = pdr.calc_importance(train_retain_loader)
        pdr.modify_weight(original_importances, sample_importances)

        return current_model

    runs = int(getattr(args, "unlearn_epochs", 1))
    selection_delta = float(getattr(args, "selection_delta", 0.0))
    min_vr_for_selection = float(baseline_acc["vr_acc"]) - selection_delta
    _baseline_state = {k: v.clone() for k, v in model.state_dict().items()}
    epoch_state_snapshots = []
    epoch_metrics = []
    final_acc = dict(baseline_acc)
    for epoch in range(1, runs + 1):
        model = _run_ssd_with_config(model)

        # ── POST-unlearning diagnostics (epoch {epoch}) ───────────────────────
        debug_forget_predictions(
            model, train_forget_loader, device,
            forget_class=forget_class, prefix=f"POST-EPOCH-{epoch}",
        )
        # ─────────────────────────────────────────────────────────────────────

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
        epoch_metrics.append(build_epoch_record(epoch, acc_epoch))
        epoch_state_snapshots.append(
            {
                "epoch": epoch,
                "acc": dict(acc_epoch),
                "state_dict": {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in model.state_dict().items()
                },
            }
        )
        final_acc = dict(acc_epoch)
        if args.print_accuracies:
            line = log_accuracies(args.results_path, f"epoch {epoch}", acc_epoch)
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
                "[SSD] No checkpoint satisfied vr_acc >= {:.4f}; "
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

    final_test_metrics = evaluate_split_metrics(model, test_loader, device, "test")
    final_acc.update(final_test_metrics)

    model.load_state_dict(selected_state_dict)
    model = model.to(device)
    model.eval()

    selected_acc = train_validation(
        model,
        train_retain_loader,
        train_forget_loader,
        valid_retain_loader,
        valid_forget_loader,
        device,
    )
    selected_test_metrics = evaluate_split_metrics(model, test_loader, device, "test")
    after_test_acc = float(selected_test_metrics["test_acc"])
    selected_acc.update(selected_test_metrics)

    report_weight_diff(_baseline_state, model.state_dict(), "SSD")

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
        print(f"   selected | test_acc: {after_test_acc:.4f}")

    if args.check_path is not None:
        check_dir = os.path.dirname(args.check_path)
        if check_dir:
            os.makedirs(check_dir, exist_ok=True)
        torch.save(model.state_dict(), args.check_path)

    summary_path, history_path = resolve_unlearning_artifact_paths(
        method="ssd",
        results_path=args.results_path,
        check_path=args.check_path,
        summary_path=args.summary_path,
        history_path=args.history_path,
        run_tag=make_run_tag(seed=int(args.seed)),
    )

    history = {
        "epoch_list": [row["epoch"] for row in epoch_metrics],
        "tr_accs": [row.get("tr_acc") for row in epoch_metrics],
        "tf_accs": [row.get("tf_acc") for row in epoch_metrics],
        "vr_accs": [row.get("vr_acc") for row in epoch_metrics],
        "vf_accs": [row.get("vf_acc") for row in epoch_metrics],
        "baseline_acc": baseline_acc,
        "final_acc": final_acc,
        "selected_acc": selected_acc,
        "best_epoch": selected_epoch,
        "selection_strategy": "min_vf_subject_to_vr_floor",
        "selection_vr_floor": min_vr_for_selection,
        "selection_delta": selection_delta,
        "selection_constraint_satisfied": used_constraint,
        "test_acc": after_test_acc,
        "forget_class": forget_class,
        "summary_path": summary_path,
        "history_csv_path": history_path,
    }

    summary = build_unlearning_summary(
        method="ssd",
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
            "unlearn_epochs": runs,
            "dampening_constant": float(getattr(args, "dampening_constant", 1.0)),
            "selection_weighting": float(getattr(args, "selection_weighting", 10.0)),
            "selection_delta": selection_delta,
            "selection_vr_floor": min_vr_for_selection,
            "selection_constraint_satisfied": used_constraint,
        },
    )
    save_unlearning_summary(summary_path, summary)
    save_unlearning_history_csv(history_path, epoch_metrics)
    history["summary"] = summary

    return model, history


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
    parser.add_argument(
        "--selection-delta",
        type=float,
        default=None,
        help="Selection rule delta: choose lowest vf_acc subject to vr_acc >= baseline_vr_acc - delta.",
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
    if cli_args.selection_delta is not None:
        overrides["selection_delta"] = cli_args.selection_delta
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
    model, history = ssd(loaders, args)

    if args.print_accuracies:
        selected_acc = history.get("selected_acc", {})
        print(f"   tr_acc: {selected_acc['tr_acc']:.4f}")
        print(f"   tf_acc: {selected_acc['tf_acc']:.4f}")
        print(f"   vr_acc: {selected_acc['vr_acc']:.4f}")
        print(f"   vf_acc: {selected_acc['vf_acc']:.4f}")
        if selected_acc.get("test_acc") is not None:
            print(f"   test_acc: {selected_acc['test_acc']:.4f}")

    return model, history


if __name__ == "__main__":
    main()
