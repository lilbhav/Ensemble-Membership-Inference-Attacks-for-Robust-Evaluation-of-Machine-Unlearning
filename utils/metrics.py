import os
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader


def compute_accuracy(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Compute accuracy for a model on a dataloader."""
    return compute_classification_metrics(model, loader, device)["accuracy"]


def _compute_macro_weighted_prf(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    if y_true.size == 0:
        return {
            "precision_macro": 0.0,
            "recall_macro": 0.0,
            "f1_macro": 0.0,
            "precision_weighted": 0.0,
            "recall_weighted": 0.0,
            "f1_weighted": 0.0,
        }

    labels = np.unique(np.concatenate([y_true, y_pred]))
    supports: List[float] = []
    precisions: List[float] = []
    recalls: List[float] = []
    f1s: List[float] = []

    for label in labels:
        true_pos = float(np.sum((y_true == label) & (y_pred == label)))
        false_pos = float(np.sum((y_true != label) & (y_pred == label)))
        false_neg = float(np.sum((y_true == label) & (y_pred != label)))
        support = float(np.sum(y_true == label))

        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0.0
        recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        supports.append(support)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    support_sum = float(np.sum(supports))
    weights = [s / support_sum if support_sum > 0 else 0.0 for s in supports]

    return {
        "precision_macro": float(np.mean(precisions)),
        "recall_macro": float(np.mean(recalls)),
        "f1_macro": float(np.mean(f1s)),
        "precision_weighted": float(np.sum(np.array(precisions) * np.array(weights))),
        "recall_weighted": float(np.sum(np.array(recalls) * np.array(weights))),
        "f1_weighted": float(np.sum(np.array(f1s) * np.array(weights))),
    }


def compute_classification_metrics(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Compute accuracy, macro PRF, and weighted PRF for a model on a dataloader."""
    predictions: List[int] = []
    targets_list: List[int] = []

    was_training = model.training
    model.eval()

    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                inputs, _, targets = batch
            else:
                inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            predicted = torch.argmax(outputs, dim=1)

            predictions.extend(predicted.detach().cpu().numpy().astype(int).tolist())
            targets_list.extend(targets.detach().cpu().numpy().astype(int).tolist())

    if was_training:
        model.train()

    if not targets_list:
        return {
            "accuracy": 0.0,
            "precision_macro": 0.0,
            "recall_macro": 0.0,
            "f1_macro": 0.0,
            "precision_weighted": 0.0,
            "recall_weighted": 0.0,
            "f1_weighted": 0.0,
        }

    y_true = np.array(targets_list, dtype=np.int64)
    y_pred = np.array(predictions, dtype=np.int64)
    accuracy = float(np.mean(y_true == y_pred))
    prf = _compute_macro_weighted_prf(y_true, y_pred)

    return {
        "accuracy": accuracy,
        **prf,
    }


def evaluate_split_metrics(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    prefix: str,
) -> Dict[str, float]:
    """Compute split metrics and return namespaced keys for result summaries."""
    metrics = compute_classification_metrics(model, loader, device)
    return {
        f"{prefix}_acc": float(metrics["accuracy"]),
        f"{prefix}_precision_macro": float(metrics["precision_macro"]),
        f"{prefix}_recall_macro": float(metrics["recall_macro"]),
        f"{prefix}_f1_macro": float(metrics["f1_macro"]),
        f"{prefix}_precision_weighted": float(metrics["precision_weighted"]),
        f"{prefix}_recall_weighted": float(metrics["recall_weighted"]),
        f"{prefix}_f1_weighted": float(metrics["f1_weighted"]),
    }


def log_accuracies(log_path: str, label: str, acc_dict: dict) -> str:
    """Format and optionally append accuracy metrics to a log file."""
    line = (
        f"{label} | tr_acc: {acc_dict['tr_acc']:.4f} "
        f"tf_acc: {acc_dict['tf_acc']:.4f} "
        f"vr_acc: {acc_dict['vr_acc']:.4f} "
        f"vf_acc: {acc_dict['vf_acc']:.4f}"
    )

    # Append macro P/R/F1 when available without breaking old log formats.
    macro_keys = [
        "tr_precision_macro",
        "tr_recall_macro",
        "tr_f1_macro",
        "tf_precision_macro",
        "tf_recall_macro",
        "tf_f1_macro",
        "vr_precision_macro",
        "vr_recall_macro",
        "vr_f1_macro",
        "vf_precision_macro",
        "vf_recall_macro",
        "vf_f1_macro",
    ]
    if all(key in acc_dict for key in macro_keys):
        line = (
            f"{line} "
            f"tr_p: {acc_dict['tr_precision_macro']:.4f} tr_r: {acc_dict['tr_recall_macro']:.4f} tr_f1: {acc_dict['tr_f1_macro']:.4f} "
            f"tf_p: {acc_dict['tf_precision_macro']:.4f} tf_r: {acc_dict['tf_recall_macro']:.4f} tf_f1: {acc_dict['tf_f1_macro']:.4f} "
            f"vr_p: {acc_dict['vr_precision_macro']:.4f} vr_r: {acc_dict['vr_recall_macro']:.4f} vr_f1: {acc_dict['vr_f1_macro']:.4f} "
            f"vf_p: {acc_dict['vf_precision_macro']:.4f} vf_r: {acc_dict['vf_recall_macro']:.4f} vf_f1: {acc_dict['vf_f1_macro']:.4f}"
        )

    if log_path:
        log_dir = os.path.dirname(log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    return line


def report_weight_diff(
    baseline_state: dict,
    unlearned_state: dict,
    method_name: str = "unlearning",
    tol: float = 1e-6,
) -> None:
    """Print per-layer L2 parameter difference norms between baseline and unlearned models.

    Raises RuntimeError if the unlearned model is identical or nearly identical to the
    baseline checkpoint (total L2 diff < tol), which indicates that unlearning had no effect.
    """
    layer_diffs = {}
    for name in baseline_state:
        if name not in unlearned_state:
            continue
        diff = (unlearned_state[name].float() - baseline_state[name].float()).norm().item()
        layer_diffs[name] = diff

    if not layer_diffs:
        raise RuntimeError(
            f"[{method_name}] Cannot compare weights: no matching parameter names found "
            "between baseline and unlearned state dicts."
        )

    max_name_len = max(len(n) for n in layer_diffs)
    sep = "=" * (max_name_len + 28)
    print(f"\n{sep}")
    print(f"[{method_name}] Per-layer weight change (L2 norm of parameter diff)")
    print(sep)
    for name, diff in layer_diffs.items():
        marker = "  <-- UNCHANGED" if diff < tol else ""
        print(f"  {name:<{max_name_len}}  {diff:.6e}{marker}")
    total_diff = sum(layer_diffs.values())
    unchanged_layers = [n for n, d in layer_diffs.items() if d < tol]
    print(sep)
    print(f"  Total L2 diff across all layers : {total_diff:.6e}")
    print(f"  Layers with zero change         : {len(unchanged_layers)} / {len(layer_diffs)}")
    print(f"{sep}\n")

    if len(unchanged_layers) == len(layer_diffs):
        raise RuntimeError(
            f"[{method_name}] Unlearning produced NO weight changes — the unlearned model is "
            "identical to the baseline checkpoint. Verify that the unlearning strategy is "
            "actually modifying model parameters and that data loaders are non-empty."
        )

    if total_diff < tol:
        raise RuntimeError(
            f"[{method_name}] Total weight change ({total_diff:.2e}) is below tolerance "
            f"({tol:.2e}). The unlearned model is effectively identical to the baseline. "
            "Check unlearning configuration and that forget/retain loaders have data."
        )

    if unchanged_layers:
        sample = ", ".join(unchanged_layers[:5])
        ellipsis = "..." if len(unchanged_layers) > 5 else ""
        print(
            f"[{method_name}] WARNING: {len(unchanged_layers)} layer(s) had no weight change: "
            f"{sample}{ellipsis}"
        )
