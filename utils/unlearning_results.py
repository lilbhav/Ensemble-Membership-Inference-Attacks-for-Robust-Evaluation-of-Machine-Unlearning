import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


ACCURACY_KEYS: Tuple[str, ...] = ("tr_acc", "tf_acc", "vr_acc", "vf_acc", "test_acc")


def normalize_accuracy_dict(metrics: Optional[Dict[str, Any]]) -> Dict[str, float]:
    normalized: Dict[str, float] = {}
    if not metrics:
        return normalized

    for key in ACCURACY_KEYS:
        value = metrics.get(key)
        if value is not None:
            normalized[key] = float(value)

    return normalized


def build_epoch_record(epoch: int, metrics: Dict[str, Any]) -> Dict[str, float]:
    record: Dict[str, float] = {"epoch": int(epoch)}
    record.update(normalize_accuracy_dict(metrics))
    return record


def compute_metric_deltas(
    baseline_metrics: Optional[Dict[str, Any]],
    current_metrics: Optional[Dict[str, Any]],
) -> Dict[str, Optional[float]]:
    baseline = normalize_accuracy_dict(baseline_metrics)
    current = normalize_accuracy_dict(current_metrics)

    deltas: Dict[str, Optional[float]] = {}
    for key in ACCURACY_KEYS:
        baseline_value = baseline.get(key)
        current_value = current.get(key)
        if baseline_value is None or current_value is None:
            deltas[key] = None
        else:
            deltas[key] = float(current_value - baseline_value)

    return deltas


def infer_split_sizes(loaders: Optional[Dict[str, Any]]) -> Dict[str, int]:
    if not loaders:
        return {}

    split_size_keys = {
        "train_loader": "train",
        "train_retain_loader": "train_retain",
        "train_forget_loader": "train_forget",
        "valid_retain_loader": "valid_retain",
        "valid_forget_loader": "valid_forget",
        "test_loader": "test",
    }

    split_sizes: Dict[str, int] = {}
    for loader_key, split_name in split_size_keys.items():
        loader = loaders.get(loader_key)
        dataset = getattr(loader, "dataset", None)
        if dataset is not None:
            split_sizes[split_name] = int(len(dataset))

    return split_sizes


def resolve_unlearning_artifact_paths(
    method: str,
    results_path: Optional[str] = None,
    check_path: Optional[str] = None,
    summary_path: Optional[str] = None,
    history_path: Optional[str] = None,
) -> Tuple[str, str]:
    if summary_path:
        summary_file = Path(summary_path)
    elif results_path:
        results_file = Path(results_path)
        summary_file = results_file.with_name(f"{results_file.stem}_summary.json")
    elif check_path:
        checkpoint_file = Path(check_path)
        summary_file = checkpoint_file.with_name(f"{checkpoint_file.stem}_summary.json")
    else:
        summary_file = Path("results") / f"{method}_unlearning_summary.json"

    if history_path:
        history_file = Path(history_path)
    else:
        history_file = summary_file.with_name(f"{summary_file.stem}_history.csv")

    return str(summary_file), str(history_file)


def save_unlearning_summary(summary_path: str, summary: Dict[str, Any]) -> str:
    summary_file = Path(summary_path)
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    with summary_file.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return str(summary_file)


def save_unlearning_history_csv(history_path: str, history_rows: Sequence[Dict[str, Any]]) -> str:
    history_file = Path(history_path)
    history_file.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["epoch", *ACCURACY_KEYS]
    with history_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in history_rows:
            writer.writerow({key: row.get(key) for key in fieldnames})

    return str(history_file)


def build_unlearning_summary(
    method: str,
    baseline_metrics: Dict[str, Any],
    final_metrics: Dict[str, Any],
    history_rows: Sequence[Dict[str, Any]],
    selected_metrics: Optional[Dict[str, Any]] = None,
    selected_epoch: Optional[int] = None,
    selection_strategy: str = "last_epoch",
    loaders: Optional[Dict[str, Any]] = None,
    run_config: Optional[Dict[str, Any]] = None,
    artifacts: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    normalized_history: List[Dict[str, Any]] = []
    for row in history_rows:
        normalized_row: Dict[str, Any] = {"epoch": int(row["epoch"])}
        normalized_row.update(normalize_accuracy_dict(row))
        normalized_history.append(normalized_row)

    final_epoch = int(normalized_history[-1]["epoch"]) if normalized_history else None
    normalized_final = normalize_accuracy_dict(final_metrics)
    normalized_selected = normalize_accuracy_dict(selected_metrics or final_metrics)
    normalized_baseline = normalize_accuracy_dict(baseline_metrics)

    if selected_epoch is None:
        selected_epoch = final_epoch

    summary = {
        "schema_version": 1,
        "method": str(method),
        "config": run_config or {},
        "selection": {
            "strategy": str(selection_strategy),
            "selected_epoch": int(selected_epoch) if selected_epoch is not None else None,
            "final_epoch": final_epoch,
        },
        "split_sizes": infer_split_sizes(loaders),
        "metrics": {
            "baseline": normalized_baseline,
            "final": normalized_final,
            "selected": normalized_selected,
        },
        "deltas_from_baseline": {
            "final": compute_metric_deltas(normalized_baseline, normalized_final),
            "selected": compute_metric_deltas(normalized_baseline, normalized_selected),
        },
        "history": normalized_history,
        "history_length": len(normalized_history),
        "artifacts": artifacts or {},
        "extra": extra or {},
    }

    return summary