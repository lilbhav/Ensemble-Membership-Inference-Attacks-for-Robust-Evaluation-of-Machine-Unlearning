import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


SPLIT_PREFIXES: Tuple[str, ...] = ("tr", "tf", "vr", "vf", "test")
METRIC_SUFFIXES: Tuple[str, ...] = (
    "acc",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "precision_weighted",
    "recall_weighted",
    "f1_weighted",
)
METRIC_KEYS: Tuple[str, ...] = tuple(
    f"{split}_{metric}"
    for split in SPLIT_PREFIXES
    for metric in METRIC_SUFFIXES
)
ACCURACY_KEYS: Tuple[str, ...] = tuple(f"{split}_acc" for split in SPLIT_PREFIXES)


def normalize_accuracy_dict(metrics: Optional[Dict[str, Any]]) -> Dict[str, float]:
    normalized: Dict[str, float] = {}
    if not metrics:
        return normalized

    for key in METRIC_KEYS:
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


def make_run_tag(seed: Optional[int] = None, timestamp: Optional[str] = None) -> str:
    """Build a stable run tag for artifact filenames.

    When a seed is provided, use it as the deterministic tag. Otherwise fall back
    to a UTC timestamp so repeated runs do not overwrite each other.
    """
    if seed is not None:
        return f"seed{int(seed)}"
    return timestamp or datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def to_serializable_dict(config: Optional[Any]) -> Dict[str, Any]:
    """Convert config objects (dict/dataclass/namespace) to JSON-safe dicts."""
    if config is None:
        return {}

    if isinstance(config, dict):
        source: Dict[str, Any] = dict(config)
    elif hasattr(config, "__dict__"):
        source = dict(vars(config))
    else:
        return {"value": str(config)}

    serialized: Dict[str, Any] = {}
    for key, value in source.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            serialized[str(key)] = value
        elif isinstance(value, (list, tuple)):
            normalized_list = []
            for item in value:
                if isinstance(item, (str, int, float, bool)) or item is None:
                    normalized_list.append(item)
                else:
                    normalized_list.append(str(item))
            serialized[str(key)] = normalized_list
        elif isinstance(value, dict):
            nested: Dict[str, Any] = {}
            for nested_key, nested_value in value.items():
                if isinstance(nested_value, (str, int, float, bool)) or nested_value is None:
                    nested[str(nested_key)] = nested_value
                else:
                    nested[str(nested_key)] = str(nested_value)
            serialized[str(key)] = nested
        else:
            serialized[str(key)] = str(value)

    return serialized


def resolve_unlearning_artifact_paths(
    method: str,
    results_path: Optional[str] = None,
    check_path: Optional[str] = None,
    summary_path: Optional[str] = None,
    history_path: Optional[str] = None,
    run_tag: Optional[str] = None,
) -> Tuple[str, str]:
    if summary_path:
        summary_file = Path(summary_path)
    elif run_tag and results_path:
        results_file = Path(results_path)
        summary_file = results_file.with_name(f"{results_file.stem}_{run_tag}_summary.json")
    elif run_tag and check_path:
        checkpoint_file = Path(check_path)
        summary_file = checkpoint_file.with_name(f"{checkpoint_file.stem}_{run_tag}_summary.json")
    elif run_tag:
        summary_file = Path("results") / f"{method}_{run_tag}_unlearning_summary.json"
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

    fieldnames = ["epoch", *METRIC_KEYS]
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


def select_unlearning_checkpoint(
    candidates: Sequence[Dict[str, Any]],
    baseline_metrics: Dict[str, Any],
    max_valid_retain_acc_drop: Optional[float] = 0.08,
    max_test_acc_drop: Optional[float] = 0.06,
    min_valid_forget_acc_drop: float = 0.20,
    min_train_forget_acc_drop: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Select a checkpoint using guardrails that balance utility and forgetting.

    Selection order:
    1) Prefer checkpoints satisfying all configured constraints.
    2) If none satisfy all constraints, keep utility constraints and relax forgetting floor.
    3) If still none, use all checkpoints.

    Tie-break key favors lower vf_acc, then higher vr_acc, then higher test_acc.
    """
    if not candidates:
        raise ValueError("No checkpoint candidates were provided for selection")

    baseline_vr = float(baseline_metrics.get("vr_acc", 0.0))
    baseline_vf = float(baseline_metrics.get("vf_acc", 0.0))
    baseline_tf = float(baseline_metrics.get("tf_acc", 0.0))
    baseline_test = float(baseline_metrics.get("test_acc", 0.0))

    def _passes_utility(candidate: Dict[str, Any]) -> bool:
        acc = candidate["acc"]
        if max_valid_retain_acc_drop is not None:
            vr_drop = baseline_vr - float(acc.get("vr_acc", 0.0))
            if vr_drop > float(max_valid_retain_acc_drop):
                return False
        if max_test_acc_drop is not None and "test_acc" in acc:
            test_drop = baseline_test - float(acc.get("test_acc", 0.0))
            if test_drop > float(max_test_acc_drop):
                return False
        return True

    def _passes_forgetting(candidate: Dict[str, Any]) -> bool:
        acc = candidate["acc"]
        vf_drop = baseline_vf - float(acc.get("vf_acc", 0.0))
        if vf_drop < float(min_valid_forget_acc_drop):
            return False
        if min_train_forget_acc_drop is not None:
            tf_drop = baseline_tf - float(acc.get("tf_acc", 0.0))
            if tf_drop < float(min_train_forget_acc_drop):
                return False
        return True

    strict = [c for c in candidates if _passes_utility(c) and _passes_forgetting(c)]
    utility_only = [c for c in candidates if _passes_utility(c)]

    selected_pool: Sequence[Dict[str, Any]]
    fallback_reason = "all_constraints"
    if strict:
        selected_pool = strict
    elif utility_only:
        selected_pool = utility_only
        fallback_reason = "utility_only_relaxed_forgetting"
    else:
        selected_pool = list(candidates)
        fallback_reason = "all_candidates_relaxed_all_constraints"

    best = min(
        selected_pool,
        key=lambda candidate: (
            float(candidate["acc"].get("vf_acc", 1.0)),
            -float(candidate["acc"].get("vr_acc", 0.0)),
            -float(candidate["acc"].get("test_acc", 0.0)),
            int(candidate["epoch"]),
        ),
    )

    return {
        "candidate": best,
        "pool_size": int(len(selected_pool)),
        "total_candidates": int(len(candidates)),
        "fallback_reason": fallback_reason,
        "constraints": {
            "max_valid_retain_acc_drop": max_valid_retain_acc_drop,
            "max_test_acc_drop": max_test_acc_drop,
            "min_valid_forget_acc_drop": min_valid_forget_acc_drop,
            "min_train_forget_acc_drop": min_train_forget_acc_drop,
        },
    }
