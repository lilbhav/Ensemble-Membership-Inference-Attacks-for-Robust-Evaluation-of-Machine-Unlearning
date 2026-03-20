#!/usr/bin/env python3
"""Generate summary plots for unlearning and MIA experiment logs.

This script parses text logs in the top-level `results/` directory and creates
figures for utility, privacy leakage, and hyperparameter sweeps.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt


UTILITY_KEYS = ("tr_acc", "tf_acc", "vr_acc", "vf_acc")
UNLEARNING_METRIC_KEYS = (
    "tr_acc",
    "tf_acc",
    "vr_acc",
    "vf_acc",
    "test_acc",
    "tr_precision_macro",
    "tf_precision_macro",
    "vr_precision_macro",
    "vf_precision_macro",
    "test_precision_macro",
    "tr_recall_macro",
    "tf_recall_macro",
    "vr_recall_macro",
    "vf_recall_macro",
    "test_recall_macro",
    "tr_f1_macro",
    "tf_f1_macro",
    "vr_f1_macro",
    "vf_f1_macro",
    "test_f1_macro",
    "tr_precision_weighted",
    "tf_precision_weighted",
    "vr_precision_weighted",
    "vf_precision_weighted",
    "test_precision_weighted",
    "tr_recall_weighted",
    "tf_recall_weighted",
    "vr_recall_weighted",
    "vf_recall_weighted",
    "test_recall_weighted",
    "tr_f1_weighted",
    "tf_f1_weighted",
    "vr_f1_weighted",
    "vf_f1_weighted",
    "test_f1_weighted",
)
METHOD_ALIASES = {
    "amnesiac_unlearned_model": "amnesiac",
    "bad_teacher_unlearned_model": "bad_teacher",
    "scrub_unlearned_model": "scrub",
    "ssd_unlearned_model": "ssd",
}


def _method_name(path_name: str) -> str:
    return METHOD_ALIASES.get(path_name, path_name)


def _extract_last_float(text: str, pattern: str) -> Optional[float]:
    matches = re.findall(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
    if not matches:
        return None
    return float(matches[-1])


def parse_utility_metrics(results_txt: Path) -> Dict[str, Optional[float]]:
    text = results_txt.read_text(encoding="utf-8", errors="ignore")
    metrics: Dict[str, Optional[float]] = {k: None for k in UTILITY_KEYS}

    for key in UTILITY_KEYS:
        # Captures patterns like "tr_acc: 0.8415" or "tr_acc = 0.8415".
        metrics[key] = _extract_last_float(text, rf"\b{re.escape(key)}\b\s*[:=]\s*([0-9]*\.?[0-9]+)")

    if metrics["tr_acc"] is None:
        metrics["tr_acc"] = _extract_last_float(
            text, r"Final\s+train\s+retain\s+acc\s*:\s*([0-9]*\.?[0-9]+)"
        )
    if metrics["vf_acc"] is None:
        metrics["vf_acc"] = _extract_last_float(
            text, r"Final\s+valid\s+forget\s+acc\s*:\s*([0-9]*\.?[0-9]+)"
        )

    return metrics


def parse_baseline_utility_metrics(results_txt: Path) -> Dict[str, Optional[float]]:
    """Parse baseline utility metrics needed for drop calculations."""
    text = results_txt.read_text(encoding="utf-8", errors="ignore")

    # Supports both formats:
    #  - baseline | tr_acc: ... tf_acc: ... vr_acc: ...
    #  - Baseline - tr_acc: ..., tf_acc: ..., vr_acc: ...
    baseline_tf = _extract_last_float(
        text,
        r"baseline[^\n]*\btf_acc\b\s*[:=]\s*([0-9]*\.?[0-9]+)",
    )
    baseline_vr = _extract_last_float(
        text,
        r"baseline[^\n]*\bvr_acc\b\s*[:=]\s*([0-9]*\.?[0-9]+)",
    )

    return {
        "baseline_tf_acc": baseline_tf,
        "baseline_vr_acc": baseline_vr,
    }


def parse_unlearning_summary(summary_path: Path) -> Optional[Dict[str, Optional[float]]]:
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    metrics = data.get("metrics")
    if not isinstance(metrics, dict):
        return None

    selected = metrics.get("selected")
    baseline = metrics.get("baseline")
    if not isinstance(selected, dict) or not isinstance(baseline, dict):
        return None

    payload: Dict[str, Optional[float]] = {
        "baseline_tf_acc": _to_float(baseline.get("tf_acc")),
        "baseline_vr_acc": _to_float(baseline.get("vr_acc")),
    }
    for key in UNLEARNING_METRIC_KEYS:
        payload[key] = _to_float(selected.get(key))
    return payload


def discover_unlearning_summary(method_dir: Path) -> Optional[Path]:
    excluded_names = {"attacks_summary.json", "config.json"}
    for candidate in sorted(method_dir.glob("*summary.json")):
        if candidate.name in excluded_names:
            continue
        parsed = parse_unlearning_summary(candidate)
        if parsed is not None:
            return candidate
    return None


def _safe_mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values) / len(values))


def _safe_std(values: List[float], mean_value: Optional[float] = None) -> Optional[float]:
    if not values:
        return None
    if mean_value is None:
        mean_value = _safe_mean(values)
    if mean_value is None:
        return None
    variance = sum((v - mean_value) ** 2 for v in values) / len(values)
    return float(variance ** 0.5)


def compute_method_attack_stats(
    attack_rows: List[Dict[str, object]],
) -> Dict[str, Dict[str, Optional[float]]]:
    """Compute AUC spread/stability and ensemble gain per method."""
    grouped: Dict[str, Dict[str, List[float]]] = {}

    for row in attack_rows:
        method = str(row.get("method"))
        attack = str(row.get("attack", "")).lower()
        auc = row.get("auc")
        if auc is None:
            continue
        auc_value = float(auc)

        if method not in grouped:
            grouped[method] = {
                "all": [],
                "single": [],
                "ensemble": [],
            }

        grouped[method]["all"].append(auc_value)
        if attack in {"union", "voting"}:
            grouped[method]["ensemble"].append(auc_value)
        else:
            grouped[method]["single"].append(auc_value)

    stats: Dict[str, Dict[str, Optional[float]]] = {}
    for method, values in grouped.items():
        all_aucs = values["all"]
        single_aucs = values["single"]
        ensemble_aucs = values["ensemble"]

        mean_auc = _safe_mean(all_aucs)
        std_auc = _safe_std(all_aucs, mean_auc)
        min_auc = float(min(all_aucs)) if all_aucs else None
        max_auc = float(max(all_aucs)) if all_aucs else None

        best_single_auc = float(max(single_aucs)) if single_aucs else None
        best_ensemble_auc = float(max(ensemble_aucs)) if ensemble_aucs else None
        ensemble_gain = None
        if best_single_auc is not None and best_ensemble_auc is not None:
            ensemble_gain = float(best_ensemble_auc - best_single_auc)

        stats[method] = {
            "auc_mean": mean_auc,
            "auc_std": std_auc,
            "auc_min": min_auc,
            "auc_max": max_auc,
            "best_single_attack_auc": best_single_auc,
            "best_ensemble_auc": best_ensemble_auc,
            "ensemble_gain": ensemble_gain,
        }

    return stats


def parse_attack_metrics(attack_txt: Path) -> Dict[str, Dict[str, float]]:
    """Parse attack metric blocks and keep the latest value for each attack."""
    text = attack_txt.read_text(encoding="utf-8", errors="ignore")
    rows = text.splitlines()

    metrics: Dict[str, Dict[str, float]] = {}
    current_attack: Optional[str] = None

    attack_header_re = re.compile(r"^\s*([a-zA-Z0-9_+\-]+):\s*$")
    value_re = re.compile(r"^\s*(auc|accuracy|tpr_at_fpr_0\.01|tpr_at_fpr_0\.001)\s*:\s*([0-9]*\.?[0-9]+)\s*$")

    for line in rows:
        normalized_line = line.strip().lower()
        if normalized_line.startswith("union") and "ensemble" in normalized_line and normalized_line.endswith(":"):
            current_attack = "union"
            metrics.setdefault(current_attack, {})
            continue
        if normalized_line.startswith("voting") and "ensemble" in normalized_line and normalized_line.endswith(":"):
            current_attack = "voting"
            metrics.setdefault(current_attack, {})
            continue

        header_match = attack_header_re.match(line)
        if header_match:
            name = header_match.group(1).lower()
            if name in {"shokri", "yeom", "lira", "calibration", "union", "voting"}:
                current_attack = name
                metrics.setdefault(current_attack, {})
            else:
                current_attack = None
            continue

        value_match = value_re.match(line)
        if value_match and current_attack is not None:
            key, value = value_match.groups()
            metrics[current_attack][key] = float(value)

    # Remove incomplete attack entries that do not have auc.
    filtered = {k: v for k, v in metrics.items() if "auc" in v}
    return filtered


def discover_sweep_csv(results_dir: Path) -> Optional[Path]:
    candidates = sorted(results_dir.glob("**/ssd_sweep_summary.csv"))
    return candidates[0] if candidates else None


def read_sweep_csv(csv_path: Path) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "dampening_constant": float(row["dampening_constant"]),
                    "selection_weighting": float(row["selection_weighting"]),
                    "score_mean": float(row["score_mean"]),
                    "forget_drop_mean": float(row["forget_drop_mean"]),
                    "retain_drop_mean": float(row["retain_drop_mean"]),
                }
            )
    return rows


def _to_float(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return float(stripped)
        except ValueError:
            return None
    return None


def _parse_unlearning_results_txt(results_txt: Path) -> Dict[str, object]:
    text = results_txt.read_text(encoding="utf-8", errors="ignore")
    line_re = re.compile(
        r"^\s*([^|]+?)\s*\|\s*"
        r"tr_acc\s*:\s*([0-9]*\.?[0-9]+)\s+"
        r"tf_acc\s*:\s*([0-9]*\.?[0-9]+)\s+"
        r"vr_acc\s*:\s*([0-9]*\.?[0-9]+)\s+"
        r"vf_acc\s*:\s*([0-9]*\.?[0-9]+)\s*$",
        flags=re.IGNORECASE,
    )

    baseline: Dict[str, float] = {}
    selected: Dict[str, float] = {}
    history: List[Dict[str, object]] = []
    step_index = 0

    for raw_line in text.splitlines():
        match = line_re.match(raw_line)
        if not match:
            continue

        label, tr_acc, tf_acc, vr_acc, vf_acc = match.groups()
        normalized = label.strip().lower()
        parsed = {
            "tr_acc": float(tr_acc),
            "tf_acc": float(tf_acc),
            "vr_acc": float(vr_acc),
            "vf_acc": float(vf_acc),
        }

        if "baseline" in normalized:
            baseline = parsed
            continue

        if any(key in normalized for key in ("selected_model", "selected_epoch", "final", "after_ssd")):
            selected = parsed

        if any(key in normalized for key in ("epoch", "forget_step", "retain_epoch")):
            step_match = re.search(r"(\d+)", normalized)
            step_index += 1
            history.append(
                {
                    "step": int(step_match.group(1)) if step_match else step_index,
                    "label": label.strip(),
                    **parsed,
                }
            )

    if not selected and history:
        selected = {
            "tr_acc": _to_float(history[-1].get("tr_acc")),
            "tf_acc": _to_float(history[-1].get("tf_acc")),
            "vr_acc": _to_float(history[-1].get("vr_acc")),
            "vf_acc": _to_float(history[-1].get("vf_acc")),
        }

    return {
        "baseline": baseline,
        "selected": selected,
        "history": history,
    }


def _read_history_csv(history_path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with history_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            step = _to_float(row.get("epoch"))
            parsed: Dict[str, object] = {
                "step": int(step) if step is not None else len(rows) + 1,
                "label": f"epoch {int(step) if step is not None else len(rows) + 1}",
                "tr_acc": _to_float(row.get("tr_acc")),
                "tf_acc": _to_float(row.get("tf_acc")),
                "vr_acc": _to_float(row.get("vr_acc")),
                "vf_acc": _to_float(row.get("vf_acc")),
                "test_acc": _to_float(row.get("test_acc")),
            }
            for key in UNLEARNING_METRIC_KEYS:
                if key not in parsed:
                    parsed[key] = _to_float(row.get(key))
            rows.append(parsed)
    return rows


def _collect_unlearning_records(unlearning_dir: Path) -> List[Dict[str, object]]:
    if not unlearning_dir.exists():
        return []

    records: List[Dict[str, object]] = []

    for results_txt in sorted(unlearning_dir.glob("*_results.txt")):
        method = results_txt.name[: -len("_results.txt")]
        text_payload = _parse_unlearning_results_txt(results_txt)
        baseline = dict(text_payload.get("baseline", {}))
        selected = dict(text_payload.get("selected", {}))
        history = list(text_payload.get("history", []))

        summary_path = unlearning_dir / f"{method}_results_summary.json"
        if summary_path.exists():
            parsed_summary = parse_unlearning_summary(summary_path)
            if parsed_summary is not None:
                baseline_tf = _to_float(parsed_summary.get("baseline_tf_acc"))
                baseline_vr = _to_float(parsed_summary.get("baseline_vr_acc"))
                if baseline_tf is not None:
                    baseline["tf_acc"] = baseline_tf
                if baseline_vr is not None:
                    baseline["vr_acc"] = baseline_vr

                for key in UNLEARNING_METRIC_KEYS:
                    value = _to_float(parsed_summary.get(key))
                    if value is not None:
                        selected[key] = value

            try:
                summary_data = json.loads(summary_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                summary_data = {}

            metrics = summary_data.get("metrics") if isinstance(summary_data, dict) else None
            if isinstance(metrics, dict):
                baseline_metrics = metrics.get("baseline")
                selected_metrics = metrics.get("selected")
                if isinstance(baseline_metrics, dict):
                    for key in UNLEARNING_METRIC_KEYS:
                        value = _to_float(baseline_metrics.get(key))
                        if value is not None:
                            baseline[key] = value
                if isinstance(selected_metrics, dict):
                    for key in UNLEARNING_METRIC_KEYS:
                        value = _to_float(selected_metrics.get(key))
                        if value is not None:
                            selected[key] = value

        history_csv = unlearning_dir / f"{method}_results_summary_history.csv"
        if history_csv.exists():
            csv_history = _read_history_csv(history_csv)
            if csv_history:
                history = csv_history

        records.append(
            {
                "method": method,
                "results_txt": results_txt.name,
                "summary_json_present": summary_path.exists(),
                "history_csv_present": history_csv.exists(),
                "baseline": baseline,
                "selected": selected,
                "history": history,
            }
        )

    return records


def _plot_unlearning_selected_metrics(records: List[Dict[str, object]], out_dir: Path, dpi: int) -> Optional[Path]:
    if not records:
        return None

    methods = [str(row["method"]) for row in records]
    metric_specs = [
        ("vr_acc", "selected vr_acc"),
        ("tf_acc", "selected tf_acc"),
        ("vf_acc", "selected vf_acc"),
        ("test_acc", "selected test_acc"),
    ]
    width = 0.2
    x = list(range(len(methods)))

    fig, ax = plt.subplots(figsize=(12, 6))
    for idx, (key, label) in enumerate(metric_specs):
        values: List[float] = []
        for row in records:
            selected = row.get("selected", {})
            if not isinstance(selected, dict):
                values.append(float("nan"))
                continue
            value = _to_float(selected.get(key))
            values.append(float("nan") if value is None else value)

        offset = [(pos + (idx - 1.5) * width) for pos in x]
        ax.bar(offset, values, width=width, label=label)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=20, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_title("Unlearning Results: Selected Metrics by Method")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    out_path = out_dir / "unlearning_selected_metrics_bar.png"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def _plot_unlearning_delta_heatmap(records: List[Dict[str, object]], out_dir: Path, dpi: int) -> Optional[Path]:
    if not records:
        return None

    metric_order = ["tr_acc", "tf_acc", "vr_acc", "vf_acc", "test_acc"]
    methods = [str(row["method"]) for row in records]

    matrix: List[List[float]] = []
    for row in records:
        baseline = row.get("baseline", {})
        selected = row.get("selected", {})
        if not isinstance(baseline, dict) or not isinstance(selected, dict):
            continue

        values: List[float] = []
        for key in metric_order:
            b_value = _to_float(baseline.get(key))
            s_value = _to_float(selected.get(key))
            if b_value is None or s_value is None:
                values.append(float("nan"))
            else:
                values.append(s_value - b_value)
        matrix.append(values)

    if not matrix:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    image = ax.imshow(matrix, aspect="auto", cmap="RdYlGn")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("selected - baseline")

    ax.set_xticks(range(len(metric_order)))
    ax.set_xticklabels(metric_order, rotation=25, ha="right")
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods)
    ax.set_title("Unlearning Metric Deltas from Baseline")

    for i, row_values in enumerate(matrix):
        for j, value in enumerate(row_values):
            if value == value:
                ax.text(j, i, f"{value:+.3f}", ha="center", va="center", fontsize=8)

    fig.tight_layout()
    out_path = out_dir / "unlearning_delta_heatmap.png"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def _plot_unlearning_forget_retain_scatter(records: List[Dict[str, object]], out_dir: Path, dpi: int) -> Optional[Path]:
    points: List[Tuple[str, float, float]] = []
    for row in records:
        selected = row.get("selected", {})
        if not isinstance(selected, dict):
            continue
        vr_acc = _to_float(selected.get("vr_acc"))
        tf_acc = _to_float(selected.get("tf_acc"))
        if vr_acc is None or tf_acc is None:
            continue
        points.append((str(row["method"]), vr_acc, tf_acc))

    if not points:
        return None

    fig, ax = plt.subplots(figsize=(8, 6))
    for method, vr_acc, tf_acc in points:
        ax.scatter([vr_acc], [tf_acc], s=90)
        ax.annotate(method, (vr_acc, tf_acc), textcoords="offset points", xytext=(5, 5))

    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Retain utility (selected vr_acc)")
    ax.set_ylabel("Forget retention (selected tf_acc, lower is better)")
    ax.set_title("Unlearning Tradeoff: Retain vs Forget Accuracy")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out_path = out_dir / "unlearning_forget_retain_scatter.png"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def _plot_unlearning_history(records: List[Dict[str, object]], out_dir: Path, dpi: int) -> Optional[Path]:
    metric_order = ["tr_acc", "tf_acc", "vr_acc", "vf_acc"]
    plotted_methods: Set[str] = set()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False, sharey=True)
    axes_flat = axes.flatten()

    for record in records:
        history = record.get("history", [])
        if not isinstance(history, list) or not history:
            continue

        method = str(record["method"])
        x_values = [int(_to_float(item.get("step")) or idx + 1) for idx, item in enumerate(history)]

        for axis, metric in zip(axes_flat, metric_order):
            y_values = [_to_float(item.get(metric)) for item in history]
            if not any(v is not None for v in y_values):
                continue
            axis.plot(
                x_values,
                [float("nan") if value is None else value for value in y_values],
                marker="o",
                linewidth=1.6,
                label=method,
            )
            axis.set_title(metric)
            axis.set_ylim(0, 1.0)
            axis.grid(alpha=0.25)
            plotted_methods.add(method)

    if not plotted_methods:
        plt.close(fig)
        return None

    for axis in axes_flat:
        axis.set_xlabel("Step")
        axis.set_ylabel("Accuracy")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(5, len(labels)))

    fig.suptitle("Unlearning Metric Trajectories from History Files", y=1.02)
    fig.tight_layout()

    out_path = out_dir / "unlearning_history_lines.png"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def _plot_unlearning_prf_macro(records: List[Dict[str, object]], out_dir: Path, dpi: int) -> Optional[Path]:
    if not records:
        return None

    methods = [str(row["method"]) for row in records]
    retain_specs = [
        ("vr_precision_macro", "retain precision"),
        ("vr_recall_macro", "retain recall"),
        ("vr_f1_macro", "retain f1"),
    ]
    forget_specs = [
        ("vf_precision_macro", "forget precision"),
        ("vf_recall_macro", "forget recall"),
        ("vf_f1_macro", "forget f1"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    x = list(range(len(methods)))
    width = 0.25

    for axis, specs, title in (
        (axes[0], retain_specs, "Selected Retain Macro PRF"),
        (axes[1], forget_specs, "Selected Forget Macro PRF"),
    ):
        for idx, (key, label) in enumerate(specs):
            values: List[float] = []
            for row in records:
                selected = row.get("selected", {})
                value = _to_float(selected.get(key)) if isinstance(selected, dict) else None
                values.append(float("nan") if value is None else value)

            offset = [pos + (idx - 1.0) * width for pos in x]
            axis.bar(offset, values, width=width, label=label)

        axis.set_xticks(x)
        axis.set_xticklabels(methods, rotation=20, ha="right")
        axis.set_ylim(0, 1.0)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)

    axes[0].set_ylabel("Score")
    axes[0].legend()
    axes[1].legend()
    fig.suptitle("Unlearning Precision/Recall/F1 Comparison (Macro)")
    fig.tight_layout()

    out_path = out_dir / "unlearning_prf_macro_bar.png"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def write_unlearning_summary_csv(records: List[Dict[str, object]], out_dir: Path) -> Optional[Path]:
    if not records:
        return None

    out_path = out_dir / "unlearning_plot_data_summary.csv"
    fieldnames = [
        "method",
        "results_txt",
        "summary_json_present",
        "history_csv_present",
        "baseline_tr_acc",
        "baseline_tf_acc",
        "baseline_vr_acc",
        "baseline_vf_acc",
        "baseline_test_acc",
        "selected_tr_acc",
        "selected_tf_acc",
        "selected_vr_acc",
        "selected_vf_acc",
        "selected_test_acc",
        "delta_tr_acc",
        "delta_tf_acc",
        "delta_vr_acc",
        "delta_vf_acc",
        "delta_test_acc",
        "selected_vr_precision_macro",
        "selected_vr_recall_macro",
        "selected_vr_f1_macro",
        "selected_vf_precision_macro",
        "selected_vf_recall_macro",
        "selected_vf_f1_macro",
        "selected_test_f1_macro",
        "selected_test_f1_weighted",
        "history_points",
    ]

    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            baseline = record.get("baseline", {})
            selected = record.get("selected", {})
            history = record.get("history", [])
            if not isinstance(baseline, dict):
                baseline = {}
            if not isinstance(selected, dict):
                selected = {}
            if not isinstance(history, list):
                history = []

            delta_values: Dict[str, Optional[float]] = {}
            for key in UNLEARNING_METRIC_KEYS:
                b_value = _to_float(baseline.get(key))
                s_value = _to_float(selected.get(key))
                if b_value is None or s_value is None:
                    delta_values[key] = None
                else:
                    delta_values[key] = s_value - b_value

            writer.writerow(
                {
                    "method": record.get("method"),
                    "results_txt": record.get("results_txt"),
                    "summary_json_present": record.get("summary_json_present"),
                    "history_csv_present": record.get("history_csv_present"),
                    "baseline_tr_acc": _to_float(baseline.get("tr_acc")),
                    "baseline_tf_acc": _to_float(baseline.get("tf_acc")),
                    "baseline_vr_acc": _to_float(baseline.get("vr_acc")),
                    "baseline_vf_acc": _to_float(baseline.get("vf_acc")),
                    "baseline_test_acc": _to_float(baseline.get("test_acc")),
                    "selected_tr_acc": _to_float(selected.get("tr_acc")),
                    "selected_tf_acc": _to_float(selected.get("tf_acc")),
                    "selected_vr_acc": _to_float(selected.get("vr_acc")),
                    "selected_vf_acc": _to_float(selected.get("vf_acc")),
                    "selected_test_acc": _to_float(selected.get("test_acc")),
                    "delta_tr_acc": delta_values.get("tr_acc"),
                    "delta_tf_acc": delta_values.get("tf_acc"),
                    "delta_vr_acc": delta_values.get("vr_acc"),
                    "delta_vf_acc": delta_values.get("vf_acc"),
                    "delta_test_acc": delta_values.get("test_acc"),
                    "selected_vr_precision_macro": _to_float(selected.get("vr_precision_macro")),
                    "selected_vr_recall_macro": _to_float(selected.get("vr_recall_macro")),
                    "selected_vr_f1_macro": _to_float(selected.get("vr_f1_macro")),
                    "selected_vf_precision_macro": _to_float(selected.get("vf_precision_macro")),
                    "selected_vf_recall_macro": _to_float(selected.get("vf_recall_macro")),
                    "selected_vf_f1_macro": _to_float(selected.get("vf_f1_macro")),
                    "selected_test_f1_macro": _to_float(selected.get("test_f1_macro")),
                    "selected_test_f1_weighted": _to_float(selected.get("test_f1_weighted")),
                    "history_points": len(history),
                }
            )

    return out_path


def _plot_utility_bars(utility_rows: List[Dict[str, object]], out_dir: Path, dpi: int) -> Optional[Path]:
    if not utility_rows:
        return None

    methods = [str(row["method"]) for row in utility_rows]
    metrics = ["tr_acc", "tf_acc", "vr_acc", "vf_acc"]
    width = 0.18
    x = list(range(len(methods)))

    fig, ax = plt.subplots(figsize=(11, 6))
    for idx, key in enumerate(metrics):
        values = [row.get(key) for row in utility_rows]
        values = [float(v) if v is not None else 0.0 for v in values]
        offset = [(pos + (idx - 1.5) * width) for pos in x]
        ax.bar(offset, values, width=width, label=key)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=20, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_title("Unlearning Utility Metrics")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    out_path = out_dir / "utility_metrics_bar.png"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def _plot_attack_auc_bars(attack_rows: List[Dict[str, object]], out_dir: Path, dpi: int) -> Optional[Path]:
    if not attack_rows:
        return None

    labels = [f"{row['method']}:{row['attack']}" for row in attack_rows]
    aucs = [float(row["auc"]) for row in attack_rows]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(range(len(labels)), aucs)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("AUC (higher = more leakage)")
    ax.set_title("MIA Attack AUC by Method")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    out_path = out_dir / "mia_auc_bar.png"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def _plot_privacy_utility_scatter(
    utility_rows: List[Dict[str, object]],
    max_auc_by_method: Dict[str, float],
    out_dir: Path,
    dpi: int,
) -> Optional[Path]:
    points: List[Tuple[str, float, float]] = []
    for row in utility_rows:
        method = str(row["method"])
        utility = row.get("vr_acc") if row.get("vr_acc") is not None else row.get("tr_acc")
        leakage = max_auc_by_method.get(method)
        if utility is None or leakage is None:
            continue
        points.append((method, float(utility), float(leakage)))

    if not points:
        return None

    fig, ax = plt.subplots(figsize=(8, 6))
    for method, utility, leakage in points:
        ax.scatter([utility], [leakage], s=80)
        ax.annotate(method, (utility, leakage), textcoords="offset points", xytext=(5, 5))

    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Utility (valid retain acc)")
    ax.set_ylabel("Privacy leakage (max attack AUC)")
    ax.set_title("Privacy-Utility Tradeoff")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out_path = out_dir / "privacy_utility_scatter.png"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def _plot_ssd_heatmap(sweep_rows: List[Dict[str, float]], out_dir: Path, dpi: int) -> Optional[Path]:
    if not sweep_rows:
        return None

    dampening_values = sorted({row["dampening_constant"] for row in sweep_rows})
    selection_values = sorted({row["selection_weighting"] for row in sweep_rows})

    matrix: List[List[float]] = []
    for dc in dampening_values:
        row_vals: List[float] = []
        for sw in selection_values:
            score = None
            for row in sweep_rows:
                if row["dampening_constant"] == dc and row["selection_weighting"] == sw:
                    score = row["score_mean"]
                    break
            row_vals.append(score if score is not None else float("nan"))
        matrix.append(row_vals)

    fig, ax = plt.subplots(figsize=(9, 6))
    image = ax.imshow(matrix, aspect="auto")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Tradeoff score")

    ax.set_xticks(range(len(selection_values)))
    ax.set_yticks(range(len(dampening_values)))
    ax.set_xticklabels([str(v) for v in selection_values], rotation=35, ha="right")
    ax.set_yticklabels([str(v) for v in dampening_values])
    ax.set_xlabel("selection_weighting")
    ax.set_ylabel("dampening_constant")
    ax.set_title("SSD Sweep Tradeoff Heatmap")

    for i, row_vals in enumerate(matrix):
        for j, value in enumerate(row_vals):
            if value == value:
                ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=8)

    fig.tight_layout()
    out_path = out_dir / "ssd_sweep_heatmap.png"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def write_summary_csv(
    utility_rows: List[Dict[str, object]],
    max_auc_by_method: Dict[str, float],
    attack_stats_by_method: Dict[str, Dict[str, Optional[float]]],
    privacy_lambda: float,
    out_dir: Path,
) -> Path:
    out_path = out_dir / "plot_data_summary.csv"
    fieldnames = [
        "method",
        "tr_acc",
        "tf_acc",
        "vr_acc",
        "vf_acc",
        "baseline_tf_acc",
        "baseline_vr_acc",
        "forget_drop",
        "retain_drop",
        "max_attack_auc",
        "privacy_utility_score",
        "auc_mean",
        "auc_std",
        "auc_min",
        "auc_max",
        "best_single_attack_auc",
        "best_ensemble_auc",
        "ensemble_gain",
    ]

    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in utility_rows:
            method = str(row["method"])
            baseline_tf = row.get("baseline_tf_acc")
            baseline_vr = row.get("baseline_vr_acc")
            final_tf = row.get("tf_acc")
            final_vr = row.get("vr_acc")

            forget_drop = None
            if baseline_tf is not None and final_tf is not None:
                forget_drop = float(baseline_tf) - float(final_tf)

            retain_drop = None
            if baseline_vr is not None and final_vr is not None:
                retain_drop = float(baseline_vr) - float(final_vr)

            max_attack_auc = max_auc_by_method.get(method)
            privacy_utility_score = None
            if max_attack_auc is not None and retain_drop is not None:
                privacy_utility_score = float(max_attack_auc + privacy_lambda * retain_drop)

            attack_stats = attack_stats_by_method.get(method, {})
            writer.writerow(
                {
                    "method": method,
                    "tr_acc": row.get("tr_acc"),
                    "tf_acc": row.get("tf_acc"),
                    "vr_acc": row.get("vr_acc"),
                    "vf_acc": row.get("vf_acc"),
                    "baseline_tf_acc": baseline_tf,
                    "baseline_vr_acc": baseline_vr,
                    "forget_drop": forget_drop,
                    "retain_drop": retain_drop,
                    "max_attack_auc": max_attack_auc,
                    "privacy_utility_score": privacy_utility_score,
                    "auc_mean": attack_stats.get("auc_mean"),
                    "auc_std": attack_stats.get("auc_std"),
                    "auc_min": attack_stats.get("auc_min"),
                    "auc_max": attack_stats.get("auc_max"),
                    "best_single_attack_auc": attack_stats.get("best_single_attack_auc"),
                    "best_ensemble_auc": attack_stats.get("best_ensemble_auc"),
                    "ensemble_gain": attack_stats.get("ensemble_gain"),
                }
            )

    return out_path


def build_records(results_dir: Path) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    utility_rows: List[Dict[str, object]] = []
    attack_rows: List[Dict[str, object]] = []

    for method_dir in sorted(results_dir.iterdir()):
        if not method_dir.is_dir():
            continue

        method = _method_name(method_dir.name)
        summary_path = discover_unlearning_summary(method_dir)
        results_txt = method_dir / "results.txt"
        if summary_path is not None:
            summary_metrics = parse_unlearning_summary(summary_path)
            if summary_metrics is not None:
                utility_rows.append(
                    {
                        "method": method,
                        **summary_metrics,
                    }
                )
        elif results_txt.exists():
            utility = parse_utility_metrics(results_txt)
            baseline_utility = parse_baseline_utility_metrics(results_txt)
            utility_rows.append(
                {
                    "method": method,
                    **utility,
                    **baseline_utility,
                }
            )

        merged_attacks: Dict[str, Dict[str, float]] = {}
        for attack_file in sorted(method_dir.glob("*.txt")):
            if attack_file.name == "results.txt":
                continue
            parsed = parse_attack_metrics(attack_file)
            for attack_name, metrics in parsed.items():
                merged_attacks[attack_name] = metrics

        for attack_name, metrics in merged_attacks.items():
            attack_rows.append(
                {
                    "method": method,
                    "attack": attack_name,
                    "auc": metrics.get("auc"),
                    "accuracy": metrics.get("accuracy"),
                    "tpr_at_fpr_0.01": metrics.get("tpr_at_fpr_0.01"),
                    "tpr_at_fpr_0.001": metrics.get("tpr_at_fpr_0.001"),
                }
            )

    return utility_rows, attack_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate visual summaries for unlearning and MIA logs")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory containing per-method experiment outputs",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results") / "plots",
        help="Directory for generated plots and summary CSV",
    )
    parser.add_argument(
        "--sweep-csv",
        type=Path,
        default=None,
        help="Optional path to ssd_sweep_summary.csv (auto-discovered if omitted)",
    )
    parser.add_argument("--dpi", type=int, default=180, help="DPI for saved PNG figures")
    parser.add_argument(
        "--unlearning-dir",
        type=Path,
        default=Path("results") / "unlearningresults",
        help="Directory containing unlearning result files (*_results.txt, *_summary.json, *_history.csv)",
    )
    parser.add_argument(
        "--privacy-lambda",
        type=float,
        default=1.0,
        help="Lambda for privacy_utility_score = max_attack_auc + lambda * retain_drop",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.results_dir.exists():
        raise FileNotFoundError(f"results directory not found: {args.results_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    utility_rows, attack_rows = build_records(args.results_dir)

    max_auc_by_method: Dict[str, float] = {}
    for row in attack_rows:
        method = str(row["method"])
        auc = row.get("auc")
        if auc is None:
            continue
        auc_value = float(auc)
        max_auc_by_method[method] = max(max_auc_by_method.get(method, 0.0), auc_value)

    attack_stats_by_method = compute_method_attack_stats(attack_rows)

    generated_paths: List[Path] = []

    for generated in (
        _plot_utility_bars(utility_rows, args.out_dir, args.dpi),
        _plot_attack_auc_bars(attack_rows, args.out_dir, args.dpi),
        _plot_privacy_utility_scatter(utility_rows, max_auc_by_method, args.out_dir, args.dpi),
    ):
        if generated is not None:
            generated_paths.append(generated)

    sweep_csv = args.sweep_csv if args.sweep_csv is not None else discover_sweep_csv(args.results_dir)
    if sweep_csv is not None and sweep_csv.exists():
        sweep_rows = read_sweep_csv(sweep_csv)
        heatmap = _plot_ssd_heatmap(sweep_rows, args.out_dir, args.dpi)
        if heatmap is not None:
            generated_paths.append(heatmap)

    summary_csv = write_summary_csv(
        utility_rows,
        max_auc_by_method,
        attack_stats_by_method,
        args.privacy_lambda,
        args.out_dir,
    )
    generated_paths.append(summary_csv)

    unlearning_records = _collect_unlearning_records(args.unlearning_dir)
    for generated in (
        _plot_unlearning_selected_metrics(unlearning_records, args.out_dir, args.dpi),
        _plot_unlearning_delta_heatmap(unlearning_records, args.out_dir, args.dpi),
        _plot_unlearning_forget_retain_scatter(unlearning_records, args.out_dir, args.dpi),
        _plot_unlearning_history(unlearning_records, args.out_dir, args.dpi),
        _plot_unlearning_prf_macro(unlearning_records, args.out_dir, args.dpi),
        write_unlearning_summary_csv(unlearning_records, args.out_dir),
    ):
        if generated is not None:
            generated_paths.append(generated)

    print("Generated files:")
    for path in generated_paths:
        print(f" - {path}")


if __name__ == "__main__":
    main()
