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
EXCLUDED_METHODS = {"finetune", "fine_tune", "finetuning"}


def _method_name(path_name: str) -> str:
    return METHOD_ALIASES.get(path_name, path_name)


def _is_excluded_method(method: str) -> bool:
    return method.strip().lower() in EXCLUDED_METHODS


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
        if _is_excluded_method(method):
            continue
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


def _configure_plot_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.facecolor": "#f7f9fb",
            "axes.facecolor": "#ffffff",
            "axes.edgecolor": "#d5dde5",
            "axes.grid": True,
            "grid.color": "#dbe4ee",
            "grid.alpha": 0.8,
            "grid.linestyle": "-",
            "axes.titleweight": "bold",
            "axes.labelweight": "semibold",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
        }
    )


def _clip01(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return max(0.0, min(1.0, float(value)))


def _safe_ratio(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return float(numerator) / float(denominator)


def _build_unlearning_method_table(records: List[Dict[str, object]]) -> List[Dict[str, Optional[float]]]:
    table: List[Dict[str, Optional[float]]] = []

    for row in records:
        method = str(row.get("method"))
        baseline = row.get("baseline", {})
        selected = row.get("selected", {})
        if not isinstance(baseline, dict):
            baseline = {}
        if not isinstance(selected, dict):
            selected = {}

        baseline_tf = _to_float(baseline.get("tf_acc"))
        baseline_vr = _to_float(baseline.get("vr_acc"))
        baseline_test = _to_float(baseline.get("test_acc"))

        selected_tf = _to_float(selected.get("tf_acc"))
        selected_vr = _to_float(selected.get("vr_acc"))
        selected_test = _to_float(selected.get("test_acc"))

        forget_reduction = None
        if baseline_tf is not None and selected_tf is not None:
            forget_reduction = baseline_tf - selected_tf

        retain_drop = None
        if baseline_vr is not None and selected_vr is not None:
            retain_drop = baseline_vr - selected_vr

        test_drop = None
        if baseline_test is not None and selected_test is not None:
            test_drop = baseline_test - selected_test

        forget_effectiveness = _clip01(_safe_ratio(forget_reduction, baseline_tf))
        retain_preservation = _clip01(_safe_ratio(selected_vr, baseline_vr))
        test_preservation = _clip01(_safe_ratio(selected_test, baseline_test))

        balanced_score = None
        score_parts = [value for value in (forget_effectiveness, retain_preservation, test_preservation) if value is not None]
        if score_parts:
            balanced_score = float(sum(score_parts) / len(score_parts))

        table.append(
            {
                "method": method,
                "baseline_tf_acc": baseline_tf,
                "baseline_vr_acc": baseline_vr,
                "baseline_test_acc": baseline_test,
                "selected_tf_acc": selected_tf,
                "selected_vr_acc": selected_vr,
                "selected_test_acc": selected_test,
                "forget_reduction": forget_reduction,
                "retain_drop": retain_drop,
                "test_drop": test_drop,
                "forget_effectiveness": forget_effectiveness,
                "retain_preservation": retain_preservation,
                "test_preservation": test_preservation,
                "balanced_score": balanced_score,
            }
        )

    return table


def _plot_unlearning_selected_metrics(records: List[Dict[str, object]], out_dir: Path, dpi: int) -> Optional[Path]:
    table = _build_unlearning_method_table(records)
    if not table:
        return None

    methods = [str(row["method"]) for row in table]
    metric_specs = [
        ("tf_acc", "Forget Acc (lower better)", "#d1495b"),
        ("vr_acc", "Retain Acc", "#2c7fb8"),
        ("test_acc", "Test Acc", "#1b9e77"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(17, 6), sharey=True)
    x = list(range(len(methods)))
    width = 0.38

    for axis, (metric_key, title, color) in zip(axes, metric_specs):
        baseline_values: List[float] = []
        selected_values: List[float] = []
        for row in records:
            baseline = row.get("baseline", {})
            selected = row.get("selected", {})
            baseline_value = _to_float(baseline.get(metric_key)) if isinstance(baseline, dict) else None
            selected_value = _to_float(selected.get(metric_key)) if isinstance(selected, dict) else None
            baseline_values.append(float("nan") if baseline_value is None else baseline_value)
            selected_values.append(float("nan") if selected_value is None else selected_value)

        axis.bar([p - width / 2 for p in x], baseline_values, width=width, label="baseline", color="#dce6f2")
        axis.bar([p + width / 2 for p in x], selected_values, width=width, label="selected", color=color)
        axis.set_xticks(x)
        axis.set_xticklabels(methods, rotation=28, ha="right")
        axis.set_ylim(0.0, 1.02)
        axis.set_title(title)
        axis.grid(axis="y")

    axes[0].set_ylabel("Accuracy")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("Baseline vs Selected Performance", y=0.995)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.95), ncol=2)
    fig.tight_layout(rect=(0.01, 0.03, 0.99, 0.86))

    out_path = out_dir / "unlearning_selected_metrics_bar.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_unlearning_delta_heatmap(records: List[Dict[str, object]], out_dir: Path, dpi: int) -> Optional[Path]:
    table = _build_unlearning_method_table(records)
    if not table:
        return None

    metric_order = [
        ("forget_effectiveness", "Forget efficacy"),
        ("retain_preservation", "Retain preservation"),
        ("test_preservation", "Test preservation"),
        ("balanced_score", "Balanced score"),
    ]
    methods = [str(row["method"]) for row in table]

    matrix: List[List[float]] = []
    for row in table:
        values: List[float] = []
        for key, _ in metric_order:
            value = _to_float(row.get(key))
            values.append(float("nan") if value is None else value)
        matrix.append(values)

    fig, ax = plt.subplots(figsize=(10, 6))
    image = ax.imshow(matrix, aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Normalized score (higher is better)")

    ax.set_xticks(range(len(metric_order)))
    ax.set_xticklabels([label for _, label in metric_order], rotation=26, ha="right")
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods)
    ax.set_title("Method Quality Heatmap (Derived Metrics)")

    for i, row_values in enumerate(matrix):
        for j, value in enumerate(row_values):
            if value == value:
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8, color="#0f172a")

    fig.tight_layout(rect=(0.01, 0.02, 0.98, 0.98))
    out_path = out_dir / "unlearning_delta_heatmap.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_unlearning_forget_retain_scatter(records: List[Dict[str, object]], out_dir: Path, dpi: int) -> Optional[Path]:
    table = _build_unlearning_method_table(records)
    points: List[Tuple[str, float, float, float, float]] = []
    for row in table:
        retain_preservation = _to_float(row.get("retain_preservation"))
        forget_effectiveness = _to_float(row.get("forget_effectiveness"))
        balanced_score = _to_float(row.get("balanced_score"))
        test_preservation = _to_float(row.get("test_preservation"))
        if retain_preservation is None or forget_effectiveness is None:
            continue
        points.append(
            (
                str(row.get("method")),
                retain_preservation,
                forget_effectiveness,
                0.0 if balanced_score is None else balanced_score,
                0.0 if test_preservation is None else test_preservation,
            )
        )

    if not points:
        return None

    fig, ax = plt.subplots(figsize=(8.5, 6.5))

    ax.axvline(0.9, color="#94a3b8", linestyle="--", linewidth=1.2)
    ax.axhline(0.9, color="#94a3b8", linestyle="--", linewidth=1.2)

    for method, retain, forget, score, test_pres in points:
        marker_size = 120 + 220 * max(0.0, min(1.0, test_pres))
        color = "#1d4ed8" if score >= 0.85 else "#0f766e" if score >= 0.7 else "#b45309"
        ax.scatter([retain], [forget], s=marker_size, c=color, alpha=0.78, edgecolors="white", linewidths=1.2)
        ax.annotate(f"{method}\nscore={score:.2f}", (retain, forget), textcoords="offset points", xytext=(6, 6), fontsize=8)

    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Retain preservation = selected_vr / baseline_vr")
    ax.set_ylabel("Forget efficacy = (baseline_tf - selected_tf) / baseline_tf")
    ax.set_title("Unlearning Tradeoff Map (bubble size = test preservation)")
    ax.grid(alpha=0.35)
    fig.tight_layout(rect=(0.01, 0.02, 0.99, 0.98))

    out_path = out_dir / "unlearning_forget_retain_scatter.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_unlearning_history(records: List[Dict[str, object]], out_dir: Path, dpi: int) -> Optional[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharex=False, sharey=True)
    plotted_methods: Set[str] = set()

    for record in records:
        history = record.get("history", [])
        if not isinstance(history, list) or not history:
            continue

        method = str(record.get("method"))
        x_values = [int(_to_float(item.get("step")) or idx + 1) for idx, item in enumerate(history)]
        tf_values = [float("nan") if _to_float(item.get("tf_acc")) is None else float(_to_float(item.get("tf_acc"))) for item in history]
        vr_values = [float("nan") if _to_float(item.get("vr_acc")) is None else float(_to_float(item.get("vr_acc"))) for item in history]

        if any(value == value for value in tf_values):
            axes[0].plot(x_values, tf_values, marker="o", linewidth=1.8, label=method)
            plotted_methods.add(method)
        if any(value == value for value in vr_values):
            axes[1].plot(x_values, vr_values, marker="o", linewidth=1.8, label=method)

    if not plotted_methods:
        plt.close(fig)
        return None

    axes[0].set_title("Forget Accuracy Over Steps (lower is better)")
    axes[1].set_title("Retain Accuracy Over Steps")
    for axis in axes:
        axis.set_xlabel("Step")
        axis.set_ylim(0.0, 1.02)
        axis.grid(alpha=0.3)
    axes[0].set_ylabel("Accuracy")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.95), ncol=min(5, len(labels)))

    fig.suptitle("Unlearning Trajectory Focus View", y=0.995)
    fig.tight_layout(rect=(0.01, 0.03, 0.99, 0.88))

    out_path = out_dir / "unlearning_history_lines.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_unlearning_prf_macro(records: List[Dict[str, object]], out_dir: Path, dpi: int) -> Optional[Path]:
    table = _build_unlearning_method_table(records)
    if not table:
        return None

    ranked = sorted(table, key=lambda row: -1.0 if row.get("balanced_score") is None else float(row.get("balanced_score")))
    methods = [str(row["method"]) for row in ranked]
    scores = [float("nan") if row.get("balanced_score") is None else float(row.get("balanced_score")) for row in ranked]
    retain = [float("nan") if row.get("retain_preservation") is None else float(row.get("retain_preservation")) for row in ranked]
    forget = [float("nan") if row.get("forget_effectiveness") is None else float(row.get("forget_effectiveness")) for row in ranked]

    y = list(range(len(methods)))
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), sharey=True)

    axes[0].barh(y, scores, color="#1d4ed8", alpha=0.9)
    axes[0].set_xlim(0.0, 1.0)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(methods)
    axes[0].invert_yaxis()
    axes[0].set_title("Balanced Unlearning Score")
    axes[0].set_xlabel("Score")

    axes[1].barh([v + 0.18 for v in y], retain, height=0.34, color="#0f766e", label="retain preservation")
    axes[1].barh([v - 0.18 for v in y], forget, height=0.34, color="#b45309", label="forget efficacy")
    axes[1].set_xlim(0.0, 1.0)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(methods)
    axes[1].invert_yaxis()
    axes[1].set_title("Score Components")
    axes[1].set_xlabel("Normalized component")
    axes[1].legend(loc="lower right")

    fig.suptitle("Unlearning Leaderboard", y=0.995)
    fig.tight_layout(rect=(0.01, 0.03, 0.99, 0.93))

    out_path = out_dir / "unlearning_prf_macro_bar.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def write_unlearning_summary_csv(records: List[Dict[str, object]], out_dir: Path) -> Optional[Path]:
    if not records:
        return None

    derived_by_method: Dict[str, Dict[str, Optional[float]]] = {
        str(item.get("method")): item for item in _build_unlearning_method_table(records)
    }

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
        "baseline_test_acc",
        "forget_reduction",
        "retain_drop",
        "test_drop",
        "forget_effectiveness",
        "retain_preservation",
        "test_preservation",
        "balanced_score",
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
                    "baseline_test_acc": _to_float(baseline.get("test_acc")),
                    "forget_reduction": _to_float(derived_by_method.get(str(record.get("method")), {}).get("forget_reduction")),
                    "retain_drop": _to_float(derived_by_method.get(str(record.get("method")), {}).get("retain_drop")),
                    "test_drop": _to_float(derived_by_method.get(str(record.get("method")), {}).get("test_drop")),
                    "forget_effectiveness": _to_float(
                        derived_by_method.get(str(record.get("method")), {}).get("forget_effectiveness")
                    ),
                    "retain_preservation": _to_float(
                        derived_by_method.get(str(record.get("method")), {}).get("retain_preservation")
                    ),
                    "test_preservation": _to_float(
                        derived_by_method.get(str(record.get("method")), {}).get("test_preservation")
                    ),
                    "balanced_score": _to_float(derived_by_method.get(str(record.get("method")), {}).get("balanced_score")),
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

    fig, ax = plt.subplots(figsize=(12, 6.5))
    for idx, key in enumerate(metrics):
        values = [row.get(key) for row in utility_rows]
        values = [float(v) if v is not None else 0.0 for v in values]
        offset = [(pos + (idx - 1.5) * width) for pos in x]
        ax.bar(offset, values, width=width, label=key)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_title("Unlearning Utility Metrics")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(rect=(0.02, 0.05, 0.99, 0.98))

    out_path = out_dir / "utility_metrics_bar.png"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def _plot_attack_auc_bars(attack_rows: List[Dict[str, object]], out_dir: Path, dpi: int) -> Optional[Path]:
    if not attack_rows:
        return None

    labels = [f"{row['method']}:{row['attack']}" for row in attack_rows]
    aucs = [float(row["auc"]) for row in attack_rows]

    fig, ax = plt.subplots(figsize=(14, 6.5))
    ax.bar(range(len(labels)), aucs)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("AUC (higher = more leakage)")
    ax.set_title("MIA Attack AUC by Method")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(rect=(0.01, 0.10, 0.99, 0.98))

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
    fig.tight_layout(rect=(0.02, 0.02, 0.99, 0.98))

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

    fig, ax = plt.subplots(figsize=(10, 6.5))
    image = ax.imshow(matrix, aspect="auto")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Tradeoff score")

    ax.set_xticks(range(len(selection_values)))
    ax.set_yticks(range(len(dampening_values)))
    ax.set_xticklabels([str(v) for v in selection_values], rotation=40, ha="right")
    ax.set_yticklabels([str(v) for v in dampening_values])
    ax.set_xlabel("selection_weighting")
    ax.set_ylabel("dampening_constant")
    ax.set_title("SSD Sweep Tradeoff Heatmap")

    for i, row_vals in enumerate(matrix):
        for j, value in enumerate(row_vals):
            if value == value:
                ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=8)

    fig.tight_layout(rect=(0.01, 0.03, 0.99, 0.98))
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
        if _is_excluded_method(method):
            continue
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

    if not args.unlearning_dir.exists():
        raise FileNotFoundError(f"unlearning results directory not found: {args.unlearning_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _configure_plot_style()

    generated_paths: List[Path] = []

    unlearning_records = _collect_unlearning_records(args.unlearning_dir)

    # Build method-level utility rows only from unlearning summaries/history artifacts.
    utility_rows: List[Dict[str, object]] = []
    for record in unlearning_records:
        baseline = record.get("baseline", {})
        selected = record.get("selected", {})
        if not isinstance(baseline, dict):
            baseline = {}
        if not isinstance(selected, dict):
            selected = {}

        utility_rows.append(
            {
                "method": str(record.get("method")),
                "tr_acc": _to_float(selected.get("tr_acc")),
                "tf_acc": _to_float(selected.get("tf_acc")),
                "vr_acc": _to_float(selected.get("vr_acc")),
                "vf_acc": _to_float(selected.get("vf_acc")),
                "baseline_tf_acc": _to_float(baseline.get("tf_acc")),
                "baseline_vr_acc": _to_float(baseline.get("vr_acc")),
            }
        )

    for generated in (
        _plot_utility_bars(utility_rows, args.out_dir, args.dpi),
    ):
        if generated is not None:
            generated_paths.append(generated)

    summary_csv = write_summary_csv(
        utility_rows,
        {},
        {},
        args.privacy_lambda,
        args.out_dir,
    )
    generated_paths.append(summary_csv)

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
