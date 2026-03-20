#!/usr/bin/env python3
"""Aggregate unlearning summary metrics across seeds by method.

Reads per-run summary JSON files produced by experiment wrappers and computes
mean/std statistics for selected, baseline, and delta metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


DEFAULT_METRICS: Tuple[str, ...] = (
    "tr_acc",
    "tf_acc",
    "vr_acc",
    "vf_acc",
    "test_acc",
    "tr_f1_macro",
    "tf_f1_macro",
    "vr_f1_macro",
    "vf_f1_macro",
    "test_f1_macro",
)


def _to_float(value: Any) -> Optional[float]:
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
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    return float(math.sqrt(variance))


def _discover_summary_files(root: Path, recursive: bool) -> List[Path]:
    patterns = ("*_summary.json", "*unlearning_summary.json")
    candidates: List[Path] = []
    for pattern in patterns:
        if recursive:
            candidates.extend(root.glob(f"**/{pattern}"))
        else:
            candidates.extend(root.glob(pattern))

    unique = sorted({path.resolve() for path in candidates})
    filtered: List[Path] = []
    for path in unique:
        name = path.name.lower()
        if "attacks_summary" in name or name == "config.json":
            continue
        filtered.append(Path(path))
    return filtered


def _extract_seed(payload: Dict[str, Any]) -> Optional[int]:
    config = payload.get("config")
    if isinstance(config, dict) and "seed" in config:
        seed_value = _to_float(config.get("seed"))
        if seed_value is not None:
            return int(seed_value)
    return None


def _collect_run_records(summary_files: Iterable[Path]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []

    for path in summary_files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        if not isinstance(payload, dict):
            continue

        method = str(payload.get("method", "unknown"))
        metrics = payload.get("metrics")
        if not isinstance(metrics, dict):
            continue

        baseline = metrics.get("baseline") if isinstance(metrics.get("baseline"), dict) else {}
        selected = metrics.get("selected") if isinstance(metrics.get("selected"), dict) else {}
        final = metrics.get("final") if isinstance(metrics.get("final"), dict) else {}

        records.append(
            {
                "method": method,
                "seed": _extract_seed(payload),
                "summary_path": str(path),
                "baseline": baseline,
                "selected": selected,
                "final": final,
            }
        )

    return records


def _aggregate_method(records: List[Dict[str, Any]], metrics: Iterable[str]) -> Dict[str, Any]:
    output: Dict[str, Any] = {
        "num_runs": len(records),
        "seeds": sorted(
            [int(record["seed"]) for record in records if record.get("seed") is not None]
        ),
        "summary_paths": [record.get("summary_path") for record in records],
        "metrics": {},
    }

    for metric_name in metrics:
        baseline_values: List[float] = []
        selected_values: List[float] = []
        final_values: List[float] = []
        delta_selected_values: List[float] = []

        for record in records:
            baseline_value = _to_float(record.get("baseline", {}).get(metric_name))
            selected_value = _to_float(record.get("selected", {}).get(metric_name))
            final_value = _to_float(record.get("final", {}).get(metric_name))

            if baseline_value is not None:
                baseline_values.append(baseline_value)
            if selected_value is not None:
                selected_values.append(selected_value)
            if final_value is not None:
                final_values.append(final_value)
            if baseline_value is not None and selected_value is not None:
                delta_selected_values.append(selected_value - baseline_value)

        baseline_mean = _safe_mean(baseline_values)
        selected_mean = _safe_mean(selected_values)
        final_mean = _safe_mean(final_values)
        delta_selected_mean = _safe_mean(delta_selected_values)

        output["metrics"][metric_name] = {
            "baseline_mean": baseline_mean,
            "baseline_std": _safe_std(baseline_values, baseline_mean),
            "baseline_n": len(baseline_values),
            "selected_mean": selected_mean,
            "selected_std": _safe_std(selected_values, selected_mean),
            "selected_n": len(selected_values),
            "final_mean": final_mean,
            "final_std": _safe_std(final_values, final_mean),
            "final_n": len(final_values),
            "delta_selected_mean": delta_selected_mean,
            "delta_selected_std": _safe_std(delta_selected_values, delta_selected_mean),
            "delta_selected_n": len(delta_selected_values),
        }

    return output


def aggregate_runs(records: List[Dict[str, Any]], metrics: Iterable[str]) -> Dict[str, Any]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for record in records:
        method = str(record.get("method", "unknown"))
        grouped.setdefault(method, []).append(record)

    per_method: Dict[str, Any] = {}
    for method, method_records in sorted(grouped.items()):
        per_method[method] = _aggregate_method(method_records, metrics)

    return {
        "schema_version": 1,
        "num_total_runs": len(records),
        "num_methods": len(per_method),
        "methods": per_method,
    }


def write_json(summary: Dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return output_path


def write_csv(summary: Dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "method",
        "num_runs",
        "seeds",
        "metric",
        "baseline_mean",
        "baseline_std",
        "baseline_n",
        "selected_mean",
        "selected_std",
        "selected_n",
        "final_mean",
        "final_std",
        "final_n",
        "delta_selected_mean",
        "delta_selected_std",
        "delta_selected_n",
    ]

    methods = summary.get("methods", {})
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for method, method_payload in sorted(methods.items()):
            seeds = method_payload.get("seeds", [])
            metric_payload = method_payload.get("metrics", {})
            for metric_name, values in sorted(metric_payload.items()):
                writer.writerow(
                    {
                        "method": method,
                        "num_runs": method_payload.get("num_runs"),
                        "seeds": ",".join(str(seed) for seed in seeds),
                        "metric": metric_name,
                        "baseline_mean": values.get("baseline_mean"),
                        "baseline_std": values.get("baseline_std"),
                        "baseline_n": values.get("baseline_n"),
                        "selected_mean": values.get("selected_mean"),
                        "selected_std": values.get("selected_std"),
                        "selected_n": values.get("selected_n"),
                        "final_mean": values.get("final_mean"),
                        "final_std": values.get("final_std"),
                        "final_n": values.get("final_n"),
                        "delta_selected_mean": values.get("delta_selected_mean"),
                        "delta_selected_std": values.get("delta_selected_std"),
                        "delta_selected_n": values.get("delta_selected_n"),
                    }
                )

    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute cross-seed mean/std by method from run summaries")
    parser.add_argument(
        "--summary-root",
        type=Path,
        default=Path("results") / "unlearningresults",
        help="Directory containing per-run *_summary.json artifacts",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("results") / "plots" / "seed_summary.json",
        help="Output JSON file for aggregated summary",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("results") / "plots" / "seed_summary.csv",
        help="Output CSV file for aggregated summary",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=list(DEFAULT_METRICS),
        help="Metric keys to aggregate (default includes acc and macro-F1 metrics)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan summary-root for summary files",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.summary_root.exists():
        raise FileNotFoundError(f"Summary directory not found: {args.summary_root}")

    summary_files = _discover_summary_files(args.summary_root, recursive=bool(args.recursive))
    run_records = _collect_run_records(summary_files)
    aggregated = aggregate_runs(run_records, metrics=args.metrics)

    json_path = write_json(aggregated, args.out_json)
    csv_path = write_csv(aggregated, args.out_csv)

    print(f"Discovered run summaries: {len(run_records)}")
    print(f"Methods aggregated: {aggregated.get('num_methods', 0)}")
    print(f"Wrote JSON summary: {json_path}")
    print(f"Wrote CSV summary: {csv_path}")


if __name__ == "__main__":
    main()
