from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# Add project root to path so imports work regardless of cwd
sys.path.insert(0, str(Path(__file__).parent.parent))

from adapters.io_utils import ensure_dir, load_config, resolve_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate run outputs into compact summary tables")
    p.add_argument("--config", default="configs/experiment.yaml")
    return p.parse_args()


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    # 1) Load config and create aggregate output directory
    args = parse_args()
    cfg = load_config(args.config)
    results_root = resolve_path(cfg["paths"]["results_root"])

    aggregate_dir = ensure_dir(results_root / "aggregate")

    # 2) Collect model metric JSON files into one table
    model_metric_files = sorted(results_root.glob("models/**/*.metrics.json"))
    model_rows = []
    import json

    for p in model_metric_files:
        with p.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        model_rows.append(payload)

    if model_rows:
        with (aggregate_dir / "model_metrics.csv").open("w", newline="", encoding="utf-8") as f:
            fields = sorted({k for r in model_rows for k in r.keys()})
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(model_rows)

    # 3) Concatenate all per-run MIA prediction CSVs
    mia_rows = []
    for p in sorted(results_root.glob("mia/**/*.csv")):
        if p.name.endswith(".csv"):
            rows = read_csv(p)
            for idx, row in enumerate(rows, start=1):
                for key in ["split_mode", "target_class", "forget_count", "forget_fraction"]:
                    if key not in row or row[key] in {"", None}:
                        raise ValueError(
                            f"Missing required split metadata '{key}' in {p} at row {idx}. "
                            "Regenerate MIA outputs with targeted_random metadata."
                        )
            mia_rows.extend(rows)

    if mia_rows:
        with (aggregate_dir / "mia_predictions_all.csv").open("w", newline="", encoding="utf-8") as f:
            base_fields = [
                "sample_id",
                "true_membership",
                "split_name",
                "split_mode",
                "target_class",
                "forget_count",
                "forget_fraction",
                "model_name",
                "unlearning_method",
                "attack_name",
                "attack_seed",
                "score",
                "prediction",
                "dataset",
                "base_seed",
            ]
            # Keep canonical columns first, then append any new per-row fields emitted by newer MIA runs.
            extra_fields = sorted({k for row in mia_rows for k in row.keys() if k not in base_fields})
            fields = base_fields + extra_fields
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(mia_rows)

    # 4) Concatenate ensemble metric summaries
    ensemble_metric_rows = []
    for p in sorted(results_root.glob("ensemble/**/ensemble_metrics.csv")):
        ensemble_metric_rows.extend(read_csv(p))

    if ensemble_metric_rows:
        with (aggregate_dir / "ensemble_metrics_all.csv").open("w", newline="", encoding="utf-8") as f:
            fields = ["dataset", "base_seed", "unlearning_method", "target", "rule", "k", "m", "tpr", "fpr", "accuracy"]
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(ensemble_metric_rows)

    print(f"Saved aggregate outputs in: {aggregate_dir}")


if __name__ == "__main__":
    main()
