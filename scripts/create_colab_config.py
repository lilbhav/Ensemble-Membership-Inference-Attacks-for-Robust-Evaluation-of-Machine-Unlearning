from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create a Colab-friendly config with Google Drive output paths")
    p.add_argument("--template", default="configs/experiment.yaml")
    p.add_argument("--out", default="configs/experiment_colab.yaml")
    p.add_argument("--repo-root", default="/content/Ensemble-Membership-Inference-Attacks-for-Robust-Evaluation-of-Machine-Unlearning")
    p.add_argument("--drive-root", default="/content/drive/MyDrive/unlearning_runs")
    p.add_argument("--dataset", default="Cifar10")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    template_path = Path(args.template)
    if not template_path.exists():
        raise FileNotFoundError(f"Missing template config: {template_path}")

    with template_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    repo_root = Path(args.repo_root)
    drive_root = Path(args.drive_root)

    # Point third-party engine repos to the Colab clone location.
    cfg["paths"]["machine_unlearning_repo"] = str(repo_root / "Third_Party_Code" / "MachineUnlearning")
    cfg["paths"]["mia_disparity_repo"] = str(repo_root / "Third_Party_Code" / "mia-disparity")

    # Persist data/results/checkpoints to Google Drive so runs survive runtime resets.
    cfg["paths"]["data_root"] = str(drive_root / "data")
    cfg["paths"]["results_root"] = str(drive_root / "results")

    # Default to one dataset for quicker single-run Colab workflows.
    cfg["experiment"]["datasets"] = [args.dataset]
    cfg["experiment"]["device"] = "cuda"

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    print(f"Wrote Colab config: {out_path}")


if __name__ == "__main__":
    main()
