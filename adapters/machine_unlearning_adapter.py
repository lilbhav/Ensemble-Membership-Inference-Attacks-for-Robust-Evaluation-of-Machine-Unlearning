from __future__ import annotations

from pathlib import Path

from adapters.io_utils import run_subprocess


class MachineUnlearningAdapter:
    def __init__(self, project_root: Path, bridge_script: Path) -> None:
        # Keep both locations so calls are independent of current working directory
        self.project_root = project_root
        self.bridge_script = bridge_script

    def run_baseline(
        self,
        dataset: str,
        seed: int,
        split_file: Path,
        model_out: Path,
        data_root: Path,
        engine_repo: Path,
        training_cfg: dict,
        device: str,
    ) -> None:
        # Build bridge command line for baseline training
        cmd = [
            "python",
            str(self.bridge_script),
            "--mode",
            "baseline",
            "--dataset",
            dataset,
            "--seed",
            str(seed),
            "--split-file",
            str(split_file),
            "--model-out",
            str(model_out),
            "--data-root",
            str(data_root),
            "--engine-repo",
            str(engine_repo),
            "--device",
            device,
            "--epochs",
            str(training_cfg["epochs"]),
            "--batch-size",
            str(training_cfg["batch_size"]),
            "--lr",
            str(training_cfg["lr"]),
            "--optimizer",
            str(training_cfg["optimizer"]),
            "--momentum",
            str(training_cfg["momentum"]),
        ]
        # Execute from repo root so relative imports/paths in bridge remain stable
        run_subprocess(cmd, cwd=self.project_root)

    def run_unlearning(
        self,
        dataset: str,
        seed: int,
        unlearning_method: str,
        split_file: Path,
        baseline_model_path: Path,
        model_out: Path,
        data_root: Path,
        engine_repo: Path,
        training_cfg: dict,
        device: str,
    ) -> None:
        # Build bridge command line for unlearning run
        cmd = [
            "python",
            str(self.bridge_script),
            "--mode",
            "unlearn",
            "--dataset",
            dataset,
            "--seed",
            str(seed),
            "--unlearning-method",
            unlearning_method,
            "--split-file",
            str(split_file),
            "--baseline-model",
            str(baseline_model_path),
            "--model-out",
            str(model_out),
            "--data-root",
            str(data_root),
            "--engine-repo",
            str(engine_repo),
            "--device",
            device,
            "--epochs",
            str(training_cfg["epochs"]),
            "--batch-size",
            str(training_cfg["batch_size"]),
            "--lr",
            str(training_cfg["lr"]),
            "--optimizer",
            str(training_cfg["optimizer"]),
            "--momentum",
            str(training_cfg["momentum"]),
        ]
        run_subprocess(cmd, cwd=self.project_root)
