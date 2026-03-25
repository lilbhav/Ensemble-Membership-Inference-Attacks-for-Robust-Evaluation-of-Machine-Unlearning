from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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
        required_baseline_keys = ["epochs", "batch_size", "lr", "optimizer", "momentum"]
        missing = [k for k in required_baseline_keys if k not in training_cfg or training_cfg[k] is None]
        if missing:
            raise ValueError(f"Missing baseline config values: {missing}")

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
        if training_cfg.get("weight_decay") is not None:
            cmd += ["--weight-decay", str(training_cfg["weight_decay"])]
        if training_cfg.get("lr_scheduler"):
            cmd += ["--lr-scheduler", str(training_cfg["lr_scheduler"])]
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
        run_name: str | None = None,
    ) -> None:
        supported_methods = {"scrub", "ssd", "bad_teacher", "amnesiac"}
        if unlearning_method not in supported_methods:
            raise ValueError(
                f"Unsupported unlearning method '{unlearning_method}'. "
                f"Supported methods: {sorted(supported_methods)}"
            )

        required_unlearning_keys = ["epochs", "batch_size", "lr", "optimizer", "momentum"]
        missing = [k for k in required_unlearning_keys if k not in training_cfg or training_cfg[k] is None]
        if missing:
            raise ValueError(f"Missing unlearning config values: {missing}")

        method_cfg_root: dict[str, Any] | None = training_cfg.get("methods")
        if not isinstance(method_cfg_root, dict):
            raise ValueError(
                "Missing training.unlearning.methods config block. "
                "Define per-method dictionaries for scrub, ssd, bad_teacher, and amnesiac."
            )
        if unlearning_method not in method_cfg_root or method_cfg_root[unlearning_method] is None:
            raise ValueError(
                f"Missing method-specific config: training.unlearning.methods.{unlearning_method}"
            )
        method_cfg = method_cfg_root[unlearning_method]
        if not isinstance(method_cfg, dict):
            raise ValueError(
                f"training.unlearning.methods.{unlearning_method} must be a dictionary."
            )

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
            "--method-config-json",
            json.dumps(method_cfg),
        ]

        if run_name:
            cmd.extend(["--run-name", run_name])

        run_subprocess(cmd, cwd=self.project_root)
