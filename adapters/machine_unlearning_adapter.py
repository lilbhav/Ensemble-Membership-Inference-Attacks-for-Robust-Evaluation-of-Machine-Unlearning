from __future__ import annotations

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
        run_name: str | None = None,
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

        if run_name:
            cmd.extend(["--run-name", run_name])

        if unlearning_method in {"scrub", "scrub_original", "scrub_teacher_loaded"}:
            scrub_cfg: dict[str, Any] = training_cfg.get("scrub", {})
            scrub_mode = scrub_cfg.get("mode")
            if unlearning_method == "scrub_original":
                scrub_mode = "original"
            elif unlearning_method == "scrub_teacher_loaded":
                scrub_mode = "teacher_loaded"

            if scrub_mode:
                cmd.extend(["--scrub-mode", str(scrub_mode)])

            scalar_mappings = {
                "epochs": "--scrub-epochs",
                "lr": "--scrub-lr",
                "distill_weight": "--scrub-distill-weight",
                "forget_loss_weight": "--scrub-forget-loss-weight",
                "maximize_epochs": "--scrub-maximize-epochs",
                "maximize_steps": "--scrub-maximize-steps",
                "minimize_steps": "--scrub-minimize-steps",
                "kd_temperature": "--scrub-kd-temperature",
                "weight_decay": "--scrub-weight-decay",
                "momentum": "--scrub-momentum",
                "lr_decay_rate": "--scrub-lr-decay-rate",
            }
            for key, flag in scalar_mappings.items():
                if key in scrub_cfg and scrub_cfg[key] is not None:
                    cmd.extend([flag, str(scrub_cfg[key])])

            if "lr_decay_epochs" in scrub_cfg:
                decay_epochs = ",".join(str(x) for x in scrub_cfg["lr_decay_epochs"])
                cmd.extend(["--scrub-lr-decay-epochs", decay_epochs])

        run_subprocess(cmd, cwd=self.project_root)
