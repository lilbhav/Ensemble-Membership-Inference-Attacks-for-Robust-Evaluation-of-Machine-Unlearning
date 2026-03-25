from __future__ import annotations

from pathlib import Path

from adapters.io_utils import run_subprocess


class MiaDisparityAdapter:
    def __init__(self, project_root: Path, bridge_script: Path) -> None:
        # Store stable paths used by every subprocess invocation
        self.project_root = project_root
        self.bridge_script = bridge_script

    def run_attack(
        self,
        dataset: str,
        base_seed: int,
        attack_name: str,
        attack_seed: int,
        target_name: str,
        split_file: Path,
        model_path: Path,
        output_csv: Path,
        data_root: Path,
        engine_repo: Path,
        attack_epochs: int,
        batch_size: int,
        device: str,
        unlearning_method: str,
        model_name: str,
    ) -> None:
        # Build bridge command line for one attack/target/model combination
        cmd = [
            "python",
            str(self.bridge_script),
            "--dataset",
            dataset,
            "--base-seed",
            str(base_seed),
            "--attack-name",
            attack_name,
            "--attack-seed",
            str(attack_seed),
            "--target-name",
            target_name,
            "--split-file",
            str(split_file),
            "--model-path",
            str(model_path),
            "--output-csv",
            str(output_csv),
            "--data-root",
            str(data_root),
            "--engine-repo",
            str(engine_repo),
            "--attack-epochs",
            str(attack_epochs),
            "--batch-size",
            str(batch_size),
            "--device",
            device,
            "--unlearning-method",
            unlearning_method,
            "--model-name",
            model_name,
        ]
        # Run from project root to keep path assumptions consistent
        run_subprocess(cmd, cwd=self.project_root)
