from __future__ import annotations
"""Adapter for MIA execution via the mia_disparity bridge.

This class converts framework-level attack parameters into a stable CLI contract
for external/mia_disparity_bridge.py.
"""

from pathlib import Path

from adapters.io_utils import run_subprocess


class MiaDisparityAdapter:
    """Thin wrapper that launches one MIA attack run per invocation."""

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
        num_shadow_models: int = 10,
        target_fpr: float = 0.01,
    ) -> None:
        """Run one (dataset, seed, method, attack, target) MIA evaluation.

        The bridge owns data loading, attack execution, score orientation, and
        output schema. This adapter focuses on argument normalization and launch.
        """
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
            "--num-shadow-models",
            str(num_shadow_models),
            "--target-fpr",
            str(target_fpr),
        ]

        # Run from project root to keep path assumptions consistent
        run_subprocess(cmd, cwd=self.project_root)
