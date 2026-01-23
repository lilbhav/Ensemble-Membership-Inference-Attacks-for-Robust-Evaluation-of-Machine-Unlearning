from miae.unlearning.base import UnlearningMethod
from miae.utils.path_utils import add_third_party_to_path
add_third_party_to_path()

import subprocess
import os


class FineTuneUnlearning(UnlearningMethod):
    """
    Wrapper for FT baseline unlearning using third-party baselines.py script.
    """

    def __init__(self, config):
        super().__init__(config)
        self.script_path = config.get("script_path", "third_party/baselines.py")

    def run(self, model_checkpoint_path, dataset_name, output_dir):
        """
        Run fine-tuning unlearning via subprocess call to third-party script.
        """

        cmd = [
            "python", self.script_path,
            "--dataset", dataset_name,
            "--model_checkpoints", model_checkpoint_path,
            "--unlearn_method", "gradient_ascent",  # or random_label, GA+KL, etc.
            "--output_path", output_dir,
            "--num_epochs", str(self.config.get("num_epochs", 1)),
            "--lr", str(self.config.get("lr", 1e-4)),
            "--train_batch_size", str(self.config.get("batch_size", 32)),
            "--eval_batch_size", str(self.config.get("batch_size", 32)),
        ]

        print("Running FT unlearning with command:")
        print(" ".join(cmd))

        subprocess.run(cmd, check=True)

        return output_dir
