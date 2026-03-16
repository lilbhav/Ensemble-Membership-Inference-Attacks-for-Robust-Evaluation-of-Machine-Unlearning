"""
Membership Inference Attack (MIA) Runner for Machine Unlearning Evaluation.

This module provides a flexible framework for running various MIA attacks against
unlearning algorithms. It's designed to:

1. Execute individual attacks on unlearned models
2. Collect and aggregate attack predictions
3. Evaluate attack performance metrics (TPR, FPR, AUC)
4. Support future ensemble attack combinations (union, k-of-M voting)

The framework is architecture-agnostic and can work with different model types
and unlearning algorithms.
"""

import os
import json
import logging
import pickle
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict, field
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# Import utilities from reference code
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../Third_Party_Code/mia-disparity'))


@dataclass
class AttackConfig:
    """Configuration for a single attack."""
    name: str  # e.g., "shokri", "yeom", "lira", "reference"
    model_access: str = "white_box"  # white_box, black_box, gray_box, label_only
    params: Dict[str, Any] = field(default_factory=dict)  # Attack-specific parameters
    seed: int = 42


@dataclass
class MIARunnerConfig:
    """Configuration for the MIA runner."""
    dataset_name: str
    model_architecture: str
    unlearning_method: str
    attacks: List[AttackConfig] = field(default_factory=list)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    output_dir: str = "./results/mia"
    log_dir: str = "./logs/mia"


class AttackResult:
    """Container for results from a single attack."""

    def __init__(
        self,
        attack_name: str,
        attack_config: AttackConfig,
        member_scores: np.ndarray,
        nonmember_scores: np.ndarray,
        all_predictions: np.ndarray,  # Binary predictions for all samples
        member_indices: np.ndarray,  # Indices of member samples
    ):
        """
        Initialize attack result.

        Args:
            attack_name: Name of the attack for identification
            attack_config: Configuration used for this attack
            member_scores: Membership scores for member samples
            nonmember_scores: Membership scores for non-member samples
            all_predictions: Binary predictions for all samples (0=non-member, 1=member)
            member_indices: Original indices of member samples in dataset
        """
        self.attack_name = attack_name
        self.attack_config = attack_config
        self.member_scores = member_scores
        self.nonmember_scores = nonmember_scores
        self.all_predictions = all_predictions
        self.member_indices = member_indices

        # Metrics (populated after evaluation)
        self.metrics = {}

    def compute_metrics(self, ground_truth_labels: np.ndarray) -> Dict[str, float]:
        """
        Compute evaluation metrics (TPR, FPR, AUC).

        Args:
            ground_truth_labels: Binary labels (1=member, 0=non-member)

        Returns:
            Dictionary of metrics
        """
        from sklearn.metrics import roc_auc_score, roc_curve

        if len(np.unique(ground_truth_labels)) < 2:
            logging.warning("Ground truth has only one class; metrics cannot be computed.")
            return {}

        # Combine all scores for AUC calculation
        # Order must match ground_truth_labels construction in run_mia_experiment:
        # [members, non-members]
        all_scores = np.concatenate([self.member_scores, self.nonmember_scores])

        # Compute AUC
        auc = roc_auc_score(ground_truth_labels, all_scores)

        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(ground_truth_labels, all_scores)

        # Compute TPR at target FPR with interpolation instead of nearest-point snapping.
        # Nearest-point can under-report when ROC has coarse steps under class imbalance.
        def _interp_tpr_at_fpr(target_fpr: float) -> float:
            target_fpr = float(np.clip(target_fpr, 0.0, 1.0))

            # Keep monotone envelope in case of repeated FPR values.
            fpr_unique, first_idx = np.unique(fpr, return_index=True)
            tpr_unique = tpr[first_idx]
            tpr_unique = np.maximum.accumulate(tpr_unique)

            return float(np.interp(target_fpr, fpr_unique, tpr_unique))

        tpr_at_fpr_001 = _interp_tpr_at_fpr(0.01)
        tpr_at_fpr_0001 = _interp_tpr_at_fpr(0.001)

        # Useful for interpreting low-FPR metrics under extreme imbalance.
        num_negative = int(np.sum(ground_truth_labels == 0))
        min_nonzero_fpr = (1.0 / num_negative) if num_negative > 0 else 1.0

        # Compute accuracy
        accuracy = np.mean(self.all_predictions == ground_truth_labels)

        # Store metrics
        self.metrics = {
            "auc": float(auc),
            "tpr_at_fpr_0.01": float(tpr_at_fpr_001),
            "tpr_at_fpr_0.001": float(tpr_at_fpr_0001),
            "accuracy": float(accuracy),
            "min_nonzero_fpr": float(min_nonzero_fpr),
        }

        return self.metrics

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "attack_name": self.attack_name,
            "attack_config": asdict(self.attack_config),
            "metrics": self.metrics,
            "member_scores_mean": float(np.mean(self.member_scores)),
            "member_scores_std": float(np.std(self.member_scores)),
            "nonmember_scores_mean": float(np.mean(self.nonmember_scores)),
            "nonmember_scores_std": float(np.std(self.nonmember_scores)),
        }

    def save(self, path: str) -> None:
        """Save attack result to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "AttackResult":
        """Load attack result from disk."""
        with open(path, "rb") as f:
            return pickle.load(f)


class MIARunner:
    """
    Main orchestrator for running MIA attacks against unlearning algorithms.

    This class:
    1. Manages attack execution
    2. Collects predictions from multiple attacks
    3. Provides infrastructure for ensemble methods (future)
    4. Evaluates and reports results
    """

    def __init__(self, config: MIARunnerConfig):
        """
        Initialize the MIA runner.

        Args:
            config: Configuration for the runner
        """
        self.config = config
        self._setup_logging()
        self._create_output_dirs()

        # Store all attack results for ensemble operations
        self.attack_results: Dict[str, AttackResult] = {}

        self.logger.info(f"MIA Runner initialized for {config.dataset_name}")
        self.logger.info(f"Unlearning method: {config.unlearning_method}")
        self.logger.info(f"Model architecture: {config.model_architecture}")

    def _setup_logging(self) -> None:
        """Configure logging."""
        os.makedirs(self.config.log_dir, exist_ok=True)

        log_file = os.path.join(
            self.config.log_dir,
            f"mia_{self.config.unlearning_method}_{self.config.dataset_name}.log",
        )

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # File handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            "%(message)s"
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def _create_output_dirs(self) -> None:
        """Create output directories."""
        os.makedirs(self.config.output_dir, exist_ok=True)
        os.makedirs(self.config.log_dir, exist_ok=True)

    def run_attack(
        self,
        target_model: torch.nn.Module,
        attack_config: AttackConfig,
        train_data: Dataset,
        test_data: Dataset,
        member_indices: Optional[np.ndarray] = None,
    ) -> AttackResult:
        """
        Execute a single attack against the target model.

        Args:
            target_model: The unlearned model to attack
            attack_config: Configuration for this attack
            train_data: Training dataset used by original model
            test_data: Test/validation dataset
            member_indices: Indices of samples known to be in training set.
                          If None, assumes first half of train_data are members.

        Returns:
            AttackResult containing predictions and evaluation metrics
        """
        self.logger.info(f"Running {attack_config.name} attack...")
        self.logger.info(f"  Model access: {attack_config.model_access}")
        self.logger.info(f"  Parameters: {attack_config.params}")

        # TODO: Implement attack dispatch based on attack_config.name
        # This will be populated with actual attack implementations
        
        # Placeholder: return dummy result
        member_scores = np.random.uniform(0.5, 1.0, len(train_data))
        nonmember_scores = np.random.uniform(0.0, 0.5, len(test_data))
        all_predictions = np.concatenate([
            np.ones(len(train_data), dtype=int),  # Predicted members
            np.zeros(len(test_data), dtype=int),  # Predicted non-members
        ])

        if member_indices is None:
            member_indices = np.arange(len(train_data))

        result = AttackResult(
            attack_name=attack_config.name,
            attack_config=attack_config,
            member_scores=member_scores,
            nonmember_scores=nonmember_scores,
            all_predictions=all_predictions,
            member_indices=member_indices,
        )

        self.logger.info(f"  Completed {attack_config.name} attack")
        return result

    def run_all_attacks(
        self,
        target_model: torch.nn.Module,
        train_data: Dataset,
        test_data: Dataset,
        member_indices: Optional[np.ndarray] = None,
    ) -> Dict[str, AttackResult]:
        """
        Execute all configured attacks.

        Args:
            target_model: The unlearned model to attack
            train_data: Training dataset
            test_data: Test dataset
            member_indices: Indices of members (if None, first half of train_data)

        Returns:
            Dictionary mapping attack names to AttackResult objects
        """
        self.logger.info(f"Starting attack execution for {len(self.config.attacks)} attacks...")

        for attack_config in self.config.attacks:
            result = self.run_attack(
                target_model=target_model,
                attack_config=attack_config,
                train_data=train_data,
                test_data=test_data,
                member_indices=member_indices,
            )
            self.attack_results[attack_config.name] = result

        self.logger.info(f"Completed all {len(self.attack_results)} attacks")
        return self.attack_results

    def evaluate_attacks(
        self, ground_truth_labels: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all attack results.

        Args:
            ground_truth_labels: Binary labels (1=member, 0=non-member)

        Returns:
            Dictionary mapping attack names to their metrics
        """
        self.logger.info("Evaluating all attacks...")

        all_metrics = {}
        for attack_name, result in self.attack_results.items():
            metrics = result.compute_metrics(ground_truth_labels)
            all_metrics[attack_name] = metrics
            self.logger.info(f"{attack_name}: {metrics}")

        return all_metrics

    def get_ensemble_predictions_union(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Get ensemble predictions using Union (OR) logic.

        A sample is predicted as member if ANY attack flags it as member.

        Returns:
            Tuple of (ensemble_predictions, metadata)
        """
        if not self.attack_results:
            raise ValueError("No attacks have been run yet")

        # Stack all predictions
        predictions_list = [
            result.all_predictions for result in self.attack_results.values()
        ]
        predictions_array = np.stack(predictions_list, axis=0)

        # Union: sample is member if any attack says it's member
        ensemble_predictions = np.max(predictions_array, axis=0).astype(int)

        metadata = {
            "method": "union",
            "num_attacks": len(self.attack_results),
            "attack_names": list(self.attack_results.keys()),
        }

        return ensemble_predictions, metadata

    def get_ensemble_predictions_voting(
        self, k: int
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Get ensemble predictions using k-of-M voting.

        A sample is predicted as member if at least k attacks flag it as member.

        Args:
            k: Minimum number of attacks that must agree for member prediction

        Returns:
            Tuple of (ensemble_predictions, metadata)
        """
        if not self.attack_results:
            raise ValueError("No attacks have been run yet")

        num_attacks = len(self.attack_results)
        if k > num_attacks:
            raise ValueError(f"k ({k}) cannot be greater than number of attacks ({num_attacks})")

        # Stack all predictions
        predictions_list = [
            result.all_predictions for result in self.attack_results.values()
        ]
        predictions_array = np.stack(predictions_list, axis=0)

        # Count votes per sample
        vote_counts = np.sum(predictions_array, axis=0)

        # Sample is member if at least k attacks agree
        ensemble_predictions = (vote_counts >= k).astype(int)

        metadata = {
            "method": "k-of-M voting",
            "k": k,
            "num_attacks": num_attacks,
            "attack_names": list(self.attack_results.keys()),
        }

        return ensemble_predictions, metadata

    def save_results(self) -> None:
        """Save all attack results and metadata."""
        self.logger.info(f"Saving results to {self.config.output_dir}...")

        # Save individual attack results
        attacks_dir = os.path.join(self.config.output_dir, "attacks")
        os.makedirs(attacks_dir, exist_ok=True)

        summary = {}
        for attack_name, result in self.attack_results.items():
            # Save picklized attack result
            result_path = os.path.join(attacks_dir, f"{attack_name}_result.pkl")
            result.save(result_path)

            # Add summary
            summary[attack_name] = result.to_dict()

        # Save summary to JSON
        summary_path = os.path.join(self.config.output_dir, "attacks_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        # Save configuration
        config_path = os.path.join(self.config.output_dir, "config.json")
        
        # Helper function to make objects JSON-serializable
        def make_serializable(obj):
            if isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [make_serializable(item) for item in obj]
            elif hasattr(obj, '__dict__'):  # Convert dataclass/object to dict recursively
                return make_serializable(vars(obj))
            else:
                # Convert non-serializable types to string
                return str(obj)
        
        config_dict = {
            "dataset": self.config.dataset_name,
            "model": self.config.model_architecture,
            "unlearning_method": self.config.unlearning_method,
            "attacks": [asdict(a) for a in self.config.attacks],
            "device": str(self.config.device),
            "seed": self.config.seed,
        }
        
        # Make config dict fully JSON-serializable
        config_dict = make_serializable(config_dict)
        
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)

        self.logger.info(f"Results saved to {self.config.output_dir}")

    def generate_report(self, ground_truth_labels: np.ndarray) -> str:
        """
        Generate a comprehensive report of the MIA evaluation.

        Args:
            ground_truth_labels: Binary labels for evaluation

        Returns:
            Formatted report string
        """
        metrics = self.evaluate_attacks(ground_truth_labels)

        report = []
        report.append("\n" + "="*80)
        report.append("MEMBERSHIP INFERENCE ATTACK EVALUATION REPORT")
        report.append("="*80)
        report.append(f"Dataset: {self.config.dataset_name}")
        report.append(f"Model: {self.config.model_architecture}")
        report.append(f"Unlearning Method: {self.config.unlearning_method}")
        report.append(f"Date: {datetime.now().isoformat(timespec='seconds')}")
        report.append("")

        report.append("INDIVIDUAL ATTACK RESULTS:")
        report.append("-"*80)
        for attack_name, attack_metrics in metrics.items():
            report.append(f"\n{attack_name}:")
            for metric_name, metric_value in attack_metrics.items():
                report.append(f"  {metric_name}: {metric_value:.4f}")

        report.append("\n" + "="*80)

        report_str = "\n".join(report)
        self.logger.info(report_str)
        return report_str

    def __repr__(self) -> str:
        return (
            f"MIARunner(dataset={self.config.dataset_name}, "
            f"method={self.config.unlearning_method}, "
            f"attacks={[a.name for a in self.config.attacks]})"
        )


class AttackDispatcher:
    """
    Dispatcher for instantiating attacks based on configuration.

    This class will be extended with implementations of various attacks
    as they are integrated.
    """

    # Registry of available attacks
    AVAILABLE_ATTACKS = {
        "shokri": None,  # To be populated with actual class
        "yeom": None,
        "lira": None,
        "reference": None,
        "calibration": None,
        "augmented": None,
    }

    @classmethod
    def get_attack(
        cls,
        attack_config: AttackConfig,
        target_model: torch.nn.Module,
        device: str,
    ):
        """
        Get an attack instance based on configuration.

        Args:
            attack_config: Configuration for the attack
            target_model: Target model to attack
            device: Device to run on

        Returns:
            Instantiated attack object
        """
        attack_name = attack_config.name.lower()

        if attack_name not in cls.AVAILABLE_ATTACKS:
            raise ValueError(
                f"Unknown attack: {attack_name}. "
                f"Available attacks: {list(cls.AVAILABLE_ATTACKS.keys())}"
            )

        # TODO: Implement actual attack instantiation
        # For now, raise NotImplementedError
        raise NotImplementedError(
            f"Attack '{attack_name}' not yet integrated. "
            "Implementation in progress."
        )

    @classmethod
    def register_attack(cls, name: str, attack_class) -> None:
        """Register a new attack class."""
        cls.AVAILABLE_ATTACKS[name.lower()] = attack_class

    @classmethod
    def list_available_attacks(cls) -> List[str]:
        """List all available attacks."""
        return [name for name, cls in cls.AVAILABLE_ATTACKS.items()]


if __name__ == "__main__":
    # Example usage
    config = MIARunnerConfig(
        dataset_name="cifar10",
        model_architecture="resnet18",
        unlearning_method="scrub",
        attacks=[
            AttackConfig(name="shokri", params={"num_shadow_models": 10}),
            AttackConfig(name="yeom", params={}),
            AttackConfig(name="lira", params={"num_runs": 32}),
        ],
    )

    runner = MIARunner(config)
    print(runner)
    print("Available attacks:", AttackDispatcher.list_available_attacks())
