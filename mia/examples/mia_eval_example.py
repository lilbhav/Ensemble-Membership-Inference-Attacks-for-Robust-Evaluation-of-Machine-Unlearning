"""
Example script: Running MIA attacks against an unlearning algorithm.

This script demonstrates how to:
1. Configure and run MIA attacks against an unlearned model
2. Evaluate individual attack performance
3. Generate ensemble predictions (union and voting)
4. Produce comprehensive evaluation reports

Usage:
    python mia_eval_example.py --config config.yaml --model-path model.pt
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Import our MIA framework
from mia_runner import MIARunner, MIARunnerConfig, AttackConfig
from attack_integrations import AttackFactory


def create_dummy_model(num_classes: int = 10) -> nn.Module:
    """Create a simple model for demonstration."""
    return nn.Sequential(
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, num_classes)
    )


def create_dummy_datasets(
    num_train: int = 1000,
    num_test: int = 500,
    input_dim: int = 784,
    num_classes: int = 10,
):
    """Create dummy datasets for demonstration."""
    # Create random data
    X_train = torch.randn(num_train, input_dim)
    y_train = torch.randint(0, num_classes, (num_train,))

    X_test = torch.randn(num_test, input_dim)
    y_test = torch.randint(0, num_classes, (num_test,))

    # Create datasets
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    return train_dataset, test_dataset, X_train, y_train, X_test, y_test


def main():
    """Run MIA evaluation example."""
    parser = argparse.ArgumentParser(
        description="Run MIA attacks on unlearning algorithms"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        help="Dataset name (default: cifar10)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="resnet18",
        help="Model architecture (default: resnet18)"
    )
    parser.add_argument(
        "--unlearning-method",
        type=str,
        default="scrub",
        help="Unlearning method to evaluate (default: scrub)"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to unlearned model checkpoint"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results/mia_eval",
        help="Output directory for results"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--attacks",
        type=str,
        nargs="+",
        default=["yeom", "shokri"],
        help="Attacks to run"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for attack execution"
    )

    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    logger.info("=" * 80)
    logger.info("MIA EVALUATION SCRIPT")
    logger.info("=" * 80)
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Unlearning method: {args.unlearning_method}")
    logger.info(f"Attacks: {args.attacks}")
    logger.info(f"Device: {args.device}")

    # Create or load model
    if args.model_path and os.path.exists(args.model_path):
        logger.info(f"Loading model from {args.model_path}")
        target_model = torch.load(args.model_path)
    else:
        logger.info("Creating dummy model for demonstration")
        target_model = create_dummy_model(num_classes=10)

    # Create datasets
    logger.info("Creating datasets...")
    train_dataset, test_dataset, X_train, y_train, X_test, y_test = create_dummy_datasets()

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Configure attacks
    attack_configs = [
        AttackConfig(
            name=attack,
            model_access="white_box",
            params={"batch_size": args.batch_size}
        )
        for attack in args.attacks
    ]

    # Create MIA runner configuration
    mia_config = MIARunnerConfig(
        dataset_name=args.dataset,
        model_architecture=args.model,
        unlearning_method=args.unlearning_method,
        attacks=attack_configs,
        device=args.device,
        seed=args.seed,
        output_dir=args.output_dir,
    )

    # Initialize runner
    logger.info("Initializing MIA runner...")
    runner = MIARunner(mia_config)

    # Run attacks
    logger.info("Running attacks...")
    attack_factory = AttackFactory()

    for attack_config in attack_configs:
        logger.info(f"Executing {attack_config.name} attack...")
        try:
            result = attack_factory.create_attack(
                attack_config=attack_config,
                target_model=target_model,
                train_dataloader=train_loader,
                test_dataloader=test_loader,
                device=args.device
            )
            runner.attack_results[attack_config.name] = result
            logger.info(f"✓ {attack_config.name} completed")
        except Exception as e:
            logger.error(f"✗ {attack_config.name} failed: {e}")

    # Create ground truth labels (1=member, 0=non-member)
    num_train = len(train_dataset)
    num_test = len(test_dataset)
    ground_truth = np.concatenate([
        np.ones(num_train, dtype=int),   # Members
        np.zeros(num_test, dtype=int)    # Non-members
    ])

    # Evaluate individual attacks
    logger.info("\n" + "="*80)
    logger.info("EVALUATING INDIVIDUAL ATTACKS")
    logger.info("="*80)

    metrics = runner.evaluate_attacks(ground_truth)

    for attack_name, attack_metrics in metrics.items():
        logger.info(f"\n{attack_name}:")
        for metric_name, metric_value in attack_metrics.items():
            logger.info(f"  {metric_name}: {metric_value:.4f}")

    # Generate ensemble predictions if multiple attacks
    if len(runner.attack_results) > 1:
        logger.info("\n" + "="*80)
        logger.info("ENSEMBLE PREDICTIONS")
        logger.info("="*80)

        # Union (OR) ensemble
        logger.info("\nUnion (OR) Ensemble:")
        union_pred, union_meta = runner.get_ensemble_predictions_union()
        union_result = runner.attack_results[list(runner.attack_results.keys())[0]]
        union_result.all_predictions = union_pred
        union_metrics = union_result.compute_metrics(ground_truth)
        for metric_name, metric_value in union_metrics.items():
            logger.info(f"  {metric_name}: {metric_value:.4f}")

        # k-of-M voting ensemble
        logger.info("\nk-of-M Voting Ensemble (k=2):")
        voting_pred, voting_meta = runner.get_ensemble_predictions_voting(k=2)
        voting_result = runner.attack_results[list(runner.attack_results.keys())[1]]
        voting_result.all_predictions = voting_pred
        voting_metrics = voting_result.compute_metrics(ground_truth)
        for metric_name, metric_value in voting_metrics.items():
            logger.info(f"  {metric_name}: {metric_value:.4f}")

    # Save results
    logger.info("\n" + "="*80)
    logger.info("SAVING RESULTS")
    logger.info("="*80)

    runner.save_results()

    # Generate report
    report = runner.generate_report(ground_truth)

    # Save report to file
    report_path = os.path.join(args.output_dir, "evaluation_report.txt")
    with open(report_path, "w") as f:
        f.write(report)

    logger.info(f"\nReport saved to {report_path}")
    logger.info("=" * 80)
    logger.info("MIA EVALUATION COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
