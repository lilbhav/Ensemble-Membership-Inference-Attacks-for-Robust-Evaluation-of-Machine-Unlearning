"""
Unified MIA Experiment Runner

Run complete MIA evaluation experiments from config files.
Supports any combination of attacks and unlearning algorithms.

Usage:
    python run_mia_experiment.py --config configs/scrub_experiment.yaml
    python run_mia_experiment.py --config configs/experiment.yaml --output results/my_experiment
    python run_mia_experiment.py --config configs/finetune_experiment.yaml --attacks calibration yeom
"""

import os
import sys
import yaml
import json
import logging
import argparse
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
from torch.utils.data import DataLoader

# Add paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from miae.mia_runner import MIARunner, MIARunnerConfig, AttackConfig
from miae.attack_integrations import AttackFactory
from data.loaders import load_dataset, get_num_classes
from utils.splits import create_retain_forget_split


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_logging(log_dir: str, experiment_name: str) -> logging.Logger:
    """Setup logging for the experiment."""
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # File handler
    log_file = os.path.join(log_dir, f"{experiment_name}.log")
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


def load_model(config: Dict[str, Any], device: str) -> torch.nn.Module:
    """Load or create model based on config."""
    logger = logging.getLogger(__name__)
    
    model_cfg = config['model']
    checkpoint_path = model_cfg.get('checkpoint_path')
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        logger.info(f"Loading model from {checkpoint_path}")
        model = torch.load(checkpoint_path, map_location=device)
    else:
        logger.warning(f"Model checkpoint not found at {checkpoint_path}")
        logger.warning("Using random model initialization for demonstration")
        
        import torchvision.models as models
        
        arch = model_cfg.get('architecture', 'resnet18')
        num_classes = get_num_classes(config['dataset']['name'])
        
        if arch == 'resnet18':
            model = models.resnet18(pretrained=False)
            model.conv1 = torch.nn.Conv2d(
                3, 64, kernel_size=3, stride=1, padding=1, bias=False
            )
            model.maxpool = torch.nn.Identity()
            model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        else:
            raise ValueError(f"Unsupported architecture: {arch}")
    
    return model.to(device).eval()


def load_unlearned_model(config: Dict[str, Any], device: str) -> Optional[torch.nn.Module]:
    """Try to load pre-unlearned model, or return None to use original."""
    logger = logging.getLogger(__name__)
    
    unlearning_cfg = config.get('unlearning', {})
    method = unlearning_cfg.get('method', 'scrub')
    
    # Try to load pre-computed unlearned model
    dataset_name = config['dataset']['name']
    model_arch = config['model']['architecture']
    
    possible_paths = [
        f"./checkpoints/{method}_unlearned_model.pt",
        f"./results/{method}_unlearned_model/{method}_unlearned_model.pt",
        unlearning_cfg.get('checkpoint_path'),
    ]
    
    for path in possible_paths:
        if path and os.path.exists(path):
            logger.info(f"Loading unlearned model from {path}")
            return torch.load(path, map_location=device).to(device).eval()
    
    logger.warning(f"No pre-unlearned model found for {method}")
    logger.warning("You must provide an unlearned model or train one first")
    return None


def prepare_data(config: Dict[str, Any]) -> tuple:
    """Prepare train/test dataloaders."""
    logger = logging.getLogger(__name__)
    
    logger.info("Loading dataset...")
    
    dataset_cfg = config['dataset']
    dataset_name = dataset_cfg['name']
    forget_fraction = dataset_cfg.get('forget_fraction', 0.2)
    batch_size = config.get('batch_size', 64)
    
    # Load dataset - load_dataset() returns single dataset, not tuple
    train_data = load_dataset(dataset_name, train=True)
    test_data = load_dataset(dataset_name, train=False)
    
    # Create split
    forget_indices, retain_indices = create_retain_forget_split(
        train_data,
        forget_fraction=forget_fraction,
        seed=config.get('seed', 42)
    )
    
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"  Training samples: {len(train_data)}")
    logger.info(f"  Test samples: {len(test_data)}")
    logger.info(f"  Forget set: {len(forget_indices)} ({forget_fraction*100:.0f}%)")
    logger.info(f"  Retain set: {len(retain_indices)} ({(1-forget_fraction)*100:.0f}%)")
    
    # For MIA: members (training data) and non-members (test data)
    member_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
    nonmember_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    return member_loader, nonmember_loader, train_data, test_data


def create_attack_configs(config: Dict[str, Any], 
                         override_attacks: Optional[List[str]] = None) -> List[AttackConfig]:
    """Create attack configurations from config file or overrides."""
    logger = logging.getLogger(__name__)
    
    mia_cfg = config.get('mia', {})
    attack_defs = mia_cfg.get('attacks', [])
    
    if override_attacks:
        # Command-line override: just names, use defaults
        logger.info(f"Using command-line attack override: {override_attacks}")
        attack_configs = [
            AttackConfig(
                name=attack,
                model_access="white_box",
                params={},
                seed=config.get('seed', 42)
            )
            for attack in override_attacks
        ]
    else:
        # Use attacks from config file
        attack_configs = [
            AttackConfig(
                name=attack_def['name'],
                model_access=attack_def.get('model_access', 'white_box'),
                params=attack_def.get('params', {}),
                seed=config.get('seed', 42)
            )
            for attack_def in attack_defs
        ]
    
    logger.info(f"Attacks to run: {[a.name for a in attack_configs]}")
    return attack_configs


def run_mia_experiment(config_path: str,
                      output_dir: Optional[str] = None,
                      attacks: Optional[List[str]] = None,
                      model: Optional[torch.nn.Module] = None,
                      unlearned_model: Optional[torch.nn.Module] = None):
    """
    Run complete MIA experiment from config.
    
    Args:
        config_path: Path to YAML config file
        output_dir: Optional override for output directory
        attacks: Optional list of attack names to override config
        model: Optional pre-loaded original model
        unlearned_model: Optional pre-loaded unlearned model
    
    Returns:
        Dictionary with results
    """
    
    # Load config
    config = load_config(config_path)
    experiment_name = config.get('experiment', {}).get('name', 'mia_experiment')
    
    # Setup logging
    log_dir = output_dir or config.get('output', {}).get('log_dir', './logs/mia')
    logger = setup_logging(log_dir, experiment_name)
    
    logger.info("="*80)
    logger.info(f"MIA EXPERIMENT: {experiment_name}")
    logger.info("="*80)
    
    # Device
    device = torch.device(config.get('experiment', {}).get('device', 'cuda'))
    logger.info(f"Device: {device}")
    
    # Prepare data
    member_loader, nonmember_loader, train_data, test_data = prepare_data(config)
    
    # Load models if not provided
    if model is None:
        model = load_model(config, device)
    if unlearned_model is None:
        unlearned_model = load_unlearned_model(config, device)
        if unlearned_model is None:
            logger.warning("Using original model as unlearned model")
            unlearned_model = model
    
    # Create attack configs
    attack_configs = create_attack_configs(config, override_attacks=attacks)
    
    # Create MIA runner config
    output_path = output_dir or config.get('output', {}).get('save_dir', './results/mia')
    
    mia_config = MIARunnerConfig(
        dataset_name=config['dataset']['name'],
        model_architecture=config['model'].get('architecture', 'resnet18'),
        unlearning_method=config.get('unlearning', {}).get('method', 'unknown'),
        attacks=attack_configs,
        device=device,
        seed=config.get('seed', 42),
        output_dir=output_path,
        log_dir=log_dir,
    )
    
    logger.info(f"Output directory: {output_path}")
    
    # Run MIA
    logger.info("\n" + "="*80)
    logger.info("EXECUTING MIA ATTACKS")
    logger.info("="*80)
    
    runner = MIARunner(mia_config)
    factory = AttackFactory()
    
    for i, attack_config in enumerate(attack_configs, 1):
        logger.info(f"\n[{i}/{len(attack_configs)}] Running {attack_config.name}...")
        
        try:
            result = factory.create_attack(
                attack_config=attack_config,
                target_model=unlearned_model,
                train_dataloader=member_loader,
                test_dataloader=nonmember_loader,
                device=device,
            )
            runner.attack_results[attack_config.name] = result
            logger.info(f"  ✓ {attack_config.name} completed")
        except Exception as e:
            logger.error(f"  ✗ {attack_config.name} failed: {e}")
    
    if not runner.attack_results:
        logger.error("No attacks completed successfully!")
        return None
    
    # Evaluate
    logger.info("\n" + "="*80)
    logger.info("EVALUATION")
    logger.info("="*80)
    
    num_train = len(train_data)
    num_test = len(test_data)
    ground_truth = np.concatenate([
        np.ones(num_train),
        np.zeros(num_test)
    ])
    
    metrics = runner.evaluate_attacks(ground_truth)
    
    logger.info("\nAttack Performance:")
    for attack_name, attack_metrics in metrics.items():
        logger.info(f"\n{attack_name}:")
        for metric_name, metric_value in attack_metrics.items():
            logger.info(f"  {metric_name}: {metric_value:.4f}")
    
    # Save results
    logger.info("\n" + "="*80)
    logger.info("SAVING RESULTS")
    logger.info("="*80)
    
    runner.save_results()
    report = runner.generate_report(ground_truth)
    
    # Save report to file
    report_path = os.path.join(output_path, "evaluation_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    
    logger.info(f"Report saved to: {report_path}")
    
    # Ensemble results (if multiple attacks)
    if len(runner.attack_results) > 1:
        logger.info("\n" + "="*80)
        logger.info("ENSEMBLE RESULTS")
        logger.info("="*80)
        
        # Union ensemble
        union_pred, _ = runner.get_ensemble_predictions_union()
        temp_result = list(runner.attack_results.values())[0]
        temp_result.all_predictions = union_pred
        union_metrics = temp_result.compute_metrics(ground_truth)
        
        logger.info("\nUnion (OR) Ensemble:")
        for metric_name, metric_value in union_metrics.items():
            logger.info(f"  {metric_name}: {metric_value:.4f}")
        
        # Voting ensemble
        voting_pred, _ = runner.get_ensemble_predictions_voting(k=2)
        temp_result2 = list(runner.attack_results.values())[1]
        temp_result2.all_predictions = voting_pred
        voting_metrics = temp_result2.compute_metrics(ground_truth)
        
        logger.info("\nVoting (k=2) Ensemble:")
        for metric_name, metric_value in voting_metrics.items():
            logger.info(f"  {metric_name}: {metric_value:.4f}")
    
    logger.info("\n" + "="*80)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("="*80)
    
    return {
        'config': mia_config,
        'runner': runner,
        'metrics': metrics,
        'ground_truth': ground_truth,
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run unified MIA experiment from config file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with config file defaults
  python run_mia_experiment.py --config configs/scrub_experiment.yaml
  
  # Override output directory
  python run_mia_experiment.py --config configs/scrub_experiment.yaml --output results/my_run
  
  # Override attacks (command-line takes precedence)
  python run_mia_experiment.py --config configs/scrub_experiment.yaml --attacks calibration yeom
  
  # Run all unlearning methods on same dataset
  python run_mia_experiment.py --config configs/finetune_experiment.yaml -o results/finetune
  python run_mia_experiment.py --config configs/scrub_experiment.yaml -o results/scrub
  python run_mia_experiment.py --config configs/ssd_experiment.yaml -o results/ssd
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to YAML config file'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Override output directory from config'
    )
    parser.add_argument(
        '--attacks',
        type=str,
        nargs='+',
        default=None,
        help='Override attacks to run (from config): calibration yeom augmentation etc.'
    )
    
    args = parser.parse_args()
    
    # Validate config file exists
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)
    
    # Run experiment
    run_mia_experiment(
        config_path=args.config,
        output_dir=args.output,
        attacks=args.attacks,
    )


if __name__ == "__main__":
    main()
