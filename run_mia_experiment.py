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
import logging
import argparse
import random
import torch
import numpy as np
from typing import Dict, Any, Optional, List
from torch.utils.data import DataLoader

# Add paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mia.mia_runner import MIARunner, MIARunnerConfig, AttackConfig, AttackResult
from mia.attack_integrations import AttackFactory
from data.loaders import load_dataset, get_num_classes
from utils.splits import create_retain_forget_split, load_split


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
        "%(message)s"
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


def _get_seed(config: Dict[str, Any]) -> int:
    """Get experiment seed from nested or flat config formats."""
    experiment_cfg = config.get('experiment', {})
    return int(experiment_cfg.get('seed', config.get('seed', 42)))


def _set_global_determinism(seed: int) -> None:
    """Set deterministic seeds for reproducible splits and dataloading."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _get_split_dir(config: Dict[str, Any]) -> str:
    """Resolve retain/forget split directory from config.

    Priority:
      1) top-level split_dir
      2) dataset.split_dir
      3) ./data/splits
    """
    dataset_cfg = config.get('dataset', {}) if isinstance(config.get('dataset', {}), dict) else {}
    return str(config.get('split_dir') or dataset_cfg.get('split_dir') or "./data/splits")


def create_model(architecture: str, dataset_name: str, device: str) -> torch.nn.Module:
    """Create a fresh model with given architecture and number of classes."""
    import torchvision.models as models
    
    num_classes = get_num_classes(dataset_name)
    
    if architecture == 'resnet18':
        model = models.resnet18(pretrained=False)
        model.conv1 = torch.nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        model.maxpool = torch.nn.Identity()
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")
    
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
            checkpoint = torch.load(path, map_location=device)
            
            # Check if it's a state_dict (from SCRUB) or a full model
            if isinstance(checkpoint, dict) and not hasattr(checkpoint, 'to'):
                # It's a state_dict, need to create model first
                logger.info("Checkpoint is a state_dict; creating model and loading weights...")
                model = create_model(model_arch, dataset_name, device)
                model.load_state_dict(checkpoint)
                return model.eval()
            else:
                # It's a full model object
                return checkpoint.to(device).eval()
    
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
    seed = _get_seed(config)
    split_dir = _get_split_dir(config)
    
    # Load dataset - load_dataset() returns single dataset, not tuple
    train_data = load_dataset(dataset_name, train=True)
    test_data = load_dataset(dataset_name, train=False)
    
    # Try to load existing splits first (from SCRUB unlearning)
    if os.path.exists(os.path.join(split_dir, "forget_idx.npy")) and os.path.exists(os.path.join(split_dir, "retain_idx.npy")):
        logger.info(f"Loading existing retain/forget splits from disk ({split_dir})...")
        retain_data, forget_data = load_split(train_data, split_dir)

        expected_forget = int(len(train_data) * forget_fraction)
        expected_retain = len(train_data) - expected_forget
        if len(forget_data) != expected_forget or len(retain_data) != expected_retain:
            logger.warning(
                "Existing split sizes do not match config forget_fraction. "
                f"Expected retain/forget = {expected_retain}/{expected_forget}, "
                f"got {len(retain_data)}/{len(forget_data)}. Recreating split for consistency."
            )
            retain_data, forget_data = create_retain_forget_split(
                train_data,
                forget_fraction=forget_fraction,
                seed=seed,
                save_dir=split_dir
            )
    else:
        # Create new split if doesn't exist
        logger.info(f"Creating retain/forget splits in {split_dir}...")
        member_data, forget_data = create_retain_forget_split(
            train_data,
            forget_fraction=forget_fraction,
            seed=seed,
            save_dir=split_dir
        )
        retain_data = member_data
    
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"  Training samples: {len(train_data)}")
    logger.info(f"  Test samples: {len(test_data)}")
    logger.info(f"  Forget set: {len(forget_data)} ({forget_fraction*100:.0f}%)")
    logger.info(f"  Retain set (members): {len(retain_data)} ({(1-forget_fraction)*100:.0f}%)")
    
    # For MIA: members are RETAIN set (what SCRUB kept), non-members are TEST set
    member_loader = DataLoader(retain_data, batch_size=batch_size, shuffle=False)
    nonmember_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    # Return: member_loader, nonmember_loader, member_data (retain set), test_data (non-members)
    return member_loader, nonmember_loader, retain_data, test_data


def create_attack_configs(config: Dict[str, Any], 
                         override_attacks: Optional[List[str]] = None) -> List[AttackConfig]:
    """Create attack configurations from config file or overrides."""
    logger = logging.getLogger(__name__)
    
    mia_cfg = config.get('mia', {})
    attack_defs = mia_cfg.get('attacks', [])
    attack_defs_by_name = {
        attack_def.get('name', '').lower(): attack_def
        for attack_def in attack_defs
        if isinstance(attack_def, dict) and attack_def.get('name')
    }
    
    if override_attacks:
        # Command-line override: keep configured params/model_access when available
        logger.info(f"Using command-line attack override: {override_attacks}")
        attack_configs = []
        for attack in override_attacks:
            configured = attack_defs_by_name.get(attack.lower(), {})
            attack_configs.append(
                AttackConfig(
                    name=attack,
                    model_access=configured.get('model_access', 'white_box'),
                    params=configured.get('params', {}),
                    seed=_get_seed(config)
                )
            )
    else:
        # Use attacks from config file
        attack_configs = [
            AttackConfig(
                name=attack_def['name'],
                model_access=attack_def.get('model_access', 'white_box'),
                params=attack_def.get('params', {}),
                seed=_get_seed(config)
            )
            for attack_def in attack_defs
        ]
    
    logger.info(f"Attacks to run: {[a.name for a in attack_configs]}")
    return attack_configs


def run_mia_experiment(config_path: str,
                      output_dir: Optional[str] = None,
                      attacks: Optional[List[str]] = None,
                      unlearned_model: Optional[torch.nn.Module] = None):
    """
    Run complete MIA experiment from config.
    
    Args:
        config_path: Path to YAML config file
        output_dir: Optional override for output directory
        attacks: Optional list of attack names to override config
        unlearned_model: Optional pre-loaded unlearned model
    
    Returns:
        Dictionary with results
    """
    
    # Load config
    config = load_config(config_path)
    seed = _get_seed(config)
    _set_global_determinism(seed)
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
    logger.info(f"Seed: {seed}")
    logger.info(f"Split directory: {_get_split_dir(config)}")
    
    # Prepare data
    member_loader, nonmember_loader, member_data, test_data = prepare_data(config)
    
    # Load unlearned model if not provided
    if unlearned_model is None:
        unlearned_model = load_unlearned_model(config, device)
        if unlearned_model is None:
            logger.error("Failed to load unlearned model. Exiting.")
            return None
    
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
        seed=seed,
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
    
    num_members = len(member_data)      # RETAIN set size (what SCRUB trained on)
    num_nonmembers = len(test_data)     # TEST set size (unseen by SCRUB)
    
    # Ground truth: 1 for members, 0 for non-members
    # Order must match AttackResult.all_predictions: [member_preds | nonmember_preds]
    ground_truth = np.concatenate([
        np.ones(num_members),
        np.zeros(num_nonmembers)
    ])
    
    logger.info(f"Ground truth: {num_members} members, {num_nonmembers} non-members")
    logger.info(f"Total samples in ground truth: {len(ground_truth)}")
    
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
        union_result = AttackResult(
            attack_name="union_ensemble",
            attack_config=AttackConfig(name="union_ensemble"),
            member_scores=union_pred[:num_members].astype(float),
            nonmember_scores=union_pred[num_members:].astype(float),
            all_predictions=union_pred,
            member_indices=np.arange(num_members),
        )
        union_metrics = union_result.compute_metrics(ground_truth)
        
        logger.info("\nUnion (OR) Ensemble:")
        for metric_name, metric_value in union_metrics.items():
            logger.info(f"  {metric_name}: {metric_value:.4f}")
        
        # Voting ensemble
        voting_pred, _ = runner.get_ensemble_predictions_voting(k=2)
        voting_result = AttackResult(
            attack_name="voting_ensemble",
            attack_config=AttackConfig(name="voting_ensemble"),
            member_scores=voting_pred[:num_members].astype(float),
            nonmember_scores=voting_pred[num_members:].astype(float),
            all_predictions=voting_pred,
            member_indices=np.arange(num_members),
        )
        voting_metrics = voting_result.compute_metrics(ground_truth)
        
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
