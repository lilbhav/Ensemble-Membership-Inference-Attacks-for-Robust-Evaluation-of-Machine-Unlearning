"""
Unified MIA Experiment Runner

Run complete MIA evaluation experiments from config files.
Supports any combination of attacks and unlearning algorithms.

Usage:
    python run_mia_experiment.py --config configs/scrub_experiment.yaml
    python run_mia_experiment.py --config configs/experiment.yaml --output results/my_experiment
    python run_mia_experiment.py --config configs/scrub_experiment.yaml --attacks calibration yeom
"""

import os
import sys
import yaml
import json
import csv
import logging
import argparse
import random
import re
import torch
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from torch.utils.data import DataLoader

# Add paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mia.mia_runner import MIARunner, MIARunnerConfig, AttackConfig, AttackResult
from mia.attack_integrations import AttackFactory
from data.loaders import load_dataset, get_num_classes
from utils.splits import ensure_retain_forget_split, ensure_targeted_random_unlearning_split, ensure_fully_random_unlearning_split


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_logging(log_dir: str, experiment_name: str) -> logging.Logger:
    """Setup logging for the experiment."""
    os.makedirs(log_dir, exist_ok=True)

    class SafeStreamHandler(logging.StreamHandler):
        """StreamHandler that silently disables itself if notebook stdout disconnects."""

        def emit(self, record):
            try:
                super().emit(record)
            except OSError:
                # In Colab/Jupyter, stdout can disconnect transiently (Errno 107).
                try:
                    self.acquire()
                    self.stream = None
                finally:
                    self.release()

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Avoid duplicated logs when setup_logging is called multiple times in one session.
    logger.handlers.clear()
    logger.propagate = False

    # File handler
    log_file = os.path.join(log_dir, f"{experiment_name}.log")
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)

    # Console handler
    ch = SafeStreamHandler()
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


def _get_dataset_cfg(config: Dict[str, Any]) -> Dict[str, Any]:
    dataset_cfg = config.get('dataset', {})
    if isinstance(dataset_cfg, dict):
        return dataset_cfg
    if isinstance(dataset_cfg, str):
        return {'name': dataset_cfg}
    return {}


def _get_dataset_name(config: Dict[str, Any]) -> str:
    dataset_cfg = _get_dataset_cfg(config)
    return str(dataset_cfg.get('name', config.get('dataset_name', 'cifar10')))


def _get_model_architecture(config: Dict[str, Any]) -> str:
    model_cfg = config.get('model', {})
    if isinstance(model_cfg, dict):
        return str(model_cfg.get('architecture', config.get('model_architecture', 'resnet18')))
    return str(config.get('model_architecture', 'resnet18'))


def _get_unlearning_cfg(config: Dict[str, Any]) -> Dict[str, Any]:
    return config.get('unlearning', {}) if isinstance(config.get('unlearning', {}), dict) else {}


def _get_unlearning_params(config: Dict[str, Any]) -> Dict[str, Any]:
    unlearning_cfg = _get_unlearning_cfg(config)
    params = unlearning_cfg.get('params', {}) if isinstance(unlearning_cfg.get('params', {}), dict) else {}
    return params


def _get_mia_cfg(config: Dict[str, Any]) -> Dict[str, Any]:
    return config.get('mia', {}) if isinstance(config.get('mia', {}), dict) else {}


def _get_mia_evaluation_targets(config: Dict[str, Any]) -> List[str]:
    """Get one or more evaluation targets for MIA experiments."""
    mia_cfg = _get_mia_cfg(config)
    configured_targets = mia_cfg.get('evaluation_targets')

    if configured_targets is None:
        configured_targets = [mia_cfg.get('evaluation_target', 'forget_vs_test')]
    elif isinstance(configured_targets, str):
        configured_targets = [configured_targets]
    elif not isinstance(configured_targets, list):
        raise ValueError("mia.evaluation_targets must be a list or string when provided")

    supported = {'forget_vs_test', 'retain_vs_test', 'forget_vs_retain'}
    normalized_targets: List[str] = []
    for raw_target in configured_targets:
        target = str(raw_target).strip().lower()
        if target not in supported:
            raise ValueError(
                f"Unsupported evaluation target '{target}'. Supported: {sorted(supported)}"
            )
        if target not in normalized_targets:
            normalized_targets.append(target)

    return normalized_targets


def _get_mia_evaluation_target(config: Dict[str, Any]) -> str:
    """Backward-compatible helper for single-target callers."""
    return _get_mia_evaluation_targets(config)[0]


def _get_ensemble_config(config: Dict[str, Any]) -> Dict[str, Any]:
    mia_cfg = _get_mia_cfg(config)
    ensemble_cfg = mia_cfg.get('ensembles', {}) if isinstance(mia_cfg.get('ensembles', {}), dict) else {}

    methods = ensemble_cfg.get('methods', ['union', 'voting', 'average_score'])
    if isinstance(methods, str):
        methods = [methods]

    valid_methods = {'union', 'voting', 'average_score'}
    normalized_methods: List[str] = []
    for raw_method in methods:
        method = str(raw_method).strip().lower()
        if method not in valid_methods:
            raise ValueError(
                f"Unsupported ensemble method '{method}'. Supported: {sorted(valid_methods)}"
            )
        if method not in normalized_methods:
            normalized_methods.append(method)

    k = int(ensemble_cfg.get('k', ensemble_cfg.get('voting_k', 2)))
    return {
        'enabled': bool(ensemble_cfg.get('enabled', True)),
        'methods': normalized_methods,
        'k': k,
    }


def _get_model_checkpoint_default(config: Dict[str, Any], model_type: str) -> Optional[str]:
    model_cfg = config.get('model', {}) if isinstance(config.get('model', {}), dict) else {}
    unlearning_cfg = _get_unlearning_cfg(config)

    if model_type == 'baseline':
        return model_cfg.get('checkpoint_path') or config.get('model_path')
    return unlearning_cfg.get('checkpoint_path') or config.get('check_path')


def _normalize_model_entry(config: Dict[str, Any], entry: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(entry, dict):
        raise ValueError("Each entry in models_to_evaluate must be a mapping")

    model_type = str(entry.get('model_type', 'unlearned')).strip().lower()
    if model_type not in {'baseline', 'unlearned'}:
        raise ValueError(
            f"Unsupported model_type '{model_type}'. Supported: ['baseline', 'unlearned']"
        )

    name = str(entry.get('name', '')).strip()
    if not name:
        raise ValueError("Each model entry must include a non-empty name")

    default_method = 'baseline' if model_type == 'baseline' else _get_unlearning_cfg(config).get('method', 'unknown')
    unlearning_method = str(entry.get('unlearning_method', default_method)).strip()
    checkpoint_path = entry.get('checkpoint_path') or _get_model_checkpoint_default(config, model_type)

    if not checkpoint_path:
        raise ValueError(f"Model '{name}' does not define checkpoint_path")

    return {
        'name': name,
        'model_type': model_type,
        'unlearning_method': unlearning_method,
        'checkpoint_path': str(checkpoint_path),
    }


def _get_models_to_evaluate(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    configured_models = config.get('models_to_evaluate')
    if configured_models is None:
        configured_models = [
            {
                'name': _get_unlearning_cfg(config).get('method', 'model'),
                'model_type': 'unlearned',
                'unlearning_method': _get_unlearning_cfg(config).get('method', 'unknown'),
                'checkpoint_path': _get_model_checkpoint_default(config, 'unlearned'),
            }
        ]

    if not isinstance(configured_models, list) or not configured_models:
        raise ValueError("models_to_evaluate must be a non-empty list")

    normalized_models = [_normalize_model_entry(config, entry) for entry in configured_models]
    seen_names = set()
    for entry in normalized_models:
        if entry['name'] in seen_names:
            raise ValueError(f"Duplicate model name in models_to_evaluate: {entry['name']}")
        seen_names.add(entry['name'])
    return normalized_models


def _safe_name(value: str) -> str:
    return re.sub(r'[^A-Za-z0-9._-]+', '_', value.strip()).strip('_') or 'item'


def _torch_load_checkpoint(checkpoint_path: str, device: torch.device):
    try:
        return torch.load(checkpoint_path, map_location=device)
    except Exception as exc:
        if "Weights only load failed" not in str(exc):
            raise

    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location=device)


def _extract_state_dict(checkpoint: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(checkpoint, dict):
        return None

    for key in ('state_dict', 'model_state_dict', 'net', 'model'):
        maybe_state = checkpoint.get(key)
        if isinstance(maybe_state, dict):
            return maybe_state

    tensor_values = [value for value in checkpoint.values() if torch.is_tensor(value)]
    if tensor_values and len(tensor_values) == len(checkpoint):
        return checkpoint

    return None


def _resolve_evaluation_sets(
    evaluation_target: str,
    retain_data,
    forget_data,
    test_data,
):
    """Resolve member/non-member datasets for selected evaluation objective."""
    if evaluation_target == 'forget_vs_test':
        return forget_data, test_data, 'forget', 'test'
    if evaluation_target == 'retain_vs_test':
        return retain_data, test_data, 'retain', 'test'
    if evaluation_target == 'forget_vs_retain':
        return forget_data, retain_data, 'forget', 'retain'
    raise ValueError(f"Unhandled evaluation target: {evaluation_target}")


def _minmax_normalize(scores: np.ndarray) -> np.ndarray:
    """Min-max normalize a 1D score array to [0, 1]."""
    min_score = float(np.min(scores))
    max_score = float(np.max(scores))
    if max_score <= min_score:
        # Degenerate case: all scores are identical, use neutral confidence.
        return np.full_like(scores, 0.5, dtype=float)
    return (scores - min_score) / (max_score - min_score)


def _build_score_based_ensembles(
    attack_results: Dict[str, AttackResult],
    k: int = 2,
    methods: Optional[List[str]] = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Build score-based ensemble outputs from individual attack scores.

    Returns dict entries with keys:
      - "scores": continuous score used for ROC/AUC
      - "predictions": hard labels used for accuracy
    """
    if not attack_results:
        raise ValueError("No attack results available for ensemble construction")

    methods = methods or ['union', 'voting', 'average_score']
    first_result = next(iter(attack_results.values()))
    num_members = len(first_result.member_scores)

    normalized_scores = []
    for result in attack_results.values():
        combined_scores = np.concatenate([result.member_scores, result.nonmember_scores]).astype(float)
        normalized_scores.append(_minmax_normalize(combined_scores))

    scores_matrix = np.stack(normalized_scores, axis=0)
    num_attacks = scores_matrix.shape[0]

    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    if k > num_attacks:
        raise ValueError(f"k ({k}) cannot be greater than number of attacks ({num_attacks})")

    def _rank_predictions(scores: np.ndarray) -> np.ndarray:
        """Predict the top-num_members scoring samples as members."""
        predictions = np.zeros(len(scores), dtype=int)
        if num_members > 0:
            top_idx = np.argsort(scores)[::-1][:num_members]
            predictions[top_idx] = 1
        return predictions

    ensemble_outputs: Dict[str, Dict[str, np.ndarray]] = {}

    if 'union' in methods:
        union_scores = np.max(scores_matrix, axis=0)
        ensemble_outputs['union'] = {
            'scores': union_scores,
            'predictions': _rank_predictions(union_scores),
        }

    if 'voting' in methods:
        vote_counts = np.sum(scores_matrix >= 0.5, axis=0)
        voting_scores = vote_counts.astype(float) / float(num_attacks)
        ensemble_outputs['voting'] = {
            'scores': voting_scores,
            'predictions': (vote_counts >= k).astype(int),
        }

    if 'average_score' in methods:
        average_scores = np.mean(scores_matrix, axis=0)
        ensemble_outputs['average_score'] = {
            'scores': average_scores,
            'predictions': _rank_predictions(average_scores),
        }

    return ensemble_outputs


def _prior_rank_predictions(member_scores: np.ndarray, nonmember_scores: np.ndarray) -> np.ndarray:
    """Create binary predictions by selecting top-K scores, where K=#members."""
    member_scores = np.asarray(member_scores, dtype=float)
    nonmember_scores = np.asarray(nonmember_scores, dtype=float)
    all_scores = np.concatenate([member_scores, nonmember_scores])
    num_members = len(member_scores)

    predictions = np.zeros(len(all_scores), dtype=int)
    if num_members > 0:
        top_idx = np.argsort(all_scores)[::-1][:num_members]
        predictions[top_idx] = 1
    return predictions


def _orient_scores_member_high(result: AttackResult) -> bool:
    """
    Ensure score direction is consistent: higher score => more member-like.

    Returns True when orientation was flipped.
    """
    member_mean = float(np.mean(result.member_scores))
    nonmember_mean = float(np.mean(result.nonmember_scores))

    if member_mean >= nonmember_mean:
        return False

    # Monotonic inversion preserves ranking while fixing direction.
    result.member_scores = -np.asarray(result.member_scores, dtype=float)
    result.nonmember_scores = -np.asarray(result.nonmember_scores, dtype=float)
    result.all_predictions = _prior_rank_predictions(result.member_scores, result.nonmember_scores)
    return True


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


def load_model_to_evaluate(
    config: Dict[str, Any],
    model_spec: Dict[str, Any],
    device: torch.device,
) -> torch.nn.Module:
    """Load a configured baseline or unlearned model checkpoint."""
    logger = logging.getLogger(__name__)
    checkpoint_path = model_spec['checkpoint_path']

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found for model '{model_spec['name']}': {checkpoint_path}")

    logger.info("Loading model '%s' from %s", model_spec['name'], checkpoint_path)
    checkpoint = _torch_load_checkpoint(checkpoint_path, device)

    if hasattr(checkpoint, 'to'):
        return checkpoint.to(device).eval()

    state_dict = _extract_state_dict(checkpoint)
    if state_dict is None:
        raise ValueError(
            f"Unsupported checkpoint format for model '{model_spec['name']}' at {checkpoint_path}"
        )

    model = create_model(_get_model_architecture(config), _get_dataset_name(config), str(device))
    model.load_state_dict(state_dict)
    return model.to(device).eval()


def load_unlearned_model(config: Dict[str, Any], device: str) -> Optional[torch.nn.Module]:
    """Backward-compatible single-model loader."""
    try:
        return load_model_to_evaluate(
            config,
            _normalize_model_entry(
                config,
                {
                    'name': _get_unlearning_cfg(config).get('method', 'model'),
                    'model_type': 'unlearned',
                    'unlearning_method': _get_unlearning_cfg(config).get('method', 'unknown'),
                    'checkpoint_path': _get_model_checkpoint_default(config, 'unlearned'),
                },
            ),
            torch.device(device),
        )
    except Exception:
        logger = logging.getLogger(__name__)
        logger.warning(
            "No compatible unlearned model could be loaded from config for method '%s'",
            _get_unlearning_cfg(config).get('method', 'unknown'),
        )
        return None


def prepare_data(config: Dict[str, Any]) -> tuple:
    """Prepare split datasets and optional auxiliary data."""
    logger = logging.getLogger(__name__)
    
    logger.info("Loading dataset...")
    
    dataset_cfg = _get_dataset_cfg(config)
    dataset_name = _get_dataset_name(config)
    forget_fraction = dataset_cfg.get('forget_fraction', 0.2)
    batch_size = int(config.get('batch_size', dataset_cfg.get('batch_size', 64)))
    seed = _get_seed(config)
    split_dir = _get_split_dir(config)
    unlearning_params = _get_unlearning_params(config)
    
    # Load dataset - load_dataset() returns single dataset, not tuple
    data_root = str(dataset_cfg.get('data_root', config.get('dataroot', './data/raw')))
    train_data = load_dataset(dataset_name, root=data_root, train=True)
    test_data = load_dataset(dataset_name, root=data_root, train=False)
    
    split_protocol = str(unlearning_params.get('split_protocol', 'random')).strip().lower()

    has_count_keys = (
        unlearning_params.get('forget_count') is not None
        and (
            unlearning_params.get('retain_count') is not None
            or unlearning_params.get('retain_per_class') is not None
        )
        and (
            unlearning_params.get('left_out_count') is not None
            or unlearning_params.get('left_out_per_class') is not None
        )
    )

    aux_data = None

    if split_protocol == 'fully_random' and has_count_keys:
        num_classes = get_num_classes(dataset_name)
        forget_count = int(unlearning_params['forget_count'])
        if unlearning_params.get('retain_count') is not None:
            retain_count = int(unlearning_params['retain_count'])
        else:
            retain_count = int(unlearning_params['retain_per_class']) * (num_classes - 1)

        if unlearning_params.get('left_out_count') is not None:
            left_out_count = int(unlearning_params['left_out_count'])
        else:
            left_out_count = int(unlearning_params['left_out_per_class']) * (num_classes - 1)

        retain_data, forget_data, left_out_data, recreated = ensure_fully_random_unlearning_split(
            dataset=train_data,
            split_dir=split_dir,
            retain_count=retain_count,
            forget_count=forget_count,
            left_out_count=left_out_count,
            seed=seed,
            verbose=True,
        )
        logger.info(
            "Fully-random split loaded: retain=%d, forget=%d, left_out=%d%s.",
            len(retain_data), len(forget_data), len(left_out_data),
            " (recreated)" if recreated else " (from disk)",
        )
        aux_data = left_out_data
        logger.info("  Forget set: %d (from any class)", len(forget_data))
        logger.info("  Retain set (members): %d", len(retain_data))

    elif split_protocol == 'targeted_random' and has_count_keys and unlearning_params.get('forget_class') is not None:
        num_classes = get_num_classes(dataset_name)
        forget_class = int(unlearning_params['forget_class'])
        forget_count = int(unlearning_params['forget_count'])
        classes_excluding_forget = num_classes - 1

        if unlearning_params.get('retain_count') is not None:
            retain_count = int(unlearning_params['retain_count'])
        else:
            retain_count = int(unlearning_params['retain_per_class']) * classes_excluding_forget

        if unlearning_params.get('left_out_count') is not None:
            left_out_count = int(unlearning_params['left_out_count'])
        else:
            left_out_count = int(unlearning_params['left_out_per_class']) * classes_excluding_forget

        retain_data, forget_data, left_out_data, recreated = ensure_targeted_random_unlearning_split(
            dataset=train_data,
            split_dir=split_dir,
            forget_class=forget_class,
            retain_count=retain_count,
            forget_count=forget_count,
            left_out_count=left_out_count,
            seed=seed,
            verbose=True,
        )
        logger.info(
            "Targeted-random split loaded: retain=%d, forget=%d, left_out=%d%s.",
            len(retain_data), len(forget_data), len(left_out_data),
            " (recreated)" if recreated else " (from disk)",
        )
        aux_data = left_out_data
        logger.info("  Forget set: %d (target class=%d)", len(forget_data), forget_class)
        logger.info("  Retain set (members): %d", len(retain_data))

    else:
        if split_protocol in ['targeted_random', 'fully_random']:
            logger.warning(
                "split_protocol=%s requested but required keys are missing; falling back to fraction-based random split.",
                split_protocol,
            )

        logger.info(
            "Using random retain/forget split with forget_fraction=%s (split_protocol=%s).",
            forget_fraction,
            split_protocol,
        )
        retain_data, forget_data, recreated = ensure_retain_forget_split(
            train_data,
            split_dir=split_dir,
            forget_fraction=forget_fraction,
            seed=seed,
            verbose=False,
        )
        if recreated:
            logger.info(
                f"Created/recreated retain/forget splits in {split_dir} "
                f"for seed={seed}, forget_fraction={forget_fraction}."
            )
        else:
            logger.info(f"Loaded validated retain/forget splits from disk ({split_dir}).")

        logger.info(f"  Forget set: {len(forget_data)} ({forget_fraction*100:.0f}%)")
        logger.info(f"  Retain set (members): {len(retain_data)} ({(1-forget_fraction)*100:.0f}%)")
    
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"  Training samples: {len(train_data)}")
    logger.info(f"  Test samples: {len(test_data)}")
    
    # Return raw datasets. Evaluation target selection happens in run_mia_experiment.
    return retain_data, forget_data, test_data, aux_data, batch_size


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


def _create_ensemble_result(
    ensemble_name: str,
    ensemble_scores: np.ndarray,
    ensemble_predictions: np.ndarray,
    num_members: int,
) -> AttackResult:
    return AttackResult(
        attack_name=ensemble_name,
        attack_config=AttackConfig(name=ensemble_name),
        member_scores=ensemble_scores[:num_members].astype(float),
        nonmember_scores=ensemble_scores[num_members:].astype(float),
        all_predictions=ensemble_predictions.astype(int),
        member_indices=np.arange(num_members),
    )


def _build_structured_result_row(
    model_spec: Dict[str, Any],
    evaluation_target: str,
    attack_name: str,
    metric_values: Dict[str, float],
    seed: int,
) -> Dict[str, Any]:
    return {
        'model_name': model_spec['name'],
        'model_type': model_spec['model_type'],
        'unlearning_method': model_spec['unlearning_method'],
        'evaluation_target': evaluation_target,
        'attack_name': attack_name,
        'auc': float(metric_values.get('auc')) if metric_values.get('auc') is not None else None,
        'accuracy': float(metric_values.get('accuracy')) if metric_values.get('accuracy') is not None else None,
        'precision': float(metric_values.get('precision')) if metric_values.get('precision') is not None else None,
        'recall': float(metric_values.get('recall')) if metric_values.get('recall') is not None else None,
        'f1': float(metric_values.get('f1')) if metric_values.get('f1') is not None else None,
        'tpr_at_1pct_fpr': float(metric_values.get('tpr_at_1pct_fpr', metric_values.get('tpr_at_fpr_0.01'))) if metric_values.get('tpr_at_1pct_fpr', metric_values.get('tpr_at_fpr_0.01')) is not None else None,
        'seed': int(seed),
    }


def _write_structured_outputs(
    rows: List[Dict[str, Any]],
    json_path: str,
    csv_path: str,
    payload: Dict[str, Any],
) -> None:
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    with open(json_path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2)

    fieldnames = [
        'model_name',
        'model_type',
        'unlearning_method',
        'evaluation_target',
        'attack_name',
        'auc',
        'accuracy',
        'precision',
        'recall',
        'f1',
        'tpr_at_1pct_fpr',
        'seed',
    ]
    with open(csv_path, 'w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


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
        unlearned_model: Optional pre-loaded model for single-model runs
    
    Returns:
        Dictionary with combined experiment results
    """

    config = load_config(config_path)
    seed = _get_seed(config)
    _set_global_determinism(seed)
    experiment_name = config.get('experiment', {}).get('name', 'mia_experiment')
    output_cfg = config.get('output', {}) if isinstance(config.get('output', {}), dict) else {}
    output_path = output_dir or output_cfg.get('save_dir', './results/mia')
    log_dir = output_cfg.get('log_dir', os.path.join(output_path, 'logs'))
    logger = setup_logging(log_dir, experiment_name)

    logger.info("=" * 80)
    logger.info(f"MIA EXPERIMENT: {experiment_name}")
    logger.info("=" * 80)

    device = torch.device(config.get('experiment', {}).get('device', 'cuda'))
    attack_configs = create_attack_configs(config, override_attacks=attacks)
    model_specs = _get_models_to_evaluate(config)
    evaluation_targets = _get_mia_evaluation_targets(config)
    ensemble_cfg = _get_ensemble_config(config)

    logger.info(f"Device: {device}")
    logger.info(f"Seed: {seed}")
    logger.info(f"Split directory: {_get_split_dir(config)}")
    logger.info(f"Models to evaluate: {[model_spec['name'] for model_spec in model_specs]}")
    logger.info(f"Evaluation targets: {evaluation_targets}")
    logger.info(f"Output directory: {output_path}")

    if unlearned_model is not None and len(model_specs) != 1:
        logger.warning(
            "A pre-loaded model was provided, but models_to_evaluate contains %d entries. The provided model will be ignored.",
            len(model_specs),
        )

    retain_data, forget_data, test_data, aux_data, batch_size = prepare_data(config)

    combined_rows: List[Dict[str, Any]] = []
    combined_results: Dict[str, Any] = {
        'schema_version': 2,
        'experiment_name': experiment_name,
        'seed': int(seed),
        'models_to_evaluate': model_specs,
        'evaluation_targets': evaluation_targets,
        'ensemble_config': ensemble_cfg,
        'attacks': [
            {
                'name': attack_cfg.name,
                'model_access': attack_cfg.model_access,
                'params': dict(attack_cfg.params),
            }
            for attack_cfg in attack_configs
        ],
        'targets': {},
        'rows': combined_rows,
        'config': config,
    }

    for evaluation_target in evaluation_targets:
        logger.info("\n" + "=" * 80)
        logger.info("EVALUATION TARGET: %s", evaluation_target)
        logger.info("=" * 80)

        member_data, nonmember_data, member_name, nonmember_name = _resolve_evaluation_sets(
            evaluation_target,
            retain_data,
            forget_data,
            test_data,
        )
        member_loader = DataLoader(member_data, batch_size=batch_size, shuffle=False)
        nonmember_loader = DataLoader(nonmember_data, batch_size=batch_size, shuffle=False)
        aux_loader = DataLoader(aux_data, batch_size=batch_size, shuffle=False) if aux_data is not None else None

        if evaluation_target == 'forget_vs_test':
            shadow_aux_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
            logger.info(
                "Shadow auxiliary: using test_data (%d samples) to cover all classes including forget class.",
                len(test_data),
            )
        else:
            shadow_aux_loader = aux_loader

        num_members = len(member_data)
        num_nonmembers = len(nonmember_data)
        ground_truth = np.concatenate([
            np.ones(num_members),
            np.zeros(num_nonmembers),
        ])

        logger.info("Member set (%s): %d", member_name, num_members)
        logger.info("Non-member set (%s): %d", nonmember_name, num_nonmembers)

        target_result_payload = {
            'member_set_name': member_name,
            'nonmember_set_name': nonmember_name,
            'member_count': int(num_members),
            'nonmember_count': int(num_nonmembers),
            'models': {},
        }
        combined_results['targets'][evaluation_target] = target_result_payload

        for model_index, model_spec in enumerate(model_specs, start=1):
            logger.info("\n[%d/%d] Model: %s", model_index, len(model_specs), model_spec['name'])

            model_output_dir = os.path.join(
                output_path,
                'per_model',
                _safe_name(evaluation_target),
                _safe_name(model_spec['name']),
            )
            model_log_dir = os.path.join(
                log_dir,
                _safe_name(evaluation_target),
                _safe_name(model_spec['name']),
            )

            if unlearned_model is not None and len(model_specs) == 1:
                target_model = unlearned_model.to(device).eval()
            else:
                target_model = load_model_to_evaluate(config, model_spec, device)

            mia_config = MIARunnerConfig(
                dataset_name=_get_dataset_name(config),
                model_architecture=_get_model_architecture(config),
                unlearning_method=model_spec['unlearning_method'],
                attacks=attack_configs,
                device=device,
                seed=seed,
                output_dir=model_output_dir,
                log_dir=model_log_dir,
            )

            runner = MIARunner(mia_config)
            factory = AttackFactory()

            for attack_position, attack_config in enumerate(attack_configs, start=1):
                logger.info("  [%d/%d] Running %s", attack_position, len(attack_configs), attack_config.name)
                try:
                    result = factory.create_attack(
                        attack_config=attack_config,
                        target_model=target_model,
                        train_dataloader=member_loader,
                        test_dataloader=nonmember_loader,
                        aux_dataloader=shadow_aux_loader,
                        device=device,
                    )

                    if _orient_scores_member_high(result):
                        logger.info(
                            "    ↺ %s scores were inverted to enforce member-high orientation.",
                            attack_config.name,
                        )

                    runner.attack_results[attack_config.name] = result
                    logger.info("    ✓ %s completed", attack_config.name)
                except Exception as exc:
                    logger.error("    ✗ %s failed: %s", attack_config.name, exc)

            if not runner.attack_results:
                logger.error("No attacks completed successfully for model '%s'", model_spec['name'])
                continue

            solo_metrics = runner.evaluate_attacks(ground_truth)
            inverted_attacks = [
                attack_name
                for attack_name, attack_metric_values in solo_metrics.items()
                if float(attack_metric_values.get('auc', 0.5)) < 0.5
            ]
            if inverted_attacks:
                logger.warning(
                    "Detected inverted attacks (AUC < 0.5) for model '%s': %s",
                    model_spec['name'],
                    inverted_attacks,
                )

            if ensemble_cfg['enabled'] and len(runner.attack_results) > 1:
                logger.info("  Building ensembles: %s (k=%d)", ensemble_cfg['methods'], ensemble_cfg['k'])
                ensemble_outputs = _build_score_based_ensembles(
                    attack_results=runner.attack_results,
                    k=ensemble_cfg['k'],
                    methods=ensemble_cfg['methods'],
                )
                for ensemble_name, ensemble_output in ensemble_outputs.items():
                    result_name = f"{ensemble_name}_ensemble"
                    runner.attack_results[result_name] = _create_ensemble_result(
                        ensemble_name=result_name,
                        ensemble_scores=ensemble_output['scores'],
                        ensemble_predictions=ensemble_output['predictions'],
                        num_members=num_members,
                    )

            all_metrics = runner.evaluate_attacks(ground_truth)
            model_rows = [
                _build_structured_result_row(
                    model_spec=model_spec,
                    evaluation_target=evaluation_target,
                    attack_name=attack_name,
                    metric_values=metric_values,
                    seed=seed,
                )
                for attack_name, metric_values in all_metrics.items()
            ]
            combined_rows.extend(model_rows)

            runner.save_results()
            report = runner.generate_report(ground_truth)
            report_path = os.path.join(model_output_dir, 'evaluation_report.txt')
            with open(report_path, 'w', encoding='utf-8') as handle:
                handle.write(report)

            model_payload = {
                'model_name': model_spec['name'],
                'model_type': model_spec['model_type'],
                'unlearning_method': model_spec['unlearning_method'],
                'checkpoint_path': model_spec['checkpoint_path'],
                'rows': model_rows,
                'metrics': {attack_name: dict(metric_values) for attack_name, metric_values in all_metrics.items()},
            }
            target_result_payload['models'][model_spec['name']] = model_payload

            _write_structured_outputs(
                rows=model_rows,
                json_path=os.path.join(model_output_dir, 'evaluation_summary.json'),
                csv_path=os.path.join(model_output_dir, 'evaluation_summary.csv'),
                payload={
                    'schema_version': 2,
                    'experiment_name': experiment_name,
                    'seed': int(seed),
                    'evaluation_target': evaluation_target,
                    'member_set_name': member_name,
                    'nonmember_set_name': nonmember_name,
                    'member_count': int(num_members),
                    'nonmember_count': int(num_nonmembers),
                    'model': model_payload,
                },
            )

    combined_json_path = os.path.join(output_path, 'combined_evaluation_summary.json')
    combined_csv_path = os.path.join(output_path, 'combined_evaluation_summary.csv')
    _write_structured_outputs(
        rows=combined_rows,
        json_path=combined_json_path,
        csv_path=combined_csv_path,
        payload=combined_results,
    )

    logger.info("\n" + "=" * 80)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("=" * 80)
    logger.info("Combined summary JSON saved to: %s", combined_json_path)
    logger.info("Combined summary CSV saved to: %s", combined_csv_path)

    return {
        'config_path': config_path,
        'output_dir': output_path,
        'rows': combined_rows,
        'targets': combined_results['targets'],
        'models_to_evaluate': model_specs,
        'evaluation_targets': evaluation_targets,
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
  python run_mia_experiment.py --config configs/scrub_experiment.yaml -o results/scrub
  python run_mia_experiment.py --config configs/ssd_experiment.yaml -o results/ssd
    python run_mia_experiment.py --config configs/bad_teacher_experiment.yaml -o results/bad_teacher
    python run_mia_experiment.py --config configs/amnesiac_experiment.yaml -o results/amnesiac
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
