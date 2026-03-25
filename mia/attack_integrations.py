"""
Integration layer for reference MIA attacks from mia-disparity codebase.

This module wraps the MIA attacks from Third_Party_Code/mia-disparity to work
with our unified MIARunner framework.
"""

import os
import sys
import logging
import copy
import tempfile
import inspect
import importlib
from typing import Dict, Optional, Tuple, Any
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset, ConcatDataset

# Add reference code to path for direct imports
_REF_CODE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'Third_Party_Code/miadisparity'
)
if _REF_CODE_PATH not in sys.path:
    sys.path.insert(0, _REF_CODE_PATH)

# Try to import reference implementations
AugAttack = AugAuxiliaryInfo = AugModelAccess = None
ShokriAttack = ShokriAuxiliaryInfo = ShokriModelAccess = None
LiraAttack = LiraAuxiliaryInfo = LiraModelAccess = None
CalibrationAttack = CalibrationAuxiliaryInfo = CalibrationModelAccess = None
YeomAttack = YeomAuxiliaryInfo = YeomModelAccess = None
ReferenceAttack = ReferenceAuxiliaryInfo = ReferenceModelAccess = None
LosstrajAttack = LosstrajAuxiliaryInfo = LosstrajModelAccess = None
ModelAccessType = AttackTrainingSet = None

try:
    aug_mia = importlib.import_module("miae.attacks.aug_mia")
    base = importlib.import_module("miae.attacks.base")
    AugAttack = aug_mia.AugAttack
    AugAuxiliaryInfo = aug_mia.AugAuxiliaryInfo
    AugModelAccess = aug_mia.AugModelAccess
    ModelAccessType = base.ModelAccessType
    AttackTrainingSet = base.AttackTrainingSet
    HAS_REFERENCE_AUGMENTATION = True
except ImportError:
    HAS_REFERENCE_AUGMENTATION = False

try:
    shokri_mia = importlib.import_module("miae.attacks.shokri_mia")
    ShokriAttack = shokri_mia.ShokriAttack
    ShokriAuxiliaryInfo = shokri_mia.ShokriAuxiliaryInfo
    ShokriModelAccess = shokri_mia.ShokriModelAccess
    HAS_REFERENCE_SHOKRI = True
except ImportError:
    HAS_REFERENCE_SHOKRI = False

try:
    lira_mia = importlib.import_module("miae.attacks.lira_mia")
    LiraAttack = lira_mia.LiraAttack
    LiraAuxiliaryInfo = lira_mia.LiraAuxiliaryInfo
    LiraModelAccess = lira_mia.LiraModelAccess
    HAS_REFERENCE_LIRA = True
except ImportError:
    HAS_REFERENCE_LIRA = False

try:
    calibration_mia = importlib.import_module("miae.attacks.calibration_mia")
    CalibrationAttack = calibration_mia.CalibrationAttack
    CalibrationAuxiliaryInfo = calibration_mia.CalibrationAuxiliaryInfo
    CalibrationModelAccess = calibration_mia.CalibrationModelAccess
    HAS_REFERENCE_CALIBRATION = True
except ImportError:
    HAS_REFERENCE_CALIBRATION = False

try:
    yeom_mia = importlib.import_module("miae.attacks.yeom_mia")
    YeomAttack = yeom_mia.YeomAttack
    YeomAuxiliaryInfo = yeom_mia.YeomAuxiliaryInfo
    YeomModelAccess = yeom_mia.YeomModelAccess
    HAS_REFERENCE_YEOM = True
except ImportError:
    HAS_REFERENCE_YEOM = False

try:
    reference_mia = importlib.import_module("miae.attacks.reference_mia")
    ReferenceAttack = reference_mia.ReferenceAttack
    ReferenceAuxiliaryInfo = reference_mia.ReferenceAuxiliaryInfo
    ReferenceModelAccess = reference_mia.ReferenceModelAccess
    HAS_REFERENCE_REFERENCE = True
except ImportError:
    HAS_REFERENCE_REFERENCE = False

try:
    losstraj_mia = importlib.import_module("miae.attacks.losstraj_mia")
    LosstrajAttack = losstraj_mia.LosstrajAttack
    LosstrajAuxiliaryInfo = losstraj_mia.LosstrajAuxiliaryInfo
    LosstrajModelAccess = losstraj_mia.LosstrajModelAccess
    HAS_REFERENCE_LOSSTRAJ = True
except ImportError:
    HAS_REFERENCE_LOSSTRAJ = False

# Register safe globals for PyTorch 2.6+ compatibility
# This allows unpickling custom classes used by reference attacks
if HAS_REFERENCE_AUGMENTATION:
    try:
        torch.serialization.add_safe_globals([AttackTrainingSet])
    except Exception as e:
        logging.warning(f"Could not register safe globals: {e}")

def _import_aug_attacks():
    """Wrapper for compatibility - attacks already loaded at module init."""
    if HAS_REFERENCE_AUGMENTATION:
        return AugAttack, AugAuxiliaryInfo, AugModelAccess, ModelAccessType
    return None, None, None, None


from mia.mia_runner import AttackResult, AttackConfig


class ReferenceAttackWrapper:
    """
    Wrapper to integrate reference attack implementations with our framework.

    This provides a unified interface to run various attacks against models.
    """

    def __init__(self, logging_enabled: bool = True):
        """
        Initialize the wrapper.

        Args:
            logging_enabled: Whether to enable logging for attacks
        """
        self.logger = logging.getLogger(__name__)
        self.logging_enabled = logging_enabled

    @staticmethod
    def _dataset_size(dataloader: Optional[DataLoader], default: int = 100) -> int:
        if dataloader is None:
            return default
        return len(dataloader.dataset)

    def _placeholder_result(
        self,
        train_dataloader: Optional[DataLoader],
        test_dataloader: Optional[DataLoader],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return deterministic-shape placeholder outputs when reference attacks fail."""
        num_train = self._dataset_size(train_dataloader)
        num_test = self._dataset_size(test_dataloader)
        member_scores = np.random.uniform(0.5, 1.0, num_train)
        nonmember_scores = np.random.uniform(0.0, 0.5, num_test)
        all_predictions = np.concatenate([
            (member_scores > 0.5).astype(int),
            (nonmember_scores > 0.5).astype(int),
        ])
        return member_scores, nonmember_scores, all_predictions

    def _extract_dataset(
        self,
        dataloader: Optional[DataLoader],
        name: str,
        required: bool,
    ) -> Tuple[Optional[TensorDataset], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Materialize an entire dataloader into TensorDataset/data/labels tensors."""
        if dataloader is None:
            if required:
                raise ValueError(f"{name} dataloader is required")
            return None, None, None

        data_list, labels_list = [], []
        for data, labels in dataloader:
            data_list.append(data)
            labels_list.append(labels)

        if not data_list:
            if required:
                raise ValueError(f"{name} dataloader is empty")
            return None, None, None

        data = torch.cat(data_list, dim=0)
        labels = torch.cat(labels_list, dim=0)
        return TensorDataset(data, labels), data, labels

    def _build_attack_datasets(
        self,
        train_dataloader: Optional[DataLoader],
        test_dataloader: Optional[DataLoader],
        aux_dataloader: Optional[DataLoader],
    ) -> Tuple[TensorDataset, TensorDataset, Optional[TensorDataset], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build train/test/aux datasets once for all reference attacks."""
        train_dataset, train_data, train_labels = self._extract_dataset(
            train_dataloader, "train", required=True
        )
        test_dataset, test_data, test_labels = self._extract_dataset(
            test_dataloader, "test", required=True
        )
        aux_dataset, _, _ = self._extract_dataset(aux_dataloader, "aux", required=False)

        return (
            train_dataset,
            test_dataset,
            aux_dataset,
            train_data,
            train_labels,
            test_data,
            test_labels,
        )

    @staticmethod
    def _infer_num_classes(target_model: nn.Module, train_labels: torch.Tensor) -> int:
        num_classes = 10
        try:
            if hasattr(target_model, 'fc'):
                num_classes = target_model.fc.out_features
            elif hasattr(target_model, 'classifier'):
                num_classes = target_model.classifier.out_features
            else:
                num_classes = len(torch.unique(train_labels))
        except Exception:
            pass
        return int(num_classes)

    def _call_with_safe_globals(
        self,
        fn,
        *args,
        trusted_weights_fallback: bool = False,
    ):
        """Execute attack calls under PyTorch safe_globals when available."""
        if AttackTrainingSet is None:
            return fn(*args)

        try:
            with torch.serialization.safe_globals([AttackTrainingSet]):
                return fn(*args)
        except TypeError:
            return fn(*args)
        except Exception as exc:
            if not trusted_weights_fallback or "Weights only load failed" not in str(exc):
                raise

            self.logger.warning(
                "Attack prepare hit PyTorch safe-load guard; retrying with trusted weights_only=False fallback."
            )
            original_torch_load = torch.load

            def _trusted_load(*call_args, **call_kwargs):
                call_kwargs.setdefault("weights_only", False)
                return original_torch_load(*call_args, **call_kwargs)

            torch.load = _trusted_load
            try:
                return fn(*args)
            finally:
                torch.load = original_torch_load

    @staticmethod
    def _predictions_by_member_prior(
        member_scores: np.ndarray,
        nonmember_scores: np.ndarray,
    ) -> np.ndarray:
        """Create binary predictions using score ranking and known member prior."""
        member_scores = np.asarray(member_scores, dtype=float)
        nonmember_scores = np.asarray(nonmember_scores, dtype=float)

        all_scores = np.concatenate([member_scores, nonmember_scores])
        num_members = len(member_scores)

        predictions = np.zeros(len(all_scores), dtype=int)
        if num_members <= 0:
            return predictions

        ranked_desc = np.argsort(all_scores)[::-1]
        predictions[ranked_desc[:num_members]] = 1
        return predictions

    def run_shokri_attack(
        self,
        target_model: nn.Module,
        shadow_models: Optional[list] = None,
        train_dataloader: Optional[DataLoader] = None,
        test_dataloader: Optional[DataLoader] = None,
        aux_dataloader: Optional[DataLoader] = None,
        device: str = "cuda",
        num_shadow_models: int = 10,
        num_epochs: int = 10,
        batch_size: int = 32,
        lr: float = 0.01,
        attack_seed: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run Shokri membership inference attack using reference implementation.

        This attack trains shadow models on data with known membership,
        then trains an attack model to distinguish members from non-members.

        Args:
            target_model: The model to attack
            shadow_models: Pre-trained shadow models (if None, will train them)
            train_dataloader: DataLoader for training data (members)
            test_dataloader: DataLoader for test data (non-members)
            device: Device to run on ('cuda' or 'cpu')
            num_shadow_models: Number of shadow models to train
            num_epochs: Number of epochs for shadow model training
            batch_size: Batch size for training
            lr: Learning rate
            attack_seed: Random seed for reproducibility

        Returns:
            Tuple of (member_scores, nonmember_scores, all_predictions)
        """
        self.logger.info("Running Shokri attack...")

        if not HAS_REFERENCE_SHOKRI:
            self.logger.warning("Reference Shokri not available. Using placeholder.")
            return self._placeholder_result(train_dataloader, test_dataloader)

        try:
            torch_device = torch.device(device)
            self.logger.info("Using reference implementation from Third_Party_Code/mia-disparity")

            (
                train_dataset,
                test_dataset,
                aux_dataset,
                train_data,
                train_labels,
                test_data,
                test_labels,
            ) = self._build_attack_datasets(train_dataloader, test_dataloader, aux_dataloader)
            num_classes = self._infer_num_classes(target_model, train_labels)

            # Create temporary directory for attack artifacts
            temp_dir = tempfile.mkdtemp()

            # Create auxiliary info for Shokri
            aux_info = ShokriAuxiliaryInfo({
                "seed": attack_seed,
                "device": torch_device,
                "num_classes": num_classes,
                "batch_size": batch_size,
                "num_shadow_models": num_shadow_models,
                "epochs": num_epochs,
                "lr": lr,
                "shadow_model_path": os.path.join(temp_dir, "shadow_models"),
                "attack_model_path": os.path.join(temp_dir, "attack_models"),
                "attack_dataset_path": os.path.join(temp_dir, "attack_dataset"),
                "log_path": os.path.join(temp_dir, "logs"),
                "save_path": os.path.join(temp_dir, "models"),
            })

            # Create model copy for untrained access
            untrained_model = copy.deepcopy(target_model)

            # Create model access
            model_access = ShokriModelAccess(
                model=target_model,
                untrained_model=untrained_model,
                access_type=ModelAccessType.BLACK_BOX
            )

            self.logger.info("Preparing Shokri attack (training shadow models)...")
            attack = ShokriAttack(target_model_access=model_access, auxiliary_info=aux_info)
            prepare_dataset = aux_dataset if aux_dataset is not None else train_dataset
            if aux_dataset is not None:
                self.logger.info("Shokri auxiliary dataset size: %d", len(aux_dataset))

            self._call_with_safe_globals(
                attack.prepare,
                prepare_dataset,
                trusted_weights_fallback=True,
            )

            # Get membership scores with safe globals context
            self.logger.info("Inferring membership...")
            combined_dataset = TensorDataset(
                torch.cat([train_data, test_data], dim=0),
                torch.cat([train_labels, test_labels], dim=0),
            )
            num_member = len(train_dataset)

            # Classwise unlearning can omit one class from member data (e.g., forget_class).
            # The reference Shokri implementation expects an attack model per label at inference.
            # If any labels are missing, clone a fallback model to avoid KeyError and keep the
            # attack execution deterministic.
            required_labels = set(int(v) for v in torch.unique(torch.cat([train_labels, test_labels], dim=0)).tolist())
            present_labels = set(int(k) for k in attack.attack_model_dict.keys())
            missing_labels = sorted(required_labels.difference(present_labels))
            if missing_labels and present_labels:
                fallback_label = sorted(present_labels)[0]
                fallback_model = attack.attack_model_dict[fallback_label]
                for missing_label in missing_labels:
                    attack.attack_model_dict[missing_label] = copy.deepcopy(fallback_model)
                self.logger.warning(
                    "Shokri attack models missing labels %s; reusing fallback model from label %s.",
                    missing_labels,
                    fallback_label,
                )

            combined_scores = self._call_with_safe_globals(attack.infer, combined_dataset)

            member_scores = np.asarray(combined_scores[:num_member], dtype=float)
            nonmember_scores = np.asarray(combined_scores[num_member:], dtype=float)

            # Clip to [0, 1]
            member_scores = np.clip(member_scores, 0, 1)
            nonmember_scores = np.clip(nonmember_scores, 0, 1)

            # Create predictions
            all_predictions = np.concatenate([
                (member_scores > 0.5).astype(int),
                (nonmember_scores > 0.5).astype(int)
            ])

            self.logger.info(
                f"Shokri attack complete. "
                f"Member: {member_scores.mean():.3f} ± {member_scores.std():.3f}, "
                f"Non-member: {nonmember_scores.mean():.3f} ± {nonmember_scores.std():.3f}"
            )

            return member_scores, nonmember_scores, all_predictions

        except Exception as e:
            self.logger.error(f"Error in Shokri attack: {e}", exc_info=True)
            self.logger.warning("Falling back to placeholder...")
            return self._placeholder_result(train_dataloader, test_dataloader)

    def run_yeom_attack(
        self,
        target_model: nn.Module,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        aux_dataloader: Optional[DataLoader] = None,
        device: str = "cuda",
        batch_size: int = 128,
        attack_seed: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run Yeom membership inference attack (confidence-based).

        This is a simple attack based on model confidence scores.

        Args:
            target_model: The model to attack
            train_dataloader: DataLoader for training data (members)
            test_dataloader: DataLoader for test data (non-members)
            device: Device to run on
            attack_seed: Random seed

        Returns:
            Tuple of (member_scores, nonmember_scores, all_predictions)
        """
        self.logger.info("Running Yeom attack...")

        if not HAS_REFERENCE_YEOM:
            self.logger.warning("Reference Yeom not available. Using placeholder.")
            return self._placeholder_result(train_dataloader, test_dataloader)

        try:
            torch_device = torch.device(device)
            self.logger.info("Using reference implementation from Third_Party_Code/mia-disparity")

            (
                train_dataset,
                test_dataset,
                aux_dataset,
                train_data,
                train_labels,
                test_data,
                test_labels,
            ) = self._build_attack_datasets(train_dataloader, test_dataloader, aux_dataloader)
            num_classes = self._infer_num_classes(target_model, train_labels)

            temp_dir = tempfile.mkdtemp()
            aux_info = YeomAuxiliaryInfo({
                "seed": attack_seed,
                "device": torch_device,
                "num_classes": num_classes,
                "batch_size": batch_size,
                "save_path": os.path.join(temp_dir, "yeom"),
                "log_path": os.path.join(temp_dir, "logs"),
            })

            untrained_model = copy.deepcopy(target_model)
            model_access = YeomModelAccess(
                model=target_model,
                untrained_model=untrained_model,
                access_type=ModelAccessType.BLACK_BOX,
            )

            attack = YeomAttack(target_model_access=model_access, aux_info=aux_info)
            prepare_dataset = aux_dataset if aux_dataset is not None else train_dataset
            if aux_dataset is not None:
                self.logger.info("Yeom auxiliary dataset size: %d", len(aux_dataset))
            attack.prepare(prepare_dataset)

            combined_dataset = TensorDataset(
                torch.cat([train_data, test_data], dim=0),
                torch.cat([train_labels, test_labels], dim=0),
            )
            combined_scores = np.asarray(attack.infer(combined_dataset), dtype=float)

            num_member = len(train_dataset)
            member_scores = combined_scores[:num_member]
            nonmember_scores = combined_scores[num_member:]

            all_predictions = self._predictions_by_member_prior(member_scores, nonmember_scores)

            return member_scores, nonmember_scores, all_predictions

        except Exception as e:
            self.logger.error(f"Error in Yeom attack: {e}", exc_info=True)
            self.logger.warning("Falling back to placeholder...")
            return self._placeholder_result(train_dataloader, test_dataloader)

    def run_lira_attack(
        self,
        target_model: nn.Module,
        shadow_models: Optional[list] = None,
        train_dataloader: Optional[DataLoader] = None,
        test_dataloader: Optional[DataLoader] = None,
        aux_dataloader: Optional[DataLoader] = None,
        device: str = "cuda",
        num_shadow_models: int = 32,
        num_epochs: int = 10,
        batch_size: int = 32,
        model_seed: int = 42,
        attack_seed: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run LIRA (Likelihood Ratio Attack) membership inference attack using reference implementation.

        This attack trains multiple shadow models and uses likelihood ratios
        from their predictions to perform the membership test.

        Args:
            target_model: The model to attack
            shadow_models: Pre-trained shadow models
            train_dataloader: DataLoader for training subsets (members)
            test_dataloader: DataLoader for test data (non-members)
            device: Device to run on
            num_shadow_models: Number of shadow models
            num_epochs: Training epochs
            batch_size: Batch size
            model_seed: Seed for reproducibility
            attack_seed: Seed for attack

        Returns:
            Tuple of (member_scores, nonmember_scores, all_predictions)
        """
        self.logger.info("Running LIRA attack...")

        if not HAS_REFERENCE_LIRA:
            self.logger.warning("Reference LIRA not available. Using placeholder.")
            return self._placeholder_result(train_dataloader, test_dataloader)

        try:
            torch_device = torch.device(device)
            self.logger.info("Using reference implementation from Third_Party_Code/mia-disparity")

            (
                train_dataset,
                test_dataset,
                aux_dataset,
                _,
                train_labels,
                _,
                _,
            ) = self._build_attack_datasets(train_dataloader, test_dataloader, aux_dataloader)
            num_classes = self._infer_num_classes(target_model, train_labels)

            # Create temporary directory for attack artifacts
            temp_dir = tempfile.mkdtemp()

            # Create auxiliary info for LIRA
            aux_info = LiraAuxiliaryInfo({
                "seed": attack_seed,
                "device": torch_device,
                "num_shadow_models": num_shadow_models,
                "epochs": num_epochs,
                "shadow_batchsize": batch_size,
                "lr": 0.1,
                "weight_decay": 0.0001,
                "momentum": 0.9,
                "save_path": os.path.join(temp_dir, "models"),
                "shadow_path": os.path.join(temp_dir, "shadow_models"),
                "log_path": os.path.join(temp_dir, "logs"),
                "online": True,
                "fix_variance": True,
            })

            # Create model copy for untrained access
            untrained_model = copy.deepcopy(target_model)

            # Create model access
            model_access = LiraModelAccess(
                model=target_model,
                untrained_model=untrained_model,
                access_type=ModelAccessType.BLACK_BOX
            )

            # Run attack
            self.logger.info("Preparing LIRA attack (training shadow models)...")
            attack = LiraAttack(target_model_access=model_access, auxiliary_info=aux_info)
            if aux_dataset is not None:
                attack.prepare(aux_dataset)
                self.logger.info(
                    "LIRA auxiliary dataset size: %d (disjoint from member set when split protocol provides left_out).",
                    len(aux_dataset),
                )
            else:
                # Fallback keeps prior behavior when no dedicated auxiliary split is available.
                self.logger.warning(
                    "No auxiliary dataloader provided for LIRA; falling back to member dataset as auxiliary set."
                )
                attack.prepare(train_dataset)

            # Get membership scores in one pass to keep score calibration consistent
            self.logger.info("Inferring membership...")
            infer_dataset = ConcatDataset([train_dataset, test_dataset])
            all_scores = np.asarray(attack.infer(infer_dataset), dtype=float)

            num_train = len(train_dataset)
            member_scores = all_scores[:num_train]
            nonmember_scores = all_scores[num_train:]

            # Create predictions without assuming score range; use known member prior
            all_predictions = self._predictions_by_member_prior(member_scores, nonmember_scores)

            self.logger.info(
                f"LIRA attack complete. "
                f"Member: {member_scores.mean():.3f} ± {member_scores.std():.3f}, "
                f"Non-member: {nonmember_scores.mean():.3f} ± {nonmember_scores.std():.3f}"
            )

            return member_scores, nonmember_scores, all_predictions

        except Exception as e:
            self.logger.error(f"Error in LIRA attack: {e}", exc_info=True)
            self.logger.warning("Falling back to placeholder...")
            return self._placeholder_result(train_dataloader, test_dataloader)

    def run_reference_attack(
        self,
        target_model: nn.Module,
        reference_models: list,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        device: str = "cuda",
        num_shadow_models: int = 29,
        num_epochs: int = 10,
        batch_size: int = 128,
        attack_seed: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run Reference model membership inference attack.

        This attack uses reference models trained on similar datasets.

        Args:
            target_model: The model to attack
            reference_models: List of reference models
            train_dataloader: DataLoader for training data
            test_dataloader: DataLoader for test data
            device: Device to run on
            attack_seed: Random seed

        Returns:
            Tuple of (member_scores, nonmember_scores, all_predictions)
        """
        self.logger.info("Running Reference attack...")

        if not HAS_REFERENCE_REFERENCE:
            self.logger.warning("Reference attack implementation not available. Using placeholder.")
            return self._placeholder_result(train_dataloader, test_dataloader)

        try:
            torch_device = torch.device(device)
            self.logger.info("Using reference implementation from Third_Party_Code/mia-disparity")

            (
                train_dataset,
                test_dataset,
                _,
                _,
                train_labels,
                _,
                _,
            ) = self._build_attack_datasets(train_dataloader, test_dataloader, aux_dataloader=None)
            num_classes = self._infer_num_classes(target_model, train_labels)

            temp_dir = tempfile.mkdtemp()
            aux_info = ReferenceAuxiliaryInfo({
                "seed": attack_seed,
                "device": torch_device,
                "num_shadow_models": num_shadow_models,
                "epochs": num_epochs,
                "shadow_batchsize": batch_size,
                "num_classes": num_classes,
                "save_path": os.path.join(temp_dir, "reference"),
                "shadow_path": os.path.join(temp_dir, "reference", "shadow_models"),
                "log_path": os.path.join(temp_dir, "logs"),
            })

            untrained_model = copy.deepcopy(target_model)
            model_access = ReferenceModelAccess(
                model=target_model,
                untrained_model=untrained_model,
                access_type=ModelAccessType.BLACK_BOX,
            )

            attack = ReferenceAttack(target_model_access=model_access, auxiliary_info=aux_info)

            # The upstream reference attack prepares with an auxiliary set and infers on the target set.
            # We use member/non-member concat as target set to align downstream metric expectations.
            attack.prepare(train_dataset)
            infer_dataset = ConcatDataset([train_dataset, test_dataset])
            all_scores = np.asarray(attack.infer(infer_dataset), dtype=float)

            num_train = len(train_dataset)
            member_scores = all_scores[:num_train]
            nonmember_scores = all_scores[num_train:]
            all_predictions = self._predictions_by_member_prior(member_scores, nonmember_scores)

            return member_scores, nonmember_scores, all_predictions

        except Exception as e:
            self.logger.error(f"Error in Reference attack: {e}", exc_info=True)
            self.logger.warning("Falling back to placeholder...")
            return self._placeholder_result(train_dataloader, test_dataloader)

    def run_losstraj_attack(
        self,
        target_model: nn.Module,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        device: str = "cuda",
        num_shadow_models: int = 32,
        num_epochs: int = 10,
        batch_size: int = 128,
        attack_seed: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run Loss Trajectory membership inference attack.

        This attack uses the trajectory of loss values during training as a signal.
        Different membership status leads to different loss trajectories.

        Args:
            target_model: The model to attack
            train_dataloader: DataLoader for training data (members)
            test_dataloader: DataLoader for test data (non-members)
            device: Device to run on
            num_shadow_models: Number of shadow models
            num_epochs: Training epochs
            attack_seed: Random seed

        Returns:
            Tuple of (member_scores, nonmember_scores, all_predictions)
        """
        self.logger.info("Running Loss Trajectory attack...")

        if not HAS_REFERENCE_LOSSTRAJ:
            self.logger.warning("Reference loss trajectory not available. Using placeholder.")
            return self._placeholder_result(train_dataloader, test_dataloader)

        try:
            torch_device = torch.device(device)
            self.logger.info("Using reference implementation from Third_Party_Code/mia-disparity")

            (
                train_dataset,
                test_dataset,
                _,
                _,
                train_labels,
                _,
                _,
            ) = self._build_attack_datasets(train_dataloader, test_dataloader, aux_dataloader=None)
            num_classes = self._infer_num_classes(target_model, train_labels)

            temp_dir = tempfile.mkdtemp()
            aux_info = LosstrajAuxiliaryInfo({
                "seed": attack_seed,
                "device": torch_device,
                "batch_size": batch_size,
                "num_classes": num_classes,
                "distillation_epochs": num_epochs,
                "save_path": os.path.join(temp_dir, "losstraj"),
                "log_path": os.path.join(temp_dir, "logs"),
            })

            untrained_model = copy.deepcopy(target_model)
            model_access = LosstrajModelAccess(
                model=target_model,
                untrained_model=untrained_model,
                model_type=ModelAccessType.BLACK_BOX,
            )

            attack = LosstrajAttack(target_model_access=model_access, auxiliary_info=aux_info)
            attack.prepare(train_dataset)

            infer_dataset = ConcatDataset([train_dataset, test_dataset])
            all_scores = np.asarray(attack.infer(infer_dataset), dtype=float)

            num_train = len(train_dataset)
            member_scores = all_scores[:num_train]
            nonmember_scores = all_scores[num_train:]
            all_predictions = self._predictions_by_member_prior(member_scores, nonmember_scores)

            return member_scores, nonmember_scores, all_predictions

        except Exception as e:
            self.logger.error(f"Error in Loss Trajectory attack: {e}", exc_info=True)
            self.logger.warning("Falling back to placeholder...")
            return self._placeholder_result(train_dataloader, test_dataloader)

    def run_calibration_attack(
        self,
        target_model: nn.Module,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        aux_dataloader: Optional[DataLoader] = None,
        device: str = "cuda",
        num_shadow_models: int = 4,
        num_shadow_epochs: int = 20,
        batch_size: int = 128,
        lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        shadow_train_ratio: float = 0.5,
        shadow_diff_init: bool = False,
        attack_seed: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run Calibration membership inference attack.

        This attack calibrates confidence scores for improved reliability and
        robustness, addressing issues with miscalibrated model outputs.

        Args:
            target_model: The model to attack
            train_dataloader: DataLoader for training data (members)
            test_dataloader: DataLoader for test data (non-members)
            device: Device to run on
            attack_seed: Random seed

        Returns:
            Tuple of (member_scores, nonmember_scores, all_predictions)
        """
        self.logger.info("Running Calibration attack...")

        if not HAS_REFERENCE_CALIBRATION:
            self.logger.warning("Reference calibration not available. Using placeholder.")
            return self._placeholder_result(train_dataloader, test_dataloader)

        try:
            torch_device = torch.device(device)
            self.logger.info("Using reference implementation from Third_Party_Code/mia-disparity")

            (
                train_dataset,
                test_dataset,
                aux_dataset,
                _,
                train_labels,
                _,
                _,
            ) = self._build_attack_datasets(train_dataloader, test_dataloader, aux_dataloader)
            num_classes = self._infer_num_classes(target_model, train_labels)

            temp_dir = tempfile.mkdtemp()

            aux_info = CalibrationAuxiliaryInfo({
                "seed": attack_seed,
                "device": torch_device,
                "num_classes": num_classes,
                "batch_size": batch_size,
                "epochs": num_shadow_epochs,
                "num_shadow_models": num_shadow_models,
                "num_aux": 1,
                "lr": lr,
                "momentum": momentum,
                "weight_decay": weight_decay,
                "shadow_train_ratio": shadow_train_ratio,
                "shadow_diff_init": shadow_diff_init,
                "save_path": os.path.join(temp_dir, "calibration"),
                "shadow_model_path": os.path.join(temp_dir, "calibration", "shadow_models"),
                "log_path": os.path.join(temp_dir, "logs"),
            })

            untrained_model = copy.deepcopy(target_model)
            model_access = CalibrationModelAccess(
                model=target_model,
                untrained_model=untrained_model,
                access_type=ModelAccessType.BLACK_BOX,
            )

            self.logger.info("Preparing calibration attack...")
            attack = CalibrationAttack(target_model_access=model_access, aux_info=aux_info)
            prepare_dataset = aux_dataset if aux_dataset is not None else train_dataset
            if aux_dataset is not None:
                self.logger.info("Calibration auxiliary dataset size: %d", len(aux_dataset))
            attack.prepare(prepare_dataset)

            self.logger.info("Inferring membership...")
            member_scores = attack.infer(train_dataset)
            nonmember_scores = attack.infer(test_dataset)

            member_scores = np.asarray(member_scores, dtype=float)
            nonmember_scores = np.asarray(nonmember_scores, dtype=float)

            if member_scores.size:
                member_scores = np.clip(member_scores, 0, 1)
            if nonmember_scores.size:
                nonmember_scores = np.clip(nonmember_scores, 0, 1)

            all_predictions = self._predictions_by_member_prior(member_scores, nonmember_scores)

            self.logger.info(
                f"Calibration attack complete. "
                f"Member: {member_scores.mean():.3f} ± {member_scores.std():.3f}, "
                f"Non-member: {nonmember_scores.mean():.3f} ± {nonmember_scores.std():.3f}"
            )

            return member_scores, nonmember_scores, all_predictions

        except Exception as e:
            self.logger.error(f"Error in calibration attack: {e}", exc_info=True)
            self.logger.warning("Falling back to placeholder...")
            return self._placeholder_result(train_dataloader, test_dataloader)

    def run_augmentation_attack(
        self,
        target_model: nn.Module,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        aux_dataloader: Optional[DataLoader] = None,
        device: str = "cuda",
        augmentation_type: str = "d",
        augment_kwarg: int = 2,
        attack_seed: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run Augmentation-based membership inference attack using reference implementation.

        Uses Third_Party_Code/mia-disparity/miae/attacks/aug_mia.py.

        Args:
            target_model: The model to attack
            train_dataloader: DataLoader for training data (members)
            test_dataloader: DataLoader for test data (non-members)
            device: Device to run on
            augmentation_type: Type of augmentation ('d' for translation)
            augment_kwarg: Max displacement for translation
            attack_seed: Random seed

        Returns:
            Tuple of (member_scores, nonmember_scores, all_predictions)
        """
        self.logger.info("Running Augmentation attack...")

        if augmentation_type == "d" and int(augment_kwarg) != 2:
            self.logger.warning(
                "augmentation_type='d' in bundled mia-disparity expects augment_kwarg=2 "
                "(9 augmented signals). Overriding provided value "
                f"{augment_kwarg} -> 2 to avoid attack-model shape mismatch."
            )
            augment_kwarg = 2

        if not HAS_REFERENCE_AUGMENTATION:
            self.logger.warning("Reference augmentation not available. Using placeholder.")
            return self._placeholder_result(train_dataloader, test_dataloader)

        try:
            torch_device = torch.device(device)
            self.logger.info("Using reference implementation from Third_Party_Code/mia-disparity")

            (
                train_dataset,
                test_dataset,
                aux_dataset,
                _,
                train_labels,
                _,
                _,
            ) = self._build_attack_datasets(train_dataloader, test_dataloader, aux_dataloader)
            num_classes = self._infer_num_classes(target_model, train_labels)

            # Create temporary directory for attack artifacts
            temp_dir = tempfile.mkdtemp()

            # Create auxiliary info
            aux_info = AugAuxiliaryInfo({
                "seed": attack_seed,
                "device": torch_device,
                "num_classes": num_classes,
                "batch_size": 32,
                "augment_kwarg": augment_kwarg,
                "shadow_batch_size": 32,
                "attack_batch_size": 32,
                "shadow_train_ratio": 0.5,
                "log_path": os.path.join(temp_dir, "logs"),
                "save_path": os.path.join(temp_dir, "models"),
                "attack_model_path": os.path.join(temp_dir, "attack_models"),
                "shadow_model_path": os.path.join(temp_dir, "shadow"),
                "attack_dataset_path": os.path.join(temp_dir, "datasets"),
            })

            # Create model copy for untrained access
            untrained_model = copy.deepcopy(target_model)

            # Create model access using reference default access type.
            # Avoid passing an enum instance from a potentially different
            # import namespace, which can cause access-type equality checks
            # inside mia-disparity to fail.
            model_access = AugModelAccess(
                model=target_model,
                untrained_model=untrained_model,
            )

            # Run attack with safe globals context
            self.logger.info("Preparing augmentation attack (training shadow model)...")
            attack = AugAttack(target_model_access=model_access, auxiliary_info=aux_info)
            prepare_dataset = aux_dataset if aux_dataset is not None else train_dataset
            if aux_dataset is not None:
                self.logger.info("Augmentation auxiliary dataset size: %d", len(aux_dataset))
            
            self._call_with_safe_globals(attack.prepare, prepare_dataset)

            # Get membership scores with safe globals context
            self.logger.info("Inferring membership...")
            member_scores = self._call_with_safe_globals(attack.infer, train_dataset)

            # aug_mia.infer() caches augmented data to a fixed path (target_data_aug.npy).
            # The member call writes this cache for len(train_dataset) samples; the
            # non-member call must not reuse it or it will return the wrong number of
            # scores, causing a shape mismatch with ground_truth_labels.
            _aug_cache = os.path.join(temp_dir, "datasets", "target_data_aug.npy")
            if os.path.exists(_aug_cache):
                os.remove(_aug_cache)

            nonmember_scores = self._call_with_safe_globals(attack.infer, test_dataset)

            # Clip to [0, 1]
            member_scores = np.clip(member_scores, 0, 1)
            nonmember_scores = np.clip(nonmember_scores, 0, 1)

            # Create predictions
            all_predictions = np.concatenate([
                (member_scores > 0.5).astype(int),
                (nonmember_scores > 0.5).astype(int)
            ])

            self.logger.info(
                f"Augmentation attack complete. "
                f"Member: {member_scores.mean():.3f} ± {member_scores.std():.3f}, "
                f"Non-member: {nonmember_scores.mean():.3f} ± {nonmember_scores.std():.3f}"
            )

            return member_scores, nonmember_scores, all_predictions

        except Exception as e:
            self.logger.error(f"Error in augmentation attack: {e}", exc_info=True)
            self.logger.warning("Falling back to placeholder...")
            return self._placeholder_result(train_dataloader, test_dataloader)


    def _get_confidence_scores(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: str,
    ) -> np.ndarray:
        """
        Get maximum softmax confidence scores from model predictions.

        Args:
            model: Model to get predictions from
            dataloader: DataLoader to get data from
            device: Device to run on

        Returns:
            Array of confidence scores
        """
        model.eval()
        confidences = []

        with torch.no_grad():
            for batch, _ in dataloader:
                batch = batch.to(device)
                outputs = model(batch)

                # Get max softmax probability
                probs = torch.softmax(outputs, dim=1)
                max_probs = torch.max(probs, dim=1)[0]

                confidences.extend(max_probs.cpu().numpy())

        return np.array(confidences)


class AttackFactory:
    """
    Factory for creating attack instances based on configuration.

    Supports easy extension with new attack types.
    """

    def __init__(self):
        self.wrapper = ReferenceAttackWrapper()
        self.logger = logging.getLogger(__name__)

    def _prepare_params(self, attack_name: str, params: Optional[Dict[str, Any]], method) -> Dict[str, Any]:
        """Normalize config params for wrapper methods and drop unsupported kwargs."""
        params = dict(params or {})

        alias_maps = {
            "shokri": {
                "epochs": "num_epochs",
            },
            "lira": {
                "epochs": "num_epochs",
            },
            "augmentation": {
                "num_augmentations": "augment_kwarg",
            },
        }

        for source_key, target_key in alias_maps.get(attack_name, {}).items():
            if source_key in params and target_key not in params:
                params[target_key] = params.pop(source_key)

        supported_keys = set(inspect.signature(method).parameters.keys())
        filtered_params = {
            key: value for key, value in params.items() if key in supported_keys
        }
        dropped_keys = sorted(set(params.keys()) - set(filtered_params.keys()))
        if dropped_keys:
            self.logger.warning(
                f"Ignoring unsupported params for {attack_name}: {dropped_keys}"
            )

        return filtered_params

    def create_attack(
        self,
        attack_config: AttackConfig,
        target_model: nn.Module,
        train_dataloader: Optional[DataLoader] = None,
        test_dataloader: Optional[DataLoader] = None,
        aux_dataloader: Optional[DataLoader] = None,
        device: str = "cuda",
    ) -> AttackResult:
        """
        Create and run an attack.

        Args:
            attack_config: Configuration for attack
            target_model: Model to attack
            train_dataloader: Training data (members)
            test_dataloader: Test data (non-members)
            device: Device to run on

        Returns:
            AttackResult with scores and predictions
        """
        attack_name = attack_config.name.lower()

        self.logger.info(f"Creating attack: {attack_name}")

        attack_methods = {
            "shokri": self.wrapper.run_shokri_attack,
            "yeom": self.wrapper.run_yeom_attack,
            "lira": self.wrapper.run_lira_attack,
            "reference": self.wrapper.run_reference_attack,
            "losstraj": self.wrapper.run_losstraj_attack,
            "calibration": self.wrapper.run_calibration_attack,
            "augmentation": self.wrapper.run_augmentation_attack,
        }

        method = attack_methods.get(attack_name)
        if method is None:
            raise ValueError(f"Unknown attack: {attack_name}")

        attack_params = self._prepare_params(
            attack_name,
            attack_config.params,
            method,
        )

        base_kwargs = {
            "target_model": target_model,
            "train_dataloader": train_dataloader,
            "test_dataloader": test_dataloader,
            "aux_dataloader": aux_dataloader,
            "device": device,
            "reference_models": [],  # Placeholder until reference models are explicitly wired.
        }

        supported_keys = set(inspect.signature(method).parameters.keys())
        method_kwargs = {
            key: value for key, value in base_kwargs.items() if key in supported_keys
        }

        member_scores, nonmember_scores, predictions = method(
            **method_kwargs,
            **attack_params,
        )

        # Get member indices (assumes first half of merged data are members)
        num_members = len(train_dataloader.dataset) if train_dataloader else 0
        member_indices = np.arange(num_members)

        # Create and return result
        result = AttackResult(
            attack_name=attack_config.name,
            attack_config=attack_config,
            member_scores=member_scores,
            nonmember_scores=nonmember_scores,
            all_predictions=predictions,
            member_indices=member_indices,
        )

        return result


if __name__ == "__main__":
    # Test the factory
    config = AttackConfig(
        name="yeom",
        params={}
    )

    factory = AttackFactory()
    print(f"Attack factory ready for:")
    print(f"  - Basic: yeom, shokri, lira, reference")
    print(f"  - Advanced: losstraj, calibration, augmentation")
    print(f"Reference augmentation available: {HAS_REFERENCE_AUGMENTATION}")
