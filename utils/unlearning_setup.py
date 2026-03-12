import copy
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset, Subset

from data.loaders import get_num_classes, load_dataset


def _clean_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cleaned: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            cleaned[key[len("module."):]] = value
        else:
            cleaned[key] = value
    return cleaned


def _extract_state_dict(checkpoint) -> Dict[str, torch.Tensor]:
    if hasattr(checkpoint, "state_dict"):
        checkpoint = checkpoint.state_dict()
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
        checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint is not a state_dict-like object.")
    return _clean_state_dict(checkpoint)


def create_resnet18_for_cifar(num_classes: int) -> nn.Module:
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _load_checkpoint(path: str, map_location: torch.device) -> Dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location=map_location)
    return _extract_state_dict(checkpoint)


def _load_transfer_initialized_model(
    source_checkpoint_cifar100: str,
    device: torch.device,
) -> nn.Module:
    model_100 = create_resnet18_for_cifar(get_num_classes("cifar100"))
    source_state = _load_checkpoint(source_checkpoint_cifar100, map_location=device)
    model_100.load_state_dict(source_state, strict=True)

    model_10 = create_resnet18_for_cifar(get_num_classes("cifar10"))
    target_state = model_10.state_dict()
    source_model_state = model_100.state_dict()
    for key in target_state.keys():
        if key.startswith("fc."):
            continue
        target_state[key] = source_model_state[key]
    model_10.load_state_dict(target_state, strict=True)
    return model_10.to(device)


def fine_tune_transfer_model_on_cifar10(
    model: nn.Module,
    dataroot: str,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    momentum: float,
    num_workers: int,
    pin_memory: bool,
) -> nn.Module:
    train_data = load_dataset("cifar10", root=dataroot, train=True)
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=True,
    )

    model.train()
    for epoch in range(1, int(epochs) + 1):
        running_loss = 0.0
        total = 0
        correct = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * int(labels.size(0))
            preds = outputs.argmax(dim=1)
            total += int(labels.size(0))
            correct += int((preds == labels).sum().item())

        epoch_loss = running_loss / max(total, 1)
        epoch_acc = correct / max(total, 1)
        print(
            f"[CIFAR-10 fine-tune] epoch {epoch}/{epochs} - "
            f"loss: {epoch_loss:.6f}, acc: {epoch_acc:.4f}"
        )

    return model.eval()


def load_or_create_transfer_model(
    device: torch.device,
    cifar10_checkpoint_path: str,
    source_checkpoint_cifar100: Optional[str],
    dataroot: str,
    finetune_epochs: int,
    finetune_batch_size: int,
    finetune_learning_rate: float,
    finetune_weight_decay: float,
    finetune_momentum: float,
    num_workers: int,
    pin_memory: bool,
) -> nn.Module:
    if os.path.exists(cifar10_checkpoint_path):
        model = create_resnet18_for_cifar(get_num_classes("cifar10"))
        state = _load_checkpoint(cifar10_checkpoint_path, map_location=device)
        model.load_state_dict(state, strict=True)
        return model.to(device).eval()

    if not source_checkpoint_cifar100:
        raise FileNotFoundError(
            "CIFAR-10 checkpoint is missing and source_checkpoint_cifar100 was not provided."
        )
    if not os.path.exists(source_checkpoint_cifar100):
        raise FileNotFoundError(
            f"Source CIFAR-100 checkpoint not found: {source_checkpoint_cifar100}"
        )

    print("CIFAR-10 checkpoint not found. Building transfer model from CIFAR-100 checkpoint...")
    model = _load_transfer_initialized_model(source_checkpoint_cifar100, device=device)

    model = fine_tune_transfer_model_on_cifar10(
        model=model,
        dataroot=dataroot,
        device=device,
        epochs=finetune_epochs,
        batch_size=finetune_batch_size,
        learning_rate=finetune_learning_rate,
        weight_decay=finetune_weight_decay,
        momentum=finetune_momentum,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    ckpt_dir = os.path.dirname(cifar10_checkpoint_path)
    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(model.state_dict(), cifar10_checkpoint_path)
    print(f"Saved fine-tuned CIFAR-10 checkpoint to: {cifar10_checkpoint_path}")
    return model


def get_dataset_targets(dataset: Dataset) -> np.ndarray:
    if isinstance(dataset, Subset):
        base_targets = get_dataset_targets(dataset.dataset)
        return base_targets[np.asarray(dataset.indices, dtype=np.int64)]

    if hasattr(dataset, "targets"):
        return np.asarray(dataset.targets, dtype=np.int64)

    if hasattr(dataset, "labels"):
        return np.asarray(dataset.labels, dtype=np.int64)

    raise ValueError("Dataset does not expose targets/labels and is not a Subset.")


def create_classwise_unlearning_splits(
    dataset: Dataset,
    forget_class: int,
    retain_per_class: int,
    forget_count: int,
    left_out_per_class: int,
    seed: int,
) -> Tuple[Subset, Subset, Subset, Dict[str, List[int]]]:
    targets = get_dataset_targets(dataset)
    rng = np.random.default_rng(int(seed))

    classes = sorted(int(c) for c in np.unique(targets).tolist())
    if int(forget_class) not in classes:
        raise ValueError(f"forget_class={forget_class} is not present in dataset classes {classes}.")

    non_forget_classes = [c for c in classes if c != int(forget_class)]

    retain_indices: List[int] = []
    left_out_indices: List[int] = []

    per_non_forget_required = int(retain_per_class) + int(left_out_per_class)
    for cls in non_forget_classes:
        cls_indices = np.where(targets == cls)[0]
        if int(cls_indices.shape[0]) < per_non_forget_required:
            raise ValueError(
                f"Class {cls} has only {int(cls_indices.shape[0])} samples, "
                f"but {per_non_forget_required} are required "
                f"({retain_per_class} retain + {left_out_per_class} left_out)."
            )
        shuffled = rng.permutation(cls_indices)
        retain_indices.extend(shuffled[: int(retain_per_class)].tolist())
        left_out_indices.extend(
            shuffled[int(retain_per_class): int(retain_per_class) + int(left_out_per_class)].tolist()
        )

    forget_cls_indices = np.where(targets == int(forget_class))[0]
    if int(forget_cls_indices.shape[0]) < int(forget_count):
        raise ValueError(
            f"Forget class {forget_class} has only {int(forget_cls_indices.shape[0])} samples, "
            f"but forget_count={forget_count} was requested."
        )
    forget_indices = rng.permutation(forget_cls_indices)[: int(forget_count)].tolist()

    retain_set = Subset(dataset, retain_indices)
    forget_set = Subset(dataset, forget_indices)
    left_out_set = Subset(dataset, left_out_indices)

    split_info = {
        "retain_indices": retain_indices,
        "forget_indices": forget_indices,
        "left_out_indices": left_out_indices,
    }
    return retain_set, forget_set, left_out_set, split_info
