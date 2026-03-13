import os
import random
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _build_resnet18(num_classes: int) -> nn.Module:
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _get_cifar10_finetune_loader(
    dataroot: str,
    batch_size: int,
    seed: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    g = torch.Generator().manual_seed(int(seed))
    transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
        ]
    )
    dataset = datasets.CIFAR10(root=dataroot, train=True, transform=transform, download=True)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=g,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
    )


def ensure_cifar10_from_cifar100_transfer_checkpoint(
    source_checkpoint_cifar100: str,
    target_checkpoint_cifar10: str,
    dataroot: str,
    finetune_epochs: int,
    finetune_batch_size: int,
    finetune_learning_rate: float,
    seed: int,
    num_workers: int = 2,
    pin_memory: bool = True,
    force_rebuild: bool = False,
    device: Optional[torch.device] = None,
) -> str:
    """
    Build a CIFAR-10 checkpoint from a CIFAR-100-pretrained ResNet18 checkpoint.

    Steps:
    1) Load CIFAR-100 checkpoint into a 100-class head.
    2) Replace the final layer with a new 10-class head.
    3) Fine-tune on CIFAR-10 training set.
    """
    if not force_rebuild and os.path.exists(target_checkpoint_cifar10):
        return target_checkpoint_cifar10

    if not os.path.exists(source_checkpoint_cifar100):
        raise FileNotFoundError(
            f"Source CIFAR-100 checkpoint not found: {source_checkpoint_cifar100}. "
            "Train it first with scripts/train_resnet18_cifar100.py"
        )

    _set_seed(int(seed))
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load CIFAR-100 backbone+head first.
    source_model = _build_resnet18(num_classes=100)
    source_model.load_state_dict(torch.load(source_checkpoint_cifar100, map_location=device))

    # Transfer backbone to CIFAR-10 head.
    target_model = _build_resnet18(num_classes=10)
    target_state = target_model.state_dict()
    source_state = source_model.state_dict()

    for name, tensor in source_state.items():
        if name.startswith("fc."):
            continue
        target_state[name] = tensor
    target_model.load_state_dict(target_state)
    target_model = target_model.to(device)

    train_loader = _get_cifar10_finetune_loader(
        dataroot=dataroot,
        batch_size=int(finetune_batch_size),
        seed=int(seed),
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        target_model.parameters(),
        lr=float(finetune_learning_rate),
        momentum=0.9,
        weight_decay=5e-4,
    )

    target_model.train()
    for _ in range(int(finetune_epochs)):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = target_model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

    os.makedirs(os.path.dirname(target_checkpoint_cifar10) or ".", exist_ok=True)
    torch.save(target_model.state_dict(), target_checkpoint_cifar10)
    return target_checkpoint_cifar10
