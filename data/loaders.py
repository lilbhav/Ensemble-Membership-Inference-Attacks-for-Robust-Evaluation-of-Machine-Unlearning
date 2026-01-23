# miae/datasets/loaders.py

import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset


def get_transforms(dataset_name: str):
    if dataset_name.lower() in ["cifar10", "cifar100"]:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                 std=[0.2470, 0.2435, 0.2616])
        ])
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


def load_dataset(dataset_name: str, root: str = "./data/raw", train: bool = True):
    transform = get_transforms(dataset_name)

    if dataset_name.lower() == "cifar10":
        dataset = datasets.CIFAR10(
            root=root,
            train=train,
            transform=transform,
            download=True
        )
    elif dataset_name.lower() == "cifar100":
        dataset = datasets.CIFAR100(
            root=root,
            train=train,
            transform=transform,
            download=True
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return dataset


def get_num_classes(dataset_name: str) -> int:
    if dataset_name.lower() == "cifar10":
        return 10
    elif dataset_name.lower() == "cifar100":
        return 100
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
