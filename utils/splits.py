# miae/utils/splits.py

import os
import numpy as np
import torch
from torch.utils.data import Subset


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def create_retain_forget_split(
    dataset,
    forget_fraction: float = 0.1,
    seed: int = 0,
    save_dir: str = None
):
    """
    Splits dataset into retain and forget subsets.
    Optionally saves indices to disk.
    """
    set_seed(seed)
    n = len(dataset)
    indices = np.random.permutation(n)
    n_forget = int(n * forget_fraction)

    forget_idx = indices[:n_forget]
    retain_idx = indices[n_forget:]

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, "forget_idx.npy"), forget_idx)
        np.save(os.path.join(save_dir, "retain_idx.npy"), retain_idx)

    forget_set = Subset(dataset, forget_idx)
    retain_set = Subset(dataset, retain_idx)

    return retain_set, forget_set


def create_auxiliary_split(
    dataset,
    aux_fraction: float = 0.5,
    seed: int = 0,
    save_dir: str = None
):
    """
    Creates auxiliary dataset indices for attacker models.
    """
    set_seed(seed)
    n = len(dataset)
    indices = np.random.permutation(n)
    n_aux = int(n * aux_fraction)

    aux_idx = indices[:n_aux]

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, "aux_idx.npy"), aux_idx)

    aux_set = Subset(dataset, aux_idx)
    return aux_set


def load_split(dataset, split_dir: str):
    """
    Loads retain/forget splits from saved indices.
    """
    forget_idx = np.load(os.path.join(split_dir, "forget_idx.npy"))
    retain_idx = np.load(os.path.join(split_dir, "retain_idx.npy"))

    forget_set = Subset(dataset, forget_idx)
    retain_set = Subset(dataset, retain_idx)

    return retain_set, forget_set


def load_auxiliary(dataset, aux_dir: str):
    """
    Loads auxiliary split from saved indices.
    """
    aux_idx = np.load(os.path.join(aux_dir, "aux_idx.npy"))
    aux_set = Subset(dataset, aux_idx)
    return aux_set
