import os
import torch
from torch.utils.data import DataLoader


def compute_accuracy(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Compute accuracy for a model on a dataloader."""
    correct = 0
    total = 0
    was_training = model.training
    model.eval()

    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                inputs, _, targets = batch
            else:
                inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    if was_training:
        model.train()

    return correct / total if total > 0 else 0.0


def log_accuracies(log_path: str, label: str, acc_dict: dict) -> str:
    """Format and optionally append accuracy metrics to a log file."""
    line = (
        f"{label} | tr_acc: {acc_dict['tr_acc']:.4f} "
        f"tf_acc: {acc_dict['tf_acc']:.4f} "
        f"vr_acc: {acc_dict['vr_acc']:.4f} "
        f"vf_acc: {acc_dict['vf_acc']:.4f}"
    )

    if log_path:
        log_dir = os.path.dirname(log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    return line
