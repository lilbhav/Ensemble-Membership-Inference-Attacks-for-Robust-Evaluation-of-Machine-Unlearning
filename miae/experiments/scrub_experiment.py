# Imports
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import sys
from types import SimpleNamespace
import os
import yaml

# Adding necessary paths to the system path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Framework imports
from data.loaders import load_dataset, get_num_classes
from utils.splits import create_retain_forget_split
from torch.utils.data import DataLoader, random_split
import torchvision.models as models

# Third party code imports
from Third_Party_Code.SCRUB.thirdparty.repdistiller.distiller_zoo.KD import DistillKL
from Third_Party_Code.SCRUB.thirdparty.repdistiller.helper.loops import train_distill


def freeze_bn(model):
    """Freeze BatchNorm running stats."""
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()


def scrub(loaders, args):
    """
    Perform SCRUB unlearning using knowledge distillation.
    """
    # Load a pre-trained ResNet-18 model checkpoint (CIFAR-10)
    model_checkpoint = "./models/pretrained_cifar10.pt"

    if not os.path.exists(model_checkpoint):
        raise FileNotFoundError(
            f"Model checkpoint not found at {model_checkpoint}. "
            f"Please run: python scripts/train_resnet18_cifar10.py"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create model architecture and adapt for CIFAR-10
    model = models.resnet18(pretrained=False)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, get_num_classes("cifar10"))

    # Load checkpoint
    model.load_state_dict(torch.load(model_checkpoint, map_location=device))
    model = model.to(device)

    # Extract loaders
    train_forget_loader = loaders['train_forget_loader']
    train_retain_loader = loaders['train_retain_loader']
    valid_forget_loader = loaders['valid_forget_loader']
    valid_retain_loader = loaders['valid_retain_loader']

    # Hyperparameters
    kd_T = args.kd_T
    learning_rate = args.learning_rate
    epochs = args.epochs
    msteps = args.msteps
    weight_decay = getattr(args, "weight_decay", 1e-4)

    # Teacher and student
    model_t = copy.deepcopy(model).to(device)
    model_s = copy.deepcopy(model).to(device)

    # Freeze teacher
    model_t.eval()
    for p in model_t.parameters():
        p.requires_grad = False

    # Module lists
    module_list = nn.ModuleList([model_s, model_t])
    trainable_list = nn.ModuleList([model_s])

    # Loss functions
    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(kd_T)
    criterion_kd = DistillKL(kd_T)  # Placeholder for interface consistency
    criterion_list = nn.ModuleList([criterion_cls, criterion_div, criterion_kd])

    # Optimizers
    optimizer_forget = optim.SGD(
        trainable_list.parameters(),
        lr=learning_rate * 2.0,
        momentum=0.9,
        weight_decay=weight_decay,
        nesterov=True,
    )

    optimizer_retain = optim.SGD(
        trainable_list.parameters(),
        lr=learning_rate,
        momentum=0.9,
        weight_decay=weight_decay,
        nesterov=True,
    )

    # Track metrics
    tf_accs, tr_accs, vf_accs, vr_accs = [], [], [], []
    losses, epoch_list = [], []

    # Training args
    t_opt = SimpleNamespace()
    t_opt.distill = 'kd'
    t_opt.gamma = args.t_opt_gamma
    t_opt.alpha = args.t_opt_alpha
    t_opt.beta = 0
    t_opt.print_freq = 0

    def fmt_metric(value, precision=6):
        if value is None:
            return "None"
        try:
            return f"{value:.{precision}f}"
        except (TypeError, ValueError):
            return repr(value)

    # =======================
    # Phase 1: Forget Phase
    # =======================
    if msteps > 0:
        print("\n===== FORGET PHASE =====")
        freeze_bn(model_s)

        for f_epoch in range(1, msteps + 1):
            print(f"[Forget Phase {f_epoch}/{msteps}] Starting epoch...")
            sys.stdout.flush()
            print("  - Maximizing loss on forget set...")
            sys.stdout.flush()

            try:
                maximize_loss = train_distill(
                    f_epoch,
                    train_forget_loader,
                    module_list,
                    None,
                    criterion_list,
                    optimizer_forget,
                    t_opt,
                    "maximize",
                    quiet=False,
                )
                print(f"    Done: maximize_loss = {maximize_loss:.4f}")
                sys.stdout.flush()
            except Exception as e:
                print(f"    ERROR during maximize: {e}")
                sys.stdout.flush()
                raise

    # =======================
    # Phase 2: Retain Phase
    # =======================
    print("\n===== RETAIN PHASE =====")
    for epoch in range(1, epochs + 1):
        print(f"[Retain Phase {epoch}/{epochs}] Starting epoch...")
        sys.stdout.flush()
        print("  - Minimizing loss on retain set...")
        sys.stdout.flush()

        try:
            train_acc, train_loss = train_distill(
                epoch,
                train_retain_loader,
                module_list,
                None,
                criterion_list,
                optimizer_retain,
                t_opt,
                "minimize",
                quiet=False,
            )
            print(f"    Done: train_loss = {fmt_metric(train_loss, precision=8)}")
            print(f"    Raw: train_loss={train_loss!r}, train_acc={train_acc!r}")
            sys.stdout.flush()
        except Exception as e:
            print(f"    ERROR during minimize: {e}")
            sys.stdout.flush()
            raise

        losses.append(train_loss)
        epoch_list.append(epoch)

        # Evaluation
        acc_dict = None
        if args.eval_every and (epoch % args.eval_every == 0):
            model_s.eval()
            model_t.eval()

            acc_dict = {
                'tr_acc': compute_accuracy(model_s, train_retain_loader, device),
                'tf_acc': compute_accuracy(model_s, train_forget_loader, device),
                'vr_acc': compute_accuracy(model_s, valid_retain_loader, device),
                'vf_acc': compute_accuracy(model_s, valid_forget_loader, device),
            }

            tr_accs.append(acc_dict['tr_acc'])
            tf_accs.append(acc_dict['tf_acc'])
            vr_accs.append(acc_dict['vr_acc'])
            vf_accs.append(acc_dict['vf_acc'])
        else:
            tr_accs.append(None)
            tf_accs.append(None)
            vr_accs.append(None)
            vf_accs.append(None)

        print(
            f"Epoch {epoch}: minimize loss: {fmt_metric(train_loss, precision=8)}, "
            f"train_acc: {fmt_metric(train_acc, precision=6)}"
        )

        if args.print_accuracies and acc_dict is not None:
            print(f"   tr_acc: {acc_dict['tr_acc']:.4f}")
            print(f"   tf_acc: {acc_dict['tf_acc']:.4f}")
            print(f"   vr_acc: {acc_dict['vr_acc']:.4f}")
            print(f"   vf_acc: {acc_dict['vf_acc']:.4f}")

    # Save student model
    if hasattr(args, 'check_path') and args.check_path is not None:
        os.makedirs(os.path.dirname(args.check_path), exist_ok=True)
        torch.save(model_s.state_dict(), args.check_path)

    history = {
        'losses': losses,
        'epoch_list': epoch_list,
        'tr_accs': tr_accs,
        'tf_accs': tf_accs,
        'vr_accs': vr_accs,
        'vf_accs': vf_accs
    }

    return model_s, history


def compute_accuracy(model, loader, device):
    """Compute accuracy on a given loader."""
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    return correct / total if total > 0 else 0.0


def main():
    """Example usage of SCRUB unlearning."""
    import argparse

    parser = argparse.ArgumentParser(description="Run SCRUB unlearning experiment")
    parser.add_argument('--config', type=str, default='./configs/scrub_experiment.yaml',
                        help='Path to YAML config file')

    cli_args = parser.parse_args()

    # Load config
    if not os.path.exists(cli_args.config):
        raise FileNotFoundError(f"Config file not found: {cli_args.config}")

    with open(cli_args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    args = SimpleNamespace(**config_dict)

    # ========== 1. LOAD DATA ==========
    print("Loading dataset...")
    dataset = load_dataset(
        dataset_name=args.dataset,
        root=args.dataroot,
        train=True
    )

    # ========== 2. CREATE SPLITS ==========
    print("Creating retain/forget splits...")
    retain_set, forget_set = create_retain_forget_split(
        dataset,
        forget_fraction=args.forget_fraction,
        seed=args.seed,
        save_dir="./data/splits"
    )

    print(f"  Retain set size: {len(retain_set)}")
    print(f"  Forget set size: {len(forget_set)}")

    # ========== 3. CREATE TRAIN/VAL SPLITS ==========
    retain_len = len(retain_set)
    forget_len = len(forget_set)

    retain_train_len = int(0.9 * retain_len)
    forget_train_len = int(0.9 * forget_len)

    retain_train, retain_val = random_split(retain_set, [retain_train_len, retain_len - retain_train_len])
    forget_train, forget_val = random_split(forget_set, [forget_train_len, forget_len - forget_train_len])

    print(f"  Retain train: {len(retain_train)}, Retain val: {len(retain_val)}")
    print(f"  Forget train: {len(forget_train)}, Forget val: {len(forget_val)}")

    # ========== 4. CREATE DATA LOADERS ==========
    print("Creating data loaders...")
    pin_memory = args.pin_memory and torch.cuda.is_available()
    loaders = {
        'train_retain_loader': DataLoader(
            retain_train,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=pin_memory
        ),
        'train_forget_loader': DataLoader(
            forget_train,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=pin_memory
        ),
        'valid_retain_loader': DataLoader(
            retain_val,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory
        ),
        'valid_forget_loader': DataLoader(
            forget_val,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory
        ),
    }

    print(f"Data loaders created. Training will start now...")
    print(f"Configuration: epochs={args.epochs}, batch_size={args.batch_size}, "
          f"device={torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")

    # ========== 5. RUN SCRUB UNLEARNING ==========
    print("Running SCRUB unlearning...")
    model, history = scrub(loaders, args)

    print("\nSCRUB unlearning completed!")
    if history['tr_accs'][-1] is not None:
        print(f"Final train retain acc: {history['tr_accs'][-1]:.4f}")
    if history['vf_accs'][-1] is not None:
        print(f"Final valid forget acc: {history['vf_accs'][-1]:.4f}")

    return model, history


if __name__ == "__main__":
    main()
