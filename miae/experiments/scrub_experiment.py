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
from torch.utils.data import DataLoader, Subset
import torchvision.models as models

# Third party code imports
from Third_Party_Code.SCRUB.thirdparty.repdistiller.distiller_zoo.KD import DistillKL
from Third_Party_Code.SCRUB.thirdparty.repdistiller.helper.loops import train_distill


def scrub(loaders, args):
    """
    Perform SCRUB unlearning using knowledge distillation.
    
    Args:
        loaders: Dictionary containing train_forget_loader, train_retain_loader,
                 valid_forget_loader, valid_retain_loader
        args: Object with attributes for hyperparameters (epochs, learning_rate,
              kd_T, msteps, t_opt_gamma, t_opt_alpha, check_path, print_accuracies)
    
    Returns:
        model_s: The unlearned student model
        history: Dictionary containing training history
    """
    # Load a pre-trained ResNet-18 model checkpoint (CIFAR-10)
    model_checkpoint = "./models/pretrained_cifar10.pt"

    if not os.path.exists(model_checkpoint):
        raise FileNotFoundError(f"Model checkpoint not found at {model_checkpoint}. Please run: python scripts/train_resnet18_cifar10.py")

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

    # Extracting the loaders that we want
    train_forget_loader = loaders['train_forget_loader']
    train_retain_loader = loaders['train_retain_loader']
    valid_forget_loader = loaders['valid_forget_loader']
    valid_retain_loader = loaders['valid_retain_loader']

    # Defining hyperparameters
    kd_T = args.kd_T
    learning_rate = args.learning_rate
    epochs = args.epochs
    msteps = args.msteps

    # Define teacher and student models
    model_t = copy.deepcopy(model).to(device)
    model_s = copy.deepcopy(model).to(device)

    # Module lists
    module_list = nn.ModuleList([model_s])
    trainable_list = nn.ModuleList([model_s])

    # Define loss functions
    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(kd_T)
    criterion_kd = DistillKL(kd_T)          # Placeholder for consistency
    criterion_list = nn.ModuleList([criterion_cls, criterion_div, criterion_kd])

    # Define the optimizer (configurable)
    weight_decay = getattr(args, "weight_decay", 1e-4)
    optimizer = optim.AdamW(trainable_list.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Add teacher model to the module list
    module_list.append(model_t)

    # Track accuracy
    tf_accs = []
    tr_accs = []
    vf_accs = []
    vr_accs = []
    losses = []
    epoch_list = []

    # Define validate args
    v_opt = SimpleNamespace()
    v_opt.print_freq = 0

    # Define train distill args
    t_opt = SimpleNamespace()
    t_opt.distill = 'kd'
    t_opt.gamma = args.t_opt_gamma      # Classification weight
    t_opt.alpha = args.t_opt_alpha      # KL divergence weight
    t_opt.beta = 0
    t_opt.print_freq = 0

    def fmt_metric(value, precision=6):
        if value is None:
            return "None"
        try:
            return f"{value:.{precision}f}"
        except (TypeError, ValueError):
            return repr(value)

    # Training loop
    for epoch in range(1, epochs + 1):
        print(f"[Epoch {epoch}/{epochs}] Starting epoch...")
        sys.stdout.flush()
        
        # Train model
        maximize_loss = 0
        if epoch <= msteps:
            print(f"  - Maximizing loss on forget set...")
            sys.stdout.flush()
            try:
                maximize_loss = train_distill(epoch, train_forget_loader, module_list, None, 
                                             criterion_list, optimizer, t_opt, "maximize", quiet=False)
                print(f"    Done: maximize_loss = {maximize_loss:.4f}")
                sys.stdout.flush()
            except Exception as e:
                print(f"    ERROR during maximize: {e}")
                sys.stdout.flush()
                raise
        
        print(f"  - Minimizing loss on retain set...")
        sys.stdout.flush()
        try:
            train_acc, train_loss = train_distill(epoch, train_retain_loader, module_list, None, 
                                                 criterion_list, optimizer, t_opt, "minimize", quiet=False)
            print(f"    Done: train_loss = {fmt_metric(train_loss, precision=8)}")
            print(f"    Raw: train_loss={train_loss!r}, train_acc={train_acc!r}")
            sys.stdout.flush()
        except Exception as e:
            print(f"    ERROR during minimize: {e}")
            sys.stdout.flush()
            raise

        losses.append(train_loss)
        scheduler.step()
        epoch_list.append(epoch)
        
        # Compute accuracies on all sets (optional)
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
            f"Epoch {epoch}: maximize loss: {fmt_metric(maximize_loss, precision=8)}, "
            f"minimize loss: {fmt_metric(train_loss, precision=8)}, "
            f"train_acc: {fmt_metric(train_acc, precision=6)}"
        )

        # Print epoch progress
        if args.print_accuracies and acc_dict is not None:
            print(f"   tr_acc: {acc_dict['tr_acc']:.4f}")
            print(f"   tf_acc: {acc_dict['tf_acc']:.4f}")
            print(f"   vr_acc: {acc_dict['vr_acc']:.4f}")
            print(f"   vf_acc: {acc_dict['vf_acc']:.4f}")

    # Save a copy of the student model to use in evaluation
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

    # Load config from YAML
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

    eval_retain_set = retain_set
    eval_forget_set = forget_set

    print(f"  Retain set size (used): {len(retain_set)}")
    print(f"  Forget set size (used): {len(forget_set)}")

    # ========== 3. CREATE DATA LOADERS ==========
    print("Creating data loaders...")
    pin_memory = args.pin_memory and torch.cuda.is_available()
    loaders = {
        'train_retain_loader': DataLoader(
            retain_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=pin_memory
        ),
        'train_forget_loader': DataLoader(
            forget_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=pin_memory
        ),
        'valid_retain_loader': DataLoader(
            eval_retain_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory
        ),
        'valid_forget_loader': DataLoader(
            eval_forget_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory
        ),
    }
    print(f"Data loaders created. Training will start now...")
    print(f"Configuration: epochs={args.epochs}, batch_size={args.batch_size}, device={torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")

    # ========== 4. RUN SCRUB UNLEARNING ==========
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