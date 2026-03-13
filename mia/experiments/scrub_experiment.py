# Imports
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import sys
from types import SimpleNamespace
import os
import random
import numpy as np
import yaml

# Adding necessary paths to the system path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Framework imports
from data.loaders import load_dataset, get_num_classes
from utils.splits import ensure_retain_forget_split, ensure_targeted_random_unlearning_split, ensure_fully_random_unlearning_split
from utils.metrics import compute_accuracy, log_accuracies
from utils.transfer_setup import ensure_cifar10_from_cifar100_transfer_checkpoint
from torch.utils.data import DataLoader
import torchvision.models as models

# Third party code imports
from Third_Party_Code.SCRUB.thirdparty.repdistiller.distiller_zoo.KD import DistillKL
from Third_Party_Code.SCRUB.thirdparty.repdistiller.helper.loops import train_distill


def set_global_determinism(seed: int) -> None:
    """Set deterministic seeds for reproducible dataset splits and training order."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def freeze_bn(model):
    """Freeze BatchNorm running stats."""
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()


def scrub(loaders, args):
    """
    Perform SCRUB unlearning using knowledge distillation.
    """
    # Build/load CIFAR-10 checkpoint from a CIFAR-100-pretrained source when configured.
    model_checkpoint = getattr(args, "model_path", "./models/pretrained_cifar10.pt")
    source_checkpoint_cifar100 = getattr(args, "source_checkpoint_cifar100", None)
    if source_checkpoint_cifar100:
        model_checkpoint = ensure_cifar10_from_cifar100_transfer_checkpoint(
            source_checkpoint_cifar100=source_checkpoint_cifar100,
            target_checkpoint_cifar10=model_checkpoint,
            dataroot=getattr(args, "dataroot", "./data/raw"),
            finetune_epochs=int(getattr(args, "transfer_finetune_epochs", 10)),
            finetune_batch_size=int(getattr(args, "transfer_finetune_batch_size", 128)),
            finetune_learning_rate=float(getattr(args, "transfer_finetune_learning_rate", 0.001)),
            seed=int(getattr(args, "seed", 42)),
            num_workers=int(getattr(args, "num_workers", 2)),
            pin_memory=bool(getattr(args, "pin_memory", True)),
            force_rebuild=bool(getattr(args, "rebuild_transfer_checkpoint", False)),
        )

    if not os.path.exists(model_checkpoint):
        raise FileNotFoundError(
            f"Model checkpoint not found at {model_checkpoint}. "
            f"Please run: python scripts/train_resnet18_cifar10.py"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create model architecture and adapt for CIFAR-10
    model = models.resnet18(weights=None)
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

    results_path = getattr(args, "results_path", None)

    # Baseline evaluation (pre-unlearning)
    base_acc = None
    try:
        model.eval()
        base_acc = {
            'tr_acc': compute_accuracy(model, train_retain_loader, device),
            'tf_acc': compute_accuracy(model, train_forget_loader, device),
            'vr_acc': compute_accuracy(model, valid_retain_loader, device),
            'vf_acc': compute_accuracy(model, valid_forget_loader, device),
        }
        print(
            "Baseline - tr_acc: {tr:.4f}, tf_acc: {tf:.4f}, vr_acc: {vr:.4f}, vf_acc: {vf:.4f}".format(
                tr=base_acc['tr_acc'], tf=base_acc['tf_acc'], vr=base_acc['vr_acc'], vf=base_acc['vf_acc']
            )
        )
        if args.print_accuracies:
            log_accuracies(results_path, "baseline", base_acc)
    except Exception as e:
        print(f"Warning: baseline evaluation failed: {e}")

    # Hyperparameters
    kd_T = float(getattr(args, "kd_T", 1.0))
    learning_rate = float(getattr(args, "sgda_learning_rate", getattr(args, "learning_rate", 0.001)))
    epochs = int(getattr(args, "sgda_epochs", getattr(args, "epochs", 10)))
    msteps = int(getattr(args, "msteps", 10))
    weight_decay = float(getattr(args, "sgda_weight_decay", getattr(args, "weight_decay", 0.0)))
    momentum = float(getattr(args, "sgda_momentum", 0.9))
    forget_refresh_steps = int(getattr(args, "forget_refresh_steps", 0))
    forget_refresh_every = int(getattr(args, "forget_refresh_every", 1))
    forget_lr_multiplier = float(getattr(args, "forget_lr_multiplier", 0.5))
    refresh_lr_multiplier = float(getattr(args, "refresh_lr_multiplier", 0.05))
    max_forget_loss_magnitude = float(getattr(args, "max_forget_loss_magnitude", 1e4))
    min_selected_vr = float(getattr(args, "min_selected_vr", 0.0))
    min_forget_drop = float(getattr(args, "min_forget_drop", 0.0))
    max_retain_drop = float(getattr(args, "max_retain_drop", 1.0))

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
        lr=learning_rate * forget_lr_multiplier,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=True,
    )

    optimizer_retain = optim.SGD(
        trainable_list.parameters(),
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=True,
    )

    optimizer_refresh = optim.SGD(
        trainable_list.parameters(),
        lr=learning_rate * refresh_lr_multiplier,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=True,
    )

    # Track metrics
    tf_accs, tr_accs, vf_accs, vr_accs = [], [], [], []
    forget_phase_metrics = []
    losses, epoch_list = [], []

    selection_weight = float(getattr(args, "selection_weight", 1.0))
    delta_selection_weight = float(getattr(args, "delta_selection_weight", 1.0))
    best_tradeoff_score = float("-inf")
    best_epoch = "retain:0"
    best_state_dict = None
    best_vf_under_retain_floor = float("inf")
    best_epoch_under_retain_floor = "none"
    best_state_under_retain_floor = None
    best_gap_score = float("-inf")
    best_gap_epoch = "none"
    best_gap_state = None
    best_delta_score = float("-inf")
    best_delta_epoch = "none"
    best_delta_state = None

    def _update_selection_candidates(phase_label: str, acc_dict: dict) -> None:
        nonlocal best_tradeoff_score, best_epoch, best_state_dict
        nonlocal best_vf_under_retain_floor, best_epoch_under_retain_floor, best_state_under_retain_floor
        nonlocal best_gap_score, best_gap_epoch, best_gap_state
        nonlocal best_delta_score, best_delta_epoch, best_delta_state

        tradeoff_score = float(acc_dict['vr_acc']) - selection_weight * float(acc_dict['vf_acc'])
        if tradeoff_score > best_tradeoff_score:
            best_tradeoff_score = tradeoff_score
            best_epoch = phase_label
            best_state_dict = copy.deepcopy(model_s.state_dict())

        # Baseline-aware constraints for random-sample unlearning selection.
        meets_baseline_constraints = True
        retain_drop = 0.0
        forget_drop = 0.0
        if base_acc is not None:
            retain_drop = float(base_acc['vr_acc']) - float(acc_dict['vr_acc'])
            forget_drop = float(base_acc['vf_acc']) - float(acc_dict['vf_acc'])
            meets_baseline_constraints = (
                retain_drop <= max_retain_drop and forget_drop >= min_forget_drop
            )

            # Positive is better: more forgetting with less retain damage.
            delta_score = forget_drop - delta_selection_weight * retain_drop
            if (
                float(acc_dict['vr_acc']) >= min_selected_vr
                and meets_baseline_constraints
                and delta_score > best_delta_score
            ):
                best_delta_score = delta_score
                best_delta_epoch = phase_label
                best_delta_state = copy.deepcopy(model_s.state_dict())

        gap_score = float(acc_dict['vr_acc']) - float(acc_dict['vf_acc'])
        if (
            float(acc_dict['vr_acc']) >= min_selected_vr
            and meets_baseline_constraints
            and gap_score > best_gap_score
        ):
            best_gap_score = gap_score
            best_gap_epoch = phase_label
            best_gap_state = copy.deepcopy(model_s.state_dict())

        if float(acc_dict['vr_acc']) >= min_selected_vr and float(acc_dict['vf_acc']) < best_vf_under_retain_floor:
            best_vf_under_retain_floor = float(acc_dict['vf_acc'])
            best_epoch_under_retain_floor = phase_label
            best_state_under_retain_floor = copy.deepcopy(model_s.state_dict())

    # Training args
    t_opt = SimpleNamespace()
    t_opt.distill = 'kd'
    t_opt.gamma = float(getattr(args, "gamma", getattr(args, "t_opt_gamma", 1.0)))
    t_opt.alpha = float(getattr(args, "alpha", getattr(args, "t_opt_alpha", 1.0)))
    t_opt.beta = float(getattr(args, "beta", 0.0))
    t_opt.print_freq = 0

    # =======================
    # Phase 1: Forget Phase
    # =======================
    if msteps > 0:
        print("\n===== FORGET PHASE =====")
        freeze_bn(model_s)

        for f_epoch in range(1, msteps + 1):
            print(f"[Forget Phase {f_epoch}/{msteps}]")

            pre_step_state = copy.deepcopy(model_s.state_dict())
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
            print(f"  maximize_loss: {float(maximize_loss):.4f}")

            acc_dict = {
                'tr_acc': compute_accuracy(model_s, train_retain_loader, device),
                'tf_acc': compute_accuracy(model_s, train_forget_loader, device),
                'vr_acc': compute_accuracy(model_s, valid_retain_loader, device),
                'vf_acc': compute_accuracy(model_s, valid_forget_loader, device),
            }
            forget_phase_metrics.append(acc_dict)
            if args.print_accuracies:
                line = log_accuracies(results_path, f"forget_step {f_epoch}", acc_dict)
                print(f"  {line}")
            _update_selection_candidates(f"forget:{f_epoch}", acc_dict)

            max_loss_val = abs(float(maximize_loss))
            if (not np.isfinite(max_loss_val)) or max_loss_val > max_forget_loss_magnitude:
                print(
                    f"  WARNING: maximize_loss unstable ({maximize_loss}); restoring previous state and stopping forget phase."
                )
                model_s.load_state_dict(pre_step_state)
                break

    # =======================
    # Phase 2: Retain Phase
    # =======================
    print("\n===== RETAIN PHASE =====")
    for epoch in range(1, epochs + 1):
        print(f"[Retain Phase {epoch}/{epochs}]")
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

        if forget_refresh_steps > 0 and (forget_refresh_every <= 1 or epoch % forget_refresh_every == 0):
            freeze_bn(model_s)
            for refresh_step in range(1, forget_refresh_steps + 1):
                pre_refresh_state = copy.deepcopy(model_s.state_dict())
                refresh_loss = train_distill(
                    epoch,
                    train_forget_loader,
                    module_list,
                    None,
                    criterion_list,
                    optimizer_refresh,
                    t_opt,
                    "maximize",
                    quiet=False,
                )
                print(
                    f"  Forget refresh {refresh_step}/{forget_refresh_steps}: "
                    f"maximize_loss = {float(refresh_loss):.8f}"
                )
                try:
                    refresh_abs = abs(float(refresh_loss))
                    if (not np.isfinite(refresh_abs)) or refresh_abs > max_forget_loss_magnitude:
                        print(
                            f"  WARNING: forget refresh unstable ({refresh_loss}); restoring previous state and disabling refresh."
                        )
                        model_s.load_state_dict(pre_refresh_state)
                        forget_refresh_steps = 0
                        break
                except Exception:
                    pass

        losses.append(float(train_loss))
        epoch_list.append(epoch)

        # Evaluation
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

        print(
            f"  train_loss: {float(train_loss):.8f}, train_acc: {float(train_acc):.6f}"
        )

        if not np.isfinite(float(train_loss)):
            print("WARNING: train_loss became non-finite; stopping retain phase early to avoid model collapse.")
            break

        if args.print_accuracies:
            line = log_accuracies(results_path, f"retain_epoch {epoch}", acc_dict)
            print(f"  {line}")

        _update_selection_candidates(f"retain:{epoch}", acc_dict)

    if best_delta_state is not None:
        model_s.load_state_dict(best_delta_state)
        print(
            "Selected best epoch by delta criterion "
            f"(maximize forget_drop - {delta_selection_weight:.2f}*retain_drop with "
            f"vr_acc >= {min_selected_vr:.4f}, min_forget_drop >= {min_forget_drop:.4f}, "
            f"max_retain_drop <= {max_retain_drop:.4f}): "
            f"{best_delta_epoch} with score {best_delta_score:.4f}"
        )
    elif best_gap_state is not None:
        model_s.load_state_dict(best_gap_state)
        print(
            "Selected best epoch by constrained gap criterion "
            f"(maximize vr_acc-vf_acc with vr_acc >= {min_selected_vr:.4f}, "
            f"min_forget_drop >= {min_forget_drop:.4f}, max_retain_drop <= {max_retain_drop:.4f}): "
            f"{best_gap_epoch} with gap {best_gap_score:.4f}"
        )
    elif best_state_under_retain_floor is not None:
        model_s.load_state_dict(best_state_under_retain_floor)
        print(
            f"Selected best epoch by constrained criterion (min vf_acc with vr_acc >= {min_selected_vr:.4f}): "
            f"{best_epoch_under_retain_floor} with vf_acc {best_vf_under_retain_floor:.4f}"
        )
    elif best_state_dict is not None:
        model_s.load_state_dict(best_state_dict)
        print(
            f"Selected best epoch by tradeoff (vr_acc - {selection_weight:.2f}*vf_acc): "
            f"{best_epoch} with score {best_tradeoff_score:.4f}"
        )

    model_s.eval()
    selected_acc = {
        'tr_acc': compute_accuracy(model_s, train_retain_loader, device),
        'tf_acc': compute_accuracy(model_s, train_forget_loader, device),
        'vr_acc': compute_accuracy(model_s, valid_retain_loader, device),
        'vf_acc': compute_accuracy(model_s, valid_forget_loader, device),
    }
    print(
        "Selected model - tr_acc: {tr:.4f}, tf_acc: {tf:.4f}, vr_acc: {vr:.4f}, vf_acc: {vf:.4f}".format(
            tr=selected_acc['tr_acc'], tf=selected_acc['tf_acc'], vr=selected_acc['vr_acc'], vf=selected_acc['vf_acc']
        )
    )
    if args.print_accuracies:
        line = log_accuracies(results_path, "selected_model", selected_acc)
        print(f"   {line}")

    # Save student model
    if hasattr(args, 'check_path') and args.check_path is not None:
        check_dir = os.path.dirname(args.check_path)
        if check_dir:
            os.makedirs(check_dir, exist_ok=True)
        torch.save(model_s.state_dict(), args.check_path)

    history = {
        'losses': losses,
        'epoch_list': epoch_list,
        'tr_accs': tr_accs,
        'tf_accs': tf_accs,
        'vr_accs': vr_accs,
        'vf_accs': vf_accs,
        'forget_phase_metrics': forget_phase_metrics,
        'selected_acc': selected_acc,
        'best_epoch': best_epoch,
        'best_tradeoff_score': best_tradeoff_score,
        'best_epoch_under_retain_floor': best_epoch_under_retain_floor,
        'best_vf_under_retain_floor': best_vf_under_retain_floor,
        'best_gap_epoch': best_gap_epoch,
        'best_gap_score': best_gap_score,
        'best_delta_epoch': best_delta_epoch,
        'best_delta_score': best_delta_score,
    }

    return model_s, history


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
    if not hasattr(args, "split_dir"):
        args.split_dir = "./data/splits"

    set_global_determinism(int(args.seed))

    # ========== 1. LOAD DATA ==========
    print("Loading dataset...")
    dataset = load_dataset(
        dataset_name=args.dataset,
        root=args.dataroot,
        train=True
    )

    # ========== 2. CREATE SPLITS ==========
    split_dir = args.split_dir
    split_protocol = str(getattr(args, "split_protocol", "targeted_random")).strip().lower()

    has_count_keys = (
        getattr(args, "forget_count", None) is not None
        and (
            getattr(args, "retain_count", None) is not None
            or getattr(args, "retain_per_class", None) is not None
        )
        and (
            getattr(args, "left_out_count", None) is not None
            or getattr(args, "left_out_per_class", None) is not None
        )
    )

    if split_protocol == "fully_random" and has_count_keys:
        if getattr(args, "retain_count", None) is not None:
            retain_count = int(args.retain_count)
        else:
            retain_count = int(args.retain_per_class) * (int(get_num_classes(args.dataset)) - 1)
        if getattr(args, "left_out_count", None) is not None:
            left_out_count = int(args.left_out_count)
        else:
            left_out_count = int(args.left_out_per_class) * (int(get_num_classes(args.dataset)) - 1)
        forget_count = int(args.forget_count)

        retain_set, forget_set, left_out_set, _ = ensure_fully_random_unlearning_split(
            dataset=dataset,
            split_dir=split_dir,
            retain_count=retain_count,
            forget_count=forget_count,
            left_out_count=left_out_count,
            seed=int(args.seed),
            verbose=True,
        )
        print(
            "  Using fully-random protocol (forget from any class): "
            f"retain={len(retain_set)}, forget={len(forget_set)}, left_out={len(left_out_set)}"
        )
    elif getattr(args, "forget_class", None) is not None and has_count_keys:
        forget_class = int(args.forget_class)
        classes_excluding_forget = int(get_num_classes(args.dataset)) - 1
        if getattr(args, "retain_count", None) is not None:
            retain_count = int(args.retain_count)
        else:
            retain_count = int(args.retain_per_class) * classes_excluding_forget
        if getattr(args, "left_out_count", None) is not None:
            left_out_count = int(args.left_out_count)
        else:
            left_out_count = int(args.left_out_per_class) * classes_excluding_forget
        forget_count = int(args.forget_count)

        retain_set, forget_set, left_out_set, _ = ensure_targeted_random_unlearning_split(
            dataset=dataset,
            split_dir=split_dir,
            forget_class=forget_class,
            retain_count=retain_count,
            forget_count=forget_count,
            left_out_count=left_out_count,
            seed=int(args.seed),
            verbose=True,
        )
        print(
            "  Using targeted-random protocol (not per-class quotas): "
            f"retain={len(retain_set)}, forget={len(forget_set)}, left_out={len(left_out_set)}"
        )
    else:
        retain_set, forget_set, _ = ensure_retain_forget_split(
            dataset,
            split_dir=split_dir,
            forget_fraction=args.forget_fraction,
            seed=args.seed,
            verbose=True,
        )
        split_generator = torch.Generator().manual_seed(int(args.seed))
        retain_train_len = int(0.9 * len(retain_set))
        forget_train_len = int(0.9 * len(forget_set))
        retain_set, left_out_set = torch.utils.data.random_split(
            retain_set,
            [retain_train_len, len(retain_set) - retain_train_len],
            generator=split_generator,
        )
        forget_set, _ = torch.utils.data.random_split(
            forget_set,
            [forget_train_len, len(forget_set) - forget_train_len],
            generator=split_generator,
        )
        print(
            "  Using fallback random protocol: "
            f"retain={len(retain_set)}, forget={len(forget_set)}, left_out={len(left_out_set)}"
        )

    # ========== 4. CREATE DATA LOADERS ==========
    print("Creating data loaders...")
    pin_memory = bool(getattr(args, "pin_memory", True)) and torch.cuda.is_available()
    num_workers = int(getattr(args, "num_workers", 2))
    base_batch_size = int(getattr(args, "batch_size", 64))
    split_generator = torch.Generator().manual_seed(int(args.seed))

    sgda_batch_size = int(getattr(args, "sgda_batch_size", base_batch_size))
    del_batch_size = int(getattr(args, "del_batch_size", base_batch_size))
    loaders = {
        'train_retain_loader': DataLoader(
            retain_set,
            batch_size=sgda_batch_size,
            shuffle=True,
            generator=split_generator,
            num_workers=num_workers,
            pin_memory=pin_memory
        ),
        'train_forget_loader': DataLoader(
            forget_set,
            batch_size=del_batch_size,
            shuffle=True,
            generator=split_generator,
            num_workers=num_workers,
            pin_memory=pin_memory
        ),
        'valid_retain_loader': DataLoader(
            left_out_set,
            batch_size=sgda_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        ),
        'valid_forget_loader': DataLoader(
            forget_set,
            batch_size=del_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        ),
    }

    print(f"Data loaders created. Training will start now...")
    print(f"Configuration: epochs={getattr(args, 'sgda_epochs', getattr(args, 'epochs', 10))}, batch_size={base_batch_size}, "
          f"device={torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")

    # ========== 5. RUN SCRUB UNLEARNING ==========
    print("Running SCRUB unlearning...")
    model, history = scrub(loaders, args)

    print("\nSCRUB unlearning completed!")
    selected_acc = history.get('selected_acc')
    if selected_acc is not None:
        print(f"Selected train retain acc: {selected_acc['tr_acc']:.4f}")
        print(f"Selected valid retain acc: {selected_acc['vr_acc']:.4f}")
        print(f"Selected train forget acc: {selected_acc['tf_acc']:.4f}")
        print(f"Selected valid forget acc: {selected_acc['vf_acc']:.4f}")
    else:
        if history['tr_accs'][-1] is not None:
            print(f"Final train retain acc: {history['tr_accs'][-1]:.4f}")
        if history['vf_accs'][-1] is not None:
            print(f"Final valid forget acc: {history['vf_accs'][-1]:.4f}")

    return model, history


if __name__ == "__main__":
    main()
