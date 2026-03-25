from __future__ import annotations

import argparse
import csv
import json
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import collections.abc

import numpy as np
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Subset
from tqdm import tqdm

# Python 3.12 requires random.sample's population to be a collections.abc.Sequence.
# torch.utils.data.Subset supports __getitem__ and __len__ but is not registered as
# a Sequence, so bad_teacher's random.sample call fails.  Register it once here
# so all strategy calls in this bridge work without touching third-party code.
collections.abc.Sequence.register(Subset)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bridge to MachineUnlearning engine")
    p.add_argument("--mode", required=True, choices=["baseline", "unlearn"])
    p.add_argument("--dataset", required=True, choices=["Cifar10", "Cifar100"])
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--split-file", required=True)
    p.add_argument("--model-out", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--engine-repo", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--epochs", type=int, required=True)
    p.add_argument("--batch-size", type=int, required=True)
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--optimizer", default="adam")
    p.add_argument("--momentum", type=float, default=0.5)

    p.add_argument(
        "--unlearning-method",
        choices=["scrub", "scrub_original", "scrub_teacher_loaded", "ssd", "bad_teacher", "amnesiac"],
    )
    p.add_argument("--baseline-model")
    p.add_argument("--run-name")

    p.add_argument("--scrub-mode", choices=["original", "teacher_loaded"])
    p.add_argument("--scrub-epochs", type=int)
    p.add_argument("--scrub-lr", type=float)
    p.add_argument("--scrub-distill-weight", type=float)
    p.add_argument("--scrub-forget-loss-weight", type=float)
    p.add_argument("--scrub-maximize-epochs", type=int)
    p.add_argument("--scrub-maximize-steps", type=int)
    p.add_argument("--scrub-minimize-steps", type=int)
    p.add_argument("--scrub-kd-temperature", type=float)
    p.add_argument("--scrub-weight-decay", type=float)
    p.add_argument("--scrub-momentum", type=float)
    p.add_argument("--scrub-lr-decay-epochs")
    p.add_argument("--scrub-lr-decay-rate", type=float)
    return p.parse_args()


def train_baseline(model, train_loader, test_loader, device, epochs, lr, optimizer_name, momentum):
    # Match optimizer choice from config while keeping defaults simple
    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    loss_func = nn.CrossEntropyLoss().to(device)
    best_model = model
    max_test_acc = -1.0

    for _ in tqdm(range(1, epochs + 1), desc="Baseline training"):
        # Standard supervised training pass
        model.train()
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.long().to(device)
            model.zero_grad()
            output = model(images)
            loss = loss_func(output, labels)
            loss.backward()
            optimizer.step()

        # Evaluate each epoch and keep best checkpoint by test accuracy
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                logits = model(images)
                pred = torch.argmax(logits, dim=1)
                correct += (pred == labels).sum().item()
                total += labels.numel()
        test_acc = correct / total if total > 0 else 0.0
        if test_acc >= max_test_acc:
            max_test_acc = test_acc
            best_model = model

    return best_model


def parse_int_list(value: str | None, default: list[int]) -> list[int]:
    if not value:
        return default
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def evaluate_accuracy_and_loss(model, loader, device):
    criterion = nn.CrossEntropyLoss().to(device)
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.long().to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            batch_size = labels.numel()
            loss_sum += loss.item() * batch_size
            total += batch_size
            pred = torch.argmax(logits, dim=1)
            correct += (pred == labels).sum().item()
    if total == 0:
        return 0.0, 0.0
    return 100.0 * correct / total, loss_sum / total


def adjust_scrub_learning_rate(epoch, optimizer, lr, lr_decay_epochs, lr_decay_rate):
    steps = np.sum(epoch > np.asarray(lr_decay_epochs))
    new_lr = lr * (lr_decay_rate ** steps) if steps > 0 else lr
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr
    return new_lr


def run_scrub_phase(
    model_s,
    model_t,
    loader,
    optimizer,
    criterion_cls,
    criterion_div,
    device,
    phase,
    distill_weight,
    forget_loss_weight,
):
    model_s.train()
    model_t.eval()

    total = 0
    correct = 0
    loss_sum = 0.0

    for images, labels in loader:
        images = images.float().to(device)
        labels = labels.long().to(device)

        logits_s = model_s(images)
        with torch.no_grad():
            logits_t = model_t(images)

        loss_cls = criterion_cls(logits_s, labels)
        loss_div = criterion_div(logits_s, logits_t)

        if phase == "maximize":
            loss = -forget_loss_weight * loss_div
        else:
            loss = loss_cls + distill_weight * loss_div

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = labels.numel()
        total += batch_size
        loss_sum += loss.item() * batch_size
        pred = torch.argmax(logits_s, dim=1)
        correct += (pred == labels).sum().item()

    if total == 0:
        return 0.0, 0.0
    return 100.0 * correct / total, loss_sum / total


def run_scrub(
    model,
    teacher_model,
    retain_loader,
    forget_loader,
    test_loader,
    device,
    scrub_cfg,
):
    model_s = deepcopy(model)
    model_t = deepcopy(teacher_model)

    criterion_cls = nn.CrossEntropyLoss().to(device)
    criterion_div = nn.KLDivLoss(reduction="batchmean")

    kd_temperature = float(scrub_cfg["kd_temperature"])

    class TemperatureDistillKL(nn.Module):
        def __init__(self, temperature: float):
            super().__init__()
            self.temperature = temperature

        def forward(self, y_s, y_t):
            p_s = torch.log_softmax(y_s / self.temperature, dim=1)
            p_t = torch.softmax(y_t / self.temperature, dim=1)
            return criterion_div(p_s, p_t) * (self.temperature ** 2)

    criterion_kd = TemperatureDistillKL(kd_temperature).to(device)

    optimizer = torch.optim.SGD(
        model_s.parameters(),
        lr=float(scrub_cfg["lr"]),
        momentum=float(scrub_cfg["momentum"]),
        weight_decay=float(scrub_cfg["weight_decay"]),
    )

    history_rows = []
    for epoch in range(1, int(scrub_cfg["epochs"]) + 1):
        current_lr = adjust_scrub_learning_rate(
            epoch=epoch,
            optimizer=optimizer,
            lr=float(scrub_cfg["lr"]),
            lr_decay_epochs=list(scrub_cfg["lr_decay_epochs"]),
            lr_decay_rate=float(scrub_cfg["lr_decay_rate"]),
        )

        maximize_loss = 0.0
        if epoch <= int(scrub_cfg["maximize_epochs"]):
            maximize_losses = []
            for _ in range(int(scrub_cfg["maximize_steps"])):
                _, max_loss = run_scrub_phase(
                    model_s=model_s,
                    model_t=model_t,
                    loader=forget_loader,
                    optimizer=optimizer,
                    criterion_cls=criterion_cls,
                    criterion_div=criterion_kd,
                    device=device,
                    phase="maximize",
                    distill_weight=float(scrub_cfg["distill_weight"]),
                    forget_loss_weight=float(scrub_cfg["forget_loss_weight"]),
                )
                maximize_losses.append(max_loss)
            maximize_loss = float(np.mean(maximize_losses)) if maximize_losses else 0.0

        retain_train_losses = []
        for _ in range(int(scrub_cfg["minimize_steps"])):
            _, retain_train_loss = run_scrub_phase(
                model_s=model_s,
                model_t=model_t,
                loader=retain_loader,
                optimizer=optimizer,
                criterion_cls=criterion_cls,
                criterion_div=criterion_kd,
                device=device,
                phase="minimize",
                distill_weight=float(scrub_cfg["distill_weight"]),
                forget_loss_weight=float(scrub_cfg["forget_loss_weight"]),
            )
            retain_train_losses.append(retain_train_loss)

        forget_acc, forget_loss = evaluate_accuracy_and_loss(model_s, forget_loader, device)
        retain_acc, retain_loss = evaluate_accuracy_and_loss(model_s, retain_loader, device)
        test_acc, test_loss = evaluate_accuracy_and_loss(model_s, test_loader, device)

        row = {
            "epoch": epoch,
            "lr": round(current_lr, 8),
            "maximize_forget_loss": round(maximize_loss, 6),
            "retain_train_loss": round(float(np.mean(retain_train_losses)) if retain_train_losses else 0.0, 6),
            "forget_acc": round(forget_acc, 6),
            "retain_acc": round(retain_acc, 6),
            "test_acc": round(test_acc, 6),
            "forget_loss": round(forget_loss, 6),
            "retain_loss": round(retain_loss, 6),
            "test_loss": round(test_loss, 6),
        }
        history_rows.append(row)
        print(
            "SCRUB epoch {epoch}: lr={lr:.6f} forget_acc={forget_acc:.4f} retain_acc={retain_acc:.4f} "
            "test_acc={test_acc:.4f} forget_loss={forget_loss:.6f} retain_loss={retain_loss:.6f}".format(**row)
        )

    return model_s, history_rows


def main() -> None:
    args = parse_args()

    # Load third-party engine modules from configured repository path
    engine_repo = Path(args.engine_repo).resolve()
    if not engine_repo.exists():
        raise FileNotFoundError(f"MachineUnlearning repo not found: {engine_repo}")
    if not (engine_repo / "src" / "__init__.py").exists():
        raise FileNotFoundError(
            f"MachineUnlearning repo is missing expected package layout at: {engine_repo / 'src'}"
        )

    sys.path.insert(0, str(engine_repo))
    from src import dataset as mu_dataset  # type: ignore
    from src import metrics as mu_metrics  # type: ignore
    from src import utils as mu_utils  # type: ignore
    from model import models as mu_models  # type: ignore
    from unlearn_strategies import strategies as mu_strategies  # type: ignore

    mu_utils.set_seed(args.seed)

    # Read canonical split generated by scripts/prepare_splits.py
    split_data = np.load(args.split_file)
    retain_indices = split_data["retain_indices"].tolist()
    forget_indices = split_data["forget_indices"].tolist()
    test_indices = split_data["test_indices"].tolist()

    split_path = Path(args.split_file)
    split_meta_path = split_path.with_suffix(".meta.json")
    if not split_meta_path.exists():
        raise FileNotFoundError(f"Missing split metadata file: {split_meta_path}")

    with split_meta_path.open("r", encoding="utf-8") as f:
        split_meta = json.load(f)

    if split_meta.get("split_mode") != "targeted_random":
        raise ValueError(
            f"Unsupported split_mode in metadata ({split_meta.get('split_mode')}). "
            "Only targeted_random is supported."
        )
    if "target_class" not in split_meta:
        raise ValueError(f"Split metadata is missing required key 'target_class': {split_meta_path}")

    target_class = int(split_meta["target_class"])

    train_dataset, test_dataset, num_classes, num_channels = mu_dataset.get_dataset(
        dataset_name=args.dataset,
        root=args.data_root,
        augment=False,
    )

    # Build subset datasets for retain/forget/test partitions
    retain_ds = Subset(train_dataset, retain_indices)
    forget_ds = Subset(train_dataset, forget_indices)
    test_ds = Subset(test_dataset, test_indices)

    baseline_train_ds = ConcatDataset([retain_ds, forget_ds])

    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")

    model = getattr(mu_models, "ResNet18")(num_classes=num_classes, input_channels=num_channels).to(device)
    scrub_history_rows = None
    scrub_settings = None
    scrub_teacher_source = None
    scrub_student_source = None

    if args.mode == "baseline":
        # Train a fresh baseline model on retain+forget train subset
        train_loader = DataLoader(baseline_train_ds, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

        model = train_baseline(
            model,
            train_loader,
            test_loader,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            optimizer_name=args.optimizer,
            momentum=args.momentum,
        )
        unlearning_method = "baseline"
    else:
        # Unlearning mode starts from baseline checkpoint and applies selected strategy
        if not args.unlearning_method:
            raise ValueError("--unlearning-method is required for unlearn mode")
        if not args.baseline_model:
            raise ValueError("--baseline-model is required for unlearn mode")

        model.load_state_dict(torch.load(args.baseline_model, map_location=device))

        unlearning_teacher = getattr(mu_models, "ResNet18")(num_classes=num_classes, input_channels=num_channels).to(device)
        retain_loader = DataLoader(retain_ds, batch_size=args.batch_size, shuffle=True)
        forget_loader = DataLoader(forget_ds, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

        if len(forget_ds) == 0:
            raise ValueError("forget_ds is empty; targeted_random split must include at least one forget sample.")

        # Guard against stale/incorrect split files by validating forget labels against metadata target_class.
        observed_labels = [int(forget_ds[i][1]) for i in range(len(forget_ds))]
        if not all(lbl == target_class for lbl in observed_labels):
            raise AssertionError("Split metadata target_class does not match forget-set labels.")

        strategy_args = SimpleNamespace(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            optimizer=args.optimizer,
            momentum=args.momentum,
            report_training=False,
            report_interval=5,
            seed=args.seed,
            gpu=args.device == "cuda",
        )

        if args.unlearning_method in {"scrub", "scrub_original", "scrub_teacher_loaded"}:
            scrub_mode = args.scrub_mode
            if args.unlearning_method == "scrub_original":
                scrub_mode = "original"
            elif args.unlearning_method == "scrub_teacher_loaded":
                scrub_mode = "teacher_loaded"
            if scrub_mode is None:
                scrub_mode = "original"

            if scrub_mode == "teacher_loaded":
                unlearning_teacher.load_state_dict(torch.load(args.baseline_model, map_location=device))
                scrub_teacher_source = str(args.baseline_model)
            else:
                scrub_teacher_source = "random_init_resnet18"
            scrub_student_source = str(args.baseline_model)

            scrub_settings = {
                "mode": scrub_mode,
                "epochs": args.scrub_epochs if args.scrub_epochs is not None else 3,
                "lr": args.scrub_lr if args.scrub_lr is not None else 0.0005,
                "distill_weight": args.scrub_distill_weight if args.scrub_distill_weight is not None else 0.001,
                "forget_loss_weight": args.scrub_forget_loss_weight if args.scrub_forget_loss_weight is not None else 1.0,
                "maximize_epochs": args.scrub_maximize_epochs if args.scrub_maximize_epochs is not None else 2,
                "maximize_steps": args.scrub_maximize_steps if args.scrub_maximize_steps is not None else 1,
                "minimize_steps": args.scrub_minimize_steps if args.scrub_minimize_steps is not None else 1,
                "kd_temperature": args.scrub_kd_temperature if args.scrub_kd_temperature is not None else 4.0,
                "weight_decay": args.scrub_weight_decay if args.scrub_weight_decay is not None else 5e-4,
                "momentum": args.scrub_momentum if args.scrub_momentum is not None else 0.9,
                "lr_decay_epochs": parse_int_list(args.scrub_lr_decay_epochs, [3, 5, 9]),
                "lr_decay_rate": args.scrub_lr_decay_rate if args.scrub_lr_decay_rate is not None else 0.1,
            }

            print(
                f"SCRUB setup: mode={scrub_mode} student_source={scrub_student_source} "
                f"teacher_source={scrub_teacher_source} forget_samples={len(forget_ds)} retain_samples={len(retain_ds)}"
            )
            print(
                f"SCRUB label check: target_class={target_class} forget_unique_labels={sorted(set(observed_labels))}"
            )

            model, scrub_history_rows = run_scrub(
                model=model,
                teacher_model=unlearning_teacher,
                retain_loader=retain_loader,
                forget_loader=forget_loader,
                test_loader=test_loader,
                device=device,
                scrub_cfg=scrub_settings,
            )
        else:
            strategy_fn = getattr(mu_strategies, args.unlearning_method)
            model = strategy_fn(
                args=strategy_args,
                model=model,
                unlearning_teacher=unlearning_teacher,
                unlearn_class=target_class,
                unlearn_loader=forget_loader,
                retain_loader=retain_loader,
                test_loader=test_loader,
                num_channels=num_channels,
                num_classes=num_classes,
                device=device,
            )
        unlearning_method = args.unlearning_method

    # Report basic utility metrics for downstream aggregation
    train_loader_eval = DataLoader(baseline_train_ds, batch_size=args.batch_size, shuffle=False)
    retain_loader_eval = DataLoader(retain_ds, batch_size=args.batch_size, shuffle=False)
    forget_loader_eval = DataLoader(forget_ds, batch_size=args.batch_size, shuffle=False)
    test_loader_eval = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    metrics = {
        "train_acc": mu_metrics.evaluate(model, train_loader_eval, device)["Acc"],
        "retain_acc": mu_metrics.evaluate(model, retain_loader_eval, device)["Acc"],
        "forget_acc": mu_metrics.evaluate(model, forget_loader_eval, device)["Acc"],
        "test_acc": mu_metrics.evaluate(model, test_loader_eval, device)["Acc"],
        "dataset": args.dataset,
        "seed": args.seed,
        "mode": args.mode,
        "unlearning_method": unlearning_method,
        "split_mode": "targeted_random",
        "target_class": target_class,
        "forget_count": int(split_meta.get("forget_count", len(forget_indices))),
        "forget_fraction": split_meta.get("forget_fraction"),
    }
    if args.run_name:
        metrics["run_name"] = args.run_name
    if scrub_settings is not None:
        metrics["scrub_mode"] = scrub_settings["mode"]
        metrics["scrub_settings"] = scrub_settings
        metrics["scrub_teacher_source"] = scrub_teacher_source
        metrics["scrub_student_source"] = scrub_student_source

    # Save model checkpoint and adjacent metrics JSON
    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_out)

    if scrub_history_rows is not None:
        scrub_history_path = model_out.with_suffix(".scrub_history.csv")
        with scrub_history_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(scrub_history_rows[0].keys()))
            writer.writeheader()
            writer.writerows(scrub_history_rows)
        metrics["scrub_history_csv"] = str(scrub_history_path)

    metrics_path = model_out.with_suffix(".metrics.json")
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
