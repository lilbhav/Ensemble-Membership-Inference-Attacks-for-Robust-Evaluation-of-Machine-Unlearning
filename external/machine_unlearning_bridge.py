from __future__ import annotations

import argparse
import collections.abc
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Subset
from tqdm import tqdm

# Python 3.12 requires random.sample's population to be a collections.abc.Sequence.
# torch.utils.data.Subset supports __getitem__ and __len__ but is not registered as
# a Sequence, so bad_teacher's random.sample call fails. Register once here so all
# strategy calls work without patching third-party files.
collections.abc.Sequence.register(Subset)

SUPPORTED_UNLEARNING_METHODS = ("scrub", "ssd", "bad_teacher", "amnesiac")


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
    p.add_argument("--optimizer", required=True)
    p.add_argument("--momentum", type=float, required=True)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--lr-scheduler", default="none", choices=["none", "cosine"])

    p.add_argument("--unlearning-method", choices=list(SUPPORTED_UNLEARNING_METHODS))
    p.add_argument("--baseline-model")
    p.add_argument("--run-name")
    p.add_argument("--method-config-json")
    return p.parse_args()


def train_baseline(
    model, train_loader, test_loader, device, epochs, lr, optimizer_name, momentum,
    weight_decay: float = 0.0, lr_scheduler: str = "none"
):
    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
        )
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay if weight_decay > 0 else 1e-4
        )
    else:
        raise ValueError(f"Unsupported baseline optimizer '{optimizer_name}'. Expected one of: ['sgd', 'adam']")

    if lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    else:
        scheduler = None

    loss_func = nn.CrossEntropyLoss().to(device)
    best_state = None
    max_test_acc = -1.0

    for _ in tqdm(range(1, epochs + 1), desc="Baseline training"):
        model.train()
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.long().to(device)
            model.zero_grad()
            output = model(images)
            loss = loss_func(output, labels)
            loss.backward()
            optimizer.step()
        if scheduler is not None:
            scheduler.step()

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
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def _load_method_cfg(args: argparse.Namespace) -> dict[str, Any]:
    if args.method_config_json is None:
        raise ValueError("Missing --method-config-json; method-specific config must be passed explicitly.")
    try:
        cfg = json.loads(args.method_config_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in --method-config-json: {exc}") from exc
    if not isinstance(cfg, dict):
        raise ValueError("--method-config-json must deserialize into a JSON object.")
    return cfg


def _validate_split_and_meta(
    args: argparse.Namespace,
    split_file: Path,
    split_data: Any,
    split_meta: dict[str, Any],
    train_size: int,
    test_size: int,
    train_labels: np.ndarray,
) -> dict[str, Any]:
    required_meta_keys = [
        "dataset",
        "seed",
        "split_mode",
        "target_class",
        "forget_count",
        "forget_fraction",
        "train_size",
        "retain_size",
        "test_size",
    ]
    missing_meta = [k for k in required_meta_keys if k not in split_meta]
    if missing_meta:
        raise ValueError(f"Split metadata missing required keys {missing_meta}: {split_file.with_suffix('.meta.json')}")

    if split_meta.get("split_mode") != "targeted_random":
        raise ValueError(f"Unsupported split_mode={split_meta.get('split_mode')}. Only targeted_random is supported.")
    if str(split_meta.get("dataset")) != str(args.dataset):
        raise ValueError(
            f"Split metadata dataset mismatch: metadata={split_meta.get('dataset')} args={args.dataset}"
        )
    if int(split_meta.get("seed")) != int(args.seed):
        raise ValueError(f"Split metadata seed mismatch: metadata={split_meta.get('seed')} args={args.seed}")

    retain_indices = np.asarray(split_data["retain_indices"], dtype=np.int64)
    forget_indices = np.asarray(split_data["forget_indices"], dtype=np.int64)
    test_indices = np.asarray(split_data["test_indices"], dtype=np.int64)

    if np.intersect1d(retain_indices, forget_indices).size > 0:
        raise AssertionError("retain_indices and forget_indices overlap; split file is invalid/stale.")

    expected_train_indices = np.arange(train_size, dtype=np.int64)
    observed_train_indices = np.sort(np.concatenate([retain_indices, forget_indices]))
    if not np.array_equal(observed_train_indices, expected_train_indices):
        raise AssertionError("retain_indices union forget_indices does not equal full training set.")

    expected_test_indices = np.arange(test_size, dtype=np.int64)
    if not np.array_equal(np.sort(test_indices), expected_test_indices):
        raise AssertionError("test_indices are not canonical; expected full unchanged test set [0..N-1].")

    forget_labels = train_labels[forget_indices] if forget_indices.size > 0 else np.array([], dtype=np.int64)
    target_class = int(split_meta["target_class"])
    if not np.all(forget_labels == target_class):
        unique_labels = sorted({int(x) for x in forget_labels.tolist()})
        raise AssertionError(
            f"target_class mismatch: metadata target_class={target_class}, forget labels={unique_labels}"
        )

    if int(split_meta["forget_count"]) != int(forget_indices.size):
        raise AssertionError(
            f"Metadata forget_count={split_meta['forget_count']} mismatches split forget size={forget_indices.size}."
        )
    if int(split_meta["retain_size"]) != int(retain_indices.size):
        raise AssertionError(
            f"Metadata retain_size={split_meta['retain_size']} mismatches split retain size={retain_indices.size}."
        )
    if int(split_meta["train_size"]) != int(train_size):
        raise AssertionError(
            f"Metadata train_size={split_meta['train_size']} mismatches dataset train size={train_size}."
        )
    if int(split_meta["test_size"]) != int(test_size):
        raise AssertionError(f"Metadata test_size={split_meta['test_size']} mismatches dataset test size={test_size}.")

    return {
        "retain_indices": retain_indices,
        "forget_indices": forget_indices,
        "test_indices": test_indices,
        "forget_labels": forget_labels,
        "target_class": target_class,
    }


def _ignored_parameters(
    unlearning_method: str,
    method_cfg: dict[str, Any],
    consumed_by_bridge: set[str] | None = None,
) -> list[dict[str, str]]:
    # Third-party strategy implementations for these methods currently use hard-coded
    # constants instead of args/method_cfg values.
    consumed_by_bridge = consumed_by_bridge or set()
    consumed_by_strategy_map: dict[str, set[str]] = {
        "scrub": {
            "gamma",
            "epochs",
            "lr",
            "distill_weight",
            "forget_loss_weight",
            "maximize_epochs",
            "maximize_steps",
            "minimize_steps",
            "post_maximize_repair_scale",
            "kd_temperature",
            "weight_decay",
            "momentum",
            "lr_decay_epochs",
            "lr_decay_rate",
        },
        "bad_teacher": {
            "epochs",
            "lr",
            "batch_size",
            "optimizer",
            "momentum",
            "kl_temperature",
            "retain_subset_fraction",
        },
        "amnesiac": {
            "epochs",
            "batch_size",
            "optimizer",
        },
        "ssd": {
            "lr",
            "optimizer",
            "momentum",
            "lower_bound",
            "exponent",
            "magnitude_diff",
            "min_layer",
            "max_layer",
            "forget_threshold",
            "dampening_constant",
            "selection_weighting",
        },
    }
    consumed_by_strategy = consumed_by_strategy_map.get(unlearning_method, set())

    ignored = []
    for key in sorted(method_cfg.keys()):
        if key in consumed_by_bridge:
            continue
        if key in consumed_by_strategy:
            continue
        ignored.append(
            {
                "parameter": key,
                "reason": "Not consumed by the third-party strategy implementation",
                "location": f"Third_Party_Code/MachineUnlearning/unlearn_strategies/strategies.py::{unlearning_method}",
            }
        )
    return ignored


def main() -> None:
    args = parse_args()

    engine_repo = Path(args.engine_repo).resolve()
    if not engine_repo.exists():
        raise FileNotFoundError(f"MachineUnlearning repo not found: {engine_repo}")
    if not (engine_repo / "src" / "__init__.py").exists():
        raise FileNotFoundError(
            f"MachineUnlearning repo is missing expected package layout at: {engine_repo / 'src'}"
        )

    sys.path.insert(0, str(engine_repo))
    from model import models as mu_models  # type: ignore
    from src import dataset as mu_dataset  # type: ignore
    from src import metrics as mu_metrics  # type: ignore
    from src import utils as mu_utils  # type: ignore
    from unlearn_strategies import strategies as mu_strategies  # type: ignore

    mu_utils.set_seed(args.seed)

    split_file = Path(args.split_file)
    if not split_file.exists():
        raise FileNotFoundError(f"Missing split file: {split_file}")
    split_data = np.load(split_file)

    split_meta_path = split_file.with_suffix(".meta.json")
    if not split_meta_path.exists():
        raise FileNotFoundError(f"Missing split metadata file: {split_meta_path}")
    with split_meta_path.open("r", encoding="utf-8") as f:
        split_meta = json.load(f)

    train_dataset, test_dataset, num_classes, num_channels = mu_dataset.get_dataset(
        dataset_name=args.dataset,
        root=args.data_root,
        augment=False,
    )
    train_size = len(train_dataset)
    test_size = len(test_dataset)
    train_labels = np.array([int(train_dataset[i][1]) for i in range(train_size)], dtype=np.int64)

    split_validated = _validate_split_and_meta(
        args=args,
        split_file=split_file,
        split_data=split_data,
        split_meta=split_meta,
        train_size=train_size,
        test_size=test_size,
        train_labels=train_labels,
    )

    retain_ds = Subset(train_dataset, split_validated["retain_indices"].tolist())
    forget_ds = Subset(train_dataset, split_validated["forget_indices"].tolist())
    test_ds = Subset(test_dataset, split_validated["test_indices"].tolist())
    baseline_train_ds = ConcatDataset([retain_ds, forget_ds])

    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    model = getattr(mu_models, "ResNet18")(num_classes=num_classes, input_channels=num_channels).to(device)

    strategy_fn_name = None
    method_cfg: dict[str, Any] = {}
    strategy_args_payload: dict[str, Any] = {}
    ignored_params: list[dict[str, str]] = []
    consumed_method_cfg_keys_by_bridge: set[str] = set()
    scrub_teacher_source = None

    if args.mode == "baseline":
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
            weight_decay=args.weight_decay,
            lr_scheduler=args.lr_scheduler,
        )
        unlearning_method = "baseline"
    else:
        if not args.unlearning_method:
            raise ValueError("--unlearning-method is required for unlearn mode")
        if args.unlearning_method not in SUPPORTED_UNLEARNING_METHODS:
            raise ValueError(
                f"Unsupported unlearning method {args.unlearning_method}; supported={list(SUPPORTED_UNLEARNING_METHODS)}"
            )
        if not args.baseline_model:
            raise ValueError("--baseline-model is required for unlearn mode")

        baseline_model_path = Path(args.baseline_model)
        if not baseline_model_path.exists():
            raise FileNotFoundError(f"Missing baseline checkpoint path: {baseline_model_path}")
        model.load_state_dict(torch.load(baseline_model_path, map_location=device))

        if len(forget_ds) == 0:
            raise ValueError("forget_ds is empty; targeted_random split must include at least one forget sample.")

        method_cfg = _load_method_cfg(args)
        strategy_args_payload = {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "optimizer": args.optimizer,
            "momentum": args.momentum,
            "report_training": False,
            "report_interval": 5,
            "seed": args.seed,
            "gpu": args.device == "cuda",
            "method_config": method_cfg,
            **method_cfg,
        }
        strategy_args = SimpleNamespace(**strategy_args_payload)

        unlearning_teacher = getattr(mu_models, "ResNet18")(num_classes=num_classes, input_channels=num_channels).to(device)
        if args.unlearning_method == "scrub":
            scrub_mode = str(method_cfg.get("mode", "teacher_loaded"))
            if scrub_mode not in {"teacher_loaded", "random_init"}:
                raise ValueError(
                    "Invalid scrub mode in method config. Supported values: ['teacher_loaded', 'random_init']"
                )
            consumed_method_cfg_keys_by_bridge.add("mode")
            if scrub_mode == "teacher_loaded":
                unlearning_teacher.load_state_dict(torch.load(baseline_model_path, map_location=device))
                scrub_teacher_source = str(baseline_model_path)
            else:
                scrub_teacher_source = "random_init_resnet18"

        retain_loader = DataLoader(retain_ds, batch_size=args.batch_size, shuffle=True)
        forget_loader = DataLoader(forget_ds, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

        strategy_fn_name = f"unlearn_strategies.strategies.{args.unlearning_method}"
        ignored_params = _ignored_parameters(
            args.unlearning_method,
            method_cfg,
            consumed_by_bridge=consumed_method_cfg_keys_by_bridge,
        )
        print("Unlearning bridge verification")
        print(f"  method={args.unlearning_method}")
        print(f"  strategy_called={strategy_fn_name}")
        print(f"  baseline_checkpoint={baseline_model_path}")
        print(f"  split_metadata={split_meta_path}")
        print(f"  target_class={split_validated['target_class']}")
        print(f"  forget_count={len(forget_ds)}")
        print(
            "  loader_sizes="
            f"retain_samples={len(retain_ds)}, forget_samples={len(forget_ds)}, test_samples={len(test_ds)}, "
            f"retain_batches={len(retain_loader)}, forget_batches={len(forget_loader)}, test_batches={len(test_loader)}"
        )
        if scrub_teacher_source is not None:
            print(f"  scrub_teacher_source={scrub_teacher_source}")
        print(f"  method_hyperparameters_passed={method_cfg}")
        if ignored_params:
            print("  ignored_method_parameters_detected=True")
            for item in ignored_params:
                print(
                    f"    - {item['parameter']} ignored in {item['location']} ({item['reason']})"
                )

        strategy_fn = getattr(mu_strategies, args.unlearning_method)
        model = strategy_fn(
            args=strategy_args,
            model=model,
            unlearning_teacher=unlearning_teacher,
            unlearn_class=split_validated["target_class"],
            unlearn_loader=forget_loader,
            retain_loader=retain_loader,
            test_loader=test_loader,
            num_channels=num_channels,
            num_classes=num_classes,
            device=device,
        )
        unlearning_method = args.unlearning_method

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
        "target_class": int(split_meta["target_class"]),
        "forget_count": int(split_meta["forget_count"]),
        "forget_fraction": float(split_meta["forget_fraction"]),
    }
    if args.run_name:
        metrics["run_name"] = args.run_name

    debug_payload = {
        "mode": args.mode,
        "method": unlearning_method,
        "strategy_called": strategy_fn_name,
        "baseline_checkpoint_path_loaded": None if not args.baseline_model else str(Path(args.baseline_model)),
        "split_file_loaded": str(split_file),
        "split_metadata_path_loaded": str(split_meta_path),
        "target_class": int(split_validated["target_class"]),
        "forget_count": int(len(forget_ds)),
        "loader_sizes": {
            "retain_samples": int(len(retain_ds)),
            "forget_samples": int(len(forget_ds)),
            "test_samples": int(len(test_ds)),
            "retain_batches": int(len(DataLoader(retain_ds, batch_size=args.batch_size, shuffle=False))),
            "forget_batches": int(len(DataLoader(forget_ds, batch_size=args.batch_size, shuffle=False))),
            "test_batches": int(len(DataLoader(test_ds, batch_size=args.batch_size, shuffle=False))),
        },
        "method_hyperparameters_passed": method_cfg,
        "consumed_method_cfg_keys_by_bridge": sorted(consumed_method_cfg_keys_by_bridge),
        "scrub_teacher_source": scrub_teacher_source,
        "strategy_args_payload": strategy_args_payload,
        "ignored_method_parameters": ignored_params,
    }

    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_out)

    metrics_path = model_out.with_suffix(".metrics.json")
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    debug_path = model_out.with_suffix(".debug.json")
    with debug_path.open("w", encoding="utf-8") as f:
        json.dump(debug_payload, f, indent=2)


if __name__ == "__main__":
    main()
