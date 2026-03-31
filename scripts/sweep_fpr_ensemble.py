from __future__ import annotations

import argparse
import csv
from itertools import combinations
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sweep calibrated ensemble behavior across multiple target FPR values.")
    p.add_argument("--results-root", default="results")
    p.add_argument("--dataset", default="Cifar10")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--method", default="bad_teacher")
    p.add_argument("--target", default="forget_vs_test")
    p.add_argument("--attack-seed", type=int, default=0)
    p.add_argument("--fprs", nargs="*", type=float, default=[0.05, 0.02, 0.01, 0.005, 0.001])
    p.add_argument("--out-dir", default=None)
    return p.parse_args()


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def threshold_at_target_fpr(y: np.ndarray, s: np.ndarray, target_fpr: float) -> float:
    fpr, tpr, thresholds = roc_curve(y, s)
    eligible = [
        (float(th), float(tv), float(fv))
        for fv, tv, th in zip(fpr, tpr, thresholds)
        if fv <= target_fpr
    ]
    if eligible:
        th, _, _ = max(eligible, key=lambda item: (item[1], -item[2], item[0]))
        return th
    return float(np.max(s) + 1e-12)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def confusion(y: np.ndarray, p: np.ndarray) -> tuple[float, float, float]:
    pos = y == 1
    neg = y == 0
    tp = int(((p == 1) & pos).sum())
    fp = int(((p == 1) & neg).sum())
    tn = int(((p == 0) & neg).sum())
    fn = int(((p == 0) & pos).sum())
    tpr = tp / max(1, tp + fn)
    fpr = fp / max(1, fp + tn)
    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    return float(tpr), float(fpr), float(acc)


def jaccard(a: set[int], b: set[int]) -> float:
    denom = len(a | b)
    if denom == 0:
        return 0.0
    return float(len(a & b) / denom)


def safe_auc(y: np.ndarray, s: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def main() -> None:
    args = parse_args()
    root = Path(args.results_root)
    out_dir = Path(args.out_dir) if args.out_dir else (root / "aggregate")

    mia_dir = root / "mia" / args.dataset / f"seed_{args.seed}" / args.method
    if not mia_dir.exists():
        print(f"Missing MIA directory: {mia_dir}")
        return

    files = sorted(mia_dir.glob(f"*/{args.target}_attack_seed_{args.attack_seed}.csv"))
    if not files:
        print(
            "No per-attack score files found for slice: "
            f"dataset={args.dataset}, seed={args.seed}, method={args.method}, target={args.target}, attack_seed={args.attack_seed}"
        )
        return

    attack_to_scores: dict[str, np.ndarray] = {}
    labels: np.ndarray | None = None
    sample_ids: np.ndarray | None = None

    for f in files:
        attack = f.parent.name
        rows = read_rows(f)
        if not rows:
            continue
        s = np.array([float(r["score"]) for r in rows], dtype=float)
        y = np.array([int(r["true_membership"]) for r in rows], dtype=int)
        sid = np.array([int(r["sample_id"]) for r in rows], dtype=int)

        if labels is None:
            labels = y
            sample_ids = sid
        else:
            if len(y) != len(labels) or not np.array_equal(y, labels) or not np.array_equal(sid, sample_ids):
                raise ValueError(f"Mismatched sample order/labels across attacks; offending file: {f}")

        attack_to_scores[attack] = s

    if not attack_to_scores or labels is None:
        print("No valid attack files loaded.")
        return

    attacks = sorted(attack_to_scores.keys())
    y = labels
    sid = sample_ids if sample_ids is not None else np.arange(len(y))

    # Global score-direction normalization per attack: if AUC(score) < 0.5, flip sign.
    attack_to_scores_oriented: dict[str, np.ndarray] = {}
    attack_flip_info: dict[str, dict] = {}
    for attack in attacks:
        s = attack_to_scores[attack]
        auc_orig = safe_auc(y, s)
        flip = bool(not np.isnan(auc_orig) and auc_orig < 0.5)
        s_used = -s if flip else s
        auc_after = safe_auc(y, s_used)
        attack_to_scores_oriented[attack] = s_used
        attack_flip_info[attack] = {
            "sign_flip": flip,
            "auc_after_flip": auc_after,
        }

    sweep_rows: list[dict] = []
    coverage_rows: list[dict] = []
    activation_rows: list[dict] = []
    jaccard_rows: list[dict] = []
    attack_summary_5pct_rows: list[dict] = []

    first_active: dict[str, float | None] = {a: None for a in attacks}

    for target_fpr in args.fprs:
        attack_preds: dict[str, np.ndarray] = {}
        attack_positive_sets: dict[str, set[int]] = {}

        for attack in attacks:
            s = attack_to_scores_oriented[attack]
            thr = threshold_at_target_fpr(y, s, target_fpr)
            p = (s >= thr).astype(int)
            attack_preds[attack] = p
            cov = float(np.mean(p))
            positives = set(sid[p == 1].tolist())
            attack_positive_sets[attack] = positives
            if cov > 0 and first_active[attack] is None:
                first_active[attack] = target_fpr

            coverage_rows.append(
                {
                    "dataset": args.dataset,
                    "base_seed": args.seed,
                    "unlearning_method": args.method,
                    "target": args.target,
                    "attack_seed": args.attack_seed,
                    "target_fpr": target_fpr,
                    "attack": attack,
                    "coverage_fraction": round(cov, 6),
                    "threshold": float(thr),
                    "sign_flip": bool(attack_flip_info[attack]["sign_flip"]),
                    "auc_after_flip": round(float(attack_flip_info[attack]["auc_after_flip"]), 6),
                }
            )

            if abs(target_fpr - 0.05) < 1e-12:
                attack_summary_5pct_rows.append(
                    {
                        "attack": attack,
                        "sign_flip": bool(attack_flip_info[attack]["sign_flip"]),
                        "AUC_after_flip": round(float(attack_flip_info[attack]["auc_after_flip"]), 6),
                        "threshold": float(thr),
                        "coverage@5%": round(cov, 6),
                    }
                )

        for a1, a2 in combinations(attacks, 2):
            jaccard_rows.append(
                {
                    "dataset": args.dataset,
                    "base_seed": args.seed,
                    "unlearning_method": args.method,
                    "target": args.target,
                    "attack_seed": args.attack_seed,
                    "target_fpr": target_fpr,
                    "attack_a": a1,
                    "attack_b": a2,
                    "jaccard": round(jaccard(attack_positive_sets[a1], attack_positive_sets[a2]), 6),
                }
            )

        votes = np.zeros_like(y, dtype=int)
        for attack in attacks:
            votes += attack_preds[attack]

        p_or = (votes >= 1).astype(int)
        p_k2 = (votes >= 2).astype(int)
        p_k3 = (votes >= 3).astype(int)

        tpr_or, fpr_or, acc_or = confusion(y, p_or)
        tpr_k2, fpr_k2, acc_k2 = confusion(y, p_k2)
        tpr_k3, fpr_k3, acc_k3 = confusion(y, p_k3)

        sweep_rows.append(
            {
                "dataset": args.dataset,
                "base_seed": args.seed,
                "unlearning_method": args.method,
                "target": args.target,
                "attack_seed": args.attack_seed,
                "target_fpr": target_fpr,
                "coverage_or": round(float(np.mean(p_or)), 6),
                "coverage_k2": round(float(np.mean(p_k2)), 6),
                "coverage_k3": round(float(np.mean(p_k3)), 6),
                "tpr_or": round(tpr_or, 6),
                "fpr_or": round(fpr_or, 6),
                "acc_or": round(acc_or, 6),
                "tpr_k2": round(tpr_k2, 6),
                "fpr_k2": round(fpr_k2, 6),
                "acc_k2": round(acc_k2, 6),
                "tpr_k3": round(tpr_k3, 6),
                "fpr_k3": round(fpr_k3, 6),
                "acc_k3": round(acc_k3, 6),
            }
        )

    for attack in attacks:
        activation_rows.append(
            {
                "dataset": args.dataset,
                "base_seed": args.seed,
                "unlearning_method": args.method,
                "target": args.target,
                "attack_seed": args.attack_seed,
                "attack": attack,
                "first_active_target_fpr": first_active[attack] if first_active[attack] is not None else "never_active",
            }
        )

    base = f"fpr_sweep_{args.dataset}_seed_{args.seed}_{args.method}_{args.target}_attackseed_{args.attack_seed}"
    write_csv(
        out_dir / f"{base}.csv",
        sweep_rows,
        [
            "dataset", "base_seed", "unlearning_method", "target", "attack_seed", "target_fpr",
            "coverage_or", "coverage_k2", "coverage_k3",
            "tpr_or", "fpr_or", "acc_or",
            "tpr_k2", "fpr_k2", "acc_k2",
            "tpr_k3", "fpr_k3", "acc_k3",
        ],
    )
    write_csv(
        out_dir / f"{base}_coverage_per_attack.csv",
        coverage_rows,
        [
            "dataset", "base_seed", "unlearning_method", "target", "attack_seed", "target_fpr", "attack", "coverage_fraction", "threshold", "sign_flip", "auc_after_flip",
        ],
    )
    write_csv(
        out_dir / f"{base}_pairwise_jaccard.csv",
        jaccard_rows,
        [
            "dataset", "base_seed", "unlearning_method", "target", "attack_seed", "target_fpr", "attack_a", "attack_b", "jaccard",
        ],
    )
    write_csv(
        out_dir / f"{base}_activation_order.csv",
        activation_rows,
        [
            "dataset", "base_seed", "unlearning_method", "target", "attack_seed", "attack", "first_active_target_fpr",
        ],
    )
    write_csv(
        out_dir / f"{base}_attack_flip_5pct_table.csv",
        attack_summary_5pct_rows,
        [
            "attack", "sign_flip", "AUC_after_flip", "threshold", "coverage@5%",
        ],
    )

    print(f"Saved sweep summary: {out_dir / f'{base}.csv'}")
    print(f"Saved per-attack coverage: {out_dir / f'{base}_coverage_per_attack.csv'}")
    print(f"Saved pairwise Jaccard: {out_dir / f'{base}_pairwise_jaccard.csv'}")
    print(f"Saved activation order: {out_dir / f'{base}_activation_order.csv'}")
    print(f"Saved 5% attack table: {out_dir / f'{base}_attack_flip_5pct_table.csv'}")


if __name__ == "__main__":
    main()
