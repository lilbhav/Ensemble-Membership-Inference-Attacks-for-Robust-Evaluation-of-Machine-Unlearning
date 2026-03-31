from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze silent attacks and recommend score-sign flips.")
    p.add_argument("--results-root", default="results")
    p.add_argument("--dataset", default="Cifar10")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--method", default="bad_teacher")
    p.add_argument("--target", default="forget_vs_test")
    p.add_argument("--attack-seed", type=int, default=0)
    p.add_argument("--attacks", nargs="*", default=["aug", "losstraj", "reference", "shokri", "yeom"])
    p.add_argument("--target-fpr", type=float, default=0.05)
    p.add_argument("--flip-margin", type=float, default=0.02, help="Meaningful AUC margin for sign flip recommendation.")
    p.add_argument("--near-half", type=float, default=0.03, help="AUC near-chance tolerance around 0.5.")
    p.add_argument("--low-std", type=float, default=0.02, help="Low-variance threshold for collapse classification.")
    p.add_argument("--out-dir", default=None)
    return p.parse_args()


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def threshold_at_fpr(y: np.ndarray, s: np.ndarray, target_fpr: float) -> float:
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


def safe_auc(y: np.ndarray, s: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def main() -> None:
    args = parse_args()
    root = Path(args.results_root)
    out_dir = Path(args.out_dir) if args.out_dir else (root / "aggregate")

    base = root / "mia" / args.dataset / f"seed_{args.seed}" / args.method
    if not base.exists():
        print(f"Missing MIA directory: {base}")
        return

    diagnosis_rows: list[dict] = []
    updated_rows: list[dict] = []

    for attack in args.attacks:
        path = base / attack / f"{args.target}_attack_seed_{args.attack_seed}.csv"
        if not path.exists():
            diagnosis_rows.append(
                {
                    "dataset": args.dataset,
                    "base_seed": args.seed,
                    "unlearning_method": args.method,
                    "target": args.target,
                    "attack_seed": args.attack_seed,
                    "attack": attack,
                    "status": "missing_file",
                }
            )
            continue

        rows = read_rows(path)
        if not rows:
            diagnosis_rows.append(
                {
                    "dataset": args.dataset,
                    "base_seed": args.seed,
                    "unlearning_method": args.method,
                    "target": args.target,
                    "attack_seed": args.attack_seed,
                    "attack": attack,
                    "status": "empty_file",
                }
            )
            continue

        y = np.array([int(r["true_membership"]) for r in rows], dtype=int)
        s = np.array([float(r["score"]) for r in rows], dtype=float)
        member = s[y == 1]
        nonmember = s[y == 0]

        auc_pos = safe_auc(y, s)
        auc_neg = safe_auc(y, -s)
        auc_gap = auc_neg - auc_pos
        flip = bool(auc_gap >= args.flip_margin)

        member_mean = float(np.mean(member)) if len(member) else float("nan")
        nonmember_mean = float(np.mean(nonmember)) if len(nonmember) else float("nan")
        member_std = float(np.std(member)) if len(member) else float("nan")
        nonmember_std = float(np.std(nonmember)) if len(nonmember) else float("nan")

        near_chance = (
            not np.isnan(auc_pos)
            and not np.isnan(auc_neg)
            and abs(auc_pos - 0.5) <= args.near_half
            and abs(auc_neg - 0.5) <= args.near_half
        )
        low_var = (
            not np.isnan(member_std)
            and not np.isnan(nonmember_std)
            and member_std <= args.low_std
            and nonmember_std <= args.low_std
        )
        collapsed = bool(near_chance and low_var)

        s_used = -s if flip else s
        auc_used = safe_auc(y, s_used)
        thr = threshold_at_fpr(y, s_used, args.target_fpr)
        p = (s_used >= thr).astype(int)
        coverage = float(np.mean(p))

        diagnosis_rows.append(
            {
                "dataset": args.dataset,
                "base_seed": args.seed,
                "unlearning_method": args.method,
                "target": args.target,
                "attack_seed": args.attack_seed,
                "attack": attack,
                "auc_score": round(auc_pos, 6),
                "auc_neg_score": round(auc_neg, 6),
                "auc_neg_minus_auc": round(auc_gap, 6),
                "mean_member": round(member_mean, 6),
                "mean_nonmember": round(nonmember_mean, 6),
                "std_member": round(member_std, 6),
                "std_nonmember": round(nonmember_std, 6),
                "flip_recommended": flip,
                "collapsed": collapsed,
                "status": "ok",
            }
        )

        updated_rows.append(
            {
                "dataset": args.dataset,
                "base_seed": args.seed,
                "unlearning_method": args.method,
                "target": args.target,
                "attack_seed": args.attack_seed,
                "attack": attack,
                "target_fpr": args.target_fpr,
                "score_direction_used": "negated" if flip else "original",
                "threshold": round(float(thr), 10),
                "auc": round(float(auc_used), 6),
                "coverage_at_target_fpr": round(float(coverage), 6),
                "collapsed": collapsed,
            }
        )

    base_name = (
        f"silent_attack_analysis_{args.dataset}_seed_{args.seed}_{args.method}_{args.target}_attackseed_{args.attack_seed}"
    )
    diagnosis_path = out_dir / f"{base_name}.csv"
    updated_path = out_dir / f"{base_name}_updated_5pct_table.csv"

    write_csv(
        diagnosis_path,
        diagnosis_rows,
        [
            "dataset",
            "base_seed",
            "unlearning_method",
            "target",
            "attack_seed",
            "attack",
            "auc_score",
            "auc_neg_score",
            "auc_neg_minus_auc",
            "mean_member",
            "mean_nonmember",
            "std_member",
            "std_nonmember",
            "flip_recommended",
            "collapsed",
            "status",
        ],
    )
    write_csv(
        updated_path,
        updated_rows,
        [
            "dataset",
            "base_seed",
            "unlearning_method",
            "target",
            "attack_seed",
            "attack",
            "target_fpr",
            "score_direction_used",
            "threshold",
            "auc",
            "coverage_at_target_fpr",
            "collapsed",
        ],
    )

    print(f"Saved diagnosis: {diagnosis_path}")
    print(f"Saved updated table: {updated_path}")


if __name__ == "__main__":
    main()
