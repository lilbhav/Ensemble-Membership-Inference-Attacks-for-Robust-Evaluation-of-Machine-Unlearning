from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve


DEFAULT_ATTACKS = ["yeom", "shokri", "lira", "reference", "losstraj"]
DEFAULT_TARGETS = ["forget_vs_test", "retain_vs_test", "forget_vs_retain"]


@dataclass
class AttackFile:
    dataset: str
    seed: int
    method: str
    attack: str
    target: str
    attack_seed: int
    path: Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Diagnose MIA calibration and ensemble drift.")
    p.add_argument("--results-root", default="results")
    p.add_argument("--dataset", default="Cifar10")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--methods", nargs="*", default=["bad_teacher", "amnesiac"])
    p.add_argument("--attacks", nargs="*", default=DEFAULT_ATTACKS)
    p.add_argument("--targets", nargs="*", default=DEFAULT_TARGETS)
    p.add_argument("--target-fprs", nargs="*", type=float, default=[0.01, 0.001])
    p.add_argument("--k-values", nargs="*", type=int, default=[2, 3])
    p.add_argument("--out-dir", default=None)
    return p.parse_args()


def load_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def discover_files(root: Path, dataset: str, seed: int, methods: list[str], attacks: list[str], targets: list[str]) -> list[AttackFile]:
    files: list[AttackFile] = []
    base = root / "mia" / dataset / f"seed_{seed}"
    if not base.exists():
        return files

    for method in methods:
        for attack in attacks:
            attack_dir = base / method / attack
            if not attack_dir.exists():
                continue
            for path in sorted(attack_dir.glob("*_attack_seed_*.csv")):
                stem = path.stem
                if "_attack_seed_" not in stem:
                    continue
                target, attack_seed_str = stem.rsplit("_attack_seed_", 1)
                if target not in targets:
                    continue
                files.append(
                    AttackFile(
                        dataset=dataset,
                        seed=seed,
                        method=method,
                        attack=attack,
                        target=target,
                        attack_seed=int(attack_seed_str),
                        path=path,
                    )
                )
    return files


def safe_auc(y: np.ndarray, s: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def tpr_at_fpr(y: np.ndarray, s: np.ndarray, fpr_target: float) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y, s)
    eligible = [float(tv) for fv, tv in zip(fpr, tpr) if fv <= fpr_target]
    return max(eligible) if eligible else 0.0


def select_threshold_for_target_fpr(y: np.ndarray, s: np.ndarray, fpr_target: float) -> float:
    fpr, _, thresholds = roc_curve(y, s)
    idx = int(np.argmin(np.abs(fpr - fpr_target)))
    return float(thresholds[idx])


def confusion_rates(y: np.ndarray, p: np.ndarray) -> Tuple[float, float]:
    pos = y == 1
    neg = y == 0
    tpr = float((p[pos] == 1).sum() / max(1, int(pos.sum())))
    fpr = float((p[neg] == 1).sum() / max(1, int(neg.sum())))
    return tpr, fpr


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def jaccard(a: set[int], b: set[int]) -> float:
    denom = len(a | b)
    if denom == 0:
        return 0.0
    return float(len(a & b) / denom)


def quantiles(x: np.ndarray) -> dict:
    return {
        "min": float(np.min(x)),
        "p01": float(np.quantile(x, 0.01)),
        "p05": float(np.quantile(x, 0.05)),
        "p25": float(np.quantile(x, 0.25)),
        "p50": float(np.quantile(x, 0.50)),
        "p75": float(np.quantile(x, 0.75)),
        "p95": float(np.quantile(x, 0.95)),
        "p99": float(np.quantile(x, 0.99)),
        "max": float(np.max(x)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
    }


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root)
    out_dir = Path(args.out_dir) if args.out_dir else (results_root / "debug_mia")

    files = discover_files(
        results_root,
        args.dataset,
        args.seed,
        args.methods,
        args.attacks,
        args.targets,
    )
    if not files:
        print(
            "No per-attack score CSV files found under "
            f"{results_root / 'mia' / args.dataset / f'seed_{args.seed}'}. "
            "Run scripts/run_mia.py first."
        )
        return

    distribution_rows: list[dict] = []
    threshold_rows: list[dict] = []
    metrics_rows: list[dict] = []
    lira_rows: list[dict] = []
    silent_rows: list[dict] = []

    raw_vote_map: Dict[Tuple[str, str, int], Dict[int, Dict[str, np.ndarray]]] = {}
    calibrated_vote_map: Dict[Tuple[str, str, int, float], Dict[int, Dict[str, np.ndarray]]] = {}

    for af in files:
        rows = load_rows(af.path)
        if not rows:
            continue
        scores = np.array([float(r["score"]) for r in rows], dtype=float)
        labels = np.array([int(r["true_membership"]) for r in rows], dtype=int)
        hard = np.array([int(r["prediction"]) for r in rows], dtype=int)
        sample_ids = np.array([int(r["sample_id"]) for r in rows], dtype=int)
        split_names = [r.get("split_name", "") for r in rows]

        # Step 1: per-attack score distribution by split and membership label.
        for split in sorted(set(split_names)):
            for membership in [0, 1]:
                mask = np.array([(s == split and int(y) == membership) for s, y in zip(split_names, labels)], dtype=bool)
                if not mask.any():
                    continue
                q = quantiles(scores[mask])
                distribution_rows.append(
                    {
                        "dataset": af.dataset,
                        "base_seed": af.seed,
                        "unlearning_method": af.method,
                        "attack": af.attack,
                        "target": af.target,
                        "attack_seed": af.attack_seed,
                        "split_name": split,
                        "membership": membership,
                        "n": int(mask.sum()),
                        **q,
                    }
                )

        # Step 2 and 5: threshold audit and per-attack metrics.
        raw_tpr, raw_fpr = confusion_rates(labels, hard)
        score_threshold_hard_match = float(np.mean((scores >= 0.5).astype(int) == hard))
        base_metric = {
            "dataset": af.dataset,
            "base_seed": af.seed,
            "unlearning_method": af.method,
            "attack": af.attack,
            "target": af.target,
            "attack_seed": af.attack_seed,
            "auc": safe_auc(labels, scores),
            "balanced_accuracy_raw": float(balanced_accuracy_score(labels, hard)),
            "tpr_raw": raw_tpr,
            "fpr_raw": raw_fpr,
            "tpr_at_1pct_fpr": tpr_at_fpr(labels, scores, 0.01),
            "tpr_at_01pct_fpr": tpr_at_fpr(labels, scores, 0.001),
        }
        metrics_rows.append(base_metric)

        threshold_base = {
            "dataset": af.dataset,
            "base_seed": af.seed,
            "unlearning_method": af.method,
            "attack": af.attack,
            "target": af.target,
            "attack_seed": af.attack_seed,
            "raw_prediction_fpr": raw_fpr,
            "raw_prediction_tpr": raw_tpr,
            "raw_matches_score_ge_0p5_fraction": score_threshold_hard_match,
            "score_min": float(np.min(scores)),
            "score_max": float(np.max(scores)),
            "score_unique": int(np.unique(scores).size),
        }
        for tfpr in args.target_fprs:
            thr = select_threshold_for_target_fpr(labels, scores, tfpr)
            cal = (scores >= thr).astype(int)
            c_tpr, c_fpr = confusion_rates(labels, cal)
            threshold_rows.append(
                {
                    **threshold_base,
                    "target_fpr": tfpr,
                    "calibrated_threshold": thr,
                    "calibrated_fpr": c_fpr,
                    "calibrated_tpr": c_tpr,
                    "calibrated_balanced_accuracy": float(balanced_accuracy_score(labels, cal)),
                }
            )

            vote_key = (af.method, af.target, af.attack_seed, tfpr)
            calibrated_vote_map.setdefault(vote_key, {})[sample_ids[0] if len(sample_ids) > 0 else 0] = {}

        # Accumulate raw vote map by method/target/attack_seed.
        vote_key_raw = (af.method, af.target, af.attack_seed)
        if vote_key_raw not in raw_vote_map:
            raw_vote_map[vote_key_raw] = {}
        for sid, y, p in zip(sample_ids, labels, hard):
            raw_vote_map[vote_key_raw].setdefault(int(sid), {})["label"] = np.array([int(y)], dtype=int)
            raw_vote_map[vote_key_raw][int(sid)][af.attack] = np.array([int(p)], dtype=int)

        # Step 3: LiRA direction check.
        if af.attack in {"lira", "lira_offline"}:
            lira_rows.append(
                {
                    "dataset": af.dataset,
                    "base_seed": af.seed,
                    "unlearning_method": af.method,
                    "target": af.target,
                    "attack_seed": af.attack_seed,
                    "auc_score": safe_auc(labels, scores),
                    "auc_negated_score": safe_auc(labels, -scores),
                    "fraction_score_ge_0p5": float(np.mean(scores >= 0.5)),
                    "score_min": float(np.min(scores)),
                    "score_max": float(np.max(scores)),
                }
            )

        # Step 4: detect silent/degenerate attacks.
        if af.attack in {"reference", "losstraj"}:
            silent_rows.append(
                {
                    "dataset": af.dataset,
                    "base_seed": af.seed,
                    "unlearning_method": af.method,
                    "target": af.target,
                    "attack": af.attack,
                    "attack_seed": af.attack_seed,
                    "score_min": float(np.min(scores)),
                    "score_max": float(np.max(scores)),
                    "score_unique": int(np.unique(scores).size),
                    "fraction_score_ge_0p5": float(np.mean(scores >= 0.5)),
                    "fraction_pred_positive": float(np.mean(hard == 1)),
                    "is_constant_score": bool(np.unique(scores).size == 1),
                }
            )

    # Step 6 and 7 are left in table form from threshold outputs and seed-level availability.
    multi_seed_rows: list[dict] = []
    grouped: Dict[Tuple[str, str, str], Dict[int, set[int]]] = {}
    for af in files:
        if af.attack_seed is None:
            continue
        rows = load_rows(af.path)
        scores = np.array([float(r["score"]) for r in rows], dtype=float)
        labels = np.array([int(r["true_membership"]) for r in rows], dtype=int)
        sample_ids = np.array([int(r["sample_id"]) for r in rows], dtype=int)
        for tfpr in args.target_fprs:
            thr = select_threshold_for_target_fpr(labels, scores, tfpr)
            pos_set = set(sample_ids[scores >= thr].tolist())
            key = (af.method, af.attack, af.target, tfpr)
            if key not in grouped:
                grouped[key] = {}
            grouped[key][af.attack_seed] = pos_set

    for (method, attack, target, tfpr), seed_to_set in grouped.items():
        seeds = sorted(seed_to_set.keys())
        if len(seeds) < 2:
            multi_seed_rows.append(
                {
                    "unlearning_method": method,
                    "attack": attack,
                    "target": target,
                    "target_fpr": tfpr,
                    "seed_a": "",
                    "seed_b": "",
                    "jaccard": "",
                    "note": "fewer_than_2_seeds",
                }
            )
            continue
        for a, b in combinations(seeds, 2):
            multi_seed_rows.append(
                {
                    "unlearning_method": method,
                    "attack": attack,
                    "target": target,
                    "target_fpr": tfpr,
                    "seed_a": a,
                    "seed_b": b,
                    "jaccard": jaccard(seed_to_set[a], seed_to_set[b]),
                    "note": "",
                }
            )

    write_csv(
        out_dir / "score_distribution_stats.csv",
        distribution_rows,
        [
            "dataset", "base_seed", "unlearning_method", "attack", "target", "attack_seed",
            "split_name", "membership", "n",
            "min", "p01", "p05", "p25", "p50", "p75", "p95", "p99", "max", "mean", "std",
        ],
    )
    write_csv(
        out_dir / "threshold_audit.csv",
        threshold_rows,
        [
            "dataset", "base_seed", "unlearning_method", "attack", "target", "attack_seed",
            "target_fpr", "calibrated_threshold", "calibrated_fpr", "calibrated_tpr", "calibrated_balanced_accuracy",
            "raw_prediction_fpr", "raw_prediction_tpr", "raw_matches_score_ge_0p5_fraction",
            "score_min", "score_max", "score_unique",
        ],
    )
    write_csv(
        out_dir / "attack_metrics.csv",
        metrics_rows,
        [
            "dataset", "base_seed", "unlearning_method", "attack", "target", "attack_seed",
            "auc", "balanced_accuracy_raw", "tpr_raw", "fpr_raw", "tpr_at_1pct_fpr", "tpr_at_01pct_fpr",
        ],
    )
    write_csv(
        out_dir / "lira_direction_audit.csv",
        lira_rows,
        [
            "dataset", "base_seed", "unlearning_method", "target", "attack_seed",
            "auc_score", "auc_negated_score", "fraction_score_ge_0p5", "score_min", "score_max",
        ],
    )
    write_csv(
        out_dir / "silent_attack_audit.csv",
        silent_rows,
        [
            "dataset", "base_seed", "unlearning_method", "target", "attack", "attack_seed",
            "score_min", "score_max", "score_unique", "fraction_score_ge_0p5", "fraction_pred_positive", "is_constant_score",
        ],
    )
    write_csv(
        out_dir / "multi_seed_jaccard.csv",
        multi_seed_rows,
        ["unlearning_method", "attack", "target", "target_fpr", "seed_a", "seed_b", "jaccard", "note"],
    )

    print(f"Diagnostics written to: {out_dir}")
    print("Generated files: score_distribution_stats.csv, threshold_audit.csv, attack_metrics.csv, lira_direction_audit.csv, silent_attack_audit.csv, multi_seed_jaccard.csv")


if __name__ == "__main__":
    main()
