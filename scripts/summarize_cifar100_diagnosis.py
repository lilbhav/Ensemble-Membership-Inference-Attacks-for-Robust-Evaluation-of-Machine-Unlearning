from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build CIFAR100 diagnosis tables: utility vs privacy, per-attack AUC, "
            "and ensemble OR coverage."
        )
    )
    p.add_argument("--results-root", default="results")
    p.add_argument("--dataset", default="Cifar100")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--target", default="forget_vs_test")
    p.add_argument("--attack-seed", type=int, default=0)
    p.add_argument("--target-fpr", type=float, default=0.05)
    p.add_argument(
        "--methods",
        nargs="*",
        default=["scrub", "bad_teacher", "ssd", "amnesiac"],
        help="Methods to include in summary tables.",
    )
    p.add_argument(
        "--tag",
        default="",
        help="Optional suffix tag (e.g., forget250 or forget500) used in output filenames.",
    )
    return p.parse_args()


def _suffix(tag: str) -> str:
    return f"_{tag}" if tag else ""


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def main() -> None:
    args = parse_args()

    results_root = Path(args.results_root)
    agg_dir = results_root / "aggregate"
    out_dir = agg_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    mia_metrics = load_csv(agg_dir / "mia_score_metrics.csv")
    model_metrics = load_csv(agg_dir / "model_metrics.csv")

    if mia_metrics.empty:
        print("No mia_score_metrics.csv found. Run compute_mia_metrics.py first.")
        return

    subset = mia_metrics[
        (mia_metrics["dataset"] == args.dataset)
        & (mia_metrics["base_seed"] == args.seed)
        & (mia_metrics["target"] == args.target)
        & (mia_metrics["attack_seed"].astype(str) == str(args.attack_seed))
    ].copy()

    if subset.empty:
        print(
            f"No MIA metric rows for dataset={args.dataset}, seed={args.seed}, "
            f"target={args.target}, attack_seed={args.attack_seed}."
        )
        return

    # 1) Per-attack AUC comparison with random baseline reference (AUC=0.5)
    auc_table = (
        subset[subset["unlearning_method"].isin(args.methods + ["baseline"])]
        .groupby(["unlearning_method", "attack_name"], as_index=False)["auc"]
        .mean()
    )
    auc_table["random_auc"] = 0.5
    auc_table["auc_minus_random"] = auc_table["auc"] - auc_table["random_auc"]
    auc_table["signal_strength"] = pd.cut(
        auc_table["auc_minus_random"],
        bins=[-10, 0.02, 0.08, 10],
        labels=["near-random", "weak", "strong"],
        include_lowest=True,
    )
    auc_out = out_dir / f"cifar100_per_attack_auc_comparison{_suffix(args.tag)}.csv"
    auc_table.sort_values(["unlearning_method", "auc"], ascending=[True, False]).to_csv(auc_out, index=False)

    # 2) Utility vs privacy summary
    util_rows = []
    if not model_metrics.empty:
        mm = model_metrics[
            (model_metrics["dataset"] == args.dataset)
            & (model_metrics["seed"] == args.seed)
            & (model_metrics["unlearning_method"].isin(args.methods + ["baseline"]))
        ].copy()
        if not mm.empty:
            baseline = mm[mm["unlearning_method"] == "baseline"]
            baseline_test = float(baseline["test_acc"].iloc[0]) if not baseline.empty else float("nan")
            baseline_forget = float(baseline["forget_acc"].iloc[0]) if not baseline.empty else float("nan")

            attack_auc = (
                subset[subset["unlearning_method"].isin(args.methods)]
                .groupby("unlearning_method", as_index=False)["auc"]
                .mean()
                .rename(columns={"auc": "mean_attack_auc"})
            )

            for _, row in mm.iterrows():
                method = str(row["unlearning_method"])
                if method == "baseline":
                    continue
                method_auc = attack_auc[attack_auc["unlearning_method"] == method]
                mean_auc = float(method_auc["mean_attack_auc"].iloc[0]) if not method_auc.empty else float("nan")
                util_rows.append(
                    {
                        "dataset": args.dataset,
                        "seed": args.seed,
                        "method": method,
                        "forget_count": row.get("forget_count", ""),
                        "test_acc": row.get("test_acc", ""),
                        "forget_acc": row.get("forget_acc", ""),
                        "retain_acc": row.get("retain_acc", ""),
                        "test_acc_drop_vs_baseline": (baseline_test - float(row["test_acc"])) if baseline_test == baseline_test else "",
                        "forget_acc_drop_vs_baseline": (baseline_forget - float(row["forget_acc"])) if baseline_forget == baseline_forget else "",
                        "mean_attack_auc": mean_auc,
                        "mean_auc_minus_random": (mean_auc - 0.5) if mean_auc == mean_auc else "",
                    }
                )

    util_df = pd.DataFrame(util_rows)
    util_out = out_dir / f"cifar100_utility_vs_privacy_summary{_suffix(args.tag)}.csv"
    util_df.to_csv(util_out, index=False)

    # 3) Ensemble OR coverage comparison at target FPR
    ens_rows = []
    for method in args.methods:
        fpr_file = (
            agg_dir
            / f"fpr_sweep_{args.dataset}_seed_{args.seed}_{method}_{args.target}_attackseed_{args.attack_seed}.csv"
        )
        if not fpr_file.exists():
            continue
        fdf = pd.read_csv(fpr_file)
        if fdf.empty:
            continue
        nearest = fdf.iloc[(fdf["target_fpr"] - args.target_fpr).abs().argsort().iloc[0]]
        ens_rows.append(
            {
                "dataset": args.dataset,
                "seed": args.seed,
                "method": method,
                "target_fpr_requested": args.target_fpr,
                "target_fpr_used": float(nearest["target_fpr"]),
                "coverage_or": float(nearest["coverage_or"]),
                "coverage_k2": float(nearest["coverage_k2"]),
                "coverage_k3": float(nearest["coverage_k3"]),
                "or_minus_best_single_vote": float(nearest["coverage_or"] - max(nearest["coverage_k2"], nearest["coverage_k3"])),
            }
        )

    ens_df = pd.DataFrame(ens_rows)
    ens_out = out_dir / f"cifar100_ensemble_or_coverage_comparison{_suffix(args.tag)}.csv"
    ens_df.to_csv(ens_out, index=False)

    # 4) Lightweight diagnosis note
    diagnosis_lines = []
    diagnosis_lines.append("# CIFAR100 Weak-Signal Diagnosis")
    diagnosis_lines.append("")
    diagnosis_lines.append(f"Dataset: {args.dataset}, seed: {args.seed}, target: {args.target}, attack_seed: {args.attack_seed}")
    diagnosis_lines.append("")

    if util_df.empty:
        diagnosis_lines.append("- Utility summary unavailable: no matching model_metrics rows found.")
    else:
        avg_forget_drop = util_df["forget_acc_drop_vs_baseline"].replace("", pd.NA).dropna().astype(float).mean()
        avg_auc_gap = util_df["mean_auc_minus_random"].replace("", pd.NA).dropna().astype(float).mean()
        diagnosis_lines.append(f"- Average forget-accuracy drop vs baseline: {avg_forget_drop:.4f}")
        diagnosis_lines.append(f"- Average mean-attack AUC minus random (0.5): {avg_auc_gap:.4f}")
        if avg_forget_drop < 20:
            diagnosis_lines.append("- Signal suggests insufficient forgetting pressure (supports hypothesis a).")
        if avg_auc_gap < 0.05:
            diagnosis_lines.append("- Signal suggests attack separability is weak on this setting (supports b and/or c).")

    if ens_df.empty:
        diagnosis_lines.append("- Ensemble OR coverage summary unavailable: no matching fpr_sweep files found.")
    else:
        mean_or_gain = ens_df["or_minus_best_single_vote"].mean()
        diagnosis_lines.append(f"- Mean OR-vs-best-single coverage gain at target FPR: {mean_or_gain:.4f}")
        if mean_or_gain > 0:
            diagnosis_lines.append("- Ensemble improves coverage, indicating single attacks are partially complementary.")
        else:
            diagnosis_lines.append("- Ensemble gain is limited, indicating attack diversity is weak.")

    md_out = out_dir / f"cifar100_diagnosis_note{_suffix(args.tag)}.md"
    md_out.write_text("\n".join(diagnosis_lines) + "\n", encoding="utf-8")

    print(f"Wrote: {auc_out}")
    print(f"Wrote: {util_out}")
    print(f"Wrote: {ens_out}")
    print(f"Wrote: {md_out}")


if __name__ == "__main__":
    main()
