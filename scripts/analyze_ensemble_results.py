"""
Utility script to analyze ensemble MIA plots and generate quick summaries.
Provides insight into key findings from the visualization suite.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Paths
RESULTS_DIR = Path('results')
AGGREGATE_DIR = RESULTS_DIR / 'aggregate'
ENSEMBLE_DIR = RESULTS_DIR / 'ensembles'
METHODS = ['bad_teacher', 'amnesiac']


def analyze_method(method):
    """Generate comprehensive analysis for a method"""
    print(f"\n{'='*70}")
    print(f"ANALYSIS: {method.upper()} UNLEARNING METHOD")
    print(f"{'='*70}\n")
    
    # Load coverage data
    coverage_file = AGGREGATE_DIR / f"fpr_sweep_Cifar10_seed_0_{method}_forget_vs_test_attackseed_0_coverage_per_attack.csv"
    if coverage_file.exists():
        df_coverage = pd.read_csv(coverage_file)
        df_5pct = df_coverage[df_coverage['target_fpr'] == 0.05]
        
        print("📊 INDIVIDUAL ATTACK PERFORMANCE AT 5% FPR")
        print("-" * 70)
        attack_coverage = df_5pct.groupby('attack')['coverage_fraction'].first().sort_values(ascending=False)
        
        for attack, coverage in attack_coverage.items():
            status = "✓ ACTIVE" if coverage > 0 else "✗ SILENT"
            print(f"  {attack:15s}  Coverage: {coverage:6.4f}  [{status}]")
        
        print(f"\n  Active attacks:  {len(attack_coverage[attack_coverage > 0])} / {len(attack_coverage)}")
        print(f"  Max coverage:    {attack_coverage.max():.4f}")
        print(f"  Min coverage:    {attack_coverage.min():.4f}")
    
    # Load AUC data
    flip_file = AGGREGATE_DIR / f"fpr_sweep_Cifar10_seed_0_{method}_forget_vs_test_attackseed_0_attack_flip_5pct_table.csv"
    if flip_file.exists():
        df_flip = pd.read_csv(flip_file)
        
        print("\n🎯 ATTACK STRENGTH (AUC) AND SIGN CORRECTION")
        print("-" * 70)
        df_flip_sorted = df_flip.sort_values('AUC_after_flip', ascending=False)
        
        for _, row in df_flip_sorted.iterrows():
            flip_status = "⚠ FLIPPED" if row['sign_flip'] else "✓ NATIVE"
            print(f"  {row['attack']:15s}  AUC: {row['AUC_after_flip']:5.3f}  [{flip_status}]")
        
        flipped_count = df_flip['sign_flip'].sum()
        print(f"\n  Attacks needing flip:  {flipped_count} / {len(df_flip)}")
        print(f"  Avg AUC (flipped):     {df_flip[df_flip['sign_flip']]['AUC_after_flip'].mean():.3f}")
        print(f"  Avg AUC (native):      {df_flip[~df_flip['sign_flip']]['AUC_after_flip'].mean():.3f}")
    
    # Load activation data
    activation_file = AGGREGATE_DIR / f"fpr_sweep_Cifar10_seed_0_{method}_forget_vs_test_attackseed_0_activation_order.csv"
    if activation_file.exists():
        df_activation = pd.read_csv(activation_file)
        
        print("\n🚀 ATTACK ACTIVATION (First FPR where attack activates)")
        print("-" * 70)
        
        active_attacks = df_activation[df_activation['first_active_target_fpr'] != 'never_active'].copy()
        active_attacks['fpr_val'] = active_attacks['first_active_target_fpr'].astype(float)
        active_attacks = active_attacks.sort_values('fpr_val', ascending=True)
        
        for _, row in active_attacks.iterrows():
            print(f"  {row['attack']:15s}  Activates at FPR: {float(row['first_active_target_fpr']):6.3f}")
        
        never_active = df_activation[df_activation['first_active_target_fpr'] == 'never_active']['attack'].tolist()
        if never_active:
            print(f"\n  Never active attacks: {', '.join(never_active)}")
        
        print(f"  Early activation (FPR<0.02):  {len(active_attacks[active_attacks['fpr_val'] < 0.02])}")
    
    # Load ensemble data
    ensemble_file = ENSEMBLE_DIR / method / 'forget_vs_test' / 'coverage_per_attack.csv'
    if ensemble_file.exists():
        df_ensemble = pd.read_csv(ensemble_file)
        
        print("\n🔗 ENSEMBLE COVERAGE (Lowest Calibration Point)")
        print("-" * 70)
        
        or_coverage = df_ensemble[df_ensemble['attack'] == 'union_or']['coverage_fraction'].values
        if len(or_coverage) > 0:
            or_cov = or_coverage[0]
            individual = df_ensemble[df_ensemble['attack'] != 'union_or'].copy()
            individual_sorted = individual.sort_values('coverage_fraction', ascending=False)
            
            strongest = individual_sorted.iloc[0] if len(individual_sorted) > 0 else None
            
            if strongest is not None:
                improvement = (or_cov - strongest['coverage_fraction']) / strongest['coverage_fraction'] * 100
                print(f"  Strongest individual attack: {strongest['attack']} ({strongest['coverage_fraction']:.4f})")
                print(f"  OR union coverage:          {or_cov:.4f}")
                print(f"  Ensemble improvement:       +{improvement:.1f}%")
            
            print(f"\n  Individual attack coverages:")
            for _, row in individual_sorted.iterrows():
                if row['coverage_fraction'] > 0:
                    contribution = (row['coverage_fraction'] / or_cov) * 100 if or_cov > 0 else 0
                    print(f"    {row['attack']:15s}  Coverage: {row['coverage_fraction']:6.4f}  ({contribution:5.1f}% of union)")
    
    # Load Jaccard data
    jaccard_file = ENSEMBLE_DIR / method / 'forget_vs_test' / 'disparity_pairwise_jaccard.csv'
    if jaccard_file.exists():
        df_jaccard = pd.read_csv(jaccard_file)
        
        print("\n🔀 ATTACK DIVERSITY (Pairwise Jaccard Similarity)")
        print("-" * 70)
        
        # Find high and low overlap pairs
        df_jaccard_sorted = df_jaccard.sort_values('jaccard', ascending=False)
        
        high_overlap = df_jaccard_sorted[df_jaccard_sorted['jaccard'] > 0.25]
        low_overlap = df_jaccard_sorted[df_jaccard_sorted['jaccard'] < 0.1]
        
        print(f"  High overlap pairs (Jaccard > 0.25):  {len(high_overlap)}")
        if len(high_overlap) > 0:
            for _, row in high_overlap.head(3).iterrows():
                print(f"    {row['attack_a']} ↔ {row['attack_b']}: {row['jaccard']:.3f}")
        
        print(f"\n  Low overlap pairs (Jaccard < 0.1):   {len(low_overlap)}")
        if len(low_overlap) > 0:
            for _, row in low_overlap.head(3).iterrows():
                print(f"    {row['attack_a']} ↔ {row['attack_b']}: {row['jaccard']:.3f}")
        
        avg_jaccard = df_jaccard['jaccard'].mean()
        print(f"\n  Average Jaccard similarity: {avg_jaccard:.3f}")
        print(f"  Diversity assessment: ", end="")
        if avg_jaccard < 0.15:
            print("✓ HIGH (attacks complement each other)")
        elif avg_jaccard < 0.30:
            print("⚠ MODERATE (some overlap)")
        else:
            print("✗ LOW (attacks are redundant)")


def compare_methods():
    """Generate comparative analysis"""
    print(f"\n{'='*70}")
    print("COMPARATIVE ANALYSIS: bad_teacher vs amnesiac")
    print(f"{'='*70}\n")
    
    print("📋 KEY COMPARISON METRICS")
    print("-" * 70)
    
    comparisons = []
    for method in METHODS:
        # Coverage
        coverage_file = AGGREGATE_DIR / f"fpr_sweep_Cifar10_seed_0_{method}_forget_vs_test_attackseed_0_coverage_per_attack.csv"
        ensemble_file = ENSEMBLE_DIR / method / 'forget_vs_test' / 'coverage_per_attack.csv'
        flip_file = AGGREGATE_DIR / f"fpr_sweep_Cifar10_seed_0_{method}_forget_vs_test_attackseed_0_attack_flip_5pct_table.csv"
        
        coverage_main = None
        or_cov = None
        flipped_pct = None
        active_count = None
        
        if coverage_file.exists():
            df = pd.read_csv(coverage_file)
            df_5pct = df[df['target_fpr'] == 0.05]
            coverage_main = df_5pct['coverage_fraction'].max()
            active_count = len(df_5pct[df_5pct['coverage_fraction'] > 0])
        
        if ensemble_file.exists():
            df = pd.read_csv(ensemble_file)
            or_vals = df[df['attack'] == 'union_or']['coverage_fraction'].values
            if len(or_vals) > 0:
                or_cov = or_vals[0]
        
        if flip_file.exists():
            df = pd.read_csv(flip_file)
            flipped_pct = (df['sign_flip'].sum() / len(df)) * 100
        
        comparisons.append({
            'method': method,
            'max_coverage': coverage_main,
            'or_coverage': or_cov,
            'flipped_pct': flipped_pct,
            'active_attacks': active_count
        })
    
    # Print comparison
    print(f"\n{'Metric':<30} {'bad_teacher':>15} {'amnesiac':>15} {'Comparison':<15}")
    print("-" * 70)
    
    for comp in comparisons:
        if comp['method'] == 'bad_teacher':
            bt = comp
    for comp in comparisons:
        if comp['method'] == 'amnesiac':
            am = comp
    
    # Max coverage
    if bt['max_coverage'] and am['max_coverage']:
        better = "bad_teacher" if bt['max_coverage'] > am['max_coverage'] else "amnesiac"
        print(f"{'Max individual coverage':30s} {bt['max_coverage']:14.4f} {am['max_coverage']:14.4f}  → {better} (more vulnerable)")
    
    # OR coverage
    if bt['or_coverage'] and am['or_coverage']:
        better = "bad_teacher" if bt['or_coverage'] > am['or_coverage'] else "amnesiac"
        print(f"{'OR union coverage':30s} {bt['or_coverage']:14.4f} {am['or_coverage']:14.4f}  → {better} (ensemble weaker)")
    
    # Sign flips
    if bt['flipped_pct'] is not None and am['flipped_pct'] is not None:
        better = "bad_teacher" if bt['flipped_pct'] < am['flipped_pct'] else "amnesiac"
        print(f"{'Attacks needing flip (%)':30s} {bt['flipped_pct']:14.1f} {am['flipped_pct']:14.1f}  → {better} (more stable)")
    
    # Active attacks
    if bt['active_attacks'] is not None and am['active_attacks'] is not None:
        better = "bad_teacher" if bt['active_attacks'] > am['active_attacks'] else "amnesiac"
        print(f"{'Active attacks @ 5% FPR':30s} {bt['active_attacks']:14d} {am['active_attacks']:14d}  → {better} (more threats)")
    
    print("\n" + "=" * 70)
    print("RECOMMENDATION:")
    print("-" * 70)
    
    # Assess which method is more robust
    bt_vulnerability = sum([
        bt['max_coverage'] > 0.1,  # has significant coverage
        bt['or_coverage'] > 0.1,
        bt['active_attacks'] > 3,  # multiple active attacks
    ])
    
    am_vulnerability = sum([
        am['max_coverage'] > 0.1,
        am['or_coverage'] > 0.1,
        am['active_attacks'] > 3,
    ])
    
    if bt_vulnerability > am_vulnerability:
        print("\n  🛡️  AMNESIAC appears more robust to ensemble attacks")
        print("     → Lower coverage and fewer active attacks")
    elif am_vulnerability > bt_vulnerability:
        print("\n  🛡️  BAD TEACHER appears more robust to ensemble attacks")
        print("     → Lower coverage and fewer active attacks")
    else:
        print("\n  ⚖️   BOTH methods show similar vulnerability profiles")
        print("     → Consider complementary defenses")


def main():
    """Run complete analysis"""
    print("\n" + "█" * 70)
    print("ENSEMBLE MIA EVALUATION - QUICK ANALYSIS TOOL")
    print("█" * 70)
    
    # Analyze each method
    for method in METHODS:
        analyze_method(method)
    
    # Compare methods
    compare_methods()
    
    print("\n" + "=" * 70)
    print("📚 For detailed interpretations, see: PLOT_INTERPRETATION_GUIDE.md")
    print("📊 For visualization descriptions, see: README.md")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
