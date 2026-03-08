#!/usr/bin/env python3
"""Hypothesis testing — compare two groups with automatic test selection.

Usage:
    python3 hypothesis_test.py data.csv --col value --group segment --a control --b treatment
"""

import argparse
import sys

import pandas as pd
import numpy as np
from scipy import stats


def run_test(group_a: pd.Series, group_b: pd.Series, alpha: float = 0.05):
    # Check assumptions
    stat, p_normal = stats.shapiro(group_a.sample(min(len(group_a), 5000)))
    stat, p_var = stats.levene(group_a, group_b)

    # Select and run test
    if p_normal > 0.05:
        stat, p_value = stats.ttest_ind(group_a, group_b, equal_var=(p_var > 0.05))
        test_name = "Independent t-test" if p_var > 0.05 else "Welch's t-test"
    else:
        stat, p_value = stats.mannwhitneyu(group_a, group_b, alternative="two-sided")
        test_name = "Mann-Whitney U"

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((group_a.std() ** 2 + group_b.std() ** 2) / 2)
    effect_size = (group_a.mean() - group_b.mean()) / pooled_std if pooled_std > 0 else 0

    if abs(effect_size) < 0.2:
        effect_label = "Negligible"
    elif abs(effect_size) < 0.5:
        effect_label = "Small"
    elif abs(effect_size) < 0.8:
        effect_label = "Medium"
    else:
        effect_label = "Large"

    # Confidence interval for difference
    se_diff = np.sqrt(group_a.var() / len(group_a) + group_b.var() / len(group_b))
    ci_low = (group_a.mean() - group_b.mean()) - 1.96 * se_diff
    ci_high = (group_a.mean() - group_b.mean()) + 1.96 * se_diff

    print(f"=== Hypothesis Test ===")
    print(f"Test: {test_name}")
    print(f"Group A: mean={group_a.mean():.4f}, n={len(group_a)}")
    print(f"Group B: mean={group_b.mean():.4f}, n={len(group_b)}")
    print(f"Statistic: {stat:.4f}")
    print(f"p-value: {p_value:.4f}")
    print(f"Effect size (Cohen's d): {effect_size:.3f} ({effect_label})")
    print(f"95% CI for difference: [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"Result: {'Significant' if p_value < alpha else 'Not significant'} at alpha={alpha}")


def main():
    parser = argparse.ArgumentParser(description="Run hypothesis test comparing two groups")
    parser.add_argument("file", help="Path to CSV file")
    parser.add_argument("--col", required=True, help="Numeric column to compare")
    parser.add_argument("--group", required=True, help="Column containing group labels")
    parser.add_argument("--a", required=True, help="Label for group A")
    parser.add_argument("--b", required=True, help="Label for group B")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level")
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.file)
    except Exception as e:
        print(f"Error reading {args.file}: {e}", file=sys.stderr)
        sys.exit(1)

    group_a = df[df[args.group] == args.a][args.col].dropna()
    group_b = df[df[args.group] == args.b][args.col].dropna()

    if len(group_a) == 0 or len(group_b) == 0:
        print(f"Error: One or both groups are empty.", file=sys.stderr)
        sys.exit(1)

    run_test(group_a, group_b, args.alpha)


if __name__ == "__main__":
    main()
