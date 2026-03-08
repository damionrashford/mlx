#!/usr/bin/env python3
"""A/B test analysis — compare control vs treatment with statistical rigor.

Usage:
    python3 ab_test.py data.csv --col converted --group variant --control A --treatment B
    python3 ab_test.py data.csv --col revenue --group variant --control A --treatment B --metric continuous
"""

import argparse
import sys

import pandas as pd
import numpy as np
from scipy import stats


def ab_test_analysis(control: pd.Series, treatment: pd.Series, metric: str = "conversion", alpha: float = 0.05):
    n_c, n_t = len(control), len(treatment)
    mean_c, mean_t = control.mean(), treatment.mean()

    # Relative lift
    lift = (mean_t - mean_c) / mean_c * 100 if mean_c != 0 else float("inf")

    # Statistical test
    if metric == "conversion":
        p_pool = (control.sum() + treatment.sum()) / (n_c + n_t)
        se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_c + 1 / n_t))
        z = (mean_t - mean_c) / se if se > 0 else 0
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        test_name = "Z-test for proportions"
    else:
        stat, p_value = stats.ttest_ind(control, treatment)
        test_name = "Independent t-test"

    # Confidence interval for difference
    se_diff = np.sqrt(control.var() / n_c + treatment.var() / n_t)
    ci_low = (mean_t - mean_c) - 1.96 * se_diff
    ci_high = (mean_t - mean_c) + 1.96 * se_diff

    print(f"=== A/B Test Analysis ===")
    print(f"Test: {test_name}")
    print(f"Control:   {mean_c:.4f} (n={n_c})")
    print(f"Treatment: {mean_t:.4f} (n={n_t})")
    print(f"Lift: {lift:+.2f}%")
    print(f"p-value: {p_value:.4f}")
    print(f"95% CI for difference: [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"Result: {'Significant' if p_value < alpha else 'Not significant'} at alpha={alpha}")


def main():
    parser = argparse.ArgumentParser(description="Run A/B test analysis")
    parser.add_argument("file", help="Path to CSV file")
    parser.add_argument("--col", required=True, help="Metric column (0/1 for conversion, numeric for continuous)")
    parser.add_argument("--group", required=True, help="Column containing variant labels")
    parser.add_argument("--control", required=True, help="Label for control group")
    parser.add_argument("--treatment", required=True, help="Label for treatment group")
    parser.add_argument("--metric", choices=["conversion", "continuous"], default="conversion")
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.file)
    except Exception as e:
        print(f"Error reading {args.file}: {e}", file=sys.stderr)
        sys.exit(1)

    control = df[df[args.group] == args.control][args.col].dropna()
    treatment = df[df[args.group] == args.treatment][args.col].dropna()

    if len(control) == 0 or len(treatment) == 0:
        print("Error: One or both groups are empty.", file=sys.stderr)
        sys.exit(1)

    ab_test_analysis(control, treatment, args.metric, args.alpha)


if __name__ == "__main__":
    main()
