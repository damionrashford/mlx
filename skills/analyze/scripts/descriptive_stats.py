#!/usr/bin/env python3
"""Descriptive statistics — aggregations, group-bys, percentiles.

Usage:
    python3 descriptive_stats.py data.csv
    python3 descriptive_stats.py data.csv --group segment --value revenue
"""

import argparse
import sys

import pandas as pd
import numpy as np


def descriptive_stats(df: pd.DataFrame, group_col: str = None, value_col: str = None):
    print(f"=== Descriptive Statistics ===\n")
    print(f"Shape: {df.shape}")
    print(f"\nNumeric summary:\n{df.describe().round(2)}")
    print(f"\nMissing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

    if group_col and value_col:
        summary = df.groupby(group_col).agg(
            count=(value_col, "count"),
            mean=(value_col, "mean"),
            median=(value_col, "median"),
            total=(value_col, "sum"),
            std=(value_col, "std"),
        ).round(2)
        print(f"\nGrouped by {group_col}:\n{summary}")

    # Percentiles for numeric columns
    numeric = df.select_dtypes(include="number")
    for col in numeric.columns:
        percentiles = df[col].quantile([0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
        mean_val = df[col].mean()
        median_val = df[col].median()
        if abs(mean_val - median_val) > 0.1 * abs(mean_val) and mean_val != 0:
            print(f"\n{col}: mean={mean_val:.2f}, median={median_val:.2f} — data is skewed")


def main():
    parser = argparse.ArgumentParser(description="Run descriptive statistics")
    parser.add_argument("file", help="Path to CSV file")
    parser.add_argument("--group", help="Column to group by")
    parser.add_argument("--value", help="Value column to aggregate")
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.file)
    except Exception as e:
        print(f"Error reading {args.file}: {e}", file=sys.stderr)
        sys.exit(1)

    descriptive_stats(df, args.group, args.value)


if __name__ == "__main__":
    main()
