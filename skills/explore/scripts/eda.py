#!/usr/bin/env python3
"""Exploratory Data Analysis — run a full EDA pipeline on a CSV dataset.

Usage:
    python3 eda.py data/train.csv
    python3 eda.py data/train.csv --target price
"""

import argparse
import sys

import pandas as pd
import numpy as np


def overview(df: pd.DataFrame) -> None:
    print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"Memory: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    print(f"\nColumn types:\n{df.dtypes.value_counts()}")
    print(f"\nFirst 5 rows:\n{df.head()}")


def missing_values(df: pd.DataFrame) -> None:
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    report = pd.DataFrame({"count": missing, "percent": missing_pct})
    has_missing = report.query("count > 0").sort_values("percent", ascending=False)
    if len(has_missing):
        print(f"\nMissing values:\n{has_missing}")
    else:
        print("\nNo missing values.")


def numeric_features(df: pd.DataFrame) -> None:
    numeric = df.select_dtypes(include="number")
    if numeric.empty:
        print("\nNo numeric columns.")
        return
    print(f"\nNumeric summary:\n{numeric.describe().T[['count','mean','std','min','25%','50%','75%','max']]}")
    for col in numeric.columns:
        if (df[col] == 0).sum() > len(df) * 0.5:
            print(f"  WARNING: {col} is >50% zeros")
        if (df[col] < 0).sum() > 0:
            print(f"  NOTE: {col} has negative values")


def categorical_features(df: pd.DataFrame) -> None:
    cats = df.select_dtypes(include=["object", "category"])
    if cats.empty:
        print("\nNo categorical columns.")
        return
    print("\nCategorical summary:")
    for col in cats.columns:
        n = df[col].nunique()
        top = df[col].mode().iloc[0] if not df[col].mode().empty else "N/A"
        print(f"  {col}: {n} unique, top='{top}'")
        if n > 50:
            print(f"    HIGH CARDINALITY")
        if n == len(df):
            print(f"    UNIQUE PER ROW — likely ID")


def correlations(df: pd.DataFrame) -> None:
    numeric = df.select_dtypes(include="number")
    if len(numeric.columns) < 2:
        return
    corr = numeric.corr()
    print("\nHigh correlations (|r| > 0.7):")
    found = False
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            r = corr.iloc[i, j]
            if abs(r) > 0.7:
                print(f"  {corr.columns[i]} <-> {corr.columns[j]}: r={r:.3f}")
                found = True
    if not found:
        print("  None found.")


def target_analysis(df: pd.DataFrame, target: str) -> None:
    if target not in df.columns:
        print(f"\nTarget column '{target}' not found.")
        return
    print(f"\nTarget analysis: {target}")
    if df[target].dtype in ["object", "category", "bool"]:
        print("  Task: Classification")
        print(df[target].value_counts(normalize=True).round(3))
        if df[target].value_counts(normalize=True).min() < 0.1:
            print("  WARNING: Imbalanced classes!")
    else:
        print(f"  Task: Regression")
        print(f"  mean={df[target].mean():.4f}, std={df[target].std():.4f}")


def duplicates(df: pd.DataFrame) -> None:
    dupes = df.duplicated().sum()
    print(f"\nExact duplicates: {dupes} ({dupes / len(df) * 100:.2f}%)")


def completeness_scoring(df: pd.DataFrame) -> None:
    print("\nCompleteness scoring:")
    for col in df.columns:
        pct = (1 - df[col].isnull().mean()) * 100
        if pct > 99:
            grade = "GREEN (complete)"
        elif pct >= 95:
            grade = "YELLOW (investigate nulls)"
        elif pct >= 80:
            grade = "ORANGE (may need imputation)"
        else:
            grade = "RED (likely unusable without imputation)"
        if pct < 100:
            print(f"  {col}: {pct:.1f}% — {grade}")


def accuracy_red_flags(df: pd.DataFrame) -> None:
    print("\nAccuracy red flags:")
    found = False
    for col in df.select_dtypes(include="number").columns:
        for sentinel in [0, -1, 999, 9999, 99999]:
            n = (df[col] == sentinel).sum()
            if n > len(df) * 0.01:
                print(f"  {col}: {n} rows ({n / len(df) * 100:.1f}%) equal {sentinel} — possible placeholder")
                found = True
        non_null = df[col].dropna()
        if len(non_null) > 100:
            round_pct = ((non_null % 5 == 0).sum() / len(non_null)) * 100
            if round_pct > 80:
                print(f"  {col}: {round_pct:.0f}% of values are multiples of 5 — possible estimation")
                found = True

    for col in df.select_dtypes(include="object").columns:
        for placeholder in ["N/A", "TBD", "test", "xxx", "unknown", "null", "none"]:
            n = df[col].str.lower().eq(placeholder.lower()).sum()
            if n > 0:
                print(f"  {col}: {n} rows contain '{placeholder}'")
                found = True
    if not found:
        print("  None found.")


def main():
    parser = argparse.ArgumentParser(description="Run EDA on a CSV dataset")
    parser.add_argument("file", help="Path to CSV file")
    parser.add_argument("--target", "-t", help="Target column for supervised analysis")
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.file)
    except Exception as e:
        print(f"Error reading {args.file}: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"=== EDA Report: {args.file} ===\n")
    overview(df)
    missing_values(df)
    numeric_features(df)
    categorical_features(df)
    correlations(df)
    if args.target:
        target_analysis(df, args.target)
    duplicates(df)
    completeness_scoring(df)
    accuracy_red_flags(df)


if __name__ == "__main__":
    main()
