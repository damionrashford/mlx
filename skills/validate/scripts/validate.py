#!/usr/bin/env python3
"""Analysis validation — automated checks for common data analysis pitfalls.

Usage:
    python3 validate.py data.csv
    python3 validate.py data.csv --join-check other.csv --join-key user_id
"""

import argparse
import sys

import pandas as pd
import numpy as np


def check_join_explosion(df_a: pd.DataFrame, df_b: pd.DataFrame, key: str) -> None:
    """Check if joining two DataFrames causes row inflation."""
    before = len(df_a)
    merged = df_a.merge(df_b, on=key)
    after = len(merged)
    if after > before:
        print(f"  WARNING: Join explosion — {before} -> {after} rows ({after/before:.1f}x)")
    else:
        print(f"  OK: Join preserved row count ({before} -> {after})")


def check_magnitude(df: pd.DataFrame) -> None:
    """Check for suspicious magnitudes in numeric columns."""
    print("\nMagnitude checks:")
    found = False
    for col in df.select_dtypes(include="number").columns:
        # Negative values where unexpected
        neg = (df[col] < 0).sum()
        if neg > 0 and col.lower() in ("revenue", "price", "cost", "amount", "quantity", "count", "age"):
            print(f"  WARNING: {col} has {neg} negative values")
            found = True

        # Percentages out of range
        if "pct" in col.lower() or "rate" in col.lower() or "percent" in col.lower():
            out_of_range = ((df[col] < 0) | (df[col] > 100)).sum()
            if out_of_range > 0:
                print(f"  WARNING: {col} has {out_of_range} values outside 0-100%")
                found = True

        # Exact round numbers (possible defaults)
        n_round = (df[col] % 1000 == 0).sum()
        if n_round > len(df) * 0.5 and df[col].nunique() < 10:
            print(f"  WARNING: {col} is >50% exact thousands — possible placeholder values")
            found = True

    if not found:
        print("  All numeric magnitudes look reasonable.")


def check_duplicates(df: pd.DataFrame) -> None:
    """Check for exact and near-duplicates."""
    print("\nDuplicate checks:")
    exact = df.duplicated().sum()
    if exact > 0:
        print(f"  WARNING: {exact} exact duplicate rows ({exact/len(df)*100:.1f}%)")
    else:
        print(f"  OK: No exact duplicates.")

    # Check ID columns
    for col in df.columns:
        if "id" in col.lower():
            n_unique = df[col].nunique()
            n_total = df[col].notna().sum()
            if n_unique < n_total:
                print(f"  NOTE: {col} has {n_total - n_unique} duplicate values (may be intentional for non-PK)")


def check_time_consistency(df: pd.DataFrame) -> None:
    """Check date columns for gaps and partial periods."""
    print("\nTime consistency checks:")
    date_cols = []
    for col in df.columns:
        if "date" in col.lower() or "time" in col.lower() or "created" in col.lower() or "updated" in col.lower():
            try:
                df[col] = pd.to_datetime(df[col])
                date_cols.append(col)
            except (ValueError, TypeError):
                pass

    if not date_cols:
        print("  No date columns detected.")
        return

    for col in date_cols:
        min_date = df[col].min()
        max_date = df[col].max()
        print(f"  {col}: {min_date} to {max_date}")

        # Check for future dates
        future = (df[col] > pd.Timestamp.now()).sum()
        if future > 0:
            print(f"    WARNING: {future} future dates detected")


def check_missing_patterns(df: pd.DataFrame) -> None:
    """Check for systematic missing data patterns."""
    print("\nMissing value patterns:")
    missing = df.isnull().sum()
    has_missing = missing[missing > 0]
    if len(has_missing) == 0:
        print("  No missing values.")
        return

    for col, count in has_missing.items():
        pct = count / len(df) * 100
        print(f"  {col}: {count} missing ({pct:.1f}%)")


def check_average_of_averages(df: pd.DataFrame) -> None:
    """Flag columns that look like pre-aggregated averages."""
    print("\nPre-aggregation check:")
    found = False
    for col in df.columns:
        if "avg" in col.lower() or "average" in col.lower() or "mean" in col.lower():
            print(f"  NOTE: {col} appears to be a pre-computed average — do NOT average this column directly")
            found = True
    if not found:
        print("  No pre-aggregated columns detected.")


def main():
    parser = argparse.ArgumentParser(description="Validate data analysis for common pitfalls")
    parser.add_argument("file", help="Path to CSV file to validate")
    parser.add_argument("--join-check", help="Second CSV to check for join explosion")
    parser.add_argument("--join-key", help="Key column for join check")
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.file)
    except Exception as e:
        print(f"Error reading {args.file}: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"=== Validation Report: {args.file} ===")
    print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns\n")

    if args.join_check and args.join_key:
        print("Join explosion check:")
        try:
            df_b = pd.read_csv(args.join_check)
            check_join_explosion(df, df_b, args.join_key)
        except Exception as e:
            print(f"  Error reading {args.join_check}: {e}")

    check_magnitude(df)
    check_duplicates(df)
    check_time_consistency(df)
    check_missing_patterns(df)
    check_average_of_averages(df)


if __name__ == "__main__":
    main()
