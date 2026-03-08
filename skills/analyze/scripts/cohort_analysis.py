#!/usr/bin/env python3
"""Cohort analysis — retention-style cohort table from transactional data.

Usage:
    python3 cohort_analysis.py data.csv --user user_id --date order_date --value revenue
    python3 cohort_analysis.py data.csv --user user_id --date event_date --freq W
"""

import argparse
import sys

import pandas as pd


def cohort_analysis(df: pd.DataFrame, user_col: str, date_col: str, freq: str = "M"):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    # Assign cohort (first activity period)
    df["cohort"] = df.groupby(user_col)[date_col].transform("min").dt.to_period(freq)
    df["period"] = df[date_col].dt.to_period(freq)
    df["cohort_age"] = (df["period"] - df["cohort"]).apply(lambda x: x.n)

    # Build cohort table
    cohort_table = df.groupby(["cohort", "cohort_age"])[user_col].nunique().reset_index()
    cohort_table = cohort_table.pivot(index="cohort", columns="cohort_age", values=user_col)

    # Retention rates
    cohort_sizes = cohort_table[0]
    retention = cohort_table.div(cohort_sizes, axis=0).round(3)

    print("=== Cohort Analysis ===\n")
    print(f"Cohorts: {len(cohort_sizes)}")
    print(f"Frequency: {freq}")
    print(f"\nCohort sizes:\n{cohort_sizes}\n")
    print(f"Retention rates:\n{retention}")
    return retention


def main():
    parser = argparse.ArgumentParser(description="Run cohort retention analysis")
    parser.add_argument("file", help="Path to CSV file")
    parser.add_argument("--user", required=True, help="User/customer ID column")
    parser.add_argument("--date", required=True, help="Date column")
    parser.add_argument("--freq", default="M", choices=["D", "W", "M", "Q", "Y"], help="Period frequency")
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.file)
    except Exception as e:
        print(f"Error reading {args.file}: {e}", file=sys.stderr)
        sys.exit(1)

    cohort_analysis(df, args.user, args.date, args.freq)


if __name__ == "__main__":
    main()
