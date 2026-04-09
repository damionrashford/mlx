#!/usr/bin/env python3
# /// script
# dependencies = [
#   "pandas>=2.0",
# ]
# requires-python = ">=3.10"
# ///
"""RFM segmentation — Recency, Frequency, Monetary customer segmentation.

Usage:
    uv run rfm_segmentation.py data.csv --customer customer_id --date order_date --value revenue
    uv run rfm_segmentation.py data.csv --customer customer_id --date order_date --value revenue --segments 5
"""

import argparse
import sys

import pandas as pd


def rfm_segmentation(df: pd.DataFrame, customer_col: str, date_col: str, value_col: str, n_segments: int = 4):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    now = df[date_col].max() + pd.Timedelta(days=1)

    rfm = df.groupby(customer_col).agg(
        recency=(date_col, lambda x: (now - x.max()).days),
        frequency=(date_col, "count"),
        monetary=(value_col, "sum"),
    )

    def _qcut_score(series: pd.Series, n: int, ascending: bool = True) -> pd.Series:
        """Quantile-cut with safe label count (handles duplicates reducing bin count)."""
        _, bins = pd.qcut(series, n, retbins=True, duplicates="drop")
        n_bins = len(bins) - 1
        labels = list(range(1, n_bins + 1)) if ascending else list(range(n_bins, 0, -1))
        return pd.cut(series, bins=bins, labels=labels, include_lowest=True)

    # Score each dimension (1=worst, n_segments=best)
    for col in ["frequency", "monetary"]:
        rfm[f"{col}_score"] = _qcut_score(rfm[col], n_segments, ascending=True)
    rfm["recency_score"] = _qcut_score(rfm["recency"], n_segments, ascending=False)

    rfm["rfm_score"] = (
        rfm["recency_score"].astype(int) + rfm["frequency_score"].astype(int) + rfm["monetary_score"].astype(int)
    )

    print("=== RFM Segmentation ===\n")
    print(f"Customers: {len(rfm)}")
    print(f"Segments: {n_segments}\n")
    print(f"RFM Summary:\n{rfm[['recency', 'frequency', 'monetary', 'rfm_score']].describe().round(2)}\n")
    print(f"Score distribution:\n{rfm['rfm_score'].value_counts().sort_index()}\n")
    print(f"Top 10 customers:\n{rfm.sort_values('rfm_score', ascending=False).head(10)}")
    return rfm


def main():
    parser = argparse.ArgumentParser(description="Run RFM customer segmentation")
    parser.add_argument("file", help="Path to CSV file")
    parser.add_argument("--customer", required=True, help="Customer ID column")
    parser.add_argument("--date", required=True, help="Date column")
    parser.add_argument("--value", required=True, help="Monetary value column")
    parser.add_argument("--segments", type=int, default=4, help="Number of segments per dimension")
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.file)
    except Exception as e:
        print(f"Error reading {args.file}: {e}", file=sys.stderr)
        sys.exit(1)

    rfm_segmentation(df, args.customer, args.date, args.value, args.segments)


if __name__ == "__main__":
    main()
