#!/usr/bin/env python3
"""Trend analysis — moving averages, growth rates, and seasonal decomposition.

Usage:
    python3 trend_analysis.py data.csv --date date --value revenue
    python3 trend_analysis.py data.csv --date date --value revenue --window 30
"""

import argparse
import sys

import pandas as pd
import numpy as np


def trend_analysis(df: pd.DataFrame, date_col: str, value_col: str, window: int = 7):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)
    series = df.set_index(date_col)[value_col]

    # Moving averages
    ma = series.rolling(window=window, center=True).mean()

    # Growth rates
    pct_change = series.pct_change(periods=window) * 100

    print(f"=== Trend Analysis: {value_col} ===\n")
    print(f"Period: {series.index.min()} to {series.index.max()}")
    print(f"Data points: {len(series)}")
    print(f"Window: {window}\n")

    print(f"Overall: {series.iloc[0]:.2f} -> {series.iloc[-1]:.2f}")
    total_change = (series.iloc[-1] - series.iloc[0]) / series.iloc[0] * 100 if series.iloc[0] != 0 else 0
    print(f"Total change: {total_change:+.1f}%\n")

    print(f"Moving average (last 5):\n{ma.dropna().tail()}\n")
    print(f"Period-over-period growth (last 5):\n{pct_change.dropna().tail().round(2)}\n")

    # Decomposition if enough data
    if len(series) >= 2 * window:
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose

            decomp = seasonal_decompose(series, period=window, model="additive")
            print("Seasonal decomposition computed.")
            print(f"Trend range: {decomp.trend.dropna().min():.2f} to {decomp.trend.dropna().max():.2f}")
            print(f"Seasonal amplitude: {decomp.seasonal.max() - decomp.seasonal.min():.2f}")
        except ImportError:
            print("Install statsmodels for seasonal decomposition: pip install statsmodels")


def main():
    parser = argparse.ArgumentParser(description="Run trend analysis on time series data")
    parser.add_argument("file", help="Path to CSV file")
    parser.add_argument("--date", required=True, help="Date column")
    parser.add_argument("--value", required=True, help="Value column to analyze")
    parser.add_argument("--window", type=int, default=7, help="Rolling window size")
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.file)
    except Exception as e:
        print(f"Error reading {args.file}: {e}", file=sys.stderr)
        sys.exit(1)

    trend_analysis(df, args.date, args.value, args.window)


if __name__ == "__main__":
    main()
