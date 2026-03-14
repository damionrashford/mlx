# /// script
# dependencies = [
#   "pandas>=2.0",
#   "numpy>=1.24",
# ]
# requires-python = ">=3.10"
# ///
"""
Feature engineering CLI — apply standard transforms to a CSV dataset.

Supports numeric, categorical, datetime, text, interaction, time series,
and group aggregation features. Outputs an enriched CSV.
"""

import argparse
import sys

import numpy as np
import pandas as pd


def engineer_numeric(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df = df.copy()
    if df[col].min() >= 0:
        df[f"{col}_log"] = np.log1p(df[col])
    df[f"{col}_binned"] = pd.qcut(df[col], q=5, labels=False, duplicates="drop")
    df[f"{col}_zscore"] = (df[col] - df[col].mean()) / df[col].std()
    q1, q99 = df[col].quantile([0.01, 0.99])
    df[f"{col}_clipped"] = df[col].clip(q1, q99)
    return df


def engineer_categorical(
    df: pd.DataFrame, col: str, target: str | None = None
) -> pd.DataFrame:
    df = df.copy()
    freq = df[col].value_counts(normalize=True)
    df[f"{col}_freq"] = df[col].map(freq)
    if target and target in df.columns:
        means = df.groupby(col)[target].mean()
        counts = df.groupby(col)[target].count()
        smooth = 20
        global_mean = df[target].mean()
        df[f"{col}_target_enc"] = df[col].map(
            (counts * means + smooth * global_mean) / (counts + smooth)
        )
    if df[col].nunique() <= 10:
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
        df = pd.concat([df, dummies], axis=1)
    return df


def engineer_datetime(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df = df.copy()
    dt = pd.to_datetime(df[col])
    df[f"{col}_year"] = dt.dt.year
    df[f"{col}_month"] = dt.dt.month
    df[f"{col}_dayofweek"] = dt.dt.dayofweek
    df[f"{col}_hour"] = dt.dt.hour
    df[f"{col}_is_weekend"] = dt.dt.dayofweek.isin([5, 6]).astype(int)
    df[f"{col}_month_sin"] = np.sin(2 * np.pi * dt.dt.month / 12)
    df[f"{col}_month_cos"] = np.cos(2 * np.pi * dt.dt.month / 12)
    return df


def engineer_text(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df = df.copy()
    text = df[col].fillna("")
    df[f"{col}_length"] = text.str.len()
    df[f"{col}_word_count"] = text.str.split().str.len()
    df[f"{col}_has_numbers"] = text.str.contains(r"\d").astype(int)
    df[f"{col}_unique_words"] = text.apply(
        lambda x: len(set(x.lower().split())) if x.strip() else 0
    )
    return df


def engineer_interactions(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    numeric = [c for c in cols if df[c].dtype in ["int64", "float64"]]
    for i, c1 in enumerate(numeric):
        for c2 in numeric[i + 1 :]:
            df[f"{c1}_x_{c2}"] = df[c1] * df[c2]
            df[f"{c1}_div_{c2}"] = df[c1] / df[c2].replace(0, np.nan)
    return df


def engineer_timeseries(
    df: pd.DataFrame, col: str, windows: list[int] | None = None
) -> pd.DataFrame:
    if windows is None:
        windows = [7, 14, 30]
    df = df.copy()
    for w in windows:
        df[f"{col}_roll_mean_{w}"] = df[col].rolling(w).mean()
        df[f"{col}_roll_std_{w}"] = df[col].rolling(w).std()
    for lag in [1, 3, 7]:
        df[f"{col}_lag_{lag}"] = df[col].shift(lag)
    df[f"{col}_diff"] = df[col].diff()
    df[f"{col}_pct_change"] = df[col].pct_change()
    return df


def engineer_aggregations(
    df: pd.DataFrame, group_col: str, agg_col: str
) -> pd.DataFrame:
    df = df.copy()
    agg = df.groupby(group_col)[agg_col].agg(["mean", "std", "min", "max"])
    agg.columns = [f"{group_col}_{agg_col}_{s}" for s in agg.columns]
    df = df.merge(agg, left_on=group_col, right_index=True, how="left")
    df[f"{agg_col}_dev_from_{group_col}"] = (
        df[agg_col] - df[f"{group_col}_{agg_col}_mean"]
    )
    return df


def detect_column_types(df: pd.DataFrame) -> dict[str, list[str]]:
    types: dict[str, list[str]] = {
        "numeric": [],
        "categorical": [],
        "datetime": [],
        "text": [],
    }
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            types["numeric"].append(col)
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            types["datetime"].append(col)
        else:
            # Try parsing as datetime
            try:
                pd.to_datetime(df[col].dropna().head(20))
                types["datetime"].append(col)
                continue
            except (ValueError, TypeError):
                pass
            # Check if text (high cardinality) or categorical (low cardinality)
            if df[col].nunique() > 50 and df[col].str.len().mean() > 20:
                types["text"].append(col)
            else:
                types["categorical"].append(col)
    return types


def main():
    parser = argparse.ArgumentParser(
        description="Feature engineering CLI for tabular datasets.",
        epilog=(
            "Examples:\n"
            "  %(prog)s data.csv -o features.csv\n"
            "  %(prog)s data.csv --target price --types numeric categorical\n"
            "  %(prog)s data.csv --cols age income --interactions\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", help="Path to input CSV file")
    parser.add_argument("-o", "--output", help="Output CSV path (default: stdout)")
    parser.add_argument("--target", help="Target column for target encoding")
    parser.add_argument(
        "--cols",
        nargs="+",
        help="Specific columns to engineer (default: auto-detect all)",
    )
    parser.add_argument(
        "--types",
        nargs="+",
        choices=["numeric", "categorical", "datetime", "text", "timeseries"],
        help="Feature types to generate (default: all detected)",
    )
    parser.add_argument(
        "--interactions",
        action="store_true",
        help="Generate interaction features for numeric columns",
    )
    parser.add_argument(
        "--group",
        nargs=2,
        metavar=("GROUP_COL", "AGG_COL"),
        help="Generate group aggregation features",
    )
    parser.add_argument(
        "--json", action="store_true", help="Output summary as JSON instead of CSV"
    )

    args = parser.parse_args()

    try:
        df = pd.read_csv(args.input)
    except FileNotFoundError:
        print(f"Error: File not found: {args.input}", file=sys.stderr)
        print(f"Usage: python {sys.argv[0]} <path-to-csv>", file=sys.stderr)
        sys.exit(1)

    original_cols = set(df.columns)
    col_types = detect_column_types(df)
    allowed_types = set(args.types) if args.types else {"numeric", "categorical", "datetime", "text"}

    cols_to_process = args.cols if args.cols else df.columns.tolist()

    # Apply transforms
    for col in cols_to_process:
        if col not in df.columns:
            print(f"Warning: column '{col}' not found, skipping", file=sys.stderr)
            continue
        if col in col_types["numeric"] and "numeric" in allowed_types:
            df = engineer_numeric(df, col)
        if col in col_types["categorical"] and "categorical" in allowed_types:
            df = engineer_categorical(df, col, args.target)
        if col in col_types["datetime"] and "datetime" in allowed_types:
            df = engineer_datetime(df, col)
        if col in col_types["text"] and "text" in allowed_types:
            df = engineer_text(df, col)

    if args.interactions:
        numeric_cols = [c for c in cols_to_process if c in col_types["numeric"]]
        df = engineer_interactions(df, numeric_cols)

    if args.group:
        df = engineer_aggregations(df, args.group[0], args.group[1])

    if "timeseries" in (args.types or []):
        for col in cols_to_process:
            if col in col_types["numeric"]:
                df = engineer_timeseries(df, col)

    new_cols = set(df.columns) - original_cols

    if args.json:
        import json

        print(
            json.dumps(
                {
                    "input": args.input,
                    "original_columns": len(original_cols),
                    "new_features": len(new_cols),
                    "total_columns": len(df.columns),
                    "rows": len(df),
                    "new_feature_names": sorted(new_cols),
                },
                indent=2,
            )
        )
    else:
        print(f"Engineered {len(new_cols)} new features", file=sys.stderr)
        if args.output:
            df.to_csv(args.output, index=False)
            print(f"Saved to {args.output}", file=sys.stderr)
        else:
            df.to_csv(sys.stdout, index=False)


if __name__ == "__main__":
    main()
