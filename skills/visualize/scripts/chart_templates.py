#!/usr/bin/env python3
"""Chart templates — generate common chart types from CSV data.

Usage:
    python3 chart_templates.py data.csv --type bar --x category --y value -o bar_chart.png
    python3 chart_templates.py data.csv --type line --x date --y value --hue segment -o trend.png
    python3 chart_templates.py data.csv --type hist --x value -o distribution.png
    python3 chart_templates.py data.csv --type heatmap -o correlations.png
    python3 chart_templates.py data.csv --type scatter --x feature_a --y target -o scatter.png
    python3 chart_templates.py data.csv --type box --x group --y value -o box.png
"""

import argparse
import sys

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def setup_style():
    sns.set_theme(style="whitegrid", palette="husl", font_scale=1.1)
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["figure.dpi"] = 150


def bar_chart(df, x, y, hue=None, output="bar_chart.png"):
    fig, ax = plt.subplots()
    sns.barplot(data=df, x=x, y=y, hue=hue, ax=ax)
    ax.set_title(f"{y} by {x}")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output}")


def line_chart(df, x, y, hue=None, output="line_chart.png"):
    fig, ax = plt.subplots()
    if hue:
        for group in df[hue].unique():
            subset = df[df[hue] == group]
            ax.plot(subset[x], subset[y], label=group, marker="o", markersize=4)
        ax.legend()
    else:
        ax.plot(df[x], df[y], marker="o", markersize=4)
    ax.set_title(f"{y} over {x}")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output}")


def histogram(df, x, output="histogram.png"):
    fig, ax = plt.subplots()
    sns.histplot(df[x], kde=True, ax=ax)
    ax.set_title(f"Distribution of {x}")
    plt.tight_layout()
    plt.savefig(output, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output}")


def heatmap(df, output="heatmap.png"):
    numeric = df.select_dtypes(include="number")
    corr = numeric.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, square=True, linewidths=0.5, ax=ax)
    ax.set_title("Feature Correlations")
    plt.tight_layout()
    plt.savefig(output, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output}")


def scatter_chart(df, x, y, hue=None, output="scatter.png"):
    fig, ax = plt.subplots()
    sns.regplot(data=df, x=x, y=y, scatter_kws={"alpha": 0.5}, ax=ax)
    ax.set_title(f"{x} vs {y}")
    plt.tight_layout()
    plt.savefig(output, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output}")


def box_chart(df, x, y, output="box.png"):
    fig, ax = plt.subplots()
    sns.boxplot(data=df, x=x, y=y, ax=ax)
    ax.set_title(f"{y} by {x}")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output}")


CHART_TYPES = {
    "bar": bar_chart,
    "line": line_chart,
    "hist": histogram,
    "heatmap": heatmap,
    "scatter": scatter_chart,
    "box": box_chart,
}


def main():
    parser = argparse.ArgumentParser(description="Generate charts from CSV data")
    parser.add_argument("file", help="Path to CSV file")
    parser.add_argument("--type", "-t", required=True, choices=CHART_TYPES.keys(), help="Chart type")
    parser.add_argument("--x", help="X-axis column")
    parser.add_argument("--y", help="Y-axis column")
    parser.add_argument("--hue", help="Color grouping column")
    parser.add_argument("--output", "-o", default=None, help="Output file path")
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.file)
    except Exception as e:
        print(f"Error reading {args.file}: {e}", file=sys.stderr)
        sys.exit(1)

    setup_style()
    output = args.output or f"{args.type}_chart.png"

    if args.type == "heatmap":
        heatmap(df, output)
    elif args.type == "hist":
        if not args.x:
            print("Error: --x required for histogram", file=sys.stderr)
            sys.exit(1)
        histogram(df, args.x, output)
    else:
        if not args.x or not args.y:
            print(f"Error: --x and --y required for {args.type} chart", file=sys.stderr)
            sys.exit(1)
        CHART_TYPES[args.type](df, args.x, args.y, **({} if args.type in ("hist", "heatmap", "box") else {"hue": args.hue}), output=output)


if __name__ == "__main__":
    main()
