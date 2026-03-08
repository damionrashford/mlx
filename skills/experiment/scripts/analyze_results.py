#!/usr/bin/env python3
"""Analyze experiment results from a TSV file.

Usage:
    python3 analyze_results.py results.tsv
"""
import sys
import csv
from collections import defaultdict


def analyze(filepath: str) -> None:
    rows = []
    with open(filepath, "r") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(row)

    if not rows:
        print("No experiments found.")
        return

    total = len(rows)
    kept = [r for r in rows if r.get("status", "").upper() == "KEEP"]
    discarded = [r for r in rows if r.get("status", "").upper() == "DISCARD"]
    crashed = [r for r in rows if r.get("status", "").upper() == "CRASH"]

    print(f"=== Experiment Analysis: {filepath} ===\n")
    print(f"Total experiments: {total}")
    print(f"  Kept:      {len(kept)}")
    print(f"  Discarded: {len(discarded)}")
    print(f"  Crashed:   {len(crashed)}")

    # Find the metric column
    metric_col = "val_score"
    if metric_col not in rows[0] and "val_bpb" in rows[0]:
        metric_col = "val_bpb"

    # Best score
    valid_scores = []
    for r in kept:
        try:
            score = float(r.get(metric_col, 0))
            if score > 0:
                valid_scores.append((score, r))
        except (ValueError, TypeError):
            continue

    if valid_scores:
        valid_scores.sort(key=lambda x: x[0])

        # Determine if lower is better (bpb, loss) or higher (accuracy, f1)
        lower_is_better = metric_col in ("val_bpb", "loss", "rmse", "mae", "mse")

        if lower_is_better:
            best_score, best_row = valid_scores[0]
            baseline_score = valid_scores[-1][0] if len(valid_scores) > 1 else best_score
        else:
            best_score, best_row = valid_scores[-1]
            baseline_score = valid_scores[0][0] if len(valid_scores) > 1 else best_score

        print(f"\nMetric: {metric_col} ({'lower' if lower_is_better else 'higher'} is better)")
        print(f"  Baseline: {baseline_score:.6f}")
        print(f"  Best:     {best_score:.6f}")

        if lower_is_better:
            improvement = baseline_score - best_score
        else:
            improvement = best_score - baseline_score

        print(f"  Improvement: {improvement:.6f}")
        print(f"  Best experiment: {best_row.get('experiment_id', best_row.get('commit', 'N/A'))}")
        print(f"  Description: {best_row.get('description', 'N/A')}")

        # Top 5 experiments
        if lower_is_better:
            top5 = valid_scores[:5]
        else:
            top5 = valid_scores[-5:][::-1]

        print(f"\nTop 5 experiments:")
        for score, row in top5:
            exp_id = row.get("experiment_id", row.get("commit", "?"))
            desc = row.get("description", "")
            print(f"  {exp_id}: {score:.6f} — {desc}")
    else:
        print(f"\nNo valid {metric_col} scores found in kept experiments.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <results.tsv>", file=sys.stderr)
        sys.exit(1)
    analyze(sys.argv[1])
