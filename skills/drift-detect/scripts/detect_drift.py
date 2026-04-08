#!/usr/bin/env python3
# detect_drift.py — Run PSI + KS on numeric columns, chi-squared on categoricals.
# Outputs drift report to stdout and drift_report.html.

import sys
import os
import json
import math
import argparse
from pathlib import Path


def load_csv(path: str):
    """Load CSV using stdlib csv module."""
    import csv
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows


def to_numeric(values: list[str]) -> list[float]:
    """Convert string values to floats, skipping non-numeric."""
    result = []
    for v in values:
        try:
            result.append(float(v))
        except (ValueError, TypeError):
            pass
    return result


def is_numeric_column(values: list[str]) -> bool:
    """Check if a column is mostly numeric."""
    numeric = to_numeric(values[:100])
    return len(numeric) > 0.8 * min(100, len(values))


def compute_psi(reference: list[float], current: list[float], bins: int = 10) -> float:
    """Population Stability Index. >0.2 = significant, 0.1-0.2 = moderate."""
    if not reference or not current:
        return 0.0
    min_val = min(min(reference), min(current))
    max_val = max(max(reference), max(current))
    if min_val == max_val:
        return 0.0

    step = (max_val - min_val) / bins
    breakpoints = [min_val + i * step for i in range(bins + 1)]

    def bucket(values, bps):
        counts = [0] * bins
        for v in values:
            for i in range(bins):
                if bps[i] <= v < bps[i + 1]:
                    counts[i] += 1
                    break
            else:
                counts[-1] += 1
        return counts

    ref_counts = bucket(reference, breakpoints)
    cur_counts = bucket(current, breakpoints)

    psi = 0.0
    n_ref, n_cur = len(reference), len(current)
    for r, c in zip(ref_counts, cur_counts):
        r_pct = max(r / n_ref, 1e-4)
        c_pct = max(c / n_cur, 1e-4)
        psi += (c_pct - r_pct) * math.log(c_pct / r_pct)
    return psi


def ks_statistic(reference: list[float], current: list[float]) -> tuple[float, float]:
    """Kolmogorov-Smirnov statistic using stdlib. Returns (stat, approx_p)."""
    if not reference or not current:
        return 0.0, 1.0

    all_vals = sorted(set(reference + current))
    n, m = len(reference), len(current)
    ref_sorted = sorted(reference)
    cur_sorted = sorted(current)

    def ecdf(sorted_vals, x):
        lo, hi = 0, len(sorted_vals)
        while lo < hi:
            mid = (lo + hi) // 2
            if sorted_vals[mid] <= x:
                lo = mid + 1
            else:
                hi = mid
        return lo / len(sorted_vals)

    ks_stat = max(abs(ecdf(ref_sorted, x) - ecdf(cur_sorted, x)) for x in all_vals)

    # Approximate p-value via Kolmogorov distribution
    en = math.sqrt(n * m / (n + m))
    t = (en + 0.12 + 0.11 / en) * ks_stat
    # Approximate: p ≈ 2 * exp(-2 * t^2) for large n
    p_value = min(1.0, 2.0 * math.exp(-2.0 * t * t))
    return ks_stat, p_value


def chi2_test(reference: list[str], current: list[str]) -> tuple[float, float]:
    """Chi-squared test for categorical columns."""
    all_cats = set(reference) | set(current)
    n_ref, n_cur = len(reference), len(current)

    ref_counts = {c: 0 for c in all_cats}
    cur_counts = {c: 0 for c in all_cats}
    for v in reference:
        ref_counts[v] = ref_counts.get(v, 0) + 1
    for v in current:
        cur_counts[v] = cur_counts.get(v, 0) + 1

    chi2 = 0.0
    df = 0
    for cat in all_cats:
        r = ref_counts.get(cat, 0)
        c = cur_counts.get(cat, 0)
        # Scale current to same total as reference
        expected = (r + c) / 2
        if expected > 0:
            chi2 += ((r - expected) ** 2 + (c - expected) ** 2) / expected
            df += 1

    df = max(1, df - 1)
    # Approximate p-value via chi2 CDF (simplified for small df)
    # Use conservative approximation
    p_value = math.exp(-chi2 / 2) if chi2 < 50 else 0.0
    return chi2, p_value


def generate_html_report(results: list[dict], ref_path: str, cur_path: str) -> str:
    rows_html = ""
    for r in results:
        status_class = "drift-significant" if r["alert"] == "ALERT" else (
            "drift-moderate" if r["alert"] == "WARN" else "drift-none")
        rows_html += f"""
        <tr class="{status_class}">
          <td>{r['column']}</td>
          <td>{r['type']}</td>
          <td>{r.get('psi', 'N/A')}</td>
          <td>{r.get('ks_stat', 'N/A')}</td>
          <td>{r.get('ks_p', r.get('chi2_p', 'N/A'))}</td>
          <td><strong>{r['alert']}</strong> {r['message']}</td>
        </tr>"""

    return f"""<!DOCTYPE html>
<html>
<head>
<title>Drift Report</title>
<style>
body {{ font-family: sans-serif; margin: 2em; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
th {{ background: #333; color: white; }}
.drift-significant {{ background: #ffcccc; }}
.drift-moderate {{ background: #fff3cc; }}
.drift-none {{ background: #ccffcc; }}
</style>
</head>
<body>
<h1>Drift Detection Report</h1>
<p>Reference: <code>{ref_path}</code> &nbsp;|&nbsp; Current: <code>{cur_path}</code></p>
<table>
<tr><th>Column</th><th>Type</th><th>PSI</th><th>KS stat</th><th>p-value</th><th>Status</th></tr>
{rows_html}
</table>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="Detect data drift between two CSV datasets")
    parser.add_argument("reference", help="Reference dataset (training data)")
    parser.add_argument("current", help="Current dataset (production data)")
    parser.add_argument("--output", default="drift_report.html", help="HTML report output path")
    parser.add_argument("--psi-warn", type=float, default=0.1)
    parser.add_argument("--psi-alert", type=float, default=0.2)
    parser.add_argument("--p-threshold", type=float, default=0.05)
    args = parser.parse_args()

    print(f"Loading reference: {args.reference}")
    ref_rows = load_csv(args.reference)
    print(f"Loading current: {args.current}")
    cur_rows = load_csv(args.current)

    if not ref_rows or not cur_rows:
        sys.exit("Empty dataset")

    columns = list(ref_rows[0].keys())
    results = []
    alerts = 0
    warnings = 0

    print(f"\nAnalyzing {len(columns)} columns ({len(ref_rows)} ref vs {len(cur_rows)} cur rows)\n")

    for col in columns:
        ref_vals = [r.get(col, "") for r in ref_rows]
        cur_vals = [r.get(col, "") for r in cur_rows]

        if is_numeric_column(ref_vals):
            ref_num = to_numeric(ref_vals)
            cur_num = to_numeric(cur_vals)

            psi = compute_psi(ref_num, cur_num)
            ks_stat, ks_p = ks_statistic(ref_num, cur_num)

            if psi > args.psi_alert or ks_p < args.p_threshold:
                alert = "ALERT"
                alerts += 1
            elif psi > args.psi_warn:
                alert = "WARN"
                warnings += 1
            else:
                alert = "OK"

            msg = f"PSI={psi:.3f}, KS={ks_stat:.3f}, p={ks_p:.3f}"
            results.append({
                "column": col, "type": "numeric",
                "psi": f"{psi:.3f}", "ks_stat": f"{ks_stat:.3f}", "ks_p": f"{ks_p:.4f}",
                "alert": alert, "message": msg,
            })
            status_sym = "🔴" if alert == "ALERT" else ("🟡" if alert == "WARN" else "🟢")
            print(f"  {status_sym} {col:<30} {alert:<8} {msg}")

        else:
            # Categorical
            chi2, p = chi2_test(ref_vals, cur_vals)
            if p < args.p_threshold:
                alert = "ALERT"
                alerts += 1
            else:
                alert = "OK"

            msg = f"chi2={chi2:.3f}, p={p:.4f}"
            results.append({
                "column": col, "type": "categorical",
                "chi2_p": f"{p:.4f}",
                "alert": alert, "message": msg,
            })
            status_sym = "🔴" if alert == "ALERT" else "🟢"
            print(f"  {status_sym} {col:<30} {alert:<8} {msg}")

    print(f"\nSummary: {alerts} ALERT, {warnings} WARN, {len(results) - alerts - warnings} OK")

    # Write HTML report
    html = generate_html_report(results, args.reference, args.current)
    with open(args.output, "w") as f:
        f.write(html)
    print(f"Report saved: {args.output}")


if __name__ == "__main__":
    main()
