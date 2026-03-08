# /// script
# dependencies = [
#   "pandas>=2.0",
#   "numpy>=1.24",
# ]
# requires-python = ">=3.10"
# ///
"""
Data cleaning CLI — remove duplicates, fix types, handle missing values,
remove outliers, and validate a CSV dataset. Outputs a cleaned CSV with
a report of all changes.
"""

import argparse
import json
import sys

import numpy as np
import pandas as pd


class DataPreparer:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.report: dict = {"original_shape": list(df.shape), "steps": []}

    def run(self) -> pd.DataFrame:
        self._remove_duplicates()
        self._fix_types()
        self._handle_missing()
        self._remove_outliers()
        self._validate()
        self.report["final_shape"] = list(self.df.shape)
        self.report["rows_removed"] = (
            self.report["original_shape"][0] - self.report["final_shape"][0]
        )
        return self.df

    def _remove_duplicates(self):
        before = len(self.df)
        self.df = self.df.drop_duplicates()
        removed = before - len(self.df)
        if removed > 0:
            self._log(f"Removed {removed} duplicate rows")

    def _fix_types(self):
        for col in self.df.select_dtypes(include="object"):
            try:
                self.df[col] = pd.to_datetime(self.df[col])
                self._log(f"Converted {col} to datetime")
            except (ValueError, TypeError):
                pass

    def _handle_missing(self):
        for col in self.df.select_dtypes(include="number"):
            n = self.df[col].isnull().sum()
            if n > 0:
                self.df[col] = self.df[col].fillna(self.df[col].median())
                self._log(f"Filled {n} nulls in {col} (median)")
        for col in self.df.select_dtypes(include=["object", "category"]):
            n = self.df[col].isnull().sum()
            if n > 0:
                self.df[col] = self.df[col].fillna(self.df[col].mode().iloc[0])
                self._log(f"Filled {n} nulls in {col} (mode)")

    def _remove_outliers(self):
        for col in self.df.select_dtypes(include="number"):
            q1, q3 = self.df[col].quantile(0.25), self.df[col].quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue
            mask = (self.df[col] >= q1 - 1.5 * iqr) & (
                self.df[col] <= q3 + 1.5 * iqr
            )
            removed = (~mask).sum()
            if removed > 0:
                self.df = self.df[mask]
                self._log(f"Removed {removed} outliers from {col} (IQR)")

    def _validate(self):
        self._log(f"Final shape: {self.df.shape[0]} rows x {self.df.shape[1]} columns")

    def _log(self, msg: str):
        self.report["steps"].append(msg)


def data_quality_checks(df: pd.DataFrame) -> dict:
    issues = []
    missing = df.isnull().sum()
    if missing.sum() > 0:
        cols = missing[missing > 0]
        issues.append(
            {
                "type": "missing_values",
                "detail": {col: int(n) for col, n in cols.items()},
            }
        )
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        issues.append({"type": "duplicate_rows", "count": int(dup_count)})
    for col in df.columns:
        if "id" in col.lower() and df[col].duplicated().any():
            issues.append({"type": "duplicate_ids", "column": col})
    return {"passed": len(issues) == 0, "issues": issues}


def main():
    parser = argparse.ArgumentParser(
        description="Clean a CSV dataset: deduplicate, fix types, handle missing values, remove outliers.",
        epilog=(
            "Examples:\n"
            "  %(prog)s data.csv -o clean.csv\n"
            "  %(prog)s data.csv --no-outliers --report report.json\n"
            "  %(prog)s data.csv --check-only\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", help="Path to input CSV file")
    parser.add_argument("-o", "--output", help="Output CSV path (default: stdout)")
    parser.add_argument("--report", help="Save cleaning report as JSON to this path")
    parser.add_argument(
        "--no-outliers",
        action="store_true",
        help="Skip outlier removal",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Run quality checks only, do not clean (outputs JSON)",
    )

    args = parser.parse_args()

    try:
        df = pd.read_csv(args.input)
    except FileNotFoundError:
        print(f"Error: File not found: {args.input}", file=sys.stderr)
        print(f"Usage: python {sys.argv[0]} <path-to-csv>", file=sys.stderr)
        sys.exit(1)

    if args.check_only:
        result = data_quality_checks(df)
        print(json.dumps(result, indent=2))
        sys.exit(0 if result["passed"] else 1)

    preparer = DataPreparer(df)
    if args.no_outliers:
        preparer._remove_outliers = lambda: None  # type: ignore[method-assign]

    cleaned = preparer.run()
    report = preparer.report

    # Print report to stderr
    print("=== Data Cleaning Report ===", file=sys.stderr)
    print(
        f"Original: ({report['original_shape'][0]}, {report['original_shape'][1]}) "
        f"→ Final: ({report['final_shape'][0]}, {report['final_shape'][1]})",
        file=sys.stderr,
    )
    for i, step in enumerate(report["steps"], 1):
        print(f"  {i}. {step}", file=sys.stderr)
    pct = (
        report["rows_removed"] / report["original_shape"][0] * 100
        if report["original_shape"][0] > 0
        else 0
    )
    print(
        f"Rows removed: {report['rows_removed']} ({pct:.2f}%)",
        file=sys.stderr,
    )

    if args.report:
        with open(args.report, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to {args.report}", file=sys.stderr)

    if args.output:
        cleaned.to_csv(args.output, index=False)
        print(f"Cleaned data saved to {args.output}", file=sys.stderr)
    else:
        cleaned.to_csv(sys.stdout, index=False)


if __name__ == "__main__":
    main()
