---
name: prepare
description: >
  Generate data cleaning and preprocessing pipelines for pandas, polars, or PySpark.
  Handles missing values, duplicates, outliers, type fixing, encoding, and validation.
  Part of the mlx workbench. Use when the user wants to clean data, handle missing values,
  remove duplicates, fix data types, or preprocess a dataset.
allowed-tools: Bash, Read, Write, Glob, Grep
user-invocable: true
argument-hint: path to dataset to clean (e.g. "data/raw.csv")
---

# Data Cleaning & Preprocessing

Reference templates and guidelines for cleaning datasets.

## Cleaning order (always follow this sequence)

| Step | Operation | Strategy |
|------|-----------|----------|
| 1 | Remove duplicates | `drop_duplicates(subset=key_cols)` |
| 2 | Fix data types | Auto-detect dates, cast numerics, strip strings |
| 3 | Handle missing | Numeric: median. Categorical: mode/"unknown". Critical: drop |
| 4 | Remove outliers | IQR (1.5x) or Z-score (threshold=3) |
| 5 | Normalize text | Lowercase, strip whitespace |
| 6 | Encode categoricals | Label (ordinal) or one-hot (nominal) |
| 7 | Validate ranges | Domain constraints (age>0, price>=0) |
| 8 | Generate report | Before/after stats |

## Pipeline template (pandas)

```python
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class DataPreparer:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.report = {"original_shape": df.shape, "steps": []}

    def run(self) -> pd.DataFrame:
        self._remove_duplicates()
        self._fix_types()
        self._handle_missing()
        self._remove_outliers()
        self._validate()
        self.report["final_shape"] = self.df.shape
        return self.df

    def _remove_duplicates(self):
        before = len(self.df)
        self.df = self.df.drop_duplicates()
        removed = before - len(self.df)
        self._log(f"Removed {removed} duplicates")

    def _fix_types(self):
        for col in self.df.select_dtypes(include='object'):
            try:
                self.df[col] = pd.to_datetime(self.df[col])
                self._log(f"Converted {col} to datetime")
            except (ValueError, TypeError):
                pass

    def _handle_missing(self):
        for col in self.df.select_dtypes(include='number'):
            n = self.df[col].isnull().sum()
            if n > 0:
                self.df[col] = self.df[col].fillna(self.df[col].median())
                self._log(f"Filled {n} nulls in {col} (median)")
        for col in self.df.select_dtypes(include=['object', 'category']):
            n = self.df[col].isnull().sum()
            if n > 0:
                self.df[col] = self.df[col].fillna(self.df[col].mode().iloc[0])
                self._log(f"Filled {n} nulls in {col} (mode)")

    def _remove_outliers(self):
        for col in self.df.select_dtypes(include='number'):
            q1 = self.df[col].quantile(0.25)
            q3 = self.df[col].quantile(0.75)
            iqr = q3 - q1
            mask = (self.df[col] >= q1 - 1.5 * iqr) & (self.df[col] <= q3 + 1.5 * iqr)
            removed = (~mask).sum()
            if removed > 0:
                self.df = self.df[mask]
                self._log(f"Removed {removed} outliers from {col} (IQR)")

    def _validate(self):
        self._log(f"Final shape: {self.df.shape}")

    def _log(self, msg: str):
        self.report["steps"].append(msg)
        logger.info(msg)
```

## Alternative frameworks

### Polars (large datasets, faster)
```python
import polars as pl

def prepare(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.unique()
        .with_columns([
            pl.col(c).fill_null(pl.col(c).median()) for c in df.select(pl.col(pl.Float64)).columns
        ])
        .with_columns([
            pl.col(c).fill_null("unknown") for c in df.select(pl.col(pl.Utf8)).columns
        ])
    )
```

### PySpark (distributed)
```python
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, mean

def prepare(df: DataFrame) -> DataFrame:
    df = df.dropDuplicates()
    for field in df.schema.fields:
        if field.dataType.simpleString() in ("double", "float", "int"):
            avg = df.select(mean(col(field.name))).first()[0]
            df = df.fillna({field.name: avg or 0})
        elif field.dataType.simpleString() == "string":
            df = df.fillna({field.name: "unknown"})
    return df
```

## Quality report format

After cleaning, produce a report in this format:
```
=== Data Cleaning Report ===
Original: (10000, 25) → Final: (9847, 25)
  1. Removed 42 duplicates
  2. Fixed 3 date columns
  3. Filled 156 numeric nulls (median)
  4. Filled 89 categorical nulls (mode)
  5. Removed 111 outliers (IQR)
Rows removed: 153 (1.53%)
```

## Data quality checks

Run after cleaning to validate the result:

```python
def data_quality_checks(df):
    issues = []
    missing = df.isnull().sum()
    if missing.sum() > 0:
        issues.append(f"Missing values:\n{missing[missing > 0]}")
    dupes = df.duplicated().sum()
    if dupes > 0:
        issues.append(f"{dupes} duplicate rows")
    if 'id' in df.columns and df['id'].duplicated().any():
        issues.append("Duplicate IDs found")
    if issues:
        for i in issues: print(f"WARNING: {i}")
    else:
        print("All quality checks passed")
    return len(issues) == 0
```

## Pipeline config (YAML template)

```yaml
cleaning_pipeline:
  remove_duplicates:
    enabled: true
    subset: ['id', 'email']
    keep: 'first'
  missing_values:
    strategy: auto
    drop_threshold: 50  # Drop columns >50% missing
    numeric_fill: median
    categorical_fill: mode
  outliers:
    method: iqr
    threshold: 1.5
    columns: ['age', 'price']
  validation:
    ranges:
      age: [0, 120]
      price: [0, 1000000]
    required_columns: ['id', 'name']
```

## Rules

- Always keep original data — clean into a copy
- Log every step with counts
- Remove duplicates BEFORE handling nulls
- Test on a sample before full dataset
