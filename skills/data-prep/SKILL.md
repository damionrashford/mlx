---
name: data-prep
description: >
  Explore and clean datasets end-to-end: statistical profiling, distribution checks,
  missing value analysis, duplicate detection, outlier removal, type fixing, and encoding.
  Supports pandas, polars, and PySpark. Use when the user wants to explore data, profile
  columns, understand a dataset, clean data, handle missing values, remove duplicates, fix
  data types, or preprocess a dataset before modeling.
allowed-tools: Bash, Read, Write, Glob, Grep
argument-hint: path to dataset (e.g. "data/train.csv")
---

# Data Preparation: Explore & Clean

Two-phase workflow for systematic data preparation. Always run EDA first — findings drive the cleaning strategy.

## Available scripts

| Script | Usage |
|--------|-------|
| [eda.py](scripts/eda.py) | `python3 ${CLAUDE_SKILL_DIR}/scripts/eda.py data.csv --target price` |
| [clean.py](scripts/clean.py) | `python3 ${CLAUDE_SKILL_DIR}/scripts/clean.py data.csv -o clean.csv` |

---

## Phase 1: Exploratory Data Analysis

### Quick start

```bash
python3 ${CLAUDE_SKILL_DIR}/scripts/eda.py $ARGUMENTS
python3 ${CLAUDE_SKILL_DIR}/scripts/eda.py data/train.csv --target price
```

The [eda.py](scripts/eda.py) script runs all 9 checks below in order and prints a complete report.

### Column Classification

Before profiling, classify each column:

| Type | Description | Examples |
|------|-------------|---------|
| **Identifier** | Unique keys, foreign keys | user_id, order_id, session_id |
| **Dimension** | Categorical attributes for grouping | status, region, category, plan_type |
| **Metric** | Quantitative values for measurement | revenue, count, duration, score |
| **Temporal** | Dates and timestamps | created_at, event_date, updated_at |
| **Text** | Free-form text fields | description, notes, comment |
| **Boolean** | True/false flags | is_active, has_subscription |

This classification drives which profiling checks to run on each column.

### EDA Pipeline Steps

1. **Overview** — shape, memory, types, first rows
2. **Missing values** — count and percentage per column, sorted by severity
3. **Numeric features** — describe stats, zero-heavy warnings, negative value notes
4. **Categorical features** — unique counts, top values, high cardinality, ID detection
5. **Correlations** — pairs with |r| > 0.7 flagged for multicollinearity
6. **Target analysis** — classification vs regression, imbalance check (with `--target`)
7. **Duplicates** — exact duplicate row count and percentage
8. **Completeness scoring** — GREEN (>99%) / YELLOW (95-99%) / ORANGE (80-95%) / RED (<80%)
9. **Accuracy red flags** — placeholder values (0, -1, 999), round number bias, text placeholders

### EDA report format

```
=== EDA Report: {filename} ===

Overview: 50,000 rows x 25 columns (15 numeric, 8 categorical, 2 datetime)

Data quality:
  - Missing: col_a (12%), col_b (8%), col_c (6%)
  - Duplicates: 42 rows (0.08%)
  - High cardinality: user_id (unique per row)

Key findings:
  1. feature_x and feature_y correlated (r=0.89) — multicollinearity
  2. Target imbalanced: 92% class 0, 8% class 1
  3. price has 500 zeros — possibly coded missing values
  4. date_created spans 2019-2024 — check for temporal leakage

Recommendations:
  1. Drop user_id (ID column, no predictive value)
  2. Handle class imbalance (SMOTE, class weights, stratified sampling)
  3. Investigate zero prices
  4. Use feature_x OR feature_y, not both
```

### Red flags to always check

- Columns that are 100% null
- Numeric columns stored as strings
- ID columns that could leak information
- Future data in features (temporal leakage)
- Constant columns (zero variance)
- Extreme class imbalance (>10:1)
- Placeholder values (0, -1, 999, 9999, "N/A", "TBD", "test")
- Default value dominance (one value has suspiciously high frequency)
- Round number bias (all values multiples of 5 or 10 — suggests estimation)
- Stale data (updated_at shows no recent changes in an active system)

---

## Phase 2: Data Cleaning

### Quick start

```bash
# Full cleaning pipeline
python3 ${CLAUDE_SKILL_DIR}/scripts/clean.py data.csv -o clean.csv

# Clean without outlier removal
python3 ${CLAUDE_SKILL_DIR}/scripts/clean.py data.csv --no-outliers -o clean.csv

# Save cleaning report as JSON
python3 ${CLAUDE_SKILL_DIR}/scripts/clean.py data.csv -o clean.csv --report report.json

# Quality check only (no cleaning)
python3 ${CLAUDE_SKILL_DIR}/scripts/clean.py data.csv --check-only
```

The [clean.py](scripts/clean.py) script runs the full cleaning pipeline: deduplication, type fixing, missing value handling, outlier removal, and validation. It prints a report to stderr and outputs the cleaned CSV.

### Cleaning order (always follow this sequence)

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

### Alternative frameworks

#### Polars (large datasets, faster)
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

#### PySpark (distributed)
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

### Pipeline config (YAML template)

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
- EDA findings drive cleaning decisions — don't clean blind
