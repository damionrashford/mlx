---
name: explore
description: >
  Perform exploratory data analysis on datasets. Generates statistical summaries,
  distribution checks, correlation analysis, missing value reports, and feature profiling.
  Part of the mlx workbench. Use when the user wants to explore data, analyze a dataset,
  profile columns, check distributions, or understand their data before modeling.
allowed-tools: Bash, Read, Write, Glob, Grep
user-invocable: true
argument-hint: path to dataset (e.g. "data/train.csv")
---

# Exploratory Data Analysis

Reference protocol for systematic dataset exploration. Follow this order.

## Quick start

Run the full EDA pipeline:

```bash
python3 ${CLAUDE_SKILL_DIR}/scripts/eda.py $ARGUMENTS
python3 ${CLAUDE_SKILL_DIR}/scripts/eda.py data/train.csv --target price
```

The [eda.py](scripts/eda.py) script runs all 9 checks below in order and prints a complete report.

## Column Classification

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

## EDA Pipeline Steps

1. **Overview** — shape, memory, types, first rows
2. **Missing values** — count and percentage per column, sorted by severity
3. **Numeric features** — describe stats, zero-heavy warnings, negative value notes
4. **Categorical features** — unique counts, top values, high cardinality, ID detection
5. **Correlations** — pairs with |r| > 0.7 flagged for multicollinearity
6. **Target analysis** — classification vs regression, imbalance check (with `--target`)
7. **Duplicates** — exact duplicate row count and percentage
8. **Completeness scoring** — GREEN (>99%) / YELLOW (95-99%) / ORANGE (80-95%) / RED (<80%)
9. **Accuracy red flags** — placeholder values (0, -1, 999), round number bias, text placeholders

## Report format

Present findings in this structure:

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

## Red flags to always check

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
