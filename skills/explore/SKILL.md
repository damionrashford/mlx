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

## 1. Overview
```python
import pandas as pd
import numpy as np

df = pd.read_csv("$ARGUMENTS")

print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"Memory: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
print(f"\nColumn types:\n{df.dtypes.value_counts()}")
print(f"\nFirst 5 rows:\n{df.head()}")
```

## 2. Missing values
```python
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
report = pd.DataFrame({'count': missing, 'percent': missing_pct})
print(report.query('count > 0').sort_values('percent', ascending=False))
```

## 3. Numeric features
```python
numeric = df.select_dtypes(include='number')
print(numeric.describe().T[['count','mean','std','min','25%','50%','75%','max']])

for col in numeric.columns:
    if (df[col] == 0).sum() > len(df) * 0.5:
        print(f"  WARNING: {col} is >50% zeros")
    if (df[col] < 0).sum() > 0:
        print(f"  NOTE: {col} has negative values")
```

## 4. Categorical features
```python
cats = df.select_dtypes(include=['object', 'category'])
for col in cats.columns:
    n = df[col].nunique()
    top = df[col].mode().iloc[0] if not df[col].mode().empty else "N/A"
    print(f"  {col}: {n} unique, top='{top}'")
    if n > 50: print(f"    HIGH CARDINALITY")
    if n == len(df): print(f"    UNIQUE PER ROW — likely ID")
```

## 5. Correlations
```python
if len(numeric.columns) > 1:
    corr = numeric.corr()
    for i in range(len(corr.columns)):
        for j in range(i+1, len(corr.columns)):
            r = corr.iloc[i, j]
            if abs(r) > 0.7:
                print(f"  {corr.columns[i]} <-> {corr.columns[j]}: r={r:.3f}")
```

## 6. Target analysis
```python
# If user specifies a target column:
target = "target_col"
if df[target].dtype in ['object', 'category', 'bool']:
    print(f"Task: Classification")
    print(df[target].value_counts(normalize=True).round(3))
    if df[target].value_counts(normalize=True).min() < 0.1:
        print("  WARNING: Imbalanced classes!")
else:
    print(f"Task: Regression")
    print(f"  mean={df[target].mean():.4f}, std={df[target].std():.4f}")
```

## 7. Duplicates
```python
dupes = df.duplicated().sum()
print(f"Exact duplicates: {dupes} ({dupes/len(df)*100:.2f}%)")
```

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
