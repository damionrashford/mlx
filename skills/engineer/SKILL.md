---
name: engineer
description: >
  Generate feature engineering code for tabular, time series, text, and categorical data.
  Creates transformations, encodings, interaction terms, and aggregations.
  Use when the user wants to create features, encode categories, transform columns, add
  rolling windows, or build interaction terms.
allowed-tools: Bash, Read, Write, Glob, Grep
argument-hint: path to clean dataset (e.g. "data/clean.csv")
---

# Feature Engineering

Transforms clean data into model-ready features. Run the script for automated engineering, or use the recipes below for custom transforms.

## Available scripts

- **[engineer_features.py](scripts/engineer_features.py)** — Auto-detect column types and apply standard transforms to a CSV dataset

### Quick start

```bash
# Auto-engineer all columns
python3 ${CLAUDE_SKILL_DIR}/scripts/engineer_features.py data/clean.csv -o data/features.csv

# Engineer specific columns with target encoding
python3 ${CLAUDE_SKILL_DIR}/scripts/engineer_features.py data/clean.csv --cols age income category --target price -o features.csv

# Generate interaction features
python3 ${CLAUDE_SKILL_DIR}/scripts/engineer_features.py data/clean.csv --interactions -o features.csv

# Time series features
python3 ${CLAUDE_SKILL_DIR}/scripts/engineer_features.py data/clean.csv --cols revenue --types timeseries -o features.csv

# Group aggregations
python3 ${CLAUDE_SKILL_DIR}/scripts/engineer_features.py data/clean.csv --group segment revenue -o features.csv

# Summary as JSON
python3 ${CLAUDE_SKILL_DIR}/scripts/engineer_features.py data/clean.csv --json
```

Flags: `--cols` (specific columns), `--types` (numeric, categorical, datetime, text, timeseries), `--target` (target column for encoding), `--interactions`, `--group GROUP_COL AGG_COL`, `--json`, `-o OUTPUT`

## Decision guide

| Data type | Transform |
|-----------|-----------|
| Skewed numeric | Log, sqrt |
| High cardinality categorical | Target/frequency encoding |
| Low cardinality categorical | One-hot |
| Datetime | Year/month/day + cyclical |
| Free text | Length, word count |
| Multiple numeric | Interactions, ratios |
| Time series | Rolling stats, lags, diffs |
| Grouped data | Aggregations, deviation from mean |

## Feature selection (after engineering)

```python
from sklearn.feature_selection import mutual_info_classif

mi = mutual_info_classif(X.fillna(0), y, random_state=42)
top = pd.Series(mi, index=X.columns).sort_values(ascending=False).head(20)
print(top)
```
