---
name: engineer
description: >
  Generate feature engineering code for tabular, time series, text, and categorical data.
  Creates transformations, encodings, interaction terms, and aggregations.
  Part of the mlx workbench. Use when the user wants to create features, encode categories,
  transform columns, add rolling windows, or build interaction terms.
allowed-tools: Bash, Read, Write, Glob, Grep
user-invocable: true
argument-hint: path to clean dataset (e.g. "data/clean.csv")
---

# Feature Engineering

Code recipes and patterns for transforming clean data into model-ready features.

## Recipes by data type

### Numeric
```python
def engineer_numeric(df, col):
    df = df.copy()
    if df[col].min() >= 0:
        df[f'{col}_log'] = np.log1p(df[col])
    df[f'{col}_binned'] = pd.qcut(df[col], q=5, labels=False, duplicates='drop')
    df[f'{col}_zscore'] = (df[col] - df[col].mean()) / df[col].std()
    q1, q99 = df[col].quantile([0.01, 0.99])
    df[f'{col}_clipped'] = df[col].clip(q1, q99)
    return df
```

### Categorical
```python
def engineer_categorical(df, col, target=None):
    df = df.copy()
    freq = df[col].value_counts(normalize=True)
    df[f'{col}_freq'] = df[col].map(freq)
    if target and target in df.columns:
        means = df.groupby(col)[target].mean()
        counts = df.groupby(col)[target].count()
        smooth = 20
        global_mean = df[target].mean()
        df[f'{col}_target_enc'] = df[col].map(
            (counts * means + smooth * global_mean) / (counts + smooth)
        )
    if df[col].nunique() <= 10:
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
        df = pd.concat([df, dummies], axis=1)
    return df
```

### Datetime
```python
def engineer_datetime(df, col):
    df = df.copy()
    dt = pd.to_datetime(df[col])
    df[f'{col}_year'] = dt.dt.year
    df[f'{col}_month'] = dt.dt.month
    df[f'{col}_dayofweek'] = dt.dt.dayofweek
    df[f'{col}_hour'] = dt.dt.hour
    df[f'{col}_is_weekend'] = dt.dt.dayofweek.isin([5, 6]).astype(int)
    df[f'{col}_month_sin'] = np.sin(2 * np.pi * dt.dt.month / 12)
    df[f'{col}_month_cos'] = np.cos(2 * np.pi * dt.dt.month / 12)
    return df
```

### Text
```python
def engineer_text(df, col):
    df = df.copy()
    text = df[col].fillna('')
    df[f'{col}_length'] = text.str.len()
    df[f'{col}_word_count'] = text.str.split().str.len()
    df[f'{col}_has_numbers'] = text.str.contains(r'\d').astype(int)
    df[f'{col}_unique_words'] = text.apply(
        lambda x: len(set(x.lower().split())) if x.strip() else 0
    )
    return df
```

### Interactions
```python
def engineer_interactions(df, cols):
    df = df.copy()
    numeric = [c for c in cols if df[c].dtype in ['int64', 'float64']]
    for i, c1 in enumerate(numeric):
        for c2 in numeric[i+1:]:
            df[f'{c1}_x_{c2}'] = df[c1] * df[c2]
            df[f'{c1}_div_{c2}'] = df[c1] / df[c2].replace(0, np.nan)
    return df
```

### Time series
```python
def engineer_timeseries(df, col, windows=[7, 14, 30]):
    df = df.copy()
    for w in windows:
        df[f'{col}_roll_mean_{w}'] = df[col].rolling(w).mean()
        df[f'{col}_roll_std_{w}'] = df[col].rolling(w).std()
    for lag in [1, 3, 7]:
        df[f'{col}_lag_{lag}'] = df[col].shift(lag)
    df[f'{col}_diff'] = df[col].diff()
    df[f'{col}_pct_change'] = df[col].pct_change()
    return df
```

### Group aggregations
```python
def engineer_aggregations(df, group_col, agg_col):
    df = df.copy()
    agg = df.groupby(group_col)[agg_col].agg(['mean', 'std', 'min', 'max'])
    agg.columns = [f'{group_col}_{agg_col}_{s}' for s in agg.columns]
    df = df.merge(agg, left_on=group_col, right_index=True, how='left')
    df[f'{agg_col}_dev_from_{group_col}'] = df[agg_col] - df[f'{group_col}_{agg_col}_mean']
    return df
```

## Feature selection (after engineering)

```python
from sklearn.feature_selection import mutual_info_classif

mi = mutual_info_classif(X.fillna(0), y, random_state=42)
top = pd.Series(mi, index=X.columns).sort_values(ascending=False).head(20)
print(top)
```

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
