---
name: analyze
description: >
  Statistical analysis, hypothesis testing, A/B testing, cohort analysis,
  segmentation, trend detection, and business metrics. Use when the user asks
  to "analyze this data", "run a statistical test", "compare groups", "find trends",
  "do A/B test analysis", "segment customers", "calculate KPIs", or mentions
  hypothesis testing, significance testing, cohort analysis, or business analytics.
allowed-tools: Bash, Read, Write, Glob, Grep
user-invocable: true
argument-hint: path to dataset or description of analysis (e.g. "data/sales.csv" or "compare conversion rates")
---

# Statistical & Business Analysis

Frameworks for answering business questions with data: descriptive statistics, hypothesis testing, cohort analysis, segmentation, trend detection, and KPI calculation.

## Analysis type selection

| Question | Analysis type |
|----------|--------------|
| What happened? | Descriptive statistics, aggregations |
| Why did it happen? | Diagnostic analysis, drill-downs, segmentation |
| Is this difference real? | Hypothesis testing (t-test, chi-square) |
| Did the change work? | A/B test analysis |
| How do groups behave over time? | Cohort analysis |
| What are the natural groupings? | Segmentation / clustering |
| What are the trends? | Time series decomposition, rolling averages |
| What should we track? | KPI definition and dashboarding |

## Descriptive statistics

```python
import pandas as pd
import numpy as np

df = pd.read_csv('data.csv')

# Quick overview
print(f"Shape: {df.shape}")
print(f"\nNumeric summary:\n{df.describe()}")
print(f"\nMissing values:\n{df.isnull().sum()}")

# Aggregations by group
summary = df.groupby('segment').agg(
    count=('id', 'count'),
    mean_value=('value', 'mean'),
    median_value=('value', 'median'),
    total_value=('value', 'sum'),
    std_value=('value', 'std'),
).round(2)
print(summary)

# Percentiles
percentiles = df['value'].quantile([0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
print(f"\nPercentiles:\n{percentiles}")
```

## Hypothesis testing

### Choosing the right test

| Scenario | Test | Python |
|----------|------|--------|
| Compare 2 group means (normal) | Independent t-test | `scipy.stats.ttest_ind` |
| Compare 2 group means (non-normal) | Mann-Whitney U | `scipy.stats.mannwhitneyu` |
| Compare 2 paired measurements | Paired t-test | `scipy.stats.ttest_rel` |
| Compare 3+ group means | One-way ANOVA | `scipy.stats.f_oneway` |
| Compare proportions | Chi-square test | `scipy.stats.chi2_contingency` |
| Test correlation | Pearson / Spearman | `scipy.stats.pearsonr` / `spearmanr` |
| Test normality | Shapiro-Wilk | `scipy.stats.shapiro` |

### Template

```python
from scipy import stats

# 1. State hypotheses
# H0: No difference between groups
# H1: There is a difference

# 2. Check assumptions
stat, p_normal = stats.shapiro(group_a)  # normality
stat, p_var = stats.levene(group_a, group_b)  # equal variance

# 3. Run test
if p_normal > 0.05:  # normal distribution
    stat, p_value = stats.ttest_ind(group_a, group_b, equal_var=(p_var > 0.05))
    test_name = "t-test"
else:  # non-normal
    stat, p_value = stats.mannwhitneyu(group_a, group_b, alternative='two-sided')
    test_name = "Mann-Whitney U"

# 4. Interpret
alpha = 0.05
effect_size = (group_a.mean() - group_b.mean()) / np.sqrt(
    (group_a.std()**2 + group_b.std()**2) / 2
)  # Cohen's d

print(f"Test: {test_name}")
print(f"Statistic: {stat:.4f}, p-value: {p_value:.4f}")
print(f"Effect size (Cohen's d): {effect_size:.3f}")
print(f"Result: {'Significant' if p_value < alpha else 'Not significant'} at alpha={alpha}")
```

### Effect size interpretation

| Cohen's d | Interpretation |
|-----------|---------------|
| < 0.2 | Negligible |
| 0.2 - 0.5 | Small |
| 0.5 - 0.8 | Medium |
| > 0.8 | Large |

## A/B test analysis

```python
def ab_test_analysis(control, treatment, metric='conversion', alpha=0.05):
    """Complete A/B test analysis with sample size check."""
    n_c, n_t = len(control), len(treatment)
    mean_c, mean_t = control.mean(), treatment.mean()

    # Relative lift
    lift = (mean_t - mean_c) / mean_c * 100

    # Statistical test
    if metric == 'conversion':  # proportions
        # Z-test for proportions
        p_pool = (control.sum() + treatment.sum()) / (n_c + n_t)
        se = np.sqrt(p_pool * (1 - p_pool) * (1/n_c + 1/n_t))
        z = (mean_t - mean_c) / se
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    else:  # continuous metric
        stat, p_value = stats.ttest_ind(control, treatment)

    # Confidence interval for difference
    se_diff = np.sqrt(control.var()/n_c + treatment.var()/n_t)
    ci_low = (mean_t - mean_c) - 1.96 * se_diff
    ci_high = (mean_t - mean_c) + 1.96 * se_diff

    print(f"Control: {mean_c:.4f} (n={n_c})")
    print(f"Treatment: {mean_t:.4f} (n={n_t})")
    print(f"Lift: {lift:+.2f}%")
    print(f"p-value: {p_value:.4f}")
    print(f"95% CI for difference: [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"Result: {'Significant' if p_value < alpha else 'Not significant'}")
```

## Cohort analysis

```python
def cohort_analysis(df, user_col, date_col, value_col, freq='M'):
    """Retention-style cohort analysis."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    # Assign cohort (first activity month)
    df['cohort'] = df.groupby(user_col)[date_col].transform('min').dt.to_period(freq)
    df['period'] = df[date_col].dt.to_period(freq)
    df['cohort_age'] = (df['period'] - df['cohort']).apply(lambda x: x.n)

    # Build cohort table
    cohort_table = df.groupby(['cohort', 'cohort_age'])[user_col].nunique().reset_index()
    cohort_table = cohort_table.pivot(index='cohort', columns='cohort_age', values=user_col)

    # Retention rates
    cohort_sizes = cohort_table[0]
    retention = cohort_table.div(cohort_sizes, axis=0).round(3)

    return retention
```

## Segmentation

```python
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def rfm_segmentation(df, customer_col, date_col, value_col, n_segments=4):
    """RFM (Recency, Frequency, Monetary) segmentation."""
    now = df[date_col].max() + pd.Timedelta(days=1)

    rfm = df.groupby(customer_col).agg(
        recency=(date_col, lambda x: (now - x.max()).days),
        frequency=(date_col, 'count'),
        monetary=(value_col, 'sum'),
    )

    # Score each dimension (1=worst, n_segments=best)
    for col in ['frequency', 'monetary']:
        rfm[f'{col}_score'] = pd.qcut(rfm[col], n_segments, labels=range(1, n_segments+1))
    rfm['recency_score'] = pd.qcut(rfm['recency'], n_segments, labels=range(n_segments, 0, -1))

    rfm['rfm_score'] = (rfm['recency_score'].astype(int) +
                         rfm['frequency_score'].astype(int) +
                         rfm['monetary_score'].astype(int))
    return rfm
```

## Trend analysis

```python
def trend_analysis(series, window=7):
    """Decompose time series into trend, seasonality, and residuals."""
    from statsmodels.tsa.seasonal import seasonal_decompose

    # Moving averages
    ma = series.rolling(window=window, center=True).mean()

    # Growth rates
    pct_change = series.pct_change(periods=window) * 100

    # Decomposition (if enough data)
    if len(series) >= 2 * window:
        decomp = seasonal_decompose(series, period=window, model='additive')
        return {'trend': decomp.trend, 'seasonal': decomp.seasonal,
                'residual': decomp.resid, 'ma': ma, 'growth': pct_change}

    return {'ma': ma, 'growth': pct_change}
```

## KPI framework

### Common business KPIs

| Category | KPI | Formula |
|----------|-----|---------|
| Revenue | MRR | Sum of monthly recurring revenue |
| Revenue | ARPU | Total revenue / active users |
| Growth | MoM Growth | (this_month - last_month) / last_month |
| Retention | Churn Rate | Lost customers / start customers |
| Retention | Retention Rate | 1 - churn rate |
| Engagement | DAU/MAU | Daily active / monthly active |
| Efficiency | CAC | Marketing spend / new customers |
| Efficiency | LTV | ARPU * avg lifetime months |
| Efficiency | LTV:CAC | LTV / CAC (target: > 3:1) |
| Conversion | Conversion Rate | Conversions / visitors |
| Conversion | Funnel Drop-off | Lost at each stage / entered stage |

## Analysis report format

```
=== Analysis Report ===
Question: [What business question are we answering?]
Data: [Dataset, date range, filters applied]
Method: [Statistical test / analysis type used]

Key Findings:
1. [Most important finding with numbers]
2. [Second finding]
3. [Third finding]

Statistical Evidence:
- Test: [name], p-value: [value], effect size: [value]
- Confidence interval: [range]

Caveats:
- [Sample size limitations]
- [Selection bias concerns]
- [Missing data impact]

Recommendation:
[Actionable next step based on findings]
```

## Rules

- State the business question BEFORE running any analysis
- Always check sample sizes — small samples produce unreliable results
- Report effect sizes alongside p-values — statistical significance is not practical significance
- Use confidence intervals, not just point estimates
- Segment before aggregating — averages hide important patterns
- Check for confounding variables before claiming causation
- Round results appropriately — false precision erodes trust
- Explain findings in plain language — stakeholders don't read code
