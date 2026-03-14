# Analysis Methods Quick Reference

## Analysis Type Selection

| Question | Analysis Type | Script |
|----------|--------------|--------|
| What happened? | Descriptive statistics, aggregations | descriptive_stats.py |
| Why did it happen? | Diagnostic analysis, segmentation | rfm_segmentation.py |
| Is this difference real? | Hypothesis testing (t-test, chi-square) | hypothesis_test.py |
| Did the change work? | A/B test analysis | ab_test.py |
| How do groups behave over time? | Cohort analysis | cohort_analysis.py |
| What are the natural groupings? | Segmentation / clustering | rfm_segmentation.py |
| What are the trends? | Time series, rolling averages | trend_analysis.py |
| What should we track? | KPI definition, dashboarding | descriptive_stats.py |
| Is this ready to share? | Pre-delivery QA, sanity checking | validate.py |

## Statistical Test Selection

| Scenario | Test |
|----------|------|
| Compare 2 group means (normal) | Independent t-test |
| Compare 2 group means (non-normal) | Mann-Whitney U |
| Compare 2 paired measurements | Paired t-test |
| Compare 3+ group means | One-way ANOVA |
| Compare proportions | Chi-square test |
| Test correlation | Pearson / Spearman |
| Test normality | Shapiro-Wilk |

## Measure of Center Selection

| Situation | Use |
|-----------|-----|
| Symmetric distribution, no outliers | Mean |
| Skewed distribution (revenue, duration) | Median |
| Categorical or ordinal data | Mode |
| Highly skewed with outliers | Median + mean (gap shows skew) |

## Forecasting Methods

| Method | When to Use |
|--------|-------------|
| Naive (tomorrow = today) | Baseline comparison |
| Seasonal naive (same day last period) | Seasonal data |
| Linear trend | Clearly linear trends |
| Moving average | Noisy data |
