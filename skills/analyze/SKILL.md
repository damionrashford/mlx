---
name: analyze
description: >
  Statistical analysis, hypothesis testing, A/B testing, cohort analysis,
  segmentation, trend detection, business metrics, and pre-delivery validation.
  Use when the user asks to "analyze this data", "run a statistical test",
  "compare groups", "find trends", "do A/B test analysis", "segment customers",
  "calculate KPIs", "validate this analysis", "check my work", "QA this",
  "sanity check", "review my numbers", or mentions hypothesis testing,
  significance testing, cohort analysis, business analytics, or data validation.
allowed-tools: Bash, Read, Write, Glob, Grep
argument-hint: path to dataset or description of analysis (e.g. "data/sales.csv" or "compare conversion rates")
---

# Statistical & Business Analysis

Frameworks for answering business questions with data: descriptive statistics, hypothesis testing, cohort analysis, segmentation, trend detection, KPI calculation, and pre-delivery QA.

## Scripts

| Script | Usage |
|--------|-------|
| [descriptive_stats.py](scripts/descriptive_stats.py) | `python3 ${CLAUDE_SKILL_DIR}/scripts/descriptive_stats.py data.csv --group segment --value revenue` |
| [hypothesis_test.py](scripts/hypothesis_test.py) | `python3 ${CLAUDE_SKILL_DIR}/scripts/hypothesis_test.py data.csv --col value --group segment --a control --b treatment` |
| [ab_test.py](scripts/ab_test.py) | `python3 ${CLAUDE_SKILL_DIR}/scripts/ab_test.py data.csv --col converted --group variant --control A --treatment B` |
| [cohort_analysis.py](scripts/cohort_analysis.py) | `python3 ${CLAUDE_SKILL_DIR}/scripts/cohort_analysis.py data.csv --user user_id --date order_date` |
| [rfm_segmentation.py](scripts/rfm_segmentation.py) | `python3 ${CLAUDE_SKILL_DIR}/scripts/rfm_segmentation.py data.csv --customer customer_id --date order_date --value revenue` |
| [trend_analysis.py](scripts/trend_analysis.py) | `python3 ${CLAUDE_SKILL_DIR}/scripts/trend_analysis.py data.csv --date date --value revenue --window 30` |
| [validate.py](scripts/validate.py) | `python3 ${CLAUDE_SKILL_DIR}/scripts/validate.py data.csv` |

## Analysis type selection

| Question | Analysis type | Script |
|----------|--------------|--------|
| What happened? | Descriptive statistics, aggregations | descriptive_stats.py |
| Why did it happen? | Diagnostic analysis, drill-downs, segmentation | rfm_segmentation.py |
| Is this difference real? | Hypothesis testing (t-test, chi-square) | hypothesis_test.py |
| Did the change work? | A/B test analysis | ab_test.py |
| How do groups behave over time? | Cohort analysis | cohort_analysis.py |
| What are the natural groupings? | Segmentation / clustering | rfm_segmentation.py |
| What are the trends? | Time series decomposition, rolling averages | trend_analysis.py |
| What should we track? | KPI definition and dashboarding | descriptive_stats.py |
| Is this ready to share? | Pre-delivery QA, sanity checking | validate.py |

## Choosing the right measure of center

| Situation | Use | Why |
|---|---|---|
| Symmetric distribution, no outliers | Mean | Most efficient estimator |
| Skewed distribution (revenue, duration) | Median | Robust to outliers |
| Categorical or ordinal data | Mode | Only option for non-numeric |
| Highly skewed with outliers | Median + mean | The gap shows skew |

**Always report mean and median together for business metrics.** If they diverge significantly, the data is skewed and the mean alone is misleading.

## Choosing the right test

| Scenario | Test |
|----------|------|
| Compare 2 group means (normal) | Independent t-test |
| Compare 2 group means (non-normal) | Mann-Whitney U |
| Compare 2 paired measurements | Paired t-test |
| Compare 3+ group means | One-way ANOVA |
| Compare proportions | Chi-square test |
| Test correlation | Pearson / Spearman |
| Test normality | Shapiro-Wilk |

The [hypothesis_test.py](scripts/hypothesis_test.py) script auto-selects the right test based on normality checks and reports p-value, effect size (Cohen's d), and confidence interval.

### Effect size interpretation

| Cohen's d | Interpretation |
|-----------|---------------|
| < 0.2 | Negligible |
| 0.2 - 0.5 | Small |
| 0.5 - 0.8 | Medium |
| > 0.8 | Large |

## KPI framework

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

## Simple forecasting (for analysts, not data scientists)

| Method | How | When |
|--------|-----|------|
| Naive | Tomorrow = today | Baseline |
| Seasonal naive | Tomorrow = same day last week/year | Seasonal data |
| Linear trend | Fit a line to historical data | Clearly linear trends |
| Moving average | Trailing average as forecast | Noisy data |

**Always communicate uncertainty** — provide a range, not a point estimate:
- "We expect 10K-12K signups next month based on the 3-month trend"
- NOT "We will get exactly 11,234 signups next month"

**When to escalate to a data scientist**: Non-linear trends, multiple seasonalities, external factors, or when forecast accuracy matters for resource allocation.

## Statistical pitfalls to watch for

### Simpson's Paradox
A trend in aggregated data can reverse when segmented. Always check whether conclusions hold across key segments.

### Multiple Comparisons Problem
Testing 20 metrics at p=0.05 means ~1 will be falsely significant. Apply Bonferroni correction (alpha / number of tests) or report how many tests were run.

### Ecological Fallacy
Aggregate trends may not apply to individuals. "Countries with higher X have higher Y" does NOT mean individuals with higher X have higher Y.

### Anchoring on False Precision
- "Churn will be 4.73% next quarter" implies more certainty than warranted
- Prefer ranges: "We expect churn between 4-6%"

### Correlation vs Causation
When you find a correlation, consider:
- **Reverse causation**: Maybe B causes A, not A causes B
- **Confounding**: Maybe C causes both A and B
- **Coincidence**: With enough variables, spurious correlations are inevitable

**What you can say**: "Users who use feature X have 30% higher retention"
**What you cannot say**: "Feature X causes 30% higher retention"

## Rules

- State the business question BEFORE running any analysis
- Always check sample sizes — small samples produce unreliable results
- Report effect sizes alongside p-values — statistical significance is not practical significance
- Use confidence intervals, not just point estimates
- Segment before aggregating — averages hide important patterns
- Check for confounding variables before claiming causation
- Round results appropriately — false precision erodes trust
- Explain findings in plain language — stakeholders don't read code

## Pre-Delivery QA & Validation

Pre-delivery QA checklist, common data analysis pitfalls, result sanity checking, and documentation standards.

### Pre-Delivery QA Checklist

Run through before sharing any analysis with stakeholders.

#### Data Quality Checks

- [ ] **Source verification**: Confirmed which tables/data sources were used. Are they the right ones?
- [ ] **Freshness**: Data is current enough. Noted the "as of" date.
- [ ] **Completeness**: No unexpected gaps in time series or missing segments.
- [ ] **Null handling**: Checked null rates in key columns. Nulls handled appropriately (excluded, imputed, or flagged).
- [ ] **Deduplication**: No double-counting from bad joins or duplicate source records.
- [ ] **Filter verification**: All filters are correct. No unintended exclusions.

#### Calculation Checks

- [ ] **Aggregation logic**: GROUP BY includes all non-aggregated columns. Aggregation level matches analysis grain.
- [ ] **Denominator correctness**: Rate and percentage calculations use the right denominator. Denominators are non-zero.
- [ ] **Date alignment**: Comparisons use same time period length. Partial periods excluded or noted.
- [ ] **Join correctness**: JOIN types appropriate. Many-to-many joins haven't inflated counts.
- [ ] **Metric definitions**: Metrics match how stakeholders define them. Deviations noted.
- [ ] **Subtotals sum**: Parts add up to the whole where expected. If not, explained why.

#### Reasonableness Checks

- [ ] **Magnitude**: Numbers in plausible range. No negative revenue. Percentages between 0-100%.
- [ ] **Trend continuity**: No unexplained jumps or drops in time series.
- [ ] **Cross-reference**: Key numbers match other known sources (dashboards, prior reports).
- [ ] **Order of magnitude**: Totals in the right ballpark vs known figures.
- [ ] **Edge cases**: Checked boundaries — empty segments, zero-activity periods, new entities.

#### Presentation Checks

- [ ] **Chart accuracy**: Bar charts start at zero. Axes labeled. Scales consistent across panels.
- [ ] **Number formatting**: Appropriate precision. Consistent formatting. Thousands separators.
- [ ] **Title clarity**: Titles state the insight, not just the metric. Date ranges specified.
- [ ] **Caveat transparency**: Known limitations and assumptions stated explicitly.
- [ ] **Reproducibility**: Someone else could recreate this analysis from the documentation.

### Common Data Analysis Pitfalls

#### Join Explosion
A many-to-many join silently multiplies rows, inflating counts and sums. Always check row counts after joins. Use `COUNT(DISTINCT id)` instead of `COUNT(*)` when counting entities through joins.

#### Survivorship Bias
Analyzing only entities that exist today, ignoring those that churned, failed, or were deleted. Ask "who is NOT in this dataset?" before drawing conclusions.

#### Incomplete Period Comparison
Comparing a partial period to a full period. "January revenue is $500K vs December's $800K" — but January isn't over yet. Filter to complete periods, or compare same-number-of-days.

#### Denominator Shifting
The denominator changes between periods, making rates incomparable. Use consistent definitions across all compared periods. Document any changes.

#### Average of Averages
Averaging pre-computed averages gives wrong results when group sizes differ. Always aggregate from raw data. Never average pre-aggregated averages.

#### Timezone Mismatches
Different data sources use different timezones, causing misalignment. Standardize all timestamps to a single timezone (UTC recommended) before analysis.

#### Selection Bias in Segmentation
Segments defined by the outcome you're measuring, creating circular logic. Define segments based on pre-treatment characteristics, not outcomes.

### Result Sanity Checking

#### Magnitude Checks

| Metric Type | Sanity Check |
|---|---|
| User counts | Match known MAU/DAU figures? |
| Revenue | Right order of magnitude vs known totals? |
| Rates | Between 0% and 100%? Match dashboard? |
| Growth rates | Is 50%+ MoM realistic or a data issue? |
| Averages | Reasonable given the distribution? |
| Percentages | Segment percentages sum to ~100%? |

#### Cross-Validation Techniques

1. **Calculate the same metric two different ways** and verify they match
2. **Spot-check individual records** — pick specific entities and trace manually
3. **Compare to known benchmarks** — match against dashboards, prior reports
4. **Reverse engineer** — if total revenue is X, does per-user revenue times user count equal X?
5. **Boundary checks** — filter to a single day/user/category. Are micro-results sensible?

#### Red Flags That Warrant Investigation

- Any metric changed >50% period-over-period without obvious cause
- Counts or sums that are exact round numbers (filter or default value issue)
- Rates exactly at 0% or 100% (incomplete data)
- Results that perfectly confirm the hypothesis (reality is messier)
- Identical values across time periods or segments (query ignoring a dimension)

### Documentation Template

Every non-trivial analysis should include:

```
## Analysis: [Title]

### Question
[The specific question being answered]

### Data Sources
- Table/file: [name] (as of [date])

### Definitions
- [Metric A]: [How it's calculated]
- [Segment X]: [How membership is determined]
- [Time period]: [Start] to [end], [timezone]

### Methodology
1. [Step 1]
2. [Step 2]

### Assumptions and Limitations
- [Assumption and why it's reasonable]
- [Limitation and its impact on conclusions]

### Key Findings
1. [Finding with evidence]

### Caveats
- [Things the reader should know before acting on this]
```

### Validation Rules

- Run the full checklist before sharing — skipping steps is how errors ship
- Check row counts before and after every join
- Never average pre-aggregated averages — go back to raw data
- Always ask "who is missing from this dataset?"
- Cross-reference key numbers against at least one other source
- State assumptions explicitly — unstated assumptions become invisible errors
- Round appropriately — false precision erodes trust
