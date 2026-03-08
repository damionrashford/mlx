---
name: validate
description: >
  QA an analysis before sharing — methodology checks, calculation verification,
  bias detection, and sanity checking. Use when the user asks to "validate this
  analysis", "check my work", "QA this before I share it", "review my numbers",
  "sanity check these results", or mentions data validation, pre-delivery review,
  analysis QA, or checking for common pitfalls.
allowed-tools: Bash, Read, Write, Glob, Grep
user-invocable: true
argument-hint: path to analysis script, notebook, or results (e.g. "analysis.py" or "results.csv")
---

# Analysis Validation & QA

Pre-delivery QA checklist, common data analysis pitfalls, result sanity checking, and documentation standards.

## Quick start

Run automated validation checks on a dataset:

```bash
python3 ${CLAUDE_SKILL_DIR}/scripts/validate.py data.csv
python3 ${CLAUDE_SKILL_DIR}/scripts/validate.py data.csv --join-check other.csv --join-key user_id
```

The [validate.py](scripts/validate.py) script checks: magnitude issues, duplicates, time consistency, missing patterns, pre-aggregated averages, and join explosions.

## Pre-Delivery QA Checklist

Run through before sharing any analysis with stakeholders.

### Data Quality Checks

- [ ] **Source verification**: Confirmed which tables/data sources were used. Are they the right ones?
- [ ] **Freshness**: Data is current enough. Noted the "as of" date.
- [ ] **Completeness**: No unexpected gaps in time series or missing segments.
- [ ] **Null handling**: Checked null rates in key columns. Nulls handled appropriately (excluded, imputed, or flagged).
- [ ] **Deduplication**: No double-counting from bad joins or duplicate source records.
- [ ] **Filter verification**: All filters are correct. No unintended exclusions.

### Calculation Checks

- [ ] **Aggregation logic**: GROUP BY includes all non-aggregated columns. Aggregation level matches analysis grain.
- [ ] **Denominator correctness**: Rate and percentage calculations use the right denominator. Denominators are non-zero.
- [ ] **Date alignment**: Comparisons use same time period length. Partial periods excluded or noted.
- [ ] **Join correctness**: JOIN types appropriate. Many-to-many joins haven't inflated counts.
- [ ] **Metric definitions**: Metrics match how stakeholders define them. Deviations noted.
- [ ] **Subtotals sum**: Parts add up to the whole where expected. If not, explained why.

### Reasonableness Checks

- [ ] **Magnitude**: Numbers in plausible range. No negative revenue. Percentages between 0-100%.
- [ ] **Trend continuity**: No unexplained jumps or drops in time series.
- [ ] **Cross-reference**: Key numbers match other known sources (dashboards, prior reports).
- [ ] **Order of magnitude**: Totals in the right ballpark vs known figures.
- [ ] **Edge cases**: Checked boundaries — empty segments, zero-activity periods, new entities.

### Presentation Checks

- [ ] **Chart accuracy**: Bar charts start at zero. Axes labeled. Scales consistent across panels.
- [ ] **Number formatting**: Appropriate precision. Consistent formatting. Thousands separators.
- [ ] **Title clarity**: Titles state the insight, not just the metric. Date ranges specified.
- [ ] **Caveat transparency**: Known limitations and assumptions stated explicitly.
- [ ] **Reproducibility**: Someone else could recreate this analysis from the documentation.

## Common Data Analysis Pitfalls

### Join Explosion
A many-to-many join silently multiplies rows, inflating counts and sums. Always check row counts after joins. Use `COUNT(DISTINCT id)` instead of `COUNT(*)` when counting entities through joins.

### Survivorship Bias
Analyzing only entities that exist today, ignoring those that churned, failed, or were deleted. Ask "who is NOT in this dataset?" before drawing conclusions.

### Incomplete Period Comparison
Comparing a partial period to a full period. "January revenue is $500K vs December's $800K" — but January isn't over yet. Filter to complete periods, or compare same-number-of-days.

### Denominator Shifting
The denominator changes between periods, making rates incomparable. Use consistent definitions across all compared periods. Document any changes.

### Average of Averages
Averaging pre-computed averages gives wrong results when group sizes differ. Always aggregate from raw data. Never average pre-aggregated averages.

### Timezone Mismatches
Different data sources use different timezones, causing misalignment. Standardize all timestamps to a single timezone (UTC recommended) before analysis.

### Selection Bias in Segmentation
Segments defined by the outcome you're measuring, creating circular logic. Define segments based on pre-treatment characteristics, not outcomes.

## Result Sanity Checking

### Magnitude Checks

| Metric Type | Sanity Check |
|---|---|
| User counts | Match known MAU/DAU figures? |
| Revenue | Right order of magnitude vs known totals? |
| Rates | Between 0% and 100%? Match dashboard? |
| Growth rates | Is 50%+ MoM realistic or a data issue? |
| Averages | Reasonable given the distribution? |
| Percentages | Segment percentages sum to ~100%? |

### Cross-Validation Techniques

1. **Calculate the same metric two different ways** and verify they match
2. **Spot-check individual records** — pick specific entities and trace manually
3. **Compare to known benchmarks** — match against dashboards, prior reports
4. **Reverse engineer** — if total revenue is X, does per-user revenue times user count equal X?
5. **Boundary checks** — filter to a single day/user/category. Are micro-results sensible?

### Red Flags That Warrant Investigation

- Any metric changed >50% period-over-period without obvious cause
- Counts or sums that are exact round numbers (filter or default value issue)
- Rates exactly at 0% or 100% (incomplete data)
- Results that perfectly confirm the hypothesis (reality is messier)
- Identical values across time periods or segments (query ignoring a dimension)

## Documentation Template

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

## Rules

- Run the full checklist before sharing — skipping steps is how errors ship
- Check row counts before and after every join
- Never average pre-aggregated averages — go back to raw data
- Always ask "who is missing from this dataset?"
- Cross-reference key numbers against at least one other source
- State assumptions explicitly — unstated assumptions become invisible errors
- Round appropriately — false precision erodes trust
