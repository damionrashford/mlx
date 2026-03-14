# ML Monitoring and Drift Detection

## Learning Objectives

By the end of this lesson, you will:

**Understand:**
- The four types of drift (data drift, concept drift, feature drift, label drift) and why ML models degrade over time
- Standard drift detection methods: Kolmogorov-Smirnov test, Population Stability Index (PSI), and sliding-window performance monitoring
- Feedback loops in production ML systems and why they are the most insidious failure mode

**Apply:**
- Implement a drift detection pipeline using KS tests and PSI, with appropriate severity thresholds and alerting
- Design a monitoring dashboard covering model performance, prediction quality, data quality, infrastructure, and business metrics

**Analyze:**
- Evaluate retraining strategies (scheduled, triggered, continuous) for a given use case based on the rate of change in the underlying domain and the availability of ground-truth labels

---

## Prerequisites

Before starting this lesson, you should be comfortable with:
- **Model serving** -- serving patterns, deployment strategies, and how predictions are delivered in production (see [Model Serving](./model-serving/COURSE.md))
- **Evaluation metrics** -- precision, recall, AUC, and how to interpret performance changes (see [Evaluation Metrics](../04-classical-ml/evaluation-metrics/COURSE.md))
- **Probability and statistics** -- distributions, hypothesis testing, p-values, and the KS test (see [Probability and Statistics](../01-foundations/probability-statistics/COURSE.md))

---

## Why This Matters

A model that was 95% accurate at deployment will degrade. The question is not if, but when and how fast. Unlike traditional software where a bug is a bug forever, ML model failures are gradual, silent, and context-dependent. The model still returns predictions — they're just wrong.

In production e-commerce systems, product catalogs change, shopping patterns shift seasonally, new merchant categories emerge, and fraud patterns evolve adversarially. If you deploy and forget, you will serve bad predictions within weeks.

---

## Why ML Models Degrade

### The Core Problem

A model learns a mapping from inputs to outputs based on historical data. When the real world changes, that mapping becomes stale.

```
Training time:   Input distribution P(X) → Learned mapping f(X) → Accurate predictions
Production:      Input distribution P'(X) → Same mapping f(X)  → Degraded predictions
```

### Categories of Change

| Type | What Changes | Example | Speed |
|------|-------------|---------|-------|
| Data drift | Input distribution | New product categories appear | Gradual |
| Concept drift | Input-output relationship | What constitutes fraud changes | Can be sudden |
| Feature drift | Individual feature distributions | A feature pipeline breaks | Sudden |
| Label drift | Distribution of outcomes | Fraud rate doubles due to attack | Variable |
| Upstream data change | Data schema or semantics | A column gets renamed | Sudden |

---

## Data Drift

Data drift occurs when the distribution of input features changes from what the model was trained on.

### Why It Happens

- **Seasonality**: Black Friday shopping patterns differ from January
- **User base changes**: the platform onboards merchants in new countries
- **Product evolution**: new product categories that didn't exist during training
- **External events**: pandemic shifts shopping from in-store to online

### Example

```
Training data (2023):
  avg_order_value:  mean=$45, std=$30
  mobile_pct:       mean=0.55

Production data (2024):
  avg_order_value:  mean=$62, std=$45    ← drift!
  mobile_pct:       mean=0.72            ← drift!
```

The model learned patterns at $45 average orders. At $62, its calibration is off.

### Detection: Kolmogorov-Smirnov Test

The KS test measures the maximum distance between two cumulative distribution functions.

```python
from scipy.stats import ks_2samp
import numpy as np

def detect_data_drift(reference_data, production_data, features, alpha=0.05):
    """
    Compare feature distributions between reference (training) and production data.
    Returns features with statistically significant drift.
    """
    drift_report = []

    for feature in features:
        ref_values = reference_data[feature].dropna()
        prod_values = production_data[feature].dropna()

        statistic, p_value = ks_2samp(ref_values, prod_values)

        drift_report.append({
            'feature': feature,
            'ks_statistic': round(statistic, 4),
            'p_value': round(p_value, 6),
            'drift_detected': p_value < alpha,
            'severity': 'HIGH' if statistic > 0.2 else 'MEDIUM' if statistic > 0.1 else 'LOW',
        })

    return sorted(drift_report, key=lambda x: x['ks_statistic'], reverse=True)

# Usage
drift = detect_data_drift(
    reference_data=training_df,
    production_data=last_week_df,
    features=['avg_order_value', 'order_count_30d', 'mobile_pct', 'product_count'],
)
for d in drift:
    if d['drift_detected']:
        print(f"DRIFT: {d['feature']} — KS={d['ks_statistic']}, severity={d['severity']}")
```

### Detection: Population Stability Index (PSI)

PSI measures how much a distribution has shifted. Widely used in credit risk and fraud.

```python
import numpy as np

def calculate_psi(reference, production, bins=10):
    """
    Population Stability Index.
    PSI < 0.1: no significant change
    PSI 0.1-0.25: moderate change, investigate
    PSI > 0.25: significant change, action needed
    """
    # Create bins from reference distribution
    breakpoints = np.percentile(reference, np.linspace(0, 100, bins + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    # Count proportions in each bin
    ref_counts = np.histogram(reference, bins=breakpoints)[0] / len(reference)
    prod_counts = np.histogram(production, bins=breakpoints)[0] / len(production)

    # Avoid division by zero
    ref_counts = np.clip(ref_counts, 0.001, None)
    prod_counts = np.clip(prod_counts, 0.001, None)

    # PSI formula
    psi = np.sum((prod_counts - ref_counts) * np.log(prod_counts / ref_counts))

    return round(psi, 4)

# Interpretation
psi = calculate_psi(training_df['avg_order_value'], production_df['avg_order_value'])
if psi > 0.25:
    print(f"ALERT: Significant drift detected (PSI={psi})")
elif psi > 0.1:
    print(f"WARNING: Moderate drift detected (PSI={psi})")
else:
    print(f"OK: No significant drift (PSI={psi})")
```

---

### Check Your Understanding: Data Drift Detection

**1. What does a KS statistic of 0.25 with a p-value of 0.001 tell you about a feature, and what action should you take?**

<details>
<summary>Answer</summary>

A KS statistic of 0.25 means the maximum distance between the cumulative distribution functions of the reference (training) and production data for that feature is 0.25 -- a substantial shift. The p-value of 0.001 means there is only a 0.1% chance this difference occurred by random sampling, so the drift is statistically significant. This would be classified as "HIGH" severity (KS > 0.2). Action: investigate the root cause (data pipeline change? real-world shift?), check if it correlates with model performance degradation, and consider retraining on recent data if multiple features show similar drift.
</details>

**2. What are the PSI thresholds and what does each range indicate?**

<details>
<summary>Answer</summary>

PSI < 0.1: no significant change -- the distribution is stable, no action needed. PSI 0.1-0.25: moderate change -- investigate the cause but do not necessarily retrain. It may be expected seasonality or a minor shift. PSI > 0.25: significant change -- action is needed. The feature distribution has shifted meaningfully from the reference, which likely affects model performance. Investigate and likely trigger retraining.
</details>

**3. Why might data drift exist without model performance degradation, and vice versa?**

<details>
<summary>Answer</summary>

Data drift without performance degradation: a feature may shift, but if the model does not rely heavily on it (low feature importance) or the shift is in a region of feature space where the decision boundary is robust, predictions remain accurate. Performance degradation without data drift: concept drift -- the relationship between inputs and outputs changes even though input distributions remain stable. For example, the same transaction patterns may shift from legitimate to fraudulent due to new fraud tactics, but the feature distributions look identical.
</details>

---

## Concept Drift

Concept drift occurs when the relationship between inputs and outputs changes — even if input distributions stay the same.

### Why It Happens

- **Fraud evolution**: Fraudsters adapt tactics when they learn the model catches them
- **Market shifts**: what predicts churn in a growing market differs from a saturated one
- **Policy changes**: the platform changes refund policies, altering return patterns
- **Cultural shifts**: pandemic changed which product categories sell well

### Example

```
Training period:
  High velocity + new account → 90% chance of fraud

Production (6 months later):
  Fraudsters now use aged accounts, so:
  High velocity + new account → only 40% chance of fraud
  Model still flags new accounts → false positives spike
```

### Detection

Concept drift is harder to detect than data drift because you need outcome labels (which are often delayed).

```python
def detect_concept_drift(predictions_with_labels, window_size=7, threshold=0.05):
    """
    Monitor model accuracy over time using a sliding window.
    Requires ground-truth labels (which may be delayed).
    """
    results = []

    for window_end in range(window_size, len(predictions_with_labels)):
        window = predictions_with_labels[window_end - window_size : window_end]

        accuracy = (window['prediction'] == window['actual']).mean()
        precision = (window[(window['prediction'] == 1)]['actual'] == 1).mean()
        recall = (window[(window['actual'] == 1)]['prediction'] == 1).mean()

        results.append({
            'date': window['date'].iloc[-1],
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
        })

    df = pd.DataFrame(results)

    # Detect significant drops
    baseline_accuracy = df['accuracy'].iloc[:window_size].mean()
    recent_accuracy = df['accuracy'].iloc[-window_size:].mean()

    if baseline_accuracy - recent_accuracy > threshold:
        return {'drift_detected': True, 'drop': baseline_accuracy - recent_accuracy}

    return {'drift_detected': False}
```

### Types of Concept Drift

```
Sudden drift:      ──────┐
                         └──────     (policy change, external event)

Gradual drift:     ────────╲
                            ╲────   (slow behavioral change)

Recurring drift:   ──╱╲──╱╲──╱╲──  (seasonal patterns)

Incremental drift: ────────────╲──  (very slow change over months)
```

---

## Feature Drift

Feature drift is when individual feature distributions change, often due to upstream pipeline issues.

### Common Causes

| Cause | Example | Detection |
|-------|---------|-----------|
| Pipeline bug | Feature computation changes silently | Distribution comparison |
| Schema change | Column renamed, type changed | Schema validation |
| Data source outage | Third-party API returns nulls | Null rate monitoring |
| Upstream model change | An upstream model that produces a feature gets updated | Output distribution check |
| Logging change | Client-side event format changes | Schema + volume checks |

### Example: Silent Pipeline Failure

```
Day 1-100: Feature "days_since_last_order" computed correctly
Day 101:   Pipeline bug — feature returns 0 for all merchants
Day 101+:  Model receives all zeros, makes bad predictions
           No error, no crash, just wrong values
```

### Detection

```python
def monitor_feature_health(feature_name, current_values, reference_stats):
    """
    Check a single feature against reference statistics.
    Returns alerts if anomalous.
    """
    alerts = []

    # Null rate check
    null_rate = current_values.isnull().mean()
    if null_rate > reference_stats['max_null_rate'] * 1.5:
        alerts.append(f"Null rate spike: {null_rate:.2%} vs expected < {reference_stats['max_null_rate']:.2%}")

    # Range check
    current_min = current_values.min()
    current_max = current_values.max()
    if current_min < reference_stats['expected_min']:
        alerts.append(f"Below expected range: min={current_min} (expected >= {reference_stats['expected_min']})")
    if current_max > reference_stats['expected_max']:
        alerts.append(f"Above expected range: max={current_max} (expected <= {reference_stats['expected_max']})")

    # Mean shift check
    current_mean = current_values.mean()
    expected_mean = reference_stats['mean']
    expected_std = reference_stats['std']
    z_score = abs(current_mean - expected_mean) / expected_std
    if z_score > 3:
        alerts.append(f"Mean shift: current={current_mean:.2f}, expected={expected_mean:.2f} (z={z_score:.1f})")

    # Constant value check (feature stuck)
    if current_values.nunique() <= 1:
        alerts.append(f"Feature appears stuck: only {current_values.nunique()} unique value(s)")

    return alerts
```

---

### Check Your Understanding: Concept Drift and Feature Drift

**1. Why is concept drift harder to detect than data drift?**

<details>
<summary>Answer</summary>

Data drift can be detected immediately by comparing input feature distributions -- you do not need outcome labels. Concept drift, however, is a change in the relationship between inputs and outputs (P(Y|X) changes). To detect it, you need ground-truth labels to measure whether the model's predictions are becoming less accurate. In many production systems (fraud, churn), labels are delayed by days, weeks, or months, so concept drift may go undetected for a long time. This is why monitoring prediction distributions (available immediately) is important as a proxy signal.
</details>

**2. In the feature drift example, the "days_since_last_order" feature returns 0 for all merchants due to a pipeline bug. Why would a standard mean/std check catch this but a null rate check would not?**

<details>
<summary>Answer</summary>

A null rate check only detects missing values (NaN/NULL). The pipeline bug returns 0 -- a valid numeric value -- so the null rate is unchanged. However, a mean shift check would detect this immediately because the mean drops dramatically from its expected value to near zero, and a constant-value check (number of unique values = 1) would also flag it. A distribution check (KS test or PSI) would also catch this as a massive distributional shift. This illustrates why multiple types of checks are needed -- no single check catches all failure modes.
</details>

**3. What is the difference between sudden concept drift and gradual concept drift, and how does each affect your retraining strategy?**

<details>
<summary>Answer</summary>

Sudden concept drift happens abruptly (e.g., a policy change or external event), causing immediate performance degradation. It requires triggered retraining as soon as the drop is detected. Gradual concept drift happens slowly over weeks or months (e.g., evolving user behavior), making it harder to detect with short monitoring windows. Scheduled retraining (weekly or monthly) with a sliding training window handles gradual drift well. For sudden drift, you need sensitive alerting and fast retraining pipelines; for gradual drift, regular scheduled retraining and trend monitoring are more appropriate.
</details>

---

## Building a Monitoring Dashboard

### What to Monitor

```
┌──────────────────────────────────────────────────────────────────┐
│                    ML Monitoring Dashboard                       │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  MODEL PERFORMANCE (requires labels, may be delayed)             │
│  ├── Accuracy / AUC / Precision / Recall (rolling window)       │
│  ├── Performance by segment (merchant size, country, category)   │
│  └── Comparison to baseline model                               │
│                                                                  │
│  PREDICTION QUALITY (available immediately)                      │
│  ├── Prediction distribution (histogram over time)              │
│  ├── Prediction confidence distribution                         │
│  ├── Prediction rate (% positive predictions)                   │
│  └── Prediction latency (p50, p95, p99)                        │
│                                                                  │
│  DATA QUALITY (available immediately)                            │
│  ├── Feature distributions vs reference                         │
│  ├── Null rates per feature                                     │
│  ├── Feature freshness (when was each feature last updated?)    │
│  └── Input volume (requests per minute)                         │
│                                                                  │
│  INFRASTRUCTURE                                                  │
│  ├── Serving latency (p50, p95, p99)                           │
│  ├── Error rate (5xx, timeouts)                                │
│  ├── CPU/Memory/GPU utilization                                │
│  └── Cache hit rate                                             │
│                                                                  │
│  BUSINESS METRICS (the ultimate truth)                           │
│  ├── Conversion rate (for recommendations)                     │
│  ├── Chargeback rate (for fraud)                               │
│  ├── Click-through rate (for search ranking)                   │
│  └── Revenue impact                                             │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### Tools for Monitoring

| Tool | Type | Strengths |
|------|------|-----------|
| Evidently AI | Open source ML monitoring | Drift detection, data quality reports |
| Whylabs / whylogs | ML observability | Lightweight profiling, anomaly detection |
| Grafana + Prometheus | Infrastructure monitoring | Custom dashboards, alerting |
| Datadog | Full-stack monitoring | ML model monitoring add-on |
| Arize AI | ML observability platform | Embedding drift, performance tracking |
| NannyML | Concept drift detection | Works without ground truth labels |

---

## Alerting: When to Page, When to Auto-Fix

### Alert Severity Levels

```
CRITICAL (page on-call immediately):
  - Model serving is returning errors (> 1% error rate)
  - Prediction distribution collapsed (all predictions same value)
  - Feature store is returning stale data (> 24 hours old)
  - Latency exceeds SLA by 2x

HIGH (notify team within 1 hour):
  - Data drift detected on multiple features (PSI > 0.25)
  - Model accuracy dropped > 5% from baseline
  - Null rate spike on important features

MEDIUM (investigate within 1 day):
  - Data drift on single feature (PSI > 0.1)
  - Model accuracy dropped 2-5% from baseline
  - Prediction distribution shifting gradually

LOW (review in weekly meeting):
  - Minor feature distribution changes
  - Model accuracy within expected seasonal range
  - Infrastructure metrics within normal bounds
```

### Alert Design Principles

1. **Alert on symptoms, not just causes.** "Conversion rate dropped 3%" is more actionable than "feature X has KS > 0.1."
2. **Avoid alert fatigue.** If you alert on everything, people ignore everything.
3. **Include context in alerts.** "Model accuracy dropped 5% — top contributing features: avg_order_value (PSI=0.3), mobile_pct (PSI=0.2)."
4. **Auto-remediate when safe.** If a feature is stale, fall back to a default. If a model is degraded, fall back to a simpler baseline.

---

## Retraining Strategies

### Scheduled Retraining

Retrain on a fixed schedule regardless of performance.

```
Schedule: Every Sunday at 2am
Pipeline: Extract last 90 days → Train → Evaluate → If better than prod → Deploy
```

**Pros:** Simple, predictable
**Cons:** May retrain unnecessarily, may not retrain fast enough when drift is rapid

### Triggered Retraining

Retrain when drift is detected or performance drops below threshold.

```
Monitor detects: PSI > 0.25 on 3+ features
  → Trigger retraining pipeline
  → Train on latest data
  → Evaluate against production model
  → If improvement > 1%: deploy via canary
  → If no improvement: alert team for investigation
```

**Pros:** Efficient, responsive to changes
**Cons:** Complex triggering logic, risk of too-frequent retraining

### Continuous Training

Incrementally update the model as new data arrives.

```
New labeled data arrives (streaming)
  → Incremental model update (online learning)
  → Automatic evaluation
  → Automatic deployment if passing
```

**Pros:** Always up-to-date, no retraining delay
**Cons:** Complex infrastructure, risk of catastrophic forgetting, hard to debug

### Decision Framework

```
┌────────────────────────────────────────────────────────────────┐
│ How fast does the world change for your use case?              │
│                                                                │
│ Slowly (months)          → Scheduled (monthly/quarterly)       │
│ Moderately (weeks)       → Scheduled (weekly) + triggered      │
│ Fast (days)              → Triggered + short retrain windows   │
│ Very fast (hours)        → Continuous / online learning        │
│ Adversarial (fraud)      → Triggered + rules for zero-day     │
└────────────────────────────────────────────────────────────────┘
```

---

### Check Your Understanding: Monitoring, Alerting, and Retraining

**1. Why should you "alert on symptoms, not just causes"?**

<details>
<summary>Answer</summary>

A cause-based alert like "feature X has KS > 0.1" tells you that a feature distribution shifted, but not whether it matters. The feature may have low importance, or the shift may be in a benign direction. A symptom-based alert like "conversion rate dropped 3%" is directly actionable because it tells you something the business cares about has changed. The best monitoring systems combine both: symptom-based alerts for severity, with cause-based diagnostics to help debug once the symptom is identified.
</details>

**2. In the retraining decision framework, why does the "adversarial" use case (fraud) need triggered retraining plus rules, rather than just continuous online learning?**

<details>
<summary>Answer</summary>

In adversarial settings, attackers deliberately change their behavior to evade the model, which means concept drift can be sudden and targeted. Pure online learning risks being manipulated -- if fraudsters flood the system with a new pattern, online learning may adapt in the wrong direction (treating the new fraud pattern as normal). Triggered retraining with human oversight provides a checkpoint before adapting. Rules provide zero-day defense against known patterns that emerge faster than the model can retrain on labeled data (since fraud labels are often delayed by 30-90 days).
</details>

---

## Feedback Loops

### Positive Feedback Loop (Dangerous)

A model's predictions influence the data it will be trained on.

```
Recommendation model recommends Product A
  → Users click Product A (because it's shown)
  → Training data shows Product A is popular
  → Model recommends Product A even more
  → Product B never gets shown, never gets clicked
  → Model thinks Product B is bad
  → Filter bubble. Catalog coverage collapses.
```

**Mitigation:**
- Exploration: show random items to a small % of users
- Counterfactual logging: record what would have been shown under a random policy
- Diversity constraints in post-processing
- Monitor catalog coverage and recommendation diversity metrics

### Feedback Loop in Fraud Detection

```
Model flags transaction as fraud → Transaction blocked →
  We never learn if it was truly fraud → Label = "fraud" (self-fulfilling) →
  Model reinforces its own biases → False positive spiral
```

**Mitigation:**
- Random hold-out: let a small % of flagged transactions through (costly but necessary for ground truth)
- Human review: get independent labels from manual investigation
- Track chargeback rate independently from model predictions

---

## Production Context: Monitoring Challenges

### Seasonal Patterns

```
             Black Friday /
Normal  →    Cyber Monday     →   Holiday   →   January
─────────   ┌──────────────┐    ──────────    ──────────
            │ 10x volume   │
            │ New patterns  │
            │ Gift buying   │
            └──────────────┘

Challenge: Is this drift or expected seasonality?
Solution: Compare to same period last year, not last week.
         Maintain seasonal baselines for drift detection.
```

### Multi-Tenant Complexity

A large e-commerce platform serves millions of merchants. Each merchant is effectively a different distribution.

```
Challenge: Global model monitors miss merchant-specific drift.
  - Fashion merchant patterns differ from electronics merchants
  - Drift in one vertical may not show up in global aggregates

Solution: Monitor at multiple granularities:
  - Global: overall model health
  - Segment: by merchant vertical, size tier, country
  - Individual: top merchants get dedicated monitoring
```

### New Merchant Categories

```
Challenge: The platform onboards merchants in new verticals continuously.
  Cannabis, NFTs, subscription boxes — patterns not in training data.

Solution:
  - Monitor prediction confidence for underrepresented segments
  - Trigger retraining when new categories reach significant volume
  - Cold-start fallbacks for novel categories
```

---

## Implementing a Complete Monitoring Pipeline

```python
# Complete daily monitoring job — runs via Airflow

def daily_model_monitoring(execution_date: str):
    """Daily monitoring for the merchant churn prediction model."""

    # 1. Load reference stats and production data
    reference = load_training_data_stats()  # Computed during training
    production = load_production_predictions(date=execution_date)

    # 2. Data drift checks
    drift_results = []
    for feature in reference['features']:
        psi = calculate_psi(
            reference['distributions'][feature],
            production[feature].values,
        )
        drift_results.append({'feature': feature, 'psi': psi})

    significant_drift = [d for d in drift_results if d['psi'] > 0.25]

    # 3. Prediction distribution check
    pred_mean = production['prediction'].mean()
    pred_shift = abs(pred_mean - reference['prediction_mean']) / reference['prediction_std']

    # 4. Performance check (if labels available — may be delayed)
    labeled = load_labeled_outcomes(date=execution_date, lookback_days=30)
    if len(labeled) > 0:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(labeled['actual'], labeled['predicted_probability'])
        performance_drop = reference['auc'] - auc
    else:
        performance_drop = None

    # 5. Generate and send alerts
    alerts = []
    if len(significant_drift) >= 3:
        alerts.append({'severity': 'HIGH', 'message': f'{len(significant_drift)} features drifted'})
    if pred_shift > 3.0:
        alerts.append({'severity': 'CRITICAL', 'message': f'Prediction shift: {pred_shift:.1f} std'})
    if performance_drop and performance_drop > 0.05:
        alerts.append({'severity': 'HIGH', 'message': f'AUC drop: {performance_drop:.3f}'})

    for alert in alerts:
        send_alert(alert)

    # 6. Trigger retraining if critical
    if any(a['severity'] == 'CRITICAL' for a in alerts):
        trigger_retraining_pipeline()

    log_monitoring_results(execution_date, drift_results, alerts)
```

---

## Common Pitfalls

**1. Treating all drift as equally urgent.** Not all feature drift affects model performance. A low-importance feature may drift significantly without impacting predictions, while a small shift in a high-importance feature could be critical. Prioritize drift alerts by feature importance and correlate drift detection with actual performance monitoring.

**2. Comparing to last week instead of the same period last year for seasonal businesses.** Detecting "drift" during Black Friday by comparing to a normal November week produces false alarms. Seasonal baselines are essential for e-commerce and any business with cyclical patterns. Maintain reference distributions for each season.

**3. Not monitoring for feedback loops.** Recommendation systems and fraud models are especially susceptible -- the model's predictions influence the data it will be retrained on. Without explicit mitigation (exploration, counterfactual logging, diversity constraints), the model converges on a narrow, biased view of the world and catalog coverage collapses.

**4. Setting drift thresholds too aggressively, causing alert fatigue.** If every minor feature distribution change triggers an alert, the team will learn to ignore all alerts, including critical ones. Start with conservative thresholds (PSI > 0.25 for action, > 0.1 for investigation) and tune based on observed correlation between drift and actual performance degradation.

---

## Hands-On Exercises

### Exercise 1: Build a Drift Detection Pipeline

Using synthetic data:

1. Generate a "reference" dataset with 5 features drawn from known distributions (e.g., normal, uniform, exponential)
2. Generate a "production" dataset where 2 of the 5 features have shifted (changed mean, increased variance, or changed distribution shape)
3. Implement both KS test and PSI-based drift detection
4. Compare the results: do both methods detect the same drifted features? Which is more sensitive to different types of shifts?
5. Implement the severity classification (LOW/MEDIUM/HIGH) and generate a drift report

### Exercise 2: Design a Monitoring Strategy

For a production churn prediction model that serves 1M merchants daily:

1. Define specific monitoring checks at each level (model performance, prediction quality, data quality, infrastructure, business metrics)
2. Set alert thresholds and severity levels for each check, explaining your reasoning
3. Describe your retraining strategy: what triggers retraining, what data window to use, and how to validate the retrained model before deployment
4. Identify potential feedback loops and describe mitigation strategies

---

## Practice Interview Questions

1. "Your recommendation model's click-through rate dropped 10% this week. Walk through your debugging process."
2. "How would you monitor a fraud model where labels are delayed 30-90 days?"
3. "It's November 20th. Should you retrain before Black Friday? Why or why not?"
4. "Describe a feedback loop in a production ML system and how to mitigate it."
5. "What's the difference between data drift and concept drift? Give an e-commerce example of each."

---

## Key Takeaways

1. Models degrade. Plan for it from day one with monitoring infrastructure.
2. Data drift (input changes) is detectable immediately. Concept drift (relationship changes) requires labels.
3. PSI and KS test are the standard drift detection methods. Know both.
4. Monitor at multiple levels: features, predictions, model metrics, business metrics.
5. Alert design matters. Too many alerts = alert fatigue = ignored alerts.
6. Retraining strategy depends on how fast the world changes for your use case.
7. Feedback loops are the most insidious failure mode. Design for exploration and independent measurement.

---

## Summary and What's Next

This lesson covered why ML models degrade, the four types of drift (data, concept, feature, label), detection methods (KS test, PSI, sliding-window performance monitoring), monitoring dashboard design, alerting best practices, retraining strategies (scheduled, triggered, continuous), and feedback loop mitigation. Monitoring is not an afterthought -- it is the system that keeps your ML product working after deployment.

**Where to go from here:**
- **System Design** (./system-design/COURSE.md) -- revisit how monitoring fits into the complete ML system design framework, and practice designing systems with monitoring as a first-class component
- **Data Pipelines** (./data-pipelines/COURSE.md) -- understand how data validation in pipelines serves as the first line of defense against the data quality issues that cause drift
- **Experiment Tracking** (./experiment-tracking/COURSE.md) -- connect monitoring insights back to the experiment lifecycle: when drift triggers retraining, experiment tracking ensures the new model is rigorously evaluated before deployment
