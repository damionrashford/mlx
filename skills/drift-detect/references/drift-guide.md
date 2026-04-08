# Drift Detection Guide

## Population Stability Index (PSI)

Measures distribution shift for continuous features.

```python
import numpy as np

def compute_psi(reference, current, bins=10):
    """PSI > 0.2: significant drift. 0.1-0.2: moderate. < 0.1: stable."""
    min_val = min(reference.min(), current.min())
    max_val = max(reference.max(), current.max())
    breakpoints = np.linspace(min_val, max_val, bins + 1)

    ref_pct = np.histogram(reference, bins=breakpoints)[0] / len(reference)
    cur_pct = np.histogram(current, bins=breakpoints)[0] / len(current)

    # Replace zeros to avoid log(0)
    ref_pct = np.where(ref_pct == 0, 1e-4, ref_pct)
    cur_pct = np.where(cur_pct == 0, 1e-4, cur_pct)

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return psi

# Interpretation
# PSI < 0.1:  no significant change
# PSI 0.1-0.2: moderate change — monitor
# PSI > 0.2:  significant change — investigate
```

## KS-test (Kolmogorov-Smirnov)

Tests if two samples come from the same distribution.

```python
from scipy.stats import ks_2samp

stat, p_value = ks_2samp(reference_col, current_col)
if p_value < 0.05:
    print(f"Drift detected: KS stat={stat:.4f}, p={p_value:.4f}")
```

KS statistic: maximum difference between CDFs. P-value < 0.05 = reject null hypothesis
(same distribution) at 95% confidence.

## Chi-squared (categorical features)

```python
from scipy.stats import chi2_contingency
import pandas as pd

def chi2_drift(reference_cat, current_cat):
    ref_counts = reference_cat.value_counts()
    cur_counts = current_cat.value_counts()
    all_cats = ref_counts.index.union(cur_counts.index)
    contingency = pd.DataFrame({
        "reference": ref_counts.reindex(all_cats, fill_value=0),
        "current": cur_counts.reindex(all_cats, fill_value=0),
    })
    stat, p_value, _, _ = chi2_contingency(contingency.T)
    return stat, p_value
```

## Wasserstein distance

Earth mover's distance — how much "work" to transform one distribution to another.

```python
from scipy.stats import wasserstein_distance

dist = wasserstein_distance(reference_col, current_col)
# Higher = more drift. Threshold depends on feature scale.
```

## evidently: automated drift reports

```python
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset

# Data drift
report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=ref_df, current_data=cur_df)
report.save_html("drift_report.html")

# Target drift (when you have labels)
target_report = Report(metrics=[TargetDriftPreset()])
target_report.run(reference_data=ref_df, current_data=cur_df, column_mapping=column_mapping)
```

## nannyml: no-label monitoring

CBPE (Confidence-Based Performance Estimation) estimates model performance
without ground truth labels.

```python
import nannyml

estimator = nannyml.CBPE(
    y_pred_proba="pred_proba",
    y_pred="pred_label",
    y_true="label",        # only needed for reference set
    metrics=["roc_auc", "f1"],
    chunk_size=500,        # samples per monitoring window
    problem_type="binary_classification",
)
estimator.fit(reference_df)
results = estimator.estimate(current_df)
results.plot()
```

## Concept drift detectors (river library)

```python
from river import drift

# DDM (Drift Detection Method) — based on error rate
ddm = drift.DDM()
for y_true, y_pred in zip(y_trues, y_preds):
    error = int(y_true != y_pred)
    ddm.update(error)
    if ddm.drift_detected:
        print("Drift detected — retrain model")

# ADWIN (Adaptive Windowing) — more sensitive
adwin = drift.ADWIN()
for value in stream:
    adwin.update(value)
    if adwin.drift_detected:
        print(f"Drift at step {adwin.n_samples}")
```

## Alert thresholds and escalation

```python
THRESHOLDS = {
    "psi_warn": 0.1,
    "psi_alert": 0.2,
    "ks_pvalue": 0.05,
    "chi2_pvalue": 0.05,
}

def check_drift(ref_df, cur_df, numeric_cols, categorical_cols):
    issues = []
    for col in numeric_cols:
        psi = compute_psi(ref_df[col], cur_df[col])
        if psi > THRESHOLDS["psi_alert"]:
            issues.append(f"ALERT: {col} PSI={psi:.3f} (significant drift)")
        elif psi > THRESHOLDS["psi_warn"]:
            issues.append(f"WARN: {col} PSI={psi:.3f} (moderate drift)")
    for col in categorical_cols:
        _, p = chi2_drift(ref_df[col], cur_df[col])
        if p < THRESHOLDS["chi2_pvalue"]:
            issues.append(f"ALERT: {col} chi2 p={p:.4f} (categorical drift)")
    return issues
```

## Inference pipeline integration

```python
# Add to inference pipeline
class DriftMonitoredPredictor:
    def __init__(self, model, reference_data, alert_threshold=0.2):
        self.model = model
        self.reference = reference_data
        self.buffer = []
        self.threshold = alert_threshold

    def predict(self, X):
        self.buffer.append(X)
        if len(self.buffer) >= 1000:
            current = pd.concat(self.buffer)
            for col in current.columns:
                psi = compute_psi(self.reference[col], current[col])
                if psi > self.threshold:
                    self.alert(col, psi)
            self.buffer = []
        return self.model.predict(X)
```
