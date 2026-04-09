---
name: drift-detect
description: >
  Detect data drift, concept drift, and model performance degradation in production.
  Uses PSI, KS-test, and chi-squared for statistical drift, plus evidently and
  nannyml for automated reports. Use when monitoring a deployed model or comparing
  training vs production data distributions.
allowed-tools: >
  Bash(uv run * scripts/detect_drift.py *)
  Bash, Read, Write, Edit, Glob, Grep
argument-hint: reference dataset and current dataset paths (e.g. "data/train.csv data/production.csv")
model: sonnet
effort: medium
compatibility: ">=1.0"
metadata:
  category: mlops
  tags: [drift-detection, monitoring, psi, ks-test, evidently, nannyml, production]
  phase: monitor
---

# Drift Detect Skill

Detect data drift, concept drift, and model degradation in production.

## Quick start

```bash
# Run full drift analysis
uv run ${CLAUDE_SKILL_DIR}/scripts/detect_drift.py data/train.csv data/production.csv
# Output: stdout report + drift_report.html
```

## Drift types

| Type | What it measures | Tool |
|------|-----------------|------|
| Data drift | Input feature distribution shift | PSI, KS-test, chi-squared |
| Concept drift | P(y|x) relationship change | DDM, ADWIN |
| Target drift | Label distribution shift | evidently TargetDriftPreset |
| Model degradation | Performance drop in production | nannyml CBPE |

## Thresholds

- **PSI > 0.2**: significant drift — investigate immediately
- **PSI 0.1-0.2**: moderate drift — monitor closely
- **KS p-value < 0.05**: statistically significant distribution shift
- **Chi-squared p-value < 0.05**: categorical feature drift

## Integration

Add drift check to inference pipeline:
```python
from scripts.detect_drift import compute_psi
psi = compute_psi(reference_col, current_col)
if psi > 0.2:
    alert("Significant drift detected")
```

See `references/drift-guide.md` for complete PSI formula, evidently, nannyml, and alert patterns.
