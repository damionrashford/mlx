---
name: explain
description: >
  Explain model predictions with SHAP, LIME, integrated gradients, and permutation
  importance. Generates summary plots, waterfall charts, and force plots. Use when
  debugging predictions, auditing for bias, or communicating model behavior to
  stakeholders.
allowed-tools: >
  Bash(uv run * scripts/shap_explain.py *)
  Bash, Read, Write, Edit, Glob, Grep
argument-hint: model file and dataset path (e.g. "model.joblib data/test.csv")
model: sonnet
effort: medium
paths: "**/*.joblib,**/*.pkl,**/*.pt,**/*.pth,**/*.onnx"
compatibility: ">=1.0"
metadata:
  category: model-evaluation
  tags: [shap, lime, explainability, interpretability, feature-importance, bias-audit]
  phase: evaluate
---

# Explain Skill

Generate model explanations with SHAP, LIME, integrated gradients, and permutation importance.

## Quick start

```bash
# Auto-detect model type and run SHAP
uv run ${CLAUDE_SKILL_DIR}/scripts/shap_explain.py model.joblib data/test.csv
# Output: explanations/shap_summary.png
```

## Methods by model type

| Model type | Recommended explainer |
|------------|----------------------|
| sklearn tree (RF, XGBoost) | SHAP TreeExplainer |
| sklearn linear | SHAP LinearExplainer |
| PyTorch/TF | SHAP DeepExplainer or captum |
| Any black-box | SHAP KernelExplainer (slow) or LIME |

## Plots

- `shap.summary_plot()` — global feature importance (beeswarm)
- `shap.waterfall_plot()` — single prediction breakdown
- `shap.force_plot()` — interactive prediction visualization
- `shap.dependence_plot()` — feature interaction effects
- `PartialDependenceDisplay` — marginal effect of one feature

## When to use each

- **SHAP**: most accurate, works for any model, gold standard
- **LIME**: fast approximation, good for text and images
- **Integrated gradients**: neural nets only, attribution to input features
- **Permutation importance**: model-agnostic, measures drop in metric when feature shuffled

See `references/explainability-guide.md` for complete documentation and code examples.
