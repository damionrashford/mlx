# Evaluation Metrics Quick Reference

## Classification

| Metric | When to use | Formula |
|--------|-------------|---------|
| Accuracy | Balanced classes | Correct / Total |
| Precision | Cost of false positives high | TP / (TP + FP) |
| Recall | Cost of false negatives high | TP / (TP + FN) |
| F1 | Balance precision and recall | 2 * P * R / (P + R) |
| AUC-ROC | Threshold-independent comparison | Area under ROC curve |
| Log Loss | Probability calibration matters | -mean(y*log(p) + (1-y)*log(1-p)) |

## Regression

| Metric | When to use | Notes |
|--------|-------------|-------|
| RMSE | Penalize large errors | Same units as target |
| MAE | Robust to outliers | Same units as target |
| MAPE | Percentage interpretation needed | Undefined when y=0 |
| R-squared | Proportion of variance explained | 0-1, higher better |

## LLM / AI System Evaluation

| Pattern | When to use |
|---------|-------------|
| LLM-as-judge | Subjective quality (coherence, helpfulness) |
| Reference-based | Ground truth exists (exact match, BLEU, ROUGE) |
| Rubric scoring | Multi-dimensional assessment |
| A/B comparison | Relative ranking of two outputs |
| Bias detection | Fairness across demographic groups |
