---
name: evaluate
description: >
  Systematic evaluation of ML models, experiments, and AI system outputs.
  Multi-dimensional rubrics, LLM-as-judge, bias detection, and structured
  comparison frameworks. Use when the user asks to "evaluate model performance",
  "compare models", "build evaluation rubrics", "assess output quality",
  "detect model bias", or mentions evaluation frameworks, LLM-as-judge,
  model comparison, or quality assessment.
allowed-tools: Bash, Read, Write, Glob, Grep
argument-hint: path to predictions, results.tsv, or model outputs (e.g. "results.tsv" or "predictions.csv")
---

# Model & System Evaluation

Frameworks for systematic evaluation of ML models, experiment results, and AI system outputs. Covers traditional ML metrics, LLM-as-judge patterns, bias detection, and structured comparison.

## When to use

- Comparing multiple trained models beyond val_score
- Evaluating LLM/AI application outputs (RAG quality, prompt effectiveness)
- Building evaluation rubrics for subjective tasks
- Detecting bias in model predictions
- Creating test sets for systematic assessment
- Deciding between experiment results in results.tsv

## Traditional ML evaluation

### Classification metrics

| Metric | When to use | Formula |
|--------|------------|---------|
| Accuracy | Balanced classes | correct / total |
| Precision | False positives costly | TP / (TP + FP) |
| Recall | False negatives costly | TP / (TP + FN) |
| F1 | Imbalanced classes | 2 * P * R / (P + R) |
| AUC-ROC | Ranking / threshold selection | Area under ROC curve |
| Log loss | Probability calibration | -mean(y*log(p)) |

### Regression metrics

| Metric | When to use | Sensitivity |
|--------|------------|-------------|
| RMSE | Penalize large errors | Outlier sensitive |
| MAE | Robust to outliers | Linear penalty |
| R-squared | Variance explained | Scale independent |
| MAPE | Percentage error | Zero sensitive |

### Beyond single metrics

```python
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Multi-dimensional evaluation
report = classification_report(y_true, y_pred, output_dict=True)

# Per-class performance (find weak spots)
cm = confusion_matrix(y_true, y_pred)
per_class_acc = cm.diagonal() / cm.sum(axis=1)

# Calibration (are probabilities reliable?)
from sklearn.calibration import calibration_curve
fraction_pos, mean_predicted = calibration_curve(y_true, y_prob, n_bins=10)

# Fairness across subgroups
for group in ['group_a', 'group_b']:
    mask = df['subgroup'] == group
    group_score = metric(y_true[mask], y_pred[mask])
    print(f"{group}: {group_score:.4f}")
```

## Experiment comparison framework

When comparing experiments in results.tsv:

### Multi-dimensional comparison table

```
| Experiment | val_score | test_score | memory_mb | train_time | complexity |
|------------|-----------|------------|-----------|------------|------------|
| exp000     | 0.8523    | 0.8401     | 4096      | 2m         | low        |
| exp003     | 0.8634    | 0.8521     | 4352      | 5m         | medium     |
| exp007     | 0.8641    | 0.8530     | 8192      | 45m        | high       |
```

### Decision criteria (beyond raw score)

1. **Statistical significance**: Is the improvement real or noise?
   - Bootstrap confidence intervals (1000 resamples)
   - Paired t-test or Wilcoxon for matched samples
   - If delta < 2x std of CV folds, it's likely noise

2. **Efficiency trade-off**: Score per resource unit
   - Score improvement / memory increase
   - Score improvement / training time increase
   - Diminishing returns detection

3. **Robustness**: Does it hold across conditions?
   - Cross-validation variance (low = robust)
   - Performance on minority classes
   - Sensitivity to random seed

4. **Deployability**: Can it run in production?
   - Inference latency (ms per prediction)
   - Memory footprint at serving time
   - Dependency complexity

## LLM-as-judge evaluation

For evaluating LLM application outputs (RAG, chatbots, agents):

### Direct scoring

```python
EVAL_PROMPT = """Rate the following AI response on a scale of 1-5 for each dimension.

**Task**: {task_description}
**Input**: {user_input}
**Response**: {model_output}
**Reference** (if available): {reference}

Rate each dimension:
- Relevance (1-5): Does the response address the question?
- Accuracy (1-5): Are the facts correct?
- Completeness (1-5): Does it cover all aspects?
- Clarity (1-5): Is it well-organized and clear?
- Helpfulness (1-5): Would this actually help the user?

Output as JSON:
{{"relevance": X, "accuracy": X, "completeness": X, "clarity": X, "helpfulness": X, "reasoning": "..."}}
"""
```

### Pairwise comparison

More reliable than direct scoring for subjective quality:

```python
PAIRWISE_PROMPT = """Which response better answers the question?

**Question**: {question}

**Response A**: {response_a}
**Response B**: {response_b}

Choose: A is better / B is better / Tie
Explain your reasoning in 2-3 sentences.
"""
```

**Bias mitigation**: Run twice with A/B swapped. If results disagree, mark as tie.

### Known biases in LLM judges

| Bias | Description | Mitigation |
|------|-------------|------------|
| Position | Prefers first response | Swap positions, average |
| Length | Prefers longer responses | Normalize by length |
| Self-enhancement | Prefers own model's style | Use different judge model |
| Verbosity | Equates detail with quality | Explicit rubric criteria |
| Authority | Prefers confident tone | Focus on factual accuracy |

### Evaluation taxonomy: when to use each approach

| Approach | Best for | Limitation |
|----------|----------|------------|
| **Direct scoring** | Objective criteria — factual accuracy, instruction following, toxicity | Score calibration drift, inconsistent scale interpretation |
| **Pairwise comparison** | Subjective preferences — tone, style, persuasiveness, overall quality | Position bias, length bias |
| **Rubric-based** | Multi-dimensional quality with defined criteria | Requires upfront rubric design |

Research (MT-Bench, Zheng et al. 2023): pairwise comparison achieves higher agreement with human judges than direct scoring for preference-based evaluation. Use direct scoring for objective criteria with clear ground truth; use pairwise for subjective quality comparisons.

### Metric selection by evaluation task

| Task type | Primary metrics | Secondary |
|-----------|----------------|-----------|
| Binary pass/fail | Recall, Precision, F1 | Cohen's κ |
| Ordinal scale (1-5) | Spearman's ρ, Kendall's τ | Cohen's κ (weighted) |
| Pairwise preference | Agreement rate, position consistency | Confidence calibration |
| Multi-label | Macro-F1, Micro-F1 | Per-label precision/recall |

### Production evaluation pipeline

```python
def evaluate_with_bias_mitigation(question, response_a, response_b, judge_model):
    # Forward pass
    result_1 = judge_model(PAIRWISE_PROMPT.format(
        question=question, response_a=response_a, response_b=response_b
    ))
    # Swapped pass (position bias mitigation)
    result_2 = judge_model(PAIRWISE_PROMPT.format(
        question=question, response_a=response_b, response_b=response_a
    ))
    # Normalize result_2 (A/B labels are swapped)
    if result_1 == result_2:
        return result_1  # Consistent — high confidence
    else:
        return "tie"  # Inconsistent — call it a tie

def build_rubric(criterion, weight, levels):
    """
    criterion: "Factual Accuracy"
    weight: 0.40
    levels: {5: "All claims verified", 3: "Mostly accurate", 1: "Multiple errors"}
    """
    return {"criterion": criterion, "weight": weight, "levels": levels}
```

### Performance driver — token budget matters

Research (BrowseComp benchmark) shows three factors explain **95% of agent performance variance**:
- **Token usage: 80% of variance** — more context/turns = better performance
- Number of tool calls: ~10%
- Model choice: ~5%

Implication: **evaluate with realistic token budgets**, not unlimited resources. Upgrading from an older model to Claude Sonnet 4.5 or GPT-5.2 provides larger gains than doubling token budget on the same model.

## Test set design

### Complexity stratification

Build test sets covering multiple difficulty levels:

```python
test_set = {
    "simple": [
        # Single-step, clear answer, common patterns
    ],
    "medium": [
        # Multi-step, some ambiguity, less common patterns
    ],
    "complex": [
        # Many steps, significant ambiguity, edge cases
    ],
    "adversarial": [
        # Deliberately tricky, boundary conditions, known failure modes
    ]
}
```

### Coverage checklist

- [ ] Happy path (typical inputs)
- [ ] Edge cases (boundary values, empty inputs, extreme values)
- [ ] Subgroup fairness (performance across demographics/categories)
- [ ] Out-of-distribution (inputs unlike training data)
- [ ] Adversarial (inputs designed to fool the model)
- [ ] Temporal (does performance change with data from different time periods?)

## Evaluation report format

```
=== Evaluation Report ===
Task: [classification/regression/generation/retrieval]
Models compared: [list]
Test set: [size, composition]

Best model: [name]
Primary metric: [metric] = [value] (95% CI: [low, high])

Multi-dimensional comparison:
| Dimension       | Model A | Model B | Winner |
|-----------------|---------|---------|--------|
| Primary metric  |         |         |        |
| Inference speed |         |         |        |
| Memory usage    |         |         |        |
| Robustness      |         |         |        |
| Fairness        |         |         |        |

Recommendation: [which model and why]
Caveats: [limitations of this evaluation]
```

## Rules

- Never evaluate on training data
- Use multiple metrics, not just one number
- Report confidence intervals, not just point estimates
- Check subgroup performance, not just aggregate
- Test set is touched ONCE for final evaluation
- Document evaluation methodology alongside results
- Acknowledge what the evaluation does NOT cover
