# Part III: Generalization and Regularization

## Overview

Training error is easy to minimize. The real challenge is **generalization** - performing well on unseen data. This part develops the theory and practice of building models that generalize.

## Chapters in This Part

- **[Chapter 8: Generalization](ch08-generalization.md)** - Bias-variance tradeoff, learning theory
- **[Chapter 9: Regularization & Model Selection](ch09-regularization.md)** - Preventing overfitting

## The Fundamental Question

> Why should doing well on training data tell us anything about unseen data?

## Core Concepts

### Overfitting vs Underfitting

| Issue | Training Error | Test Error | Solution |
|-------|---------------|------------|----------|
| **Underfitting** | High | High | More complex model |
| **Good fit** | Low | Low | - |
| **Overfitting** | Low | High | Regularization, more data |

### The Bias-Variance Tradeoff

```
Expected Test Error = Bias² + Variance + Irreducible Noise
```

- **Bias**: Error from wrong assumptions (model too simple)
- **Variance**: Error from sensitivity to training data (model too complex)
- **Tradeoff**: Reducing one often increases the other

### Key Quantities

| Quantity | Definition |
|----------|------------|
| **Training error** ε̂(h) | Error on training set |
| **Generalization error** ε(h) | Expected error on new data |
| **Hypothesis class** H | Set of all models considered |
| **VC dimension** | Complexity measure for H |

## Learning Theory Results

### Finite Hypothesis Class

With probability ≥ 1-δ:
```
ε(ĥ) ≤ min_{h∈H} ε(h) + 2√((1/2n) log(2k/δ))
```

Where k = |H| and n = training set size.

### Sample Complexity

To achieve error within γ of optimal with probability 1-δ:
```
n ≥ O((1/γ²) log(k/δ))
```

Need only **logarithmic** samples in hypothesis class size!

## Regularization Techniques

| Technique | How It Works |
|-----------|--------------|
| **L2 (Ridge)** | Penalize large weights |
| **L1 (Lasso)** | Encourage sparse weights |
| **Dropout** | Randomly drop units during training |
| **Early stopping** | Stop before training converges |
| **Data augmentation** | Artificially expand training set |

## Model Selection

**Cross-validation**: Split data into folds, train on k-1, validate on 1, rotate.

```
1. For each hyperparameter setting:
   a. Run k-fold cross-validation
   b. Compute average validation error
2. Select setting with lowest average error
3. Retrain on full training set
```

## Why This Matters

1. **Debugging**: Diagnose if model is under/overfitting
2. **Model selection**: Choose appropriate complexity
3. **Confidence**: Understand when predictions are reliable
4. **Efficiency**: Know how much data you need
