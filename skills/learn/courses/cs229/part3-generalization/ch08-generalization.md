# Chapter 8: Generalization

## Introduction

Machine learning cares about **generalization error**, not training error. This chapter develops the theory of why and when learning algorithms work.

## 8.1 Bias-Variance Tradeoff

### The Three Types of Error

```
Test Error = Bias² + Variance + Irreducible Noise
```

### Definitions (for regression)

**Bias**: How far is the average prediction from the truth?
```
Bias(x) = E_S[ĥ_S(x)] - h*(x)
```

**Variance**: How much do predictions vary across training sets?
```
Var(x) = E_S[(ĥ_S(x) - E_S[ĥ_S(x)])²]
```

Where:
- S = training set (random variable)
- ĥ_S = hypothesis learned from S
- h* = true function

### The Tradeoff

| Model Complexity | Bias | Variance |
|-----------------|------|----------|
| Low (simple) | High | Low |
| High (complex) | Low | High |

**Sweet spot**: Balance bias and variance for minimum total error.

### Visual Intuition

```
Error
  ↑
  │      Total Error
  │        ╱╲
  │       ╱  ╲
  │      ╱    ╲______
  │     ╱      Variance
  │────────────────────
  │ Bias²
  └──────────────────→ Model Complexity
        Optimal
```

### 8.1.1 Mathematical Decomposition

For squared error loss:
```
E[(y - ĥ(x))²] = (E[ĥ(x)] - h*(x))² + E[(ĥ(x) - E[ĥ(x)])²] + σ²
                 \_________Bias²________/   \______Variance______/   \Noise/
```

## 8.2 The Double Descent Phenomenon

### Classical View

Test error decreases, then increases with model complexity (U-shaped curve).

### Modern Discovery

With very high-capacity models (deep networks), test error:
1. Decreases (underfitting region)
2. Peaks at interpolation threshold (n ≈ d)
3. **Decreases again** in overparameterized regime!

```
Error
  ↑
  │     ╱╲
  │    ╱  ╲
  │   ╱    ╲    ←── Peak at n ≈ d
  │  ╱      ╲
  │ ╱        ╲_____
  │╱               ╲____
  └──────────────────────→ # Parameters
    Under-      Over-
  parameterized parameterized
```

### Why Double Descent?

In the overparameterized regime:
- Many solutions interpolate the training data
- Gradient descent/SGD finds a **minimum norm** solution
- This implicit regularization prevents overfitting!

**Key insight**: Overparameterization + implicit regularization can generalize well.

## 8.3 Sample Complexity Bounds

### Setup

- Training set S = {(x^(i), y^(i))} of size n
- Hypothesis class H
- Training error: ε̂(h) = (1/n) Σᵢ 1{h(x^(i)) ≠ y^(i)}
- Generalization error: ε(h) = P(h(x) ≠ y)

### Key Lemmas

**Union Bound**: P(A₁ ∪ ... ∪ Aₖ) ≤ P(A₁) + ... + P(Aₖ)

**Hoeffding Inequality**: For IID Bernoulli(φ) variables:
```
P(|φ̂ - φ| > γ) ≤ 2exp(-2γ²n)
```

### 8.3.2 Finite Hypothesis Class

**Theorem**: For |H| = k, with probability ≥ 1-δ:
```
ε(ĥ) ≤ min_{h∈H} ε(h) + 2√((1/2n) log(2k/δ))
```

**Interpretation**:
- First term: Best possible error in H
- Second term: Generalization gap (decreases with n)

**Sample complexity**: To achieve ε(ĥ) ≤ ε(h*) + 2γ with probability 1-δ:
```
n ≥ (1/2γ²) log(2k/δ)
```

Only **logarithmic** in k!

### 8.3.3 Infinite Hypothesis Class

For hypothesis classes parameterized by d real numbers:
```
n ≥ O(d/γ²)
```

Sample complexity is **linear in dimension d**.

### VC Dimension

**Definition**: VC(H) = size of largest set that H can shatter.

**Shattering**: H shatters a set S if it can realize all 2^|S| labelings.

**Example**: Linear classifiers in 2D have VC dimension 3.
- Can shatter any 3 non-collinear points
- Cannot shatter any 4 points

**Sample Complexity Bound** (VC theory):
```
n ≥ O((VC(H)/γ²) log(1/(δγ)))
```

**Key insight**: Sample complexity depends on VC dimension, not number of hypotheses!

### Important VC Dimension Results

| Hypothesis Class | VC Dimension |
|-----------------|--------------|
| Linear classifiers in ℝᵈ | d + 1 |
| Linear classifiers through origin | d |
| Neural network with W weights | O(W log W) |
| Decision trees with k nodes | O(k) |

## Implications for Practice

### Diagnosing Problems

| Symptom | Diagnosis | Solution |
|---------|-----------|----------|
| High train error, high test error | High bias (underfitting) | More complex model, more features |
| Low train error, high test error | High variance (overfitting) | Regularization, more data |
| Low train error, low test error | Good fit! | - |

### Model Selection Strategy

1. Start simple
2. Increase complexity until training error is low
3. If test error is high, add regularization or get more data
4. Use cross-validation to select hyperparameters

### The Modern Regime

For deep learning:
- Overparameterization is often fine (double descent)
- Implicit regularization from SGD helps
- Still need validation for hyperparameter tuning
- Early stopping provides regularization

## Key Takeaways

1. **Bias-variance tradeoff** is fundamental
2. **Sample complexity** grows logarithmically with |H|
3. **VC dimension** measures hypothesis class complexity
4. **Double descent** explains why overparameterization works
5. **Generalization bounds** guide model selection

## Practical Notes

- **Learning curves**: Plot train/test error vs. training size
- **Validation curves**: Plot train/test error vs. model complexity
- **Cross-validation**: Essential for model selection
- **Ensemble methods**: Reduce variance without increasing bias

