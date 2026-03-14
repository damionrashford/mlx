# Chapter 2: Classification and Logistic Regression

## Introduction

**Classification** is predicting discrete labels instead of continuous values.

**Example**: Spam classification
- Input x^(i): features of an email
- Output y ∈ {0, 1}: spam (1) or not spam (0)

**Terminology**:
- 0 = negative class (-)
- 1 = positive class (+)
- y^(i) = label

## 2.1 Logistic Regression

### Why Not Linear Regression?

Linear regression predicts h_θ(x) ∈ ℝ, but for classification we need h_θ(x) ∈ [0, 1].

### The Sigmoid (Logistic) Function

```
h_θ(x) = g(θᵀx) = 1 / (1 + e^(-θᵀx))
```

Where g(z) = 1/(1 + e^(-z)) is the **sigmoid function**.

**Properties**:
- g(z) → 1 as z → ∞
- g(z) → 0 as z → -∞
- g(0) = 0.5
- 0 < g(z) < 1 for all z
- g'(z) = g(z)(1 - g(z))

**Interpretation**: h_θ(x) = P(y = 1 | x; θ)

### Decision Rule

Predict y = 1 if h_θ(x) ≥ 0.5, equivalently if θᵀx ≥ 0.

### Probabilistic Model

```
P(y = 1 | x; θ) = h_θ(x)
P(y = 0 | x; θ) = 1 - h_θ(x)
```

Compactly: p(y | x; θ) = (h_θ(x))^y · (1 - h_θ(x))^(1-y)

### Maximum Likelihood Estimation

**Log-likelihood**:
```
ℓ(θ) = Σᵢⁿ [y^(i) log h(x^(i)) + (1 - y^(i)) log(1 - h(x^(i)))]
```

This is the **cross-entropy loss** (negated).

### Gradient Ascent

To maximize ℓ(θ):
```
θⱼ := θⱼ + α · ∂ℓ(θ)/∂θⱼ
```

The gradient:
```
∂ℓ(θ)/∂θⱼ = (y - h_θ(x)) · xⱼ
```

**Stochastic gradient ascent update**:
```
θⱼ := θⱼ + α(y^(i) - h_θ(x^(i))) · xⱼ^(i)
```

**Remarkable observation**: This looks identical to the LMS update rule, but h_θ(x) is now the sigmoid function!

### Alternative View: Logistic Loss

The logistic loss function:
```
ℓ_logistic(t, y) = y·log(1 + e^(-t)) + (1 - y)·log(1 + e^t)
```

Where t = θᵀx (the "logit").

Derivative: ∂ℓ/∂t = 1/(1 + e^(-t)) - y = h_θ(x) - y

## 2.2 The Perceptron Learning Algorithm

**Historical note**: Proposed in 1960s as a model of neural computation.

### Modified Decision Function

```
g(z) = { 1  if z ≥ 0
       { 0  if z < 0
```

### Update Rule

```
θⱼ := θⱼ + α(y^(i) - h_θ(x^(i))) · xⱼ^(i)
```

**Key difference from logistic regression**:
- Output is hard 0/1, not probability
- No meaningful probabilistic interpretation
- Cannot be derived as MLE

## 2.3 Multi-class Classification

Extend to y ∈ {1, 2, ..., k} using **softmax**.

### Softmax Function

```
softmax(t₁, ..., tₖ)ᵢ = exp(tᵢ) / Σⱼ exp(tⱼ)
```

Properties:
- Output sums to 1
- Each output ∈ (0, 1)
- Converts logits to probabilities

### Multi-class Model

Parameters: θ₁, θ₂, ..., θₖ (one per class)

```
P(y = i | x; θ) = exp(θᵢᵀx) / Σⱼ exp(θⱼᵀx)
```

### Loss Function

**Cross-entropy loss**:
```
L = -Σᵢⁿ Σⱼᵏ 1{y^(i) = j} · log P(y = j | x^(i))
```

## 2.4 Newton's Method for Optimization

Alternative to gradient descent with faster convergence.

### The Update Rule

```
θ := θ - H⁻¹∇_θℓ(θ)
```

Where H is the **Hessian matrix**:
```
Hᵢⱼ = ∂²ℓ(θ) / (∂θᵢ∂θⱼ)
```

### Properties

- **Quadratic convergence**: doubles correct digits per iteration
- Requires computing/inverting Hessian: O(d³) per iteration
- For logistic regression: **Newton-Raphson** or **Fisher scoring**

### When to Use

| Method | Cost per Iteration | Convergence |
|--------|-------------------|-------------|
| Gradient Descent | O(nd) | Linear |
| Newton's Method | O(nd² + d³) | Quadratic |

Use Newton's when d is small and n is moderate.

## Key Takeaways

1. **Logistic regression** models P(y|x) using sigmoid
2. **Cross-entropy loss** comes from MLE derivation
3. **Same update rule** as linear regression (different h)
4. **Softmax** extends to multi-class
5. **Newton's method** converges faster when Hessian is tractable

## Practical Notes

- **Feature scaling** helps convergence
- **Regularization** (L1/L2) prevents overfitting
- **Class imbalance**: use weighted loss or resampling
- **Calibration**: logistic regression outputs are well-calibrated probabilities
- Modern frameworks (scikit-learn) use L-BFGS or other quasi-Newton methods

