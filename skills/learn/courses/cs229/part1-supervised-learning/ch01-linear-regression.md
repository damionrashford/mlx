# Chapter 1: Linear Regression

## Introduction

Linear regression is the starting point for understanding machine learning. It answers: **How do we predict a continuous output from input features?**

**Example**: Predicting house prices from living area and number of bedrooms.

| Living area (ft²) | #Bedrooms | Price ($1000s) |
|-------------------|-----------|----------------|
| 2104 | 3 | 400 |
| 1600 | 3 | 330 |
| 2400 | 3 | 369 |
| 1416 | 2 | 232 |
| 3000 | 4 | 540 |

## The Hypothesis

We model y as a linear function of x:

```
h_θ(x) = θ₀ + θ₁x₁ + θ₂x₂ = θᵀx
```

Where:
- θᵢ are the **parameters** (weights)
- x₀ = 1 (intercept term)
- θᵀx is the inner product

## 1.1 LMS Algorithm (Least Mean Squares)

### The Cost Function

```
J(θ) = (1/2) Σᵢⁿ (h_θ(x^(i)) - y^(i))²
```

This is the **least-squares cost function** - it measures how far our predictions are from the actual values.

### Gradient Descent

To minimize J(θ), we use gradient descent:

```
θⱼ := θⱼ - α · ∂J(θ)/∂θⱼ
```

For a single training example:
```
∂J(θ)/∂θⱼ = (h_θ(x) - y) · xⱼ
```

This gives the **LMS update rule**:
```
θⱼ := θⱼ + α(y^(i) - h_θ(x^(i))) · xⱼ^(i)
```

**Intuition**: The update is proportional to the error (y - h(x)). Large errors → large updates.

### Batch vs Stochastic Gradient Descent

**Batch Gradient Descent**:
```
Repeat until convergence:
    θⱼ := θⱼ + α Σᵢⁿ (y^(i) - h_θ(x^(i))) · xⱼ^(i)
```
- Uses ALL examples per update
- Guaranteed convergence to global minimum (J is convex)

**Stochastic Gradient Descent (SGD)**:
```
Loop:
    for i = 1 to n:
        θⱼ := θⱼ + α(y^(i) - h_θ(x^(i))) · xⱼ^(i)
```
- Uses ONE example per update
- Faster progress, especially for large datasets
- May oscillate around minimum

## 1.2 The Normal Equations

Instead of iterative gradient descent, we can solve directly.

### Matrix Calculus Notation

For f: ℝⁿˣᵈ → ℝ, the gradient is:
```
∇_A f(A) = [∂f/∂A₁₁  ...  ∂f/∂A₁ᵈ]
           [  ⋮      ⋱      ⋮    ]
           [∂f/∂Aₙ₁  ...  ∂f/∂Aₙᵈ]
```

### Derivation

Define the design matrix X and target vector ỹ:
```
X = [— (x^(1))ᵀ —]     ỹ = [y^(1)]
    [— (x^(2))ᵀ —]         [y^(2)]
    [     ⋮      ]         [  ⋮  ]
    [— (x^(n))ᵀ —]         [y^(n)]
```

The cost function in matrix form:
```
J(θ) = (1/2)(Xθ - ỹ)ᵀ(Xθ - ỹ)
```

Taking the gradient and setting to zero:
```
∇_θ J(θ) = XᵀXθ - Xᵀỹ = 0
```

### The Normal Equations Solution

```
θ = (XᵀX)⁻¹Xᵀỹ
```

**This is a closed-form solution!** No iteration needed.

**Caveat**: XᵀX must be invertible (full rank).

## 1.3 Probabilistic Interpretation

**Why squared error?** Here's a principled justification.

### Assumptions
1. Target and input are related: y^(i) = θᵀx^(i) + ε^(i)
2. Error terms ε^(i) are IID Gaussian: ε^(i) ~ N(0, σ²)

### Likelihood

The probability of y given x:
```
p(y^(i)|x^(i); θ) = (1/√(2πσ)) exp(-(y^(i) - θᵀx^(i))²/(2σ²))
```

The likelihood of the parameters:
```
L(θ) = Πᵢⁿ p(y^(i)|x^(i); θ)
```

### Maximum Likelihood Estimation

Taking the log likelihood:
```
ℓ(θ) = n·log(1/√(2πσ)) - (1/2σ²) Σᵢⁿ (y^(i) - θᵀx^(i))²
```

Maximizing ℓ(θ) is equivalent to minimizing:
```
(1/2) Σᵢⁿ (y^(i) - θᵀx^(i))² = J(θ)
```

**Key Insight**: Least squares regression = MLE under Gaussian noise!

## 1.4 Locally Weighted Linear Regression (Optional)

### The Problem with Linear Models

Linear models can **underfit** (too simple) or **overfit** (too complex):

- **Underfitting**: y = θ₀ + θ₁x doesn't capture curvature
- **Overfitting**: y = Σⱼ⁵ θⱼxʲ passes through all points but generalizes poorly

### Locally Weighted Linear Regression (LWR)

Instead of fitting one global θ, fit a local θ for each query point.

**Standard Linear Regression**:
1. Fit θ to minimize Σᵢ (y^(i) - θᵀx^(i))²
2. Output θᵀx

**LWR**:
1. Fit θ to minimize Σᵢ w^(i)(y^(i) - θᵀx^(i))²
2. Output θᵀx

Where weights are:
```
w^(i) = exp(-(x^(i) - x)² / (2τ²))
```

- Points close to query x get high weight
- Points far from x get low weight
- τ (bandwidth) controls the decay rate

### Parametric vs Non-parametric

| Type | Memory | Example |
|------|--------|---------|
| **Parametric** | Store θ only | Linear Regression |
| **Non-parametric** | Store all training data | LWR |

LWR is **non-parametric**: the hypothesis grows with the training set size.

## Key Takeaways

1. **Linear regression** is the foundation - understand it deeply
2. **Gradient descent** is the workhorse optimization algorithm
3. **Normal equations** give a closed-form solution when feasible
4. **MLE interpretation** justifies squared error under Gaussian noise
5. **LWR** shows the parametric vs non-parametric tradeoff

## Practical Notes for Production

- **Feature scaling** is crucial for gradient descent convergence
- **Regularization** (covered later) prevents overfitting
- **Normal equations** are O(d³) - use gradient descent for large d
- **SGD** is preferred for large datasets

