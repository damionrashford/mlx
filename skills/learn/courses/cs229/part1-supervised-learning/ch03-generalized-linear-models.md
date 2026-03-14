# Chapter 3: Generalized Linear Models (GLMs)

## Introduction

Both linear regression (Gaussian) and logistic regression (Bernoulli) are special cases of a broader family: **Generalized Linear Models**.

GLMs provide a unified framework for deriving learning algorithms.

## 3.1 The Exponential Family

A distribution belongs to the **exponential family** if it can be written as:

```
p(y; η) = b(y) · exp(ηᵀT(y) - a(η))
```

Where:
- η = **natural parameter** (canonical parameter)
- T(y) = **sufficient statistic** (often T(y) = y)
- a(η) = **log partition function** (normalizer)
- b(y) = **base measure**

### Examples

**Bernoulli Distribution** (y ∈ {0, 1}):
```
p(y; φ) = φʸ(1-φ)^(1-y)
        = exp(y·log(φ/(1-φ)) + log(1-φ))
```
- η = log(φ/(1-φ)) → φ = 1/(1 + e^(-η)) (sigmoid!)
- T(y) = y
- a(η) = -log(1-φ) = log(1 + e^η)
- b(y) = 1

**Gaussian Distribution** (fixed σ²):
```
p(y; μ) = (1/√(2πσ²)) · exp(-(y-μ)²/(2σ²))
```
- η = μ
- T(y) = y
- a(η) = μ²/(2σ²) = η²/(2σ²)
- b(y) = (1/√(2πσ²)) · exp(-y²/(2σ²))

**Other members**: Poisson, Exponential, Gamma, Beta, Dirichlet, ...

## 3.2 Constructing GLMs

Given a classification/regression problem, we derive a GLM using three assumptions:

### Design Assumptions

1. **y | x; θ ~ ExponentialFamily(η)** - output follows exponential family
2. **h(x) = E[y | x]** - predict the expected value
3. **η = θᵀx** - natural parameter is linear in x

### 3.2.1 Ordinary Least Squares as GLM

Choose Gaussian N(μ, σ²):
```
h_θ(x) = E[y | x; θ]
       = μ           (Gaussian mean)
       = η           (for Gaussian: μ = η)
       = θᵀx         (Assumption 3)
```

This recovers **linear regression**!

### 3.2.2 Logistic Regression as GLM

Choose Bernoulli(φ):
```
h_θ(x) = E[y | x; θ]
       = φ           (Bernoulli mean = probability of 1)
       = 1/(1 + e^(-η))  (from exponential family form)
       = 1/(1 + e^(-θᵀx)) (Assumption 3)
```

This recovers **logistic regression**!

### Canonical Response and Link Functions

| Function | Definition | Example |
|----------|------------|---------|
| **Canonical Response** g(η) | E[T(y); η] | Identity for Gaussian, Sigmoid for Bernoulli |
| **Canonical Link** g⁻¹(μ) | Maps mean to η | Identity for Gaussian, Logit for Bernoulli |

## Softmax Regression as GLM

For multiclass with k outcomes, use **Multinomial distribution**:

```
P(y = i | x; θ) = exp(θᵢᵀx) / Σⱼ exp(θⱼᵀx)
```

This is the softmax function applied to the logits θᵢᵀx.

## Why GLMs Matter

1. **Unified framework**: Same recipe for many problems
2. **Principled approach**: MLE derivation is automatic
3. **Theoretical guarantees**: Convex optimization, consistent estimation
4. **Extensibility**: Easy to add new distributions

## Common GLMs in Practice

| Distribution | Typical Use | Link Function | Response |
|--------------|-------------|---------------|----------|
| Gaussian | Continuous outcomes | Identity | μ = θᵀx |
| Bernoulli | Binary classification | Logit | φ = σ(θᵀx) |
| Multinomial | Multi-class | Softmax | φᵢ = softmax(θᵀx)ᵢ |
| Poisson | Count data | Log | λ = exp(θᵀx) |
| Exponential | Time to event | Inverse | λ = 1/(θᵀx) |

## Key Takeaways

1. **Exponential family** unifies many common distributions
2. **Three assumptions** give us GLM construction recipe
3. **Linear and logistic regression** are just special GLMs
4. **Canonical response** function comes naturally from the math
5. Framework extends to Poisson regression, softmax, and more

## Practical Notes

- Use **Poisson GLM** for count data (clicks, purchases)
- Use **Multinomial GLM** for multi-class classification
- **Overdispersion**: when variance exceeds mean, consider negative binomial
- GLMs in scikit-learn: `LinearRegression`, `LogisticRegression`, `PoissonRegressor`

