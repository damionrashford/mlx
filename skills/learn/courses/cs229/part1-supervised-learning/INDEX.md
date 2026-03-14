# Part I: Supervised Learning

## Overview

Supervised learning is the foundation of predictive machine learning. Given a training set of input-output pairs, the goal is to learn a function h: X → Y that maps inputs to outputs.

**Key Question**: Given data like housing prices, how can we learn to predict prices of other houses?

## Chapters in This Part

1. **[Chapter 1: Linear Regression](ch01-linear-regression.md)** - The simplest predictive model
2. **[Chapter 2: Classification & Logistic Regression](ch02-logistic-regression.md)** - Binary classification
3. **[Chapter 3: Generalized Linear Models](ch03-generalized-linear-models.md)** - Unified framework
4. **[Chapter 4: Generative Learning Algorithms](ch04-generative-algorithms.md)** - GDA, Naive Bayes
5. **[Chapter 5: Kernel Methods](ch05-kernel-methods.md)** - Feature maps and kernel trick
6. **[Chapter 6: Support Vector Machines](ch06-support-vector-machines.md)** - Maximum margin classifiers

## Core Concepts

### Problem Formulation
- **Regression**: y is continuous (e.g., housing prices)
- **Classification**: y is discrete (e.g., spam/not-spam)

### Standard Notation
```
x^(i)  = Input features for i-th example
y^(i)  = Output/target for i-th example
(x^(i), y^(i)) = Training example
n      = Number of training examples
d      = Number of features
X      = Design matrix ∈ ℝ^(n×d)
θ      = Model parameters
h_θ(x) = Hypothesis function
```

### The Learning Pipeline
```
Training Set → Learning Algorithm → h (hypothesis)
                                    ↓
                              x → h(x) → predicted y
```

## Key Algorithms

| Algorithm | Problem Type | Loss Function | Key Property |
|-----------|--------------|---------------|--------------|
| Linear Regression | Regression | Squared Error | Closed-form solution |
| Logistic Regression | Classification | Cross-Entropy | Probabilistic output |
| GDA | Classification | Likelihood | Generative model |
| Naive Bayes | Classification | Likelihood | Feature independence |
| SVM | Classification | Hinge Loss | Maximum margin |

## Mathematical Foundations

### Cost Function (Linear Regression)
```
J(θ) = (1/2) Σᵢ (h_θ(x^(i)) - y^(i))²
```

### Gradient Descent
```
θⱼ := θⱼ - α · ∂J(θ)/∂θⱼ
```
Where α is the learning rate.

### Probabilistic Interpretation
Least squares regression = Maximum Likelihood Estimation (MLE) under Gaussian noise assumption.

## Discriminative vs Generative

| Approach | Models | Examples |
|----------|--------|----------|
| **Discriminative** | P(y\|x) directly | Logistic Regression, SVM |
| **Generative** | P(x\|y) and P(y) | GDA, Naive Bayes |

Generative models can be converted to discriminative using Bayes' rule:
```
P(y|x) = P(x|y)P(y) / P(x)
```

## Why This Matters for Production ML

1. **Understanding fundamentals** enables debugging complex models
2. **Linear models** are often sufficient and more interpretable
3. **Probabilistic interpretation** enables uncertainty quantification
4. **SVMs with kernels** can model complex decision boundaries efficiently

