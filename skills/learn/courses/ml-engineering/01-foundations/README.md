# 01 — Mathematical Foundations

> The math that makes ML work. You don't need to derive proofs — you need to understand what's happening and why.

## Why This Matters

Without this, you can USE XGBoost but can't explain WHY it works, debug WHY a model isn't converging, or design a new loss function. Interviewers will test this.

## Subdirectories

```
01-foundations/
├── linear-algebra/       # Vectors, matrices, eigenvalues, SVD — how neural nets represent data
├── calculus/              # Derivatives, chain rule, gradients — how models learn (backpropagation)
├── probability-statistics/ # Distributions, Bayes, hypothesis testing, maximum likelihood
└── optimization/          # Gradient descent, convex optimization, loss functions
```

## Key Resources

| Resource | What it covers | Time | Type |
|---|---|---|---|
| 3Blue1Brown: Essence of Linear Algebra | Visual intuition for vectors, matrices, transforms | ~3.5 hrs | YouTube (free) |
| 3Blue1Brown: Essence of Calculus | Derivatives, integrals, chain rule visually | ~3 hrs | YouTube (free) |
| StatQuest (Josh Starmer) | Statistics and ML math explained simply | Ongoing | YouTube (free) |
| Mathematics for ML & Data Science Specialization | Formal course covering all areas | ~4 months | DeepLearning.AI |

## What to Learn (Not Memorize)

### Linear Algebra — Know These Concepts
- What a vector is and what vector spaces mean
- Matrix multiplication — what it geometrically represents (transformation)
- Eigenvalues/eigenvectors — directions that don't change under transformation
- SVD — how to decompose any matrix (used in dimensionality reduction, recommenders)
- Dot product — how similarity is measured

### Calculus — Know These Concepts
- What a derivative means (rate of change, slope)
- Chain rule — how backpropagation works (the core of neural network training)
- Partial derivatives — how gradients work in multi-variable functions
- What a gradient is — direction of steepest ascent

### Probability & Statistics — Know These Concepts
- Probability distributions (normal, Bernoulli, categorical)
- Bayes' theorem — updating beliefs with evidence
- Maximum likelihood estimation — how models find optimal parameters
- Expectation and variance
- Hypothesis testing basics (for A/B testing models)

### Optimization — Know These Concepts
- Gradient descent — walk downhill on the loss surface
- Learning rate — step size (most important hyperparameter)
- Convexity — why some problems are easier to optimize
- Local vs global minima — why deep learning still works despite non-convexity
- Adam optimizer — why it works well in practice

## Practice Format

For each concept, write a brief entry:
1. **What is it?** (1-2 sentences)
2. **Why does it matter for ML?** (1-2 sentences)
3. **When would I encounter this?** (practical scenario)
