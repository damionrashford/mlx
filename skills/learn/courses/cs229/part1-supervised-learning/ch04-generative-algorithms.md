# Chapter 4: Generative Learning Algorithms

## Introduction

So far we've studied **discriminative** algorithms that model P(y|x) directly.

Now we explore **generative** algorithms that model P(x|y) and P(y), then use Bayes' rule.

## Discriminative vs Generative

### Discriminative Approach
- Model P(y|x) directly
- Find decision boundary separating classes
- Examples: Logistic Regression, SVM, Neural Networks

### Generative Approach
- Model P(x|y) - what does each class "look like"?
- Model P(y) - prior probability of each class
- Use Bayes' rule: P(y|x) = P(x|y)P(y) / P(x)

**Intuition (Elephant vs Dog Classification)**:
- **Discriminative**: Find a line separating elephants from dogs
- **Generative**: Build a model of what elephants look like, build a model of what dogs look like, then match new animals to each model

### Making Predictions

```
arg max_y P(y|x) = arg max_y P(x|y)P(y) / P(x)
                 = arg max_y P(x|y)P(y)
```

We don't need P(x) for classification!

## 4.1 Gaussian Discriminant Analysis (GDA)

Assume features are continuous and class-conditional densities are Gaussian.

### 4.1.1 Multivariate Normal Distribution

```
p(x; μ, Σ) = (1/((2π)^(d/2)|Σ|^(1/2))) · exp(-½(x-μ)ᵀΣ⁻¹(x-μ))
```

Where:
- μ ∈ ℝᵈ = mean vector
- Σ ∈ ℝᵈˣᵈ = covariance matrix (symmetric, positive semi-definite)

**Properties**:
- E[X] = μ
- Cov(X) = Σ

### 4.1.2 The GDA Model

**Assumptions**:
```
y ~ Bernoulli(φ)
x | y = 0 ~ N(μ₀, Σ)
x | y = 1 ~ N(μ₁, Σ)
```

Note: Same covariance Σ for both classes.

**Parameters**: φ, μ₀, μ₁, Σ

### Maximum Likelihood Estimation

**Log-likelihood**:
```
ℓ(φ, μ₀, μ₁, Σ) = log Πᵢⁿ p(x^(i)|y^(i)) · p(y^(i))
```

**Closed-form solutions**:
```
φ = (1/n) Σᵢⁿ 1{y^(i) = 1}

μ₀ = Σᵢⁿ 1{y^(i) = 0} x^(i) / Σᵢⁿ 1{y^(i) = 0}

μ₁ = Σᵢⁿ 1{y^(i) = 1} x^(i) / Σᵢⁿ 1{y^(i) = 1}

Σ = (1/n) Σᵢⁿ (x^(i) - μ_{y^(i)})(x^(i) - μ_{y^(i)})ᵀ
```

**Intuition**:
- φ = fraction of positive examples
- μ₀ = mean of class 0 examples
- μ₁ = mean of class 1 examples
- Σ = average covariance

### Decision Boundary

GDA produces a **linear** decision boundary (when Σ is shared).

The boundary is where P(y=1|x) = P(y=0|x), which is a hyperplane.

### 4.1.3 GDA vs Logistic Regression

**Connection**: If P(x|y) is Gaussian (with shared Σ), then P(y|x) follows logistic function!

**Comparison**:

| Aspect | GDA | Logistic Regression |
|--------|-----|---------------------|
| Model | P(x\|y), P(y) | P(y\|x) |
| Assumption | Gaussian x\|y | None on x |
| Efficiency | More efficient if assumption holds | Robust to violations |
| Data needed | Less (stronger assumption) | More (weaker assumption) |

**Rule of thumb**:
- GDA: Use when P(x|y) is approximately Gaussian
- Logistic: Use when assumption is uncertain or violated

## 4.2 Naive Bayes

For **discrete** features (e.g., text classification).

### The Model

Assume features are **conditionally independent** given class:

```
P(x|y) = P(x₁|y) · P(x₂|y) · ... · P(xₐ|y) = Πⱼᵈ P(xⱼ|y)
```

**For text (binary features: word present/absent)**:
```
P(xⱼ = 1|y = 1) = φⱼ|y=1
P(xⱼ = 1|y = 0) = φⱼ|y=0
```

### Maximum Likelihood Estimates

```
φⱼ|y=1 = Σᵢⁿ 1{xⱼ^(i) = 1 ∧ y^(i) = 1} / Σᵢⁿ 1{y^(i) = 1}

φⱼ|y=0 = Σᵢⁿ 1{xⱼ^(i) = 1 ∧ y^(i) = 0} / Σᵢⁿ 1{y^(i) = 0}

φy = Σᵢⁿ 1{y^(i) = 1} / n
```

### Making Predictions

```
P(y = 1|x) ∝ P(x|y=1)P(y=1)
           = P(y=1) · Πⱼ P(xⱼ|y=1)
```

Take argmax over y ∈ {0, 1}.

### 4.2.1 Laplace Smoothing

**Problem**: If a word never appears in training for a class, P(word|class) = 0, making P(class|doc) = 0.

**Solution**: Add pseudocounts (Laplace smoothing):

```
φⱼ|y=1 = (Σᵢⁿ 1{xⱼ^(i) = 1 ∧ y^(i) = 1} + 1) / (Σᵢⁿ 1{y^(i) = 1} + 2)
```

General form: Add 1 to numerator, add k (number of values) to denominator.

### 4.2.2 Event Models for Text

**Multivariate Bernoulli**: Binary features (word present/absent)
**Multinomial**: Count features (word counts)

For long documents, multinomial often works better.

## Why "Naive"?

The conditional independence assumption is almost always **wrong**. Words like "machine" and "learning" are correlated.

But Naive Bayes often works well anyway! The probabilities may be wrong, but the **ranking** (argmax) is often correct.

## Key Takeaways

1. **Generative models** model P(x|y), then use Bayes' rule
2. **GDA** assumes Gaussian class-conditionals with shared covariance
3. **Naive Bayes** assumes feature independence (often wrong, usually works)
4. **Laplace smoothing** prevents zero probabilities
5. Generative models can be more **data-efficient** with correct assumptions

## Practical Notes

- **Text classification**: Naive Bayes is fast and surprisingly effective
- **Feature engineering**: Log-probabilities prevent underflow
- **Imbalanced classes**: Prior P(y) matters!
- **scikit-learn**: `GaussianNB`, `MultinomialNB`, `BernoulliNB`

