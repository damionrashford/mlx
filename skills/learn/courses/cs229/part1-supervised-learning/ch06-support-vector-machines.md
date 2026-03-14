# Chapter 6: Support Vector Machines

## Introduction

SVMs are among the best "off-the-shelf" supervised learning algorithms. They find the decision boundary that **maximizes the margin** between classes.

## 6.1 Margins: Intuition

### Confidence in Predictions

In logistic regression, P(y=1|x) = g(θᵀx).

- If θᵀx >> 0: Very confident y = 1
- If θᵀx << 0: Very confident y = 0
- If θᵀx ≈ 0: Not confident (near decision boundary)

**Intuition**: Points far from the decision boundary → confident predictions.

### Geometric View

Consider training examples labeled x (positive) and o (negative):

```
       x    x
          A    <- far from boundary, confident
      x
---------B----  <- decision boundary (θᵀx = 0)
    C o    o   <- C is close, less confident
       o    o
```

**Goal**: Find a decision boundary that keeps ALL points as far away as possible.

## 6.2 Notation

For SVM, we use:

- y ∈ {-1, +1} (instead of {0, 1})
- Parameters: w, b (instead of θ)
- Classifier: h\_{w,b}(x) = g(wᵀx + b)

Where g(z) = +1 if z ≥ 0, else -1.

## 6.3 Functional and Geometric Margins

### Functional Margin

For a training example (x^(i), y^(i)):

```
γ̂^(i) = y^(i)(wᵀx^(i) + b)
```

**Properties**:

- γ̂^(i) > 0 ⟹ correct classification
- Larger γ̂ = more confident

**Problem**: Scaling w and b by 2 doubles γ̂ without changing predictions.

### Geometric Margin

The actual distance from x to the hyperplane:

```
γ^(i) = y^(i) · (wᵀx^(i) + b) / ||w||
```

**Properties**:

- Invariant to scaling of w, b
- True distance to decision boundary
- γ^(i) = γ̂^(i) / ||w||

### Margin of Training Set

```
γ̂ = min_i γ̂^(i)    (smallest functional margin)
γ = min_i γ^(i)     (smallest geometric margin)
```

## 6.4 The Optimal Margin Classifier

**Goal**: Find w, b that maximize the geometric margin.

### Optimization Problem (v1)

```
max_{γ,w,b} γ
subject to: y^(i)(wᵀx^(i) + b) ≥ γ,  i = 1,...,n
            ||w|| = 1
```

**Problem**: ||w|| = 1 is non-convex.

### Optimization Problem (v2)

Use the scaling freedom. Set functional margin γ̂ = 1, then geometric margin γ = 1/||w||.

Maximizing 1/||w|| = minimizing ||w||² = minimizing ½||w||²:

```
min_{w,b} ½||w||²
subject to: y^(i)(wᵀx^(i) + b) ≥ 1,  i = 1,...,n
```

**This is a convex quadratic program!**

## 6.5 Lagrange Duality

### The Primal Problem

```
min_w f(w)
subject to: gᵢ(w) ≤ 0,  i = 1,...,k
            hᵢ(w) = 0,  i = 1,...,l
```

### The Lagrangian

```
L(w, α, β) = f(w) + Σᵢᵏ αᵢgᵢ(w) + Σᵢˡ βᵢhᵢ(w)
```

Where α, β are **Lagrange multipliers**.

### Primal vs Dual

**Primal**: min*w max*{α,β: α≥0} L(w, α, β) = p\*

**Dual**: max\_{α,β: α≥0} min_w L(w, α, β) = d\*

**Weak duality**: d* ≤ p* (always)

**Strong duality**: d* = p* (under conditions - Slater's condition)

### KKT Conditions

At optimum (w*, α*, β\*):

1. ∂L/∂wᵢ = 0 (stationarity)
2. ∂L/∂βᵢ = 0 (stationarity)
3. αᵢ*gᵢ(w*) = 0 (complementary slackness)
4. gᵢ(w\*) ≤ 0 (primal feasibility)
5. αᵢ\* ≥ 0 (dual feasibility)

**Key insight from (3)**: αᵢ > 0 ⟹ gᵢ(w) = 0 (constraint is active)

## 6.6 The Dual Form

### SVM Lagrangian

```
L(w, b, α) = ½||w||² - Σᵢⁿ αᵢ[y^(i)(wᵀx^(i) + b) - 1]
```

### Deriving the Dual

Setting ∂L/∂w = 0: w = Σᵢⁿ αᵢy^(i)x^(i)

Setting ∂L/∂b = 0: Σᵢⁿ αᵢy^(i) = 0

Substituting back:

```
max_α W(α) = Σᵢⁿ αᵢ - ½Σᵢ,ⱼⁿ y^(i)y^(j)αᵢαⱼ⟨x^(i), x^(j)⟩

subject to: αᵢ ≥ 0,  i = 1,...,n
            Σᵢⁿ αᵢy^(i) = 0
```

**Key observation**: The dual depends only on inner products ⟨x^(i), x^(j)⟩!

### Making Predictions

```
wᵀx + b = (Σᵢⁿ αᵢy^(i)x^(i))ᵀx + b
        = Σᵢⁿ αᵢy^(i)⟨x^(i), x⟩ + b
```

Only inner products needed - **ready for kernels**!

### Support Vectors

From KKT complementary slackness: αᵢ > 0 ⟹ y^(i)(wᵀx^(i) + b) = 1

Points with αᵢ > 0 lie exactly on the margin - these are **support vectors**.

**Key property**: Only support vectors matter for the decision boundary!

## 6.7 Regularization (Soft Margin)

What if data isn't linearly separable?

### L1-SVM (Soft Margin)

Allow some constraint violations:

```
min_{w,b,ξ} ½||w||² + C Σᵢⁿ ξᵢ

subject to: y^(i)(wᵀx^(i) + b) ≥ 1 - ξᵢ
            ξᵢ ≥ 0
```

- ξᵢ = "slack variable" (how much constraint i is violated)
- C = regularization parameter (higher = fewer violations allowed)

### Dual with Regularization

```
max_α Σᵢⁿ αᵢ - ½Σᵢ,ⱼⁿ y^(i)y^(j)αᵢαⱼ⟨x^(i), x^(j)⟩

subject to: 0 ≤ αᵢ ≤ C,  i = 1,...,n  (box constraint!)
            Σᵢⁿ αᵢy^(i) = 0
```

## 6.8 The SMO Algorithm

**Sequential Minimal Optimization**: Efficient algorithm for solving SVM dual.

### Coordinate Ascent

Update one αᵢ at a time, holding others fixed.

**Problem for SVM**: Constraint Σᵢ αᵢy^(i) = 0 means we can't change just one α!

### SMO Key Idea

Update **two** αᵢ's at a time:

1. Select α₁, α₂ to update
2. Optimize over α₁, α₂ while satisfying constraints
3. This is a simple quadratic optimization with box constraints

### Coordinate Selection Heuristics

Choose αᵢ that violates KKT conditions most.

**Practical notes**: SMO is very efficient, making SVM training practical for large datasets.

## Key Takeaways

1. **Maximum margin** principle gives robust classifiers
2. **Support vectors** are the only points that matter
3. **Dual form** uses only inner products → kernels work!
4. **Soft margin** handles non-separable data
5. **SMO** makes training efficient

## Practical Notes

- **C parameter**: Low C = more regularization, smoother boundary
- **Kernel choice**: RBF is default, try linear first for high-d data
- **Feature scaling**: Critical for SVMs
- **scikit-learn**: `SVC(kernel='rbf', C=1.0)`, `LinearSVC`
- **Large datasets**: Consider `SGDClassifier(loss='hinge')` or approximations
