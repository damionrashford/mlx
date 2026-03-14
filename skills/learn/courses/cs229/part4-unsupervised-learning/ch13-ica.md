# Chapter 13: Independent Components Analysis

## Introduction

**ICA** (Independent Components Analysis) recovers independent source signals from mixed observations.

**Classic example**: Cocktail party problem
- Microphones record mixed voices
- ICA separates individual speakers

## The Model

**Observations**: x = As

Where:
- s ∈ ℝⁿ = source signals (unknown)
- A ∈ ℝⁿˣⁿ = mixing matrix (unknown)
- x ∈ ℝⁿ = observed mixtures

**Goal**: Given only x^(1), ..., x^(m), recover A (and hence s = A⁻¹x).

### Assumption

Source components s₁, ..., sₙ are **statistically independent**:
```
P(s) = Πᵢⁿ p_i(sᵢ)
```

## 13.1 ICA Ambiguities

### What ICA Cannot Determine

1. **Permutation**: Can recover sources in any order
2. **Scaling**: Can multiply source by constant (absorbed into A)
3. **Sign**: Can flip sign of source (absorbed into A)

### Gaussian Sources Don't Work

If sources are Gaussian, the joint distribution is:
```
P(x) = N(0, AAᵀ)
```

This only depends on AAᵀ, so A cannot be uniquely identified (rotation ambiguity).

**Key requirement**: Sources must be **non-Gaussian** (except at most one).

## 13.2 Densities and Linear Transformations

If s has density p_s(s) and x = As, then:
```
p_x(x) = p_s(A⁻¹x) · |det(A⁻¹)|
       = p_s(Wx) · |det(W)|
```

Where W = A⁻¹ is the **unmixing matrix**.

## 13.3 ICA Algorithm

### Maximum Likelihood Formulation

Given observations x^(1), ..., x^(m):
```
ℓ(W) = Σᵢᵐ [Σⱼⁿ log p_j(wⱼᵀx^(i)) + log|det(W)|]
```

Where wⱼ is the j-th row of W.

### Choosing Source Distributions

Use a **fixed** non-Gaussian distribution for p_j.

Common choice (sigmoid CDF):
```
p(s) = d/ds [1/(1 + e^(-s))] = e^(-s)/(1 + e^(-s))²
```

Corresponding log-derivative:
```
d/ds log p(s) = 1 - 2·sigmoid(s)
```

### Gradient Ascent

Update rule:
```
W := W + α[(1 - 2g(Wx^(i)))x^(i)ᵀ + (Wᵀ)⁻¹]
```

Where g = sigmoid function applied elementwise.

### Natural Gradient (Faster)

Multiply by WWᵀ (right-multiplication):
```
W := W + α[(1 - 2g(Wx^(i)))(Wx^(i))ᵀ + I]W
```

This converges faster and is easier to compute (no matrix inverse).

## PCA vs ICA

| Aspect | PCA | ICA |
|--------|-----|-----|
| **Goal** | Uncorrelated components | Independent components |
| **Orthogonal** | Yes | No |
| **Order** | By variance | No ordering |
| **Gaussian** | Works fine | Requires non-Gaussian |
| **Unique** | Yes (up to sign) | No (permutation, scaling) |

**Key difference**: Uncorrelated ≠ Independent

- Uncorrelated: E[XY] = E[X]E[Y] (second moment)
- Independent: P(X,Y) = P(X)P(Y) (all moments)

## Applications

1. **Blind source separation**: Audio, EEG signals
2. **Feature extraction**: Images, text
3. **Artifact removal**: EEG/MEG noise removal
4. **Financial data**: Factor models

## Key Takeaways

1. **ICA** recovers independent sources from mixtures
2. **Non-Gaussian** sources required (except at most one)
3. **Ambiguities**: permutation, scaling, sign
4. **Maximum likelihood** formulation enables gradient optimization
5. **Stronger than PCA**: Independent > uncorrelated

## Practical Notes

- **Preprocess**: Center and whiten data before ICA
- **Whitening**: Transform data to have identity covariance
- **Number of sources**: Usually assume same as observations
- **FastICA**: Popular algorithm (fixed-point iteration)
- **scikit-learn**: `FastICA(n_components=k)`

