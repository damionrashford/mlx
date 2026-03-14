# Chapter 12: Principal Components Analysis

## Introduction

**PCA** finds the directions of maximum variance in data, enabling:
- Dimensionality reduction
- Visualization
- Noise reduction
- Feature extraction

## Motivation

**Example**: Car data with "max speed (mph)" and "max speed (kph)" features.

These are nearly linearly dependent - the data really lies in a lower-dimensional subspace.

**Goal**: Automatically discover such redundancy and find the "true" dimensions.

## Preprocessing

Before PCA, normalize data:

1. **Zero-center**: x_j^(i) ← x_j^(i) - μ_j

2. **Scale** (optional): x_j^(i) ← x_j^(i) / σ_j

This ensures PCA treats all features equally.

## Finding the Principal Components

### Goal

Find unit vector u such that projecting data onto u **maximizes variance**.

### Setup

Projection of x onto u: x_proj = (xᵀu)u

Length of projection: xᵀu

Variance of projections:
```
(1/n) Σᵢⁿ (x^(i)ᵀu)² = uᵀ((1/n) Σᵢⁿ x^(i)(x^(i))ᵀ)u = uᵀΣu
```

Where Σ is the **sample covariance matrix**.

### Optimization Problem

```
max_u uᵀΣu
subject to: ||u|| = 1
```

### Solution

By Lagrange multipliers:
```
Σu = λu
```

u must be an **eigenvector** of Σ, and the variance is the eigenvalue λ.

**First principal component**: Eigenvector with largest eigenvalue.

### All Principal Components

Sort eigenvectors by eigenvalue: λ₁ ≥ λ₂ ≥ ... ≥ λ_d

- u₁ = first principal component (direction of max variance)
- u₂ = second principal component (max variance orthogonal to u₁)
- etc.

## The PCA Algorithm

```
1. Preprocess: zero-center (and optionally scale) the data

2. Compute covariance matrix: Σ = (1/n) XᵀX

3. Compute eigenvectors/eigenvalues of Σ

4. Sort eigenvectors by eigenvalue (descending)

5. Project data onto top k eigenvectors: X_reduced = XU_k
   where U_k = [u₁ | u₂ | ... | u_k]
```

## Variance Explained

**Fraction of variance explained by top k components**:
```
(Σⱼ₌₁ᵏ λⱼ) / (Σⱼ₌₁ᵈ λⱼ)
```

**Rule of thumb**: Choose k to explain 95-99% of variance.

### Scree Plot

Plot eigenvalues vs. component number:
```
λ
↑
│●
│ ●
│  ●
│   ●●●●●●●●●●
└──────────────→ Component
```

Look for "elbow" where eigenvalues level off.

## Alternative View: Reconstruction Error

PCA also minimizes **reconstruction error**:
```
min_U (1/n) Σᵢⁿ ||x^(i) - UUᵀx^(i)||²
```

Subject to U having k orthonormal columns.

## Properties of PCA

1. **Principal components are orthogonal**
2. **PCA is linear**: Projections are linear combinations of features
3. **Unique** (up to sign flips)
4. **Optimal** for linear dimensionality reduction (in variance/reconstruction sense)

## Computation

### Eigendecomposition

```
Σ = UΛUᵀ
```

where U contains eigenvectors, Λ is diagonal with eigenvalues.

**Complexity**: O(d³)

### SVD Approach

For large n, use SVD of data matrix X:
```
X = USVᵀ
```

Columns of V are principal components.
Singular values s_i = √(n·λ_i)

**Advantage**: Numerically stable, works when n < d.

## Limitations of PCA

1. **Linear only**: Cannot capture nonlinear structure
2. **Variance-based**: Ignores discriminative information
3. **Sensitive to scaling**: Preprocessing matters
4. **Orthogonality**: May not match natural data structure

## Extensions

| Method | Extension |
|--------|-----------|
| **Kernel PCA** | Nonlinear via kernel trick |
| **Sparse PCA** | Interpretable components |
| **Probabilistic PCA** | Generative model |
| **ICA** | Independent (not just uncorrelated) components |
| **t-SNE/UMAP** | Nonlinear for visualization |

## Key Takeaways

1. **PCA** finds directions of maximum variance
2. **Eigenvectors** of covariance matrix are principal components
3. **Eigenvalues** indicate variance explained
4. **Choose k** to retain sufficient variance (95-99%)
5. **Use SVD** for numerical stability

## Practical Notes

- **Always center data** before PCA
- **Scale features** if they have different units
- **Check variance explained** to choose k
- **Visualize**: First 2-3 components often reveal structure
- **scikit-learn**: `PCA(n_components=k)`, `pca.explained_variance_ratio_`

