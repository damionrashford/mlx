# Chapter 5: Kernel Methods

## Introduction

**Problem**: Linear models are limited. What if the decision boundary is nonlinear?

**Solution 1**: Add polynomial features manually (x, x², x³, ..., x₁x₂, ...)

**Solution 2**: Use **kernels** - implicitly work in high-dimensional feature space without computing features explicitly.

## 5.1 Feature Maps

A **feature map** φ: X → ℝᵖ transforms inputs into a higher-dimensional space.

**Example** (quadratic features for x ∈ ℝ²):
```
φ(x) = [x₁², √2·x₁x₂, x₂², √2·x₁, √2·x₂, 1]ᵀ ∈ ℝ⁶
```

Now fit a linear model in the new space:
```
h_θ(x) = θᵀφ(x)
```

This is **linear in φ(x)** but **nonlinear in x**!

## 5.2 LMS with Features

Standard LMS update:
```
θ := θ + α(y^(i) - θᵀx^(i))x^(i)
```

With features:
```
θ := θ + α(y^(i) - θᵀφ(x^(i)))φ(x^(i))
```

### The Representor Theorem

Starting from θ = 0, after n updates:
```
θ = Σᵢⁿ βᵢ φ(x^(i))
```

θ is always a linear combination of the training examples (in feature space)!

## 5.3 LMS with the Kernel Trick

### Key Insight

Instead of storing θ ∈ ℝᵖ, store β ∈ ℝⁿ!

**Prediction**:
```
θᵀφ(x) = (Σᵢⁿ βᵢ φ(x^(i)))ᵀ φ(x)
        = Σᵢⁿ βᵢ φ(x^(i))ᵀφ(x)
        = Σᵢⁿ βᵢ K(x^(i), x)
```

Where **K(x, z) = φ(x)ᵀφ(z)** is the **kernel function**.

### The Kernel Function

Instead of computing φ(x) explicitly, we only need K(x, z).

**Example**: For the quadratic feature map above:
```
K(x, z) = φ(x)ᵀφ(z) = (xᵀz + 1)²
```

Computing (xᵀz + 1)² is O(d), but computing φ(x)ᵀφ(z) directly would be O(d²)!

### LMS Update with Kernels

```
βᵢ := βᵢ + α(y^(i) - Σⱼⁿ βⱼ K(x^(j), x^(i)))
```

Or in matrix form, let K be the n×n **kernel matrix** with Kᵢⱼ = K(x^(i), x^(j)):
```
β := β + α(y - Kβ)
```

## 5.4 Properties of Kernels

### What Makes a Valid Kernel?

**Mercer's Theorem**: K is a valid kernel if and only if for any {x^(1), ..., x^(n)}, the kernel matrix K is positive semi-definite.

### Common Kernels

| Kernel | Formula | Feature Space |
|--------|---------|---------------|
| **Linear** | K(x,z) = xᵀz | Original |
| **Polynomial** | K(x,z) = (xᵀz + c)ᵈ | All monomials up to degree d |
| **RBF/Gaussian** | K(x,z) = exp(-\|\|x-z\|\|²/(2σ²)) | **Infinite dimensional!** |
| **Laplacian** | K(x,z) = exp(-\|\|x-z\|\|₁/σ) | Infinite dimensional |

### The RBF Kernel

```
K(x, z) = exp(-||x - z||² / (2σ²))
```

**Properties**:
- K(x, x) = 1 (points are maximally similar to themselves)
- K(x, z) → 0 as ||x - z|| → ∞
- σ controls the "bandwidth" (locality)

**Remarkable**: This corresponds to an **infinite-dimensional** feature space!

### Constructing New Kernels

If K₁ and K₂ are valid kernels, so are:
- K₁ + K₂
- c · K₁ (for c > 0)
- K₁ · K₂
- f(x)K₁(x,z)f(z) for any function f
- K(φ(x), φ(z)) for any feature map φ

## The Kernel Trick

**Big Idea**: If your algorithm can be written using only inner products ⟨x, z⟩, replace them with K(x, z) to work in high/infinite-dimensional space!

**Applicable to**:
- Linear regression → Kernel regression
- Perceptron → Kernel perceptron
- SVM → Kernel SVM (next chapter)
- PCA → Kernel PCA
- K-means → Kernel K-means

## Computational Considerations

| Approach | Computation | Storage |
|----------|-------------|---------|
| Explicit features | O(p) per prediction | O(p) for θ |
| Kernel | O(n) per prediction | O(n) for β, O(n²) for K |

**Trade-off**: Kernels are better when p >> n (high-dimensional features, few examples).

## Key Takeaways

1. **Feature maps** transform inputs to enable nonlinear models
2. **Kernel trick** avoids explicit feature computation
3. **RBF kernel** corresponds to infinite-dimensional space
4. **Any algorithm using only inner products** can be kernelized
5. **Mercer's theorem** characterizes valid kernels

## Practical Notes

- **RBF kernel** is the most popular for general use
- **σ (bandwidth)** is a critical hyperparameter - tune via cross-validation
- **Kernel matrix** can be precomputed for efficiency
- **Approximations** (Random Fourier Features) scale kernels to large datasets
- **scikit-learn**: `kernel='rbf'` in SVC, `kernel_ridge.KernelRidge`

