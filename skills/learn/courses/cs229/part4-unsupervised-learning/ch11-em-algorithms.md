# Chapter 11: EM Algorithms

## Introduction

**Expectation-Maximization (EM)** is a general algorithm for maximum likelihood estimation when data has **latent (hidden) variables**.

## 11.1 EM for Mixture of Gaussians

### The Model

Data comes from k Gaussian distributions:
```
P(x) = Σⱼᵏ φⱼ · N(x; μⱼ, Σⱼ)
```

Where:
- φⱼ = mixing proportion (φⱼ ≥ 0, Σⱼ φⱼ = 1)
- μⱼ = mean of component j
- Σⱼ = covariance of component j

**Latent variable**: z^(i) ∈ {1, ..., k} = which component generated x^(i)

### The EM Algorithm for GMM

**E-step**: Compute posterior probabilities (responsibilities)
```
w_j^(i) = P(z^(i) = j | x^(i); φ, μ, Σ)
        = (φⱼ · N(x^(i); μⱼ, Σⱼ)) / (Σₗ φₗ · N(x^(i); μₗ, Σₗ))
```

**M-step**: Update parameters using weighted MLE
```
φⱼ = (1/n) Σᵢⁿ w_j^(i)

μⱼ = (Σᵢⁿ w_j^(i) · x^(i)) / (Σᵢⁿ w_j^(i))

Σⱼ = (Σᵢⁿ w_j^(i) · (x^(i) - μⱼ)(x^(i) - μⱼ)ᵀ) / (Σᵢⁿ w_j^(i))
```

**Repeat** until convergence.

### Comparison to K-Means

| K-Means | GMM-EM |
|---------|--------|
| Hard assignments (0 or 1) | Soft assignments (probabilities) |
| Spherical clusters | Elliptical clusters (full covariance) |
| Heuristic objective | Maximum likelihood |
| Always converges | Always converges |

## 11.2 Jensen's Inequality

**Key mathematical tool** for proving EM convergence.

### Statement

For a **convex** function f and random variable X:
```
f(E[X]) ≤ E[f(X)]
```

For **concave** f (like log):
```
f(E[X]) ≥ E[f(X)]
```

**Equality** holds when X is constant.

### Graphical Intuition

For convex f, the chord lies above the curve.

## 11.3 General EM Algorithm

### The Setup

- Observed data: x^(1), ..., x^(n)
- Latent variables: z^(1), ..., z^(n)
- Parameters: θ
- Goal: Maximize log P(x; θ) = log Σ_z P(x, z; θ)

### The Problem

Direct maximization of log Σ_z P(x, z; θ) is hard because log doesn't pass through sum.

### The ELBO (Evidence Lower Bound)

For any distribution Q over z:
```
log P(x; θ) ≥ E_Q[log P(x, z; θ)] - E_Q[log Q(z)]
            = ELBO(Q, θ)
```

Equality when Q(z) = P(z|x; θ).

### General EM Algorithm

**E-step**: Set Q(z) = P(z|x; θ) (makes ELBO tight)

**M-step**: Maximize ELBO over θ:
```
θ := argmax_θ E_Q[log P(x, z; θ)]
```

### Convergence Guarantee

EM monotonically increases log-likelihood:
```
log P(x; θ^(t+1)) ≥ log P(x; θ^(t))
```

**Caveat**: Converges to local maximum, not necessarily global.

## 11.4 Mixture of Gaussians Revisited

GMM-EM is a special case:

**E-step**: Q(z^(i) = j) = w_j^(i) (responsibilities)

**M-step**: Weighted MLE for Gaussian parameters

## 11.5 Variational Inference and VAE (Optional)

### When Exact E-Step is Intractable

For complex models, P(z|x; θ) is intractable. 

**Solution**: Approximate with a simpler Q from a tractable family Q.

```
max_{Q ∈ Q} max_θ ELBO(Q, θ)
```

### Variational Autoencoders (VAEs)

**Model**:
```
z ~ N(0, I)
x | z ~ P(x | g(z; θ))  where g is a neural network (decoder)
```

**Approximate posterior**:
```
Q(z|x) = N(μ(x; φ), diag(σ(x; φ)²))  (encoder)
```

μ and σ are neural networks.

### The Reparameterization Trick

To backpropagate through sampling:
```
z = μ(x; φ) + σ(x; φ) ⊙ ε,  where ε ~ N(0, I)
```

This makes z differentiable with respect to φ.

### VAE Loss (Negative ELBO)

```
L = -E_{z~Q}[log P(x|z; θ)] + KL(Q(z|x; φ) || P(z))
    \_________Reconstruction________/  \_____Regularization____/
```

## Key Takeaways

1. **EM** handles latent variable models via alternating optimization
2. **E-step** computes posterior over latent variables
3. **M-step** maximizes expected complete-data log-likelihood
4. **ELBO** provides a lower bound on log-likelihood
5. **VAEs** extend EM with neural network parameterization

## Practical Notes

- **Initialization**: Multiple random restarts for GMM
- **Convergence**: Monitor log-likelihood, use tolerance
- **Model selection**: Use BIC/AIC to choose number of components
- **scikit-learn**: `GaussianMixture(n_components=k)`
- **Deep learning**: VAE implementations in PyTorch/TensorFlow

