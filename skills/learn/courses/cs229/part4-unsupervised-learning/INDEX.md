# Part IV: Unsupervised Learning

## Overview

Unsupervised learning discovers structure in data **without labels**. Instead of predicting y from x, we find patterns, clusters, or representations in x alone.

## Chapters in This Part

- **[Chapter 10: Clustering and K-Means](ch10-clustering.md)** - Partitioning data into groups
- **[Chapter 11: EM Algorithms](ch11-em-algorithms.md)** - Learning with latent variables
- **[Chapter 12: Principal Components Analysis](ch12-pca.md)** - Dimensionality reduction
- **[Chapter 13: Independent Components Analysis](ch13-ica.md)** - Blind source separation
- **[Chapter 14: Self-Supervised Learning & Foundation Models](ch14-foundation-models.md)** - Modern pretraining

## Core Tasks

| Task | Goal | Example |
|------|------|---------|
| **Clustering** | Group similar examples | Customer segmentation |
| **Dimensionality Reduction** | Find low-dimensional representation | Visualization, compression |
| **Density Estimation** | Model P(x) | Anomaly detection |
| **Representation Learning** | Learn useful features | Transfer learning |

## Key Algorithms

| Algorithm | Task | Key Idea |
|-----------|------|----------|
| **K-Means** | Clustering | Minimize within-cluster variance |
| **GMM + EM** | Clustering + Density | Mixture of Gaussians |
| **PCA** | Dim. Reduction | Maximum variance directions |
| **ICA** | Source Separation | Independent components |
| **VAE** | Representation | Variational inference |
| **Contrastive Learning** | Representation | Similar pairs close, different far |

## Mathematical Framework

### Latent Variable Models

Many unsupervised methods model:
```
P(x) = Σ_z P(x|z)P(z)    (discrete z)
P(x) = ∫ P(x|z)P(z)dz    (continuous z)
```

Where z is a **latent variable** (cluster assignment, hidden cause, etc.)

### The EM Algorithm

General framework for latent variable models:

**E-step**: Compute posterior over latent variables
```
Q(z) = P(z|x; θ)
```

**M-step**: Maximize expected log-likelihood
```
θ = argmax_θ E_Q[log P(x, z; θ)]
```

## Connection to Supervised Learning

| Unsupervised | Supervised Application |
|--------------|----------------------|
| Clustering | Semi-supervised learning |
| PCA | Feature preprocessing |
| Representation learning | Transfer learning, fine-tuning |
| Foundation models | Few-shot learning |

## Why Unsupervised Learning Matters

1. **Labels are expensive**: Most data is unlabeled
2. **Structure discovery**: Find patterns humans might miss
3. **Preprocessing**: Better representations for downstream tasks
4. **Foundation models**: Pretrain once, adapt many times

