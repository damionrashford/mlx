# Chapter 10: Clustering and the K-Means Algorithm

## Introduction

**Clustering** partitions data into groups where examples within a group are similar and examples across groups are different.

**No labels needed** - purely based on input features x.

## The K-Means Algorithm

### Goal

Partition n examples into k clusters to minimize within-cluster variance.

### Algorithm

```
1. Initialize cluster centroids μ₁, ..., μₖ randomly

2. Repeat until convergence:
   
   a. Assignment step: For each example i, assign to nearest centroid
      c^(i) = argmin_j ||x^(i) - μⱼ||²
   
   b. Update step: Recompute centroids as cluster means
      μⱼ = (Σᵢ 1{c^(i) = j} · x^(i)) / (Σᵢ 1{c^(i) = j})
```

### Properties

**Convergence**: K-means always converges (objective decreases monotonically).

**Local optima**: Result depends on initialization - not globally optimal.

**Complexity**: O(n · k · d · iterations), typically fast in practice.

### Objective Function

K-means minimizes:
```
J(c, μ) = Σᵢⁿ ||x^(i) - μ_{c^(i)}||²
```

This is also called the **distortion** or **inertia**.

## Initialization Strategies

### Random Initialization

Pick k random examples as initial centroids.

**Problem**: Can lead to poor local optima.

### K-Means++

Smart initialization that spreads out initial centroids:

1. Choose first centroid uniformly at random
2. For each subsequent centroid:
   - Compute D(x) = distance to nearest existing centroid
   - Choose new centroid with probability ∝ D(x)²

**Guarantees**: O(log k) approximation to optimal.

### Multiple Restarts

Run k-means multiple times with different initializations, keep best result.

## Choosing K

K-means requires specifying k in advance. How to choose?

### The Elbow Method

Plot distortion vs. k, look for "elbow" where improvement slows.

```
Distortion
    ↑
    │╲
    │ ╲
    │  ╲___   ← Elbow at k=3
    │      ╲__________
    └─────────────────→ k
       1  2  3  4  5
```

### Silhouette Score

For each point, compare:
- a = average distance to points in same cluster
- b = average distance to points in nearest other cluster

Silhouette = (b - a) / max(a, b)

Range: [-1, 1], higher is better.

### Gap Statistic

Compare within-cluster dispersion to that expected under null reference.

## Limitations of K-Means

1. **Assumes spherical clusters**: Uses Euclidean distance
2. **Sensitive to outliers**: Means are pulled by outliers
3. **Fixed k**: Must specify number of clusters
4. **Hard assignments**: Each point belongs to exactly one cluster

## Variants and Extensions

| Variant | Improvement |
|---------|-------------|
| **K-Medoids** | Use medoids (actual data points) instead of means |
| **Fuzzy C-Means** | Soft cluster assignments |
| **Spectral Clustering** | Handle non-convex clusters |
| **DBSCAN** | Density-based, no need to specify k |
| **Hierarchical** | Produces tree of clusters |

## Key Takeaways

1. **K-means** is simple, fast, and often effective
2. **Initialization matters**: Use k-means++ or multiple restarts
3. **Choosing k**: Use elbow method or silhouette score
4. **Assumptions**: Spherical clusters, sensitive to outliers

## Practical Notes

- **Scale features**: K-means is sensitive to feature scales
- **Use k-means++ initialization**: Standard in scikit-learn
- **Try multiple k values**: Compare using silhouette or elbow
- **scikit-learn**: `KMeans(n_clusters=k, init='k-means++', n_init=10)`

