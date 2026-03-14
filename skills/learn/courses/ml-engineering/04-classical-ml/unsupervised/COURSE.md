# Unsupervised Learning: Finding Structure Without Labels

## Learning Objectives

By the end of this lesson, you will:

**Understand:**
- How clustering algorithms (K-Means, DBSCAN, hierarchical) discover groupings and when each is appropriate
- How dimensionality reduction techniques (PCA, t-SNE, UMAP) work and what properties they preserve or sacrifice
- How anomaly detection methods (Isolation Forest, One-Class SVM, autoencoders) identify outliers through different mechanisms

**Apply:**
- Select and tune the right clustering algorithm based on cluster shape, density, and data scale
- Use PCA for preprocessing and UMAP for visualization to explore high-dimensional data

**Analyze:**
- Evaluate clustering quality without ground-truth labels using internal metrics and business validation, and decide when unsupervised methods should be combined with supervised learning

## Prerequisites

- **Linear Algebra** — eigenvalues, eigenvectors, and matrix decomposition are the mathematical foundation of PCA and underpin how dimensionality reduction preserves variance (see [Linear Algebra](../01-foundations/linear-algebra/COURSE.md))
- **Supervised Learning** — understanding supervised algorithms is necessary to appreciate how unsupervised methods complement them (e.g., clustering as preprocessing, anomaly detection as an alternative to classification) (see [Supervised Learning](./supervised/COURSE.md))

## Why This Matters

Unsupervised learning finds patterns in data without labeled examples. No one tells the model what's right — it discovers groupings, structure, and representations on its own. This is foundational for:

- **Production ML**: customer segmentation, anomaly detection, recommendation diversity
- **Preprocessing**: dimensionality reduction, feature extraction, denoising
- **Data exploration**: understanding your data before building supervised models

In large-scale e-commerce platforms, unsupervised methods power merchant segmentation, product clustering, anomaly detection in payments, embedding visualization, and cold-start recommendations. Interviewers test this because it shows you think beyond "train a classifier."

---

## 1. K-Means Clustering

**What it is:** Partition N data points into K clusters by iteratively assigning points to the nearest centroid and updating centroids to be the mean of their assigned points.

**Algorithm:**
```
1. Initialize K centroids (random or K-means++)
2. Repeat until convergence:
   a. ASSIGN: each point -> nearest centroid (Euclidean distance)
   b. UPDATE: each centroid -> mean of its assigned points
3. Converged when assignments stop changing (or max iterations)
```

**K-means++ initialization:** Instead of random centroids, pick the first randomly, then each subsequent centroid is chosen with probability proportional to distance-squared from the nearest existing centroid. This spreads initial centroids apart, dramatically improving convergence speed and solution quality. Always use it (sklearn default).

### Choosing K

**Elbow Method:**
1. Run K-means for K = 1, 2, 3, ..., 15
2. Plot inertia (total within-cluster sum of squares) vs K
3. The "elbow" where adding more clusters gives diminishing returns suggests optimal K
4. Problem: the elbow is often ambiguous — this is more art than science

**Silhouette Score:**
```
silhouette(i) = (b(i) - a(i)) / max(a(i), b(i))
```
- a(i) = average distance to points in same cluster (cohesion)
- b(i) = average distance to points in nearest other cluster (separation)
- Range: -1 (wrong cluster) to +1 (perfectly clustered)
- Pick K that maximizes average silhouette score
- More principled than the elbow method

**Gap Statistic:** Compares your clustering's inertia to the expected inertia under a null reference distribution (uniform random data). The K where the gap is largest is optimal. Most principled but computationally expensive.

### Limitations
- **Assumes spherical, equally-sized clusters**: uses Euclidean distance, so it finds round blobs. Elongated, irregular, or differently-sized clusters get split incorrectly.
- **Must specify K in advance**: requires domain knowledge or a selection method.
- **Sensitive to initialization**: K-means++ mitigates this. Also run multiple times (n_init=10 in sklearn) and pick best.
- **Sensitive to outliers**: a single outlier can pull a centroid off course. Consider removing outliers first or using K-Medoids (uses medians instead of means).
- **Sensitive to scale**: always standardize features first.

### When to Use
- Customer/merchant segmentation (group by behavior patterns)
- Vector quantization (compressing embeddings for approximate nearest neighbor search)
- Initializing more complex algorithms
- When you have a rough idea of K and expect roughly spherical clusters

---

## 2. DBSCAN (Density-Based Spatial Clustering)

**What it is:** Clusters are dense regions of points separated by sparse regions. No need to specify K — the algorithm discovers clusters automatically and labels sparse points as noise.

**Two parameters:**
- **eps (epsilon)**: radius of the neighborhood around each point
- **min_samples**: minimum points required in an eps-neighborhood to form a dense core

**Algorithm:**
```
For each unvisited point P:
  1. Find all points within distance eps of P (eps-neighborhood)
  2. If |neighborhood| >= min_samples:
     - P is a CORE point -> start a new cluster
     - Recursively add all density-reachable points to this cluster
  3. If |neighborhood| < min_samples but P is reachable from a core point:
     - P is a BORDER point -> assign to that cluster
  4. Otherwise:
     - P is NOISE -> label as -1 (outlier)
```

**Why it handles arbitrary shapes:** Because it follows density contours rather than measuring distance to a center point. DBSCAN can find crescents, rings, elongated blobs — anything that forms a continuous dense region. K-means would split these into incorrect spherical chunks.

**Choosing parameters:**
- **eps**: Use a k-distance plot. For each point, compute distance to its k-th nearest neighbor (k = min_samples). Sort and plot these distances. The "elbow" in the curve suggests eps — below this distance, points are in clusters; above, they're noise.
- **min_samples**: Rule of thumb = 2 * n_dimensions. Higher values = more conservative clustering (fewer, larger clusters). For 2D data, min_samples=5 is common.

**HDBSCAN (the modern upgrade):**
DBSCAN struggles with clusters of varying density — a single eps can't fit both tight and loose clusters. HDBSCAN solves this by running DBSCAN at all possible eps values and extracting the most stable clusters. It's almost always better than DBSCAN. Use `hdbscan` library.

### Strengths
- Finds arbitrarily shaped clusters
- Automatically detects outliers (noise points)
- No need to specify number of clusters
- Robust to outliers (labeled as noise, not forced into clusters)

### Weaknesses
- Struggles with varying-density clusters (HDBSCAN fixes this)
- Performance degrades in very high dimensions
- eps and min_samples require tuning
- Not deterministic for border points (order-dependent)

---

## 3. Hierarchical Clustering

**What it is:** Build a tree (dendrogram) of nested clusters. You can cut the tree at any level to get any number of clusters.

**Agglomerative (bottom-up) — most common:**
```
1. Start: each point is its own cluster (N clusters)
2. Find the two closest clusters and merge them
3. Repeat step 2 until one cluster remains
4. Cut the dendrogram at desired height to get K clusters
```

**Divisive (top-down):**
```
1. Start: all points in one cluster
2. Split the least cohesive cluster into two
3. Repeat until each point is its own cluster
```
Rarely used in practice — agglomerative is dominant.

**Linkage methods (how to measure distance between clusters):**
- **Single linkage**: min distance between any pair of points across clusters. Finds elongated clusters but suffers from "chaining" (linking distant clusters through a bridge of intermediate points).
- **Complete linkage**: max distance between any pair. Finds compact, roughly equal-sized clusters.
- **Average linkage**: average pairwise distance. Balanced compromise.
- **Ward's method**: merge clusters that minimize the increase in total within-cluster variance. Most commonly used. Tends to find compact, roughly equal-sized clusters.

**Dendrograms:**
The dendrogram visualizes the entire merge history. Y-axis = distance at which clusters merged. Long vertical lines = well-separated clusters. Cut where you see the biggest vertical gap.

### When to Use
- Exploring cluster structure at multiple granularities (the dendrogram is the answer)
- Small to medium datasets (< 10K points — O(n^2) memory for distance matrix)
- Building taxonomies or hierarchies (product categories, biological classification)
- When you want a deterministic result (no random initialization)

### Limitations
- O(n^2) space, O(n^3) time — doesn't scale past ~10K points
- Greedy: can't undo a merge once made
- Sensitive to noise and outliers
- Linkage choice significantly affects results

---

### Check Your Understanding

1. You run K-Means on a customer dataset and obtain clusters with a silhouette score of 0.15. What does this suggest, and what would you try next?
2. Why does DBSCAN struggle when clusters have very different densities, and how does HDBSCAN solve this?
3. A colleague uses K-Means on a dataset containing crescent-shaped clusters and gets poor results. They increase K from 2 to 6 and claim the clustering is now good because inertia decreased. What is the fundamental problem with their approach?

<details>
<summary>Answers</summary>

1. A silhouette score of 0.15 indicates poor cluster separation -- points are not clearly assigned to one cluster over another (the range is -1 to +1, with values near 0 meaning clusters overlap significantly). You should try: (a) different values of K (the current K may not match the data's structure), (b) a different algorithm like DBSCAN or HDBSCAN if clusters are non-spherical, (c) dimensionality reduction (PCA or UMAP) before clustering if the data is high-dimensional, or (d) examining whether meaningful clusters actually exist in the data.

2. DBSCAN uses a single eps (radius) for the entire dataset. If one cluster is tight (small distances between points) and another is loose (larger distances), no single eps works for both -- a small eps will fragment the loose cluster into noise, and a large eps will merge the tight cluster with its surroundings. HDBSCAN solves this by effectively running DBSCAN at all possible eps values and extracting the clusters that are most stable across a range of density thresholds, allowing it to find clusters of varying density simultaneously.

3. Inertia (within-cluster sum of squares) always decreases as K increases -- it would reach zero at K = N (one point per cluster). This does not mean the clustering is meaningful. The fundamental problem is that K-Means assumes spherical clusters and uses Euclidean distance to centroids. It will split crescent-shaped clusters into incorrect spherical chunks regardless of K. The correct fix is to use an algorithm that can handle arbitrary shapes, such as DBSCAN, HDBSCAN, or spectral clustering.

</details>

---

## 4. PCA (Principal Component Analysis)

**What it is:** Find the directions (principal components) of maximum variance in the data. Project data onto these directions to reduce dimensionality while preserving as much information as possible.

**How it works:**
1. Standardize features (zero mean, unit variance) — critical
2. Compute the covariance matrix (or use SVD directly, which is numerically more stable)
3. Find eigenvectors (directions of maximum variance) and eigenvalues (amount of variance along each direction)
4. Sort by eigenvalue descending. PC1 explains the most variance, PC2 the second most, etc.
5. Keep the top K components. Project data onto them.

**Key properties:**
- Principal components are orthogonal (uncorrelated) by construction
- Each PC is a linear combination of original features
- Total variance is preserved: sum of all eigenvalues = total variance
- Reconstruction error is minimized for any given number of components

**Variance explained:** Plot cumulative variance explained vs number of components. Common targets:
- Keep enough components for 95% of variance explained
- Pick the "elbow" in the scree plot
- With 200 engineered features, PCA might show 30-50 components capture 95% of variance — the rest is redundancy and noise

**Uses in practice:**
1. **Visualization**: project to 2-3 dimensions for plotting (quick but linear-only)
2. **Preprocessing**: reduce dimensions before KNN, SVM (cures curse of dimensionality)
3. **Noise reduction**: low-variance components are mostly noise — drop them
4. **Feature decorrelation**: PCA components are uncorrelated, which helps linear models
5. **Compression**: store and transmit data in fewer dimensions
6. **Multicollinearity removal**: PCA components are orthogonal by construction

**Limitations:**
- Linear only — can't capture non-linear structure (use kernel PCA or autoencoders)
- Components are hard to interpret: PC1 = 0.3*feature_A + 0.7*feature_B - 0.2*feature_C... what does that mean?
- Assumes variance = importance. A high-variance feature might be noise; a low-variance feature might be the most predictive.
- Must standardize first, otherwise high-magnitude features dominate

---

## 5. t-SNE and UMAP

### t-SNE (t-distributed Stochastic Neighbor Embedding)

**Purpose:** Non-linear dimensionality reduction designed specifically for visualization (2D/3D). Preserves local structure — nearby points in high-D stay nearby in the plot.

**How it works (intuition):**
1. In high-D: compute pairwise similarities using Gaussian distributions (nearby points get high similarity)
2. In low-D: compute pairwise similarities using t-distributions (heavier tails than Gaussian)
3. Minimize KL divergence between these two distributions using gradient descent
4. The heavy-tailed t-distribution prevents the "crowding problem" — it allows moderately distant points to be placed further apart in low-D

**Key parameter:** `perplexity` (5-50). Roughly = expected number of nearest neighbors to preserve. Low perplexity emphasizes very local structure; high perplexity captures more global patterns. Try multiple values.

**Critical limitations to know for interviews:**
- Stochastic — different runs produce different layouts. Always set random_state.
- Distances between clusters are meaningless. Only within-cluster structure is preserved.
- Cluster sizes in the plot don't reflect true cluster sizes.
- Slow: O(n^2) naive, O(n log n) with Barnes-Hut approximation
- Only for visualization — can't be used for preprocessing or inverse transform
- Non-parametric — can't embed new points without rerunning on entire dataset

### UMAP (Uniform Manifold Approximation and Projection)

**Purpose:** Same goal as t-SNE but faster, preserves more global structure, and can be used for general dimensionality reduction (not just visualization).

**Advantages over t-SNE:**
- 10-100x faster (scales to millions of points)
- Better preservation of global structure (cluster-to-cluster distances are more meaningful)
- Can reduce to arbitrary dimensions (not just 2-3) — use as actual preprocessing
- Supports supervised and semi-supervised modes (use labels to guide the embedding)
- Can embed new points using the learned transform
- More deterministic with fixed random seed

**Key parameters:**
- `n_neighbors`: balance local vs global structure. Low = tight local clusters. High = more global view. Similar to perplexity.
- `min_dist`: how tightly packed the embedding is. Low values = tighter, more separated clusters. High values = more spread out, preserves broader structure.
- `metric`: distance metric in the original space (euclidean, cosine, etc.)

**When to use:**
- Visualizing embeddings or high-dimensional feature spaces
- Checking if natural clusters exist before choosing a clustering algorithm
- Quality check on learned embeddings (are similar items close?)
- Dimensionality reduction for downstream tasks (UMAP to 50D -> XGBoost)
- Prefer UMAP over t-SNE for almost all cases

---

### Check Your Understanding

1. You apply PCA to a dataset with 200 features and find that the first 5 components explain 95% of the variance. Does this mean only 5 features matter? Why or why not?
2. A colleague shows you a t-SNE plot where two clusters are far apart and concludes they are very different. What is wrong with this interpretation?
3. When would you choose UMAP over PCA for dimensionality reduction as a preprocessing step before training a supervised model?

<details>
<summary>Answers</summary>

1. No. Each principal component is a linear combination of ALL original features, not a single feature. Saying 5 components explain 95% of variance means the data lies approximately on a 5-dimensional linear subspace, but each dimension is a weighted mix of the original 200 features. Many original features may contribute to those 5 components. If you need to know which individual features matter, use feature importance methods (SHAP, permutation importance) rather than PCA.

2. Distances between clusters in a t-SNE plot are meaningless -- only within-cluster local structure is preserved. Two clusters far apart in the plot may actually be close in the original space, and vice versa. Cluster sizes in t-SNE also do not reflect true cluster sizes. To assess true inter-cluster distances, use the original high-dimensional data or PCA (which preserves global distances better).

3. Choose UMAP when: (a) the data has non-linear structure that PCA (a linear method) cannot capture, (b) you have enough data to justify the computational cost, and (c) you want to reduce to more than 2-3 dimensions for use as actual features (UMAP can reduce to arbitrary dimensions, unlike t-SNE which is visualization-only). PCA is preferred when relationships are approximately linear, interpretability of components matters, or the dataset is very large and speed is critical (PCA is faster).

</details>

---

## 6. Autoencoders

**What they are:** Neural networks trained to reconstruct their input through a bottleneck. The compressed representation (bottleneck layer) captures the most important features of the data.

```
Input (d dimensions) -> Encoder -> Bottleneck (k dimensions, k << d) -> Decoder -> Reconstructed Input (d dimensions)

Loss = reconstruction_error(input, output)  typically MSE
```

**Types:**
- **Vanilla autoencoder**: simple feedforward encoder/decoder. Learns a non-linear version of PCA.
- **Variational Autoencoder (VAE)**: bottleneck is a distribution (mean + variance), not a point. Enables generation of new samples. More principled but harder to train.
- **Denoising autoencoder**: add noise to input, train to reconstruct clean input. Learns robust features.
- **Sparse autoencoder**: add L1 penalty on bottleneck activations. Forces most neurons to be inactive -> interpretable features.

**Uses:**
1. **Dimensionality reduction**: non-linear alternative to PCA. Captures complex manifold structure.
2. **Anomaly detection**: train on normal data only. Anomalies produce high reconstruction error because the autoencoder never learned to encode them. Set a threshold on reconstruction error.
3. **Feature extraction**: use the bottleneck representation as features for a downstream supervised model.
4. **Denoising**: reconstruct clean signals from noisy inputs.
5. **Pre-training**: learn representations before fine-tuning on limited labeled data.

**When to prefer over PCA:**
- When relationships between features are non-linear
- When you have enough data to train a neural network (> 10K samples)
- When you need richer, more expressive representations
- For complex data types (images, sequences)

---

## 7. Anomaly Detection

Anomaly detection identifies data points that deviate significantly from the norm. It's a critical production capability — fraud, abuse, system failures, data quality issues.

### Isolation Forest

**Core idea:** Anomalies are few and different, so random splits isolate them quickly. Normal points require many splits to isolate because they're surrounded by similar points.

```
1. Build many random trees:
   - At each node, randomly pick a feature and a random split value
   - Continue until each point is isolated (in its own leaf)
2. Anomaly score = average path length to isolate the point across all trees
3. Short path = easy to isolate = likely anomaly
4. Normalize score to [0, 1] — higher = more anomalous
```

**Why it works:** In a random partition, an outlier in a sparse region will be separated from other points in the first few splits. A normal point in a dense cluster needs many splits to be isolated. No distance computation needed — it works in high dimensions where distance-based methods fail.

**Parameters:** `n_estimators` (100-300 trees), `contamination` (expected fraction of anomalies, e.g., 0.01 for 1%). Set contamination to auto if unsure.

**Strengths:** Fast (O(n log n)), scales to millions of points, handles high dimensions, no distribution assumptions, works well out of the box.

### One-Class SVM

**Idea:** Learn a boundary around "normal" data in feature space. Anything outside the boundary is anomalous.

- Train on only normal data (no anomaly examples needed)
- Uses the kernel trick to handle non-linear boundaries in the feature space
- nu parameter controls the fraction of training data allowed to be "outliers" (essentially a soft margin)

**When to use:** Smaller datasets where you want a tighter decision boundary. Not great for large-scale (SVM scaling issues apply).

### Autoencoder-based Detection

**Idea:** Train autoencoder to reconstruct normal data well. Anomalies produce high reconstruction error.

```
1. Train autoencoder on "normal" data only
2. For each new observation, compute reconstruction error
3. If reconstruction_error > threshold: flag as anomaly
```

**Advantages:** Handles complex, high-dimensional data. Can detect subtle anomalies that simpler methods miss. The reconstruction error naturally provides a continuous anomaly score.

**Threshold selection:** Use validation data with known anomalies if available. Otherwise, use statistical methods on the reconstruction error distribution (e.g., flag observations > 3 standard deviations above mean reconstruction error).

### Statistical Methods (baselines)
- **Z-score**: flag points > 3 standard deviations from mean. Simple, assumes normality.
- **IQR method**: flag points below Q1 - 1.5*IQR or above Q3 + 1.5*IQR. More robust to non-normality.
- **Mahalanobis distance**: accounts for correlations between features. Multivariate generalization of z-score.

### Practical Decision Guide
- Start with Isolation Forest (fast, scalable, works well by default)
- For high-dimensional or complex data, try autoencoders
- For small datasets with clear normal patterns, try One-Class SVM
- For simple univariate checks, z-score or IQR

---

### Check Your Understanding

1. Isolation Forest assigns anomaly scores based on average path length. Why does a shorter path length indicate an anomaly?
2. You train an autoencoder for anomaly detection on "normal" transaction data. In production, a new type of legitimate transaction pattern appears that the autoencoder has never seen. What happens, and how would you address it?
3. When would you choose Isolation Forest over a statistical method like z-score for anomaly detection?

<details>
<summary>Answers</summary>

1. Isolation Forest builds random trees by picking random features and random split values. Anomalies are few and different from the majority -- they sit in sparse regions of the feature space. A random split is likely to isolate an anomaly from other points in just a few cuts because there are few nearby points. Normal points in dense clusters require many sequential random splits to be separated from their neighbors. Therefore, shorter path = easier to isolate = more anomalous.

2. The autoencoder would produce high reconstruction error for this new legitimate pattern (since it was never trained on it), incorrectly flagging it as anomalous. This is a fundamental limitation of anomaly detection: it conflates "novel" with "anomalous." To address this: (a) periodically retrain the autoencoder on updated normal data that includes new legitimate patterns, (b) use a human review step so flagged items are investigated rather than automatically blocked, (c) monitor the false positive rate over time to detect when the model's notion of "normal" has drifted from reality.

3. Choose Isolation Forest when: (a) data is multivariate (z-score is univariate and misses anomalies that are only unusual in combination), (b) features are not normally distributed (z-score assumes normality), (c) the dataset is large (Isolation Forest scales as O(n log n)), or (d) you want a method that works out of the box without distribution assumptions. Z-score is appropriate for quick univariate checks on approximately normal data, or as a simple first pass.

</details>

---

## 8. E-Commerce Applications

### Merchant Segmentation
Group merchants by behavior for targeted product development, marketing, and support:
- **Features**: GMV, order count, product count, industry, plan type, feature adoption, support tickets, app usage
- **Approach**: K-Means or HDBSCAN on standardized features. Validate with silhouette score. Profile each segment.
- **Output**: "High-growth merchants" (rapid GMV increase, many apps), "Struggling merchants" (declining orders, high support tickets), "Established enterprises" (stable high GMV, custom themes)
- **Business impact**: target product development and onboarding by segment

### Product Clustering
Group similar products for recommendation diversity and catalog organization:
- **Features**: product description embeddings (Sentence-BERT), price, category, images (CNN features)
- **Approach**: UMAP to reduce embedding dimensions -> HDBSCAN for variable-density clusters
- **Application**: ensure recommendation carousels show diverse product types, not 10 slightly different t-shirts

### Transaction Anomaly Detection
Flag unusual patterns in payment platform transactions:
- **Features**: transaction amount, frequency, time patterns, geographic signals, device fingerprints
- **Approach**: Isolation Forest on merchant transaction profiles. Flag merchants whose recent behavior deviates from their historical pattern.
- **Critical detail**: anomaly != fraud. Flag for review, don't auto-block. High precision matters here.

### Cold-Start Recommendations
When a new merchant has no interaction data:
- **Approach**: cluster existing merchants by profile features. For a new merchant, find the closest cluster and recommend what's popular in that cluster.
- **Transition**: as interaction data accumulates, gradually shift from content-based (cluster) to collaborative filtering recommendations.

---

## Common Pitfalls

1. **Forgetting to standardize features before K-Means or PCA.** Both algorithms rely on Euclidean distance or variance, so a feature with range 0-1,000,000 will dominate one with range 0-1. Always standardize (zero mean, unit variance) before applying distance-based or variance-based unsupervised methods.

2. **Interpreting t-SNE cluster distances as meaningful.** t-SNE preserves local neighborhood structure only. The distance between clusters, the relative sizes of clusters, and even the number of apparent clusters can be artifacts of the perplexity parameter and random initialization. Never draw conclusions about inter-cluster relationships from a t-SNE plot.

3. **Using K-Means inertia alone to evaluate clustering quality.** Inertia always decreases with more clusters and reaches zero at K=N. It tells you nothing about whether the clusters are meaningful. Use silhouette score (which measures cohesion vs separation), business validation (do the clusters make sense to domain experts?), and stability analysis (do clusters change with different random seeds or subsamples?).

4. **Treating anomaly detection output as ground truth.** Anomaly detection flags unusual points, not necessarily bad ones. A new product launch, a seasonal spike, or a legitimate high-value transaction can all be flagged as anomalies. Always route anomaly flags to a review process rather than taking automated action, and track false positive rates to calibrate thresholds.

---

## Hands-On Exercises

### Exercise 1: Clustering Comparison on Synthetic Data (20 min)

Use scikit-learn's `make_moons` and `make_blobs` to compare K-Means, DBSCAN, and HDBSCAN:

1. Generate two datasets: `make_moons(n_samples=500, noise=0.1)` and `make_blobs(n_samples=500, centers=3, cluster_std=[1.0, 2.5, 0.5])`.
2. Apply K-Means (K=2 for moons, K=3 for blobs) and DBSCAN (tune eps and min_samples) to both datasets.
3. Plot the clustering results side by side.
4. Compute silhouette scores for each method on each dataset. Which algorithm works better for which data shape, and why?

```python
from sklearn.datasets import make_moons, make_blobs
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
```

### Exercise 2: PCA Dimensionality Reduction and Visualization (15 min)

Using the scikit-learn `digits` dataset (8x8 handwritten digit images):

1. Apply PCA and plot the cumulative variance explained. How many components capture 95% of variance?
2. Project the data to 2D using PCA and plot, coloring by digit label. Which digits overlap?
3. Apply UMAP (2D) to the same data and plot. Compare the cluster separation to PCA.
4. Train a KNN classifier (K=5) on the full data, on PCA-reduced data (95% variance), and on UMAP-reduced data (2D). Compare accuracy with 5-fold cross-validation.

```python
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import umap
```

---

## 9. Interview Questions with Answers

**Q: How do you decide between K-Means and DBSCAN?**
A: K-Means when I expect roughly spherical, equally-sized clusters and have domain knowledge about K. DBSCAN when I expect arbitrary-shaped clusters, don't know K, or need built-in outlier detection. In practice, I'd start with K-Means for simplicity, then try HDBSCAN if results are poor — HDBSCAN handles varying-density clusters and requires less parameter tuning than DBSCAN.

**Q: Explain PCA to a non-technical stakeholder.**
A: Imagine you have 100 measurements for each customer. PCA finds that most of these measurements are redundant — they're telling you the same thing in slightly different ways. PCA identifies the 10-15 truly independent patterns in the data and summarizes each customer using just those patterns. You lose a little detail but gain speed and clarity.

**Q: What's the difference between PCA and t-SNE/UMAP?**
A: PCA finds the linear directions of maximum variance — fast, interpretable, good for preprocessing. t-SNE and UMAP find non-linear manifold structure — better for visualization because they preserve neighborhood relationships that PCA misses. PCA is a tool; t-SNE/UMAP are mostly for looking at things. UMAP can serve as both — it's fast enough and preserves enough global structure to use as actual preprocessing.

**Q: How would you detect anomalous merchants on an e-commerce platform?**
A: I'd build merchant behavior profiles (order patterns, login frequency, revenue trends, app usage) and use Isolation Forest to flag merchants whose profiles are statistically unusual. I'd train separate models for different merchant segments — what's normal for a high-volume enterprise is anomalous for a new small business. I'd track the anomaly score over time and alert on sudden changes, not just absolute levels. The output goes to a review queue, not automated action.

**Q: When would you use unsupervised learning as preprocessing for supervised learning?**
A: Three scenarios. (1) PCA or UMAP for dimensionality reduction before algorithms that suffer from curse of dimensionality (KNN, SVM). (2) Cluster labels as features — add cluster membership as a categorical feature for XGBoost; it captures group-level patterns the model might miss. (3) Autoencoder representations — use the bottleneck layer as a compressed feature set, especially when original features are high-dimensional embeddings.

**Q: How do you evaluate clustering when you don't have ground truth labels?**
A: Internal metrics: silhouette score (cohesion vs separation), Calinski-Harabasz index (ratio of between-cluster to within-cluster variance), Davies-Bouldin index (average similarity between clusters). External validation: do the clusters make business sense? Profile each cluster and verify with domain experts. Stability: does the clustering change drastically when you subsample the data? If clusters are fragile, they're not real structure.

**Q: What happens when you apply K-Means to high-dimensional data?**
A: It degrades because of the curse of dimensionality — in high dimensions, Euclidean distances between all pairs of points converge to similar values. The centroid (mean of many points in high-D) becomes a meaningless average that may not resemble any real data point. Solution: reduce dimensions first with PCA (to retain 95% variance) or UMAP, then cluster in the reduced space. Alternatively, use cosine similarity instead of Euclidean distance (spectral clustering or spherical K-Means).

---

## Summary

This lesson covered the major unsupervised learning techniques for discovering structure in unlabeled data:

- **Clustering:** K-Means for spherical clusters with known K, DBSCAN/HDBSCAN for arbitrary shapes and automatic outlier detection, and hierarchical clustering for exploring multi-granularity structure in small datasets.
- **Dimensionality reduction:** PCA for linear reduction and preprocessing (fast, interpretable), t-SNE for local-structure visualization, and UMAP as the modern default for both visualization and general-purpose reduction.
- **Autoencoders** provide non-linear dimensionality reduction and serve as the basis for anomaly detection on complex, high-dimensional data.
- **Anomaly detection:** Isolation Forest as the scalable default, One-Class SVM for small datasets with clear boundaries, and autoencoder-based detection for complex patterns.

The key theme is that unsupervised methods rarely stand alone in production -- they are most powerful as preprocessing for supervised models (PCA before KNN, cluster labels as features), as exploration tools (UMAP for understanding embedding quality), and as detection systems (anomaly flagging for human review).

## What's Next

- **Feature Engineering** — how to transform raw data into the features that feed both supervised and unsupervised methods, including encoding categorical variables, handling missing data, and building feature stores (see [Feature Engineering](../feature-engineering/COURSE.md))
- **Evaluation Metrics** — measuring model performance for both classification and ranking tasks, including how to evaluate clustering quality (see [Evaluation Metrics](../evaluation-metrics/COURSE.md))
