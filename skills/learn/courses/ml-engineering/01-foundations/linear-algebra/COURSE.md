## Learning Objectives

By the end of this lesson, you will:

**Understand:**
- How vectors, matrices, and their operations (dot product, matrix multiplication) form the computational backbone of neural networks
- What eigendecomposition and SVD do and why they are the basis of dimensionality reduction (PCA) and matrix factorization (recommendation systems)
- The difference between L1 and L2 norms and how each drives different regularization behavior (sparsity vs. smoothness)

**Apply:**
- Diagnose and fix shape mismatch errors in matrix multiplications by reasoning about dimension compatibility
- Choose the appropriate similarity measure (dot product, cosine similarity, Euclidean distance) for a given ML task

**Analyze:**
- Evaluate trade-offs between L1 and L2 regularization based on whether the underlying signal is sparse or distributed

## Prerequisites

No prerequisites — this is the entry point to the course.

---

# Linear Algebra for Machine Learning

## Why This Matters

Every single thing in machine learning is a vector, a matrix, or an operation on them. When you feed an image into a neural network, it's a matrix. When the network processes it, that's matrix multiplication. When you compare two word embeddings, that's a dot product. When you reduce a 1000-dimensional dataset to something plottable, that's eigendecomposition or SVD.

You don't need to hand-compute matrix inverses. You need to *think* in linear algebra — to know what's happening when you call `model.forward()`, why your embeddings cluster the way they do, and what PCA is actually doing to your data.

---

## 1. Vectors

### What Is a Vector?

Forget the physics arrows for a moment. In ML, a vector is a list of numbers that describes something.

A house might be: `[1500, 3, 2, 1985, 0.25]` — square feet, bedrooms, bathrooms, year built, lot acres. That's a 5-dimensional vector. Each number is a *feature*.

A word embedding might be: `[0.23, -0.87, 0.12, ..., 0.45]` — a 768-dimensional vector where each dimension captures some learned aspect of meaning. You didn't design what each dimension means; the model learned it.

```
A single data point = a vector
A dataset = a collection of vectors
A model's parameters = vectors (and matrices)
```

### Geometric Intuition

In 2D, a vector is an arrow from the origin to a point. In 3D, same thing. In 768D, you can't visualize it, but the *math still works the same way*. Distances, angles, projections — they all generalize perfectly to any number of dimensions.

This is the beautiful thing about linear algebra: the intuition you build in 2D and 3D carries directly into 10,000 dimensions.

### Key Vector Operations

**Addition**: `[1, 2] + [3, 1] = [4, 3]` — combine two directions. In word embeddings, the famous `king - man + woman ≈ queen` is literally vector arithmetic.

**Scalar multiplication**: `3 * [1, 2] = [3, 6]` — stretch or shrink. Same direction, different magnitude.

**Linear combination**: `a*v1 + b*v2 + c*v3` — a weighted mix of vectors. This is fundamentally what neural networks compute at every layer. Each neuron takes a linear combination of its inputs, then applies a nonlinearity.

### Why ML Uses Vectors

| Concept | Vector Representation |
|---|---|
| A data point | Feature vector |
| A word | Word embedding (Word2Vec, BERT) |
| An image | Pixel values flattened or feature maps |
| A user's preferences | Embedding from collaborative filtering |
| A model's learned parameters | Weight vector |

Everything becomes geometry once it's a vector. Similar things are close together. Different things are far apart. This is the foundation of literally everything in ML.

### Interview-Ready Explanation

> "A vector is just an ordered list of numbers that represents a point or direction in some space. In ML, we represent data as vectors — each feature is a dimension. This lets us use geometric operations like measuring distance and angles to reason about similarity, which is the basis of most ML algorithms."

---

## 2. Matrices

### What Is a Matrix?

A matrix is a grid of numbers. But that's like saying a car is a metal box — technically true, deeply unhelpful.

A matrix is a *transformation*. It takes vectors from one space and maps them to another. When you multiply a matrix by a vector, you're transforming that vector — rotating it, scaling it, projecting it, or some combination.

```
W * x = y

- x is your input (a vector)
- W is the transformation (a matrix)
- y is the output (a new vector, possibly in a different-dimensional space)
```

### What Matrix Multiplication Actually Means

Here's the key insight: **a neural network layer is a matrix multiplication followed by a nonlinearity**.

```python
# A single neural network layer, stripped bare:
y = activation(W @ x + b)
```

`W @ x` is matrix multiplication. If `W` is a 256x784 matrix and `x` is a 784-dimensional vector (a flattened 28x28 MNIST image), then `y` is a 256-dimensional vector. The matrix *projects* the input from 784 dimensions down to 256 dimensions.

Each row of `W` is a learned "template" or "detector." The dot product of that row with the input measures how much the input matches that template. A 256x784 matrix has 256 such detectors, each looking for a different pattern in the 784-dimensional input.

### Matrix Dimensions and Shape

This is where bugs live. Get this right and half your debugging is done.

```
Matrix A: shape (m, n) — m rows, n columns
Matrix B: shape (n, p) — n rows, p columns
A @ B:    shape (m, p) — inner dimensions must match!
```

The rule: `(m, n) @ (n, p) = (m, p)`. The inner dimensions (n) must match. The outer dimensions (m, p) give you the output shape.

In PyTorch:
```python
# This works:
A = torch.randn(32, 784)   # batch of 32, each 784-dim
W = torch.randn(784, 256)  # weight matrix
out = A @ W                 # shape: (32, 256)

# This crashes:
A = torch.randn(32, 784)
W = torch.randn(512, 256)  # 512 != 784
out = A @ W                 # RuntimeError: size mismatch
```

### Special Matrices

**Identity matrix**: `I` — does nothing. `I @ x = x`. Like multiplying by 1.

**Transpose**: `A^T` — flip rows and columns. In ML, you transpose weight matrices constantly. If you go forward with `W`, you go backward (in backprop) with `W^T`.

**Inverse**: `A^(-1)` — undoes the transformation. `A^(-1) @ A = I`. Used in closed-form linear regression: `w = (X^T X)^(-1) X^T y`. But in practice, we rarely compute inverses directly — it's numerically unstable and expensive.

### Why Neural Nets Are Matrix Multiplications

A deep neural network with layers `W1, W2, W3` and activations `f` computes:

```
output = f3(W3 @ f2(W2 @ f1(W1 @ x)))
```

Without the activations, `W3 @ W2 @ W1 @ x` collapses to a single matrix — that's why nonlinearities are essential. They prevent the whole network from being equivalent to one linear transformation.

### Interview-Ready Explanation

> "A matrix represents a linear transformation — it maps vectors from one space to another. In neural networks, each layer's weights form a matrix that transforms the input. Matrix multiplication is the core computation: each output neuron computes a dot product (a weighted combination) of the inputs. The shape constraints — inner dimensions must match — are why we get shape errors when layer sizes don't align."

---

### Check Your Understanding

1. A weight matrix `W` has shape `(512, 768)` and you need to multiply it with an input batch `X` of shape `(32, 768)`. Should you compute `X @ W` or `W @ X`? What is the output shape?
2. A 3-layer neural network uses weight matrices `W1 (256, 784)`, `W2 (128, 256)`, `W3 (10, 128)`. Without any activations, what single matrix would this be equivalent to, and what would its shape be?
3. Your colleague says "adding more rows to the weight matrix increases the model's input dimensionality." Is this correct?

<details>
<summary>Answers</summary>

1. You should compute `X @ W.T` (or equivalently, `(W @ X.T).T`). Since `X` is `(32, 768)` and `W` is `(512, 768)`, `X @ W.T` gives `(32, 768) @ (768, 512) = (32, 512)`. Alternatively, you could use `X @ W` if `W` were `(768, 512)`. The key is the inner dimensions must match.
2. Without activations, `W3 @ W2 @ W1` collapses to a single `(10, 784)` matrix. This is why nonlinear activations are essential — without them, any depth of linear layers is equivalent to a single linear transformation.
3. No. Adding more rows increases the *output* dimensionality (more detectors/neurons). The number of *columns* must match the input dimensionality.

</details>

---

## 3. Dot Product

### What Is the Dot Product?

Take two vectors of the same length. Multiply corresponding elements. Sum them up.

```
a = [1, 2, 3]
b = [4, 5, 6]
a · b = 1*4 + 2*5 + 3*6 = 32
```

That's it mechanically. But what does it *mean*?

### Geometric Meaning: Similarity

The dot product measures how much two vectors point in the same direction.

```
a · b = |a| * |b| * cos(θ)
```

- If they point the same way: `cos(0°) = 1`, so `a · b` is large and positive.
- If they're perpendicular: `cos(90°) = 0`, so `a · b = 0`.
- If they point opposite ways: `cos(180°) = -1`, so `a · b` is large and negative.

This is why the dot product is the universal similarity measure in ML.

### Cosine Similarity

Normalize by the magnitudes and you get cosine similarity:

```
cos_sim(a, b) = (a · b) / (|a| * |b|)
```

This strips out magnitude and keeps only the directional similarity. A value of 1 means identical direction, 0 means unrelated, -1 means opposite.

**Where you see this everywhere:**

- **Search/retrieval**: Embed query and documents, rank by cosine similarity
- **Recommendation**: User embedding dotted with item embedding = predicted preference
- **Attention mechanism**: Query dotted with Key gives attention scores
- **Contrastive learning**: CLIP matches images and text via dot product similarity

### The Dot Product IS Attention

The attention mechanism in Transformers is fundamentally dot products:

```
Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
```

`Q @ K^T` is a matrix of dot products — every query dotted with every key. High dot product = high attention = "these tokens are relevant to each other."

### Interview-Ready Explanation

> "The dot product measures the similarity between two vectors — geometrically, it's the product of their magnitudes times the cosine of the angle between them. In ML, it's everywhere: cosine similarity for embeddings, the core of the attention mechanism in transformers, and the fundamental operation in each neuron (a dot product of weights and inputs). When two embeddings have a high dot product, it means they represent similar concepts."

---

## 4. Eigenvalues and Eigenvectors

### What Are They, Intuitively?

Imagine a transformation (a matrix) that warps space — stretching, rotating, shearing. Most vectors change direction when you apply the transformation. But some special vectors only get *scaled* — they keep pointing the same way, just longer or shorter. Those special vectors are **eigenvectors**, and the scaling factors are the **eigenvalues**.

```
A @ v = λ * v

v is an eigenvector of A
λ (lambda) is the corresponding eigenvalue
```

The transformation `A` doesn't rotate `v` — it just stretches it by `λ`.

### Why They Matter for ML

**PCA (Principal Component Analysis)**: When you compute PCA, you're finding the eigenvectors of the covariance matrix of your data. The eigenvector with the largest eigenvalue points in the direction of maximum variance — that's the first principal component. The second largest eigenvalue gives the second direction, orthogonal to the first, and so on.

In other words: eigenvectors tell you the "natural axes" of your data. Instead of thinking in terms of the original features (height, weight, age...), you think in terms of the directions where your data varies the most.

**Spectral analysis of graphs**: Google's original PageRank was an eigenvector computation. Graph neural networks use spectral methods rooted in eigendecomposition.

**Understanding model behavior**: The eigenvalues of the Hessian (second-derivative matrix) of a loss function tell you about the curvature of the loss landscape — are you in a flat valley (small eigenvalues, easy optimization) or a narrow ravine (mixed eigenvalues, hard optimization)?

### A Practical Example

You have a 1000-feature dataset. Most features are correlated. PCA:

1. Compute covariance matrix (1000x1000)
2. Find its eigenvalues and eigenvectors
3. Sort by eigenvalue (largest first)
4. Keep top k eigenvectors (maybe 50 explain 95% of variance)
5. Project data onto those 50 directions

You went from 1000 dimensions to 50 while keeping 95% of the information. The eigenvalues told you *how much information* each direction carries.

### Interview-Ready Explanation

> "An eigenvector of a matrix is a direction that doesn't change when the transformation is applied — it only gets scaled by the eigenvalue. In ML, this matters most in PCA: the eigenvectors of the data's covariance matrix are the principal components (directions of maximum variance), and the eigenvalues tell you how much variance each component explains. This lets us reduce dimensionality by keeping only the directions that carry the most information."

---

### Check Your Understanding

1. You run PCA on a 500-feature dataset and find that the first 3 eigenvalues are 150, 100, 50, while the remaining 497 eigenvalues sum to 100. How much variance do the top 3 principal components explain, and would you consider this a good candidate for dimensionality reduction?
2. The eigenvalues of the Hessian at a critical point are all positive except one, which is negative. Is this a local minimum, local maximum, or saddle point?
3. A colleague applies PCA and keeps all components. They claim "PCA improved my model because it decorrelated the features." Does PCA change the information content in this case?

<details>
<summary>Answers</summary>

1. Total variance = 150 + 100 + 50 + 100 = 400. Top 3 explain (150+100+50)/400 = 75%. This is a reasonable candidate for reduction — you can capture 75% of variance with only 3 of 500 dimensions. Whether 75% is sufficient depends on the task.
2. This is a saddle point. A local minimum requires ALL eigenvalues to be non-negative (positive semi-definite Hessian). One negative eigenvalue means there is a direction along which the function curves downward.
3. No, PCA with all components preserved does not change the information content — it is a lossless rotation of the coordinate system. The features become uncorrelated (orthogonal axes), which can help some algorithms (e.g., those sensitive to feature correlation), but no information is gained or lost.

</details>

---

## 5. Singular Value Decomposition (SVD)

### What Is SVD?

Any matrix `M` can be decomposed into three matrices:

```
M = U @ S @ V^T

- U: left singular vectors (orthogonal matrix)
- S: diagonal matrix of singular values (non-negative, sorted descending)
- V^T: right singular vectors (orthogonal matrix)
```

Think of it as breaking any transformation into three steps: rotate, scale along axes, rotate again.

### Why SVD Matters

**Dimensionality reduction**: Keep only the top `k` singular values and their corresponding vectors. This gives you the best rank-k approximation of the original matrix (in terms of Frobenius norm). This is what truncated SVD does, and it's why PCA works — PCA on centered data is equivalent to SVD.

**Recommendation systems**: The Netflix Prize was famously won with matrix factorization. You have a users-by-movies matrix (mostly empty). SVD-like decompositions factor it into user-embeddings times movie-embeddings, letting you predict missing entries.

```
Ratings ≈ Users_matrix @ Sigma @ Movies_matrix^T
```

Each user gets a low-dimensional vector. Each movie gets a low-dimensional vector. Their dot product predicts the rating.

**Data compression**: An image is a matrix. SVD with k=50 components might capture 90% of the image quality with a fraction of the storage.

**NLP**: Latent Semantic Analysis (LSA) applies SVD to term-document matrices to find latent topics. This was a precursor to modern word embeddings.

### Relationship to Eigendecomposition

- Eigendecomposition works on square matrices: `A = Q @ Lambda @ Q^(-1)`
- SVD works on *any* matrix (even non-square)
- The singular values of `M` are the square roots of the eigenvalues of `M^T @ M`

### Interview-Ready Explanation

> "SVD decomposes any matrix into rotation-scale-rotation components. The singular values tell you the importance of each component, so you can truncate to get a low-rank approximation. This is the foundation of recommendation systems (matrix factorization), dimensionality reduction (it's what PCA does under the hood), and data compression. When someone says they're doing 'matrix factorization,' they're usually doing something SVD-related."

---

## 6. Norms

### What Is a Norm?

A norm measures the "size" or "length" of a vector. Different norms measure size differently.

### L2 Norm (Euclidean Norm)

```
||x||_2 = sqrt(x1^2 + x2^2 + ... + xn^2)
```

This is the straight-line distance from the origin. It's what you think of as "length."

**Where it appears:**
- **L2 regularization (Ridge, weight decay)**: Penalizes `||w||_2^2`. This adds `lambda * sum(w_i^2)` to the loss. It pushes weights toward zero but never exactly to zero. It prevents any single weight from getting too large.
- **Euclidean distance**: `||a - b||_2` is the distance between two points. k-NN uses this.
- **Gradient norm clipping**: Cap `||gradient||_2` to prevent exploding gradients.

### L1 Norm (Manhattan Norm)

```
||x||_1 = |x1| + |x2| + ... + |xn|
```

Sum of absolute values. Like walking along a city grid (Manhattan distance).

**Where it appears:**
- **L1 regularization (Lasso)**: Penalizes `||w||_1 = sum(|w_i|)`. This pushes weights *exactly* to zero, producing sparse models. It's automatic feature selection — irrelevant features get weight 0.
- **Mean Absolute Error (MAE)**: `||predictions - targets||_1 / n`

### L1 vs L2: The Key Difference

| Property | L1 (Lasso) | L2 (Ridge) |
|---|---|---|
| Pushes weights to zero? | Yes, exactly zero | No, just small |
| Feature selection? | Built-in | No |
| Solution uniqueness | May not be unique | Always unique |
| Geometric shape | Diamond (corners on axes) | Circle (smooth) |
| When to use | Sparse signal, many irrelevant features | Many small contributions |

The geometric intuition: L1's diamond shape has corners on the axes, so the optimal point is more likely to land exactly on an axis (some weight = 0). L2's circle is smooth, so the optimal point rarely hits an axis exactly.

### L0 "Norm" (Not Actually a Norm)

`||x||_0` = count of nonzero elements. The ideal for sparsity but non-convex (NP-hard to optimize). L1 is the best convex relaxation of L0, which is why we use it.

### Interview-Ready Explanation

> "Norms measure vector magnitude. L2 (Euclidean) penalizes large weights quadratically — used in Ridge/weight decay to keep weights small but nonzero. L1 (Manhattan) penalizes linearly — used in Lasso to drive weights exactly to zero, effectively selecting features. The choice between L1 and L2 regularization depends on whether you expect your signal to be sparse (few important features) or distributed (many small contributions)."

---

### Check Your Understanding

1. You have a regression model with 1000 features, but you suspect only about 20 are truly relevant. Should you use L1 or L2 regularization, and why?
2. Two embedding vectors have a high Euclidean distance but a cosine similarity of 0.98. What does this tell you about the vectors, and which metric would you trust for measuring semantic similarity?
3. Why is the L0 "norm" (count of nonzero elements) not used directly as a regularizer in practice, even though it most directly measures sparsity?

<details>
<summary>Answers</summary>

1. Use L1 (Lasso). When only ~20 of 1000 features are relevant, you want the model to automatically zero out the irrelevant 980. L1's diamond-shaped constraint region has corners on the axes, making it likely to drive weights to exactly zero. L2 would shrink all 1000 weights but keep them all nonzero.
2. The vectors point in nearly the same direction (cosine similarity 0.98) but have very different magnitudes (high Euclidean distance). For semantic similarity, cosine similarity is more appropriate because it captures directional agreement regardless of magnitude — a longer document and a shorter document about the same topic should be considered similar.
3. The L0 "norm" is non-convex and discontinuous, making it NP-hard to optimize. There is no gradient information — the count of nonzero elements does not change smoothly as weights change. L1 is used as the best convex relaxation (approximation) of L0.

</details>

---

## 7. Putting It All Together

### The ML Pipeline Through a Linear Algebra Lens

1. **Data arrives** as a matrix `X` of shape `(n_samples, n_features)` — each row is a data point (vector)
2. **Preprocessing** might include PCA (eigendecomposition/SVD) to reduce dimensions
3. **The model** is a series of matrix multiplications and nonlinearities (neural net) or a single matrix solve (linear regression)
4. **Loss computation** involves norms (MSE uses L2, MAE uses L1)
5. **Regularization** adds norm penalties to prevent overfitting
6. **Similarity search** at inference uses dot products or cosine similarity
7. **Backpropagation** propagates gradients using transposed weight matrices

### Common Interview Questions

**Q: "What happens when you multiply a matrix by a vector?"**
A: You get a new vector where each element is the dot product of a row of the matrix with the input vector. Geometrically, you're transforming the vector — projecting it into a new space. In a neural net, each output neuron computes one of these dot products.

**Q: "Explain PCA."**
A: PCA finds the directions of maximum variance in your data by computing eigenvectors of the covariance matrix. You keep the top-k directions (those with highest eigenvalues) as a low-dimensional representation. It's equivalent to truncated SVD on centered data.

**Q: "Why do we use cosine similarity instead of Euclidean distance for embeddings?"**
A: Cosine similarity cares only about direction, not magnitude. Two documents about the same topic might have different lengths (magnitudes), but their direction in embedding space should be similar. Cosine similarity captures "what the vector is about" regardless of "how much."

**Q: "What's the difference between L1 and L2 regularization?"**
A: L2 shrinks all weights toward zero proportionally — good when all features contribute. L1 can push weights to exactly zero — good for feature selection when many features are irrelevant. L1 gives sparse models; L2 gives small-weight models.

---

## Common Pitfalls

**Pitfall 1: Transposing the Wrong Matrix in Multiplication**
- Symptom: `RuntimeError: size mismatch` or silently wrong results when dimensions happen to align by accident
- Why: Confusing `X @ W` with `X @ W.T` or mixing up which matrix needs transposing. In a layer `y = Wx + b`, `W` is `(output_dim, input_dim)`, so you compute `W @ x` — but in batch mode with `X` as `(batch, input_dim)`, you need `X @ W.T`.
- Fix: Always write out the shapes explicitly. Use the rule `(m, n) @ (n, p) = (m, p)` and verify inner dimensions match before running code.

**Pitfall 2: Confusing Cosine Similarity with Dot Product**
- Symptom: Similarity scores that are not between -1 and 1, or rankings that change when embeddings are scaled
- Why: The raw dot product is affected by vector magnitude. Two identical-direction vectors with different lengths will have a different dot product than two identical-direction vectors with the same length.
- Fix: Normalize vectors to unit length before computing dot products, or explicitly use `cosine_similarity`. If magnitudes are meaningful (e.g., confidence), use dot product intentionally.

**Pitfall 3: Applying PCA Without Centering the Data**
- Symptom: The first principal component captures the mean offset rather than the direction of maximum variance
- Why: PCA assumes the data is centered (mean-subtracted). If the data has a large non-zero mean, the covariance matrix is dominated by that offset.
- Fix: Always subtract the mean before computing PCA. Most library implementations (e.g., `sklearn.decomposition.PCA`) do this automatically, but raw eigendecomposition of an uncentered covariance matrix will give wrong results.

**Pitfall 4: Assuming L2 Regularization Produces Sparse Models**
- Symptom: Expecting weights to be exactly zero after L2 regularization, then finding all weights are small but nonzero
- Why: L2 penalty gradient is proportional to the weight value (`2 * lambda * w`), which shrinks toward zero but never reaches it. Only L1 can drive weights to exact zero.
- Fix: Use L1 regularization (Lasso) or Elastic Net (L1 + L2) when you need true sparsity and feature selection.

---

## Hands-On Exercises

### Exercise 1: Matrix Dimensions and Neural Network Layers
**Goal:** Build intuition for how matrix shapes flow through a neural network.
**Task:**
1. Using NumPy, create random weight matrices for a 3-layer network: input dim 784, hidden1 dim 256, hidden2 dim 128, output dim 10.
2. Create a random batch of 32 input vectors of dimension 784.
3. Manually compute the forward pass using `@` (matrix multiply) and `np.maximum(0, x)` for ReLU.
4. Print the shape at every stage and verify they match your predictions.
5. Intentionally swap two weight matrices and observe the error message.
**Verify:** The final output should have shape `(32, 10)`. Each intermediate shape should follow from the `(m, n) @ (n, p) = (m, p)` rule.

### Exercise 2: PCA from Scratch
**Goal:** Understand what PCA does under the hood by implementing it without sklearn.
**Task:**
1. Generate a 2D dataset with clear correlation: `x1 = np.random.randn(200)`, `x2 = 0.8 * x1 + 0.2 * np.random.randn(200)`.
2. Center the data (subtract the mean of each feature).
3. Compute the covariance matrix using `np.cov`.
4. Compute eigenvalues and eigenvectors using `np.linalg.eigh`.
5. Project the data onto the first principal component.
6. Compare your result with `sklearn.decomposition.PCA(n_components=1)`.
**Verify:** Your projected values should match sklearn's output (up to a possible sign flip, which is expected since eigenvectors can point in either direction).

---

## Key Takeaways

1. **Vectors are data.** Everything in ML gets represented as vectors.
2. **Matrices are transformations.** Neural net layers are matrix multiplications.
3. **Dot products measure similarity.** This powers attention, recommendations, and search.
4. **Eigendecomposition reveals structure.** PCA uses it to find important directions.
5. **SVD works on anything.** The Swiss Army knife of matrix decomposition.
6. **Norms measure and constrain.** L1 for sparsity, L2 for smoothness.

You don't need to compute these by hand. You need to know what they *do*, when to *use* them, and what goes wrong when you *ignore* them.

---

## Summary

Linear algebra provides the computational language of machine learning: data is vectors, models are matrices, and learning is optimization over matrix operations. The single most important takeaway is that every neural network layer is a matrix multiplication followed by a nonlinearity, and nearly every ML concept — from similarity search to dimensionality reduction to regularization — reduces to an operation on vectors and matrices.

## What's Next

- **Next lesson:** [Calculus for Machine Learning](../calculus/COURSE.md) — covers derivatives, the chain rule (which IS backpropagation), and gradients, building on the vector and matrix foundations from this lesson
- **Builds on this:** [Optimization for Machine Learning](../optimization/COURSE.md) — uses gradients and matrix operations to train models via gradient descent and its variants
- **Deep dive:** [Neural Network Fundamentals](../../02-neural-networks/fundamentals/COURSE.md) — applies matrix multiplications and activation functions to build and train actual neural networks
