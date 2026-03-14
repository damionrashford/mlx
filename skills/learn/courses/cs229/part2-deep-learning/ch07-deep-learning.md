# Chapter 7: Deep Learning

## 7.1 Supervised Learning with Non-linear Models

### Beyond Linear Models

Linear models: h_θ(x) = θᵀx

**Limitation**: Decision boundary is always linear (or determined by fixed kernel).

### Adding Nonlinearity

**Approach 1**: Feature engineering
```
h_θ(x) = θᵀφ(x)
```
Where φ(x) is handcrafted. Problem: Requires domain expertise.

**Approach 2**: Kernel methods
```
h(x) = Σᵢ αᵢ K(x^(i), x)
```
Problem: Scales poorly with data size (O(n) per prediction).

**Approach 3**: Neural networks
```
h(x) = f(x; θ)  (parameterized nonlinear function)
```
Learn the features and classifier jointly!

### Logistic Loss for Nonlinear Models

For binary classification:
```
J(θ) = Σᵢⁿ ℓ_logistic(f(x^(i); θ), y^(i))
```

Where:
```
ℓ_logistic(t, y) = y·log(1 + e^(-t)) + (1-y)·log(1 + e^t)
```

**Challenge**: J(θ) is no longer convex! But still differentiable → gradient descent.

## 7.2 Neural Networks

### The Neuron (Single Unit)

```
         x₁ ─┐
              │──→ [Σ + b] ──→ [σ] ──→ output
         x₂ ─┘
              ↑
            weights w
```

Computation: output = σ(wᵀx + b)

### Multi-Layer Perceptron (MLP)

**Architecture**:
```
Input   Hidden Layer 1   Hidden Layer 2   Output
 x₁  ─→  [●]  [●]  ─→   [●]  [●]   ─→     ŷ
 x₂  ─→  [●]  [●]  ─→   [●]  [●]
 x₃  ─→  [●]  [●]  ─→   [●]  [●]
```

**Notation**:
- L = number of layers
- n^[l] = number of units in layer l
- W^[l] ∈ ℝ^(n^[l] × n^[l-1]) = weight matrix for layer l
- b^[l] ∈ ℝ^(n^[l]) = bias vector for layer l
- a^[l] = activation of layer l

**Forward propagation**:
```
z^[l] = W^[l] a^[l-1] + b^[l]
a^[l] = g(z^[l])
```

For the output layer (classification):
```
ŷ = softmax(z^[L])
```

### Activation Functions

| Function | Formula | Properties |
|----------|---------|------------|
| **Sigmoid** | σ(z) = 1/(1+e^(-z)) | Output ∈ (0,1), vanishing gradients |
| **Tanh** | tanh(z) = (e^z - e^(-z))/(e^z + e^(-z)) | Output ∈ (-1,1), zero-centered |
| **ReLU** | max(0, z) | Fast, sparse, no vanishing gradient for z>0 |
| **Leaky ReLU** | max(αz, z) | No "dying ReLU" problem |
| **GELU** | z·Φ(z) | Smooth ReLU, used in Transformers |

**Modern default**: ReLU for hidden layers, softmax/sigmoid for output.

### Why Depth Matters

**Universal Approximation Theorem**: A single hidden layer with enough units can approximate any continuous function.

**But**: Deep networks are exponentially more efficient! They learn hierarchical features:
- Layer 1: Edges
- Layer 2: Textures
- Layer 3: Parts
- Layer 4: Objects

### Loss Functions

**Regression**: Mean squared error
```
L = (1/n) Σᵢ (ŷ^(i) - y^(i))²
```

**Binary classification**: Binary cross-entropy
```
L = -(1/n) Σᵢ [y^(i) log ŷ^(i) + (1-y^(i)) log(1-ŷ^(i))]
```

**Multi-class classification**: Categorical cross-entropy
```
L = -(1/n) Σᵢ Σₖ y_k^(i) log ŷ_k^(i)
```

## 7.3 Modules in Modern Neural Networks

### Layer Normalization (LayerNorm)

Normalize activations across features for each example:
```
LN(z) = γ ⊙ ((z - μ) / σ) + β
```

Where μ, σ are computed per-example across features.

**Purpose**: Stabilizes training, enables deeper networks.

**Scale invariance**: LN(αz) = LN(z) for any α > 0!

### Batch Normalization (BatchNorm)

Normalize across batch for each feature:
```
BN(z) = γ ⊙ ((z - μ_batch) / σ_batch) + β
```

**Used in**: CNNs, computer vision

### Convolutional Layers (Conv)

**1D Convolution** (for sequences):
```
Conv1D(z)ᵢ = Σⱼ wⱼ · z_{i-ℓ+j-1}
```

Where w ∈ ℝᵏ is the filter (kernel).

**Properties**:
- Parameter sharing: Same filter applied across positions
- Translation equivariance: Shift input → shift output
- Efficient: O(km) vs O(m²) for full matrix

**2D Convolution** (for images):
```
Conv2D(Z)_{i,j} = Σ_{p,q} W_{p,q} · Z_{i+p, j+q}
```

**Multi-channel**: Sum convolutions across input channels, produce multiple output channels.

### Other Important Modules

| Module | Purpose |
|--------|---------|
| **Pooling** | Reduce spatial dimensions |
| **Dropout** | Regularization via random unit dropping |
| **Skip connections** | Enable gradient flow in deep networks |
| **Attention** | Dynamic weighting of inputs |

## 7.4 Backpropagation

### The Problem

Compute ∇_θ J(θ) for neural network with many layers.

**Naive approach**: Symbolic differentiation → expression explosion!

**Solution**: Backpropagation (reverse-mode automatic differentiation)

### Backpropagation Theorem (Informal)

> If a function f: ℝˡ → ℝ can be computed by a circuit of size N, then ∇f can be computed by a circuit of size O(N).

Gradient computation is **not more expensive** than forward computation!

### Chain Rule Review

If J depends on z via u = g(z):
```
∂J/∂zᵢ = Σⱼ (∂J/∂uⱼ) · (∂gⱼ/∂zᵢ)
```

**Key insight**: We can compute ∂J/∂z from ∂J/∂u using only information about g.

### Backward Functions

For each module g: z → u, define backward function B[g, z]:
```
∂J/∂z = B[g, z](∂J/∂u)
```

**Examples**:

**Matrix multiplication** (u = Wz):
```
∂J/∂z = Wᵀ · ∂J/∂u
∂J/∂W = ∂J/∂u · zᵀ
```

**Element-wise activation** (u = σ(z)):
```
∂J/∂z = σ'(z) ⊙ ∂J/∂u
```

**Softmax + Cross-entropy** (combined):
```
∂J/∂z = ŷ - y
```

### Backpropagation for MLPs

**Forward pass** (compute and cache):
```
for l = 1 to L:
    z^[l] = W^[l] a^[l-1] + b^[l]
    a^[l] = g(z^[l])
```

**Backward pass** (compute gradients):
```
δ^[L] = ∂J/∂z^[L]  (from loss function)

for l = L down to 1:
    ∂J/∂W^[l] = δ^[l] · (a^[l-1])ᵀ
    ∂J/∂b^[l] = δ^[l]
    δ^[l-1] = (W^[l])ᵀ δ^[l] ⊙ g'(z^[l-1])
```

### Automatic Differentiation

Modern frameworks (PyTorch, TensorFlow) implement backprop automatically:
1. Build computational graph during forward pass
2. Traverse graph backwards to compute gradients
3. User just defines forward computation!

## 7.5 Vectorization Over Training Examples

**Naive**: Loop over examples, accumulate gradients.

**Efficient**: Batch examples into matrices, use matrix operations.

```
Z^[l] = W^[l] A^[l-1] + b^[l]  (n×m matrix, m = batch size)
A^[l] = g(Z^[l])
```

**Benefits**:
- Leverages optimized BLAS/GPU kernels
- 10-100x faster than loops
- Essential for practical deep learning

## Key Takeaways

1. **Neural networks** learn features and classifier jointly
2. **Depth** enables hierarchical feature learning
3. **Backpropagation** computes gradients in O(forward pass) time
4. **Modern modules** (conv, norm, attention) encode useful inductive biases
5. **Vectorization** is essential for efficiency

## Practical Notes

- **Initialization**: Xavier for tanh, He for ReLU
- **Learning rate**: Start with 3e-4 for Adam, tune from there
- **Batch size**: Larger = more stable gradients, slower convergence
- **Early stopping**: Monitor validation loss, stop when it increases
- **Gradient clipping**: Prevent exploding gradients in RNNs
- **Frameworks**: PyTorch, TensorFlow, JAX handle backprop automatically

