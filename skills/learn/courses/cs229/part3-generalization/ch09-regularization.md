# Chapter 9: Regularization and Model Selection

## 9.1 Regularization

### The Problem

Complex models can fit training data perfectly but fail to generalize.

### The Solution

Add a penalty term that discourages complexity:

```
J_reg(θ) = J(θ) + λ · R(θ)
```

Where:
- J(θ) = original loss (e.g., squared error, cross-entropy)
- R(θ) = regularization term
- λ = regularization strength (hyperparameter)

### L2 Regularization (Ridge/Weight Decay)

```
R(θ) = ||θ||₂² = Σⱼ θⱼ²
```

**Effect**: Shrinks all weights towards zero.

**Closed form for linear regression**:
```
θ = (XᵀX + λI)⁻¹Xᵀy
```

**Properties**:
- Always has a solution (even when XᵀX is singular)
- Weights are small but rarely exactly zero
- Corresponds to Gaussian prior on θ

### L1 Regularization (Lasso)

```
R(θ) = ||θ||₁ = Σⱼ |θⱼ|
```

**Effect**: Encourages **sparse** solutions (many θⱼ = 0).

**Properties**:
- Performs feature selection automatically
- Corresponds to Laplace prior on θ
- Harder to optimize (non-differentiable at 0)

### Elastic Net

Combine L1 and L2:
```
R(θ) = α||θ||₁ + (1-α)||θ||₂²
```

**Benefits**: Sparsity of L1 + stability of L2.

### Comparison

| Regularization | Geometry | Sparsity | Stability |
|----------------|----------|----------|-----------|
| L2 (Ridge) | Sphere | No | High |
| L1 (Lasso) | Diamond | Yes | Lower |
| Elastic Net | Mix | Some | Medium |

### Choosing λ

Use **cross-validation**:
1. Try range of λ values (e.g., 10⁻⁴ to 10⁴)
2. For each λ, compute cross-validation error
3. Select λ with minimum CV error

## 9.2 Implicit Regularization (Optional)

### The Phenomenon

Even without explicit regularization, some algorithms find "simple" solutions.

**Gradient descent on overparameterized models**:
- Many solutions fit training data perfectly
- GD converges to **minimum norm** solution
- This is implicitly regularized!

### Why It Works

GD initialized at θ = 0 and run with small learning rate:
```
θ* = X†y = (XᵀX)†Xᵀy  (minimum norm solution)
```

**Implications**:
- Deep networks can generalize despite overparameterization
- Early stopping also provides implicit regularization
- SGD noise provides additional regularization

## 9.3 Model Selection via Cross-Validation

### The Problem

How to choose:
- Regularization strength λ?
- Model complexity (degree of polynomial, number of layers)?
- Other hyperparameters?

### Hold-Out Validation

Split data: Training (e.g., 70%) / Validation (30%)

1. Train models with different hyperparameters on training set
2. Evaluate on validation set
3. Select best hyperparameters
4. (Optional) Retrain on all data with best hyperparameters

**Problem**: Wastes data, high variance estimate.

### K-Fold Cross-Validation

```
Data: [Fold 1][Fold 2][Fold 3][Fold 4][Fold 5]
       Train    Train   Valid   Train   Train   → Error₃
       Train    Train   Train   Valid   Train   → Error₄
       ...
```

**Procedure**:
1. Split data into K folds
2. For each fold i:
   - Train on all folds except i
   - Validate on fold i
3. Average the K validation errors

**Typical K**: 5 or 10

### Leave-One-Out Cross-Validation (LOOCV)

K = n (each example is its own fold)

**Pros**: Maximum use of data, low bias

**Cons**: Expensive (n training runs), high variance

### Nested Cross-Validation

For final model evaluation:

**Outer loop**: Evaluate model selection procedure
**Inner loop**: Select hyperparameters

```
for outer_fold in outer_folds:
    test_set = outer_fold
    train_val_set = remaining folds
    
    # Inner loop: model selection
    for hyperparameter in hyperparameters:
        cv_error = k_fold_cv(train_val_set, hyperparameter)
    best_hyperparameter = argmin(cv_errors)
    
    # Evaluate on outer test fold
    model = train(train_val_set, best_hyperparameter)
    test_error = evaluate(model, test_set)
```

## 9.4 Bayesian Statistics and Regularization

### The Bayesian View

**Prior**: P(θ) - belief about θ before seeing data

**Likelihood**: P(data|θ) - probability of data given θ

**Posterior**: P(θ|data) ∝ P(data|θ) · P(θ)

### MAP Estimation

**Maximum A Posteriori** estimate:
```
θ_MAP = argmax_θ P(θ|data)
       = argmax_θ [log P(data|θ) + log P(θ)]
       = argmax_θ [ℓ(θ) + log P(θ)]
```

### Connection to Regularization

**Gaussian prior** (θⱼ ~ N(0, τ²)):
```
log P(θ) = const - (1/2τ²) Σⱼ θⱼ²
```

MAP estimation = L2-regularized maximum likelihood!
```
θ_MAP = argmin_θ [-ℓ(θ) + λ||θ||₂²]  where λ = 1/(2τ²)
```

**Laplace prior** (θⱼ ~ Laplace(0, b)):
```
log P(θ) = const - (1/b) Σⱼ |θⱼ|
```

MAP estimation = L1-regularized maximum likelihood!

### Full Bayesian Inference

Instead of point estimate, compute full posterior P(θ|data).

**Predictions** integrate over parameter uncertainty:
```
P(y|x, data) = ∫ P(y|x, θ) P(θ|data) dθ
```

**Benefits**: Uncertainty quantification, automatic regularization

**Challenge**: Integral is often intractable → approximations (MCMC, variational inference)

## Regularization in Neural Networks

### Dropout

During training, randomly drop units with probability p:
```
h_train = mask ⊙ h,  where mask ~ Bernoulli(1-p)
h_test = (1-p) · h   (or use inverted dropout)
```

**Interpretation**: Training an ensemble of sub-networks.

### Weight Decay

L2 regularization applied to neural network weights:
```
L_total = L_data + (λ/2) Σ_l ||W^[l]||_F²
```

### Batch Normalization

Normalizes activations → reduces internal covariate shift → acts as regularizer.

### Data Augmentation

Artificially expand training set:
- Images: rotation, flipping, cropping, color jitter
- Text: synonym replacement, back-translation
- Audio: speed change, noise addition

**Effect**: Explicit regularization through data.

### Early Stopping

Stop training before convergence:
```
while validation_error decreasing:
    train_one_epoch()
save best_model
```

**Effect**: Limits effective complexity of learned model.

## Key Takeaways

1. **Regularization** prevents overfitting by penalizing complexity
2. **L2** shrinks weights, **L1** creates sparsity
3. **Cross-validation** selects hyperparameters objectively
4. **Bayesian view** interprets regularization as priors
5. **Multiple techniques** combine: dropout + weight decay + augmentation

## Practical Notes

- **Start with L2**: Weight decay = 1e-4 to 1e-2
- **Try L1**: When feature selection is desired
- **Use dropout**: p = 0.5 for hidden layers, p = 0.2 for input
- **Early stopping**: Always monitor validation loss
- **Data augmentation**: Often more effective than explicit regularization

