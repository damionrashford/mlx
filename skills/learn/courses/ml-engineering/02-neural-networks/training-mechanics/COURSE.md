## Learning Objectives

By the end of this lesson, you will:

**Understand:**
- How weight initialization (Xavier, He, transformer-specific) preserves activation variance and prevents symmetry breaking, and why different activations require different schemes
- How optimizers (SGD, Adam, AdamW) translate gradients into weight updates, why AdamW decouples weight decay from adaptive learning rates, and how learning rate schedules (warmup + cosine) stabilize training
- How regularization techniques (dropout, weight decay, data augmentation, label smoothing) combat overfitting through different mechanisms, and when to use each

**Apply:**
- Configure a complete training pipeline with appropriate initialization, optimizer, learning rate schedule, mixed precision, gradient clipping, and early stopping
- Diagnose training failures from training curves and gradient norms, and apply the "overfit a tiny batch" debugging technique

**Analyze:**
- Given a set of training curves, distinguish between underfitting, overfitting, learning rate issues, and numerical instability, and prescribe the correct fix for each scenario

## Prerequisites

- **Neural network architecture basics** -- you need to understand layers, activations, forward/backward passes, and the training loop before learning how to tune them (see [Fundamentals](../fundamentals/COURSE.md))
- **Learning rates and regularization concepts** -- this lesson goes deep into optimization practice, building on the mathematical foundations of gradient descent and loss landscapes (see [Optimization](../../01-math-foundations/optimization/COURSE.md))

---

# Training Mechanics: How to Actually Train Neural Networks

## Why This Lesson Matters

Architecture is only half the story. You can have the perfect model design and still fail completely if you do not know how to train it. Training mechanics is the practical craft — the difference between a model that converges beautifully and one that explodes, stalls, or memorizes the training data. This lesson covers everything you need to go from "I have a model and data" to "I have a trained model that generalizes."

---

## 1. Weight Initialization: Where Training Begins

### Why Initialization Matters

If all weights start at zero, every neuron computes the same thing. Gradients are identical, updates are identical, and the network is stuck in permanent symmetry — it can never learn different features. This is called the **symmetry breaking problem**, and it is why random initialization is necessary.

But not just any random initialization will do:
- If weights start **too large**, activations explode through the layers (especially with ReLU), and the network is numerically unstable from step one
- If weights start **too small**, activations shrink through the layers, gradients vanish, and the network learns nothing

**Good initialization** preserves the **variance of activations** across layers. If the variance stays roughly constant from layer to layer, the network is in a healthy regime where gradients flow and learning can happen.

### Xavier / Glorot Initialization (2010)

Designed for layers with **sigmoid or tanh** activation:

```python
# Uniform variant:
nn.init.xavier_uniform_(layer.weight)
# Samples from Uniform(-a, a) where a = sqrt(6 / (fan_in + fan_out))

# Normal variant:
nn.init.xavier_normal_(layer.weight)
# Samples from Normal(0, sqrt(2 / (fan_in + fan_out)))
```

**The math:** Consider a layer `y = W*x` where x has fan_in dimensions and y has fan_out dimensions. If we want `Var(y) = Var(x)` (preserve variance in forward pass) AND `Var(grad_x) = Var(grad_y)` (preserve variance in backward pass), the required weight variance is:

```
Var(W) = 2 / (fan_in + fan_out)
```

This balances the forward and backward variance preservation by averaging the two constraints.

**Use for:** Sigmoid, tanh, and linear activations. The assumption is that the activation function is approximately linear around zero (which sigmoid and tanh are, with derivative close to 1 near zero).

### He / Kaiming Initialization (2015)

Designed for **ReLU** activation, which zeroes out roughly half the outputs:

```python
# Normal variant (most common):
nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
# Samples from Normal(0, sqrt(2 / fan_in))

# Uniform variant:
nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
# Samples from Uniform(-sqrt(6/fan_in), sqrt(6/fan_in))
```

**Why different from Xavier?** ReLU sets roughly half the activations to zero. This halves the variance at each layer. He initialization compensates by doubling the variance of the weights (the factor of 2 in the numerator instead of Xavier's 2/(fan_in + fan_out)).

**Use for:** ReLU and its variants (Leaky ReLU, PReLU, ELU). This is the **default initialization for most CNN and feedforward models**. PyTorch uses Kaiming uniform as the default for `nn.Linear` and `nn.Conv2d`.

### Initialization for Transformers

Modern transformers typically use custom initialization schemes:

```python
# GPT-style: small normal distribution for most weights
nn.init.normal_(module.weight, mean=0.0, std=0.02)

# Residual scaling: weights in output projections are scaled down
# to prevent residual contributions from growing with depth
nn.init.normal_(residual_proj.weight, mean=0.0, std=0.02 / sqrt(2 * n_layers))

# Embeddings: normal with small std
nn.init.normal_(embed.weight, mean=0.0, std=0.02)

# Biases: initialized to zero
nn.init.zeros_(module.bias)
```

The `1/sqrt(2 * n_layers)` scaling for residual projections is important: without it, the cumulative sum of residual contributions grows with sqrt(n_layers), making training unstable for deep models.

### The Practical Rule

| Activation | Initialization | PyTorch Default |
|-----------|---------------|-----------------|
| ReLU, Leaky ReLU | He / Kaiming | `kaiming_uniform_` (default for nn.Linear) |
| Sigmoid, Tanh | Xavier / Glorot | `xavier_uniform_` |
| GELU, SiLU (Transformers) | Small normal (0.02) | Custom |
| LSTM forget gate bias | Initialize to 1.0 | Manual |

**In practice, PyTorch's defaults are usually fine** for standard architectures. You only need to think about initialization when training very deep networks from scratch, seeing training instability, or implementing a custom architecture.

---

## 2. Learning Rate

The learning rate (LR) is the **single most important hyperparameter** in deep learning. It controls how large of a step the optimizer takes in the direction of the gradient.

```
W_new = W_old - learning_rate * gradient
```

### Too High vs Too Low

- **Too high:** Loss oscillates wildly, spikes, or goes to NaN. The model overshoots good parameter values and bounces around.
- **Too low:** Loss decreases agonizingly slowly. Training takes forever and may get stuck in poor local minima. You waste compute.
- **Just right:** Loss decreases steadily, with small oscillations that gradually smooth out.

### Why 3e-4 Is the Default for Adam

Adam's adaptive learning rates normalize the gradient by its second moment (running average of squared gradients). This effectively puts the step size in a "natural" scale. Empirically, 3e-4 (0.0003) works well as a starting point for most problems with Adam because:
- It is small enough to avoid divergence in most architectures
- It is large enough for reasonable convergence speed
- Adam's per-parameter adaptive scaling compensates for parameters at different scales

**But 3e-4 is just a starting point.** Key adjustments:

| Scenario | Typical LR |
|----------|-----------|
| Training transformer from scratch | 1e-4 to 6e-4 |
| Fine-tuning pretrained LLM | 1e-5 to 5e-5 |
| Fine-tuning pretrained CNN | 1e-4 to 1e-3 |
| SGD with momentum (ImageNet) | 0.1 (much larger — SGD needs it) |
| Very large model (70B+) | 1e-4 to 3e-4 |

### Learning Rate Warmup

Start from a very small LR (e.g., 1e-7) and linearly increase to the target LR over the first N steps:

```python
if step < warmup_steps:
    lr = max_lr * (step / warmup_steps)
```

**Why warmup helps:** In the first few steps, the model's predictions are essentially random, producing large, noisy gradients. A high learning rate applied to these noisy gradients causes the model to make large, poorly-directed updates that can permanently destabilize training (push the model into a region of the loss landscape from which it never recovers).

Warmup lets the model find a reasonable region before ramping up. Standard practice: 1-10% of total steps (e.g., 2000 steps for a 100K-step run).

### Learning Rate Schedules

A constant learning rate is almost never optimal. The learning rate should be high early (make progress fast) and low late (fine-tune the solution).

#### Cosine Annealing — The Modern Default

```python
lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(pi * step / total_steps))
```

Smooth decay from max_lr to min_lr following a cosine curve. Spends more time at lower LRs near the end of training.

```
LR
^
|     /\
|    /  \_
|   /     \_
|  /        \_
| /           \_
|/              \_______
+---+---+---+---+---+--> Steps
^   ^               ^
|   warmup          total_steps
start
```

#### Step Decay

Multiply LR by a factor (e.g., 0.1) at specific milestones:

```python
# Classic ImageNet recipe:
# LR = 0.1, decay by 10x at epochs 30, 60, 90
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
```

Common in CNN training but less common for transformers.

#### Linear Decay

```python
lr = max_lr * (1 - step / total_steps)
```

Simple and effective. Used in some LLM pretraining runs.

### The Warmup + Cosine Combination

This is the standard for transformer training:
1. Linear warmup from ~0 to max_lr over first 1-10% of steps
2. Cosine decay from max_lr to min_lr (typically max_lr/10 or max_lr/100) over remaining steps

### Learning Rate Finder (Smith, 2017)

A practical technique to find a good learning rate empirically:

1. Start with a very small LR (e.g., 1e-8)
2. Increase exponentially over one epoch
3. Plot loss vs learning rate
4. **Choose the LR where loss is decreasing fastest** (steepest downward slope)
5. This is typically one order of magnitude below where loss starts increasing

The LR finder takes minutes and can save hours of trial and error.

### LR Scaling with Batch Size

When you increase batch size, the gradient estimate becomes more accurate (less noisy), so you can afford larger steps:

```
Linear scaling rule: LR_new = LR_base * (batch_size_new / batch_size_base)
```

This rule works up to a point. At very large batch sizes (thousands), the noise reduction hurts generalization and more sophisticated techniques are needed (LARS/LAMB optimizers, additional warmup).

---

## 3. Optimizers: How Gradients Become Weight Updates

### SGD (Stochastic Gradient Descent)

The simplest optimizer:

```python
W = W - lr * gradient
```

**With momentum** (almost always used):
```python
v = momentum * v + gradient        # Accumulate velocity
W = W - lr * v                      # Step with accumulated velocity
# Default momentum = 0.9
```

Momentum smooths out noisy gradients by maintaining a running average. The optimizer builds up speed in directions with consistent gradients (like a ball rolling downhill) and dampens oscillations in directions with inconsistent gradients.

**When to use:** CNNs trained on ImageNet (the classic recipe). SGD with momentum finds flatter minima than Adam, which can mean better generalization in some settings. But it requires more careful LR tuning.

### Adam (Adaptive Moment Estimation)

The default optimizer for most deep learning:

```python
# First moment (mean of gradients — momentum):
m = beta1 * m + (1 - beta1) * gradient

# Second moment (mean of squared gradients — adaptive LR):
v = beta2 * v + (1 - beta2) * gradient^2

# Bias-corrected estimates (compensate for initialization at zero):
m_hat = m / (1 - beta1^t)
v_hat = v / (1 - beta2^t)

# Update:
W = W - lr * m_hat / (sqrt(v_hat) + eps)
```

**Why Adam works so well:**
- **Adaptive per-parameter LR:** The `sqrt(v_hat)` term normalizes each parameter's gradient by its historical magnitude. Parameters with consistently large gradients get smaller effective LRs; parameters with small gradients get larger effective LRs.
- **Momentum:** The first moment `m` provides momentum-like smoothing of gradients over time.
- **Bias correction:** Without it, the initial estimates are biased toward zero because m and v are initialized to zero. Correction compensates for this.

**Default hyperparameters:** `beta1=0.9, beta2=0.999, eps=1e-8`. These are almost always fine. Do not tune them unless you have a very specific reason.

### AdamW (Adam with Decoupled Weight Decay)

The standard optimizer for transformer training. Fixes a subtle but important issue with how Adam handles weight decay:

```python
# Adam with L2 regularization (WRONG way):
gradient += weight_decay * W    # Decay is part of the gradient
m, v = update_moments(gradient)  # Adaptive LR scales the decay term
W = W - lr * m_hat / sqrt(v_hat) # Decay is scaled by adaptive rates

# AdamW (CORRECT way):
m, v = update_moments(gradient)           # Gradient only, no decay
W = W - lr * m_hat / sqrt(v_hat)          # Adam update
W = W - lr * weight_decay * W             # Decay applied SEPARATELY
```

**Why decoupled matters:** In standard Adam, the adaptive learning rate scales the weight decay differently for each parameter. Parameters with large gradient history get less weight decay (because `sqrt(v_hat)` is large). This weakens the regularization effect inconsistently. AdamW applies weight decay uniformly to all parameters, regardless of their gradient history.

**AdamW is the standard for training transformers.** Use it unless you have a specific reason not to.

### Optimizer Selection Guide

| Task | Optimizer | LR | Notes |
|------|----------|-----|-------|
| Transformer from scratch | AdamW | 3e-4 | warmup + cosine |
| Fine-tuning pretrained LLM | AdamW | 1e-5 to 5e-5 | Low LR to avoid catastrophic forgetting |
| CNN on ImageNet | SGD + momentum (0.9) | 0.1 | Step decay at 30/60/90 epochs |
| Fine-tuning pretrained CNN | AdamW | 1e-4 to 1e-3 | Differential LR for layers |
| Quick experiments | Adam | 3e-4 | Fast and usually good enough |

---

### Check Your Understanding

1. You are fine-tuning a pretrained BERT model and a colleague suggests using SGD with a learning rate of 0.1 (the classic ImageNet recipe). Why is this likely to fail, and what optimizer and learning rate would you recommend?
2. Explain why AdamW applies weight decay separately from the gradient update. What goes wrong when weight decay is coupled with Adam's adaptive learning rates?
3. Why does learning rate warmup help during the first few hundred steps of training? What would happen if you started at the full target learning rate immediately?

<details>
<summary>Answers</summary>

1. SGD at 0.1 would likely destroy the pretrained weights because the learning rate is far too high for fine-tuning (pretrained weights are already in a good region). Use AdamW with a learning rate of 1e-5 to 5e-5. Adam's adaptive per-parameter learning rates are also better suited for fine-tuning, where different layers need different effective step sizes.
2. In standard Adam with L2 regularization, the weight decay term becomes part of the gradient, which gets scaled by Adam's adaptive learning rate (divided by sqrt(v_hat)). Parameters with large gradient history receive less weight decay, inconsistently weakening regularization. AdamW applies weight decay directly to weights after the Adam update, ensuring uniform regularization regardless of gradient magnitudes.
3. In the first few steps, model predictions are essentially random, producing large, noisy gradients. A high learning rate amplifies this noise, potentially pushing weights into a bad region from which training never recovers. Warmup starts with a near-zero LR, letting the model find a reasonable parameter region before ramping up. Without warmup, you risk permanent training instability, especially for transformers.

</details>

---

## 4. Batch Size

### What Batch Size Controls

Each training step computes the gradient on a **mini-batch** of examples. The batch size controls the tradeoff between gradient quality and computational efficiency.

### The Tradeoff

| Small Batch (8-32) | Large Batch (512-4096) |
|--------------------|----------------------|
| Noisy gradient estimates | Accurate gradient estimates |
| More update steps per epoch | Fewer update steps per epoch |
| Noise acts as regularization | Less regularization from noise |
| Finds flatter minima (often better generalization) | Can find sharper minima |
| Underutilizes GPU parallel capacity | Saturates GPU compute |
| Lower memory per step | Higher memory per step |

### The Noise-is-Good Theory

Small batch gradient noise helps the optimizer escape sharp local minima and find **flat minima** that generalize better. Flat minima are regions where the loss does not change much if you perturb the parameters slightly — meaning the model is robust to small weight changes and performs consistently on new data.

Large batch training reduces this noise, which can lead to sharper minima and worse generalization — unless compensated with techniques like learning rate scaling, warmup, or synthetic noise.

### Gradient Accumulation — Simulating Larger Batches

When your ideal batch size does not fit in GPU memory:

```python
accumulation_steps = 4  # Effective batch = micro_batch * 4
optimizer.zero_grad()

for i, (x, y) in enumerate(dataloader):
    loss = model(x, y) / accumulation_steps  # Scale loss by accumulation steps
    loss.backward()                            # Accumulate gradients

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()                       # Update with accumulated gradient
        optimizer.zero_grad()
```

This is mathematically equivalent to a 4x larger batch. The gradients accumulate over 4 mini-batches before the weight update. Almost no downside — slightly slower per effective step but the same total gradient.

**Most large model training uses gradient accumulation** because single GPU memory cannot hold realistic batch sizes.

### Practical Batch Size Selection

1. **Start with the largest power-of-2 batch that fits in memory** (for GPU efficiency)
2. **Common ranges:** 32-128 for fine-tuning, 256-4096 for pretraining
3. **For transformers, batch size is often expressed in tokens:** e.g., 1M tokens per batch for pretraining (so batch_size * seq_len = 1M)
4. **If validation performance degrades with large batches:** reduce batch size or use more warmup

---

## 5. Epochs and When to Stop Training

### What an Epoch Is

One epoch = one complete pass through the entire training dataset. If you have 10,000 training examples and batch size 100, one epoch is 100 gradient updates.

### How Many Epochs?

| Scenario | Typical Epochs | Notes |
|----------|---------------|-------|
| LLM pretraining | <1 (by tokens) | Most data seen only once. "1 epoch" over trillions of tokens |
| Large dataset (ImageNet, 1.2M) | 90-300 | Classic recipes train for many epochs |
| Fine-tuning pretrained model | 3-10 | Few epochs to adapt without forgetting |
| Small dataset (<10K) | 50-200 | More epochs needed, but watch overfitting |

### Early Stopping — The Primary Overfitting Defense

Monitor validation loss during training. When it stops improving (or starts increasing), stop training and revert to the best checkpoint:

```python
best_val_loss = float('inf')
patience = 5  # How many epochs to wait without improvement
patience_counter = 0

for epoch in range(max_epochs):
    train_one_epoch()
    val_loss = evaluate_validation()

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        save_checkpoint(model, 'best.pt')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping! No improvement for {patience} epochs.")
            break

# Always load the BEST checkpoint for final evaluation
model.load_state_dict(torch.load('best.pt'))
```

**Critical:** Save checkpoints at each improvement. The best model is usually NOT the model at the end of training — it is the one with the lowest validation loss, which may have occurred 5-10 epochs earlier.

---

## 6. Overfitting: Diagnosis and Treatment

### What Overfitting Looks Like

```
HEALTHY TRAINING:                OVERFITTING:
Loss                             Loss
^                                ^
| \.                             | \.
|   \.                           |   \.___
|     \.__                       |        \____         val (INCREASING)
|         \.__                   |             \___  /
|             \.__  train        |                 \/
|                 \.__ val       |                  \___ train (still decreasing)
+----> Epochs                    +----> Epochs

Train and val both decrease,     Train decreases but val
converging to similar values     starts INCREASING = memorization
```

### The Bias-Variance Tradeoff

- **Underfitting (high bias):** Model too simple. Both train and val loss are high and similar. The model cannot capture the patterns in the data.
- **Overfitting (high variance):** Model too complex relative to the data. Train loss is low, val loss is high. The model has memorized noise and specific training examples.
- **Sweet spot:** Both losses are low and close to each other.

### Diagnosing From Training Curves

| Symptom | Diagnosis | Action |
|---------|-----------|--------|
| Train loss high, val loss high | Underfitting | Increase model size, train longer, reduce regularization |
| Train loss low, val loss high | Overfitting | Add regularization, more data, reduce model size |
| Train loss low, val loss low but plateaued | Capacity reached | Need better architecture or more data |
| Both decreasing, gap growing | Early overfitting | Add regularization or use early stopping |
| Val loss erratic/noisy | LR too high or batch too small | Reduce LR, increase batch size |
| Train loss = 0 very early | Data leakage | Check if labels leak into features |

### Double Descent — The Modern Nuance

Classical statistics says increasing model size beyond the interpolation threshold (where the model can perfectly fit training data) always hurts generalization. But modern deep learning consistently shows **double descent**: very large, overparameterized models somehow generalize well despite memorizing the training data.

This means "make the model smaller" is not always the right response to overfitting. Sometimes making it **much larger** and training with appropriate regularization actually improves generalization. This is counterintuitive but well-documented.

---

## 7. Regularization Techniques

### Dropout

Randomly set a fraction of activations to zero during training:

```python
# During training with dropout rate p=0.1:
# Randomly zero 10% of activations
# Scale surviving activations by 1/(1-p) to maintain expected value

# In PyTorch (inverted dropout — scaling happens during training):
self.dropout = nn.Dropout(p=0.1)
x = self.dropout(x)  # During training: zero 10%, scale by 1/0.9
                       # During eval: identity (no change)
```

**Why it works:** Forces the network to not rely on any single neuron. Each neuron must contribute useful information independently. Mathematically, it is equivalent to training an ensemble of 2^n sub-networks (where n is the number of dropout-eligible neurons) and averaging their predictions at inference.

**Where to apply in transformers:**
- After attention weights (before value projection): `attn_weights = dropout(softmax(scores))`
- After each sub-layer (attention output, FFN output): `x = x + dropout(sublayer(norm(x)))`
- NOT on embeddings or normalization layers

**Typical values:** 0.1 for transformers, 0.2-0.5 for CNNs (higher in FC layers than conv layers).

**Critical:** Always set `model.eval()` during inference to disable dropout. Forgetting this is one of the most common bugs — the model produces different outputs each time and accuracy drops.

### Weight Decay (L2 Regularization)

Add a penalty that pushes weights toward zero:

```python
# In AdamW (decoupled):
W = W - lr * adam_update        # Gradient-based update
W = W - lr * weight_decay * W  # Separate decay step
# Equivalent to: W *= (1 - lr * weight_decay)
```

**Why it works:** Large weights mean the model is making aggressive, highly-specific decisions that fit the training data precisely. Weight decay pushes weights toward zero, preferring simpler, more general patterns.

**What NOT to regularize:** Bias terms, normalization parameters (gamma, beta), and embedding weights are typically excluded from weight decay:

```python
no_decay = ['bias', 'LayerNorm', 'layer_norm', 'embedding', 'rmsnorm']
param_groups = [
    {'params': [p for n, p in model.named_parameters()
                if not any(nd in n.lower() for nd in no_decay)],
     'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters()
                if any(nd in n.lower() for nd in no_decay)],
     'weight_decay': 0.0}
]
optimizer = AdamW(param_groups, lr=3e-4)
```

**Typical values:** 0.01 for transformers, 0.0001 for SGD, 0.05-0.1 for aggressive regularization.

### Data Augmentation — Free Performance

Creating synthetic variations of training data. The single highest-ROI technique for vision tasks:

| Domain | Common Augmentations |
|--------|---------------------|
| **Vision** | RandomResizedCrop, HorizontalFlip, ColorJitter, RandAugment, MixUp, CutMix |
| **NLP** | Back-translation, synonym replacement, random deletion/insertion, LLM paraphrasing |
| **Audio** | Time stretching, pitch shifting, noise addition, SpecAugment |
| **Tabular** | SMOTE (for class imbalance), feature noise injection |

**MixUp:** Blend two images and their labels:
```python
# lambda sampled from Beta(alpha, alpha)
x_mix = lambda * x_1 + (1 - lambda) * x_2
y_mix = lambda * y_1 + (1 - lambda) * y_2
```

**CutMix:** Replace a random patch of one image with a patch from another. More effective than MixUp for many tasks because it preserves more of the original spatial structure.

Data augmentation is effectively **free data** that teaches the model invariances you care about.

### Label Smoothing

Instead of hard one-hot labels, use soft labels:

```python
# Hard:  [0, 0, 1, 0, 0]  (100% confident in class 2)
# Soft:  [0.02, 0.02, 0.92, 0.02, 0.02]  (smoothing=0.1)

loss = nn.CrossEntropyLoss(label_smoothing=0.1)
```

**Why it helps:** Prevents the model from becoming overconfident (driving logits to infinity). Produces better-calibrated probabilities and slightly better generalization. Standard in transformer training.

### Early Stopping

Already covered in Section 5. Monitor validation loss and stop when it stops improving. Always revert to the best checkpoint.

### Stochastic Depth (Drop Path)

Randomly skip entire layers during training:

```python
def forward(self, x):
    if self.training and random.random() < drop_path_rate:
        return x    # Skip this layer entirely (identity via residual)
    return x + self.layer(x)
```

Acts like dropout but at the layer level. Used in EfficientNet, DeiT, and modern vision models. Typical rate: 0.1-0.3.

---

### Check Your Understanding

1. You observe that training loss is 0.05 but validation loss is 1.8. List three specific regularization techniques you would try, in order of expected impact for a vision task.
2. Why are bias terms, normalization parameters, and embedding weights typically excluded from weight decay? What would happen if you applied weight decay to them?
3. A colleague argues that label smoothing hurts performance because it "teaches the model to be less confident." Is this correct? What is the actual benefit?

<details>
<summary>Answers</summary>

1. For a vision task with severe overfitting: (1) Data augmentation (RandAugment, MixUp, CutMix) -- this is typically the highest-ROI technique because it provides effectively free training data. (2) Increase dropout (from 0.1 to 0.3-0.5) -- forces redundancy in learned features. (3) Increase weight decay (from 0.01 to 0.05-0.1) -- penalizes large weights that memorize specific training examples.
2. These parameters serve structural roles: biases set decision boundaries and need to be free to take any value; normalization gamma/beta control the learned scale and shift that normalization depends on; embeddings represent learned token identities. Applying weight decay pushes them toward zero, which undermines their function. For example, decaying the normalization beta toward zero forces all normalized outputs to be zero-centered, removing the model's ability to learn optimal activation distributions.
3. The colleague's framing is misleading. Label smoothing does reduce confidence, but this is a feature, not a bug. Without it, the model drives logits toward infinity to make softmax output approach 1.0, which wastes model capacity and overfits. Label smoothing prevents this extreme behavior, producing better-calibrated probabilities and slightly better generalization. The model can still be highly accurate without being overconfident.

</details>

---

## 8. Batch Normalization vs Layer Normalization

### Batch Normalization (BatchNorm)

Normalizes across the **batch dimension** for each feature/channel:

```python
# For each feature independently:
mean = x.mean(dim=0)       # Mean across the BATCH
std = x.std(dim=0)
x_norm = (x - mean) / (std + eps)
output = gamma * x_norm + beta   # Learned scale and shift
```

**During inference:** Uses running statistics (exponential moving average computed during training), NOT batch statistics. This means BatchNorm behaves **differently** during training vs inference.

**Where it is used:** CNNs. BatchNorm after each conv layer was the standard from 2015-2020.

**Problems:**
- Depends on batch size — small batches give noisy statistics
- Different behavior during training and eval (common source of bugs)
- Does not work with variable-length sequences (different positions should not be normalized together)
- Requires cross-GPU synchronization for distributed training (SyncBatchNorm)

### Layer Normalization (LayerNorm)

Normalizes across the **feature dimension** for each sample independently:

```python
# For each sample and position independently:
mean = x.mean(dim=-1, keepdim=True)   # Mean across d_model
std = x.std(dim=-1, keepdim=True)
x_norm = (x - mean) / (std + eps)
output = gamma * x_norm + beta
```

**Advantages over BatchNorm:**
- Independent of batch size (works with batch size 1)
- Same behavior during training and inference (no running statistics)
- Works naturally with variable-length sequences
- No cross-GPU synchronization needed

**Where it is used:** Transformers, RNNs.

### RMSNorm — The Simplified Standard

```python
rms = sqrt(mean(x^2) + eps)
output = (x / rms) * gamma
# No mean subtraction, no beta parameter
```

Drops the mean-centering step from LayerNorm. About 15% faster with no quality loss. Used in Llama, Mistral, Gemma, and virtually all modern LLMs.

### When to Use Which

| Architecture | Normalization | Why |
|-------------|--------------|-----|
| CNNs | BatchNorm | Cross-sample normalization benefits spatial features |
| Transformers | LayerNorm / RMSNorm | Variable sequences, batch-size independence |
| Small batch CNNs | GroupNorm | BatchNorm with batch size <16 is noisy |
| GANs | InstanceNorm / SpectralNorm | BatchNorm destabilizes discriminator |
| Modern LLMs (2024+) | RMSNorm | Faster, equally effective |

---

## 9. Gradient Clipping

### The Problem: Exploding Gradients

Gradient norms can spike due to pathological training examples, numerical instability, or the inherent instability of deep/recurrent networks. A single large gradient can destroy a well-trained model:

```
Normal:    W = 10.0 - 0.001 * 5.0    = 9.995   (fine)
Exploding: W = 10.0 - 0.001 * 50000  = -40.0    (catastrophic)
```

### The Solution: Clip the Gradient Norm

```python
# After loss.backward(), before optimizer.step():
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# How it works:
# 1. Compute total norm: sqrt(sum(grad.norm()^2 for all parameters))
# 2. If total_norm > max_norm: scale ALL gradients by max_norm / total_norm
# 3. This preserves gradient DIRECTION but limits MAGNITUDE
```

**Important:** Clip the **norm** (magnitude), not individual gradient values. Clipping individual values (`clamp`) changes the gradient direction. Norm clipping preserves direction and only limits step size.

### When to Use

| Architecture | Clip? | Typical max_norm |
|-------------|-------|-----------------|
| Transformers | Yes | 1.0 |
| RNNs/LSTMs | Always mandatory | 0.25 - 5.0 |
| CNNs | Usually not needed | N/A (BatchNorm stabilizes) |

### Monitoring Gradient Norms

Tracking gradient norms is one of the best training diagnostics:

```python
total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
# Log total_norm to wandb/tensorboard
```

| Pattern | Meaning |
|---------|---------|
| Stable, consistent norm | Healthy training |
| Gradually increasing norm | Model may become unstable |
| Sudden spikes | Bad data, LR too high, or instability |
| Near-zero norm | Vanishing gradients, model not learning |
| Norm = max_norm frequently | Clipping is active — consider if LR is too high |

---

## 10. Mixed Precision Training

### What It Is

Instead of FP32 (32-bit float) for all computations, use FP16 or BF16 (16-bit) for most operations and FP32 only where precision is critical:

```
FP32: 1 sign + 8 exponent + 23 mantissa = 32 bits  (high precision, large)
FP16: 1 sign + 5 exponent + 10 mantissa = 16 bits  (medium precision, limited range)
BF16: 1 sign + 8 exponent + 7 mantissa  = 16 bits  (lower precision, FP32 range)
```

### Why It Helps

1. **2x memory reduction:** Weights, activations, and gradients are half the size. You can double the batch size or train a model twice as large.
2. **2-3x speed improvement:** Modern GPUs (A100, H100, RTX 3090+) have specialized tensor cores for FP16/BF16 that compute at 2-8x the speed of FP32.
3. **No meaningful quality loss:** When done correctly, the model learns the same thing.

### FP16 vs BF16

| Property | FP16 | BF16 |
|----------|------|------|
| Precision (mantissa) | 10 bits (higher precision) | 7 bits (lower precision) |
| Range (exponent) | 5 bits (max ~65,504) | 8 bits (same as FP32: ~3.4e38) |
| Overflow risk | **High** — values > 65K overflow | **Low** — same range as FP32 |
| Loss scaling required? | **Yes** — gradient values can underflow | **No** |
| Available on | V100, A100, H100, consumer GPUs | A100, H100, RTX 3090+ |

**BF16 is the modern default** because it has the same numerical range as FP32, eliminating overflow concerns. FP16 requires careful loss scaling (multiplying loss by a large factor to prevent gradient underflow, then dividing the gradient by the same factor before the update).

### How to Use It in PyTorch

```python
# BF16 (preferred, simpler):
for batch in dataloader:
    optimizer.zero_grad()
    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
        output = model(batch)
        loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()

# FP16 (needs GradScaler for loss scaling):
scaler = torch.amp.GradScaler()
for batch in dataloader:
    optimizer.zero_grad()
    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
        output = model(batch)
        loss = loss_fn(output, target)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### What Stays in FP32

PyTorch's `autocast` automatically handles this, but conceptually:
- **Loss computation:** Small values in log-softmax can underflow in FP16
- **Softmax:** Exponentiation can overflow in FP16
- **Normalization:** Mean/variance computation accumulates errors in low precision
- **Master weights:** Optimizer maintains FP32 copies of weights for accurate updates (forward pass casts to BF16)

### When to Use

**Always, on modern GPUs.** There is essentially no quality loss with BF16, and the 2x memory + 2-3x speed improvement is enormous. Mixed precision is not optional for large model training — it is required.

---

### Check Your Understanding

1. You are training a transformer but want to switch from BatchNorm to LayerNorm. Beyond the architecture change, what behavioral differences should you expect during training and inference?
2. Why is BF16 preferred over FP16 for modern training? What specific failure mode does BF16 avoid that FP16 does not?
3. You notice that gradient norms spike to 500x their normal value every few hundred steps but the model still trains okay due to gradient clipping. Should you investigate further or leave it?

<details>
<summary>Answers</summary>

1. Key differences: (a) LayerNorm behaves identically during training and inference (no running statistics to track), eliminating a common source of bugs. (b) LayerNorm normalizes per-sample across features, so it works with batch size 1 and variable-length sequences. (c) You no longer need SyncBatchNorm for distributed training. (d) The model may need learning rate adjustment since the normalization dynamics differ.
2. BF16 has 8 exponent bits (same as FP32), giving it the same numerical range (~3.4e38). FP16 has only 5 exponent bits with a max value of ~65,504. This means FP16 can overflow when activation values or gradient values exceed 65K, causing NaN or Inf. BF16 avoids this entirely, eliminating the need for loss scaling (GradScaler). The tradeoff is that BF16 has lower precision (7 vs 10 mantissa bits), but this has negligible impact on training quality.
3. You should investigate. While gradient clipping prevents immediate damage, frequent large spikes indicate an underlying issue: possibly bad data samples (NaN, extreme outliers), learning rate too high for certain parameter groups, or numerical instability in specific layers. Check which batches trigger spikes, inspect the data in those batches, and consider whether the learning rate or architecture needs adjustment. Relying solely on clipping masks the root cause.

</details>

---

## 11. Training Curves: How to Read Them

### What Healthy Training Looks Like

```
Loss
^
|  \.
|   \.
|    \.___
|     \   \___
|      \      \____
|       \   train   \________
|        \     val            \___________
+--+--+--+--+--+--+--+--+--+--+--+--+--> Steps

Characteristics:
1. Both curves decrease steadily
2. Validation tracks training closely (small gap)
3. Curves smooth out over time
4. Eventual convergence to a plateau
5. Final train-val gap is small
```

### Common Unhealthy Patterns

**Learning rate too high — oscillating loss:**
```
Loss
^
| /\ /\ /\
|/  V  V  \  /\
|          \/  \ ...
+----> Steps
```
Fix: Reduce LR by 3-10x.

**Learning rate too low — barely decreasing:**
```
Loss
^
| \.....................................
+----> Steps
```
Fix: Increase LR, run LR finder.

**Overfitting — diverging curves:**
```
Loss
^
|   val: starts increasing
|  \.    /
|   \.  /
|    \/
|     \.___ train: keeps decreasing
+----> Steps
```
Fix: Early stopping, add regularization, more data.

**Loss spike — sudden jump:**
```
Loss
^
|            /\
|  \.__     /  \.__
|      \.__/       \.___
+----> Steps
```
Cause: Bad batch, gradient explosion, numerical instability. Fix: gradient clipping, check data, reduce LR.

**Loss goes to NaN:**
```
Loss
^
|  \._
|     \._ NaN!
+----> Steps
```
Cause: Overflow, log(0), division by zero. Fix: BF16, gradient clipping, check data for NaN, lower LR.

### What to Log and Monitor

Always track these during training:

| Metric | Why | Frequency |
|--------|-----|-----------|
| Training loss | Is the model learning? | Every step |
| Validation loss | Is it generalizing? | Every epoch or every N steps |
| Learning rate | Verify schedule works | Every step |
| Gradient norm | Detect instability | Every step |
| Task metric (accuracy, F1, BLEU) | The real objective | Every epoch |
| GPU utilization | Are you compute-efficient? | Periodically |
| Memory usage | Near the limit? | Periodically |

Use Weights & Biases, TensorBoard, or MLflow for visualization. The investment in logging pays back immediately when something goes wrong.

---

## 12. The "Overfit a Tiny Batch First" Debugging Trick

This is the **single most valuable debugging technique in deep learning.** Before training on the full dataset, always verify your model can memorize a tiny subset.

### The Procedure

```python
# Grab a single tiny batch (5-10 examples)
tiny_batch = next(iter(train_loader))

# Train on ONLY this batch for many steps
model.train()
for step in range(1000):
    optimizer.zero_grad()
    output = model(tiny_batch.input)
    loss = loss_fn(output, tiny_batch.labels)
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print(f"Step {step}: loss = {loss.item():.6f}")

# Expected: loss should approach 0 (or near theoretical minimum)
# For cross-entropy with 10 classes: initial loss ~2.3, should reach <0.01
```

### What It Tests

This single test verifies that:
- The model architecture is functional (correct shapes, activations work)
- The loss function is correct (produces a learnable signal)
- The data pipeline is correct (inputs and labels match, preprocessing is right)
- Gradients flow properly (no dead layers, no NaN)
- The learning rate is in a reasonable range (not too high to diverge, not too low to stall)
- The optimizer is configured correctly

### What Failure Means

| Symptom | Diagnosis |
|---------|-----------|
| Loss stays completely constant | Gradients are zero (dead network), or LR is effectively zero |
| Loss oscillates without decreasing | LR too high |
| Loss is NaN immediately | Data has NaN, log(0), division by zero, or LR way too high |
| Loss decreases but plateaus above zero | Architecture too restrictive, or labels are wrong |
| Loss reaches zero but predictions look wrong | Labels do not match inputs (data pipeline bug) |
| Loss decreases very slowly (>500 steps for tiny batch) | LR too low (increase 10x) |

### The Rule

**If the model cannot overfit a tiny batch, there is no point training on the full dataset.** The bug is in the pipeline, not in the hyperparameters. Fix it first. This test takes 2 minutes and can save hours.

---

## 13. Practical Training Recipe: Step by Step

### Step 1: Data Verification (30 minutes)

```
[ ] Inspect raw data: look at 20+ examples manually
[ ] Check label distribution: are classes balanced? If not, what is the imbalance ratio?
[ ] Verify data pipeline: are inputs and labels correctly paired?
[ ] Check for data leakage: is test/val data leaking into training?
[ ] Verify preprocessing: normalization values, tokenization, image transforms
[ ] Check for NaN/inf in data
```

### Step 2: Establish Baseline (1 hour)

```
[ ] Choose a pretrained model if available (almost always yes in 2026)
[ ] Run the "overfit a tiny batch" test — must reach near-zero loss
[ ] Train a simple baseline (logistic regression, small model) for comparison
[ ] Verify that the evaluation pipeline works correctly
```

### Step 3: Initial Training Run (hours to days)

```python
config = {
    'optimizer': 'AdamW',
    'lr': 3e-4,                    # 2e-5 for fine-tuning pretrained
    'weight_decay': 0.01,
    'batch_size': 32,              # Or largest that fits with grad accumulation
    'warmup_ratio': 0.05,          # 5% of total steps
    'lr_schedule': 'cosine',
    'dropout': 0.1,
    'gradient_clip_norm': 1.0,
    'precision': 'bf16',
    'max_epochs': 10,              # Short run to diagnose
    'early_stopping_patience': 3,
}
```

### Step 4: Diagnose and Iterate

Read the training curves. Based on what you see:

| Observation | Action |
|-------------|--------|
| Not learning at all | Check pipeline (overfit tiny batch test), increase LR |
| Learning but slowly | Increase LR, check if GPU is utilized |
| Overfitting | Add dropout, increase weight decay, add data augmentation |
| Underfitting | Increase model size, decrease regularization, train longer |
| Unstable (spikes/NaN) | Reduce LR, add gradient clipping, check data for issues |

Try 3-5 learning rates (e.g., 1e-5, 3e-5, 1e-4, 3e-4, 1e-3) and pick the best.

### Step 5: Final Training Run

```
[ ] Use best hyperparameters from Step 4
[ ] Train for full duration with early stopping
[ ] Save checkpoints every N steps (not just at epoch boundaries)
[ ] Final evaluation on held-out test set (run ONCE — do not iterate on test set)
[ ] Document the full training configuration for reproducibility
```

### Step 6: Sanity Checks

```
[ ] Model outperforms random baseline
[ ] Model outperforms simple baseline
[ ] Predictions on known examples are correct
[ ] Model handles edge cases (empty input, very long input, out-of-distribution)
[ ] Inference latency meets requirements
[ ] Model size fits deployment constraints
```

---

## 14. Common Training Failures and Quick Fixes

| Problem | Likely Cause | Fix |
|---------|-------------|-----|
| Loss NaN from step 1 | Data has NaN, LR too high, bad init | Check data, reduce LR by 100x, verify init |
| Loss NaN after good training | Gradient explosion | Add gradient clipping (1.0), check for bad batches |
| Loss decreases very slowly | LR too low | Increase LR, use LR finder |
| Loss oscillates wildly | LR too high | Reduce LR by 3-10x |
| Train great, val terrible | Overfitting | Add regularization: dropout, weight decay, augmentation |
| Both losses plateau high | Underfitting | Bigger model, more features, less regularization |
| All predictions identical | Dead network or class imbalance | Check init, check class balance, verify loss function |
| Training gets slower over time | Memory leak or data loading bottleneck | Check for growing lists, increase num_workers |
| GPU utilization <50% | Data loading bottleneck | More num_workers, pin_memory=True, prefetch |
| Results vary wildly between runs | High variance from randomness | Set all seeds, increase batch size |
| Good accuracy, bad calibration | Model is overconfident | Add label smoothing (0.1), temperature scaling |

---

## Common Pitfalls

**Pitfall 1: Not running the "overfit a tiny batch" test before full training**
- Symptom: You spend hours training on the full dataset only to discover the loss never decreases, or predictions are nonsensical
- Why: A bug in the data pipeline, loss function, or model architecture prevents learning entirely, but you only notice after wasting compute
- Fix: Always run the overfit-tiny-batch test first. It takes 2 minutes and catches data pipeline bugs, shape mismatches, incorrect loss functions, and dead gradients before they waste hours

**Pitfall 2: Using Adam instead of AdamW for transformer training**
- Symptom: Overfitting that seems resistant to weight decay tuning; inconsistent regularization across parameters
- Why: Adam couples weight decay with adaptive learning rates, meaning parameters with large gradient history receive less regularization. This silently undermines your weight decay setting
- Fix: Always use AdamW (decoupled weight decay) for transformers. It is the standard for a reason. In PyTorch, use `torch.optim.AdamW`, not `torch.optim.Adam` with `weight_decay`

**Pitfall 3: Forgetting to switch between model.train() and model.eval()**
- Symptom: Validation metrics are noisy and lower than expected; model produces different outputs for the same input during evaluation
- Why: Dropout remains active during evaluation (randomly zeroing neurons), and BatchNorm uses batch statistics instead of running statistics
- Fix: Call `model.eval()` before validation/inference and `model.train()` before training. Wrap inference in `torch.no_grad()` to also save memory

**Pitfall 4: Scaling learning rate without adjusting warmup when changing batch size**
- Symptom: Training diverges after increasing batch size, even though you scaled the learning rate proportionally
- Why: The linear scaling rule (LR proportional to batch size) is only valid with sufficient warmup. Larger batches with larger LR produce larger initial updates that require more warmup steps to stabilize
- Fix: When scaling batch size by factor k, also increase warmup steps proportionally. Use gradual warmup over 5-10% of total training steps

## Hands-On Exercises

### Exercise 1: Learning Rate Sensitivity Experiment
**Goal:** Build intuition for how learning rate affects training dynamics
**Task:**
1. Train a small CNN (3 conv layers + 2 FC layers) on CIFAR-10 in PyTorch
2. Run 5 training experiments with learning rates: 1e-1, 1e-2, 1e-3, 1e-4, 1e-5 (all using Adam)
3. For each run, log training loss, validation loss, and gradient norm at each step for 10 epochs
4. Plot all 5 training curves on one graph and all 5 gradient norm trajectories on another
5. Implement the LR finder (exponentially increasing LR over one epoch) and verify it suggests a value near your best-performing LR
**Verify:** The LR finder's suggested value (steepest downward slope on loss-vs-LR plot) should fall between 1e-3 and 1e-2. Training at 1e-1 should diverge or oscillate. Training at 1e-5 should barely decrease loss.

### Exercise 2: Regularization Ablation Study
**Goal:** Quantify the impact of each regularization technique on the train-val gap
**Task:**
1. Train a ResNet-18 on a small subset of CIFAR-10 (5000 training images) -- enough to overfit easily
2. Run these experiments: (a) no regularization, (b) dropout=0.3, (c) weight decay=0.01, (d) data augmentation (RandomCrop + HorizontalFlip), (e) all three combined
3. For each experiment, plot training and validation accuracy curves over 50 epochs
4. Create a summary table: final train accuracy, final val accuracy, and the train-val gap for each configuration
**Verify:** The no-regularization baseline should show a large train-val gap (>20%). Data augmentation should provide the largest single improvement. The combined configuration should have the smallest gap.

---

## 15. Interview Questions

### Conceptual

1. **Why does weight initialization matter?** If all weights are zero or the same, symmetry is never broken and the network cannot learn diverse features. If weights are too large, activations explode. If too small, they vanish. Xavier and He initialization set weight variance to preserve activation variance across layers, keeping the network in a healthy training regime.

2. **Explain the difference between Adam and AdamW.** Adam applies weight decay as L2 regularization through the gradient (coupled with adaptive learning rates). AdamW decouples weight decay from the gradient update, applying it directly to weights. This means regularization strength is consistent across parameters regardless of gradient magnitude. AdamW is the standard for transformer training.

3. **Why does learning rate warmup help?** Early in training, predictions are random and gradients are large and noisy. A high learning rate amplifies this noise, potentially pushing the model into a bad region permanently. Warmup starts with near-zero LR, letting the model stabilize before ramping to the target LR.

4. **How do you diagnose overfitting vs underfitting?** Overfitting: training loss decreases while validation loss increases (or the gap grows). The model memorizes training data. Underfitting: both losses remain high. The model cannot capture the data's patterns. Fix overfitting with regularization (dropout, weight decay, augmentation, early stopping). Fix underfitting with more capacity or longer training.

5. **What is mixed precision training and why is it important?** Use FP16/BF16 for most computations and FP32 where precision matters (loss, normalization). Benefits: 2x memory reduction, 2-3x speed on tensor cores, no quality loss. BF16 is preferred because it has FP32's numerical range, avoiding overflow. Required for efficient training of any large model.

6. **What is gradient clipping and when do you use it?** Clip the total gradient norm to a maximum value (typically 1.0) to prevent exploding gradient updates. Applied after backward() and before step(). Preserves gradient direction but limits magnitude. Essential for RNNs, standard for transformers, usually not needed for CNNs (BatchNorm stabilizes).

7. **Explain the "overfit a tiny batch" debugging trick.** Train on 5-10 examples for hundreds of steps. Loss should reach near-zero. If it cannot, something is fundamentally broken: bad loss function, data pipeline bug, dead gradients, or wrong architecture. This test takes 2 minutes and catches the most common bugs before wasting hours on full training.

### Practical

8. **Your transformer training loss spikes at step 5000. What do you do?** (1) Check gradient norm at the spike — likely a gradient explosion. (2) Add/lower gradient clipping (max_norm=1.0). (3) Check if the spike corresponds to a specific batch (bad data). (4) Reduce LR. (5) Check for numerical instability (log(0), overflow). (6) If using FP16, switch to BF16.

9. **Your model achieves 95% training accuracy but 70% validation accuracy. What is happening and how do you fix it?** This is overfitting — 25-point gap shows memorization. Fixes in priority order: (1) data augmentation (highest ROI), (2) increase dropout (0.2-0.5), (3) increase weight decay (0.05-0.1), (4) early stopping (use best checkpoint), (5) reduce model size, (6) get more training data. Start with augmentation — it is usually the single most impactful fix.

10. **Walk through your training recipe for a new task.** Verify data (inspect examples, check labels). Overfit tiny batch to verify pipeline. Start with standard config: AdamW, LR 3e-4, cosine schedule, 5% warmup, weight decay 0.01, gradient clipping 1.0, BF16. Monitor train/val curves and gradient norms. If overfitting, add regularization. If underfitting, increase capacity. Sweep LR and weight decay with random search. Final evaluation on held-out test set once.

---

## Key Takeaways

1. **Initialization:** He for ReLU, Xavier for sigmoid/tanh, small normal (0.02) for transformers. Bad init causes symmetry or gradient issues.

2. **Learning rate is the most important hyperparameter.** Start with 3e-4 for AdamW, use warmup + cosine schedule. Use the LR finder if unsure.

3. **AdamW is the default optimizer for transformers.** SGD+momentum for CNNs. The difference between Adam and AdamW matters — always use AdamW.

4. **Batch size trades noise for speed.** Use gradient accumulation to simulate larger batches. Scale LR proportionally with batch size.

5. **Early stopping prevents overfitting.** Monitor validation loss, save best checkpoint, stop when no improvement.

6. **Regularization toolkit:** Dropout (0.1-0.5), weight decay (0.01-0.1), data augmentation (always for vision), label smoothing (0.1), stochastic depth.

7. **LayerNorm/RMSNorm for transformers, BatchNorm for CNNs.** Never mix them up.

8. **Gradient clipping prevents exploding gradients.** max_norm=1.0 for transformers. Almost always worth including.

9. **Mixed precision (BF16) is required for efficient training.** 2x memory, 2-3x speed, no quality loss. Not optional.

10. **Read your training curves.** They tell you everything: overfitting, underfitting, LR issues, instability. Log everything.

11. **Always overfit a tiny batch first.** If the model cannot memorize 5 examples, nothing else matters. Fix the bug first.

12. **Follow the recipe:** Verify data -> overfit tiny batch -> standard config -> diagnose from curves -> iterate -> final evaluation once on test set.

## Summary

Training neural networks is a craft that combines principled choices (He initialization for ReLU, AdamW with warmup + cosine schedule, gradient clipping) with systematic debugging (overfit a tiny batch, read training curves, monitor gradient norms). The single most important takeaway: if the model cannot overfit 5 examples, fix the pipeline bug before doing anything else. Everything else -- learning rate tuning, regularization, mixed precision -- is optimization on top of a working foundation.

## What's Next

- **Next lesson:** [CNNs](../cnns/COURSE.md) -- applies these training mechanics to convolutional architectures, including transfer learning strategies and data augmentation techniques specific to vision
- **Builds on this:** [Transformers and Attention](../transformers-attention/COURSE.md) -- uses normalization (LayerNorm/RMSNorm), AdamW, warmup + cosine schedules, and mixed precision as standard components of the transformer training stack
