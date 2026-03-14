# Part II: Deep Learning

## Overview

Deep learning extends supervised learning using **neural networks** - compositions of many nonlinear transformations. This enables learning complex patterns that linear models cannot capture.

## Chapter in This Part

- **[Chapter 7: Deep Learning](ch07-deep-learning.md)** - Neural networks, backpropagation, modern architectures

## Core Concepts

### From Linear to Nonlinear

**Problem**: Linear models (logistic regression, SVM) have limited expressiveness.

**Solution Options**:
1. Feature engineering (manual, requires domain expertise)
2. Kernel methods (implicit high-dimensional features)
3. **Neural networks** (learned feature hierarchies)

### The Neural Network Approach

```
Input x → [Hidden Layers] → Output ŷ
         (learned features)
```

Neural networks learn the feature representation AND the classifier jointly through end-to-end training.

## Key Topics

| Topic | Description |
|-------|-------------|
| **MLPs** | Multi-layer perceptrons, fully connected networks |
| **Activation Functions** | ReLU, sigmoid, tanh - nonlinearities |
| **Backpropagation** | Efficient gradient computation via chain rule |
| **Modules** | Convolution, normalization, attention |
| **Optimization** | SGD, Adam, learning rate schedules |

## Mathematical Framework

### Forward Pass
```
z^[l] = W^[l] a^[l-1] + b^[l]    (linear transformation)
a^[l] = g(z^[l])                  (nonlinear activation)
```

### Backward Pass (Backpropagation)
```
∂L/∂W^[l] = ∂L/∂z^[l] · (a^[l-1])ᵀ
∂L/∂a^[l-1] = (W^[l])ᵀ · ∂L/∂z^[l]
```

## Why Deep Learning Works

1. **Universal Approximation**: Neural networks can approximate any continuous function
2. **Hierarchical Features**: Deep networks learn increasingly abstract representations
3. **End-to-End Learning**: No need for manual feature engineering
4. **Scalability**: Performance improves with more data and compute

## Modern Architectures

| Architecture | Use Case | Key Innovation |
|--------------|----------|----------------|
| **CNNs** | Images, spatial data | Convolutional layers |
| **RNNs/LSTMs** | Sequences, time series | Recurrent connections |
| **Transformers** | NLP, vision | Self-attention mechanism |
| **ResNets** | Deep networks | Skip connections |

## Practical Considerations

- **Initialization**: Xavier/He initialization for stable training
- **Regularization**: Dropout, weight decay, batch normalization
- **Optimization**: Adam is often a good default
- **Debugging**: Start simple, validate gradient computation
