# 02 — Neural Networks

> How neural networks work — from single neurons to transformers. Focus on intuition, not implementation.

## Why This Matters

You use LLMs as APIs. To be an ML engineer, you need to understand what's inside them — how they learn, why they fail, and how to fix them.

## Subdirectories

```
02-neural-networks/
├── fundamentals/            # Perceptrons, activation functions, backprop, loss functions
├── cnns/                    # Convolutions, pooling, architectures (ResNet, EfficientNet)
├── rnns-lstms/              # Sequence modeling, vanishing gradients, gating mechanisms
├── transformers-attention/  # Self-attention, positional encoding, encoder/decoder
└── training-mechanics/      # Learning rates, batch size, regularization, dropout, batch norm
```

## Key Resources

| Resource | What it covers | Time |
|---|---|---|
| 3Blue1Brown: Neural Networks | Visual intuition for NNs and backprop | ~1 hr |
| DeepLearning.AI: How Transformer LLMs Work | Conceptual transformer walkthrough | ~2 hrs |
| DeepLearning.AI: Attention in Transformers | Build attention from scratch | ~2 hrs |
| Jay Alammar: The Illustrated Transformer | Best visual explanation of transformers | ~1 hr read |
| DeepLearning.AI: Deep Learning Specialization | Comprehensive (courses 1-3 most relevant) | ~4 months |

## Key Concepts to Master

### Fundamentals
- [ ] What does a single neuron compute? (weighted sum + activation)
- [ ] What are activation functions and why do we need them? (non-linearity)
- [ ] What is a loss function? (the number we're minimizing)
- [ ] What is backpropagation? (chain rule applied to compute gradients)
- [ ] What is a computational graph? (how frameworks track operations for autograd)

### Training Mechanics
- [ ] Learning rate — why 3e-4 is a good default for Adam
- [ ] Batch size — tradeoff between noise and speed
- [ ] Epochs — how many passes through the data
- [ ] Overfitting — model memorizes training data, fails on new data
- [ ] Regularization — dropout, weight decay, early stopping
- [ ] Batch normalization — stabilizes training by normalizing layer inputs
- [ ] Learning rate scheduling — warmup, cosine decay, step decay

### Transformers (Critical for applied ML roles)
- [ ] Self-attention — every token attends to every other token
- [ ] Query, Key, Value — the three projections and what they represent
- [ ] Multi-head attention — why multiple attention heads help
- [ ] Positional encoding — how transformers know word order
- [ ] Feed-forward layers — the MLP after attention
- [ ] Layer normalization — where it goes and why
- [ ] Encoder vs decoder — when to use each
- [ ] Causal masking — preventing the model from seeing the future
