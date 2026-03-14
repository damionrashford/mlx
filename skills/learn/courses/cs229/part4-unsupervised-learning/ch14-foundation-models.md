# Chapter 14: Self-Supervised Learning and Foundation Models

## Introduction

**Foundation models** are large models pretrained on massive datasets, then adapted to downstream tasks.

**Key insight**: Unlabeled data is abundant; leverage it via **self-supervised learning**.

## 14.1 Pretraining and Adaptation

### The Paradigm

```
[Massive Unlabeled Data] → Pretrain → [Foundation Model] → Fine-tune → [Task-Specific Model]
```

**Benefits**:
- Learn general representations from large data
- Transfer knowledge to tasks with limited labels
- Amortize compute across many applications

### Types of Adaptation

| Method | Description | When to Use |
|--------|-------------|-------------|
| **Fine-tuning** | Update all parameters | Moderate labeled data |
| **Linear probing** | Train only final layer | Limited labeled data |
| **Prompt tuning** | Learn prompt embeddings | Very limited data |
| **In-context learning** | No parameter updates | Zero/few-shot |

## 14.2 Pretraining Methods in Computer Vision

### Contrastive Learning (SimCLR, MoCo)

**Idea**: Learn representations where similar examples are close, different examples are far.

**Method**:
1. Create augmented views of same image (positive pairs)
2. Treat different images as negative pairs
3. Train encoder to:
   - Maximize similarity of positive pairs
   - Minimize similarity of negative pairs

**InfoNCE Loss**:
```
L = -log[exp(sim(z_i, z_j)/τ) / Σₖ exp(sim(z_i, z_k)/τ)]
```

### Masked Image Modeling (MAE)

**Idea**: Mask patches of image, predict the masked patches.

**Method**:
1. Mask random patches (75%)
2. Encode visible patches with transformer
3. Decode to reconstruct masked patches

**Benefit**: Forces model to learn holistic understanding.

### CLIP (Contrastive Language-Image Pretraining)

**Idea**: Align image and text representations.

**Method**:
- Train on (image, caption) pairs from web
- Image encoder and text encoder
- Contrastive loss to match images with captions

**Benefit**: Zero-shot classification via text prompts.

## 14.3 Pretrained Large Language Models

### Language Modeling Objective

Predict next token given previous tokens:
```
P(x_t | x_1, ..., x_{t-1})
```

**Autoregressive**: GPT family

### Masked Language Modeling

Predict masked tokens given context:
```
P(x_t | x_1, ..., x_{t-1}, [MASK], x_{t+1}, ..., x_T)
```

**Bidirectional**: BERT family

### Modern LLMs

| Model | Size | Training Data | Key Innovation |
|-------|------|---------------|----------------|
| **GPT-3** | 175B | 300B tokens | Scale |
| **GPT-4** | ~1T | ~10T tokens | Multimodal |
| **LLaMA** | 7-70B | 1T tokens | Open weights |
| **Claude** | ~100B | Diverse | Constitution AI |

## 14.3.1 Transformers: Opening the Black Box

### The Attention Mechanism

For query q, keys K, values V:
```
Attention(q, K, V) = softmax(qKᵀ/√d) · V
```

**Intuition**: Weighted average of values, weights determined by query-key similarity.

### Multi-Head Attention

Multiple parallel attention operations:
```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) · W^O

head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

**Benefit**: Attend to different aspects simultaneously.

### Transformer Block

```
x → [Multi-Head Attention] → [Add & Norm] → [FFN] → [Add & Norm] → output
     └───────────────────────────┘           └───────────────────────┘
              Skip Connection                     Skip Connection
```

### Positional Encoding

Attention is permutation-invariant, so add positional information:
- **Sinusoidal**: Fixed, based on position
- **Learned**: Trained embeddings
- **Rotary (RoPE)**: Relative positions in rotation

## 14.3.2 Zero-Shot and In-Context Learning

### Zero-Shot Learning

Perform task without any examples:
```
Input: "Classify sentiment: 'This movie was amazing!'"
Output: "Positive"
```

Model uses pretrained knowledge.

### Few-Shot (In-Context) Learning

Provide examples in the prompt:
```
Input: "Classify sentiment:
        'Great film' → Positive
        'Terrible acting' → Negative
        'This movie was amazing!' → "
Output: "Positive"
```

**No gradient updates** - model generalizes from context.

### Why Does In-Context Learning Work?

Hypotheses:
1. **Implicit fine-tuning**: Attention acts like gradient descent
2. **Bayesian inference**: Posterior update given examples
3. **Retrieval**: Finding similar patterns in training data

Active research area.

## Scaling Laws

Empirically, LLM performance scales predictably:
```
L = (N/N_c)^(-α_N) + (D/D_c)^(-α_D) + irreducible error
```

Where:
- N = number of parameters
- D = training data size
- α_N ≈ 0.076, α_D ≈ 0.095

**Implication**: Bigger models trained on more data = better.

## Key Takeaways

1. **Foundation models** pretrain on massive unlabeled data
2. **Self-supervised objectives** create learning signal without labels
3. **Transformers** enable parallel processing and long-range dependencies
4. **In-context learning** adapts without gradient updates
5. **Scale** is crucial: more parameters, more data = better

## Practical Notes

- **Start with pretrained**: Almost never train from scratch
- **Choose model size**: Balance performance vs. cost
- **Fine-tuning**: Often beats in-context learning with enough data
- **Prompt engineering**: Critical for in-context learning
- **APIs**: OpenAI, Anthropic, etc. for largest models
- **Open models**: LLaMA, Mistral for local deployment

