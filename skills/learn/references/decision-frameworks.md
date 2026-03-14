# Decision Frameworks — ML System Design

> Mental models for making ML engineering decisions.
> These are what separate an ML engineer from someone who just calls APIs.

---

## 1. Model Selection: "Which model for which problem?"

```
What's your data?
├── Tabular (rows & columns)
│   ├── < 10K rows → Logistic Regression, Random Forest
│   ├── 10K-1M rows → XGBoost / LightGBM
│   └── > 1M rows → XGBoost still, or neural nets if complex patterns
│
├── Text
│   ├── Classification → Fine-tuned BERT or LLM with prompt
│   ├── Generation → LLM (GPT, Claude, Llama)
│   ├── Search/retrieval → Embeddings + vector DB
│   └── Extraction → LLM with structured output
│
├── Images
│   ├── Classification → CNN (ResNet, EfficientNet) or ViT
│   ├── Detection → YOLO, DETR
│   └── Generation → Diffusion models
│
├── Sequences / Time Series
│   ├── Short sequences → LSTM, GRU
│   ├── Long sequences → Transformer
│   └── Forecasting → Prophet, N-BEATS, temporal fusion transformer
│
└── Multi-modal (text + images + structured)
    └── Custom architecture or multimodal LLM
```

---

## 2. Fine-tune vs RAG vs Prompt Engineering

```
Is the knowledge already in the base model?
├── Yes → Prompt engineering (cheapest, fastest)
│   └── Not working well enough?
│       └── Few-shot → Chain-of-thought → then consider below
│
├── No, knowledge changes frequently
│   └── RAG (retrieve at inference time)
│       └── Not working well enough?
│           └── Better chunking → Better embeddings → Reranking → Hybrid search
│
├── No, knowledge is static and domain-specific
│   └── Fine-tuning
│       ├── Small dataset (< 1K examples) → LoRA / QLoRA
│       ├── Medium dataset (1K-100K) → Full fine-tune of small model
│       └── Large dataset (100K+) → Continued pretraining then fine-tune
│
└── Need the model to behave differently (tone, format, reasoning)
    └── RLHF / DPO / GRPO (alignment)
```

---

## 3. Single Model vs Multi-Agent

```
How complex is the task?
├── Single, well-defined task → One model, one prompt
├── Multiple steps, same domain → Chain-of-thought or workflow
├── Multiple steps, different domains → Multi-agent
│   ├── Agents need to share state? → Shared memory / context
│   ├── Agents need to negotiate? → A2A protocol
│   └── Need reliability? → Orchestrator pattern with fallbacks
└── User-facing, conversational → Single agent with tools
```

---

## 4. Model Evaluation: "Is this model actually working?"

| What to check | What it means | Red flag |
|---|---|---|
| Training loss going down | Model is learning | Plateaus early → wrong LR |
| Validation loss going down | Model generalizes | Train ↓ but val flat → overfitting |
| Val loss goes up while train drops | Classic overfitting | Need regularization, dropout, more data, simpler model |
| Both losses flat | Not learning | LR too low, data issue, or architecture wrong |
| Accuracy high but precision/recall bad | Class imbalance hiding poor performance | Check confusion matrix, use F1 or AUC-ROC |
| Metrics great on test set | Could be data leakage | Check if test data leaked into training |

---

## 5. ML Debugging Decision Tree

```
Model isn't learning (loss flat/NaN)
├── Loss is NaN
│   ├── Learning rate too high → reduce by 10x
│   ├── Data has NaN/inf values → check preprocessing
│   └── Numerical overflow → use mixed precision or gradient clipping
│
├── Loss is flat
│   ├── Learning rate too low → increase by 10x
│   ├── Model too simple → add capacity
│   ├── Data isn't shuffled → shuffle it
│   └── Bug in data loading → verify a batch manually
│
├── Loss decreases then plateaus early
│   ├── Need LR schedule → cosine decay or warmup
│   ├── Local minimum → different init or optimizer (Adam vs SGD)
│   └── Data is the bottleneck → need more/better data
│
└── Training works but bad in production
    ├── Data drift → production data ≠ training data
    ├── Train/test mismatch → check your split
    ├── Overfitting → simplify model, add regularization
    └── Label quality → garbage labels = garbage model
```

---

## 6. Key Debugging Intuitions

| Intuition | Why it matters |
|---|---|
| **Learning rate is the most important hyperparameter** | Too high = explodes, too low = never converges. Start with 3e-4 for Adam |
| **Always look at your data first** | 90% of ML bugs are data bugs, not model bugs |
| **Overfit on a tiny batch first** | If model can't memorize 10 examples, something is broken |
| **Simpler models first** | Baseline with logistic regression before building a transformer |
| **More data beats a better model** | Almost always true |

---

## 7. Architecture Sanity Check

When an agent gives you a model architecture, verify:

- [ ] Input shape matches your data
- [ ] Output shape matches your task (1 neuron for binary, N for N classes)
- [ ] Parameter count is reasonable for your data size (not overfitting risk)
- [ ] Regularization exists (dropout, weight decay)
- [ ] Loss function matches the task (CrossEntropy for classification, MSE for regression)
- [ ] Optimizer is appropriate (Adam for most things, SGD+momentum for large-scale)
- [ ] Learning rate is in a sane range (1e-5 to 1e-2)

---

## 8. Production Deployment Checklist

```
Before deploying a model:
├── Model quality
│   ├── Meets baseline performance on held-out test set
│   ├── Performance audited across subgroups (fairness)
│   ├── Error analysis completed — know where it fails
│   └── Compared against simpler alternatives
│
├── Data pipeline
│   ├── Training data pipeline is reproducible
│   ├── Feature computation matches between training and serving
│   ├── Data validation in place (schema, distribution checks)
│   └── Handles missing data and edge cases
│
├── Serving
│   ├── Latency meets requirements
│   ├── Throughput handles expected load
│   ├── Fallback behavior defined (what happens when model fails?)
│   └── A/B testing infrastructure ready
│
└── Monitoring
    ├── Model performance metrics tracked over time
    ├── Data drift detection in place
    ├── Alerting for performance degradation
    └── Retraining triggers defined
```
