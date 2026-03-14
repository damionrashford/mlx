# LLM Pretraining: Building Intelligence from Raw Text

## Learning Objectives

By the end of this lesson, you will:

**Understand:**
- The causal language modeling objective and why it is used for generative LLMs instead of masked language modeling
- How Chinchilla scaling laws relate compute budget, model size, and training tokens, and why the industry now over-trains smaller models
- Why data quality, deduplication, and data mix composition matter more than raw data quantity

**Apply:**
- Use the decision framework to determine whether pretraining, continued pretraining, or fine-tuning is appropriate for a given use case
- Estimate GPU memory requirements for training a model of a given size with AdamW

**Analyze:**
- Evaluate the tradeoffs between pretraining from scratch, continued pretraining, and fine-tuning given specific data availability and compute constraints

## Prerequisites

- **Transformers and Attention** -- Understanding of the transformer architecture, self-attention, and the causal attention mask is essential since pretraining optimizes through these mechanisms (see [Transformers and Attention](../02-neural-networks/transformers-attention/COURSE.md))
- **Optimization** -- Familiarity with gradient descent, learning rate schedules, and optimizers like AdamW, since pretraining stability depends on these techniques (see [Optimization](../01-foundations/optimization/COURSE.md))

## What Is Pretraining?

Pretraining is the foundational phase where a language model learns the statistical structure of language by reading massive amounts of text. The model starts with random weights and, through billions of gradient updates, develops an internal representation of grammar, facts, reasoning patterns, and even some degree of common sense.

The core idea is deceptively simple: **predict the next token**. Given a sequence of tokens, what comes next? That's it. Every capability you see in GPT-4 or Claude — code generation, translation, reasoning, following instructions — emerges from optimizing this single objective at enormous scale.

```
Input:  "The capital of France is"
Target: "Paris"

Input:  "def fibonacci(n):\n    if n <= 1:\n        return"
Target: " n"
```

The model never receives explicit labels like "this is a geography fact" or "this is Python code." It learns everything implicitly through prediction.

## The Training Objective: Causal Language Modeling

The specific objective is **causal language modeling** (CLM), also called autoregressive language modeling. For a sequence of tokens (x_1, x_2, ..., x_T), the model maximizes:

```
L = sum over t: log P(x_t | x_1, ..., x_{t-1})
```

Key properties of CLM:

- **Unidirectional**: Each token can only attend to previous tokens (not future ones). This is enforced by the causal attention mask — an upper-triangular matrix of -infinity values.
- **Self-supervised**: The training signal comes from the data itself. No human labeling required.
- **Teacher forcing**: During training, the model always receives the ground truth previous tokens, not its own predictions. This makes training stable but creates a train/inference mismatch.

Why causal and not bidirectional (like BERT's masked language modeling)? Because generation is inherently left-to-right. If you want a model that can *write*, it needs to be trained to predict what comes next, not what's masked in the middle.

```
Causal (GPT-style):   [The] [cat] [sat] → predicts [on]
Masked (BERT-style):  [The] [MASK] [sat] [on] → predicts [cat]
```

BERT-style is great for understanding (classification, NER). GPT-style is what powers generation.

## Data Curation: What Do Models Read?

The quality and composition of pretraining data is arguably the most important factor in model quality — more important than architecture tweaks or training tricks.

### Major Datasets

| Dataset | Size | Sources | Notes |
|---------|------|---------|-------|
| CommonCrawl | ~250B pages | Web scrape | Raw internet. Mostly garbage without filtering |
| The Pile | 825 GB | 22 diverse sources | EleutherAI. Academic papers, books, GitHub, StackExchange |
| RedPajama v2 | 30T tokens | Web + curated | Open reproduction of LLaMA training data |
| RefinedWeb | 5T tokens | Filtered CommonCrawl | Falcon's dataset. Aggressive dedup + quality filtering |
| FineWeb | 15T tokens | Filtered CommonCrawl | HuggingFace. State-of-art web filtering pipeline |
| Dolma | 3T tokens | Web + academic + code | AI2's open dataset for OLMo |

### Data Mix Matters Enormously

LLaMA's training mix (approximate):
- 67% CommonCrawl (filtered)
- 15% C4 (cleaned CommonCrawl)
- 4.5% GitHub code
- 4.5% Wikipedia
- 4.5% Books
- 2% ArXiv
- 2% StackExchange

This mix is a deliberate choice. More code in training improves reasoning ability (not just coding). More books improve long-range coherence. Wikipedia improves factual accuracy. The ratio of each source is a hyperparameter that teams spend months tuning.

---

### Check Your Understanding

1. What is teacher forcing in causal language modeling, and what mismatch does it create between training and inference?
2. Why does the composition of the data mix (e.g., percentage of code, books, and web data) matter for model capabilities beyond the specific domains included?
3. A model is pretrained with 67% web data, 4.5% code, and 4.5% Wikipedia. If you wanted to improve the model's reasoning ability without increasing total training data, what change to the mix would you consider, and why?

<details>
<summary>Answers</summary>

1. Teacher forcing means the model always receives the ground truth previous tokens during training, not its own predictions. At inference, the model must use its own generated tokens as input, creating a train/inference mismatch where errors can compound since the model never practiced recovering from its own mistakes.
2. Code training improves reasoning ability (not just coding), books improve long-range coherence, and Wikipedia improves factual accuracy. The data mix affects emergent capabilities beyond the literal domain of each source.
3. Increase the proportion of code data. Research has shown that more code in training improves general reasoning ability, not just code generation. LLaMA 3 upsampled code and math data in later training stages for exactly this reason.

</details>

---

## Data Quality: The Unsexy Secret

The single biggest lesson from 2023-2024 in LLM training: **data quality beats data quantity**. Phi-1.5 (1.3B params) trained on "textbook quality" data outperformed models 10x its size on reasoning benchmarks.

### Deduplication

Duplicate data is poison. If the same paragraph appears 100 times in your corpus, the model memorizes it verbatim instead of learning generalizable patterns.

Types of deduplication:
- **Exact dedup**: Hash each document, remove exact copies. Fast but misses near-duplicates.
- **MinHash / LSH**: Approximate similarity. Documents with >80% n-gram overlap are flagged. This is what most serious efforts use.
- **Substring dedup**: Remove documents sharing long substrings (e.g., boilerplate headers/footers).

RefinedWeb showed that aggressive deduplication of CommonCrawl alone can match curated multi-source datasets.

### Quality Filtering

Common filtering steps:
1. **Language detection**: Remove non-target-language documents (or bucket by language).
2. **Perplexity filtering**: Use a small model trained on high-quality text. Documents with very high perplexity (gibberish) or very low perplexity (repetitive boilerplate) get filtered.
3. **Heuristic rules**: Remove documents with too many special characters, too-short sentences, too many repeated lines, bad word ratios, etc.
4. **Classifier filtering**: Train a classifier on "good" vs "bad" text (Wikipedia = good, random web scrape = bad). Use it to score and threshold.
5. **PII removal**: Strip emails, phone numbers, addresses. Both for privacy and to prevent memorization.

### Contamination

A critical concern: if your evaluation benchmarks appear in your training data, your results are meaningless. Benchmark contamination is a real problem with web-scraped data. Careful teams check for n-gram overlap between training data and common benchmarks (MMLU, HumanEval, GSM8K, etc.).

## Tokenization (Brief)

Before text enters the model, it must be converted to integers. Modern LLMs use **subword tokenization** — typically BPE (Byte Pair Encoding) or SentencePiece.

```
"unhappiness" → ["un", "happiness"]  (or ["un", "happ", "iness"] depending on vocab)
```

Vocabulary sizes: GPT-2 used 50K tokens, LLaMA uses 32K, GPT-4 uses ~100K. Larger vocabularies mean each token carries more information (fewer tokens per sentence) but require more embedding parameters.

The tokenizer is trained on the pretraining corpus before model training begins. It's frozen during training. See the dedicated tokenization lesson for the full picture.

## Scaling Laws: How Big Should Your Model Be?

### The Chinchilla Result

In 2022, DeepMind's Chinchilla paper changed how the industry thinks about training. The key finding:

**For a fixed compute budget, you should scale model size and training tokens equally.**

Prior to Chinchilla, the trend (Kaplan et al., 2020) suggested making models as large as possible and training them on "enough" data. This led to models like GPT-3 (175B params, 300B tokens) — massively over-parameterized relative to training data.

Chinchilla showed that a 70B model trained on 1.4T tokens outperforms a 175B model trained on 300B tokens, using roughly the same compute. The model was smaller but saw more data.

The Chinchilla-optimal ratio: **~20 tokens per parameter**. So a 7B model should see ~140B tokens. A 70B model should see ~1.4T tokens.

### Post-Chinchilla Reality

In practice, teams now over-train smaller models well beyond the Chinchilla-optimal point. Why? Because inference cost depends on model size, not training tokens. LLaMA 7B was trained on 1T tokens (140x parameters instead of 20x). The extra training compute is a one-time cost; the smaller model saves on every inference forever.

The practical rule: **train the smallest model that meets your quality bar, and over-train it.**

### Scaling Law Formulas

Loss as a function of compute, parameters, and data:

```
L(N, D) = E + A/N^alpha + B/D^beta

Where:
  N = number of parameters
  D = number of training tokens
  E = irreducible entropy of language (~1.69 nats for English)
  alpha ~ 0.34, beta ~ 0.28 (fitted constants)
```

This tells you: double the parameters and loss drops by ~21%. Double the data and loss drops by ~18%. Both matter, but parameters give slightly more bang per buck (until Chinchilla showed you need both).

---

### Check Your Understanding

1. The Chinchilla-optimal ratio is approximately 20 tokens per parameter. A 7B model should therefore see ~140B tokens. Why do teams like Meta train LLaMA 7B on 1T tokens (7x the Chinchilla-optimal amount)?
2. What is benchmark contamination and why is it a serious problem for evaluating pretrained models?
3. Phi-1.5 (1.3B parameters) outperformed models 10x its size on reasoning benchmarks. What was the key factor that enabled this?

<details>
<summary>Answers</summary>

1. Inference cost depends on model size, not on how much training data was used. The extra training compute is a one-time cost, but the smaller model saves on every inference call forever. Over-training a smaller model is cheaper in total cost of ownership than running a larger Chinchilla-optimal model at scale.
2. Benchmark contamination occurs when evaluation benchmark questions or answers appear in the training data. The model memorizes answers rather than demonstrating genuine capability, making evaluation results meaningless. Teams must check for n-gram overlap between training data and benchmarks like MMLU, HumanEval, and GSM8K.
3. Data quality. Phi-1.5 was trained on "textbook quality" data -- carefully curated, high-quality text that taught concepts clearly. This demonstrated that data quality can substitute for model scale.

</details>

---

## Training Infrastructure

### Hardware

Pretraining a frontier model requires thousands of GPUs running for weeks to months.

| Model | GPUs | Training Time | Approx Cost |
|-------|------|---------------|-------------|
| LLaMA 65B | 2048 A100-80GB | 21 days | ~$2M |
| LLaMA 2 70B | 2000 A100-80GB | ~35 days | ~$3M |
| Llama 3 405B | 16,384 H100 | ~54 days | ~$50M+ |
| GPT-4 | ~25,000 A100 (estimated) | ~90 days | ~$50-100M |

### Parallelism Strategies

A single GPU can't hold a 70B model, let alone train it. You need parallelism:

**Data Parallelism (DP)**
- Each GPU holds a full copy of the model
- Each GPU processes a different batch of data
- Gradients are averaged across GPUs (AllReduce)
- Easy to implement, scales well, but each GPU needs to hold the full model

**Tensor Parallelism (TP)**
- Split individual weight matrices across GPUs
- E.g., a 4096x4096 matrix becomes four 4096x1024 slices on 4 GPUs
- Requires high-bandwidth interconnect (NVLink) — very communication-heavy
- Typically used within a single node (8 GPUs)

**Pipeline Parallelism (PP)**
- Split layers across GPUs: GPU 0 gets layers 0-15, GPU 1 gets layers 16-31, etc.
- Data flows through GPUs sequentially
- Problem: "pipeline bubbles" — GPUs sit idle waiting for their turn
- Micro-batching reduces bubbles but adds complexity

**Fully Sharded Data Parallelism (FSDP / ZeRO)**
- Shards optimizer states, gradients, and parameters across GPUs
- Each GPU holds only a fraction of the model
- Parameters are gathered just-in-time for each forward/backward pass
- Best memory efficiency, moderate communication overhead
- This is what most teams actually use (via DeepSpeed ZeRO or PyTorch FSDP)

### Training Stability

Large-scale training is notoriously unstable:
- **Loss spikes**: Sudden jumps in training loss. Often caused by bad data batches or numerical instability. Teams checkpoint frequently and roll back when spikes occur.
- **Learning rate warmup**: Start with a tiny learning rate, ramp up linearly over ~2000 steps, then cosine decay. This prevents early instability.
- **Gradient clipping**: Cap gradient norms (typically at 1.0) to prevent explosions.
- **BFloat16**: Use bf16 mixed precision — same range as fp32 but lower precision. More numerically stable than fp16 for training.

## Cost of Pretraining

This is why most people should NOT pretrain:

| Approach | Typical Cost | When to Use |
|----------|-------------|-------------|
| Pretraining from scratch (7B) | $500K - $2M | You have unique data at massive scale |
| Pretraining from scratch (70B) | $3M - $10M | You're a well-funded AI lab |
| Frontier model (400B+) | $50M - $500M+ | You're OpenAI, Google, or Anthropic |
| Continued pretraining | $10K - $200K | Domain adaptation (medical, legal, code) |
| Fine-tuning (LoRA) | $10 - $1K | Most people should start here |
| Prompt engineering | $0 | Start here. Seriously. |

The right question isn't "can I pretrain?" but "do I need to pretrain?"

## Continued Pretraining

Continued pretraining (also called domain-adaptive pretraining) takes an existing base model and trains it further on domain-specific data using the same CLM objective.

**When it works:**
- You have a large corpus (>1B tokens) of domain-specific text
- The domain has specialized vocabulary or reasoning patterns (medicine, law, finance)
- You need the model to deeply understand domain conventions, not just follow instructions about them

**Examples:**
- BloombergGPT: Continued pretraining on financial data
- CodeLlama: Continued pretraining of LLaMA on code
- Med-PaLM: Domain-adapted PaLM for medical knowledge

**Key decisions:**
- Learning rate: Much lower than initial pretraining (typically 10-50x lower)
- Data mix: Include some general data to prevent catastrophic forgetting (typically 10-30% general + 70-90% domain)
- Duration: Depends on domain corpus size, but typically 50B-500B additional tokens

---

### Check Your Understanding

1. A single A100-80GB GPU cannot hold a 70B model for training. Name two parallelism strategies and explain when each is preferred.
2. What is the purpose of learning rate warmup in large-scale pretraining?
3. What is FSDP (Fully Sharded Data Parallelism) and why is it the most commonly used parallelism strategy?

<details>
<summary>Answers</summary>

1. Tensor Parallelism (TP) splits individual weight matrices across GPUs and is preferred within a single node where high-bandwidth NVLink interconnects are available. Pipeline Parallelism (PP) assigns different layers to different GPUs and is suitable for cross-node setups with lower bandwidth, though it introduces pipeline bubbles. Data Parallelism (DP) is another option where each GPU holds a full model copy and processes different batches.
2. Starting with a tiny learning rate and ramping up linearly over ~2000 steps prevents early training instability. At initialization, model weights are essentially random, and large learning rates would cause gradient explosions or divergence before the model has learned basic structure.
3. FSDP shards optimizer states, gradients, and model parameters across all GPUs, so each GPU holds only a fraction of the model. Parameters are gathered just-in-time for each forward/backward pass. It provides the best memory efficiency while maintaining moderate communication overhead, making it practical for very large models.

</details>

---

## Curriculum Learning

Not all data is equally useful at every stage of training. Curriculum learning orders the training data strategically:

1. **Simple to Complex**: Start with clean, simple text (Wikipedia, textbooks). Introduce harder content later (code, math, technical papers).
2. **High quality to Mixed quality**: Train on curated data first, then gradually introduce noisier web data.
3. **Domain scheduling**: Introduce domain-specific data at specific phases.

LLaMA 3 used a sophisticated curriculum: code and math data were upsampled in later stages of training, which improved reasoning without sacrificing general capabilities.

In practice, most teams do a simpler version: carefully control the data mix ratios and adjust them during training based on evaluation metrics.

## Decision Framework: Should You Pretrain?

```
Do you need a model that deeply understands a specialized domain?
|-- No --> Don't pretrain. Fine-tune or prompt-engineer.
|-- Yes -->
|   |-- Do you have >10B tokens of domain text?
|   |   |-- No --> Don't pretrain. Fine-tune instead.
|   |   |-- Yes -->
|   |       |-- Can you afford $100K+ and weeks of GPU time?
|   |       |   |-- No --> Continued pretraining on a smaller base model
|   |       |   |-- Yes --> Consider pretraining or continued pretraining
|   |       |-- Is there an existing domain model you can build on?
|   |           |-- Yes --> Start from that, continued pretrain or fine-tune
|   |           |-- No --> Continued pretrain from a strong open base model
```

For an AI shopping assistant: you would **not** pretrain. E-commerce language is well-represented in existing models. You'd fine-tune on shopping conversations and product data, and use RAG for real-time catalog information.

## Common Pitfalls

1. **Pretraining when you should not.** The most common mistake is assuming you need to pretrain. For the vast majority of applications, fine-tuning or prompt engineering on an existing base model is far more cost-effective. Only pretrain if you have a truly unique domain with massive data (>10B tokens) and significant compute budget.
2. **Underestimating the importance of deduplication.** Duplicate data causes memorization instead of generalization. RefinedWeb demonstrated that aggressive deduplication of CommonCrawl alone can match curated multi-source datasets. Always deduplicate before training.
3. **Using the same learning rate throughout training.** Large-scale pretraining requires warmup (linear ramp over ~2000 steps) followed by cosine decay. Skipping warmup leads to early instability and loss spikes. Using too high a constant learning rate causes divergence.
4. **Ignoring data mix ratios.** Treating all training data equally leads to suboptimal results. The ratio of code, books, web, and domain-specific data is a critical hyperparameter. More code improves reasoning; more books improve coherence; Wikipedia improves factual accuracy.

## Hands-On Exercises

### Exercise 1: Estimate Training Compute (15 min)

Using the Chinchilla scaling law formula and the cost table from this lesson, answer:

1. You have a budget of $500K and access to A100-80GB GPUs at $2/hour. How many GPU-hours can you afford?
2. Using the Chinchilla-optimal ratio (20 tokens per parameter), what is the largest model you could train on 200B tokens? On 1T tokens?
3. If you instead over-train a 7B model to 1T tokens (LLaMA-style), estimate the training time on 2048 A100 GPUs, given that LLaMA 65B took 21 days on the same hardware. (Hint: the 7B model is ~9x smaller, so roughly 9x fewer FLOPs per step, but you are training on more tokens.)

### Exercise 2: Inspect Pretraining Data Quality (20 min)

Use HuggingFace datasets to load a small sample of a web-scraped dataset and evaluate its quality.

```python
# Install: pip install datasets
from datasets import load_dataset

# Load a sample of FineWeb or RedPajama
ds = load_dataset("HuggingFaceFW/fineweb", split="train", streaming=True)

# Examine 20 samples. For each, note:
# 1. Is this high-quality text you would want a model to learn from?
# 2. Is there boilerplate (navigation menus, cookie notices, ads)?
# 3. Is there potential PII (emails, phone numbers)?
# 4. Would you filter this document? Why or why not?
# Tally: what percentage of raw web data would you keep?
```

## Key Papers

1. **"Language Models are Unsupervised Multitask Learners"** (Radford et al., 2019) — GPT-2. Showed that pretraining alone produces surprisingly capable models.
2. **"Language Models are Few-Shot Learners"** (Brown et al., 2020) — GPT-3. Demonstrated emergent in-context learning at scale.
3. **"Training Compute-Optimal Large Language Models"** (Hoffmann et al., 2022) — Chinchilla. The scaling laws paper that changed how we think about training budgets.
4. **"LLaMA: Open and Efficient Foundation Language Models"** (Touvron et al., 2023) — Proved that smaller models trained on more data can compete with larger models.
5. **"Textbooks Are All You Need"** (Gunasekar et al., 2023) — Phi-1.5. Showed that data quality can substitute for model scale.
6. **"The Pile: An 800GB Dataset of Diverse Text"** (Gao et al., 2020) — Influential open dataset design.
7. **"FineWeb: decanting the web for the finest text data"** (Penedo et al., 2024) — State of the art web data filtering.

## Interview Questions

**Conceptual:**
1. What is the pretraining objective for GPT-style models? Why causal and not bidirectional?
2. Explain the Chinchilla scaling laws. If you had a fixed compute budget, how would you allocate it between model size and training tokens?
3. Why has the industry moved toward training smaller models for longer, despite Chinchilla suggesting equal scaling?
4. What's the difference between pretraining from scratch and continued pretraining? When would you choose each?
5. Why does data deduplication improve model quality? What happens if you don't deduplicate?

**Applied:**
6. You're building a code assistant for a proprietary programming language used only at your company. You have 500M tokens of code in this language. What's your training strategy?
7. Your pretraining run shows a sudden loss spike at step 50,000 out of 100,000. What do you do?
8. You notice your model is great at English but terrible at French, even though 10% of your training data is French. What might be wrong?
9. How would you detect benchmark contamination in a pretraining dataset?
10. Calculate: How much GPU memory do you need to train a 7B parameter model in bf16 with AdamW optimizer? (Hint: model weights + optimizer states + gradients)

**Answer to Q10**: 7B params x 2 bytes (bf16) = 14 GB for weights. AdamW stores 2 copies of moments in fp32: 7B x 4 bytes x 2 = 56 GB. Gradients in bf16: 14 GB. Total: ~84 GB minimum, plus activations. That's why you need FSDP or ZeRO to distribute this across GPUs.

## Summary

This lesson covered the foundational phase of LLM development. Key takeaways:

- **Causal language modeling** (predict the next token) is the universal pretraining objective for generative LLMs. All capabilities emerge from this single objective at scale.
- **Data quality beats data quantity.** Deduplication, quality filtering, and careful data mix composition are the most impactful levers for model quality.
- **Chinchilla scaling laws** showed that compute should be split equally between model size and training tokens. In practice, teams over-train smaller models because inference cost scales with model size, not training duration.
- **Training infrastructure** requires parallelism strategies (FSDP/ZeRO, tensor parallelism, pipeline parallelism) because no single GPU can hold a large model.
- **Continued pretraining** on domain-specific data is the practical middle ground between full pretraining and fine-tuning, useful when you have large domain corpora.
- **Most practitioners should not pretrain.** Start with prompt engineering, then fine-tuning, and only consider pretraining with unique data at massive scale and significant budget.

## What's Next

The next lesson, **Fine-Tuning** (see [Fine-Tuning](../fine-tuning/COURSE.md)), covers how to adapt a pretrained model to specific tasks and behaviors using supervised fine-tuning, LoRA, and QLoRA. Fine-tuning builds directly on the pretrained model produced by the process described in this lesson.
