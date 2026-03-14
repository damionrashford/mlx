# Inference Optimization: Serving LLMs at Scale

## Learning Objectives

By the end of this lesson, you will:

**Understand:**
- How the KV cache eliminates redundant computation during autoregressive generation and why it is often the memory bottleneck rather than model weights
- How Flash Attention reduces memory from O(n^2) to O(n) through tiling and online softmax, and how Paged Attention solves KV cache memory fragmentation
- The key differences between quantization formats (GPTQ, AWQ, GGUF) and when each is appropriate

**Apply:**
- Calculate KV cache memory requirements for a given model architecture, context length, and batch size
- Design a cost optimization strategy using model routing, prompt caching, and quantization for a production serving workload

**Analyze:**
- Evaluate the three-way tradeoff between latency, throughput, and cost for different inference optimization combinations, and choose the right balance for a given use case

## Prerequisites

- **Transformers and Attention** -- Understanding of multi-head attention, key/value/query projections, and the attention computation is essential since KV caching, Flash Attention, and GQA all modify how attention operates (see [Transformers and Attention](../02-neural-networks/transformers-attention/COURSE.md))
- **Pretraining** -- Familiarity with model architecture parameters (layers, heads, hidden dimensions) and the autoregressive generation process, since inference optimization targets these operations directly (see [Pretraining](../pretraining/COURSE.md))

## Why Inference Optimization Matters

Training happens once. Inference happens millions of times. For a production system like an AI personal shopper — serving recommendations to hundreds of millions of shoppers on a large-scale e-commerce platform — the cost and latency of every inference call matters enormously.

The numbers are stark:
- GPT-4 class model: ~$0.01-0.03 per response (API pricing)
- At 10M queries/day: $100K-$300K/day in API costs
- Self-hosted without optimization: even worse (GPU costs + underutilization)
- With proper optimization: 3-10x cost reduction

Inference optimization is the difference between "cool demo" and "viable product."

## How Autoregressive Generation Works

To understand optimizations, you need to understand the baseline. LLM generation is inherently sequential:

```
Step 1: Process entire prompt at once           → output token 1
Step 2: Process prompt + token 1               → output token 2
Step 3: Process prompt + token 1 + token 2     → output token 3
...
Step N: Process everything so far               → output token N
```

Each step requires a full forward pass through the model. The naive approach recomputes attention over ALL previous tokens at every step. For a 1000-token response, that's a lot of redundant computation.

This is where the KV cache comes in.

## KV Cache: The Fundamental Optimization

### What It Stores

During the attention computation, each token produces Key (K) and Value (V) vectors for every attention head in every layer. These K and V vectors don't change once computed — they depend only on the token's position and the tokens before it.

The KV cache stores these K and V vectors so they don't need to be recomputed at each generation step.

```
Without KV cache (step 100):
  Recompute K, V for all 100 tokens across all layers → generate token 101

With KV cache (step 100):
  Look up cached K, V for tokens 1-99
  Compute K, V only for token 100
  Attend to all cached K, V → generate token 101
```

This transforms generation from O(n^2) to O(n) per step in compute (though memory is still O(n) for the cache itself).

### KV Cache Memory Math

This is a common interview question. Let's work through it:

```
KV cache size per token = 2 (K and V) * n_layers * n_kv_heads * d_head * bytes_per_element

For Llama 3.1 8B:
  n_layers = 32
  n_kv_heads = 8 (GQA — 8 KV heads shared across 32 query heads)
  d_head = 128
  bytes_per_element = 2 (fp16/bf16)

KV cache per token = 2 * 32 * 8 * 128 * 2 = 131,072 bytes = 128 KB per token

For a 4096-token context: 128 KB * 4096 = 512 MB per sequence
For 128 concurrent sequences: 512 MB * 128 = 64 GB just for KV cache!
```

This is why KV cache management is critical for serving. It's often the memory bottleneck, not the model weights.

**GQA (Grouped Query Attention)** reduces KV cache by sharing K/V heads across multiple query heads. Llama 3 uses 8 KV heads instead of 32, reducing KV cache by 4x. This is a major reason GQA became standard.

## Quantization for Inference

Quantization reduces the precision of model weights, trading a small quality loss for significant memory and speed savings.

### Precision Formats

| Format | Bits | Memory (7B model) | Relative Quality | Speed |
|--------|------|-------------------|------------------|-------|
| FP32 | 32 | 28 GB | Baseline | Baseline |
| FP16/BF16 | 16 | 14 GB | ~Same as FP32 | 2x faster |
| INT8 | 8 | 7 GB | 99%+ of FP16 | 2-4x faster |
| INT4 | 4 | 3.5 GB | 95-98% of FP16 | 3-6x faster |

### Quantization Methods

**GPTQ (Post-Training Quantization)**
- Quantizes weights to INT4 using calibration data
- Layer-by-layer: minimize the output error of each layer independently
- Fast inference, good quality
- Requires a calibration dataset (~128 samples)
- GPU-only (uses CUDA kernels for INT4 matmul)

**AWQ (Activation-Aware Weight Quantization)**
- Key insight: not all weights are equally important. Weights corresponding to large activations matter more.
- Protects important weights by scaling channels before quantization
- Often slightly better quality than GPTQ at the same bit-width
- GPU inference

**GGUF (llama.cpp format)**
- Designed for CPU and mixed CPU/GPU inference
- Supports many quantization levels: Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_0
- Optimized for Apple Silicon (Metal), x86 (AVX2), and CUDA
- The go-to format for local/edge inference
- Great for running models on laptops or consumer hardware

### When to Use Which

```
Serving on GPU cluster     → GPTQ or AWQ (INT4), or FP16 if memory allows
Running locally / edge     → GGUF (Q4_K_M is the sweet spot for quality/size)
Single GPU, need max qual  → FP16 or INT8
Single GPU, need to fit    → INT4 (GPTQ/AWQ)
CPU only                   → GGUF
```

---

### Check Your Understanding

1. For Llama 3.1 8B (32 layers, 8 KV heads, d_head=128, bf16), the KV cache per token is 128 KB. If you are serving 64 concurrent requests each with a 2048-token context, how much GPU memory is consumed by the KV cache alone?
2. Why does GQA (Grouped Query Attention) reduce KV cache memory? How much does Llama 3's choice of 8 KV heads (vs. 32 query heads) save?
3. GPTQ and AWQ both quantize weights to INT4, but AWQ often achieves slightly better quality. What is AWQ's key insight?

<details>
<summary>Answers</summary>

1. 128 KB/token x 2048 tokens = 256 MB per sequence. 256 MB x 64 concurrent requests = 16 GB consumed by KV cache alone. This is in addition to the model weights themselves (~14 GB in bf16 for an 8B model), showing why KV cache is often the memory bottleneck.
2. GQA shares K and V heads across multiple query heads. With 8 KV heads instead of 32, the KV cache is 4x smaller because only 8 sets of K/V vectors are stored per layer instead of 32. This reduces KV cache from 512 KB/token to 128 KB/token.
3. AWQ observes that not all weights are equally important -- weights corresponding to large activations have a disproportionate impact on output quality. AWQ protects these important weights by scaling channels before quantization, preserving the precision where it matters most.

</details>

---

## Continuous Batching

### The Problem with Static Batching

Traditional batching waits until a batch is full, processes all requests together, and waits for the longest request to finish before returning any results.

```
Static batching:
  Request 1: 50 tokens  -------->  [done, waiting...]
  Request 2: 200 tokens -------------------------------->
  Request 3: 30 tokens  ----->     [done, waiting.........]

  All return at step 200. Requests 1 and 3 wasted GPU cycles.
```

### Continuous Batching

Also called "iteration-level batching." New requests join the batch as old ones finish. No wasted cycles.

```
Continuous batching:
  Request 1: 50 tokens  -------> [returns immediately]
  Request 4: ----------- [joins when slot opens]
  Request 2: 200 tokens -------------------------------->
  Request 3: 30 tokens  ----> [returns immediately]
  Request 5: ---------------- [joins when slot opens]
```

This maximizes GPU utilization and reduces latency for shorter requests. Every major serving framework (vLLM, TGI) implements this.

## Speculative Decoding

A clever trick: use a small, fast model to "draft" multiple tokens, then verify them in parallel with the large model.

### How It Works

```
1. Draft model (e.g., 1B params) generates 5 tokens quickly:
   "The best running shoe for"  →  "long distance is the Nike"

2. Large model (e.g., 70B params) verifies all 5 tokens in a single forward pass:
   Token 1 "long"     → matches draft (accept)
   Token 2 "distance" → matches draft (accept)
   Token 3 "is"       → matches draft (accept)
   Token 4 "the"      → matches draft (accept)
   Token 5 "Nike"     → large model prefers "Brooks" (reject, use "Brooks")

3. Accept 4 tokens, fix the 5th. Generated 4 tokens in 2 forward passes instead of 4.
```

### The Math

If the draft model matches the large model ~70% of the time, and you draft 5 tokens per step:
- Expected accepted tokens: ~3.5 per step
- Cost: 1 small model forward pass + 1 large model forward pass per step
- Speedup: ~2-3x for well-matched draft/target pairs

The quality is **mathematically guaranteed** to be identical to the large model (it's a rejection sampling scheme). You're not sacrificing quality — just reducing the number of large-model forward passes.

### When It Works Best

- Draft model is very fast (small, same GPU)
- High acceptance rate (draft and target models agree often)
- Output is predictable (factual answers, code, structured output)
- Less effective for creative/diverse generation

## Flash Attention

Standard attention has O(n^2) memory complexity because it materializes the full attention matrix. For a 32K context, that's a 32K x 32K matrix per head per layer.

### The Flash Attention Insight

Flash Attention (Dao et al., 2022) computes attention without materializing the full matrix. It uses a tiling approach — processing blocks of the attention matrix that fit in GPU SRAM (fast memory), avoiding reads/writes to slower HBM (GPU DRAM).

```
Standard attention:
  1. Compute S = Q @ K^T        (n x n matrix in HBM — slow)
  2. Compute P = softmax(S)      (n x n matrix in HBM — slow)
  3. Compute O = P @ V           (read n x n from HBM — slow)
  Memory: O(n^2)

Flash Attention:
  1. Process in tiles that fit in SRAM
  2. Never materialize the full n x n matrix
  3. Use online softmax to accumulate results
  Memory: O(n) — only the output matrix
  Speed: 2-4x faster (memory-bound → compute-bound)
```

Flash Attention is now the default in virtually all inference frameworks. It's not optional — it's a prerequisite for handling long contexts efficiently.

### Flash Attention 2 and 3

- **FA2**: Better parallelism, support for GQA, ~2x faster than FA1
- **FA3**: Hopper GPU optimizations (H100), FP8 support, even faster

---

### Check Your Understanding

1. In continuous batching, why do shorter requests benefit from reduced latency compared to static batching?
2. Speculative decoding uses a small draft model to generate candidate tokens and a large model to verify them. Why is the output quality mathematically identical to using the large model alone?
3. Flash Attention never materializes the full n x n attention matrix. How does it compute the softmax correctly without seeing all the values at once?

<details>
<summary>Answers</summary>

1. In static batching, all requests in a batch wait for the longest request to complete before any results are returned. In continuous batching, each request returns immediately when it finishes, and its slot is filled by a new request. Short requests are no longer blocked by long ones.
2. Speculative decoding is a rejection sampling scheme. The large model verifies each draft token against its own probability distribution. If the draft token matches what the large model would have generated, it is accepted. If not, it is rejected and replaced with the large model's actual prediction. The final output only contains tokens approved by the large model.
3. Flash Attention uses an "online softmax" algorithm. It processes the attention matrix in tiles that fit in GPU SRAM, maintaining running statistics (max value and sum of exponentials) that allow it to compute the correct softmax incrementally without needing the full matrix in memory.

</details>

---

## Paged Attention (vLLM)

The KV cache memory problem: different requests have different sequence lengths, and we don't know in advance how long each will be. Naive allocation wastes memory through fragmentation.

### The Virtual Memory Analogy

vLLM applies the operating system concept of virtual memory to KV cache management:

```
Traditional KV cache allocation:
  Request 1: [allocated 4096 tokens] [used 500] [wasted: 3596 tokens of memory]
  Request 2: [allocated 4096 tokens] [used 2000] [wasted: 2096 tokens]
  Memory fragmentation: ~60% wasted

Paged Attention:
  Request 1: [page 1][page 2][page 3]  — 3 pages allocated, all used
  Request 2: [page 4][page 5][page 6][page 7][page 8] — 5 pages, all used
  Free pool: [page 9][page 10]...  — allocated on demand
  Memory fragmentation: near zero
```

Each "page" holds a fixed number of tokens' KV cache (e.g., 16 tokens per page). Pages are allocated on demand and can be non-contiguous in physical memory (just like OS virtual memory).

### Impact

Paged Attention typically increases serving throughput by **2-4x** compared to naive KV cache allocation, purely through better memory utilization. More concurrent sequences fit in GPU memory.

## Tensor Parallelism vs Pipeline Parallelism for Serving

When a model doesn't fit on a single GPU, you need to split it.

**Tensor Parallelism (TP)**
- Split each layer across GPUs
- Every GPU participates in every token's computation
- Low latency (all GPUs work in parallel per token)
- High bandwidth requirement (GPUs communicate every layer)
- Use within a node (NVLink: 600+ GB/s)

**Pipeline Parallelism (PP)**
- Assign different layers to different GPUs
- Lower bandwidth requirement (only activations passed between stages)
- Higher latency (sequential through stages)
- Can work across nodes

**For serving, TP is almost always preferred** because latency matters more than bandwidth efficiency. A single request's latency with PP is the sum of all stages. With TP, it's a single stage (all GPUs in parallel).

Typical setup for serving a 70B model: 4-8 GPUs with TP within a single node.

## Model Distillation

Train a smaller "student" model to mimic a larger "teacher" model. The student learns from the teacher's output probabilities (soft labels), not just the hard labels.

```
Teacher (70B): "The best laptop for programming is..." → probability distribution over vocab
Student (7B):  Trained to match the teacher's probability distribution, not just the top-1 token
```

### Why Soft Labels Help

The teacher's probability distribution contains more information than just the correct answer:
- "Macbook" = 0.3, "ThinkPad" = 0.25, "XPS" = 0.2 tells the student that all three are reasonable
- Hard labels ("Macbook" = 1.0) lose this nuance

### Distillation in Practice

```python
# Temperature scaling reveals more information in softmax
T = 3.0  # Higher T = softer distribution

loss = alpha * KL_div(
    student_logits / T,
    teacher_logits / T
) + (1 - alpha) * cross_entropy(student_logits, hard_labels)
```

Many production models are distilled: GPT-4o-mini, Claude 3 Haiku, Llama 3.2 1B/3B. Distillation is how you get small-model pricing with large-model quality (approximately).

## Serving Frameworks Comparison

| Framework | Key Feature | Best For |
|-----------|-------------|----------|
| **vLLM** | Paged Attention, continuous batching | High-throughput serving, production |
| **TGI** (HuggingFace) | Easy deployment, HF integration | Quick setup, HF model hub models |
| **TensorRT-LLM** (NVIDIA) | Maximum GPU optimization | NVIDIA hardware, maximum performance |
| **Triton Inference Server** | Multi-model, multi-framework | Complex serving pipelines |
| **llama.cpp** | CPU/edge inference, GGUF | Local deployment, Apple Silicon |
| **Ollama** | User-friendly local serving | Developer experimentation |
| **SGLang** | Structured generation, RadixAttention | Complex prompts, constraint decoding |

### For Large-Scale E-Commerce

You'd likely use vLLM or TensorRT-LLM behind an API gateway with:
- Auto-scaling based on request queue depth
- Multiple model sizes for different use cases (simple queries use small model, complex reasoning uses large model)
- Prompt caching for repeated system prompts
- Geographic distribution for latency-sensitive markets

---

### Check Your Understanding

1. Paged Attention reduces memory fragmentation from ~60% wasted to near zero. How does it achieve this, drawing on the analogy to operating system virtual memory?
2. For serving a 70B model, why is tensor parallelism preferred over pipeline parallelism?
3. In model distillation, why do "soft labels" (the teacher's full probability distribution) contain more information than "hard labels" (just the correct token)?

<details>
<summary>Answers</summary>

1. Instead of pre-allocating a fixed maximum-length KV cache for each request (wasting memory on unused slots), Paged Attention allocates small fixed-size pages on demand. Pages can be non-contiguous in physical GPU memory, just like virtual memory pages in an OS. As a sequence grows, new pages are allocated; when a sequence finishes, its pages are freed and reused. This eliminates both internal fragmentation (unused space within allocations) and external fragmentation (unusable gaps between allocations).
2. For serving, latency per request is critical. With pipeline parallelism, a single request's latency is the sum of all pipeline stages (sequential). With tensor parallelism, all GPUs work on the same token in parallel, so latency is determined by one stage plus communication overhead. TP also requires high-bandwidth NVLink, which is available within a single node.
3. The teacher's probability distribution encodes relative likelihoods of all candidates. For example, if the teacher assigns 0.3 to "Macbook", 0.25 to "ThinkPad", and 0.2 to "XPS", the student learns that all three are reasonable answers and their relative ranking. Hard labels (1.0 for "Macbook") lose all this information about alternative valid answers.

</details>

---

## Cost Optimization Strategies

### Model Routing

Not every query needs GPT-4. Route simple queries to cheaper models:

```
User: "What's your return policy?"                         → Small model (fast, cheap)
User: "Compare these 5 laptops for my specific needs"     → Large model (better reasoning)
```

A lightweight classifier (or even keyword rules) routes requests. This alone can cut costs 50-70%.

### Prompt Caching

Many requests share the same system prompt and few-shot examples. Cache the KV cache for the shared prefix:

```
System prompt (500 tokens): computed once, cached
User message (50 tokens):   computed per request

Without caching: 550 tokens processed per request
With caching:    50 tokens processed per request (10x cheaper for prefill)
```

Anthropic's prompt caching, OpenAI's equivalent — this is a massive optimization for production systems.

### Semantic Caching

If users ask similar questions, cache the response:
```
"What are the best headphones under $100?" → cached response
"Best headphones for under 100 dollars?"   → semantic match → return cached response
```

Use embedding similarity to match queries. Cache hit = zero inference cost.

### Output Length Control

Shorter outputs = fewer tokens = lower cost and latency. Set appropriate `max_tokens` for each use case. A product recommendation doesn't need 2000 tokens.

## Latency vs Throughput vs Cost

These three are in tension. Optimizing one often hurts another:

| Optimization | Latency | Throughput | Cost |
|-------------|---------|------------|------|
| Larger batch size | Worse | Better | Better |
| Quantization (INT4) | Better | Better | Better (rare win-win-win) |
| Speculative decoding | Better | Neutral | Slightly worse |
| Model routing | Better (small model) | Better | Much better |
| Longer max_tokens | Worse | Worse | Worse |
| More GPUs (TP) | Better | Better | Worse |

### Choose Based on Use Case

- **Real-time chat**: Optimize latency (streaming, speculative decoding, small models)
- **Batch processing**: Optimize throughput (large batches, maximize GPU utilization)
- **High-volume API**: Optimize cost (routing, caching, quantization)
- **AI shopping assistant**: Optimize latency AND cost (shoppers won't wait, and margins matter)

## E-Commerce Context: Serving Recommendations at Scale

Imagine serving an AI personal shopper across a large-scale merchant ecosystem:

**Scale**: Hundreds of millions of monthly shoppers across millions of stores.

**Architecture considerations**:
1. **Multi-tier model serving**: Small model for quick product lookups and simple Q&A. Large model for complex comparison shopping and personalized recommendations.
2. **Per-merchant context**: Each store has different products, policies, and brand voice. This is a RAG problem — retrieve merchant-specific context at inference time.
3. **Prompt caching**: The merchant-specific system prompt (brand voice, policies) is the same for every shopper at that store. Cache it aggressively.
4. **Edge caching**: Cache popular product recommendations per store. Most shoppers ask similar questions.
5. **Streaming**: Stream responses token-by-token. Perceived latency drops dramatically even if total generation time is the same.
6. **Fallback chains**: If the primary model is overloaded or slow, fall back to a simpler model or cached response rather than timing out.
7. **Monitoring**: Track latency P50, P95, P99. Track cost per query. Track quality metrics (user clicks on recommended products, conversion rates).

## Common Pitfalls

1. **Forgetting that KV cache is often the memory bottleneck, not model weights.** A 7B model in bf16 is 14 GB, but KV cache for 128 concurrent requests at 4096 tokens can exceed 64 GB. Always compute KV cache requirements when planning GPU memory budgets.
2. **Over-quantizing for the use case.** INT4 quantization is tempting for memory savings, but it introduces measurable quality degradation (95-98% of fp16 quality). For applications where accuracy is critical (medical, legal, financial), INT8 or fp16 may be worth the extra memory cost.
3. **Not implementing prompt caching for repeated system prompts.** If every request to a merchant's shopping assistant shares the same 500-token system prompt, processing those 500 tokens for every request is pure waste. Prompt caching eliminates this overhead and is one of the highest-ROI optimizations.
4. **Optimizing latency when throughput is the bottleneck (or vice versa).** These require different strategies. Speculative decoding and tensor parallelism reduce latency. Larger batch sizes and continuous batching improve throughput. Quantization and model routing improve cost. Identify which dimension matters most before optimizing.

## Hands-On Exercises

### Exercise 1: KV Cache Memory Calculator (15 min)

Build a simple calculator that computes KV cache memory requirements for different model configurations.

```python
def kv_cache_memory(
    n_layers: int,
    n_kv_heads: int,
    d_head: int,
    seq_len: int,
    batch_size: int,
    bytes_per_element: int = 2,  # bf16
) -> dict:
    """Calculate KV cache memory requirements."""
    per_token = 2 * n_layers * n_kv_heads * d_head * bytes_per_element
    per_sequence = per_token * seq_len
    total = per_sequence * batch_size
    return {
        "per_token_bytes": per_token,
        "per_sequence_mb": per_sequence / (1024 ** 2),
        "total_gb": total / (1024 ** 3),
    }

# Compute for these configurations:
# 1. Llama 3.1 8B (32 layers, 8 KV heads, d_head=128) at 4096 tokens, 32 concurrent
# 2. Llama 3.1 70B (80 layers, 8 KV heads, d_head=128) at 4096 tokens, 32 concurrent
# 3. Llama 3.1 70B at 128K tokens, 1 concurrent
# Question: At what context length does a single 70B request consume more KV cache
#           memory than the model weights themselves (140 GB in bf16)?
```

### Exercise 2: Model Routing Simulation (20 min)

Simulate a model routing system and measure cost savings.

```python
import random

# Simulate 1000 shopping assistant queries
queries = [
    {"text": "What is your return policy?", "complexity": "simple"},
    {"text": "Compare these 5 laptops for video editing under $1500", "complexity": "complex"},
    # ... generate more with a mix of 70% simple, 30% complex
]

# Pricing (per 1K tokens):
small_model_cost = 0.001   # e.g., 3B model
large_model_cost = 0.01    # e.g., 70B model

# Implement a router that classifies queries as simple/complex
# (use keyword heuristics: "compare", "recommend", "explain why" -> complex)
# Calculate total cost with routing vs. sending everything to the large model
# Question: What is the cost reduction percentage?
```

## Interview Questions

**Conceptual:**
1. Explain the KV cache. What does it store and why does it speed up generation?
2. Calculate the KV cache memory for Llama 3.1 70B with a 128K context window and 32 concurrent requests.
3. What is Flash Attention? Why does it reduce memory from O(n^2) to O(n)?
4. Explain Paged Attention. How does it improve over static KV cache allocation?
5. What is speculative decoding? Under what conditions does it provide the most speedup?

**Applied:**
6. You're serving a product recommendation model. Latency needs to be under 500ms. The model currently takes 2 seconds. What optimizations would you try, in order?
7. Design a model routing system for a shopping assistant. What criteria would you use to route between a small and large model?
8. Your vLLM deployment is running out of GPU memory under peak load. You can't add more GPUs. What do you do?
9. A merchant complains that the AI assistant is slower for their store than others. Their store has 50,000 products. Diagnose and fix.
10. Compare GPTQ and AWQ quantization. When would you choose one over the other?

**System Design:**
11. Design an LLM serving infrastructure for a large-scale e-commerce platform that handles 100K requests per minute across merchants globally. Consider latency, cost, reliability, and quality.
12. You need to serve the same base model with 1000 different LoRA adapters (one per merchant). How do you do this efficiently?

**Answer to Q2**: Llama 3.1 70B has 80 layers, 8 KV heads (GQA), d_head=128, bf16. Per token: 2 * 80 * 8 * 128 * 2 = 327,680 bytes = 320 KB. For 128K context: 320 KB * 128K = 40 GB per sequence. For 32 concurrent: 40 GB * 32 = 1.28 TB. This is why 128K context with many concurrent users is extremely expensive and requires aggressive optimization (shorter contexts, KV cache eviction, quantized KV cache).

**Answer to Q8**: In order: (1) Reduce max_seq_len if possible, (2) Enable KV cache quantization (INT8 KV cache), (3) Reduce max concurrent sequences, (4) Quantize model weights further (FP16 to INT8 or INT4), (5) Implement request queuing with priority, (6) Add prompt caching to reduce KV cache buildup for shared prefixes.

## Summary

This lesson covered the critical optimizations that make LLM serving viable at production scale. Key takeaways:

- **KV cache** is the fundamental inference optimization, transforming per-step compute from O(n^2) to O(n), but it becomes the primary memory bottleneck at scale. Always calculate KV cache requirements before planning deployments.
- **Quantization** (GPTQ, AWQ for GPU; GGUF for CPU/edge) reduces memory 2-8x with minimal quality loss. INT4 is the sweet spot for most serving workloads.
- **Continuous batching** maximizes GPU utilization by allowing requests to enter and exit the batch independently, eliminating idle GPU cycles.
- **Flash Attention** reduces attention memory from O(n^2) to O(n) through tiling and online softmax. It is now a non-negotiable prerequisite for long-context models.
- **Paged Attention** (vLLM) eliminates KV cache memory fragmentation, increasing serving throughput by 2-4x.
- **Speculative decoding** provides 2-3x speedup with mathematically identical output quality by using a small draft model verified by the large model.
- **Cost optimization** through model routing, prompt caching, and semantic caching can reduce inference costs by 50-70%.
- **Latency, throughput, and cost** are in tension. The right optimization strategy depends on which dimension matters most for your use case.

## What's Next

This is the final lesson in the LLM Internals module. You now have a complete picture of the LLM lifecycle: tokenization, pretraining, fine-tuning, alignment, and inference optimization. From here, you can explore more advanced topics in system design, retrieval-augmented generation, or multi-modal models, applying the foundational understanding built across these lessons.
