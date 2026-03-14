# Quantization for ML Engineering

## Learning Objectives

By the end of this lesson, you will:

**Understand:**
- How quantization maps floating-point weights to lower-precision integer formats (INT8, INT4, NF4) and the quality-compression tradeoffs at each level
- The difference between post-training quantization (PTQ) and quantization-aware training (QAT), and when to use each
- How GPTQ, AWQ, GGUF, and bitsandbytes implement quantization with different tradeoff profiles

**Apply:**
- Select the appropriate quantization format and method for a given deployment scenario (production serving, edge deployment, fine-tuning, research)
- Use QLoRA to fine-tune large models on consumer-grade GPUs by combining 4-bit quantization with LoRA adapters

**Analyze:**
- Evaluate the cost-quality tradeoff of deploying a quantized model vs a full-precision model on different GPU configurations, considering VRAM, throughput, and task-specific quality sensitivity

---

## Prerequisites

Before starting this lesson, you should be comfortable with:
- GPU memory calculations from [GPU Computing Fundamentals](../gpu-fundamentals/COURSE.md), including VRAM budgeting for weights, gradients, and optimizer states across precision formats
- Inference optimization concepts from [Inference Optimization](../../03-llm-internals/inference-optimization/COURSE.md), including KV cache management and serving considerations

---

## What Quantization Is

Quantization reduces the numerical precision of a model's weights (and sometimes activations) from high-precision formats (FP32, FP16) to lower-precision formats (INT8, INT4). A 7B parameter model in FP16 takes 14 GB of VRAM. The same model in INT4 takes 3.5 GB — a 4x reduction.

This is not approximation in the vague sense. Quantization is a precise mathematical mapping: take a range of floating-point values, divide that range into a fixed number of discrete levels (256 for INT8, 16 for INT4), and map each weight to the nearest level. The model loses some numerical precision but gains dramatic reductions in memory, storage, and compute requirements.

Quantization has become the single most important technique for deploying large language models in production. Without it, serving a 70B model requires 2-4 expensive GPUs. With INT4 quantization, it fits on a single GPU. The cost savings are enormous.

---

## Number Formats: What Each Means

### FP32 (32-bit Floating Point)

- 1 sign bit, 8 exponent bits, 23 mantissa bits
- Range: ±3.4 × 10^38
- Precision: ~7 decimal digits
- Memory per parameter: 4 bytes
- The "full precision" standard for training

### FP16 (16-bit Floating Point)

- 1 sign bit, 5 exponent bits, 10 mantissa bits
- Range: ±65,504
- Precision: ~3.3 decimal digits
- Memory per parameter: 2 bytes
- Half the memory of FP32, 2-3x faster on Tensor Cores
- Problem: limited range causes overflow/underflow during training (needs loss scaling)

### BF16 (Brain Float 16)

- 1 sign bit, 8 exponent bits, 7 mantissa bits
- Range: same as FP32 (±3.4 × 10^38)
- Precision: ~2.4 decimal digits (less than FP16)
- Memory per parameter: 2 bytes
- Preferred for training: same range as FP32, so no loss scaling needed
- Developed by Google Brain, supported on A100+ GPUs

### INT8 (8-bit Integer)

- 8 bits, no exponent
- Range: -128 to 127 (signed) or 0 to 255 (unsigned)
- Memory per parameter: 1 byte (4x smaller than FP32)
- Each INT8 value represents a quantized weight: `weight = scale * int8_value + zero_point`
- Quality loss: minimal for most models (< 1% accuracy drop)

### INT4 (4-bit Integer)

- 4 bits, no exponent
- Range: -8 to 7 (signed) or 0 to 15 (unsigned)
- Memory per parameter: 0.5 bytes (8x smaller than FP32)
- Quality loss: noticeable but acceptable for most LLM tasks
- The sweet spot for LLM deployment — best tradeoff of size vs quality

### NF4 (Normal Float 4)

- 4-bit format where the 16 quantization levels are optimally spaced for normally-distributed weights
- Neural network weights are approximately normally distributed, so NF4 places more levels near zero (where most weights are) and fewer at the extremes
- Used by QLoRA — marginally better than uniform INT4 for neural network weights
- Memory per parameter: 0.5 bytes

### FP8 (8-bit Floating Point)

- Two variants: E4M3 (4 exponent, 3 mantissa) and E5M2 (5 exponent, 2 mantissa)
- Supported natively on H100 GPUs
- Used for training (not just inference) — H100 can train in FP8 with minimal quality loss
- Emerging standard for efficient training on next-gen hardware

---

### Check Your Understanding: Number Formats

**1. Why does BF16 not require loss scaling during training while FP16 does?**

<details>
<summary>Answer</summary>

BF16 has 8 exponent bits, giving it the same value range as FP32 (up to ~3.4 x 10^38). Gradients rarely fall outside this range. FP16 has only 5 exponent bits with a maximum value of 65,504, and small gradients can underflow to zero. Loss scaling artificially inflates the loss (and thus gradients) to keep them within FP16's representable range, then scales them back down before the weight update.
</details>

**2. INT4 has only 16 discrete levels. How can a model with billions of parameters maintain quality with so few levels per weight?**

<details>
<summary>Answer</summary>

Two key factors: (1) Group-wise quantization -- weights are quantized in small groups (e.g., 128 weights per group), each with its own scale and zero-point. This means different parts of the model can use different mappings to the 16 levels, greatly increasing effective precision. (2) Larger models have more redundancy -- the collective behavior of billions of parameters is robust to small per-weight errors. A 70B model in INT4 often outperforms a 13B model in FP16.
</details>

---

## Post-Training Quantization (PTQ)

PTQ quantizes a model after training is complete. You take a trained FP16 model and convert it to INT8 or INT4 without any additional training.

### How It Works

1. **Collect calibration data:** Run a small dataset (100-1000 samples) through the model to observe the range of values in each layer's weights and activations
2. **Determine scale and zero-point:** For each layer, compute the mapping from floating-point range to integer range
3. **Quantize weights:** Map each weight to the nearest integer value
4. **Store quantized model:** Save integer weights plus per-layer (or per-group) scale factors

### Weight-Only vs Weight-and-Activation Quantization

**Weight-only quantization:** Only weights are stored in low precision. During inference, weights are dequantized to FP16 on-the-fly for computation. This saves memory and storage but doesn't speed up compute as much.

**Weight-and-activation quantization (W8A8):** Both weights AND activations are quantized to INT8. The actual matrix multiplication happens in INT8, which is faster on hardware with INT8 support. More challenging because activation distributions vary with input.

### Quality Impact

| Format | Memory Reduction | Typical Quality Loss |
|--------|-----------------|---------------------|
| FP16 → INT8 | 2x | < 0.5% on most tasks |
| FP16 → INT4 | 4x | 1-3% on most tasks |
| FP16 → INT3 | 5.3x | 5-10%, noticeable |
| FP16 → INT2 | 8x | Significant, research only |

The quality loss is not uniform across tasks. Simple tasks (classification, sentiment) tolerate aggressive quantization. Complex tasks (math reasoning, code generation) are more sensitive.

---

## Quantization-Aware Training (QAT)

QAT simulates quantization during training. The model learns to compensate for the precision loss, producing weights that quantize more cleanly.

### How It Works

1. **Insert fake quantization nodes:** During forward pass, weights are quantized to INT8/INT4 and then dequantized back to FP16. The computation uses the dequantized (lossy) values.
2. **Backward pass uses straight-through estimator:** Gradients flow through the quantization nodes as if they weren't there (because rounding has zero gradient almost everywhere).
3. **Training continues normally:** The model adjusts its weights to work well despite the quantization.
4. **After training:** Export the quantized model. Since the model was trained with quantization simulation, the quality loss is minimal.

### QAT vs PTQ

| Aspect | PTQ | QAT |
|--------|-----|-----|
| Training required? | No | Yes (additional fine-tuning) |
| Quality at INT8 | Good | Slightly better |
| Quality at INT4 | Acceptable | Better |
| Speed | Minutes | Hours to days |
| When to use | Default choice | When PTQ quality is insufficient |

For most LLM deployments, PTQ with good algorithms (GPTQ, AWQ) is sufficient. QAT is used when every fraction of a percent matters, or for very aggressive quantization (INT4, INT2).

---

### Check Your Understanding: PTQ and QAT

**1. Weight-only quantization saves memory but may not speed up computation. Why?**

<details>
<summary>Answer</summary>

In weight-only quantization, integer weights are dequantized to FP16 on-the-fly before the matrix multiplication, which still executes in FP16. The speedup comes only from reduced memory bandwidth (loading smaller weights from VRAM). For full compute speedup, both weights AND activations must be quantized (W8A8), so the matrix multiplication itself runs in INT8 using specialized hardware.
</details>

**2. Why does QAT generally produce better quality than PTQ at aggressive quantization levels like INT4?**

<details>
<summary>Answer</summary>

QAT inserts fake quantization during training, so the model learns weights that work well despite precision loss. The training process adjusts weights to minimize the impact of quantization rounding. PTQ applies quantization after training, so the weights were never optimized to handle the precision loss. At INT8, the difference is small. At INT4 and below, the quality gap widens because the model has more to compensate for.
</details>

---

## GPTQ: Weight-Only Quantization via Hessian

GPTQ (Generalized Post-Training Quantization) is the most popular PTQ method for LLMs. Published in 2022, it quantizes weights layer by layer using second-order information (the Hessian).

### Core Idea

When you quantize a weight, you introduce an error. GPTQ minimizes the total error by:
1. Quantizing one weight at a time
2. After quantizing each weight, adjusting the remaining (not yet quantized) weights to compensate
3. Using the Hessian (second derivative of the loss) to determine which adjustments minimize error

This "error compensation" is what makes GPTQ better than naive rounding.

### Practical Details

- Quantizes a 7B model in ~5 minutes on a single GPU
- Requires a small calibration dataset (128 samples is standard)
- Produces weight-only INT4 models with group-wise quantization (groups of 128 weights share one scale factor)
- Quality is very close to FP16, especially for 4-bit with group size 128

### Using GPTQ Models

GPTQ models are widely available on HuggingFace (search for "-GPTQ" suffix). Load them with:
- `auto-gptq` library
- `transformers` with `GPTQConfig`
- `vllm` for serving (natively supports GPTQ)

---

## AWQ: Activation-Aware Weight Quantization

AWQ (2023) takes a different approach: instead of trying to compensate for quantization errors, it identifies which weights matter most and protects them.

### Core Idea

Not all weights are equally important. AWQ observes that a small percentage of weights (corresponding to large activation magnitudes) have an outsized impact on model quality. AWQ:

1. Runs calibration data to identify activation magnitudes per channel
2. Scales up important weight channels (those with large activations) before quantization
3. Scales down the corresponding activations to maintain mathematical equivalence
4. Quantizes all weights uniformly (but the important ones now have higher relative precision because they were scaled up)

### AWQ vs GPTQ

| Aspect | GPTQ | AWQ |
|--------|------|-----|
| Approach | Error compensation | Importance-aware scaling |
| Speed | Faster quantization | Slightly slower quantization |
| Quality at INT4 | Very good | Slightly better |
| Inference speed | Standard | Faster (better kernel support) |
| Popularity | Most models available | Growing rapidly |

In practice, the quality difference is small. AWQ models tend to have faster inference because the format is simpler for kernel optimization.

---

## GGUF/GGML: Local Quantized Models

GGUF (GPT-Generated Unified Format) is the format used by llama.cpp for running quantized models on consumer hardware, including CPUs.

### Why GGUF Matters

GGUF models run on:
- CPUs (no GPU required)
- Apple Silicon (Metal acceleration)
- Consumer GPUs with limited VRAM
- Any hardware via llama.cpp, ollama, or LM Studio

This is how most people run LLMs locally. A 70B model quantized to Q4_K_M (4-bit with mixed precision) is about 40 GB — it can run on a MacBook Pro with 64 GB of unified memory.

### GGUF Quantization Types

| Type | Bits/Weight | Quality | Notes |
|------|-------------|---------|-------|
| Q8_0 | 8.5 | Excellent | Barely worse than FP16 |
| Q6_K | 6.6 | Very good | |
| Q5_K_M | 5.7 | Good | Good balance for most users |
| Q4_K_M | 4.8 | Good | Most popular, recommended default |
| Q4_0 | 4.5 | Acceptable | Faster but lower quality than Q4_K_M |
| Q3_K_M | 3.9 | Noticeable loss | For memory-constrained setups |
| Q2_K | 3.4 | Significant loss | Last resort |

The "_K_" variants use k-quant, which applies different precision to different layers (more precision for important layers like the first and last). "_M" means medium, balancing quality and size.

### Creating GGUF Models

```
# Conceptual flow:
# 1. Start with HuggingFace model (FP16 safetensors)
# 2. Convert to GGUF format
# 3. Quantize to desired precision
python convert_hf_to_gguf.py model_dir --outfile model.gguf
./quantize model.gguf model-q4_k_m.gguf Q4_K_M
```

---

### Check Your Understanding: GPTQ, AWQ, and GGUF

**1. How does GPTQ's error compensation differ from AWQ's importance-aware scaling?**

<details>
<summary>Answer</summary>

GPTQ quantizes weights one at a time and adjusts the remaining unquantized weights to compensate for the error introduced by each quantization step, using the Hessian (second derivative) to determine optimal adjustments. AWQ takes a different approach: it identifies important weight channels by observing activation magnitudes during calibration, scales those channels up before quantization (so they get higher relative precision), and scales activations down correspondingly. GPTQ corrects errors after they occur; AWQ prevents the most damaging errors from occurring.
</details>

**2. What makes GGUF models different from GPTQ/AWQ models in terms of deployment targets?**

<details>
<summary>Answer</summary>

GGUF models are designed for llama.cpp and run on CPUs (no GPU required), Apple Silicon (Metal), and consumer GPUs. They support mixed-precision quantization where different layers get different bit widths. GPTQ and AWQ models are GPU-centric, requiring CUDA for efficient inference. GGUF enables local deployment on laptops and edge devices, while GPTQ/AWQ target server-side GPU deployment.
</details>

---

## bitsandbytes: Easy Quantization in Python

bitsandbytes is the simplest way to load quantized models in Python/PyTorch. Created by Tim Dettmers, it provides:

### 8-bit Quantization (LLM.int8())

Load any HuggingFace model in 8-bit:
```python
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3-8B",
    load_in_8bit=True,
    device_map="auto"
)
```

Key innovation: LLM.int8() identifies outlier features (activation channels with very large values) and keeps them in FP16 while quantizing everything else to INT8. This preserves quality even for models sensitive to quantization.

### 4-bit Quantization (NF4)

Load any model in 4-bit with NF4:
```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3-8B",
    quantization_config=bnb_config,
    device_map="auto"
)
```

`double_quant=True` quantizes the quantization constants themselves, saving an additional ~0.4 bits per parameter.

---

## QLoRA: The Efficient Fine-Tuning Recipe

QLoRA combines quantization with LoRA to enable fine-tuning large models on consumer hardware. It is the most important practical technique for accessible fine-tuning.

### The Recipe

1. **Load base model in 4-bit (NF4):** Base weights are frozen and stored in 4-bit
2. **Add LoRA adapters in BF16:** Small trainable matrices (rank 16-64) added to attention layers
3. **Train only LoRA parameters:** Gradients only flow through the small adapters
4. **Optimizer states only for LoRA:** Adam states for ~20M parameters, not 7B

### Memory Breakdown for 7B Model with QLoRA

- Base weights (NF4): 7B × 0.5 = 3.5 GB
- LoRA adapters (BF16): ~20M × 2 = 40 MB
- LoRA gradients: ~40 MB
- LoRA optimizer (Adam): ~20M × 8 = 160 MB
- Activations: ~4-8 GB
- **Total: ~8-12 GB → fits on RTX 3090/4090 (24 GB)**

### Quality

QLoRA matches full fine-tuning quality in most benchmarks. The 4-bit base model introduces some noise, but the LoRA adapters are trained in full precision and can compensate. The original QLoRA paper showed that fine-tuning a 65B model with QLoRA on a single 48GB GPU produced results competitive with full 16-bit fine-tuning on a multi-GPU setup.

### When NOT to Use QLoRA

- Pre-training from scratch (you need to update all weights)
- When you need to modify the model architecture
- Tasks where base model quality at 4-bit is insufficient (rare but possible for math-heavy tasks)

---

## Quality vs Compression Tradeoffs

### General Rules

1. **INT8 barely hurts.** For nearly all tasks and models, INT8 quantization preserves 99%+ of the original quality. It should be the default for any deployment.

2. **INT4 is noticeable but usually acceptable.** Quality drops 1-3% on most benchmarks. For conversational AI, summarization, and simple reasoning, users rarely notice. For math, code, and complex reasoning, the drop is more significant.

3. **Below INT4 is research territory.** INT3 and INT2 quantization show significant quality degradation. Active research area but not production-ready for most applications.

4. **Larger models quantize better.** A 70B model in INT4 typically outperforms a 13B model in FP16. The larger model has more redundancy that quantization can exploit.

5. **Quantization affects different capabilities unevenly.** Knowledge recall (factual questions) is robust to quantization. Mathematical reasoning degrades more. Creative writing is in between.

### Decision Framework

| Scenario | Recommended Format | Why |
|----------|-------------------|-----|
| Production serving, cost-sensitive | INT4 (AWQ/GPTQ) | 4x memory reduction, minor quality loss |
| Production serving, quality-critical | INT8 | 2x memory reduction, negligible quality loss |
| Edge/mobile deployment | INT4 or lower | Memory is extremely constrained |
| Fine-tuning base | NF4 via QLoRA | Maximum memory efficiency for training |
| Research/evaluation | FP16/BF16 | No quantization noise in measurements |
| Training | BF16 (or FP8 on H100) | Best quality-performance tradeoff |

---

## Serving Quantized Models

### vLLM

vLLM is the most popular inference engine for LLMs and natively supports GPTQ and AWQ models:
- Automatic PagedAttention for efficient KV cache management
- Continuous batching for high throughput
- Tensor parallelism for multi-GPU serving
- Load GPTQ/AWQ models with zero code changes

### TGI (Text Generation Inference)

HuggingFace's TGI supports GPTQ, AWQ, and bitsandbytes quantization. It's optimized for production deployment with features like health checks, metrics, and request queuing.

### llama.cpp / Ollama

For local or edge deployment:
- llama.cpp: C++ inference engine, runs GGUF models on CPU or GPU
- Ollama: user-friendly wrapper around llama.cpp, one-command model serving
- Both support all GGUF quantization levels

---

## Interview Questions

**Q: A stakeholder asks you to deploy a 70B parameter model for a customer-facing chatbot. You have a budget for 2 A100 80GB GPUs. How do you approach this?**

Strong answer: "In FP16, 70B requires ~140 GB, which just fits on 2x A100 80GB with tensor parallelism. But that leaves almost no room for the KV cache, limiting concurrent users. Instead, I'd quantize to INT4 using AWQ or GPTQ — that brings the model to ~35 GB, fitting comfortably on a single A100 with ample room for the KV cache and batching. I'd serve it with vLLM, which supports AWQ natively and provides continuous batching for high throughput. The second GPU becomes either a spare for redundancy or handles a second model replica for higher throughput. INT4 quantization on a 70B model produces quality very close to FP16 — larger models are more robust to quantization."

**Q: Explain the difference between GPTQ and AWQ.**

Strong answer: "Both are post-training weight quantization methods for INT4. GPTQ works by quantizing weights one at a time and adjusting remaining weights to compensate for the quantization error, using Hessian information to determine optimal adjustments. AWQ takes a different approach — it identifies important weight channels by looking at activation magnitudes, then scales those channels up before quantization so they get higher relative precision. In practice, AWQ tends to produce slightly better quality and faster inference due to simpler kernel implementations, while GPTQ is faster to quantize and has more models available. Both are excellent choices for production deployment."

**Q: When would you choose QLoRA over full fine-tuning?**

Strong answer: "Almost always, unless you have unlimited GPU budget and need to squeeze out the last fraction of a percent of quality. QLoRA lets me fine-tune a 7B model on a single 24GB GPU and a 70B model on a single 80GB GPU. Full fine-tuning of 7B needs ~120 GB (two A100s), and 70B needs a full 8-GPU node. The QLoRA paper showed quality is within 1% of full fine-tuning for most tasks. The exceptions are: pre-training from scratch (QLoRA doesn't apply), tasks where 4-bit base model quality is insufficient (rare), or when you need to modify all parameters for a specific architecture change."

---

## Key Takeaways

1. Quantization reduces model precision to cut memory 2-8x with acceptable quality loss
2. INT8 is nearly lossless. INT4 is the sweet spot for deployment. Below INT4 is research territory.
3. GPTQ and AWQ are the two leading INT4 methods — both are production-ready
4. GGUF/llama.cpp enables running quantized models on consumer hardware including CPUs
5. QLoRA (4-bit base + LoRA adapters) is the standard for efficient fine-tuning
6. Larger models are more robust to quantization — 70B in INT4 often beats 13B in FP16
7. bitsandbytes makes quantization a one-line change in HuggingFace code
8. vLLM is the standard for serving quantized models in production

---

## Common Pitfalls

**1. Quantizing without calibration data.** PTQ methods (GPTQ, AWQ) require a small calibration dataset to observe weight and activation ranges. Using no calibration or unrepresentative calibration data (e.g., random noise instead of real text) produces poor quantization parameters and degraded quality. Use 128-1000 representative samples.

**2. Assuming all tasks degrade equally under quantization.** Knowledge recall and simple classification are robust to INT4. Mathematical reasoning and code generation are more sensitive. Always benchmark on your specific task -- do not rely on general perplexity numbers alone.

**3. Using INT4 quantization for training instead of inference.** INT4 is a deployment optimization. Training requires higher precision for gradient computation and weight updates. For training, use QLoRA (4-bit frozen base + BF16 adapters) or BF16/FP32 mixed precision, not raw INT4 for all computation.

**4. Overlooking the KV cache when sizing inference memory.** Quantizing model weights to INT4 reduces the static model footprint, but the KV cache (which stores attention keys and values for each token in the sequence) is typically kept in FP16 and grows with batch size and sequence length. For long-context or high-throughput serving, the KV cache can exceed the quantized model size.

---

## Hands-On Exercises

### Exercise: Deployment Planning

A product team wants to deploy a 70B parameter model for a customer-facing application. They have the following hardware options:
- Option A: 2x A100 80GB GPUs
- Option B: 1x A100 80GB GPU
- Option C: 1x L40S 48GB GPU

For each option, determine:
1. What quantization format is needed (FP16, INT8, INT4)?
2. How much VRAM remains for KV cache and batching?
3. What serving framework would you use?
4. What is the expected quality impact?

### Exercise: QLoRA Memory Budget

Calculate the VRAM requirements for fine-tuning a 13B parameter model using QLoRA with:
- NF4 base weights with double quantization
- LoRA rank 32, applied to all attention layers (assume 40M trainable parameters)
- Adam optimizer for LoRA parameters only
- Batch size 1, sequence length 4096

Determine whether this fits on an RTX 4090 (24 GB) and what adjustments are needed if it does not.

---

## Summary

This lesson covered quantization as the key technique for efficient model deployment: number format fundamentals (FP32/FP16/BF16/INT8/INT4/NF4/FP8), post-training quantization and quantization-aware training, the GPTQ and AWQ methods for INT4 weight quantization, GGUF for local deployment, bitsandbytes for easy Python integration, QLoRA for memory-efficient fine-tuning, quality-compression tradeoffs, and serving infrastructure (vLLM, TGI, llama.cpp).

### What's Next

With the distributed GPU and quantization foundation complete, you now have the full picture of how ML models are trained at scale and deployed efficiently. These concepts connect back to the production ML topics in earlier modules: the data engineering pipelines (modules 5-6) feed into the training infrastructure covered here, and the quantized models you produce are served using the deployment patterns from production ML.
