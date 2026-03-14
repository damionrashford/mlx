# GPU Computing Fundamentals for ML Engineering

## Learning Objectives

By the end of this lesson, you will:

**Understand:**
- Why GPUs provide 50-200x speedups over CPUs for deep learning workloads (SIMT architecture, Tensor Cores)
- The components of GPU memory consumption during training: weights, gradients, optimizer states, and activations
- The tradeoffs of memory optimization techniques: mixed precision, gradient checkpointing, and gradient accumulation

**Apply:**
- Estimate GPU memory requirements for training and inference of models ranging from 7B to 70B parameters
- Select appropriate GPU hardware and cloud providers based on workload requirements and budget constraints

**Analyze:**
- Determine the optimal combination of precision format, batch size, and memory optimization techniques for a given model and GPU configuration

---

## Prerequisites

Before starting this lesson, you should be comfortable with:
- Neural network training mechanics from [Training Mechanics](../../02-neural-networks/training-mechanics/COURSE.md), including forward/backward passes, gradient descent, and optimizer behavior (Adam)

---

## Why GPUs for Machine Learning

Machine learning — particularly deep learning — is dominated by one operation: matrix multiplication. Training a neural network means multiplying weight matrices by activation vectors millions of times per batch, across millions of batches. A single forward pass through a transformer with 7 billion parameters involves hundreds of matrix multiplications, each operating on tensors with millions of elements.

CPUs are designed for sequential, branching logic. A modern CPU has 8-64 cores, each capable of complex operations with deep pipelines, branch prediction, and large caches. This is perfect for running an operating system, a web server, or a database. It is terrible for multiplying two 4096x4096 matrices together, because that operation is 4096^3 = ~69 billion multiply-add operations with almost no branching.

GPUs flip the architecture. Instead of a few powerful cores, a GPU has thousands of simple cores organized into streaming multiprocessors (SMs). An NVIDIA H100 has 16,896 CUDA cores. Each core is weak individually — no branch prediction, tiny cache — but together they can perform thousands of floating-point operations simultaneously. A matrix multiplication that takes seconds on a CPU takes milliseconds on a GPU.

This is why every serious ML training job runs on GPUs. The speedup is not 2x or 5x — it is 50-200x for typical deep learning workloads.

---

## CPU vs GPU Architecture

### CPU: Few Powerful Cores

A CPU core is a general-purpose computing engine. It has:
- Deep instruction pipelines (20+ stages)
- Branch prediction units that guess which code path to take
- Large L1/L2/L3 caches (megabytes per core)
- Out-of-order execution to maximize instruction throughput
- Support for complex control flow (if/else, loops, function calls)

A modern server CPU (AMD EPYC 9654) has 96 cores, each running at 2.4-3.7 GHz. Total theoretical FP32 throughput: ~5 TFLOPS.

### GPU: Thousands of Simple Cores

A GPU core (CUDA core) is a simple arithmetic unit. It can:
- Multiply two numbers and add a third (fused multiply-add)
- Execute the same instruction across many data elements simultaneously (SIMT)
- Access fast but small shared memory within its streaming multiprocessor

An NVIDIA H100 has 16,896 CUDA cores plus 528 Tensor Cores (specialized for matrix math). Total theoretical FP32 throughput: ~67 TFLOPS. With Tensor Cores in FP16: ~990 TFLOPS.

### The Key Insight

ML workloads are embarrassingly parallel. Every element of a matrix multiplication can be computed independently. Every sample in a batch can flow through the network independently. GPUs exploit this parallelism — they trade single-thread performance for massive throughput.

The analogy: a CPU is a sports car (fast on a single road), a GPU is a highway system (moves thousands of cars simultaneously, each slowly).

---

## CUDA: NVIDIA's Parallel Computing Platform

CUDA (Compute Unified Device Architecture) is NVIDIA's programming model for GPUs. You don't need to write CUDA code as an ML engineer — PyTorch and TensorFlow handle that — but you need to understand what it does.

**Execution model:** Code is organized into kernels — functions that run on the GPU. A kernel launches thousands of threads organized into blocks, and blocks are organized into a grid. Each thread executes the same code on different data.

**Memory hierarchy:** GPU memory has multiple levels:
- Global memory (VRAM): large but slow (80GB on H100, ~2 TB/s bandwidth)
- Shared memory: small but fast (per SM, ~200 KB, ~19 TB/s bandwidth)
- Registers: fastest (per thread)

**Why this matters:** When PyTorch says `tensor.cuda()`, it copies data from CPU RAM to GPU global memory. When you call `model(input)`, PyTorch launches CUDA kernels that perform the math on GPU. The results stay in GPU memory until you call `.cpu()`.

**cuDNN:** NVIDIA's library of optimized GPU implementations for common neural network operations (convolutions, batch normalization, attention). PyTorch calls cuDNN under the hood. This is why NVIDIA GPUs dominate ML — the software ecosystem is unmatched.

---

### Check Your Understanding: Architecture and CUDA

**1. Why are GPUs faster than CPUs for matrix multiplication but slower for complex branching logic?**

<details>
<summary>Answer</summary>

GPUs have thousands of simple cores that execute the same instruction on different data simultaneously (SIMT). Matrix multiplication is ideal because every element can be computed independently with no branching. CPUs have few cores but each has deep pipelines, branch prediction, and large caches optimized for complex sequential logic. The GPU trades single-thread capability for massive parallelism.
</details>

**2. What happens when you call `tensor.cuda()` in PyTorch?**

<details>
<summary>Answer</summary>

The tensor's data is copied from CPU RAM (system memory) to GPU global memory (VRAM). This is a data transfer over the PCIe bus. The tensor is now accessible to CUDA kernels for GPU computation. Results stay in GPU memory until explicitly moved back with `.cpu()`.
</details>

---

## GPU Memory (VRAM): The Primary Bottleneck

GPU compute is rarely the bottleneck in modern ML. Memory is. You will hit VRAM limits long before you max out compute.

### What Occupies VRAM During Training

For a model with P parameters:

**1. Model weights:** P x bytes_per_param
- FP32: P x 4 bytes
- FP16/BF16: P x 2 bytes
- A 7B parameter model in FP16: 7B x 2 = 14 GB

**2. Gradients:** Same size as weights
- One gradient value per parameter
- FP16: 7B x 2 = 14 GB

**3. Optimizer states:** Depends on optimizer
- SGD: momentum = P x 4 bytes (one buffer)
- Adam: mean + variance = P x 4 x 2 = P x 8 bytes
- Adam for 7B: 7B x 8 = 56 GB (kept in FP32 for numerical stability)

**4. Activations:** Depends on batch size and model architecture
- Every layer saves its output for the backward pass
- Scales linearly with batch size and sequence length
- For transformers: proportional to batch_size x seq_len x hidden_dim x num_layers
- Often 2-10 GB for typical training configurations

**5. Temporary buffers:** Communication buffers, workspace memory
- Typically 1-5 GB

### The Training Memory Formula

Total VRAM (Adam, FP16 mixed precision) ≈
- Weights: P x 2 bytes (FP16)
- Gradients: P x 2 bytes (FP16)
- Optimizer: P x 8 bytes (FP32 master weights + mean + variance)
- Activations: variable

For 7B parameters: 14 + 14 + 56 + activations ≈ 84 GB + activations

This is why you cannot fine-tune a 7B model on a single consumer GPU (24 GB).

### Inference Memory

Inference is much lighter — no gradients, no optimizer states:
- FP16: P x 2 bytes + KV cache
- 7B in FP16: ~14 GB + KV cache (fits on a single A100 40GB easily)
- 70B in FP16: ~140 GB (needs multiple GPUs or quantization)

---

## Memory Optimization Techniques

### Gradient Checkpointing (Activation Recomputation)

Normally, the forward pass saves every layer's activations for the backward pass. This consumes enormous memory for deep models.

Gradient checkpointing trades compute for memory: save activations at only a few "checkpoint" layers, and recompute the intermediate activations during the backward pass. This reduces activation memory from O(n) to O(sqrt(n)) layers, at the cost of ~33% more compute.

In PyTorch: `model.gradient_checkpointing_enable()` — one line and your activation memory drops dramatically.

### Mixed Precision Training

Instead of FP32 everywhere, use FP16 or BF16 for most operations:
- Forward pass in FP16/BF16
- Backward pass in FP16/BF16
- Master weights in FP32 (for numerical stability)
- Loss scaling to prevent gradients from underflowing in FP16

Benefits:
- Halve memory for weights and activations
- 2-3x faster compute on Tensor Cores
- Minimal impact on model quality

BF16 vs FP16: BF16 has the same exponent range as FP32 (so no loss scaling needed) but less precision. FP16 has more precision but smaller range (needs loss scaling). BF16 is preferred on modern hardware (A100, H100).

In PyTorch: `torch.cuda.amp.autocast()` or Hugging Face Trainer with `bf16=True`.

### Gradient Accumulation

Instead of one large batch, process multiple small batches and accumulate gradients before updating:
- Effective batch size = micro_batch_size x accumulation_steps
- Each micro-batch fits in VRAM
- Mathematically equivalent to a larger batch (ignoring batch norm)

This doesn't reduce peak memory per step, but lets you achieve large effective batch sizes without the memory cost.

---

### Check Your Understanding: Memory and Optimization

**1. For a 7B parameter model trained with Adam in FP16 mixed precision, which component uses the most VRAM: weights, gradients, or optimizer states?**

<details>
<summary>Answer</summary>

Optimizer states. Adam maintains two buffers (mean and variance) in FP32, plus the FP32 master weights copy: 7B x 4 x 3 = 84 GB. Weights in FP16 are 14 GB and gradients in FP16 are 14 GB. The optimizer states are roughly 3x larger than weights and gradients combined.
</details>

**2. Why is BF16 generally preferred over FP16 for training?**

<details>
<summary>Answer</summary>

BF16 has the same exponent range as FP32 (8 exponent bits), so it can represent the same range of values without overflow or underflow. FP16 has only 5 exponent bits and a much smaller range, requiring loss scaling to prevent gradient underflow. BF16 eliminates this complexity while providing the same 2x memory savings and Tensor Core speedup. The tradeoff is that BF16 has less mantissa precision (7 bits vs 10), but this rarely impacts training quality.
</details>

**3. How does gradient checkpointing reduce memory at the cost of compute?**

<details>
<summary>Answer</summary>

Normally, the forward pass saves every layer's activations for the backward pass, consuming O(n) memory for n layers. Gradient checkpointing saves activations at only a few checkpoint layers and recomputes intermediate activations during the backward pass. This reduces activation memory to O(sqrt(n)) but requires re-running the forward pass for uncheckpointed layers, adding roughly 33% more compute time.
</details>

---

## GPU Hardware Comparison

### Key Specifications That Matter

| GPU | VRAM | FP16 TFLOPS | Interconnect | Use Case |
|-----|------|-------------|--------------|----------|
| RTX 4090 | 24 GB | 165 | PCIe 4.0 | Consumer, inference, small fine-tuning |
| A100 40GB | 40 GB | 312 | NVLink 600 GB/s | Training standard, widely available |
| A100 80GB | 80 GB | 312 | NVLink 600 GB/s | Training larger models |
| H100 SXM | 80 GB | 990 | NVLink 900 GB/s | Current top training GPU |
| L40S | 48 GB | 362 | PCIe 4.0 | Inference-optimized, good value |
| H200 | 141 GB | 990 | NVLink 900 GB/s | Maximum memory, next-gen training |

### What Matters Most

**VRAM** is the most important spec for ML. You either fit your workload in memory or you don't. More VRAM = larger models, larger batches, less need for memory optimization tricks.

**Interconnect** matters for multi-GPU training. NVLink provides 600-900 GB/s between GPUs on the same node. PCIe provides 32-64 GB/s. The difference is 10-20x — this matters enormously for data parallelism (gradient synchronization) and model parallelism (tensor communication).

**Tensor Core throughput** determines how fast matrix multiplications execute. H100 Tensor Cores are ~3x faster than A100.

---

## Cloud GPU Pricing (Approximate, 2025-2026)

| Provider | GPU | On-Demand $/hr | Spot $/hr |
|----------|-----|----------------|-----------|
| AWS (p5) | H100 | $32.77 | ~$13 |
| AWS (p4d) | A100 80GB | $32.77 (8x) | ~$13 |
| Google Cloud (a3) | H100 | $31.22 (8x) | ~$12 |
| Google Cloud (a2) | A100 80GB | $29.39 (8x) | ~$10 |
| Lambda Labs | H100 | $2.49/GPU | N/A |
| Lambda Labs | A100 80GB | $1.29/GPU | N/A |
| RunPod | H100 | $3.89/GPU | $2.69 |
| RunPod | A100 80GB | $1.64/GPU | $1.24 |

**Key insight:** Cloud providers (AWS, GCP) charge per node (8 GPUs). Smaller providers (Lambda, RunPod) charge per GPU and are dramatically cheaper for small-scale work. For production ML at scale, cloud providers offer better reliability, networking, and managed services.

**Cost optimization strategies:**
- Use spot/preemptible instances (60-70% cheaper) with checkpointing
- Right-size your GPU (don't rent H100s for inference that fits on L40S)
- Quantize models for inference (run on cheaper GPUs)
- Use reserved instances for long-running training jobs

---

## Practical: Estimating GPU Requirements

### Fine-tuning a 7B Model (e.g., Llama-3-8B)

Full fine-tuning with Adam:
- Weights (BF16): 7B x 2 = 14 GB
- Gradients (BF16): 7B x 2 = 14 GB
- Optimizer (FP32): 7B x 12 = 84 GB (master weights + mean + variance)
- Activations: ~10 GB (batch_size=1, seq_len=2048)
- **Total: ~122 GB → needs 2x A100 80GB**

QLoRA (4-bit quantized + LoRA adapters):
- Base weights (4-bit): 7B x 0.5 = 3.5 GB
- LoRA adapters (BF16): ~20M x 2 = 40 MB
- Gradients for LoRA only: ~40 MB
- Optimizer for LoRA only: ~240 MB
- Activations: ~5 GB
- **Total: ~9 GB → fits on a single RTX 4090 (24 GB)**

This is why QLoRA changed the game — it made fine-tuning accessible on consumer hardware.

### Fine-tuning a 70B Model

Full fine-tuning: ~1.2 TB → needs a full node of 8x H100 with model parallelism
QLoRA: ~38 GB → fits on a single A100 80GB or 2x RTX 4090s

### Inference for a 70B Model

FP16: ~140 GB → 2x A100 80GB
INT8: ~70 GB → 1x A100 80GB
INT4 (GPTQ/AWQ): ~35 GB → 1x A100 40GB or 2x RTX 4090

---

## Multi-GPU Basics: When One GPU Isn't Enough

You need multiple GPUs when:
1. **Model doesn't fit in one GPU's VRAM** (e.g., 70B in FP16 = 140 GB)
2. **Training is too slow** and you want to parallelize across data
3. **You need a larger effective batch size** for training stability

The simplest multi-GPU strategy is **Data Parallel**: replicate the model on each GPU, split the batch, and average gradients. This works when the model fits on a single GPU but you want faster training.

When the model doesn't fit on one GPU, you need **Model Parallel** strategies (covered in the next lesson on distributed training).

**Scaling efficiency:** Going from 1 to 8 GPUs on the same node (connected by NVLink) typically gives 6-7.5x speedup (75-94% efficiency). Going across nodes (connected by network) gives 4-6x speedup per 8 additional GPUs due to communication overhead.

---

## Key Takeaways

1. GPUs beat CPUs for ML by 50-200x because ML is matrix multiplication, which is massively parallel
2. VRAM is the bottleneck, not compute — always calculate memory requirements first
3. Training memory = weights + gradients + optimizer states + activations
4. Mixed precision (BF16) halves memory and doubles speed with minimal quality loss
5. Gradient checkpointing trades ~33% more compute for dramatically less activation memory
6. QLoRA reduces fine-tuning memory by 10-20x, making it accessible on consumer GPUs
7. Cloud GPU costs vary 5-10x between providers — choose based on your scale and requirements

---

## Self-Check Questions

1. A colleague wants to fine-tune Llama-3-70B. They have a single A100 80GB. What do you recommend?
2. Your model trains fine on 1 GPU but OOMs when you increase batch size from 4 to 32. What are three solutions that don't require more GPUs?
3. Why is BF16 generally preferred over FP16 for training?
4. An inference endpoint for a 13B model is too expensive on A100s. How would you reduce cost?
5. You have 8x H100 GPUs on one node. Your model fits on a single GPU. What's the simplest way to speed up training 6-7x?

---

## Common Pitfalls

**1. Calculating memory only for weights and forgetting optimizer states.** The most common memory estimation error. For Adam optimizer, optimizer states (master weights + mean + variance in FP32) consume 12 bytes per parameter -- often 3-6x more than the weights themselves. Always include all four components: weights, gradients, optimizer states, and activations.

**2. Assuming more TFLOPS means faster training.** Raw compute throughput matters less than VRAM capacity and memory bandwidth for most training workloads. A GPU with more TFLOPS but less VRAM will hit OOM errors before it can use its compute advantage. VRAM is the first constraint to check.

**3. Not using mixed precision.** Training in FP32 when BF16 is available wastes half your memory and 2-3x of your Tensor Core throughput. On A100 and H100 hardware, always use BF16 mixed precision unless you have a specific numerical stability reason not to.

**4. Renting expensive GPUs for inference that fits on cheaper hardware.** A 7B model in INT4 (3.5 GB) does not need an A100. An L40S or even an RTX 4090 handles it easily at a fraction of the cost. Always quantize models for inference and right-size the GPU.

---

## Hands-On Exercises

### Exercise: Memory Budget Calculation

Calculate the total VRAM required for training a 13B parameter model with:
- BF16 mixed precision
- Adam optimizer
- Batch size of 4, sequence length 2048, hidden dimension 5120, 40 layers
- With and without gradient checkpointing

Determine which single GPU (RTX 4090, A100 40GB, A100 80GB, H100 80GB) can handle this workload, and which memory optimizations are needed for each.

### Exercise: GPU Selection and Cost Analysis

Your team needs to fine-tune a 7B model using QLoRA and then serve the quantized model for inference. Compare the cost of:
1. Using a cloud provider (AWS p5 H100 nodes) for both training and inference
2. Using a budget provider (Lambda Labs A100) for training and a smaller GPU for inference
3. Estimate the monthly cost for each option assuming 40 hours of training and 24/7 inference serving

---

## Summary

This lesson covered the hardware foundations of ML computing: GPU architecture and why it accelerates deep learning, CUDA's execution and memory model, the detailed breakdown of VRAM consumption during training and inference, memory optimization techniques (mixed precision, gradient checkpointing, gradient accumulation), GPU hardware specifications and pricing, and practical methods for estimating requirements and selecting hardware.

### What's Next

Continue to [Distributed Training](../distributed-training/COURSE.md) to learn how to scale beyond a single GPU using data parallelism, model parallelism, FSDP, and DeepSpeed -- the techniques used to train models that are too large or too slow for one GPU.
