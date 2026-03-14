# Distributed Training for ML Engineering

## Learning Objectives

By the end of this lesson, you will:

**Understand:**
- The three categories of parallelism (data, tensor, pipeline) and when each is appropriate based on model size and hardware topology
- How FSDP and DeepSpeed ZeRO shard weights, gradients, and optimizer states to reduce per-GPU memory requirements
- The communication primitives (all-reduce, all-gather, reduce-scatter) that underpin distributed training and their cost characteristics

**Apply:**
- Select the correct distributed training strategy (DDP, FSDP, DeepSpeed, 3D parallelism) given a model size, GPU count, and hardware configuration
- Configure gradient accumulation to achieve target effective batch sizes within memory constraints

**Analyze:**
- Diagnose scaling inefficiencies in multi-GPU and multi-node training by reasoning about communication overhead, pipeline bubbles, and hardware topology mismatches

---

## Prerequisites

Before starting this lesson, you should be comfortable with:
- GPU memory calculations and optimization techniques from [GPU Computing Fundamentals](../gpu-fundamentals/COURSE.md), including VRAM budgeting for weights, gradients, optimizer states, and activations
- Training mechanics from [Training Mechanics](../../02-neural-networks/training-mechanics/COURSE.md), including how gradients are computed and how optimizers update weights

---

## Why Distribute Training

There are exactly two reasons to distribute training across multiple GPUs:

1. **The model doesn't fit on one GPU.** A 70B parameter model in FP16 requires ~140 GB of VRAM just for weights. The largest single GPU (H200) has 141 GB. Add gradients and optimizer states, and you need 500+ GB. The model must be split across GPUs.

2. **Training is too slow.** Even if the model fits on one GPU, training on billions of tokens takes weeks. Distributing across 8 GPUs cuts that to days. Distributing across 64 GPUs cuts it to hours.

Every distributed strategy is a tradeoff between memory savings, compute efficiency, and communication overhead. Understanding these tradeoffs is what separates ML engineers from ML researchers.

---

## Data Parallelism: The Default Strategy

### How It Works

Data parallelism is the simplest and most common distributed strategy:

1. **Replicate** the full model on every GPU
2. **Split** each training batch into micro-batches, one per GPU
3. **Forward pass** runs independently on each GPU with its micro-batch
4. **Backward pass** computes gradients independently on each GPU
5. **All-reduce** averages gradients across all GPUs
6. **Update** weights identically on each GPU (since gradients are averaged, all replicas stay in sync)

### When to Use It

Data parallelism works when the full model (weights + gradients + optimizer states + activations for one micro-batch) fits on a single GPU. This covers most models up to ~10B parameters on 80GB GPUs.

### Efficiency

On a single 8-GPU node with NVLink: 85-95% scaling efficiency (7-7.6x speedup from 8 GPUs).

The inefficiency comes from the all-reduce step — every GPU must send its gradients to every other GPU. With NVLink (600-900 GB/s), this takes milliseconds. Across nodes with InfiniBand (200-400 Gb/s), it takes longer.

### PyTorch Implementation

PyTorch offers two APIs:
- `DataParallel` (DP): old, single-process, uses one GPU as the "master" — inefficient, do not use
- `DistributedDataParallel` (DDP): modern, multi-process, each GPU runs its own process — use this

DDP launches one process per GPU, each with its own model replica. Gradient synchronization happens automatically during the backward pass using NCCL (NVIDIA Collective Communications Library).

Key configuration:
- `world_size`: total number of GPUs across all nodes
- `rank`: global ID of this process (0 to world_size-1)
- `local_rank`: GPU ID on this specific machine (0 to num_gpus_per_node-1)

---

## Model Parallelism: When the Model Won't Fit

### Tensor Parallelism (Intra-Layer)

Tensor parallelism splits individual layers across GPUs. A linear layer with weight matrix W (shape [4096, 16384]) can be split column-wise across 4 GPUs, each holding a [4096, 4096] slice.

**How it works for a linear layer Y = XW:**
1. Split W column-wise across GPUs: W = [W1, W2, W3, W4]
2. Each GPU computes Y_i = X @ W_i (partial result)
3. Concatenate or reduce results across GPUs

For transformer attention heads, this is natural — each GPU handles a subset of attention heads.

**Communication:** Requires communication within each layer (forward AND backward), so GPUs must be connected by fast interconnect (NVLink). Tensor parallelism across nodes is almost never practical.

**Typical usage:** 2-8 way tensor parallelism within a single node.

### Pipeline Parallelism (Inter-Layer)

Pipeline parallelism assigns different layers to different GPUs:
- GPU 0: layers 0-7
- GPU 1: layers 8-15
- GPU 2: layers 16-23
- GPU 3: layers 24-31

**The bubble problem:** Without pipelining, GPU 0 processes its layers, sends output to GPU 1, then sits idle. GPU 3 is idle until all previous GPUs finish. This is extremely wasteful.

**Micro-batching solution:** Split the batch into micro-batches and pipeline them:
- Time 1: GPU 0 processes micro-batch 1
- Time 2: GPU 0 processes micro-batch 2, GPU 1 processes micro-batch 1
- Time 3: GPU 0 processes micro-batch 3, GPU 1 processes micro-batch 2, GPU 2 processes micro-batch 1
- ...and so on

This fills the "pipeline bubble" so all GPUs are busy most of the time. With enough micro-batches, efficiency approaches 100%.

**Communication:** Only between adjacent pipeline stages (layer boundaries), so lower bandwidth requirements than tensor parallelism. Can work across nodes.

**Typical usage:** 2-8 way pipeline parallelism, often combined with tensor parallelism and data parallelism in 3D parallelism configurations.

---

### Check Your Understanding: Data and Model Parallelism

**1. In data parallelism, why do all GPU replicas end up with identical weights after each step?**

<details>
<summary>Answer</summary>

All replicas start with the same weights (from initialization or the previous step). Each GPU computes gradients on different data, then all-reduce averages the gradients across all GPUs. Since every GPU applies the same averaged gradient with the same optimizer to the same starting weights, the weight update is identical on all GPUs.
</details>

**2. Why is tensor parallelism typically limited to GPUs within a single node?**

<details>
<summary>Answer</summary>

Tensor parallelism splits individual layers across GPUs, requiring communication within every layer during both forward and backward passes. This demands very high bandwidth and low latency. NVLink provides 600-900 GB/s within a node, while cross-node interconnects (InfiniBand) provide 200-400 Gb/s -- a 10-20x gap. The frequent communication in tensor parallelism makes cross-node deployment impractical due to this bandwidth limitation.
</details>

**3. What is the "pipeline bubble" in pipeline parallelism, and how is it mitigated?**

<details>
<summary>Answer</summary>

The pipeline bubble is idle time: when GPU 0 processes its layers and passes output to GPU 1, GPU 0 sits idle waiting. In a naive implementation, only one GPU works at a time. The solution is micro-batching: split the batch into multiple micro-batches and pipeline them so that while GPU 1 processes micro-batch 1, GPU 0 starts on micro-batch 2. With enough micro-batches, all GPUs stay busy and efficiency approaches 100%.
</details>

---

## FSDP: Fully Sharded Data Parallel

### The Problem with Standard Data Parallelism

In standard DDP, every GPU holds a full copy of:
- Model weights
- Gradients
- Optimizer states

For a 7B model with Adam optimizer, that's ~84 GB per GPU, even though each GPU only needs its own micro-batch of data. This is wasteful — why store redundant copies?

### The FSDP Solution

FSDP (Fully Sharded Data Parallel) shards everything:

1. **Shard weights** across GPUs: each GPU holds 1/N of the weights
2. **Before a layer's forward pass:** all-gather the full weights for that layer from all GPUs
3. **After forward pass:** discard the non-local weights (free memory)
4. **Before backward pass:** all-gather the weights again
5. **After backward pass:** reduce-scatter gradients, each GPU gets 1/N of gradients
6. **Update:** each GPU updates only its 1/N shard of weights with its 1/N of optimizer states

### Memory Savings

For N GPUs:
- Weights per GPU: P/N (instead of P)
- Gradients per GPU: P/N (instead of P)
- Optimizer per GPU: proportional to P/N (instead of P)

For 8 GPUs, FSDP reduces memory by ~8x compared to DDP. A model that needs 84 GB per GPU with DDP needs ~11 GB per GPU with FSDP.

### Tradeoff

FSDP requires more communication than DDP:
- DDP: one all-reduce per backward pass
- FSDP: all-gather before each layer (forward + backward) + reduce-scatter after backward

On NVLink within a node, this overhead is small (5-15%). Across nodes, it can be significant (15-30%).

### When to Use FSDP

Use FSDP when:
- The model fits on one GPU with DDP but you want to use larger batch sizes or sequence lengths
- The model doesn't fit on one GPU with DDP (including optimizer states)
- You're using PyTorch and want the simplest multi-GPU solution for large models

FSDP is PyTorch's recommended approach for training models that don't fit on a single GPU. It replaces the older FairScale and is built into PyTorch core (`torch.distributed.fsdp`).

---

## DeepSpeed ZeRO: The Three Stages

DeepSpeed ZeRO (Zero Redundancy Optimizer) from Microsoft follows the same insight as FSDP but was developed independently and offers more configuration options.

### ZeRO Stage 1: Shard Optimizer States

- Weights: replicated on all GPUs
- Gradients: replicated on all GPUs
- **Optimizer states: sharded across GPUs**

Memory savings: ~4x for Adam (optimizer is the largest component).
Communication: same as DDP (one all-reduce for gradients).

### ZeRO Stage 2: Shard Optimizer + Gradients

- Weights: replicated on all GPUs
- **Gradients: sharded across GPUs**
- **Optimizer states: sharded across GPUs**

Memory savings: ~8x for Adam.
Communication: slightly more than Stage 1 (reduce-scatter instead of all-reduce for gradients).

### ZeRO Stage 3: Shard Everything

- **Weights: sharded across GPUs**
- **Gradients: sharded across GPUs**
- **Optimizer states: sharded across GPUs**

Memory savings: linear with number of GPUs.
Communication: same as FSDP (all-gather weights before each layer).

### ZeRO-Offload and ZeRO-Infinity

DeepSpeed extends the sharding concept to CPU RAM and NVMe storage:
- **ZeRO-Offload:** Keep optimizer states (or even gradients) in CPU RAM, only moving them to GPU when needed. Dramatically increases effective memory at the cost of PCIe bandwidth.
- **ZeRO-Infinity:** Extend to NVMe SSDs. Even more memory, even more latency.

These are useful for training very large models on limited GPU hardware, accepting a 2-5x slowdown.

### DeepSpeed vs FSDP

| Aspect | DeepSpeed | FSDP |
|--------|-----------|------|
| Framework | Standalone library | Built into PyTorch |
| Flexibility | More config options | Simpler API |
| Offloading | ZeRO-Offload to CPU/NVMe | Limited CPU offload |
| Maturity | More battle-tested at scale | Rapidly improving |
| Integration | Works with HuggingFace, custom loops | Native PyTorch |

For most HuggingFace-based training: use DeepSpeed (better integration, more configs).
For custom PyTorch training loops: use FSDP (simpler, native).

---

### Check Your Understanding: FSDP and DeepSpeed

**1. Standard DDP for a 7B model with Adam requires ~84 GB per GPU. How much does FSDP require per GPU with 8 GPUs, and why?**

<details>
<summary>Answer</summary>

FSDP requires roughly 84/8 = ~10.5 GB per GPU (plus activations). FSDP shards weights, gradients, and optimizer states across all GPUs, so each GPU holds only 1/N of each component. Before computing a layer, FSDP uses all-gather to reconstruct the full weights temporarily, then discards them after use.
</details>

**2. What is the key difference between DeepSpeed ZeRO Stage 1 and Stage 3?**

<details>
<summary>Answer</summary>

ZeRO Stage 1 shards only optimizer states (weights and gradients are replicated on all GPUs). Stage 3 shards everything: weights, gradients, and optimizer states. Stage 1 has the same communication cost as DDP (one all-reduce) but provides ~4x memory savings from distributing the Adam optimizer states. Stage 3 provides maximum memory savings (linear with GPU count) but requires more communication (all-gather before each layer).
</details>

---

## Communication Primitives

Understanding collective operations is essential for debugging distributed training.

### All-Reduce

Every GPU sends its data (gradients) to every other GPU, and all GPUs end up with the sum (or average). Used in standard DDP for gradient synchronization.

Cost: each GPU sends and receives O(P) data, but ring all-reduce achieves O(P/N) per GPU per step.

### All-Gather

Every GPU starts with a shard. After all-gather, every GPU has the full data. Used in FSDP/ZeRO-3 to reconstruct full weights before each layer.

Cost: each GPU receives O(P * (N-1)/N) data.

### Reduce-Scatter

Every GPU starts with full data. After reduce-scatter, each GPU has 1/N of the reduced result. Used in FSDP/ZeRO-2+ to distribute gradient shards.

Cost: each GPU sends O(P * (N-1)/N) data.

### Ring Topology

Modern collective operations use ring algorithms: GPUs are arranged in a logical ring, and data flows around the ring in chunks. This maximizes bandwidth utilization — all GPUs send and receive simultaneously.

On a node with NVLink, the ring is configured to match the physical NVLink topology for maximum bandwidth.

---

## Gradient Accumulation

Gradient accumulation simulates larger batch sizes without requiring more GPU memory:

1. Run forward + backward with a small micro-batch
2. Don't update weights — accumulate the gradients
3. Repeat for K micro-batches
4. Average accumulated gradients and update weights
5. Effective batch size = micro_batch_size x K x num_GPUs

This is mathematically equivalent to training with a large batch (ignoring batch normalization statistics, which are computed per micro-batch).

**When to use:**
- You need a large batch size for training stability (common for contrastive learning, large language models)
- Each micro-batch barely fits in memory
- You're already using all GPUs you have

**The relationship:** gradient accumulation is orthogonal to parallelism strategies. You can combine it with DDP, FSDP, or DeepSpeed. For example, with 8 GPUs, micro-batch=4, and accumulation_steps=8: effective batch size = 4 x 8 x 8 = 256.

---

## Multi-Node Training

### When You Need Multiple Machines

Single-node training (1-8 GPUs on one machine) handles most workloads. You need multiple nodes when:
- You need more than 8 GPUs (e.g., pre-training a large language model)
- Your timeline requires it (training must finish in days, not weeks)

### NCCL: NVIDIA Collective Communications Library

NCCL handles all GPU-to-GPU communication, whether within a node or across nodes. It automatically uses:
- NVLink for GPUs on the same node (600-900 GB/s)
- InfiniBand or RoCE for GPUs on different nodes (200-400 Gb/s per link)

The 10-20x bandwidth gap between NVLink and network explains why communication-heavy strategies (tensor parallelism) stay within a node, while communication-light strategies (data parallelism, pipeline parallelism) go across nodes.

### InfiniBand vs Ethernet

| Aspect | InfiniBand | Ethernet |
|--------|------------|----------|
| Bandwidth | 200-400 Gb/s | 100-400 Gb/s |
| Latency | 1-2 microseconds | 10-50 microseconds |
| GPU Direct | RDMA (bypass CPU) | Requires CPU involvement |
| Cost | Expensive | Standard |
| Availability | ML-specific clusters | Everywhere |

For serious multi-node training, InfiniBand with GPU Direct RDMA is essential. The latency difference matters because collective operations involve many small messages.

### Typical 3D Parallelism Configuration

For training a 70B+ model on 64 GPUs (8 nodes x 8 GPUs):
- **Tensor parallelism: 8-way** within each node (needs NVLink)
- **Pipeline parallelism: 2-4 way** across nodes (moderate communication)
- **Data parallelism: 2-4 way** across node groups (one all-reduce per step)

This is how models like Llama, GPT-4, and Gemini are trained.

---

## Decision Framework: What Strategy When

### Model fits on 1 GPU (< ~30B parameters with optimizations)

1. **Single GPU:** Use mixed precision + gradient checkpointing + gradient accumulation
2. **Multiple GPUs, same node:** Use DDP (data parallelism)
3. **Multiple nodes:** Use DDP across nodes

### Model doesn't fit on 1 GPU (> ~30B parameters)

1. **2-8 GPUs, same node:** Use FSDP or DeepSpeed ZeRO Stage 3
2. **Limited GPUs:** Use ZeRO-Offload (offload to CPU RAM)
3. **Full node (8 GPUs):** FSDP with tensor parallelism
4. **Multiple nodes:** 3D parallelism (tensor + pipeline + data)

### Fine-tuning (not pre-training)

1. **< 13B parameters:** QLoRA on a single GPU (most memory-efficient)
2. **13B-70B parameters:** QLoRA with FSDP or DeepSpeed
3. **Full fine-tuning of any size:** same as pre-training strategies above

---

## Practical Considerations

### Debugging Distributed Code

Distributed training is hard to debug because:
- Errors may occur on only one rank (GPU process)
- Deadlocks happen when processes get out of sync
- Memory errors on one rank crash all ranks

Practical tips:
- Set `NCCL_DEBUG=INFO` to see communication details
- Use `torch.distributed.barrier()` to synchronize processes for debugging
- Test with 2 GPUs before scaling to 8 or 64
- Use `CUDA_VISIBLE_DEVICES=0,1` to limit GPUs during debugging
- Check that your batch size is divisible by world_size

### Handling Failures

At scale (hundreds of GPUs), hardware failures are common. A training run lasting weeks will likely experience:
- GPU memory errors
- Network interruptions
- Machine reboots

**Checkpointing:** Save model state every N steps. With FSDP/DeepSpeed, each rank saves its shard — consolidation happens at load time or explicitly.

**Elastic training:** PyTorch Elastic (torchrun) can restart failed processes and continue training. DeepSpeed has similar capabilities.

**Rule of thumb:** Checkpoint every 1-2 hours of training time. Don't checkpoint too frequently (it's slow) or too rarely (you lose too much progress on failure).

### Common Pitfalls

1. **Forgetting to scale the learning rate.** When you double the batch size (via more GPUs), you often need to increase the learning rate by sqrt(2) or 2x, with a warmup period.

2. **Batch normalization across GPUs.** Standard BatchNorm computes statistics per-GPU. For small per-GPU batch sizes, use SyncBatchNorm to compute statistics across all GPUs.

3. **Random seed management.** Each GPU must see different data (different random seed for data shuffling) but the same model initialization (same seed for weight init).

4. **Not accounting for communication time in throughput calculations.** Throughput = (tokens processed) / (compute time + communication time). Communication can be 10-30% of total time.

---

## Interview Questions

**Q: Your team needs to train a 13B parameter model on 4 A100 80GB GPUs on a single node. What distributed strategy would you use and why?**

Strong answer: "I'd use FSDP (or DeepSpeed ZeRO Stage 3) because a 13B model with Adam optimizer needs about 13B x 16 bytes = ~208 GB for weights + gradients + optimizer in FP16/FP32 mixed precision — that's ~52 GB per GPU with 4-way sharding, which fits in 80 GB with room for activations. FSDP shards everything and uses all-gather to reconstruct weights when needed. Within a single node, the NVLink bandwidth makes the communication overhead minimal. I'd also enable gradient checkpointing and BF16 to further reduce memory and increase throughput."

**Q: Explain the difference between data parallelism and model parallelism. When would you choose each?**

Strong answer: "Data parallelism replicates the model on every GPU and splits the data. Each GPU processes different data independently, then gradients are averaged. It's the simplest approach and works when the model fits on one GPU — you use it to speed up training. Model parallelism splits the model itself across GPUs, either by layers (pipeline parallelism) or within layers (tensor parallelism). You use it when the model doesn't fit on a single GPU. In practice, large-scale training uses both: tensor parallelism within a node for intra-layer splitting, pipeline parallelism across a few nodes, and data parallelism for the remaining GPUs."

**Q: What is the communication overhead of FSDP compared to standard DDP?**

Strong answer: "In DDP, communication happens once per backward pass — an all-reduce of gradients. In FSDP, communication happens multiple times: an all-gather of weights before each layer in both forward and backward passes, plus a reduce-scatter of gradients after the backward pass. The total data communicated is roughly 3x more than DDP. On NVLink within a node, this adds 5-15% overhead. Across nodes on InfiniBand, it can be 15-30%. The tradeoff is worth it because FSDP enables training models that wouldn't fit in memory with DDP."

---

## Key Takeaways

1. Data parallelism (DDP) is the default — use it when the model fits on one GPU
2. FSDP/DeepSpeed ZeRO shard weights + gradients + optimizer across GPUs for memory savings
3. Tensor parallelism splits layers within a node (needs NVLink), pipeline parallelism splits layers across nodes
4. 3D parallelism (tensor + pipeline + data) is how frontier models are trained
5. Communication overhead is the cost of distribution — minimize it by matching strategy to hardware topology
6. Gradient accumulation simulates larger batches without more memory
7. Checkpoint frequently at scale -- hardware failures are inevitable

---

## Common Pitfalls

**1. Forgetting to scale the learning rate when increasing the effective batch size.** When you go from 1 GPU to 8 GPUs with data parallelism, the effective batch size increases 8x. The learning rate typically needs to be scaled up (linearly or by sqrt) with a warmup period. Not adjusting leads to underfitting.

**2. Using DataParallel (DP) instead of DistributedDataParallel (DDP).** The older `torch.nn.DataParallel` uses a single process with one GPU as master, creating a bottleneck. DDP uses one process per GPU with NCCL-based gradient synchronization and is significantly faster. Never use DP for serious training.

**3. Not testing with 2 GPUs before scaling to 64.** Distributed training bugs (deadlocks, rank mismatches, incorrect batch sizes) are much harder to debug at scale. Always validate your distributed code on 2 GPUs first, then scale to a full node, then to multiple nodes.

**4. Checkpointing too infrequently at scale.** With hundreds of GPUs running for days, hardware failures are not exceptional -- they are expected. Losing 8+ hours of training to a GPU memory error or network interruption is expensive. Checkpoint every 1-2 hours of training time.

---

## Hands-On Exercises

### Exercise: Strategy Selection

For each scenario, determine the distributed training strategy, estimate per-GPU memory, and justify your choice:

1. A 3B parameter model, 4x A100 80GB GPUs on one node, goal: maximize training throughput
2. A 30B parameter model, 8x H100 80GB GPUs on one node, full fine-tuning with Adam
3. A 70B parameter model, 64 GPUs across 8 nodes (8x H100 per node), pre-training

### Exercise: Communication Cost Analysis

Consider 8 GPUs on one node with NVLink (900 GB/s bidirectional). A 13B parameter model has 13B x 2 = 26 GB of FP16 gradients.

1. How long does an all-reduce of the gradients take (approximately)?
2. If the forward + backward pass takes 500ms with micro-batch size 4, what is the communication overhead as a percentage of total step time?
3. How does this change if the GPUs are on two different nodes connected by 400 Gb/s InfiniBand?

---

## Summary

This lesson covered the distributed training strategies that enable training models beyond a single GPU: data parallelism (DDP) for throughput scaling, FSDP and DeepSpeed ZeRO for memory-efficient sharding, tensor and pipeline parallelism for splitting models across GPUs, 3D parallelism for frontier-scale training, communication primitives, gradient accumulation, multi-node considerations, and practical debugging guidance.

### What's Next

Continue to [Quantization](../quantization/COURSE.md) to learn how to reduce model precision for efficient inference and fine-tuning, including INT8/INT4 quantization methods (GPTQ, AWQ), the GGUF format for local deployment, and QLoRA for memory-efficient fine-tuning.
