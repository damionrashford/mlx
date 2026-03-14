# 07 — Distributed Computing & GPU Training

> Running ML at scale. Explicit requirement: "Experience with running machine learning in parallel environments (e.g., distributed clusters, GPU optimization)."

## Why This Matters

Large-scale platforms serve millions of users. Models need to train on massive datasets and serve at low latency. You need to understand GPU fundamentals and distributed training.

## Subdirectories

```
07-distributed-gpu/
├── gpu-fundamentals/        # CUDA concepts, GPU memory, compute architecture
├── distributed-training/    # Data parallelism, model parallelism, DeepSpeed, FSDP
└── quantization/            # INT8, INT4, GPTQ, AWQ — model compression
```

## Key Resources

| Resource | What it covers | Time |
|---|---|---|
| DeepLearning.AI: Quantization Fundamentals | Model compression basics | ~2 hrs |
| DeepLearning.AI: Quantization In Depth | Advanced quantization techniques | ~2 hrs |
| DeepLearning.AI: Introduction to On-Device AI | Edge deployment | ~2 hrs |
| DeepLearning.AI: Build and Train an LLM with JAX | Hands-on training | ~2 hrs |
| Lambda Labs / RunPod / Google Colab Pro | Actual GPU training experience | Ongoing |

## Key Concepts

### GPU Fundamentals (Know the intuition)
- **Why GPUs?** Matrix multiplication is the core of neural networks. GPUs do matrix math in parallel.
- **GPU Memory (VRAM):** The bottleneck. Model weights + activations + gradients must fit.
- **CUDA:** NVIDIA's parallel computing platform. PyTorch uses it under the hood.
- **Mixed precision (FP16/BF16):** Use half-precision floats to double effective memory and speed.

### Distributed Training Strategies
| Strategy | What it does | When to use |
|---|---|---|
| Data parallelism | Same model on each GPU, different data batches | Most common — model fits on one GPU |
| Model parallelism | Split model layers across GPUs | Model too large for one GPU |
| Pipeline parallelism | Split model into stages, pipeline the batches | Very large models |
| FSDP (Fully Sharded) | Shard model params, gradients, optimizer across GPUs | Modern default for large models |
| DeepSpeed ZeRO | Similar to FSDP, three stages of sharding | Popular in research |

### Quantization
| Format | Bits | Size reduction | Quality loss | When to use |
|---|---|---|---|---|
| FP32 | 32 | Baseline | None | Training (default) |
| FP16/BF16 | 16 | 2x smaller | Minimal | Training (mixed precision) |
| INT8 | 8 | 4x smaller | Small | Inference deployment |
| INT4 | 4 | 8x smaller | Moderate | Edge/mobile deployment |

### Hands-on Practice Plan
- [ ] Run a training job on Google Colab (free GPU)
- [ ] Run a training job on Colab Pro (A100)
- [ ] Fine-tune a small LLM using LoRA on a single GPU
- [ ] Experiment with quantization (INT8 vs INT4 inference)
