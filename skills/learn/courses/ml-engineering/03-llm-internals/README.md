# 03 — LLM Internals

> How large language models are built, trained, aligned, and deployed. The internals behind the APIs you already use.

## Why This Matters

Applied ML engineering roles require mastery in building data products using generative AI, RLHF, and fine-tuning LLMs. Understanding the full lifecycle is essential.

## Subdirectories

```
03-llm-internals/
├── pretraining/              # How LLMs are trained from scratch (data, compute, architecture)
├── fine-tuning/              # LoRA, QLoRA, full fine-tuning, when to use each
├── rlhf-alignment/           # RLHF, DPO, GRPO — making models behave
├── inference-optimization/   # KV cache, quantization, batching, serving at scale
└── tokenization/             # BPE, SentencePiece — how text becomes numbers
```

## Key Resources

| Resource | What it covers | Time |
|---|---|---|
| DeepLearning.AI: Generative AI with LLMs | Full LLM lifecycle — THE most important course | ~16 hrs |
| DeepLearning.AI: Pretraining LLMs | Training from scratch | ~2 hrs |
| DeepLearning.AI: Finetuning Large Language Models | When and how to fine-tune | ~2 hrs |
| DeepLearning.AI: Fine-tuning & RL for LLMs | Post-training techniques | ~2 hrs |
| DeepLearning.AI: Reinforcement Fine-Tuning LLMs — GRPO | Cutting-edge alignment | ~2 hrs |
| DeepLearning.AI: Efficiently Serving LLMs | Inference optimization | ~2 hrs |
| DeepLearning.AI: Quantization Fundamentals | Model compression | ~2 hrs |
| DeepLearning.AI: Build and Train an LLM with JAX | Hands-on training from scratch | ~2 hrs |

## Technique Landscape

### Pretraining
- **What:** Train model on massive text corpus to learn language patterns
- **When:** Building a foundation model (rarely done — requires massive compute)
- **Key concepts:** Next-token prediction, data curation, scaling laws, compute budgets
- **Cost:** $1M-$100M+ for frontier models

### Fine-tuning
- **What:** Adapt a pretrained model to a specific task or domain
- **When:** You have domain-specific data and need specialized behavior

| Method | What it does | When to use | GPU requirement |
|---|---|---|---|
| Full fine-tuning | Update all weights | Large dataset, need max performance | Multi-GPU |
| LoRA | Train small adapter matrices, freeze rest | Most common — good balance of cost/quality | Single GPU |
| QLoRA | LoRA on a quantized model | Limited GPU memory | Single small GPU |
| Prefix tuning | Prepend learnable tokens | Very parameter-efficient | Minimal |

### Alignment (RLHF / DPO / GRPO)
- **What:** Make the model's outputs match human preferences
- **When:** Model knows the answer but presents it wrong (tone, safety, format)
- **RLHF:** Train a reward model from human comparisons, then optimize with PPO
- **DPO:** Direct preference optimization — simpler, no reward model needed
- **GRPO:** Group relative policy optimization — newest, most efficient

### Inference Optimization
- **KV Cache:** Store previous attention computations for faster generation
- **Quantization:** Compress weights (FP32 → INT8 → INT4) for smaller/faster models
- **Batching:** Process multiple requests simultaneously
- **Speculative decoding:** Use a small model to draft, large model to verify
- **Tensor parallelism:** Split model across GPUs for large models
