---
name: compress
description: >
  Reduce model size and inference latency via quantization, pruning, and knowledge
  distillation. Covers sklearn, PyTorch, and LLM compression. Benchmarks before/after:
  latency, throughput, accuracy delta, memory footprint. Use before deploying any
  model to production.
allowed-tools: Bash, Read, Write, Edit, Glob, Grep
argument-hint: model file path (e.g. "model.joblib" or "model.pt")
---

# Compress Skill

Reduce model size and latency via quantization, pruning, and distillation.

## Quick start

```bash
# Benchmark original vs compressed
python3 scripts/benchmark_model.py model.joblib model_compressed.joblib data/test.csv
```

## Methods

### Quantization
- **Dynamic** (no calibration): `torch.quantization.quantize_dynamic` — easiest, CPU
- **Static** (calibration required): `torch.quantization.prepare` + `convert` — faster
- **4-bit** (LLMs): `bitsandbytes load_in_4bit=True` — 4× memory reduction
- **GPTQ**: `AutoGPTQ` — accurate 4-bit for LLMs

### Export
- **ONNX**: `skl2onnx` for sklearn, `torch.onnx.export` for PyTorch
- **ONNX Runtime**: `InferenceSession` — cross-platform optimized inference
- **GGUF**: `convert_hf_to_gguf.py` for local LLM inference

### Pruning
- `torch.nn.utils.prune.l1_unstructured` — remove smallest weights
- Structured pruning: remove entire filters/heads

### Knowledge distillation
- Student model learns from teacher's soft labels
- Temperature τ softens probability distribution
- Loss = α × CE(hard labels) + (1-α) × KD(soft labels, τ)

## Benchmark targets

| Metric | Goal |
|--------|------|
| Latency p50/p95 | < 2× original |
| Throughput | > 1.5× original |
| Accuracy delta | < 1% absolute |
| Memory RSS | < 50% original |

See `references/compression-guide.md` for full code examples and ONNX Runtime setup.
