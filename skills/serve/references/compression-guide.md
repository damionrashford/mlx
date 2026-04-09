# Model Compression Guide

## Dynamic quantization (no calibration data)

Easiest — quantizes weights post-training. Best for CPU deployment.

```python
import torch

model_dynamic = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},  # modules to quantize
    dtype=torch.qint8,
)
torch.save(model_dynamic.state_dict(), "model_dynamic_int8.pt")
```

No accuracy calibration needed. Works well for LSTM, Linear layers. Activations stay float32.

## Static quantization (calibration required)

Quantizes both weights and activations. Requires calibration data. Faster than dynamic.

```python
model.qconfig = torch.quantization.get_default_qconfig("fbgemm")  # CPU
torch.quantization.prepare(model, inplace=True)

# Calibration: run inference on representative data
for X_batch, _ in calibration_loader:
    model(X_batch)

torch.quantization.convert(model, inplace=True)
```

## bitsandbytes: 8-bit and 4-bit for LLMs

```python
from transformers import AutoModelForCausalLM

# 8-bit (2× memory reduction)
model = AutoModelForCausalLM.from_pretrained(
    "model_name",
    load_in_8bit=True,
    device_map="auto",
)

# 4-bit (4× memory reduction)
model = AutoModelForCausalLM.from_pretrained(
    "model_name",
    load_in_4bit=True,
    device_map="auto",
)
```

## GPTQ: accurate 4-bit LLM quantization

```python
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

quantize_config = BaseQuantizeConfig(
    bits=4,
    group_size=128,
    desc_act=True,
)

model = AutoGPTQForCausalLM.from_pretrained("model_name", quantize_config)
model.quantize(calibration_dataset)  # calibration_dataset: list of token tensors
model.save_quantized("model_gptq_4bit")
```

## AWQ: fast edge inference

```python
from awq import AutoAWQForCausalLM

model = AutoAWQForCausalLM.from_pretrained("model_name")
model.quantize(tokenizer, quant_config={"zero_point": True, "q_group_size": 128, "w_bit": 4})
model.save_quantized("model_awq_4bit")
```

## ONNX export

### sklearn models

```python
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

initial_type = [("float_input", FloatTensorType([None, n_features]))]
onnx_model = convert_sklearn(pipeline, initial_types=initial_type)
with open("model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
```

### PyTorch models

```python
dummy_input = torch.randn(1, n_features)
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=17,
)
```

## ONNX Runtime inference

```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession(
    "model.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)

# Graph optimization
opts = ort.SessionOptions()
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

input_name = session.get_inputs()[0].name
outputs = session.run(None, {input_name: X_test.astype(np.float32)})
```

## Structured pruning

```python
import torch.nn.utils.prune as prune

# Unstructured L1 pruning: remove smallest weights
prune.l1_unstructured(model.fc1, name="weight", amount=0.3)  # 30% of weights

# Structured: remove entire filters (reduces computation more)
prune.ln_structured(model.conv1, name="weight", amount=0.2, n=2, dim=0)

# Make permanent
prune.remove(model.fc1, "weight")
```

## Knowledge distillation

```python
import torch.nn.functional as F

def distillation_loss(student_logits, teacher_logits, labels, alpha=0.5, temperature=4.0):
    # Hard label loss
    hard_loss = F.cross_entropy(student_logits, labels)

    # Soft label loss (KL divergence)
    soft_student = F.log_softmax(student_logits / temperature, dim=-1)
    soft_teacher = F.softmax(teacher_logits / temperature, dim=-1)
    soft_loss = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (temperature ** 2)

    return alpha * hard_loss + (1 - alpha) * soft_loss

# Training loop
teacher.eval()
for X, y in dataloader:
    with torch.no_grad():
        teacher_logits = teacher(X)
    student_logits = student(X)
    loss = distillation_loss(student_logits, teacher_logits, y)
    loss.backward()
```

**Temperature**: higher T (4-8) makes soft targets softer, giving more information
about wrong class probabilities. **alpha**: 0.3-0.7 typical.

## Benchmarking template

```python
import time, psutil, os
import numpy as np

def benchmark(model_fn, X_test, n_warmup=10, n_runs=100):
    # Warmup
    for _ in range(n_warmup):
        model_fn(X_test[:1])

    # Measure latency
    latencies = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        model_fn(X_test[:1])
        latencies.append((time.perf_counter() - t0) * 1000)

    # Measure throughput
    t0 = time.perf_counter()
    model_fn(X_test)
    throughput = len(X_test) / (time.perf_counter() - t0)

    # Memory
    process = psutil.Process(os.getpid())
    rss_mb = process.memory_info().rss / 1024 / 1024

    return {
        "p50_ms": np.percentile(latencies, 50),
        "p95_ms": np.percentile(latencies, 95),
        "throughput_rps": throughput,
        "memory_rss_mb": rss_mb,
    }
```
