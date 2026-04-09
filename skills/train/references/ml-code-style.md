# ML Code Style Reference

Conventions from SN-WANG/ResearchSkills for clean, consistent ML research code.

## Shape annotation conventions

Use uppercase letters for tensor dimensions. Always document shapes in docstrings.

### Standard dimension names

| Symbol | Meaning |
|--------|---------|
| B | Batch size |
| T | Sequence length (time) |
| N | Number of elements (tokens, nodes, etc.) |
| C | Channels (generic) |
| H | Height |
| W | Width |
| D | Model/embedding dimension |
| E | Embedding dimension (when different from D) |
| K | Number of classes or keys |
| L | Number of layers |
| A | Number of attention heads |

### Semantic variants

When multiple channel dimensions coexist, use semantic suffixes:

```python
# Good: unambiguous with semantic names
def project(x: Tensor, W: Tensor) -> Tensor:
    """
    x: Input features. (B, N, C_IN).
    W: Projection weight. (C_IN, C_OUT).
    returns: Projected features. (B, N, C_OUT).
    """
    return x @ W

# Bad: ambiguous
def project(x, W):
    return x @ W
```

Common semantic variants: `C_IN`, `C_OUT`, `D_MODEL`, `D_HEAD`, `D_FF`, `N_HEADS`.

## Tensor docstring format

Standard format for functions operating on tensors:

```python
def attention(q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None = None) -> Tensor:
    """Multi-head scaled dot-product attention.

    q: Query tensor. (B, A, T, D_HEAD).
    k: Key tensor. (B, A, T, D_HEAD).
    v: Value tensor. (B, A, T, D_HEAD).
    mask: Attention mask, True = masked out. (B, 1, T, T). Optional.
    returns: Attended values. (B, A, T, D_HEAD).
    """
    scale = q.shape[-1] ** -0.5
    scores = (q @ k.transpose(-2, -1)) * scale
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))
    return F.softmax(scores, dim=-1) @ v
```

Rules:
- Argument name, colon, description, period, shape in parentheses
- `returns:` (lowercase) for return value
- Shapes always in parentheses on the same line as the description
- Use `|` for optional types: `Tensor | None`

## Two-line file header

Every module starts with a two-line header:

```python
# Transformer encoder stack with rotary position embeddings.
# Author: name
```

Line 1: what the module does (one sentence).
Line 2: `# Author: name` (optional but encouraged for research code).

## Section dividers for files > 200 lines

Use `# ── Section Name ──` dividers (with em-dashes) to separate logical sections:

```python
# ── Config ────────────────────────────────────────────────────────────────────
BATCH_SIZE = 32
LR = 1e-3

# ── Model ─────────────────────────────────────────────────────────────────────
class Transformer(nn.Module):
    ...

# ── Training ──────────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer):
    ...
```

## No defensive programming by default

Trust internal code and framework guarantees. Only validate at system boundaries
(user input, external APIs, file I/O).

```python
# Good: trust that caller passes correct shapes
def forward(self, x: Tensor) -> Tensor:
    return self.linear(x)

# Bad: defensive check inside trusted internals
def forward(self, x: Tensor) -> Tensor:
    if not isinstance(x, Tensor):
        raise TypeError(f"Expected Tensor, got {type(x)}")
    if x.ndim != 2:
        raise ValueError(f"Expected 2D tensor, got {x.ndim}D")
    return self.linear(x)
```

Add validation only at: CLI argument parsing, API endpoint inputs, file loading.

## Variable-first documentation

Document the variable, not the operation:

```python
# Good: describes what the variable IS
lr = 1e-3  # base learning rate before warmup scaling

# Bad: describes the operation
lr = 1e-3  # set the learning rate to 1e-3
```

## Five-part documentation spine

Structure research project documentation with these five sections:

### 1. Problem Formulation
- Task definition (input → output)
- Formal notation
- Evaluation metric(s) and their justification

### 2. Data Specification
- Dataset name, size, source, license
- Train/val/test split rationale
- Feature schema with types and ranges
- Known data quality issues

### 3. Model Specification
- Architecture diagram or pseudocode
- Parameter count and memory footprint
- Key design decisions and alternatives considered

### 4. Training Protocol
- Optimizer, learning rate schedule, batch size
- Regularization: dropout, weight decay, gradient clipping
- Stopping criteria
- Hardware and wall-clock time

### 5. Inference Specification
- Input preprocessing pipeline (must match training exactly)
- Output postprocessing
- Latency (p50, p95) and throughput
- Known failure modes and edge cases

## Import ordering

```python
# 1. Standard library
import os, sys, math, json, csv

# 2. Third-party (alphabetical within each group)
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# 3. Local
from .model import Transformer
from .data import Dataset
```
