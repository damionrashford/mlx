# Advanced Training Patterns

From karpathy/autoresearch: patterns for efficient, reproducible neural network training.

## GC freeze after step 0

Eliminates ~500ms GC pause stalls that occur at unpredictable intervals:

```python
import gc

for step in range(total_steps):
    loss = train_step(batch)

    if step == 0:
        # All long-lived objects now allocated — freeze and disable GC
        gc.collect()
        gc.freeze()
        gc.disable()

    # ... rest of training loop
```

**Why**: Python's generational GC runs at intervals determined by allocation counts.
After step 0, all model weights, optimizer states, and buffers are allocated.
Freezing them prevents the GC from scanning them on every collection cycle.
Result: no unexpected pauses during training.

**When to re-enable**: if you explicitly allocate and free large tensors inside the
training loop (e.g., dynamic computation graphs), re-enable GC selectively.

## Time-budget loop

Train for wall-clock seconds, not epochs. Enables fair comparison across different
model sizes — all get the same compute budget.

```python
import time, itertools

TIME_BUDGET = 300  # seconds
t0 = time.monotonic()

for step in itertools.count():
    if time.monotonic() - t0 >= TIME_BUDGET:
        break
    loss = train_step(batch)
```

**Why vs epochs**: a 6-layer model takes 2× per-step time vs a 3-layer model.
With epoch-based training, the smaller model gets 2× the gradient updates.
Wall-clock fairness lets you compare architectures at equal compute.

**Rate**: ~12 experiments/hour at TIME_BUDGET=300.

## EMA debiased loss

Correct exponential moving average from step 0. Without debiasing, the EMA
underestimates the true loss for the first ~1/(1-β) steps.

```python
beta = 0.9
smooth = 0.0

for step, loss in enumerate(losses):
    smooth = beta * smooth + (1 - beta) * loss
    display = smooth / (1 - beta ** (step + 1))  # debiased
    print(f"step={step} loss={display:.4f}")
```

**Without debiasing**: at step 0, `smooth = 0.1 * loss` (90% wrong).
**With debiasing**: at step 0, `display = smooth / 0.1 = loss` (correct).

## MuonAdamW optimizer

Use Muon for 2D weight matrices (linear layers, attention projections).
Use AdamW for 1D parameters (embeddings, biases, layer norms).

```python
# Separate param groups by tensor dimensionality
muon_params = [p for p in model.parameters() if p.ndim >= 2]
adamw_params = [p for p in model.parameters() if p.ndim < 2]

# Muon: Nesterov momentum + Newton-Schulz orthogonalization
# AdamW: standard adaptive moment estimation
optimizer = CombinedOptimizer([
    Muon(muon_params, lr=0.02, momentum=0.95),
    torch.optim.AdamW(adamw_params, lr=3e-4, weight_decay=0.1),
])
```

**Why**: 2D weight matrices benefit from orthogonalized gradient updates —
Muon projects gradients onto the manifold of orthogonal matrices via Newton-Schulz
iterations. Achieves better loss at equal compute vs AdamW for transformer weights.
1D parameters (embeddings, biases) work better with AdamW's element-wise adaptation.

## Best-fit packing dataloader

Concatenate variable-length sequences to fill context windows completely.
Eliminates padding waste — achieves ~100% token utilization.

```python
from itertools import chain

def pack_sequences(dataset, context_length, eos_token_id):
    """Concatenate samples with EOS separator, chunk to context_length."""
    tokens = list(chain.from_iterable(
        sample["input_ids"] + [eos_token_id]
        for sample in dataset
    ))
    # Chunk into context_length blocks
    chunks = [
        tokens[i:i + context_length]
        for i in range(0, len(tokens) - context_length, context_length)
    ]
    return [{"input_ids": chunk, "labels": chunk} for chunk in chunks]
```

**Why vs padding**: padding to max sequence length wastes 20-70% of compute
on mask operations. Packing eliminates this waste entirely.

## Warmdown LR schedule

Decay learning rate over the LAST 50% of training budget (not 10%).

```python
def get_lr(step, total_steps, lr_max, warmup_frac=0.05, warmdown_start=0.5):
    frac = step / total_steps

    if frac < warmup_frac:
        # Linear warmup
        return lr_max * frac / warmup_frac
    elif frac < warmdown_start:
        # Constant phase
        return lr_max
    else:
        # Linear warmdown from warmdown_start to end
        decay = (frac - warmdown_start) / (1.0 - warmdown_start)
        return lr_max * max(0.0, 1.0 - decay)
```

**Why 50% not 10%**: with short budgets (300s), a 10% warmdown period is too
brief for the optimizer to fully exploit the low-LR regime. 50% warmdown gives
the model more gradient steps at low LR to consolidate learned features.

## ASPECT_RATIO model sizing

Scale model dimensions proportionally to depth to maintain aspect ratio.

```python
HEAD_DIM = 64  # attention head dimension (multiple of 64 for efficient attention)

def compute_model_dim(depth: int, head_dim: int = HEAD_DIM) -> int:
    """Model dim = depth * head_dim, rounded to nearest head_dim multiple."""
    raw_dim = depth * head_dim
    return round(raw_dim / head_dim) * head_dim

# Examples:
# depth=4 → model_dim=256 (4 × 64)
# depth=6 → model_dim=384 (6 × 64)
# depth=8 → model_dim=512 (8 × 64)
```

**Why**: maintaining aspect ratio (width/depth) prevents pathological architectures.
A very wide shallow model and a very narrow deep model both underperform a model
with balanced width-to-depth ratio at the same parameter count.

## Fast-fail guard

Exit immediately on NaN loss or runaway values. Prevents wasting budget on broken runs.

```python
import math, sys

def check_loss(loss: float, step: int):
    if math.isnan(loss):
        print(f"CRASH: NaN loss at step {step}", file=sys.stderr)
        sys.exit(1)
    if loss > 100:
        print(f"CRASH: loss={loss:.1f} > 100 at step {step} (runaway)", file=sys.stderr)
        sys.exit(1)
```

Records as CRASH in results.tsv. Circuit breaker: 3 consecutive CRASHes on
the same error → stop the experiment loop, report diagnosis to user.
