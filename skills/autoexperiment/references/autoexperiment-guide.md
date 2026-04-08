# Autoexperiment Guide

Patterns from karpathy/autoresearch for autonomous time-budget experiment loops.

## TIME_BUDGET pattern

Train for a fixed wall-clock duration, not fixed epochs. Enables fair comparison across
different model sizes and configurations — all get the same wall-clock time.

```python
import time
TIME_BUDGET = int(os.environ.get("TIME_BUDGET", "300"))  # seconds
t0 = time.time()

for step in itertools.count():
    if time.time() - t0 > TIME_BUDGET:
        break
    # ... training step
```

Why wall-clock vs epochs: a larger model may need fewer epochs to converge but more
time per step. Wall-clock fairness lets you compare architectures at equal compute cost.
~12 experiments/hour at 300s each.

## EXPERIMENT.md: human-written intent file

The agent reads this before each iteration to understand the current goal and hypothesis.
Structure: goal, baseline, hypothesis, constraints, next to try.

See `references/EXPERIMENT.md.template` for the format.

## Single modifiable file

Agents edit `train.py` only. `prepare.py` (data preprocessing) is frozen.
This ensures each experiment changes exactly one thing.

## val_bpb metric

Bits-per-byte — vocabulary-independent metric for language models:
```python
import math
val_bpb = total_nats / (math.log(2) * total_bytes)
```

- `total_nats`: sum of negative log-likelihood over all tokens
- `total_bytes`: total UTF-8 bytes in the validation set
- Lower is better. Vocab-independent, unlike perplexity.

## GC freeze after step 0

Eliminates ~500ms GC pause stalls that appear at irregular intervals:
```python
import gc
# After first training step:
gc.collect()
gc.freeze()
gc.disable()
```

Why: Python GC runs generational collection at unpredictable intervals. After step 0,
all long-lived objects are allocated. Freezing them and disabling GC prevents stalls.
Re-enable only if you allocate and free large objects in the loop.

## Fast fail guard

Exit immediately on NaN loss or runaway values:
```python
if math.isnan(loss) or loss > 100:
    print(f"Fast fail: loss={loss}", file=sys.stderr)
    sys.exit(1)
```

Records as CRASH in results.tsv. Prevents wasting time budget on a broken run.

## EMA debiased loss

Correct exponential moving average from step 0 (avoids initial underestimate):
```python
b = 0.9  # smoothing factor
smooth = 0.0
for step in range(steps):
    smooth = b * smooth + (1 - b) * loss
    display = smooth / (1 - b ** (step + 1))  # debiased
```

## MuonAdamW optimizer

Use Muon for 2D weight matrices (linear layers, attention projections),
AdamW for 1D parameters (embeddings, biases, layer norms):

```python
param_groups = [
    {"params": [p for p in model.parameters() if p.ndim >= 2], "optimizer": "muon"},
    {"params": [p for p in model.parameters() if p.ndim < 2], "optimizer": "adamw"},
]
```

Muon applies Nesterov momentum then orthogonalizes the update via Newton-Schulz
iterations. Achieves better loss at equal compute vs pure AdamW for transformer weights.

## Warmdown LR schedule

Decay learning rate across the LAST 50% of the time budget (not last 10%):
```python
frac = elapsed / TIME_BUDGET
if frac > 0.5:
    lr = lr_max * (1.0 - (frac - 0.5) / 0.5)
else:
    # warmup or constant phase
    lr = lr_max * min(1.0, frac / 0.1)
```

Why 50%: with short budgets (300s), 10% warmdown is too brief for the optimizer
to settle. 50% warmdown gives the model more time at low LR to consolidate.

## ASPECT_RATIO model sizing

Scale model dimensions proportionally to depth:
```python
HEAD_DIM = 64
depth = 6  # number of layers
model_dim = depth * HEAD_DIM  # = 384 for depth=6
# Round to nearest HEAD_DIM multiple
model_dim = round(model_dim / HEAD_DIM) * HEAD_DIM
```

Why: maintains aspect ratio as models scale. Prevents very wide shallow or
very narrow deep models which underperform.

## Circuit breaker

If the same experiment crashes 3 times consecutively on the same error:
1. Stop the loop
2. Report to user: error message, stack trace, likely cause, suggested fix
3. Do NOT retry — escalate and wait for human guidance

```python
consecutive_crashes = 0
last_error = None
for exp_id in experiment_ids:
    result = run_experiment(exp_id)
    if result.status == "CRASH":
        if result.error == last_error:
            consecutive_crashes += 1
        else:
            consecutive_crashes = 1
            last_error = result.error
        if consecutive_crashes >= 3:
            report_and_stop(result.error)
            break
    else:
        consecutive_crashes = 0
        last_error = None
```
