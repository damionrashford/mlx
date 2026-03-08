---
name: experiment
description: >
  Design, run, and track ML experiments with systematic hyperparameter search.
  Manages result tracking (TSV), metric comparison, and keep/discard decisions.
  Part of the mlx workbench. Use when the user wants to run experiments, tune hyperparameters,
  compare models, track results, or iterate systematically on model performance.
allowed-tools: Bash, Read, Write, Glob, Grep
user-invocable: true
argument-hint: path to experiment script or results.tsv (e.g. "experiment.py" or "results.tsv")
context: fork
---

# Experiment Tracking & Iteration

Reference for systematic ML experimentation: tracking, search strategies, and analysis.

## Experiment cycle

```
1. Hypothesize (what change, why it might help)
2. Modify (one variable at a time)
3. Run (fixed budget: time or epochs)
4. Record (append to results.tsv)
5. Decide: KEEP (improved) or DISCARD (didn't)
6. Repeat
```

## Setup

### Initialize tracking
```bash
echo -e "id\tmetric\tval_score\ttest_score\tmemory_mb\tstatus\tdescription" > results.tsv
```

### Record a baseline
```bash
echo -e "exp000\taccuracy\t0.8523\t0.8401\t4096\tKEEP\tbaseline" >> results.tsv
```

## Results format (TSV)

```
id        metric    val_score  test_score  memory_mb  status   description
exp000    accuracy  0.8523     0.8401      4096       KEEP     baseline
exp001    accuracy  0.8612     0.8498      4096       KEEP     lr=0.001
exp002    accuracy  0.8590     -           4096       DISCARD  lr=0.003 (overfit)
exp003    accuracy  0.0000     -           0          CRASH    lr=0.01 (diverged)
exp004    accuracy  0.8634     0.8521      4352       KEEP     dropout=0.1
```

Status: `KEEP` (improved), `DISCARD` (same or worse), `CRASH` (error/OOM/NaN)

## What to try (priority order)

### High impact (try first)
1. Learning rate (3x and 0.3x current)
2. Model capacity (layers, hidden size)
3. Batch size (double or halve)
4. Regularization (dropout, weight decay)

### Medium impact
5. Optimizer (Adam -> AdamW -> SGD+momentum)
6. LR schedule (cosine, warmup, step decay)
7. Data augmentation
8. Feature selection

### Low impact (try last)
9. Activation functions
10. Normalization layers
11. Initialization schemes
12. Gradient clipping

## Search strategies

### Grid (small spaces)
```python
from itertools import product
params = {'lr': [1e-4, 3e-4, 1e-3], 'dropout': [0.0, 0.1, 0.3]}
for combo in product(*params.values()):
    config = dict(zip(params.keys(), combo))
```

### Random (large spaces)
```python
import random
def sample():
    return {
        'lr': 10 ** random.uniform(-5, -2),
        'dropout': random.uniform(0, 0.5),
        'hidden': random.choice([64, 128, 256, 512]),
    }
```

### Informed (after several runs)
Analyze results.tsv: which LR range works? Did more capacity help? Narrow search.

## Analysis

```bash
python3 ${CLAUDE_SKILL_DIR}/scripts/analyze_results.py results.tsv
```

Or inline:
```python
import pandas as pd
r = pd.read_csv("results.tsv", sep="\t")
kept = r[r.status == "KEEP"]
print(f"Total: {len(r)}, Kept: {len(kept)}, Best: {kept.val_score.max():.6f}")
print(kept.nlargest(5, 'val_score')[['id', 'val_score', 'description']])
```

## Rules

- ONE variable per experiment
- Validation set for decisions, test set only at the end
- Track memory — OOM means not viable
- Fixed random seeds
- Log everything (stdout/stderr to run.log)
- Commit code before each experiment

## Autonomous mode

For unattended experimentation:
1. Set time budget per run (e.g., 5 minutes)
2. Loop: modify -> train -> evaluate -> record -> decide
3. Never pause for input
4. If stuck, try radical changes
5. ~8-10 experiments/hour depending on training time
