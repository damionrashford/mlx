---
name: autoexperiment
description: >
  Autonomous time-budget experiment loop. Modify a training script, train for a
  fixed wall-clock budget, evaluate, record, repeat. Inspired by karpathy/autoresearch.
  Use for overnight architecture search, systematic hyperparameter sweeps, or any
  iterative model improvement workflow.
allowed-tools: Bash, Read, Write, Edit, Glob, Grep
argument-hint: path to train.py or description of experiment goal
---

# Autoexperiment Skill

Run autonomous time-budget experiment loops. Each iteration modifies `train.py`,
trains for a fixed wall-clock budget, evaluates, records in `results.tsv`, and repeats.

## Setup

1. Ensure `results.tsv` exists with a baseline (`exp000`) before iterating
2. Create `EXPERIMENT.md` with your goal, baseline, hypothesis, and constraints
3. Run: `/mlx:autoexperiment path/to/train.py`

## Protocol

### Before each iteration
1. Read `EXPERIMENT.md` for the current hypothesis
2. Read `results.tsv` for experiment history
3. Identify one change to make (ONE variable only)

### Iteration loop
1. Edit `train.py` with the single change
2. Run with TIME_BUDGET: `timeout $BUDGET python3 train.py`
3. Capture exit code and metrics
4. Record in `results.tsv`: KEEP / DISCARD / CRASH
5. If CRASH 3× in a row on the same error → stop, report diagnosis

### After each iteration
- Update `EXPERIMENT.md` "Next to try" section
- Summarize: what changed, what happened, what's next

## Templates

See `references/EXPERIMENT.md.template` for the hypothesis file format.
See `scripts/time_budget_train.py` for a complete training script template with all patterns.

## Key patterns

- **TIME_BUDGET**: wall-clock seconds, not epochs. ~12 experiments/hour at 300s each
- **val_bpb**: `total_nats / (math.log(2) * total_bytes)` — vocab-independent metric
- **GC freeze**: after step 0 eliminates ~500ms stalls
- **Fast fail**: `if math.isnan(loss) or loss > 100: sys.exit(1)`
- **Circuit breaker**: 3 consecutive CRASHes on same error → escalate to user

See `references/autoexperiment-guide.md` for full documentation.
