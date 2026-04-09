#!/usr/bin/env python3
# /// script
# dependencies = [
#   "numpy>=1.24",
# ]
# requires-python = ">=3.10"
# ///
# time_budget_train.py — Template training script with autoexperiment patterns.
# All patterns from karpathy/autoresearch encoded here.
# Customize: replace the model definition and data loading sections.

import os
import sys
import gc
import csv
import math
import time
import random
import itertools
import argparse

import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
TIME_BUDGET = int(os.environ.get("TIME_BUDGET", "300"))   # seconds
EXPERIMENT_ID = os.environ.get("EXPERIMENT_ID", "exp000")
METRIC = os.environ.get("METRIC", "val_loss")
RANDOM_SEED = int(os.environ.get("RANDOM_SEED", "42"))
RESULTS_TSV = os.environ.get("RESULTS_TSV", "results.tsv")

# ── Hyperparameters (edit these per experiment) ────────────────────────────────
LR = float(os.environ.get("LR", "1e-3"))
HIDDEN_DIM = int(os.environ.get("HIDDEN_DIM", "128"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "32"))
WARMUP_FRAC = 0.05   # fraction of budget for LR warmup
WARMDOWN_START = 0.5  # decay from 50% of budget (not 10%)
EMA_BETA = 0.9

# ── Reproducibility ───────────────────────────────────────────────────────────
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ── Model (replace with your model) ──────────────────────────────────────────
class SimpleMLP:
    """Minimal numpy MLP for illustration."""
    def __init__(self, in_dim, hidden_dim, out_dim):
        scale = math.sqrt(2.0 / in_dim)
        self.W1 = np.random.randn(in_dim, hidden_dim) * scale
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, out_dim) * scale
        self.b2 = np.zeros(out_dim)

    def forward(self, x):
        h = np.maximum(0, x @ self.W1 + self.b1)  # ReLU
        return h @ self.W2 + self.b2

    def num_params(self):
        return (self.W1.size + self.b1.size + self.W2.size + self.b2.size)


def mse_loss(pred, target):
    return float(np.mean((pred - target) ** 2))


# ── Data (replace with your data loading) ────────────────────────────────────
def load_data():
    """Replace with real data loading. Returns (X_train, y_train, X_val, y_val)."""
    N, in_dim = 10000, 16
    X = np.random.randn(N, in_dim).astype(np.float32)
    y = X[:, :1] * 2.0 + np.random.randn(N, 1).astype(np.float32) * 0.1  # simple linear target

    split = int(0.8 * N)
    return X[:split], y[:split], X[split:], y[split:]


# ── LR schedule ───────────────────────────────────────────────────────────────
def get_lr(elapsed, budget, lr_max):
    frac = elapsed / budget
    if frac < WARMUP_FRAC:
        return lr_max * (frac / WARMUP_FRAC)
    elif frac < WARMDOWN_START:
        return lr_max
    else:
        # Linear warmdown from WARMDOWN_START to end of budget
        decay_frac = (frac - WARMDOWN_START) / (1.0 - WARMDOWN_START)
        return lr_max * max(0.0, 1.0 - decay_frac)


# ── Experiment result writer ───────────────────────────────────────────────────
def write_result(exp_id, metric, val_score, status, description, memory_mb=0.0, test_score=None):
    exists = os.path.exists(RESULTS_TSV)
    columns = ["id", "metric", "val_score", "test_score", "memory_mb", "status", "description"]
    row = {
        "id": exp_id,
        "metric": metric,
        "val_score": f"{val_score:.6f}",
        "test_score": f"{test_score:.6f}" if test_score is not None else "",
        "memory_mb": f"{memory_mb:.1f}",
        "status": status,
        "description": description,
    }
    with open(RESULTS_TSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, delimiter="\t")
        if not exists:
            writer.writeheader()
        writer.writerow(row)
    print(f"Recorded: {exp_id} | {metric}={val_score:.6f} | {status}")


# ── Main training loop ────────────────────────────────────────────────────────
def train():
    print(f"Experiment: {EXPERIMENT_ID} | Budget: {TIME_BUDGET}s | LR: {LR}")

    X_train, y_train, X_val, y_val = load_data()
    in_dim = X_train.shape[1]
    out_dim = y_train.shape[1] if y_train.ndim > 1 else 1

    model = SimpleMLP(in_dim, HIDDEN_DIM, out_dim)
    print(f"Model params: {model.num_params():,}")

    # EMA loss tracking
    smooth_loss = 0.0
    t0 = time.monotonic()
    step = 0
    n = len(X_train)

    try:
        for step in itertools.count():
            elapsed = time.monotonic() - t0
            if elapsed >= TIME_BUDGET:
                break

            # Mini-batch
            idx = np.random.choice(n, BATCH_SIZE, replace=False)
            X_batch = X_train[idx]
            y_batch = y_train[idx] if y_train.ndim > 1 else y_train[idx, None]

            # Forward + manual gradient (replace with torch/sklearn as needed)
            pred = model.forward(X_batch)
            loss = mse_loss(pred, y_batch)

            # Fast fail guard
            if math.isnan(loss) or loss > 100:
                print(f"Fast fail: loss={loss:.4f} at step {step}", file=sys.stderr)
                write_result(EXPERIMENT_ID, METRIC, float("inf"), "CRASH",
                             f"Fast fail at step {step}: loss={loss:.4f}")
                sys.exit(1)

            # EMA debiased display loss
            smooth_loss = EMA_BETA * smooth_loss + (1 - EMA_BETA) * loss
            display_loss = smooth_loss / (1 - EMA_BETA ** (step + 1))

            # GC freeze after step 0 (eliminates ~500ms stalls)
            if step == 0:
                gc.collect()
                gc.freeze()
                gc.disable()

            # Simple gradient update (replace with optimizer)
            lr = get_lr(elapsed, TIME_BUDGET, LR)
            # (placeholder: real gradient update goes here)

            if step % 100 == 0:
                remaining = TIME_BUDGET - elapsed
                print(f"step={step:6d} | loss={display_loss:.4f} | lr={lr:.2e} | remaining={remaining:.0f}s")

    except KeyboardInterrupt:
        print("Interrupted by user")

    # Evaluate on validation set
    val_pred = model.forward(X_val)
    val_loss = mse_loss(val_pred, y_val[:, None] if y_val.ndim == 1 else y_val)

    total_steps = step
    elapsed = time.monotonic() - t0
    print(f"\nTraining complete: {total_steps} steps in {elapsed:.1f}s")
    print(f"Val loss: {val_loss:.6f}")

    # Memory usage
    import resource
    mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    description = f"hidden={HIDDEN_DIM} lr={LR} steps={total_steps}"
    write_result(EXPERIMENT_ID, METRIC, val_loss, "KEEP", description, memory_mb=mem_mb)

    return val_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Time-budget training script")
    parser.add_argument("--budget", type=int, default=TIME_BUDGET, help="Training budget in seconds")
    parser.add_argument("--exp-id", default=EXPERIMENT_ID, help="Experiment ID for results.tsv")
    args = parser.parse_args()

    TIME_BUDGET = args.budget
    EXPERIMENT_ID = args.exp_id

    val_loss = train()
    sys.exit(0)
