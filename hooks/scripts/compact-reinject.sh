#!/usr/bin/env bash
# SessionStart (compact) — re-inject experiment state after context compaction.
# Applies the "re-inject context after compaction" pattern from hooks documentation.
# Without this, results.tsv history is lost after compaction.

set -euo pipefail

parts=()

# Experiment log
if [ -f "results.tsv" ]; then
  total=$(awk 'NR>1' results.tsv | wc -l | tr -d ' ')
  keep=$(awk -F'\t' 'NR>1 && $6=="KEEP"' results.tsv 2>/dev/null | wc -l | tr -d ' ')
  discard=$(awk -F'\t' 'NR>1 && $6=="DISCARD"' results.tsv 2>/dev/null | wc -l | tr -d ' ')
  crash=$(awk -F'\t' 'NR>1 && $6=="CRASH"' results.tsv 2>/dev/null | wc -l | tr -d ' ')

  best_row=$(awk -F'\t' 'NR>1 && $3!=""' results.tsv 2>/dev/null | sort -t$'\t' -k3 -rn | head -1)
  best_score=$(echo "$best_row" | cut -f3)
  best_id=$(echo "$best_row" | cut -f1)

  parts+=("Experiments: ${total} total | ${keep} KEEP | ${discard} DISCARD | ${crash} CRASH")
  if [ -n "$best_score" ] && [ -n "$best_id" ]; then
    parts+=("Best run: id=${best_id}, val_score=${best_score}")
  fi
fi

# Saved models
models=$(ls *.joblib *.pt *.xgb *.onnx *.pkl 2>/dev/null | tr '\n' ' ' | sed 's/ $//')
if [ -n "$models" ]; then
  parts+=("Saved models: ${models}")
fi

# Current plan file if present
if [ -f "scratch/current_plan.yaml" ]; then
  parts+=("Plan: scratch/current_plan.yaml exists — read to restore task state")
fi

# Output only when there's something to reinject
if [ ${#parts[@]} -gt 0 ]; then
  echo "## Experiment State (restored after compaction)"
  for part in "${parts[@]}"; do
    echo "- ${part}"
  done
fi
