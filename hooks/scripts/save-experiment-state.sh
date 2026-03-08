#!/usr/bin/env bash
# PreCompact — output a structured summary before context compaction runs.
# This text is added to Claude's context BEFORE compaction, ensuring the
# compaction summary captures ML experiment state rather than losing it.

set -euo pipefail

parts=()

# results.tsv state
if [ -f "results.tsv" ]; then
  total=$(awk 'NR>1' results.tsv | wc -l | tr -d ' ')
  keep=$(awk -F'\t' 'NR>1 && $6=="KEEP"' results.tsv 2>/dev/null | wc -l | tr -d ' ')
  discard=$(awk -F'\t' 'NR>1 && $6=="DISCARD"' results.tsv 2>/dev/null | wc -l | tr -d ' ')
  crash=$(awk -F'\t' 'NR>1 && $6=="CRASH"' results.tsv 2>/dev/null | wc -l | tr -d ' ')

  best_row=$(awk -F'\t' 'NR>1 && $3!=""' results.tsv 2>/dev/null | sort -t$'\t' -k3 -rn | head -1)
  best_score=$(echo "$best_row" | cut -f3)
  best_id=$(echo "$best_row" | cut -f1)

  parts+=("results.tsv: ${total} experiments | ${keep} KEEP | ${discard} DISCARD | ${crash} CRASH")
  if [ -n "$best_score" ] && [ -n "$best_id" ]; then
    parts+=("Best experiment: id=${best_id}, val_score=${best_score}")
  fi

  # Last 3 experiments
  last3=$(awk -F'\t' 'NR>1' results.tsv 2>/dev/null | tail -3 | awk -F'\t' '{print "  id="$1" val="$3" status="$6}')
  if [ -n "$last3" ]; then
    parts+=("Recent experiments:")
    while IFS= read -r line; do
      parts+=("  ${line}")
    done <<< "$last3"
  fi
fi

# Saved model files
models=$(ls *.joblib *.pt *.xgb *.onnx *.pkl 2>/dev/null | tr '\n' ' ' | sed 's/ $//')
if [ -n "$models" ]; then
  parts+=("Saved models: ${models}")
fi

# Current working directory context
parts+=("Working directory: $(pwd)")

# Output structured summary for compaction inclusion
if [ ${#parts[@]} -gt 0 ]; then
  echo "## ML Experiment State (pre-compaction snapshot)"
  for part in "${parts[@]}"; do
    echo "- ${part}"
  done
fi
