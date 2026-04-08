#!/usr/bin/env bash
set -euo pipefail

# After compaction, re-inject experiment state AND active hypothesis from EXPERIMENT.md.

TSV="${PWD}/results.tsv"
EXPERIMENT_MD="${PWD}/EXPERIMENT.md"

if [ -f "$TSV" ]; then
  TOTAL=$(awk -F'\t' 'NR>1{c++} END{print c+0}' "$TSV")
  KEEP=$(awk -F'\t' 'NR>1 && $6=="KEEP"{c++} END{print c+0}' "$TSV")
  BEST=$(awk -F'\t' 'NR>1 && $6=="KEEP"{if($3>b) b=$3} END{print b+0}' "$TSV")
  BEST_ID=$(awk -F'\t' 'NR>1 && $6=="KEEP"{if($3>b){b=$3; id=$1}} END{print id}' "$TSV")
  echo "=== Experiment State (post-compact) ==="
  echo "Total: ${TOTAL} | KEEP: ${KEEP} | Best: ${BEST_ID} val=${BEST}"
fi

if [ -f "$EXPERIMENT_MD" ]; then
  echo ""
  echo "=== Active Hypothesis (EXPERIMENT.md) ==="
  cat "$EXPERIMENT_MD"
fi
