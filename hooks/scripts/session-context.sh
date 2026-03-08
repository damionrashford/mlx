#!/usr/bin/env bash
# SessionStart (startup) — inject ML project state summary into Claude's context.
# Outputs to stdout only when ML artifacts are found (stdout = Claude's context injection).
# No output = no injection.

set -euo pipefail

found=0
parts=()

# data directory
if [ -d "data" ]; then
  n=$(find data -maxdepth 2 -type f 2>/dev/null | wc -l | tr -d ' ')
  parts+=("data/: ${n} files")
  found=1
fi

# results.tsv experiment log
if [ -f "results.tsv" ]; then
  total=$(awk 'NR>1' results.tsv | wc -l | tr -d ' ')
  best=$(awk -F'\t' 'NR>1 && $3!="" {print $3}' results.tsv 2>/dev/null | sort -rn | head -1)
  keep=$(awk -F'\t' 'NR>1 && $6=="KEEP"' results.tsv 2>/dev/null | wc -l | tr -d ' ')
  if [ -n "$best" ]; then
    parts+=("results.tsv: ${total} experiments, ${keep} KEEP, best val=${best}")
  else
    parts+=("results.tsv: ${total} experiments")
  fi
  found=1
fi

# saved model files
models=$(ls *.joblib *.pt *.xgb *.onnx *.pkl 2>/dev/null | tr '\n' ' ' | sed 's/ $//')
if [ -n "$models" ]; then
  parts+=("models: ${models}")
  found=1
fi

# inference API
if [ -f "serve.py" ]; then
  parts+=("serve.py: inference API present")
  found=1
fi

# Docker
if [ -f "Dockerfile" ]; then
  parts+=("Dockerfile: container config present")
  found=1
fi

# Only output when ML artifacts found
if [ "$found" -eq 1 ]; then
  echo "## ML Project State"
  for part in "${parts[@]}"; do
    echo "- ${part}"
  done
fi
