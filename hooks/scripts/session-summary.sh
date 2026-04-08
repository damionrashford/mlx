#!/usr/bin/env bash
set -euo pipefail

# On session end, output ML session summary if ML artifacts are present.
# Outputs nothing if no ML artifacts found.

TSV="${PWD}/results.tsv"
MODELS=$(ls *.joblib *.pt *.xgb *.onnx 2>/dev/null || true)

if [ ! -f "$TSV" ] && [ -z "$MODELS" ]; then
  exit 0
fi

echo "=== ML Session Summary ==="

if [ -f "$TSV" ]; then
  TOTAL=$(awk -F'\t' 'NR>1{c++} END{print c+0}' "$TSV")
  KEEP=$(awk -F'\t' 'NR>1 && $6=="KEEP"{c++} END{print c+0}' "$TSV")
  BEST=$(awk -F'\t' 'NR>1 && $6=="KEEP"{if($3>b) b=$3} END{print b+0}' "$TSV")
  echo "Experiments: ${TOTAL} run, ${KEEP} KEEP, best val_score=${BEST}"
fi

if [ -n "$MODELS" ]; then
  COUNT=$(echo "$MODELS" | wc -l | tr -d ' ')
  echo "Models saved: ${COUNT} artifact(s): $(echo "$MODELS" | tr '\n' ' ')"
fi
