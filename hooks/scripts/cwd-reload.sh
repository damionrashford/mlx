#!/usr/bin/env bash
set -euo pipefail

# When CWD changes, scan new directory for ML artifacts.
# Output context if found, nothing if not an ML project.

TSV="${PWD}/results.tsv"
DATA_DIR="${PWD}/data"
MODELS=$(ls "${PWD}"/*.joblib "${PWD}"/*.pt "${PWD}"/*.xgb "${PWD}"/*.onnx 2>/dev/null || true)
NOTEBOOKS=$(ls "${PWD}"/*.ipynb 2>/dev/null || true)

FOUND=0

if [ -f "$TSV" ]; then
  TOTAL=$(awk -F'\t' 'NR>1{c++} END{print c+0}' "$TSV")
  BEST=$(awk -F'\t' 'NR>1 && $6=="KEEP"{if($3>b) b=$3} END{print b+0}' "$TSV")
  echo "ML project detected: ${TOTAL} experiments, best val_score=${BEST}"
  FOUND=1
fi

if [ -d "$DATA_DIR" ]; then
  DATA_COUNT=$(find "$DATA_DIR" -maxdepth 1 -type f 2>/dev/null | wc -l | tr -d ' ')
  echo "data/ directory: ${DATA_COUNT} file(s)"
  FOUND=1
fi

if [ -n "$MODELS" ]; then
  MODEL_COUNT=$(echo "$MODELS" | wc -l | tr -d ' ')
  echo "Model artifacts: ${MODEL_COUNT} file(s)"
  FOUND=1
fi

if [ -n "$NOTEBOOKS" ]; then
  NB_COUNT=$(echo "$NOTEBOOKS" | wc -l | tr -d ' ')
  echo "Notebooks: ${NB_COUNT} .ipynb file(s)"
  FOUND=1
fi

if [ "$FOUND" = "0" ]; then
  exit 0
fi
