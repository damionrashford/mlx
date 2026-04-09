#!/usr/bin/env bash
# Stop/SubagentStop — emit ML project state back to parent agent before subagent closes.
# Shared by ml-engineer, dl-engineer, data-scientist agents.
# Output goes to additionalContext so parent sees experiment/pipeline state.

set -euo pipefail

parts=()

# Experiment log
if [ -f "results.tsv" ]; then
  total=$(awk 'NR>1' results.tsv | wc -l | tr -d ' ')
  keep=$(awk -F'\t' 'NR>1 && $6=="KEEP"' results.tsv 2>/dev/null | wc -l | tr -d ' ')
  discard=$(awk -F'\t' 'NR>1 && $6=="DISCARD"' results.tsv 2>/dev/null | wc -l | tr -d ' ')

  best_row=$(awk -F'\t' 'NR>1 && $3!=""' results.tsv 2>/dev/null | sort -t$'\t' -k3 -rn | head -1)
  best_score=$(echo "$best_row" | cut -f3)
  best_id=$(echo "$best_row" | cut -f1)
  best_desc=$(echo "$best_row" | cut -f7)

  parts+=("Experiments: ${total} run | ${keep} KEEP | ${discard} DISCARD")
  if [ -n "$best_score" ] && [ -n "$best_id" ]; then
    parts+=("Best: id=${best_id} val=${best_score} — ${best_desc}")
  fi
fi

# Saved model artifacts
models=$(ls *.joblib *.pt *.xgb *.onnx *.pkl 2>/dev/null | tr '\n' ' ' | sed 's/ $//' || true)
if [ -n "$models" ]; then
  parts+=("Saved models: ${models}")
fi

# data directory
if [ -d "data" ]; then
  n=$(find data -maxdepth 2 -type f 2>/dev/null | wc -l | tr -d ' ')
  parts+=("data/: ${n} files")
fi

# Active EXPERIMENT.md
if [ -f "EXPERIMENT.md" ]; then
  goal=$(head -5 EXPERIMENT.md | tr '\n' ' ' | sed 's/  */ /g')
  parts+=("EXPERIMENT.md: ${goal}")
fi

if [ ${#parts[@]} -eq 0 ]; then
  exit 0
fi

summary=$(printf '%s\n' "${parts[@]}" | awk '{print "- " $0}')

python3 -c "
import json, sys
print(json.dumps({'additionalContext': 'Agent final state:\n' + sys.argv[1]}))
" "$summary"

exit 0
