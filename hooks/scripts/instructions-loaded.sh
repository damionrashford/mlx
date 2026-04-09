#!/usr/bin/env bash
# InstructionsLoaded — reinject active ML experiment state when CLAUDE.md loads.
# Fires at session start and when .claude/rules/*.md files are lazily loaded.
# Complements session-context.sh (SessionStart) with a lightweight re-injection.

set -euo pipefail

parts=()

# Active experiment hypothesis
if [ -f "EXPERIMENT.md" ]; then
  goal=$(grep -m1 -A2 "^## Hypothesis" EXPERIMENT.md 2>/dev/null \
    | grep -v "^##" | grep -v "^<!--" | head -1 | xargs || true)
  [ -n "$goal" ] && parts+=("Active hypothesis: ${goal}")
fi

# Experiment log summary
if [ -f "results.tsv" ]; then
  row_count=$(awk 'NR>1' results.tsv | wc -l | tr -d ' ')
  if [ "$row_count" -gt 0 ]; then
    keep=$(awk -F'\t' 'NR>1 && $6=="KEEP"' results.tsv 2>/dev/null | wc -l | tr -d ' ')
    best=$(awk -F'\t' 'NR>1 && $3!=""' results.tsv 2>/dev/null \
      | sort -t$'\t' -k3 -rn | head -1 | cut -f1,3,7)
    parts+=("Experiments: ${row_count} runs, ${keep} KEEP — best: ${best}")
  fi
fi

if [ ${#parts[@]} -eq 0 ]; then
  exit 0
fi

context=$(printf '%s\n' "${parts[@]}" | awk '{print "- " $0}')

python3 -c "
import json, sys
print(json.dumps({'additionalContext': 'ML context (instructions loaded):\n' + sys.argv[1]}))
" "$context"

exit 0
