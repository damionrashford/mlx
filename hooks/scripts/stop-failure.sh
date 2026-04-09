#!/usr/bin/env bash
# StopFailure — handle turns that end due to API errors.
# Logs the failure and emits recovery context so the next turn can resume cleanly.

set -euo pipefail

INPUT=$(cat 2>/dev/null || true)

ERROR="unknown error"
if [ -n "$INPUT" ]; then
  ERROR=$(echo "$INPUT" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(d.get('error', d.get('message', d.get('reason', 'unknown error'))))
except Exception:
    print('unknown error')
" 2>/dev/null || echo "unknown error")
fi

timestamp=$(date '+%Y-%m-%d %H:%M:%S')
LOG="${CLAUDE_PLUGIN_DATA}/stop-failures.log"
mkdir -p "${CLAUDE_PLUGIN_DATA}"
echo "[${timestamp}] StopFailure: ${ERROR}" >> "$LOG"

# Preserve whatever experiment state exists so the next turn can resume
context_parts=()

if [ -f "results.tsv" ] && [ "$(awk 'NR>1' results.tsv | wc -l | tr -d ' ')" -gt 0 ]; then
  total=$(awk 'NR>1' results.tsv | wc -l | tr -d ' ')
  context_parts+=("Experiment log intact: ${total} rows in results.tsv")
fi

if [ -f "EXPERIMENT.md" ]; then
  context_parts+=("EXPERIMENT.md present — hypothesis preserved")
fi

context="Previous turn ended due to API error: ${ERROR}. Resume from where you left off."
if [ ${#context_parts[@]} -gt 0 ]; then
  context+=" State: $(printf '%s; ' "${context_parts[@]}")"
fi

python3 -c "
import json, sys
print(json.dumps({'additionalContext': sys.argv[1]}))
" "$context"

exit 0
