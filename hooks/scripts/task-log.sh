#!/usr/bin/env bash
# TaskCreated / TaskCompleted — log ML task lifecycle for experiment auditing.
# Provides a trail of what the autonomous experiment loop actually did.

set -euo pipefail

INPUT=$(cat 2>/dev/null || true)
if [ -z "$INPUT" ]; then
  exit 0
fi

EVENT=$(echo "$INPUT" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(d.get('hook_event_name', d.get('event', 'TaskEvent')))
except Exception:
    print('TaskEvent')
" 2>/dev/null || echo "TaskEvent")

TASK=$(echo "$INPUT" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    t = d.get('task', d.get('title', d.get('content', d.get('description', ''))))
    if isinstance(t, dict):
        val = t.get('content', t.get('title', str(t)))
    else:
        val = str(t)
    print(val[:100].strip())
except Exception:
    print('')
" 2>/dev/null || echo "")

timestamp=$(date '+%Y-%m-%d %H:%M:%S')
LOG="${CLAUDE_PLUGIN_DATA}/task-log.txt"
mkdir -p "${CLAUDE_PLUGIN_DATA}"
echo "[${timestamp}] ${EVENT}: ${TASK}" >> "$LOG"

exit 0
