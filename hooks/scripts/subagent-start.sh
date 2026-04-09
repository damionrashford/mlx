#!/usr/bin/env bash
# SubagentStart — log when a specialized agent starts and inject any relevant context.
# Complement to subagent-log.sh (SubagentStop).

set -euo pipefail

INPUT=$(cat 2>/dev/null || true)
if [ -z "$INPUT" ]; then
  exit 0
fi

AGENT=$(echo "$INPUT" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(d.get('agent_type', d.get('agent_name', 'unknown')))
except Exception:
    print('unknown')
" 2>/dev/null || echo "unknown")

timestamp=$(date '+%Y-%m-%d %H:%M:%S')

# Log to CLAUDE_PLUGIN_DATA for audit trail
LOG="${CLAUDE_PLUGIN_DATA}/agent-log.txt"
mkdir -p "${CLAUDE_PLUGIN_DATA}"
echo "[${timestamp}] SubagentStart: ${AGENT}" >> "$LOG"

echo "Subagent started: ${AGENT}"
exit 0
