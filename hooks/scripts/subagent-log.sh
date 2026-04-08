#!/usr/bin/env bash
set -euo pipefail

# Reads SubagentStop JSON from stdin, extracts agent name, outputs one line.
# Async — never blocks.

INPUT=$(cat 2>/dev/null || true)
if [ -z "$INPUT" ]; then
  exit 0
fi

AGENT=$(echo "$INPUT" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(d.get('agent_name', d.get('name', 'unknown')))
except Exception:
    print('unknown')
" 2>/dev/null || echo "unknown")

echo "Subagent finished: ${AGENT}"
