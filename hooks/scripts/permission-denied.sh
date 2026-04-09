#!/usr/bin/env bash
# PermissionDenied — handle auto-mode blocked tool calls.
# Returns {retry: true} for non-destructive blocks so the model can rethink and retry.
# Returns nothing (silent exit 0) for destructive blocks — keep them denied.

set -euo pipefail

INPUT=$(cat 2>/dev/null || true)
if [ -z "$INPUT" ]; then
  exit 0
fi

TOOL=$(echo "$INPUT" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(d.get('tool_name', d.get('tool', '')))
except Exception:
    print('')
" 2>/dev/null || echo "")

REASON=$(echo "$INPUT" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(d.get('reason', d.get('denial_reason', d.get('message', ''))))
except Exception:
    print('')
" 2>/dev/null || echo "")

REASON_LOWER=$(echo "$REASON" | tr '[:upper:]' '[:lower:]')

# Never retry if the block was for an explicitly destructive ML operation
if echo "$REASON_LOWER" | grep -qE 'force push|reset --hard|delete.*model|rm.*artifact|kubectl delete|docker rm.*-f'; then
  exit 0
fi

# Allow retry for read/search operations — these are safe to reattempt
if echo "$TOOL" | grep -qiE '^(Read|Glob|Grep|WebFetch|WebSearch)$'; then
  python3 -c "import json; print(json.dumps({'retry': True}))"
  exit 0
fi

# Allow retry for Bash commands that look like info-gathering (not destructive)
if [ "$TOOL" = "Bash" ]; then
  COMMAND=$(echo "$INPUT" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(d.get('tool_input', {}).get('command', ''))
except Exception:
    print('')
" 2>/dev/null || echo "")
  if echo "$COMMAND" | grep -qE '^(ls|cat|head|tail|which|echo|pwd|git log|git status|git diff|uv run)'; then
    python3 -c "import json; print(json.dumps({'retry': True}))"
    exit 0
  fi
fi

exit 0
