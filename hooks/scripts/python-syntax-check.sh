#!/usr/bin/env bash
# PostToolUse (Write|Edit, if: *.py) — syntax check after writing Python.
# Feeds compile errors back to Claude as additionalContext so they're fixed immediately.
# Never blocks (PostToolUse cannot block — file already written).

set -euo pipefail

INPUT=$(cat)

FILE_PATH=$(echo "$INPUT" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(data.get('tool_input', {}).get('file_path') or data.get('file_path') or '')
except Exception:
    print('')
" 2>/dev/null || echo "")

# Only check .py files that actually exist on disk
if [[ "$FILE_PATH" != *.py ]] || [ ! -f "$FILE_PATH" ]; then
  exit 0
fi

ERRORS=$(python3 -m py_compile "$FILE_PATH" 2>&1 || true)

if [ -n "$ERRORS" ]; then
  python3 -c "
import json, sys
msg = sys.argv[1]
print(json.dumps({'additionalContext': msg}))
" "Syntax error in ${FILE_PATH}: ${ERRORS}"
fi

exit 0
