#!/usr/bin/env bash
# PostToolUse (Bash, async) — detect training metrics and errors in bash output.
# Replaces the expensive prompt-based hook. Zero LLM calls.
# Async: Claude is never blocked. Output delivered on next turn as systemMessage.

set -euo pipefail

# Read tool response JSON from stdin
INPUT=$(cat)

# Extract the bash output
OUTPUT=$(echo "$INPUT" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    # tool_response can be nested
    out = data.get('output') or data.get('stdout') or data.get('tool_response') or ''
    if isinstance(out, dict):
        out = out.get('output') or out.get('stdout') or ''
    print(str(out))
except Exception:
    print('')
" 2>/dev/null || echo "")

if [ -z "$OUTPUT" ]; then
  exit 0
fi

message=""

# Detect training metrics (loss, accuracy, val_, test_ patterns)
if echo "$OUTPUT" | grep -qiE '(loss|accuracy|val_|test_|epoch|train_loss|f1|auc|rmse|mae)'; then
  # Extract metric lines
  metrics=$(echo "$OUTPUT" | grep -iE '(loss|accuracy|val_|test_|epoch|f1|auc|rmse|mae)' | tail -5)
  message="Training metrics detected:\n${metrics}"
fi

# Detect OOM / CUDA errors (prioritize over metrics message)
if echo "$OUTPUT" | grep -qiE '(cuda out of memory|out of memory|killed|oom)'; then
  message="GPU/memory issue detected in output. Consider reducing batch_size or enabling gradient checkpointing."
fi

if [ -n "$message" ]; then
  # Output valid JSON with systemMessage for next-turn delivery
  python3 -c "
import json, sys
msg = sys.argv[1]
print(json.dumps({'systemMessage': msg}))
" "$message"
fi

exit 0
