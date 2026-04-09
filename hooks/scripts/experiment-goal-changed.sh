#!/usr/bin/env bash
# FileChanged (EXPERIMENT.md) — re-inject active hypothesis when experiment goal changes.
# Stdout is written to debug log; use additionalContext JSON to reach Claude.

set -euo pipefail

EXPERIMENT_MD="${PWD}/EXPERIMENT.md"

if [ ! -f "$EXPERIMENT_MD" ]; then
  exit 0
fi

CONTENT=$(cat "$EXPERIMENT_MD")

if [ -z "$CONTENT" ]; then
  exit 0
fi

python3 -c "
import json, sys
content = sys.argv[1]
print(json.dumps({
    'additionalContext': 'EXPERIMENT.md updated:\n' + content
}))
" "$CONTENT"

exit 0
