#!/usr/bin/env bash
# PreToolUse (Bash) — ml-ops agent safety guard.
# Blocks destructive deployment commands that are hard to reverse.
# Uses hookSpecificOutput.permissionDecision (exit 0 + JSON per docs).

set -euo pipefail

INPUT=$(cat)

COMMAND=$(echo "$INPUT" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(data.get('tool_input', {}).get('command', ''))
except Exception:
    print('')
" 2>/dev/null || echo "")

if [ -z "$COMMAND" ]; then
  exit 0
fi

reason=""

# git force push — can overwrite remote history irreversibly
if echo "$COMMAND" | grep -qE 'git push.*(--force|-f)'; then
  reason="Force push blocked: --force overwrites remote history. Use --force-with-lease if you must, or open a PR instead."

# git reset --hard — discards uncommitted work
elif echo "$COMMAND" | grep -qE 'git reset --hard'; then
  reason="git reset --hard blocked: discards all uncommitted changes. Stash first if work needs preserving."

# docker remove containers/images with force
elif echo "$COMMAND" | grep -qE 'docker (rm|rmi) .*-f|docker (rm|rmi) -f'; then
  reason="Forced docker remove blocked in ml-ops agent. Confirm container/image names before removing."

# kubectl delete on broad resource types
elif echo "$COMMAND" | grep -qE 'kubectl delete (deployment|service|namespace|pod|statefulset|daemonset)'; then
  reason="kubectl delete blocked: removing live k8s resources. Confirm this is the right environment and resource."

# rm -rf on anything that looks like model/data directories
elif echo "$COMMAND" | grep -qE 'rm -rf.*(model|data|checkpoint|artifact|weights|\.pt|\.joblib|\.onnx)'; then
  reason="Destructive rm blocked: command targets model/data files. Verify paths before deleting ML artifacts."
fi

if [ -n "$reason" ]; then
  python3 -c "
import json, sys
print(json.dumps({
    'hookSpecificOutput': {
        'hookEventName': 'PreToolUse',
        'permissionDecision': 'deny',
        'permissionDecisionReason': sys.argv[1]
    }
}))
" "$reason"
  exit 0
fi

exit 0
