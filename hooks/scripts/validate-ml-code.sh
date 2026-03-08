#!/usr/bin/env bash
# PreToolUse (Write|Edit) — ML code validation for Python files.
# Only activates on .py files. Exit 2 blocks the write; exit 0 allows it.

set -euo pipefail

# Read tool input JSON from stdin
INPUT=$(cat)

# Extract file path from tool input
FILE_PATH=$(echo "$INPUT" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(data.get('file_path') or data.get('path') or '')
except Exception:
    print('')
" 2>/dev/null || echo "")

# Only validate .py files
if [[ "$FILE_PATH" != *.py ]]; then
  exit 0
fi

# Extract content being written (if available)
CONTENT=$(echo "$INPUT" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(data.get('content') or data.get('new_string') or data.get('new_content') or '')
except Exception:
    print('')
" 2>/dev/null || echo "")

if [ -z "$CONTENT" ]; then
  exit 0
fi

errors=()

# Check 1: hardcoded absolute paths (portability killer)
if echo "$CONTENT" | grep -qE '(["'"'"'`])(/home/|/Users/)[^'"'"'"` ]+\1'; then
  errors+=("Hardcoded absolute path detected (/home/ or /Users/). Use relative paths or environment variables.")
fi

# Check 2: training scripts without reproducibility seed
# Only flag files that look like training scripts
if echo "$CONTENT" | grep -qiE '(fit|train|RandomForest|XGBClassifier|LightGBM|GradientBoosting|SGD|Adam)'; then
  if ! echo "$CONTENT" | grep -qiE '(random_state|seed|set_seed|torch\.manual_seed|np\.random\.seed)'; then
    errors+=("Training code without random_state/seed parameter. Add random_state=SEED for reproducibility.")
  fi
fi

# Check 3: data operations after test split (potential leakage)
# Look for fit/transform calls on test data (e.g. scaler.fit(X_test))
if echo "$CONTENT" | grep -qE '\.fit\s*\(\s*(X_test|y_test|test_)'; then
  errors+=("Potential data leakage: .fit() called on test data. Fit only on training data, then transform test.")
fi

# Output errors
if [ ${#errors[@]} -gt 0 ]; then
  echo "ML code validation issues found in ${FILE_PATH}:" >&2
  for err in "${errors[@]}"; do
    echo "  - ${err}" >&2
  done
  # Output JSON decision to block with reason
  python3 -c "
import json, sys
errors = sys.argv[1:]
print(json.dumps({'decision': 'block', 'reason': 'ML code validation: ' + '; '.join(errors)}))
" "${errors[@]}"
  exit 2
fi

exit 0
