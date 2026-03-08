#!/usr/bin/env bash
# PostToolUseFailure (Bash) — ML-specific error pattern advisor.
# Fires only when a Bash command fails. Exit 0 always (advisory only, never blocks).
# Outputs additionalContext JSON with actionable suggestions.

set -euo pipefail

# Read failure info from stdin
INPUT=$(cat)

# Extract error text
ERROR=$(echo "$INPUT" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    err = data.get('error') or data.get('stderr') or data.get('output') or ''
    print(str(err))
except Exception:
    print('')
" 2>/dev/null || echo "")

if [ -z "$ERROR" ]; then
  exit 0
fi

suggestion=""

# Pattern matching — most specific first
if echo "$ERROR" | grep -qiE 'cuda out of memory|RuntimeError.*CUDA|out of memory'; then
  suggestion="GPU OOM: reduce batch_size (try halving it), enable gradient checkpointing (model.gradient_checkpointing_enable()), or use mixed precision (torch.autocast). If persisting, switch to CPU or use chunked inference."

elif echo "$ERROR" | grep -qiE 'ModuleNotFoundError|No module named'; then
  mod=$(echo "$ERROR" | grep -ioE "No module named '([^']+)'" | head -1 | sed "s/No module named '//;s/'//")
  if [ -n "$mod" ]; then
    suggestion="Missing module '${mod}': run 'pip install ${mod}'. If in a notebook, use '!pip install ${mod}' in a cell."
  else
    suggestion="Missing Python module. Check import name and run 'pip install <package>'."
  fi

elif echo "$ERROR" | grep -qiE "KeyError.*column|KeyError.*'[A-Za-z_]+'.*DataFrame|columns"; then
  suggestion="Column KeyError: check available columns with df.columns.tolist(). Column names are case-sensitive and may have leading/trailing spaces. Use df.columns.str.strip() to normalize."

elif echo "$ERROR" | grep -qiE 'MemoryError|Killed|memory allocation|cannot allocate'; then
  suggestion="Memory exhaustion: process data in chunks (pd.read_csv(..., chunksize=10000)), reduce dataset size for prototyping, or use polars/dask for out-of-core processing."

elif echo "$ERROR" | grep -qiE 'could not convert string to float|ValueError.*float|invalid literal for float'; then
  suggestion="String-to-float conversion error: column contains non-numeric values. Check with df['col'].unique() or df['col'].value_counts(). Use pd.to_numeric(df['col'], errors='coerce') to force conversion, then handle NaNs."

elif echo "$ERROR" | grep -qiE 'FileNotFoundError|No such file or directory'; then
  path=$(echo "$ERROR" | grep -oE "'[^']+'" | head -1 | tr -d "'")
  if [ -n "$path" ]; then
    suggestion="File not found: '${path}'. Verify the path with ls and ensure you're in the correct working directory (pwd). Use relative paths from project root."
  else
    suggestion="File not found. Verify paths with ls and check current working directory with pwd."
  fi

elif echo "$ERROR" | grep -qiE 'ConvergenceWarning|failed to converge|max_iter'; then
  suggestion="Model convergence issue: increase max_iter (e.g., max_iter=1000), scale features with StandardScaler, or check for NaN values in input data."

elif echo "$ERROR" | grep -qiE 'DataConversionWarning|dtype|int.*float'; then
  suggestion="Data type issue: ensure feature matrix X is float32/float64 (X = X.astype(float)). Check for mixed types in columns."
fi

# Output additionalContext if suggestion found
if [ -n "$suggestion" ]; then
  python3 -c "
import json, sys
print(json.dumps({'additionalContext': sys.argv[1]}))
" "$suggestion"
fi

exit 0
