#!/usr/bin/env bash
# WorktreeRemove — archive experiment results before a worktree is torn down.
# Fires when ml-engineer or dl-engineer agents finish and their worktree is removed.
# Copies results.tsv and EXPERIMENT.md into .claude/experiments/ in the main repo.

set -euo pipefail

# Locate the main repo root (parent of the worktree)
MAIN_REPO=$(git rev-parse --show-superproject-working-tree 2>/dev/null \
  || git -C .. rev-parse --show-toplevel 2>/dev/null \
  || true)

if [ -z "$MAIN_REPO" ] || [ ! -d "$MAIN_REPO" ]; then
  exit 0
fi

ARCHIVE_DIR="${MAIN_REPO}/.claude/experiments"
mkdir -p "$ARCHIVE_DIR"

timestamp=$(date +%Y%m%d-%H%M%S)
archived=()

# Archive results.tsv if it has experiment rows (more than just the header)
if [ -f "results.tsv" ] && [ "$(awk 'NR>1' results.tsv | wc -l | tr -d ' ')" -gt 0 ]; then
  cp results.tsv "${ARCHIVE_DIR}/results-${timestamp}.tsv"
  archived+=("results.tsv → .claude/experiments/results-${timestamp}.tsv")
fi

# Archive EXPERIMENT.md
if [ -f "EXPERIMENT.md" ]; then
  cp EXPERIMENT.md "${ARCHIVE_DIR}/EXPERIMENT-${timestamp}.md"
  archived+=("EXPERIMENT.md → .claude/experiments/EXPERIMENT-${timestamp}.md")
fi

# Archive any saved model artifacts (copy names only, not the binaries — those stay in worktree)
models=$(ls *.joblib *.pt *.xgb *.onnx *.pkl *.safetensors 2>/dev/null | tr '\n' ' ' | xargs || true)
if [ -n "$models" ]; then
  echo "$models" > "${ARCHIVE_DIR}/artifacts-${timestamp}.txt"
  archived+=("artifact list → .claude/experiments/artifacts-${timestamp}.txt")
fi

if [ ${#archived[@]} -gt 0 ]; then
  echo "Archived from worktree:"
  printf '  %s\n' "${archived[@]}"
fi

exit 0
