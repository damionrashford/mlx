#!/usr/bin/env bash
# WorktreeCreate — initialize ML experiment workspace in a new git worktree.
# Fires when ml-engineer or dl-engineer agents create their isolated worktree.
# Sets up EXPERIMENT.md template, results.tsv header, and data/ directory.

set -euo pipefail

created=()

# Initialize results.tsv with header if not present
if [ ! -f "results.tsv" ]; then
  printf 'id\tmetric\tval_score\ttest_score\tmemory_mb\tstatus\tdescription\n' > results.tsv
  created+=("results.tsv")
fi

# Create EXPERIMENT.md template if not present
if [ ! -f "EXPERIMENT.md" ]; then
  cat > EXPERIMENT.md << 'TMPL'
# Experiment

## Hypothesis
<!-- What are we trying to improve and why? ONE variable only. -->

## Changes this run
<!-- What exactly changed from the last experiment? Be specific. -->

## Expected outcome
<!-- Target metric and expected direction (e.g. val_accuracy > 0.87) -->

## Results
<!-- Fill in after run: val_score, test_score, KEEP/DISCARD, reasoning -->

## Next to try
<!-- What would you try next based on these results? -->
TMPL
  created+=("EXPERIMENT.md")
fi

# Create data/ directory if absent
if [ ! -d "data" ]; then
  mkdir -p data
  created+=("data/")
fi

if [ ${#created[@]} -gt 0 ]; then
  echo "Worktree initialized: ${created[*]}"
else
  echo "Worktree ready (files already present)"
fi

exit 0
