#!/usr/bin/env bash
# UserPromptSubmit — fast ML anti-pattern detection and agent routing hints.
# Pattern matching only — no LLM calls. Fires before every user prompt is processed.
# Outputs additionalContext so Claude sees warnings before generating a response.

set -euo pipefail

INPUT=$(cat 2>/dev/null || true)
if [ -z "$INPUT" ]; then
  exit 0
fi

PROMPT=$(echo "$INPUT" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(d.get('prompt', d.get('message', d.get('content', ''))))
except Exception:
    print('')
" 2>/dev/null || echo "")

if [ -z "$PROMPT" ]; then
  exit 0
fi

PROMPT_LOWER=$(echo "$PROMPT" | tr '[:upper:]' '[:lower:]')

warnings=()
hints=()

# ── Anti-patterns ────────────────────────────────────────────────────────────

# Train on test set
if echo "$PROMPT_LOWER" | grep -qE 'train.*on.*(the )?test|fit.*on.*(the )?test|use.*test.*for.*train'; then
  warnings+=("⚠️  Anti-pattern: training on test data. Fit only on train/val sets; keep test held-out until final evaluation.")
fi

# Evaluate on full dataset (no split)
if echo "$PROMPT_LOWER" | grep -qE 'accuracy on (the )?full|eval(uate)?.*(all|entire|whole).*(data|dataset)|score.*everything'; then
  warnings+=("⚠️  Evaluate on held-out split only (val or test), not full dataset — this inflates reported metrics.")
fi

# Hyperparameter tuning on test set
if echo "$PROMPT_LOWER" | grep -qE 'tune.*(hyper)?param.*test|pick.*model.*test set|select.*threshold.*test'; then
  warnings+=("⚠️  Hyperparameter/threshold selection must use validation set, not test set — test is for final reporting only.")
fi

# ── Routing hints ─────────────────────────────────────────────────────────────

# Research / paper task
if echo "$PROMPT_LOWER" | grep -qE 'find.*paper|search.*arxiv|latest.*research|survey.*on|literature review|paper on'; then
  hints+=("💡 ml-researcher handles paper search, dataset discovery, and paper → podcast generation.")
fi

# Business analysis / KPI task
if echo "$PROMPT_LOWER" | grep -qE '\brevenue\b|\bconversion\b|\bretention\b|cohort|a/b test|\bkpi\b|dashboard|segment users|rfm'; then
  hints+=("💡 data-analyst handles business metrics, A/B tests, cohort analysis, and dashboards.")
fi

# Deep learning / neural network task
if echo "$PROMPT_LOWER" | grep -qE 'transformer|neural net|cnn|rnn|lstm|attention.*head|gradient.*vanish|backprop|gpu.*memory|cuda.*oom|mixed precision|distributed.*train'; then
  hints+=("💡 dl-engineer handles neural network design, GPU optimization, and distributed training.")
fi

# RAG / LLM application task
if echo "$PROMPT_LOWER" | grep -qE '\brag\b|retrieval.*augment|vector.*store|embedding.*search|llm.*app|chatbot|build.*with.*llm|prompt.*engineer'; then
  hints+=("💡 ai-engineer handles RAG pipelines, LLM application development, and fine-tuning.")
fi

# ── Output ────────────────────────────────────────────────────────────────────

if [ ${#warnings[@]} -eq 0 ] && [ ${#hints[@]} -eq 0 ]; then
  exit 0
fi

lines=()
for w in "${warnings[@]}"; do lines+=("$w"); done
for h in "${hints[@]}"; do lines+=("$h"); done

context=$(printf '%s\n' "${lines[@]}")

python3 -c "
import json, sys
print(json.dumps({'additionalContext': sys.argv[1]}))
" "$context"

exit 0
