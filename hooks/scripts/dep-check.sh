#!/usr/bin/env bash
# SessionStart — warm uv cache for the experiments MCP server using CLAUDE_PLUGIN_DATA.
# Checks if the server script changed since last session; pre-resolves deps if so.
# Uses the persistent data dir pattern: compare hash, re-warm only when needed.

set -euo pipefail

DATA_DIR="${CLAUDE_PLUGIN_DATA}"
mkdir -p "$DATA_DIR"

# Touch heartbeat
touch "${DATA_DIR}/.active"

SERVER="${CLAUDE_PLUGIN_ROOT}/servers/experiments.py"
if [ ! -f "$SERVER" ]; then
  exit 0
fi

STAMP_FILE="${DATA_DIR}/experiments-hash"

# Compute a short hash of the server script
SERVER_HASH=$(md5 -q "$SERVER" 2>/dev/null \
  || md5sum "$SERVER" 2>/dev/null | cut -c1-8 \
  || cksum "$SERVER" | awk '{print $1}' \
  || echo "")

if [ -z "$SERVER_HASH" ]; then
  exit 0
fi

STORED_HASH=$(cat "$STAMP_FILE" 2>/dev/null || echo "")

if [ "$SERVER_HASH" != "$STORED_HASH" ]; then
  # Script changed (or first run) — pre-warm uv dep resolution silently
  # This avoids the first-invocation delay when the MCP server starts
  uv run --no-project "$SERVER" --help >/dev/null 2>&1 || true
  echo "$SERVER_HASH" > "$STAMP_FILE"
fi

exit 0
