#!/usr/bin/env bash
set -euo pipefail

# Reads the newest row in results.tsv and outputs a one-line summary.
# Called async when results.tsv changes.

TSV="${PWD}/results.tsv"

if [ ! -f "$TSV" ]; then
  exit 0
fi

# Get last data row (skip header)
LAST=$(tail -n 1 "$TSV")
if [ -z "$LAST" ]; then
  exit 0
fi

ID=$(echo "$LAST" | awk -F'\t' '{print $1}')
VAL=$(echo "$LAST" | awk -F'\t' '{print $3}')
STATUS=$(echo "$LAST" | awk -F'\t' '{print $6}')

# Check if this is the best KEEP score
BEST=$(awk -F'\t' 'NR>1 && $6=="KEEP" {if($3>b) b=$3} END{print b+0}' "$TSV")

SUFFIX=""
if [ "$STATUS" = "KEEP" ] && [ "$(echo "$VAL >= $BEST" | awk '{print ($1>=$3)}')" = "1" ]; then
  SUFFIX=" [NEW BEST]"
fi

echo "New experiment: id=${ID} val=${VAL} status=${STATUS}${SUFFIX}"
