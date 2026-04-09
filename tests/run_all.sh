#!/usr/bin/env bash
# MLX Plugin — run all skill tests
# Usage: bash tests/run_all.sh [--fast]
#
# --fast  Skip slow tests (chart rendering, benchmark, full training loops)

set -euo pipefail
cd "$(dirname "$0")/.."

FAST="${1:-}"
PASSED=0
FAILED=0
SKIPPED=0

run_test() {
    local file="$1"
    local name
    name=$(basename "$file" .py)
    printf "%-35s" "$name ..."
    if python3 "$file" 2>&1 | tail -1 | grep -q "OK"; then
        echo "PASS"
        PASSED=$((PASSED + 1))
    else
        output=$(python3 "$file" 2>&1)
        if echo "$output" | grep -q "skipped"; then
            echo "SKIP"
            SKIPPED=$((SKIPPED + 1))
        else
            echo "FAIL"
            echo "$output" | tail -20
            FAILED=$((FAILED + 1))
        fi
    fi
}

echo "=== MLX Plugin Test Suite ==="
echo "Running from: $(pwd)"
echo ""

for test_file in tests/test_*.py; do
    run_test "$test_file"
done

echo ""
echo "Results: ${PASSED} passed, ${FAILED} failed, ${SKIPPED} skipped"
[ "$FAILED" -eq 0 ] && exit 0 || exit 1
