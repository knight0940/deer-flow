#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Load env
if [ -f "$SCRIPT_DIR/../../.env" ]; then
    set -a
    source "$SCRIPT_DIR/../../.env"
    set +a
fi

echo "=== MMLU Benchmark (DeepEval + DeerFlow Agent) ==="
echo ""

uv run --directory "$SCRIPT_DIR/../../backend" python "$SCRIPT_DIR/run.py" "$@"
