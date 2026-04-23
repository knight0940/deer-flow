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

echo "=== HumanEval Benchmark (DeerFlow Agent) ==="
echo ""

# Step 1: Generate completions via DeerFlow agent
echo "[1/2] Running DeerFlow agent on HumanEval problems..."
uv run --directory "$SCRIPT_DIR/../../backend" python "$SCRIPT_DIR/generate.py" --resume --cleanup "$@"

# Step 2: Evaluate and generate CSV report
echo ""
echo "[2/2] Evaluating and generating CSV..."
uv run --directory "$SCRIPT_DIR/../../backend" python "$SCRIPT_DIR/evaluate.py" "$@"

echo ""
echo "=== Done ==="
