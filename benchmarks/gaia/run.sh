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

echo "=== GAIA Benchmark (DeerFlow Agent) ==="
echo ""

# Step 1: Run agent on GAIA tasks
echo "[1/2] Running DeerFlow agent on GAIA tasks..."
uv run --directory "$SCRIPT_DIR/../../backend" python "$SCRIPT_DIR/run.py" --resume --cleanup "$@"

# Step 2: Print report
echo ""
echo "[2/2] Done! Check the stats file for results."
echo ""
echo "=== Complete ==="
