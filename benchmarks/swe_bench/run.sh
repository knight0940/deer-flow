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

# Default dataset
DATASET="${1:-lite}"
shift || true

echo "=== SWE-bench Benchmark (DeerFlow Agent) ==="
echo ""

# Step 1: Install dependencies if needed
if ! uv run --directory "$SCRIPT_DIR/../../backend" python -c "import datasets" 2>/dev/null; then
    echo "[0/3] Installing 'datasets' package..."
    uv add --directory "$SCRIPT_DIR/../../backend" datasets
fi

# Step 2: Generate patches using DeerFlow agent
echo "[1/3] Running DeerFlow agent on SWE-bench tasks..."
uv run --directory "$SCRIPT_DIR/../../backend" python "$SCRIPT_DIR/run.py" --dataset "$DATASET" --resume --cleanup "$@"

# Step 3: Evaluate with swebench harness (requires Docker)
echo ""
echo "[2/3] Evaluating patches with swebench harness (requires Docker)..."
echo "Note: If Docker is not available, run this step manually:"
echo "  docker pull ghcr.io/opendeerflow/swebench-runner:latest"
echo "  uv run python evaluate.py --dataset $DATASET"
uv run --directory "$SCRIPT_DIR/../../backend" python "$SCRIPT_DIR/evaluate.py" --dataset "$DATASET" "$@"

# Step 4: Report
echo ""
echo "[3/3] Done! Check the CSV report for results."
echo ""
echo "=== Complete ==="
