"""
MMLU benchmark using DeepEval + DeerFlow agent.

Uses DeepEval's MMLU benchmark for dataset loading and scoring,
with a custom model adapter that routes through DeerFlow's agent pipeline.

Architecture:
    - DeepEval handles: dataset loading, few-shot prompting, exact-match scoring
    - DeerFlow agent handles: model invocation, thinking mode, tool availability

Features:
- Run numbering: each run gets a unique results directory.
- Mode isolation: thinking/non-thinking produce separate output files.
- Memory disabled during benchmark to prevent cross-sample contamination.

Usage:
    uv run python run.py --model kimi-k2.5
    uv run python run.py --model kimi-k2.5 --n_shots 5
"""

import argparse
import json
import os
import sys
import time

from deerflow.client import DeerFlowClient
from deerflow.config import get_app_config
from deerflow.config.memory_config import get_memory_config, set_memory_config

MODEL_NAME = "kimi-k2.5"


# ---------------------------------------------------------------------------
# Quota exhaustion detection
# ---------------------------------------------------------------------------


class QuotaExhaustedError(Exception):
    """Raised when API quota is exhausted."""
    pass


def is_quota_error(error: str | Exception) -> bool:
    msg = str(error).lower()
    patterns = [
        "429", "rate limit", "rate_limit", "quota", "insufficient",
        "billing", "capacity", "overloaded", "too many requests",
        "resource_exhausted", "tokens exhausted", "account limit",
        "spending limit",
    ]
    return any(p in msg for p in patterns)


# ---------------------------------------------------------------------------
# Run directory management
# ---------------------------------------------------------------------------


def get_run_dir(model: str) -> str:
    results_base = os.path.join(os.path.dirname(__file__), "results", model)
    os.makedirs(results_base, exist_ok=True)
    existing = [d for d in os.listdir(results_base) if d.startswith("run_")]
    next_num = max((int(d.split("_")[1]) for d in existing), default=0) + 1
    run_dir = os.path.join(results_base, f"run_{next_num}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


# ---------------------------------------------------------------------------
# DeepEval custom model adapter wrapping DeerFlowClient
# ---------------------------------------------------------------------------


class DeerFlowAdapter:
    """Adapter that wraps DeerFlowClient for DeepEval's model interface."""

    def __init__(self, model_name: str, thinking_enabled: bool = False):
        self.model_name = model_name
        self.thinking_enabled = thinking_enabled
        self._client: DeerFlowClient | None = None
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def _get_client(self) -> DeerFlowClient:
        if self._client is None:
            self._client = DeerFlowClient(
                model_name=self.model_name,
                thinking_enabled=self.thinking_enabled,
                subagent_enabled=False,
                plan_mode=False,
            )
        return self._client

    def generate(self, prompt: str) -> str:
        """Synchronous generate - required by DeepEval's model interface."""
        import uuid

        client = self._get_client()
        thread_id = f"mmlu-{uuid.uuid4().hex[:12]}"

        total_input = 0
        total_output = 0
        seen_ids: set[str] = set()
        response_text = ""

        try:
            for event in client.stream(
                prompt,
                thread_id=thread_id,
                recursion_limit=10,
            ):
                if event.type == "end":
                    usage = event.data.get("usage", {})
                    total_input = usage.get("input_tokens", total_input)
                    total_output = usage.get("output_tokens", total_output)
                elif event.type == "messages-tuple" and event.data.get("type") == "ai":
                    msg_id = event.data.get("id", "")
                    meta = event.data.get("usage_metadata")
                    if meta and msg_id and msg_id not in seen_ids:
                        seen_ids.add(msg_id)
                        total_input += meta.get("input_tokens", 0)
                        total_output += meta.get("output_tokens", 0)
                    content = event.data.get("content", "")
                    if content:
                        response_text += content
        except Exception as e:
            if is_quota_error(e):
                print(f"\n\nAPI QUOTA EXHAUSTED: {e}")
                raise QuotaExhaustedError(str(e)) from e
            print(f"  [agent error: {e}]", end="", flush=True)
            return ""

        # Cleanup thread
        try:
            from deerflow.config.paths import get_paths
            get_paths().delete_thread_dir(thread_id)
        except Exception:
            pass

        self.total_input_tokens += total_input
        self.total_output_tokens += total_output
        return response_text

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return f"DeerFlow({self.model_name})"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="MMLU benchmark via DeepEval + DeerFlow agent")
    parser.add_argument("--model", default=MODEL_NAME, help="Model name in config.yaml")
    parser.add_argument("--n_shots", type=int, default=5, help="Number of few-shot examples")
    parser.add_argument("--thinking", action="store_true", help="Enable thinking mode")
    args = parser.parse_args()

    # Verify model config
    config = get_app_config()
    model_config = config.get_model_config(args.model)
    if model_config is None:
        print(f"Error: Model '{args.model}' not found in config.yaml")
        print(f"Available: {[m.name for m in config.models]}")
        sys.exit(1)

    # Disable memory
    saved_memory = get_memory_config()
    set_memory_config(saved_memory.model_copy(update={"enabled": False, "injection_enabled": False}))

    # Output directory
    mode = "thinking" if args.thinking else "no_thinking"
    run_dir = get_run_dir(args.model)

    print(f"=== MMLU Benchmark (DeepEval + DeerFlow Agent) ===")
    print()
    print(f"Model:          {args.model}")
    print(f"Thinking:       {args.thinking}")
    print(f"Few-shot:       {args.n_shots}")
    print(f"Output:         {run_dir}")
    print()

    # Install deepeval if needed
    try:
        from deepeval.benchmarks import MMLU
    except ImportError:
        print("Installing deepeval...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "deepeval"])
        from deepeval.benchmarks import MMLU

    # Create adapter
    adapter = DeerFlowAdapter(model_name=args.model, thinking_enabled=args.thinking)

    # Run MMLU benchmark
    print("Running MMLU benchmark...")
    start_time = time.time()

    benchmark = MMLU(n_shots=args.n_shots)

    try:
        benchmark.evaluate(model=adapter)
    except QuotaExhaustedError as e:
        print(f"\n\nAPI QUOTA EXHAUSTED. Benchmark stopped.")
        print("Top up your API quota and re-run to continue.")
        sys.exit(1)

    elapsed = time.time() - start_time

    # Results
    print()
    print(f"{'=' * 60}")
    print(f"MMLU Results")
    print(f"{'=' * 60}")
    print(f"Model:           {args.model}")
    print(f"Thinking:        {args.thinking}")
    print(f"Few-shot:        {args.n_shots}")
    print(f"Overall Score:   {benchmark.overall_score:.4f}")
    print(f"Input tokens:    {adapter.total_input_tokens:,}")
    print(f"Output tokens:   {adapter.total_output_tokens:,}")
    print(f"Elapsed:         {elapsed:.1f}s")
    print()

    # Save results
    results = {
        "model": args.model,
        "thinking": args.thinking,
        "n_shots": args.n_shots,
        "overall_score": benchmark.overall_score,
        "input_tokens": adapter.total_input_tokens,
        "output_tokens": adapter.total_output_tokens,
        "elapsed_seconds": round(elapsed, 1),
    }

    results_file = os.path.join(run_dir, f"{args.model}_{mode}_mmlu_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results: {results_file}")

    # Save predictions if available
    if hasattr(benchmark, "predictions"):
        predictions = benchmark.predictions
        if hasattr(predictions, "to_json"):
            pred_file = os.path.join(run_dir, f"{args.model}_{mode}_mmlu_predictions.json")
            predictions.to_json(pred_file, orient="records", indent=2)
            print(f"Predictions: {pred_file}")
        elif hasattr(predictions, "to_csv"):
            pred_file = os.path.join(run_dir, f"{args.model}_{mode}_mmlu_predictions.csv")
            predictions.to_csv(pred_file, index=False)
            print(f"Predictions: {pred_file}")


if __name__ == "__main__":
    main()
