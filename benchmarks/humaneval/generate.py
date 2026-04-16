"""
HumanEval code generation benchmark using DeerFlow harness.

Uses DeerFlow's create_chat_model() to load model config from config.yaml,
properly leveraging the model provider, thinking mode, token counting, etc.

Features:
- Checkpoint/resume: saves progress after every API call.
  If interrupted, re-run with the same args to resume automatically.
- Graceful interrupt: Ctrl+C stops after the current problem completes and saves.
- Retry: transient API errors are retried with exponential backoff.
- Resume retries: previously failed problems are retried on --resume.

Usage:
    uv run python generate.py
    uv run python generate.py --n_samples 10        # for pass@10
    uv run python generate.py --thinking             # enable thinking mode
    uv run python generate.py --resume               # resume from checkpoint (retries failures)
    uv run python generate.py --fresh                # ignore checkpoint, start over
"""

import argparse
import json
import os
import signal
import sys
import time

from langchain.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from deerflow.config import get_app_config
from deerflow.models.factory import create_chat_model
from human_eval.data import write_jsonl, read_problems

MODEL_NAME = "kimi-for-coding"

SYSTEM_PROMPT = (
    "Continue the following Python code. "
    "Output ONLY the code continuation, no explanations, no markdown fences, no comments about your approach."
)

# Global flag for graceful interrupt
_interrupted = False


def _handle_signal(signum, frame):
    global _interrupted
    print("\n\nInterrupt received! Finishing current problem and saving...")
    _interrupted = True


signal.signal(signal.SIGINT, _handle_signal)


def checkpoint_path(output_file: str) -> str:
    return output_file.replace(".jsonl", "_checkpoint.json")


def save_checkpoint(output_file: str, samples: list, stats: dict):
    cp_file = checkpoint_path(output_file)
    data = {"samples": samples, "stats": stats}
    tmp = cp_file + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, cp_file)  # atomic write


def load_checkpoint(output_file: str) -> tuple[list, dict] | None:
    cp_file = checkpoint_path(output_file)
    if not os.path.exists(cp_file):
        return None
    with open(cp_file) as f:
        data = json.load(f)
    return data["samples"], data["stats"]


def generate_completion(
    model: BaseChatModel, prompt: str, max_retries: int = 3
) -> tuple[str, int, int]:
    """Generate a code completion with retry on transient errors."""
    for attempt in range(max_retries):
        try:
            response = model.invoke(
                [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=prompt),
                ],
            )

            content = response.content or ""
            input_tokens = 0
            output_tokens = 0

            if hasattr(response, "usage_metadata") and response.usage_metadata:
                meta = response.usage_metadata
                input_tokens = meta.get("input_tokens", 0)
                output_tokens = meta.get("output_tokens", 0)
            elif hasattr(response, "response_metadata") and response.response_metadata:
                meta = response.response_metadata
                usage = meta.get("token_usage", meta.get("usage", {}))
                input_tokens = usage.get("prompt_tokens", 0)
                output_tokens = usage.get("completion_tokens", 0)

            return content, input_tokens, output_tokens

        except Exception as e:
            is_last = attempt == max_retries - 1
            if is_last:
                raise
            wait = 2 ** attempt * 2  # 2s, 4s, 8s
            print(f"RETRY {attempt+1}/{max_retries} ({e}), waiting {wait}s...", end=" ", flush=True)
            time.sleep(wait)


def extract_code(completion: str) -> str:
    # Extract code from model completion, stripping markdown fences.
    # HumanEval prompts end with the function docstring closing quotes.
    # The completion gets directly appended, so it must start with a newline
    # and proper indentation (4 spaces) to be valid function body code.
    text = completion.strip()
    if text.startswith("```python"):
        text = text[len("```python"):]
    elif text.startswith("```"):
        text = text[len("```"):]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    # Ensure the completion starts with newline + 4-space indent for function body
    if text and not text.startswith("\n"):
        text = "\n    " + text
    return text


def main():
    parser = argparse.ArgumentParser(description="HumanEval benchmark via DeerFlow harness")
    parser.add_argument("--model", default=MODEL_NAME, help="Model name in config.yaml")
    parser.add_argument("--n_samples", type=int, default=1, help="Samples per problem (for pass@k)")
    parser.add_argument("--output", default=None, help="Output JSONL file")
    parser.add_argument("--thinking", action="store_true", help="Enable thinking mode")
    parser.add_argument("--start", type=int, default=0, help="Start index")
    parser.add_argument("--end", type=int, default=-1, help="End index (-1 for all)")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between calls (seconds)")
    parser.add_argument("--max_retries", type=int, default=3, help="Max retries per API call")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint (retries failures)")
    parser.add_argument("--fresh", action="store_true", help="Ignore checkpoint, start over")
    args = parser.parse_args()

    output_file = args.output or f"{args.model}_samples.jsonl"

    # Load DeerFlow config
    config = get_app_config()
    model_config = config.get_model_config(args.model)
    if model_config is None:
        print(f"Error: Model '{args.model}' not found in config.yaml")
        print(f"Available: {[m.name for m in config.models]}")
        sys.exit(1)

    # Load HumanEval problems
    problems = read_problems()
    problem_list = list(problems.values())
    if args.end == -1:
        args.end = len(problem_list)
    problem_list = problem_list[args.start : args.end]

    # Initialize state
    samples = []
    completed_task_ids: set[str] = set()
    stats = {
        "model": args.model,
        "provider": model_config.use,
        "thinking": args.thinking,
        "n_problems": len(problem_list),
        "n_samples_per_problem": args.n_samples,
        "total_samples": 0,
        "errors": 0,
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_tokens": 0,
        "elapsed_seconds": 0,
    }

    # Resume from checkpoint
    if args.resume and not args.fresh:
        cp = load_checkpoint(output_file)
        if cp:
            saved_samples, saved_stats = cp
            samples = saved_samples
            completed_task_ids = {s["task_id"] for s in samples}
            stats["total_input_tokens"] = sum(s.get("input_tokens", 0) for s in samples)
            stats["total_output_tokens"] = sum(s.get("output_tokens", 0) for s in samples)

            print(f"Resumed from checkpoint: {len(samples)} completed, {len(problem_list) - len(completed_task_ids)} remaining")
            print()
        else:
            print("No checkpoint found, starting fresh")
            print()

    print(f"Model:          {args.model}")
    print(f"Provider:       {model_config.use}")
    print(f"Thinking:       {args.thinking}")
    print(f"Problems:       {len(problem_list)} ({args.start}-{args.start + len(problem_list) - 1})")
    print(f"Completed:      {len(completed_task_ids)}")
    print(f"Samples/eval:   {args.n_samples}")
    print(f"Max retries:    {args.max_retries}")
    print(f"Output:         {output_file}")
    print()

    # Create model via DeerFlow factory
    model = create_chat_model(name=args.model, thinking_enabled=args.thinking)

    start_time = time.time()

    for i, problem in enumerate(problem_list):
        if _interrupted:
            break

        task_id = problem["task_id"]
        prompt = problem["prompt"]

        # Skip already completed
        if task_id in completed_task_ids:
            continue

        print(f"[{len(completed_task_ids)+1}/{len(problem_list)}] {task_id} ...", end=" ", flush=True)

        problem_ok = False
        problem_in_tok = 0
        problem_out_tok = 0

        for sample_idx in range(args.n_samples):
            if _interrupted:
                break
            try:
                completion, in_tok, out_tok = generate_completion(
                    model, prompt, max_retries=args.max_retries
                )
                problem_in_tok += in_tok
                problem_out_tok += out_tok
                code = extract_code(completion)
                samples.append({
                    "task_id": task_id,
                    "completion": code,
                    "input_tokens": in_tok,
                    "output_tokens": out_tok,
                    "error": None,
                })
                print(f"OK (in={in_tok}, out={out_tok})")
                problem_ok = True
            except Exception as e:
                print(f"FAILED: {e}")
                stats["errors"] += 1
                break  # skip remaining samples for this problem

            if args.delay > 0 and not _interrupted:
                time.sleep(args.delay)

        # Only save checkpoint if this problem succeeded
        if problem_ok:
            completed_task_ids.add(task_id)
            stats["total_input_tokens"] += problem_in_tok
            stats["total_output_tokens"] += problem_out_tok
        stats["total_samples"] = len(samples)
        stats["total_tokens"] = stats["total_input_tokens"] + stats["total_output_tokens"]
        save_checkpoint(output_file, samples, stats)

    elapsed = time.time() - start_time
    stats["elapsed_seconds"] = round(elapsed, 1)
    stats["total_samples"] = len(samples)
    stats["total_tokens"] = stats["total_input_tokens"] + stats["total_output_tokens"]

    # Write final JSONL output (only task_id + completion for human-eval compatibility)
    eval_samples = [{"task_id": s["task_id"], "completion": s["completion"]} for s in samples]
    write_jsonl(output_file, eval_samples)

    print(f"\n{'='*50}")
    print(f"Generation complete" + (" (interrupted)" if _interrupted else ""))
    print(f"{'='*50}")
    print(f"Model:           {args.model}")
    print(f"Thinking:        {args.thinking}")
    print(f"Samples:         {len(samples)}")
    print(f"Passed (OK):     {len(completed_task_ids)}")
    print(f"Errors:          {stats['errors']}")
    print(f"Input tokens:    {stats['total_input_tokens']:,}")
    print(f"Output tokens:   {stats['total_output_tokens']:,}")
    print(f"Total tokens:    {stats['total_tokens']:,}")
    print(f"Elapsed time:    {elapsed:.1f}s")
    print(f"Avg time/sample: {elapsed/max(len(samples),1):.2f}s")
    print(f"Output:          {output_file}")
    print()

    # Save final stats (with full per-problem data)
    stats_file = output_file.replace(".jsonl", "_stats.json")
    with open(stats_file, "w") as f:
        json.dump({"stats": stats, "samples": samples}, f, indent=2)
    print(f"Stats:           {stats_file}")

    # Clean up checkpoint if completed all
    if not _interrupted and len(completed_task_ids) >= len(problem_list):
        cp_file = checkpoint_path(output_file)
        if os.path.exists(cp_file):
            os.remove(cp_file)
            print(f"Checkpoint cleaned up: {cp_file}")


if __name__ == "__main__":
    main()
