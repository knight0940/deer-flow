"""
GAIA benchmark using DeerFlow agent.

GAIA (General AI Assistant) is a benchmark that tests agentic capabilities:
multi-step reasoning, tool use (web browsing, code execution), and long-horizon planning.
It contains ~466 questions across 3 difficulty levels.

Uses DeerFlowClient with full tool access (bash, web_search, etc.) for each task.
Evaluation uses GAIA's official normalization and exact-match scoring.

Dataset: gaia-benchmark/GAIA on HuggingFace
  - Fields: task_id, Question, Level, Final answer, file_name, file_path
  - Levels: 1 (easy), 2 (medium), 3 (hard)
  - Scoring: exact match after normalization (numbers, strings, lists)

Features:
- Checkpoint/resume: saves progress after every task.
- Graceful interrupt: Ctrl+C stops after the current task completes.
- Run numbering: each run gets a unique results directory.
- Mode isolation: thinking/non-thinking produce separate output files.

Usage:
    uv run python run.py --model kimi-k2.5
    uv run python run.py --model kimi-k2.5 --level 1
    uv run python run.py --resume
"""

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
import uuid

from deerflow.client import DeerFlowClient
from deerflow.config import get_app_config
from deerflow.config.memory_config import get_memory_config, set_memory_config
from deerflow.config.paths import get_paths

MODEL_NAME = "kimi-k2.5"

DATASET_NAME = "gaia-benchmark/GAIA"

# Maximum agent recursion
MAX_RECURSION = 80

# Global flags for graceful interrupt and quota exhaustion
_interrupted = False
_quota_exhausted = False


def _handle_signal(signum, frame):
    global _interrupted
    print("\n\nInterrupt received! Finishing current task and saving...")
    _interrupted = True


signal.signal(signal.SIGINT, _handle_signal)


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
# Checkpoint helpers
# ---------------------------------------------------------------------------


def checkpoint_path(output_file: str) -> str:
    return output_file.replace(".jsonl", "_checkpoint.json")


def save_checkpoint(output_file: str, results: list, stats: dict):
    cp_file = checkpoint_path(output_file)
    data = {"results": results, "stats": stats}
    tmp = cp_file + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, cp_file)


def load_checkpoint(output_file: str) -> tuple[list, dict] | None:
    cp_file = checkpoint_path(output_file)
    if not os.path.exists(cp_file):
        return None
    with open(cp_file) as f:
        data = json.load(f)
    return data["results"], data["stats"]


# ---------------------------------------------------------------------------
# GAIA scoring (from official leaderboard scorer)
# ---------------------------------------------------------------------------


def normalize_answer(text: str) -> str:
    """Normalize answer for comparison."""
    if not text:
        return ""
    text = text.strip()
    # Remove common prefixes
    for prefix in ["The answer is ", "Answer: ", "answer: "]:
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip()
    return text


def normalize_number(text: str) -> float | None:
    """Try to parse text as a number, stripping $, %, commas."""
    text = text.strip().replace("$", "").replace("%", "").replace(",", "")
    try:
        return float(text)
    except ValueError:
        return None


def normalize_string(text: str) -> str:
    """Normalize string: lowercase, strip whitespace and punctuation."""
    text = text.strip().lower()
    # Remove trailing punctuation
    text = text.rstrip(".!?;:")
    return text


def score_answer(prediction: str, ground_truth: str) -> bool:
    """Score a GAIA answer using official normalization rules."""
    pred = normalize_answer(prediction)
    truth = normalize_answer(ground_truth)

    if not pred or not truth:
        return pred == truth

    # Try number comparison
    pred_num = normalize_number(pred)
    truth_num = normalize_number(truth)
    if pred_num is not None and truth_num is not None:
        return abs(pred_num - truth_num) < 1e-6

    # Try list comparison (comma or semicolon separated)
    if "," in truth or ";" in truth:
        sep = "," if "," in truth else ";"
        pred_items = [normalize_string(x) for x in pred.split(sep)]
        truth_items = [normalize_string(x) for x in truth.split(sep)]
        return sorted(pred_items) == sorted(truth_items)

    # String comparison
    return normalize_string(pred) == normalize_string(truth)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_dataset(level: int | None = None) -> list[dict]:
    """Load GAIA dataset from HuggingFace.

    level: 1, 2, or 3 for specific level. None for all.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: 'datasets' package required. pip install datasets")
        sys.exit(1)

    print(f"Loading dataset: {DATASET_NAME}...")
    ds = load_dataset(DATASET_NAME, "2023_all", split="validation")
    items = list(ds)

    if level is not None:
        items = [item for item in items if item.get("Level") == level]

    print(f"Loaded {len(items)} tasks" + (f" (Level {level})" if level else ""))
    return items


# ---------------------------------------------------------------------------
# Agent execution
# ---------------------------------------------------------------------------


AGENT_PROMPT_TEMPLATE = """\
You are a helpful AI assistant. Answer the following question.

{question}

{file_instruction}

Provide your final answer clearly at the end, prefixed with "FINAL ANSWER: ".
Be precise - the answer will be compared exactly. If the answer is a number, provide just the number.
If the answer is a list, provide comma-separated values.
"""

FILE_INSTRUCTION = """
An attached file is available at /mnt/user-data/workspace/{filename}.
Read it before answering the question if relevant.
"""


def setup_workspace(thread_id: str, task: dict) -> str | None:
    """Setup workspace and download attached file if any."""
    paths = get_paths()
    paths.ensure_thread_dirs(thread_id)
    workspace = str(paths.sandbox_work_dir(thread_id))

    file_name = task.get("file_name", "")
    file_path = task.get("file_path", "")

    if file_name and file_path:
        # Download attached file to workspace
        try:
            result = subprocess.run(
                ["curl", "-sL", "-o", os.path.join(workspace, file_name), file_path],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                print(f"[file download failed]", end=" ", flush=True)
        except Exception:
            pass

    return workspace


def extract_final_answer(response: str) -> str:
    """Extract the final answer from agent response."""
    # Look for "FINAL ANSWER: ..." pattern
    match = re.search(r"FINAL ANSWER:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Fallback: take the last non-empty line
    lines = [l.strip() for l in response.strip().split("\n") if l.strip()]
    if lines:
        return lines[-1]

    return response.strip()


def run_agent(
    client: DeerFlowClient,
    task: dict,
    thread_id: str,
    max_recursion: int = MAX_RECURSION,
) -> tuple[str, int, int, str | None]:
    """Run DeerFlow agent on a GAIA task.

    Returns (predicted_answer, input_tokens, output_tokens, error).
    """
    question = task["Question"]
    file_name = task.get("file_name", "")

    # Setup workspace
    try:
        setup_workspace(thread_id, task)
    except Exception as e:
        return "", 0, 0, f"workspace setup failed: {e}"

    # Build prompt
    file_instruction = FILE_INSTRUCTION.format(filename=file_name) if file_name else ""
    prompt = AGENT_PROMPT_TEMPLATE.format(
        question=question,
        file_instruction=file_instruction,
    )

    # Run agent
    total_input = 0
    total_output = 0
    seen_usage_ids: set[str] = set()
    response_text = ""

    try:
        for event in client.stream(
            prompt,
            thread_id=thread_id,
            recursion_limit=max_recursion,
        ):
            if event.type == "end":
                usage = event.data.get("usage", {})
                total_input = usage.get("input_tokens", total_input)
                total_output = usage.get("output_tokens", total_output)
            elif event.type == "messages-tuple" and event.data.get("type") == "ai":
                msg_id = event.data.get("id", "")
                meta = event.data.get("usage_metadata")
                if meta and msg_id and msg_id not in seen_usage_ids:
                    seen_usage_ids.add(msg_id)
                    total_input += meta.get("input_tokens", 0)
                    total_output += meta.get("output_tokens", 0)
                content = event.data.get("content", "")
                if content:
                    response_text += content
    except Exception as e:
        answer = extract_final_answer(response_text)
        return answer, total_input, total_output, f"agent error: {e}"

    answer = extract_final_answer(response_text)
    return answer, total_input, total_output, None


def cleanup_workspace(thread_id: str):
    paths = get_paths()
    paths.delete_thread_dir(thread_id)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="GAIA benchmark via DeerFlow agent")
    parser.add_argument("--model", default=MODEL_NAME, help="Model name in config.yaml")
    parser.add_argument("--level", type=int, default=None, choices=[1, 2, 3], help="Difficulty level (1-3)")
    parser.add_argument("--output", default=None, help="Output JSONL file")
    parser.add_argument("--thinking", action="store_true", help="Enable thinking mode")
    parser.add_argument("--start", type=int, default=0, help="Start index")
    parser.add_argument("--end", type=int, default=-1, help="End index (-1 for all)")
    parser.add_argument("--max_recursion", type=int, default=MAX_RECURSION, help="Max agent recursion limit")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between tasks (seconds)")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--fresh", action="store_true", help="Ignore checkpoint, start over")
    parser.add_argument("--cleanup", action="store_true", help="Delete thread workspaces after testing")
    args = parser.parse_args()

    # Output file with mode encoding
    mode = "thinking" if args.thinking else "no_thinking"
    level_str = f"_level{args.level}" if args.level else "_all"
    run_dir = get_run_dir(args.model)
    output_file = args.output or os.path.join(
        run_dir, f"gaia{level_str}_{args.model}_{mode}_results.jsonl"
    )
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

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

    # Load dataset
    items = load_dataset(args.level)
    if args.end == -1:
        args.end = len(items)
    items = items[args.start : args.end]

    # Initialize state
    results = []
    completed_ids: set[str] = set()
    stats = {
        "model": args.model,
        "provider": model_config.use,
        "thinking": args.thinking,
        "level": args.level,
        "n_tasks": len(items),
        "correct": 0,
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
            saved_results, _saved_stats = cp
            results = saved_results
            completed_ids = {r["task_id"] for r in results}
            stats["correct"] = sum(1 for r in results if r.get("correct"))
            stats["total_input_tokens"] = sum(r.get("input_tokens", 0) for r in results)
            stats["total_output_tokens"] = sum(r.get("output_tokens", 0) for r in results)
            print(f"Resumed: {len(results)} completed, {len(items) - len(completed_ids)} remaining")
            print()
        else:
            print("No checkpoint found, starting fresh")
            print()

    print(f"Model:          {args.model}")
    print(f"Provider:       {model_config.use}")
    print(f"Thinking:       {args.thinking}")
    print(f"Level:          {args.level or 'all'}")
    print(f"Tasks:          {len(items)} ({args.start}-{args.start + len(items) - 1})")
    print(f"Completed:      {len(completed_ids)}")
    print(f"Max recursion:  {args.max_recursion}")
    print(f"Output:         {output_file}")
    print()

    # Create DeerFlow client
    client = DeerFlowClient(
        model_name=args.model,
        thinking_enabled=args.thinking,
        subagent_enabled=False,
        plan_mode=False,
    )

    start_time = time.time()

    global _interrupted, _quota_exhausted
    for i, item in enumerate(items):
        if _interrupted or _quota_exhausted:
            break

        task_id = item.get("task_id", f"task_{i}")

        if task_id in completed_ids:
            continue

        level = item.get("Level", "?")
        print(f"[{len(completed_ids)+1}/{len(items)}] L{level} {task_id[:20]} ...", end=" ", flush=True)

        thread_id = f"gaia-{task_id[:20]}-{uuid.uuid4().hex[:8]}"

        try:
            prediction, in_tok, out_tok, error = run_agent(
                client, item, thread_id, max_recursion=args.max_recursion,
            )

            if error and is_quota_error(error):
                print(f"\n\nAPI QUOTA EXHAUSTED: {error}")
                print("Stopping benchmark. Use --resume to continue after topping up.")
                _quota_exhausted = True
                break

            ground_truth = item.get("Final answer", "")
            is_correct = score_answer(prediction, ground_truth) if not error else False

            mark = "PASS" if is_correct else "FAIL"
            err_str = f" err={error[:50]}" if error else ""
            print(f"{mark} (pred={prediction[:30]}, ans={ground_truth[:30]}, in={in_tok}, out={out_tok}){err_str}")

            results.append({
                "task_id": task_id,
                "level": level,
                "question": item.get("Question", "")[:200],
                "ground_truth": ground_truth,
                "prediction": prediction,
                "correct": is_correct,
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "error": error,
            })

            if is_correct:
                stats["correct"] += 1
            if error:
                stats["errors"] += 1

            completed_ids.add(task_id)
            stats["total_input_tokens"] += in_tok
            stats["total_output_tokens"] += out_tok
            stats["total_tokens"] = stats["total_input_tokens"] + stats["total_output_tokens"]

        except Exception as e:
            if is_quota_error(e):
                print(f"\n\nAPI QUOTA EXHAUSTED: {e}")
                print("Stopping benchmark. Use --resume to continue after topping up.")
                _quota_exhausted = True
            else:
                print(f"FAILED: {e}")
                stats["errors"] += 1
                results.append({
                    "task_id": task_id,
                    "level": item.get("Level", "?"),
                    "question": item.get("Question", "")[:200],
                    "ground_truth": item.get("Final answer", ""),
                    "prediction": "",
                    "correct": False,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "error": str(e),
                })
                completed_ids.add(task_id)

        # Cleanup
        if args.cleanup:
            try:
                cleanup_workspace(thread_id)
            except Exception:
                pass

        save_checkpoint(output_file, results, stats)

        if args.delay > 0 and not _interrupted:
            time.sleep(args.delay)

    elapsed = time.time() - start_time
    stats["elapsed_seconds"] = round(elapsed, 1)

    # Write final JSONL
    with open(output_file, "w") as f:
        for r in results:
            f.write(json.dumps({
                "task_id": r["task_id"],
                "level": r["level"],
                "ground_truth": r["ground_truth"],
                "prediction": r["prediction"],
                "correct": r["correct"],
            }) + "\n")

    # Save stats
    stats_file = output_file.replace(".jsonl", "_stats.json")
    with open(stats_file, "w") as f:
        json.dump({"stats": stats, "results": results}, f, indent=2)

    # Print summary
    n_completed = len([r for r in results if not r.get("error")])
    accuracy = stats["correct"] / max(n_completed, 1)
    print(f"\n{'=' * 60}")
    print(f"GAIA {'(interrupted) ' if _interrupted else ''}Complete")
    print(f"{'=' * 60}")
    print(f"Model:           {args.model}")
    print(f"Level:           {args.level or 'all'}")
    print(f"Thinking:        {args.thinking}")
    print(f"Total:           {n_completed}")
    print(f"Correct:         {stats['correct']}")
    print(f"Accuracy:        {accuracy:.4f}")
    print(f"Errors:          {stats['errors']}")
    print(f"Input tokens:    {stats['total_input_tokens']:,}")
    print(f"Output tokens:   {stats['total_output_tokens']:,}")
    print(f"Elapsed:         {elapsed:.1f}s")
    print(f"Output:          {output_file}")
    print(f"Stats:           {stats_file}")

    # Per-level breakdown
    levels = {}
    for r in results:
        lvl = r.get("level", "?")
        if lvl not in levels:
            levels[lvl] = {"total": 0, "correct": 0}
        if not r.get("error"):
            levels[lvl]["total"] += 1
        if r.get("correct"):
            levels[lvl]["correct"] += 1

    if levels:
        print()
        print("Per level:")
        for lvl in sorted(levels.keys()):
            l = levels[lvl]
            acc = l["correct"] / max(l["total"], 1)
            print(f"  Level {lvl}: {l['correct']}/{l['total']} = {acc:.4f}")

    # Clean up checkpoint
    if not _interrupted and len(completed_ids) >= len(items):
        cp_file = checkpoint_path(output_file)
        if os.path.exists(cp_file):
            os.remove(cp_file)
            print(f"\nCheckpoint cleaned up: {cp_file}")


if __name__ == "__main__":
    main()
