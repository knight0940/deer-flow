"""
HumanEval code generation benchmark using DeerFlow agent.

Uses DeerFlowClient as the agent to solve HumanEval problems.
The agent has access to bash, read_file, write_file, str_replace tools
to implement functions and verify them with tests.

Architecture (per problem):
    1. Load HumanEval problem (prompt, test, entry_point)
    2. Create thread workspace with solution.py (stub) and test_solution.py
    3. Send task description to DeerFlow agent via DeerFlowClient.stream()
    4. Agent uses tools to implement the function and run tests
    5. Verify solution by running test_solution.py in subprocess
    6. Extract completion from solution.py

Features:
- Checkpoint/resume: saves progress after every problem.
- Graceful interrupt: Ctrl+C stops after the current problem completes.
- Run numbering: each run gets a unique results directory.
- Mode isolation: thinking/non-thinking produce separate output files.

Usage:
    uv run python generate.py --model kimi-k2.5 --thinking
    uv run python generate.py --model kimi-k2.5
    uv run python generate.py --resume
    uv run python generate.py --fresh
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import uuid

from deerflow.client import DeerFlowClient
from deerflow.config import get_app_config
from deerflow.config.memory_config import get_memory_config, set_memory_config
from deerflow.config.paths import get_paths
from human_eval.data import read_problems, write_jsonl

MODEL_NAME = "kimi-k2.5"

# Maximum LangGraph recursion limit (controls agent turns)
MAX_RECURSION = 50

# Prompt template for the agent
AGENT_PROMPT = """\
You need to implement a Python function. The function stub is in /mnt/user-data/workspace/solution.py.
The test file is at /mnt/user-data/workspace/test_solution.py.

## Instructions
1. Read /mnt/user-data/workspace/solution.py to see the function stub
2. Complete the function implementation by filling in the function body after the docstring
3. Run the test: cd /mnt/user-data/workspace && python test_solution.py
4. If the test fails, read the error, fix your implementation, and try again
5. Do NOT modify test_solution.py
6. Keep the original function signature, docstring, and imports

The function name is: {entry_point}

Use the available tools (bash, read_file, write_file, str_replace) to implement and test the function.
"""

# Global flags for graceful interrupt and quota exhaustion
_interrupted = False
_quota_exhausted = False


def _handle_signal(signum, frame):
    global _interrupted
    print("\n\nInterrupt received! Finishing current problem and saving...")
    _interrupted = True


signal.signal(signal.SIGINT, _handle_signal)


def is_quota_error(error: str | Exception) -> bool:
    """Check if an error indicates API quota/rate limit exhaustion."""
    msg = str(error).lower()
    patterns = [
        "429",
        "rate limit",
        "rate_limit",
        "quota",
        "insufficient",
        "billing",
        "capacity",
        "overloaded",
        "too many requests",
        "resource_exhausted",
        "tokens exhausted",
        "account limit",
        "spending limit",
    ]
    return any(p in msg for p in patterns)


# ---------------------------------------------------------------------------
# Run directory management (principle 4: no overwrite between runs)
# ---------------------------------------------------------------------------


def get_run_dir(model: str) -> str:
    """Get the next available run directory."""
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


def save_checkpoint(output_file: str, samples: list, stats: dict):
    cp_file = checkpoint_path(output_file)
    data = {"samples": samples, "stats": stats}
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
    return data["samples"], data["stats"]


# ---------------------------------------------------------------------------
# Workspace helpers
# ---------------------------------------------------------------------------


def setup_workspace(thread_id: str, problem: dict) -> str:
    """Create thread workspace with solution.py (stub) and test_solution.py."""
    paths = get_paths()
    paths.ensure_thread_dirs(thread_id)
    workspace = str(paths.sandbox_work_dir(thread_id))

    prompt_text = problem["prompt"]
    test_code = problem["test"]
    entry_point = problem["entry_point"]

    # Write solution.py with just the stub
    with open(os.path.join(workspace, "solution.py"), "w") as f:
        f.write(prompt_text)

    # Write test_solution.py
    test_content = (
        f"import sys\n"
        f"import os\n"
        f"sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))\n"
        f"from solution import {entry_point} as candidate\n\n"
        f"{test_code}\n\n"
        f"check(candidate)\n"
        f'print("ALL TESTS PASSED")\n'
    )
    with open(os.path.join(workspace, "test_solution.py"), "w") as f:
        f.write(test_content)

    return workspace


def run_test(workspace: str) -> tuple[bool, str]:
    """Run the test in the workspace and return (passed, output)."""
    try:
        result = subprocess.run(
            [sys.executable, "test_solution.py"],
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=10,
        )
        passed = "ALL TESTS PASSED" in result.stdout and result.returncode == 0
        output = result.stdout + result.stderr
        return passed, output
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    except Exception as e:
        return False, str(e)


def extract_completion(workspace: str, original_prompt: str) -> str:
    """Read solution.py and extract the completion (content after the original stub)."""
    solution_path = os.path.join(workspace, "solution.py")
    if not os.path.exists(solution_path):
        return ""
    with open(solution_path) as f:
        content = f.read()
    if not content or content == original_prompt:
        return ""
    if content.startswith(original_prompt):
        return content[len(original_prompt):]
    # Agent rewrote the entire file - return as-is
    return content


def cleanup_workspace(thread_id: str):
    """Delete thread workspace to free disk space."""
    paths = get_paths()
    paths.delete_thread_dir(thread_id)


# ---------------------------------------------------------------------------
# Agent execution
# ---------------------------------------------------------------------------


def run_agent(
    client: DeerFlowClient,
    problem: dict,
    thread_id: str,
    max_recursion: int = MAX_RECURSION,
) -> tuple[str, int, int, bool, str | None]:
    """Run DeerFlow agent on a HumanEval problem.

    Returns (completion, input_tokens, output_tokens, passed, error).
    """
    prompt_text = problem["prompt"]
    entry_point = problem["entry_point"]

    # Setup workspace
    try:
        workspace = setup_workspace(thread_id, problem)
    except Exception as e:
        return "", 0, 0, False, f"workspace setup failed: {e}"

    # Build agent prompt
    agent_prompt = AGENT_PROMPT.format(entry_point=entry_point)

    # Run agent with streaming to track usage
    total_input = 0
    total_output = 0
    seen_usage_ids: set[str] = set()

    try:
        for event in client.stream(
            agent_prompt,
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
    except Exception as e:
        completion = extract_completion(workspace, prompt_text)
        passed, _ = run_test(workspace)
        return completion, total_input, total_output, passed, f"agent error: {e}"

    # Extract solution
    completion = extract_completion(workspace, prompt_text)

    # Verify by running the test
    passed, _ = run_test(workspace)

    return completion, total_input, total_output, passed, None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="HumanEval benchmark via DeerFlow agent")
    parser.add_argument("--model", default=MODEL_NAME, help="Model name in config.yaml")
    parser.add_argument("--output", default=None, help="Output JSONL file")
    parser.add_argument("--thinking", action="store_true", help="Enable thinking mode")
    parser.add_argument("--start", type=int, default=0, help="Start index")
    parser.add_argument("--end", type=int, default=-1, help="End index (-1 for all)")
    parser.add_argument("--max_recursion", type=int, default=MAX_RECURSION, help="Max agent recursion limit")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between problems (seconds)")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--fresh", action="store_true", help="Ignore checkpoint, start over")
    parser.add_argument("--cleanup", action="store_true", help="Delete thread workspaces after testing")
    args = parser.parse_args()

    # Output file with mode encoding (principle 2: no conflict between modes)
    mode = "thinking" if args.thinking else "no_thinking"
    run_dir = get_run_dir(args.model)
    output_file = args.output or os.path.join(run_dir, f"{args.model}_{mode}_samples.jsonl")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Verify model config
    config = get_app_config()
    model_config = config.get_model_config(args.model)
    if model_config is None:
        print(f"Error: Model '{args.model}' not found in config.yaml")
        print(f"Available: {[m.name for m in config.models]}")
        sys.exit(1)

    # Disable memory to prevent cross-sample contamination
    saved_memory_config = get_memory_config()
    memory_config = saved_memory_config.model_copy(update={"enabled": False, "injection_enabled": False})
    set_memory_config(memory_config)

    # Load HumanEval problems
    problems = read_problems()
    problem_list = list(problems.values())
    if args.end == -1:
        args.end = len(problem_list)
    problem_list = problem_list[args.start : args.end]

    # Initialize state
    samples = []
    completed_ids: set[str] = set()
    stats = {
        "model": args.model,
        "provider": model_config.use,
        "thinking": args.thinking,
        "n_problems": len(problem_list),
        "passed": 0,
        "failed": 0,
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
            saved_samples, _saved_stats = cp
            samples = saved_samples
            completed_ids = {s["task_id"] for s in samples}
            stats["passed"] = sum(1 for s in samples if s.get("passed"))
            stats["failed"] = sum(1 for s in samples if not s.get("passed") and not s.get("error"))
            stats["total_input_tokens"] = sum(s.get("input_tokens", 0) for s in samples)
            stats["total_output_tokens"] = sum(s.get("output_tokens", 0) for s in samples)
            print(f"Resumed from checkpoint: {len(samples)} completed, {len(problem_list) - len(completed_ids)} remaining")
            print()
        else:
            print("No checkpoint found, starting fresh")
            print()

    print(f"Model:          {args.model}")
    print(f"Provider:       {model_config.use}")
    print(f"Thinking:       {args.thinking}")
    print(f"Problems:       {len(problem_list)} ({args.start}-{args.start + len(problem_list) - 1})")
    print(f"Completed:      {len(completed_ids)}")
    print(f"Max recursion:  {args.max_recursion}")
    print(f"Cleanup:        {args.cleanup}")
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
    for i, problem in enumerate(problem_list):
        if _interrupted or _quota_exhausted:
            break

        task_id = problem["task_id"]

        if task_id in completed_ids:
            continue

        print(f"[{len(completed_ids)+1}/{len(problem_list)}] {task_id} ...", end=" ", flush=True)

        thread_id = f"humaneval-{task_id.replace('/', '-')}-{uuid.uuid4().hex[:8]}"

        try:
            completion, in_tok, out_tok, passed, error = run_agent(
                client, problem, thread_id, max_recursion=args.max_recursion,
            )

            if error:
                if is_quota_error(error):
                    print(f"\n\nAPI QUOTA EXHAUSTED: {error}")
                    print("Stopping benchmark. Use --resume to continue after topping up.")
                    _quota_exhausted = True
                else:
                    print(f"ERROR: {error}", end=" ")
                    stats["errors"] += 1
            elif passed:
                print(f"PASS (in={in_tok}, out={out_tok})", end=" ")
                stats["passed"] += 1
            else:
                print(f"FAIL (in={in_tok}, out={out_tok})", end=" ")
                stats["failed"] += 1

            samples.append({
                "task_id": task_id,
                "completion": completion,
                "passed": passed,
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "error": error,
            })

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
                samples.append({
                    "task_id": task_id,
                    "completion": "",
                    "passed": False,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "error": str(e),
                })
                completed_ids.add(task_id)

        # Cleanup workspace
        if args.cleanup:
            try:
                cleanup_workspace(thread_id)
            except Exception:
                pass

        # Save checkpoint
        save_checkpoint(output_file, samples, stats)

        if args.delay > 0 and not _interrupted:
            time.sleep(args.delay)

    elapsed = time.time() - start_time
    stats["elapsed_seconds"] = round(elapsed, 1)

    # Write final JSONL (task_id + completion for human-eval compatibility)
    eval_samples = [{"task_id": s["task_id"], "completion": s["completion"]} for s in samples]
    write_jsonl(output_file, eval_samples)

    # Save stats with full per-problem data
    stats_file = output_file.replace(".jsonl", "_stats.json")
    with open(stats_file, "w") as f:
        json.dump({"stats": stats, "samples": samples}, f, indent=2)

    # Print summary
    print(f"\n{'=' * 60}")
    stop_reason = " (interrupted)" if _interrupted else " (quota exhausted)" if _quota_exhausted else ""
    print(f"HumanEval Generation{stop_reason} Complete")
    print(f"{'=' * 60}")
    print(f"Model:           {args.model}")
    print(f"Thinking:        {args.thinking}")
    print(f"Total:           {len(samples)}")
    print(f"Passed:          {stats['passed']}")
    print(f"Failed:          {stats['failed']}")
    print(f"Errors:          {stats['errors']}")
    print(f"pass@1:          {stats['passed'] / max(len(samples), 1):.4f}")
    print(f"Input tokens:    {stats['total_input_tokens']:,}")
    print(f"Output tokens:   {stats['total_output_tokens']:,}")
    print(f"Total tokens:    {stats['total_tokens']:,}")
    print(f"Elapsed:         {elapsed:.1f}s")
    print(f"Avg per problem: {elapsed / max(len(samples), 1):.1f}s")
    print(f"Output:          {output_file}")
    print(f"Stats:           {stats_file}")
    print()

    # Clean up checkpoint if all completed
    if not _interrupted and len(completed_ids) >= len(problem_list):
        cp_file = checkpoint_path(output_file)
        if os.path.exists(cp_file):
            os.remove(cp_file)
            print(f"Checkpoint cleaned up: {cp_file}")


if __name__ == "__main__":
    main()
