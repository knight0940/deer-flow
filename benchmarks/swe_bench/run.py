"""
SWE-bench benchmark using DeerFlow agent.

Uses DeerFlowClient as the agent to fix issues from SWE-bench tasks.
The agent has access to bash, read_file, str_replace, write_file tools
to explore and modify the codebase.

Architecture (per task):
    1. Load SWE-bench task (repo, base_commit, problem_statement)
    2. Clone repo into thread workspace and checkout base_commit
    3. Send issue description to DeerFlow agent via DeerFlowClient.chat()
    4. Agent uses bash/read_file/str_replace tools to fix the code
    5. Run `git diff` in workspace to extract patch
    6. Save prediction to JSONL (instance_id + model_patch)

Features:
- Checkpoint/resume: saves progress after every task.
  If interrupted, re-run with --resume to continue.
- Graceful interrupt: Ctrl+C stops after the current task completes and saves.
- Token tracking: per-task token usage statistics.

Usage:
    uv run python run.py
    uv run python run.py --dataset lite        # SWE-bench Lite (25 tasks)
    uv run python run.py --dataset verified    # SWE-bench Verified (500 tasks)
    uv run python run.py --dataset pro         # SWE-bench Pro (731 tasks)
    uv run python run.py --resume              # Resume from checkpoint
    uv run python run.py --fresh               # Ignore checkpoint, start over
    uv run python run.py --start 0 --end 3     # Run first 3 tasks only
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

# Shared benchmark utilities
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils import handle_pauseable_error, is_quota_error

MODEL_NAME = "kimi-for-coding"

DATASETS = {
    "lite": "princeton-nlp/SWE-bench_Lite",
    "verified": "princeton-nlp/SWE-bench_Verified",
    "pro": "ScaleAI/SWE-bench_Pro",
}

# Prompt template for SWE-bench tasks
PROMPT_TEMPLATE = """\
I need you to fix a bug in a repository. The repository is already cloned at `/mnt/user-data/workspace/`.

Repository: {repo}
Base commit: {base_commit}

## Issue
{problem_statement}

## Instructions
1. First, explore the repository structure in /mnt/user-data/workspace/ to understand the codebase
2. Find the relevant code related to this issue
3. Make minimal, targeted changes to fix the issue
4. Do NOT modify test files unless absolutely necessary for the fix
5. Always use `cd /mnt/user-data/workspace` before running any commands

Use the available tools (bash, read_file, str_replace, write_file) to explore and modify the code.
"""

# Maximum LangGraph recursion limit (controls agent turns)
MAX_RECURSION = 100

# Timeout for git operations (seconds)
GIT_TIMEOUT = 300

# Global flags for graceful interrupt and quota exhaustion
_interrupted = False
_quota_exhausted = False


def _handle_signal(signum, frame):
    global _interrupted
    print("\n\nInterrupt received! Finishing current task and saving...")
    _interrupted = True


signal.signal(signal.SIGINT, _handle_signal)


# ---------------------------------------------------------------------------
# Checkpoint helpers (same pattern as HumanEval benchmark)
# ---------------------------------------------------------------------------


def checkpoint_path(output_file: str) -> str:
    return output_file.replace(".jsonl", "_checkpoint.json")


def save_checkpoint(output_file: str, predictions: list, stats: dict):
    cp_file = checkpoint_path(output_file)
    data = {"predictions": predictions, "stats": stats}
    tmp = cp_file + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, cp_file)


def load_checkpoint(output_file: str) -> tuple[list, dict] | None:
    cp_file = checkpoint_path(output_file)
    if not os.path.exists(cp_file):
        return None
    with open(cp_file) as f:
        data = json.load(f)
    return data["predictions"], data["stats"]


# ---------------------------------------------------------------------------
# Workspace helpers
# ---------------------------------------------------------------------------


def get_thread_workspace(thread_id: str) -> str:
    """Get the physical workspace path for a thread."""
    paths = get_paths()
    workspace = paths.sandbox_work_dir(thread_id)
    return str(workspace)


def setup_workspace(thread_id: str, repo_url: str, base_commit: str) -> str:
    """Create thread directories, clone repo, and checkout base commit."""
    paths = get_paths()
    paths.ensure_thread_dirs(thread_id)

    workspace = get_thread_workspace(thread_id)

    # Clone repo into workspace
    result = subprocess.run(
        ["git", "clone", repo_url, "."],
        cwd=workspace,
        capture_output=True,
        text=True,
        timeout=GIT_TIMEOUT,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git clone failed: {result.stderr}")

    # Checkout base commit
    result = subprocess.run(
        ["git", "checkout", base_commit],
        cwd=workspace,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git checkout {base_commit} failed: {result.stderr}")

    return workspace


def get_git_diff(workspace: str) -> str:
    """Get unified diff of all changes (staged + unstaged + untracked)."""
    # Stage all changes (including new files) to capture untracked files
    subprocess.run(
        ["git", "add", "-A"],
        cwd=workspace,
        capture_output=True,
        timeout=30,
    )
    result = subprocess.run(
        ["git", "diff", "--cached"],
        cwd=workspace,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result.stdout


def cleanup_workspace(thread_id: str) -> None:
    """Delete thread workspace to free disk space."""
    paths = get_paths()
    paths.delete_thread_dir(thread_id)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_dataset(dataset_name: str) -> list[dict]:
    """Load SWE-bench dataset from HuggingFace."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: 'datasets' package is required. Install with: pip install datasets")
        sys.exit(1)

    print(f"Loading dataset: {dataset_name}...")
    ds = load_dataset(dataset_name, split="test")
    return list(ds)


# ---------------------------------------------------------------------------
# Agent execution
# ---------------------------------------------------------------------------


def run_agent(
    client: DeerFlowClient,
    instance: dict,
    thread_id: str,
    max_recursion: int = MAX_RECURSION,
) -> tuple[str, int, int, str | None]:
    """Run DeerFlow agent on a SWE-bench task.

    Returns (patch_text, input_tokens, output_tokens, error).
    """
    repo = instance["repo"]
    base_commit = instance["base_commit"]
    problem_statement = instance["problem_statement"]
    repo_url = f"https://github.com/{repo}.git"

    # Setup workspace: clone repo at base commit
    try:
        workspace = setup_workspace(thread_id, repo_url, base_commit)
    except Exception as e:
        return "", 0, 0, f"workspace setup failed: {e}"

    # Build prompt
    prompt = PROMPT_TEMPLATE.format(
        repo=repo,
        base_commit=base_commit,
        problem_statement=problem_statement,
    )

    # Run agent with streaming to track usage.
    # We accumulate tokens from both "end" events (normal completion) and
    # intermediate "messages-tuple" events so we don't lose data if the
    # agent hits the recursion limit (no "end" event in that case).
    total_input = 0
    total_output = 0
    seen_usage_ids: set[str] = set()

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
                # Accumulate per-message usage from intermediate events
                msg_id = event.data.get("id", "")
                meta = event.data.get("usage_metadata")
                if meta and msg_id and msg_id not in seen_usage_ids:
                    seen_usage_ids.add(msg_id)
                    total_input += meta.get("input_tokens", 0)
                    total_output += meta.get("output_tokens", 0)
    except Exception as e:
        # Even if agent fails, try to get whatever diff was produced
        patch = get_git_diff(workspace)
        return patch, total_input, total_output, f"agent error: {e}"

    # Extract git diff
    patch = get_git_diff(workspace)

    return patch, total_input, total_output, None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="SWE-bench benchmark via DeerFlow agent")
    parser.add_argument("--model", default=MODEL_NAME, help="Model name in config.yaml")
    parser.add_argument("--dataset", default="lite", choices=list(DATASETS.keys()), help="Dataset variant")
    parser.add_argument("--output", default=None, help="Output JSONL file")
    parser.add_argument("--thinking", action="store_true", help="Enable thinking mode")
    parser.add_argument("--start", type=int, default=0, help="Start index")
    parser.add_argument("--end", type=int, default=-1, help="End index (-1 for all)")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--fresh", action="store_true", help="Ignore checkpoint, start over")
    parser.add_argument("--max_recursion", type=int, default=MAX_RECURSION, help="Max agent recursion limit")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between tasks (seconds)")
    parser.add_argument("--cleanup", action="store_true", help="Delete thread workspaces after extracting diff")
    args = parser.parse_args()

    dataset_name = DATASETS[args.dataset]
    output_file = args.output or f"swebench_{args.dataset}_{args.model}_predictions.jsonl"

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

    # Load dataset
    instances = load_dataset(dataset_name)
    if args.end == -1:
        args.end = len(instances)
    instances = instances[args.start : args.end]

    # Initialize state
    predictions: list[dict] = []
    completed_ids: set[str] = set()
    stats = {
        "model": args.model,
        "dataset": args.dataset,
        "dataset_name": dataset_name,
        "thinking": args.thinking,
        "n_instances": len(instances),
        "completed": 0,
        "errors": 0,
        "empty_patches": 0,
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_tokens": 0,
        "elapsed_seconds": 0,
    }

    # Resume from checkpoint
    if args.resume and not args.fresh:
        cp = load_checkpoint(output_file)
        if cp:
            saved_predictions, _saved_stats = cp
            predictions = saved_predictions
            completed_ids = {p["instance_id"] for p in predictions}
            stats["total_input_tokens"] = sum(p.get("input_tokens", 0) for p in predictions)
            stats["total_output_tokens"] = sum(p.get("output_tokens", 0) for p in predictions)
            print(f"Resumed from checkpoint: {len(predictions)} completed, {len(instances) - len(completed_ids)} remaining")
            print()
        else:
            print("No checkpoint found, starting fresh")
            print()

    print(f"Model:          {args.model}")
    print(f"Provider:       {model_config.use}")
    print(f"Dataset:        {args.dataset} ({dataset_name})")
    print(f"Thinking:       {args.thinking}")
    print(f"Instances:      {len(instances)} ({args.start}-{args.start + len(instances) - 1})")
    print(f"Completed:      {len(completed_ids)}")
    print(f"Max recursion:  {args.max_recursion}")
    print(f"Cleanup:        {args.cleanup}")
    print(f"Output:         {output_file}")
    print()

    # Create DeerFlow client (reused across tasks, thread isolation via checkpointer)
    client = DeerFlowClient(
        model_name=args.model,
        thinking_enabled=args.thinking,
        subagent_enabled=False,
        plan_mode=False,
    )

    start_time = time.time()

    global _interrupted, _quota_exhausted
    for i, instance in enumerate(instances):
        if _interrupted or _quota_exhausted:
            break

        instance_id = instance["instance_id"]

        # Skip already completed
        if instance_id in completed_ids:
            continue

        print(f"[{len(completed_ids) + 1}/{len(instances)}] {instance_id} ...", end=" ", flush=True)

        # Retry loop: pause on network/quota errors, never skip
        while True:
            if _interrupted:
                break

            # Use deterministic thread_id for reproducibility + uniqueness suffix
            thread_id = f"swebench-{instance_id}-{uuid.uuid4().hex[:8]}"

            try:
                patch, in_tok, out_tok, error = run_agent(
                    client,
                    instance,
                    thread_id,
                    max_recursion=args.max_recursion,
                )

                if error:
                    action = handle_pauseable_error(error, context=f"task={instance_id}")
                    if action != "other":
                        save_checkpoint(output_file, predictions, stats)
                        continue

                    print(f"ERROR: {error}")
                    stats["errors"] += 1
                    predictions.append({
                        "instance_id": instance_id,
                        "model_patch": patch if patch else "",
                        "model_name_or_path": args.model,
                        "input_tokens": in_tok,
                        "output_tokens": out_tok,
                        "error": str(error),
                        "thread_id": thread_id,
                    })
                elif not patch.strip():
                    print(f"EMPTY PATCH (in={in_tok}, out={out_tok})")
                    stats["empty_patches"] += 1
                    predictions.append({
                        "instance_id": instance_id,
                        "model_patch": "",
                        "model_name_or_path": args.model,
                        "input_tokens": in_tok,
                        "output_tokens": out_tok,
                        "error": "empty_patch",
                        "thread_id": thread_id,
                    })
                else:
                    print(f"OK (in={in_tok}, out={out_tok}, patch={len(patch)} chars)")
                    predictions.append({
                        "instance_id": instance_id,
                        "model_patch": patch,
                        "model_name_or_path": args.model,
                        "input_tokens": in_tok,
                        "output_tokens": out_tok,
                        "error": None,
                        "thread_id": thread_id,
                    })

                completed_ids.add(instance_id)
                stats["completed"] = len(completed_ids)
                stats["total_input_tokens"] += in_tok
                stats["total_output_tokens"] += out_tok
                stats["total_tokens"] = stats["total_input_tokens"] + stats["total_output_tokens"]
                break  # done with this task, move to next

            except Exception as e:
                action = handle_pauseable_error(e, context=f"task={instance_id}")
                if action != "other":
                    save_checkpoint(output_file, predictions, stats)
                    continue
                # other exception — record and move on
                print(f"FAILED: {e}")
                stats["errors"] += 1
                predictions.append({
                    "instance_id": instance_id,
                    "model_patch": "",
                    "model_name_or_path": args.model,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "error": str(e),
                    "thread_id": thread_id,
                })
                completed_ids.add(instance_id)
                break  # move to next task

        # Cleanup workspace to save disk space
        if args.cleanup:
            try:
                cleanup_workspace(thread_id)
            except Exception:
                pass  # Non-critical, don't fail the task

        # Save checkpoint after each task
        save_checkpoint(output_file, predictions, stats)

        if args.delay > 0 and not _interrupted:
            time.sleep(args.delay)

    elapsed = time.time() - start_time
    stats["elapsed_seconds"] = round(elapsed, 1)
    stats["completed"] = len(completed_ids)
    stats["total_tokens"] = stats["total_input_tokens"] + stats["total_output_tokens"]

    # Write final JSONL in swebench format: instance_id + model_patch + model_name_or_path
    with open(output_file, "w") as f:
        for p in predictions:
            f.write(json.dumps({
                "instance_id": p["instance_id"],
                "model_patch": p["model_patch"],
                "model_name_or_path": p["model_name_or_path"],
            }) + "\n")

    # Save stats with full per-task data
    stats_file = output_file.replace(".jsonl", "_stats.json")
    with open(stats_file, "w") as f:
        json.dump({"stats": stats, "predictions": predictions}, f, indent=2)

    # Print summary
    n_patches = sum(1 for p in predictions if p.get("model_patch", "").strip())
    n_errors = stats["errors"]

    print(f"\n{'=' * 60}")
    print(f"SWE-bench Generation {'(interrupted) ' if _interrupted else ''}Complete")
    print(f"{'=' * 60}")
    print(f"Model:           {args.model}")
    print(f"Dataset:         {args.dataset} ({dataset_name})")
    print(f"Total tasks:     {len(predictions)}")
    print(f"Patches:         {n_patches}")
    print(f"Empty patches:   {stats['empty_patches']}")
    print(f"Errors:          {n_errors}")
    print(f"Input tokens:    {stats['total_input_tokens']:,}")
    print(f"Output tokens:   {stats['total_output_tokens']:,}")
    print(f"Total tokens:    {stats['total_tokens']:,}")
    print(f"Elapsed:         {elapsed:.1f}s")
    print(f"Avg per task:    {elapsed / max(len(predictions), 1):.1f}s")
    print(f"Output:          {output_file}")
    print(f"Stats:           {stats_file}")
    print()

    # Clean up checkpoint if all tasks completed
    if not _interrupted and len(completed_ids) >= len(instances):
        cp_file = checkpoint_path(output_file)
        if os.path.exists(cp_file):
            os.remove(cp_file)
            print(f"Checkpoint cleaned up: {cp_file}")


if __name__ == "__main__":
    main()
