# Benchmarks

## Design Principles

All benchmarks MUST follow these principles:

### 1. Interruptible & Resumable

Every benchmark must support checkpoint/resume. If interrupted (Ctrl+C, OOM, crash), re-running with `--resume` picks up from the last completed problem without re-doing work.

### 2. No Overwrite Between Modes

The same model with different modes (e.g., Thinking vs Non-Thinking) must produce separate, non-conflicting output files. The output filename must encode the mode, e.g.:

- `{model}_thinking_samples.jsonl`
- `{model}_no_thinking_samples.jsonl`

Checkpoint files (`_checkpoint.json`), stats files (`_stats.json`), results files (`_results.jsonl`), and CSV reports (`_report.csv`) are all derived from the output filename, so they are automatically isolated.

### 3. Use DeerFlow Agent Harness

Benchmarks must go through DeerFlow's agent system, not direct model API calls. This means using `DeerFlowClient` (or equivalent) to run tasks through the full agent pipeline with tools (bash, read_file, write_file, str_replace, etc.) and sandbox execution. The model should be able to write code, run it, inspect errors, and iterate.

Exceptions: pure knowledge benchmarks (like multiple-choice QA) where tool use is irrelevant may use direct model calls, but this must be explicitly justified in the benchmark's documentation.

### 4. No Overwrite Between Runs

Each benchmark run produces a unique result directory/file with a sequential run number based on timestamp order. If you run the same model on the same benchmark twice, results are saved as:

```
results/{benchmark}/{model}/run_1/
results/{benchmark}/{model}/run_2/
results/{benchmark}/{model}/run_3/
```

Or alternatively with timestamp-based naming:

```
results/{benchmark}/{model}/2026-04-21T143000/
results/{benchmark}/{model}/2026-04-21T150000/
```

Never overwrite a previous run's results. The run number or timestamp is auto-assigned based on existing directories.

## Current Phase

**Only run non-thinking mode for now.** Thinking mode benchmarks will be added later.

## Current Benchmarks

| Benchmark | Location | Agent Harness | Run Numbering | Mode Isolation |
|-----------|----------|---------------|---------------|----------------|
| HumanEval | `humaneval/` | `DeerFlowClient.stream()` | `results/{model}/run_N/` | `_{mode}_samples.jsonl` |
| MMLU | `mmlu/` | DeepEval + `DeerFlowClient` | `results/{model}/run_N/` | `_{mode}_mmlu_results.json` |
| GAIA | `gaia/` | `DeerFlowClient.stream()` | `results/{model}/run_N/` | `_{mode}_results.jsonl` |
| SWE-bench | `swe_bench/` | `DeerFlowClient.stream()` | Manual `--output` | N/A |

### 5. Disable Memory During Benchmarks

All benchmarks must disable DeerFlow's memory system (`memory.enabled=False`, `memory.injection_enabled=False`) before creating the agent. This prevents cross-sample contamination where one task's context leaks into subsequent tasks.

### 6. Stop on API Quota Exhaustion

When any API quota is exhausted (model API, web search API, etc.), the benchmark must immediately stop — not continue producing errors. Detect quota errors by checking for patterns like `429`, `rate limit`, `quota`, `insufficient`, `billing`, etc. in the error message. On detection:
1. Print a clear "API QUOTA EXHAUSTED" message.
2. Save the checkpoint (so `--resume` works after topping up).
3. Do NOT mark the failed task as completed (so it gets retried on resume).
4. Exit the main loop.

### HumanEval

Each problem creates a thread workspace with `solution.py` (function stub) and `test_solution.py`. The agent uses bash/read_file/write_file/str_replace to implement the function, run tests, and iterate. After the agent finishes, the test is run one more time for a definitive pass/fail result.

```bash
bash run.sh --model=kimi-k2.5
```

### MMLU (via DeepEval)

Uses DeepEval's MMLU benchmark for dataset loading (57 subjects) and exact-match scoring, with a custom model adapter that routes through DeerFlowClient. This combines DeepEval's evaluation infrastructure with DeerFlow's agent pipeline.

```bash
bash run.sh --model=kimi-k2.5
bash run.sh --model=kimi-k2.5 --n_shots 5
```

### GAIA

GAIA (General AI Assistant) tests agentic capabilities: multi-step reasoning, tool use, and long-horizon planning. ~466 questions across 3 difficulty levels. The agent has full tool access (bash, web_search, etc.) for each task. Scoring uses GAIA's official normalization and exact-match rules.

```bash
bash run.sh --model=kimi-k2.5
bash run.sh --model=kimi-k2.5 --level 1
bash run.sh --model=kimi-k2.5 --level 2
```

### SWE-bench

Each task clones a git repo into the thread workspace at the base commit. The agent explores the codebase, identifies the bug, and produces a patch. The patch is evaluated using the swebench Docker harness.

```bash
bash run.sh --model=kimi-k2.5
```
