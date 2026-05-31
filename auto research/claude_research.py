"""Local Claude Code autoresearch runner.

This wraps Claude Code in the style of Karpathy/autoresearch:

- `prepare.py` is fixed evaluation/data loading.
- `pipeline.py` is the single file Claude is allowed to edit.
- `program.md` is the human steering surface.
- `train.py` evaluates the current pipeline and prints `reward`.

Entrypoint:
    uv run python claude_research.py --data-glob "../bci-sdk/data/*_prod*.csv"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from queue import Empty, Queue
import re
import shutil
import subprocess
from threading import Thread
import time


ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT / "runs"
AGENT_RUNS_DIR = RUNS_DIR / "claude_code"
PIPELINE_PATH = ROOT / "pipeline.py"
RESULTS_HEADER = "iteration\treward\tstatus\tdescription\tseconds\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local Claude Code as the BCI autoresearch agent.")
    parser.add_argument("--subject", default="S03")
    parser.add_argument("--stage", type=int, default=4)
    parser.add_argument("--data-glob", action="append", default=[])
    parser.add_argument("--data-path", action="append", default=[])
    parser.add_argument("--iterations", type=int, default=1, help="Number of Claude Code propose/evaluate loops.")
    parser.add_argument("--tag", default=time.strftime("%b%d-%H%M").lower())
    parser.add_argument("--model", default="sonnet", help="Claude Code model alias, e.g. sonnet or opus.")
    parser.add_argument("--max-budget-usd", type=float, default=3.0)
    parser.add_argument("--claude-timeout-seconds", type=int, default=600)
    parser.add_argument("--permission-mode", default="bypassPermissions")
    parser.add_argument("--claude-output-format", default="text", choices=["text", "json", "stream-json"])
    parser.add_argument("--dry-run", action="store_true", help="Write prompt files but do not invoke Claude.")
    args = parser.parse_args()
    if not args.data_glob and not args.data_path:
        args.data_glob = ["../bci-sdk/data/*_prod*.csv"]

    if not shutil.which("claude"):
        raise SystemExit("Claude Code CLI not found. Install/login first, then rerun.")

    AGENT_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_dir = AGENT_RUNS_DIR / args.tag
    run_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / "results.tsv"
    live_path = run_dir / "LIVE.md"
    if not results_path.exists():
        results_path.write_text(RESULTS_HEADER, encoding="utf-8")
    init_live_doc(live_path, args, run_dir)

    train_cmd = build_train_command(args)
    best_reward = evaluate_current(train_cmd, run_dir / "baseline.log", live_path, "Baseline evaluation")
    print(f"[baseline] reward={best_reward:.6f}")
    append_live(live_path, f"\n## Baseline\n\n- reward: `{best_reward:.6f}`\n- train entrypoint: `{' '.join(train_cmd)}`\n")

    for iteration in range(1, args.iterations + 1):
        prompt = build_prompt(args, iteration, best_reward, train_cmd, results_path)
        prompt_path = run_dir / f"prompt_{iteration:03d}.md"
        prompt_path.write_text(prompt, encoding="utf-8")
        before = PIPELINE_PATH.read_text(encoding="utf-8")

        if args.dry_run:
            print(f"[dry-run] wrote {prompt_path}")
            append_live(live_path, f"\n## Iteration {iteration}\n\nDry run wrote `{prompt_path.name}`; Claude was not invoked.\n")
            continue

        started = time.time()
        claude_log = run_dir / f"claude_{iteration:03d}.log"
        append_live(
            live_path,
            f"\n## Iteration {iteration}\n\n"
            f"- prompt: `{prompt_path.name}`\n"
            f"- Claude log: `{claude_log.name}`\n"
            f"- started: `{timestamp()}`\n\n"
            "### Claude stream\n\n```text\n",
        )
        claude_ok = run_claude(prompt, claude_log, live_path, args)
        append_live(live_path, "```\n")
        if not claude_ok:
            PIPELINE_PATH.write_text(before, encoding="utf-8")
            append_result(results_path, iteration, float("-inf"), "crash", "Claude Code timed out or exited nonzero", time.time() - started)
            append_live(live_path, f"\n### Result\n\n- status: `crash`\n- reason: Claude Code timed out or exited nonzero\n")
            print(f"[crash] iteration={iteration} Claude Code timed out or exited nonzero")
            continue
        reward = evaluate_current(train_cmd, run_dir / f"eval_{iteration:03d}.log", live_path, f"Evaluation {iteration}")
        elapsed = time.time() - started

        if reward > best_reward:
            status = "keep"
            best_reward = reward
            description = "Claude Code pipeline.py edit improved reward"
            subprocess.run(["git", "add", "pipeline.py"], cwd=ROOT, check=False)
        else:
            status = "discard"
            description = "Claude Code edit did not improve reward; restored pipeline.py"
            PIPELINE_PATH.write_text(before, encoding="utf-8")

        append_result(results_path, iteration, reward, status, description, elapsed)
        append_live(
            live_path,
            f"\n### Result\n\n"
            f"- status: `{status}`\n"
            f"- reward: `{reward:.6f}`\n"
            f"- best_reward: `{best_reward:.6f}`\n"
            f"- seconds: `{elapsed:.1f}`\n"
            f"- description: {description}\n",
        )
        print(f"[{status}] iteration={iteration} reward={reward:.6f} best={best_reward:.6f}")

    print(f"\nClaude Code autoresearch run dir: {run_dir}")
    print(f"Live doc: {live_path}")
    print(f"Entrypoint used: {' '.join(train_cmd)}")


def build_train_command(args: argparse.Namespace) -> list[str]:
    command = [
        "uv",
        "run",
        "python",
        "train.py",
        "--subject",
        args.subject,
        "--stage",
        str(args.stage),
        "--sealed",
        "--export-final",
    ]
    for path in args.data_path:
        command.extend(["--data-path", path])
    for pattern in args.data_glob:
        command.extend(["--data-glob", pattern])
    return command


def evaluate_current(command: list[str], log_path: Path, live_path: Path | None = None, title: str = "Evaluation") -> float:
    if live_path:
        append_live(live_path, f"\n### {title}\n\n```text\n$ {' '.join(command)}\n")
    run_streaming_command(command, ROOT, log_path, live_path, timeout_seconds=None)
    if live_path:
        append_live(live_path, "```\n")
    text = log_path.read_text(encoding="utf-8")
    match = re.search(r"^reward:\s+([-+0-9.]+)", text, flags=re.MULTILINE)
    if not match:
        return float("-inf")
    return float(match.group(1))


def run_claude(prompt: str, log_path: Path, live_path: Path, args: argparse.Namespace) -> bool:
    command = [
        "claude",
        "--print",
        "--model",
        args.model,
        "--permission-mode",
        args.permission_mode,
        "--max-budget-usd",
        str(args.max_budget_usd),
        "--output-format",
        args.claude_output_format,
        prompt,
    ]
    if args.claude_output_format == "stream-json":
        command.insert(-1, "--verbose")
    return run_streaming_command(command, ROOT, log_path, live_path, timeout_seconds=args.claude_timeout_seconds) == 0


def run_streaming_command(
    command: list[str],
    cwd: Path,
    log_path: Path,
    live_path: Path | None,
    timeout_seconds: int | None,
) -> int:
    output_queue: Queue[str] = Queue()
    process = subprocess.Popen(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    def read_output() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            output_queue.put(line)

    reader = Thread(target=read_output, daemon=True)
    reader.start()
    started = time.time()

    with log_path.open("w", encoding="utf-8") as handle:
        while True:
            try:
                line = output_queue.get(timeout=0.2)
            except Empty:
                line = ""
            if line:
                handle.write(line)
                handle.flush()
                if live_path:
                    append_live(live_path, line)

            if timeout_seconds is not None and time.time() - started > timeout_seconds:
                process.kill()
                message = f"\n[TIMEOUT] command exceeded {timeout_seconds}s\n"
                handle.write(message)
                handle.flush()
                if live_path:
                    append_live(live_path, message)
                return 124

            return_code = process.poll()
            if return_code is not None:
                while True:
                    try:
                        line = output_queue.get_nowait()
                    except Empty:
                        break
                    handle.write(line)
                    if live_path:
                        append_live(live_path, line)
                return return_code


def build_prompt(
    args: argparse.Namespace,
    iteration: int,
    best_reward: float,
    train_cmd: list[str],
    results_path: Path,
) -> str:
    return f"""# BCI Claude Code Autoresearch

You are the local Claude Code research agent for this repository, following the
Karpathy/autoresearch pattern.

Read these files before editing:
- README.md
- program.md
- prepare.py
- pipeline.py
- train.py

Rules:
- Edit ONLY `pipeline.py`.
- Do NOT edit `prepare.py`; it is the fixed evaluation/data harness.
- Do NOT install dependencies.
- Keep the action-channel constraint:
  - left_squeeze uses AF7 only (channel index 1)
  - right_squeeze uses AF8 only (channel index 0)
  - eye_blink uses CHEEK_R only (channel index 2)
  - nothing is fallback.
- Optimize reward. Higher is better.
- Current best reward before this iteration: {best_reward:.6f}

Experiment command:
```bash
{' '.join(train_cmd)} > run.log 2>&1
```

Your task for iteration {iteration}:
1. Propose one concrete research idea.
2. Modify `pipeline.py` to implement it.
3. Run the experiment command above.
4. Inspect `run.log`, especially the `reward:` line.
5. Leave `pipeline.py` in the best state you found for this one idea.

The wrapper script will independently re-run evaluation and keep/discard your
change. It logs to `{results_path}`.
"""


def append_result(results_path: Path, iteration: int, reward: float, status: str, description: str, seconds: float) -> None:
    with results_path.open("a", encoding="utf-8") as handle:
        handle.write(f"{iteration}\t{reward:.6f}\t{status}\t{description}\t{seconds:.1f}\n")


def init_live_doc(live_path: Path, args: argparse.Namespace, run_dir: Path) -> None:
    live_path.write_text(
        "# Claude Code Autoresearch Live Log\n\n"
        f"- started: `{timestamp()}`\n"
        f"- run_dir: `{run_dir}`\n"
        f"- iterations: `{args.iterations}`\n"
        f"- max_budget_usd: `{args.max_budget_usd}`\n"
        f"- claude_output_format: `{args.claude_output_format}`\n"
        f"- data_glob: `{args.data_glob}`\n"
        f"- data_path: `{args.data_path}`\n\n",
        encoding="utf-8",
    )


def append_live(live_path: Path, text: str) -> None:
    with live_path.open("a", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()


def timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":
    main()
