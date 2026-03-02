#!/usr/bin/env python3
"""Pichay experiment runner.

Single entry point: starts the proxy, launches Claude Code through it,
captures all artifacts when done.

Usage:
    # Baseline run (observe only)
    python -m pichay --treatment baseline --project /path/to/project --prompt "Build X"

    # Compact run (dead tool eviction)
    python -m pichay --treatment compact --compact --project /path/to/project --prompt "Build X"

    # With temperature control
    python -m pichay --treatment compact --compact --temperature 0 --project /path/to/project --prompt "Build X"

Artifacts are saved to experiments/{treatment}_run{N}/ in the pichay project directory.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from pichay.proxy import create_app, find_free_port


def find_project_claude_dir(project_dir: str) -> Path:
    """Map a project directory to its ~/.claude/projects/ path."""
    normalized = project_dir.replace("/", "-")
    return Path.home() / ".claude" / "projects" / normalized


# Environment variables that cause Claude Code to detect nested invocation.
_CLAUDE_NESTED_VARS = [
    "CLAUDECODE",
    "CLAUDE_CODE_ENTRYPOINT",
    "CLAUDE_CODE_SSE_PORT",
    "CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS",
]


def clean_env_for_subprocess() -> dict[str, str]:
    """Build environment for nested Claude Code, removing detection vars."""
    env = os.environ.copy()
    for var in _CLAUDE_NESTED_VARS:
        env.pop(var, None)
    return env


def run_experiment(args: argparse.Namespace) -> None:
    """Run a single experiment."""
    pichay_dir = Path(__file__).resolve().parent.parent.parent
    exp_dir = pichay_dir / "experiments" / f"{args.treatment}_run{args.run}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    log_dir = exp_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    project_dir = Path(args.project).resolve()
    if not project_dir.is_dir():
        print(f"Error: project directory not found: {project_dir}", file=sys.stderr)
        sys.exit(1)

    print("=" * 60, file=sys.stderr)
    print(f"Pichay Experiment", file=sys.stderr)
    print(f"  Treatment:    {args.treatment}", file=sys.stderr)
    print(f"  Run:          {args.run}", file=sys.stderr)
    print(f"  Project:      {project_dir}", file=sys.stderr)
    print(f"  Compact:      {args.compact}", file=sys.stderr)
    print(f"  Temperature:  {args.temperature}", file=sys.stderr)
    print(f"  Output:       {exp_dir}", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    # Save config
    config = {
        "treatment": args.treatment,
        "run": args.run,
        "project_dir": str(project_dir),
        "branch": args.branch,
        "compact": args.compact,
        "age_threshold": args.age_threshold,
        "min_size": args.min_size,
        "temperature": args.temperature,
        "prompt": args.prompt,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    (exp_dir / "config.json").write_text(json.dumps(config, indent=2))

    # Step 1: Reset project
    if args.branch:
        print(f"\n[1/5] Resetting project to {args.branch}...", file=sys.stderr)
        subprocess.run(
            ["git", "checkout", args.branch],
            cwd=project_dir, capture_output=True,
        )
        subprocess.run(
            ["git", "clean", "-fd"],
            cwd=project_dir, capture_output=True,
        )
    else:
        print(f"\n[1/5] No branch specified, using current state.", file=sys.stderr)

    # Clear previous session data
    claude_dir = find_project_claude_dir(str(project_dir))
    if args.clear_session and claude_dir.is_dir():
        for jsonl in claude_dir.glob("*.jsonl"):
            jsonl.unlink()
        print(f"  Cleared session data in {claude_dir}", file=sys.stderr)

    # Step 2: Start proxy
    print(f"\n[2/5] Starting proxy...", file=sys.stderr)
    port = find_free_port()

    app = create_app(
        log_dir,
        compact=args.compact,
        age_threshold=args.age_threshold,
        min_size=args.min_size,
    )
    if args.temperature is not None:
        app.config["temperature_override"] = args.temperature

    # Run Flask in a thread
    server_thread = threading.Thread(
        target=lambda: app.run(
            host="127.0.0.1", port=port, threaded=True, use_reloader=False,
        ),
        daemon=True,
    )
    server_thread.start()
    time.sleep(1)  # Let server bind
    print(f"  Proxy on http://localhost:{port}", file=sys.stderr)

    # Step 3: Run Claude Code
    print(f"\n[3/5] Launching Claude Code...", file=sys.stderr)
    print(f"  Prompt: {args.prompt[:80]}...", file=sys.stderr)
    print(file=sys.stderr)

    env = clean_env_for_subprocess()
    env["ANTHROPIC_BASE_URL"] = f"http://localhost:{port}"

    claude_cmd = [
        "claude",
        "-p",
        "--dangerously-skip-permissions",
        "--max-budget-usd", str(args.max_budget),
        args.prompt,
    ]
    try:
        subprocess.run(
            claude_cmd,
            cwd=project_dir,
            env=env,
            stdin=subprocess.DEVNULL,
        )
    except KeyboardInterrupt:
        print("\n  Interrupted by user.", file=sys.stderr)
    except FileNotFoundError:
        print("  Error: 'claude' not found in PATH.", file=sys.stderr)
        sys.exit(1)

    # Step 4: Capture artifacts
    print(f"\n[4/5] Capturing artifacts...", file=sys.stderr)

    # Copy session data
    if claude_dir.is_dir():
        session_dest = exp_dir / "session"
        if session_dest.exists():
            shutil.rmtree(session_dest)
        shutil.copytree(claude_dir, session_dest, dirs_exist_ok=True)

    # Git state
    subprocess.run(
        ["git", "log", "--oneline", "-20"],
        cwd=project_dir,
        stdout=open(exp_dir / "git_log.txt", "w"),
        stderr=subprocess.DEVNULL,
    )
    if args.branch:
        subprocess.run(
            ["git", "diff", f"{args.branch}..HEAD"],
            cwd=project_dir,
            stdout=open(exp_dir / "git_diff.txt", "w"),
            stderr=subprocess.DEVNULL,
        )
        subprocess.run(
            ["git", "diff", "--stat", f"{args.branch}..HEAD"],
            cwd=project_dir,
            stdout=open(exp_dir / "git_diff_stat.txt", "w"),
            stderr=subprocess.DEVNULL,
        )

    # Test results
    pyproject = project_dir / "pyproject.toml"
    if pyproject.exists():
        subprocess.run(
            ["uv", "run", "pytest", "tests/", "-v", "--tb=short"],
            cwd=project_dir,
            stdout=open(exp_dir / "test_results.txt", "w"),
            stderr=subprocess.STDOUT,
        )

    # Record end time
    config["ended_at"] = datetime.now(timezone.utc).isoformat()
    (exp_dir / "config.json").write_text(json.dumps(config, indent=2))

    # Step 5: Summary
    print(f"\n[5/5] Run complete.", file=sys.stderr)
    print(f"  Artifacts: {exp_dir}", file=sys.stderr)

    # Run eval if proxy log exists
    proxy_logs = list(log_dir.glob("proxy_*.jsonl"))
    if proxy_logs:
        from pichay.eval import analyze_run, print_run_summary
        summary = analyze_run(proxy_logs[0], label=args.treatment)
        print_run_summary(summary)


def main():
    parser = argparse.ArgumentParser(
        description="Pichay — context paging experiment runner"
    )
    parser.add_argument(
        "--treatment", required=True,
        help="Treatment label (e.g., baseline, compact, trimmed)",
    )
    parser.add_argument(
        "--run", type=int, default=1,
        help="Run number (default: 1)",
    )
    parser.add_argument(
        "--project", required=True,
        help="Target project directory",
    )
    parser.add_argument(
        "--prompt", required=True,
        help="Starting prompt for Claude Code",
    )
    parser.add_argument(
        "--branch", default=None,
        help="Git branch to reset to before each run",
    )
    parser.add_argument(
        "--compact", action="store_true",
        help="Enable dead tool result eviction",
    )
    parser.add_argument(
        "--age-threshold", type=int, default=4,
        help="Eviction age threshold in user-turns (default: 4)",
    )
    parser.add_argument(
        "--min-size", type=int, default=500,
        help="Min tool result size for eviction (default: 500)",
    )
    parser.add_argument(
        "--temperature", type=float, default=None,
        help="Override temperature (e.g., 0 for deterministic)",
    )
    parser.add_argument(
        "--max-budget", type=float, default=20.0,
        help="Maximum dollar budget for Claude Code session (default: $20)",
    )
    parser.add_argument(
        "--clear-session", action="store_true", default=True,
        help="Clear previous Claude session data (default: yes)",
    )
    parser.add_argument(
        "--no-clear-session", action="store_false", dest="clear_session",
        help="Keep previous Claude session data",
    )
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
