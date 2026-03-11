"""Summarize Phase 4 working-set realism results.

Reads a phase4 directory and outputs a CSV table with:
  - task, max_tokens_profile
  - prompt_tokens_total, fault_count, bytes_recalled
  - valid_rate, replicates
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def summarize_phase4_dir(run_dir: Path) -> list[dict]:
    """Summarize a single phase4 run directory.

    Returns list of dicts with per-task/profile aggregates.
    """
    agg_path = run_dir / "phase4_aggregate.json"
    if not agg_path.exists():
        raise FileNotFoundError(f"Missing phase4_aggregate.json in {run_dir}")

    # Load aggregate stats
    agg_data = json.loads(agg_path.read_text())

    # Scan JSONL logs to count faults and recall volume per (task, max_tokens_profile).
    fault_by_key = defaultdict(int)
    bytes_by_key = defaultdict(int)

    for jsonl_path in run_dir.glob("cognitive_step_v1_treatment_*.jsonl"):
        for line in jsonl_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Extract task and max_tokens_profile from openrouter_app.
            app = record.get("openrouter_app", "")
            if not app:
                continue
            # Format: pichay-phase4:task_name:maxXXXX:repN
            parts = app.split(":")
            if len(parts) < 3:
                continue
            task_name = parts[1]
            max_part = parts[2]
            if not max_part.startswith("max"):
                continue
            try:
                max_tokens_profile = int(max_part[len("max") :])
            except ValueError:
                continue
            key = (task_name, max_tokens_profile)

            # Count memory_fault operations (faults)
            for action in record.get("committed_actions", []):
                if isinstance(action, dict) and action.get("op") == "memory_fault":
                    ids = action.get("ids", [])
                    fault_by_key[key] += len(ids)

            # Sum bytes recalled
            recall = record.get("recall", {})
            bytes_recalled = recall.get("recalled_unit_json_chars", 0)
            if bytes_recalled:
                bytes_by_key[key] += bytes_recalled

    # Merge with aggregate stats
    results = []
    for row in agg_data:
        task = row["task"]
        key = (task, int(row["max_tokens_profile"]))
        results.append(
            {
                "task": task,
                "max_tokens_profile": row["max_tokens_profile"],
                "prompt_tokens_total": row["prompt_tokens_total"],
                "fault_count": fault_by_key.get(key, 0),
                "bytes_recalled": bytes_by_key.get(key, 0),
                "valid_rate": round(row["valid_rate"], 4),
                "valid_steps": row["valid_steps_total"],
                "steps_total": row["steps_total"],
                "replicates": row["replicates"],
            }
        )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize Phase 4 working-set realism results"
    )
    parser.add_argument(
        "phase4_dir",
        type=Path,
        help="Path to phase4 run directory (contains phase4_aggregate.json)",
    )
    parser.add_argument(
        "--format",
        choices=["csv", "json", "table"],
        default="table",
        help="Output format",
    )
    args = parser.parse_args()

    results = summarize_phase4_dir(args.phase4_dir)

    if args.format == "json":
        print(json.dumps(results, indent=2))
    elif args.format == "csv":
        print(
            "task,max_tokens_profile,prompt_tokens_total,fault_count,bytes_recalled,valid_rate,valid_steps,steps_total,replicates"
        )
        for row in results:
            print(
                f"{row['task']},{row['max_tokens_profile']},{row['prompt_tokens_total']},"
                f"{row['fault_count']},{row['bytes_recalled']},{row['valid_rate']},"
                f"{row['valid_steps']},{row['steps_total']},{row['replicates']}"
            )
    else:  # table
        print(
            f"{'Task':<25} {'Profile':>8} {'Prompt Tok':>12} {'Faults':>8} {'Bytes Recalled':>16} {'Valid %':>8}"
        )
        print("-" * 85)
        for row in results:
            print(
                f"{row['task']:<25} {row['max_tokens_profile']:>8} "
                f"{row['prompt_tokens_total']:>12} {row['fault_count']:>8} "
                f"{row['bytes_recalled']:>16} {row['valid_rate']*100:>7.1f}%"
            )


if __name__ == "__main__":
    main()
