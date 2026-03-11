from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import sys

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import cognitive_step_harness as harness  # type: ignore
from repro_meta import get_git_meta, script_hashes  # type: ignore


@dataclass
class Phase4Task:
    name: str
    teacher_trace_mode: str
    max_steps: int
    description: str


DEFAULT_TASKS = [
    Phase4Task(
        name="baseline_synthetic",
        teacher_trace_mode="synthetic",
        max_steps=8,
        description="Longer-horizon synthetic protocol durability run.",
    ),
    Phase4Task(
        name="reduced_guidance",
        teacher_trace_mode="synthetic_reduced",
        max_steps=8,
        description="Ablation under reduced exemplar guidance.",
    ),
    Phase4Task(
        name="minimal_guidance",
        teacher_trace_mode="none",
        max_steps=8,
        description="No exemplar condition for spontaneous participation check.",
    ),
]


def _summarize_log(path: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    valid = sum(1 for r in rows if r.get("ok"))
    required_misses = sum(
        len((r.get("scoreboard") or {}).get("required_ops_missing", []))
        for r in rows
    )
    semantic_rejections = sum(
        int((r.get("scoreboard") or {}).get("semantic_rejection_count", 0))
        for r in rows
    )
    content_filters = sum(1 for r in rows if "provider_content_filter" in (r.get("errors") or []))
    truncations = sum(1 for r in rows if "output_truncated_length" in (r.get("errors") or []))
    return {
        "steps": len(rows),
        "valid_steps": valid,
        "valid_rate": (valid / len(rows)) if rows else 0.0,
        "required_op_misses": required_misses,
        "semantic_rejections": semantic_rejections,
        "content_filters": content_filters,
        "truncations": truncations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4 OLMo-3 evaluation scaffold")
    parser.add_argument("--model", default="allenai/olmo-3.1-32b-instruct")
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--max-steps-override", type=int, default=0, help="0 uses task default max_steps")
    parser.add_argument("--seed-base", type=int, default=5000)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=4000)
    parser.add_argument("--timeout", type=float, default=35.0)
    parser.add_argument("--request-hard-timeout", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--openrouter-api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--openrouter-base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--openrouter-app", default="pichay-phase4")
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/cognitive_transactions/phase4"))
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help="Optional subset of task names: baseline_synthetic reduced_guidance minimal_guidance",
    )
    args = parser.parse_args()

    api_key = os.environ.get(args.openrouter_api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing API key env var: {args.openrouter_api_key_env}")

    selected = DEFAULT_TASKS
    if args.tasks:
        wanted = set(args.tasks)
        selected = [t for t in DEFAULT_TASKS if t.name in wanted]
        if not selected:
            raise RuntimeError("No valid tasks selected")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.out_dir / f"phase4_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    run_rows: list[dict[str, Any]] = []

    for task in selected:
        for rep in range(args.replicates):
            seed = args.seed_base + rep
            decode_config = {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_tokens": args.max_tokens,
                "seed": seed,
            }
            print(f"start task={task.name} rep={rep + 1}/{args.replicates}", flush=True)
            effective_max_steps = args.max_steps_override if args.max_steps_override > 0 else task.max_steps
            log_path = harness.run_experiment(
                provider="openrouter",
                base_url=args.openrouter_base_url,
                model=args.model,
                timeout_s=args.timeout,
                out_dir=run_dir,
                decode_config=decode_config,
                mode="treatment",
                proposer="lmstudio",
                teacher_trace_mode=task.teacher_trace_mode,
                openrouter_api_key=api_key,
                max_steps=effective_max_steps,
                openrouter_app_name=f"{args.openrouter_app}:{task.name}:rep{rep+1}",
                request_hard_timeout=args.request_hard_timeout,
                max_retries=args.max_retries,
            )
            summary = _summarize_log(log_path)
            run_rows.append(
                {
                    "task": task.name,
                    "task_description": task.description,
                    "replicate": rep + 1,
                    "seed": seed,
                    "teacher_trace_mode": task.teacher_trace_mode,
                    "max_steps": effective_max_steps,
                    "log_path": str(log_path),
                    **summary,
                }
            )
            print(
                f"done task={task.name} rep={rep + 1} valid={summary['valid_steps']}/{summary['steps']} "
                f"reqmiss={summary['required_op_misses']} semrej={summary['semantic_rejections']}",
                flush=True,
            )

    aggregate: dict[str, dict[str, Any]] = {}
    for row in run_rows:
        key = row["task"]
        a = aggregate.setdefault(
            key,
            {
                "task": key,
                "teacher_trace_mode": row["teacher_trace_mode"],
                "replicates": 0,
                "steps_total": 0,
                "valid_steps_total": 0,
                "required_op_misses_total": 0,
                "semantic_rejections_total": 0,
                "content_filters_total": 0,
                "truncations_total": 0,
            },
        )
        a["replicates"] += 1
        a["steps_total"] += row["steps"]
        a["valid_steps_total"] += row["valid_steps"]
        a["required_op_misses_total"] += row["required_op_misses"]
        a["semantic_rejections_total"] += row["semantic_rejections"]
        a["content_filters_total"] += row["content_filters"]
        a["truncations_total"] += row["truncations"]

    aggregate_rows = []
    for item in aggregate.values():
        steps_total = item["steps_total"]
        item["valid_rate"] = (item["valid_steps_total"] / steps_total) if steps_total else 0.0
        aggregate_rows.append(item)

    run_summary_path = run_dir / "phase4_run_summary.json"
    aggregate_path = run_dir / "phase4_aggregate.json"
    run_summary_path.write_text(json.dumps(run_rows, ensure_ascii=True, indent=2), encoding="utf-8")
    aggregate_path.write_text(json.dumps(sorted(aggregate_rows, key=lambda r: r["task"]), ensure_ascii=True, indent=2), encoding="utf-8")

    manifest = {
        "experiment_type": "phase4",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "tasks": [t.name for t in selected],
        "replicates": args.replicates,
        "seed_base": args.seed_base,
        "provider": "openrouter",
        "openrouter_base_url": args.openrouter_base_url,
        "openrouter_app": args.openrouter_app,
        "openrouter_api_key_env": args.openrouter_api_key_env,
        "timeouts": {
            "timeout": args.timeout,
            "request_hard_timeout": args.request_hard_timeout,
            "max_retries": args.max_retries,
        },
        "decode_config": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
        },
        "paths": {
            "run_summary": str(run_summary_path),
            "aggregate": str(aggregate_path),
        },
        "git": get_git_meta(Path(__file__).resolve().parent.parent),
        "script_hashes": script_hashes(
            [
                Path(__file__).resolve(),
                (Path(__file__).resolve().parent / "cognitive_step_harness.py").resolve(),
                (Path(__file__).resolve().parent / "repro_meta.py").resolve(),
            ]
        ),
    }
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=True, indent=2), encoding="utf-8")

    print(f"wrote {run_summary_path}")
    print(f"wrote {aggregate_path}")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
