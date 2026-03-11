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


ALLOWED_ACTIONS = [
    "create_unit",
    "update_unit",
    "memory_release",
    "memory_fault",
    "memory_curate",
    "emit_text",
    "halt",
]


@dataclass
class Phase4Task:
    name: str
    description: str


def _base_workspace() -> dict[str, Any]:
    return {
        "resident_units": [
            {"id": "unit-01", "type": "goal", "content": "Deliver robust fix proposal"},
            {"id": "unit-02", "type": "evidence", "content": "Profiler indicates allocator regressions"},
            {"id": "unit-03", "type": "tool_result", "content": "Benchmark table v4"},
            {"id": "unit-04", "type": "summary", "content": "Prior mitigation attempts"},
        ],
        "evicted_handles": ["unit-09", "unit-10"],
        "focus": ["unit-01", "unit-02"],
        "memory_budget_units": 3,
    }


def _step(events: list[dict[str, Any]], goal: str, assertions: dict[str, Any], workspace: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "workspace": workspace,
        "events": events,
        "policies": {
            "preserve_unresolved": True,
            "evict_tool_output_aggressively": True,
            "prefer_faults_over_compaction": True,
        },
        "allowed_actions": ALLOWED_ACTIONS,
        "goal_or_focus": goal,
        "policy_header": {"experiment": "phase4_realism"},
        "task_assertions": assertions,
    }


def scenario_multitask_benchmark() -> list[dict[str, Any]]:
    ws = _base_workspace()
    return [
        _step([{"type": "human_input", "content": "Plan a fix and testing strategy."}], "Create initial plan units.", {"require_non_halt": True, "min_committed_actions": 1}, ws),
        _step([{"type": "tool_result", "content": "New benchmark regression in allocator-heavy workload."}], "Incorporate tool evidence.", {"require_non_halt": True, "min_committed_actions": 1}),
        _step([{"type": "scheduler_wakeup", "content": "Memory pressure high; release low-value resident units."}], "Relieve pressure.", {"require_non_halt": True, "require_ops": ["memory_release"]}),
        _step([{"type": "human_input", "content": "Need historical baseline from unit-09 to validate claim."}], "Recover baseline context.", {"require_non_halt": True, "require_ops": ["memory_fault"]}),
        _step([{"type": "human_input", "content": "Reconcile contradiction between benchmark table and profiler trace."}], "Commit contradiction repair based on restored memory.", {"require_non_halt": True, "min_committed_actions": 1, "require_fault_use_recent": True}),
        _step([{"type": "human_input", "content": "Produce concise external recommendation."}], "Emit user-facing result and halt.", {"require_external_output": True}),
    ]


def scenario_long_horizon() -> list[dict[str, Any]]:
    ws = _base_workspace()
    steps = [
        _step([{"type": "human_input", "content": "Start long-horizon investigation."}], "Initialize hypothesis graph.", {"require_non_halt": True, "min_committed_actions": 1}, ws),
    ]
    for i in range(2, 10):
        if i in (4, 7):
            steps.append(_step([{"type": "scheduler_wakeup", "content": f"Pressure cycle {i}; release low-value memory."}], "Release memory under pressure.", {"require_non_halt": True, "require_ops": ["memory_release"]}))
        elif i in (5, 8):
            steps.append(_step([{"type": "human_input", "content": f"Need evicted context to continue cycle {i}."}], "Fault needed context.", {"require_non_halt": True, "require_ops": ["memory_fault"]}))
        elif i in (6, 9):
            steps.append(_step([{"type": "tool_result", "content": f"New conflicting evidence at step {i}."}], "Update state with conflict resolution.", {"require_non_halt": True, "min_committed_actions": 1, "require_fault_use_recent": True}))
        else:
            steps.append(_step([{"type": "tool_result", "content": f"Incremental evidence step {i}."}], "Maintain coherent progression.", {"require_non_halt": True, "min_committed_actions": 1}))
    steps.append(_step([{"type": "human_input", "content": "Finalize long-horizon summary."}], "Emit external summary.", {"require_external_output": True}))
    return steps


def scenario_failure_recovery() -> list[dict[str, Any]]:
    ws = _base_workspace()
    return [
        _step([{"type": "human_input", "content": "Begin repair-oriented run."}], "Build first-pass plan.", {"require_non_halt": True, "min_committed_actions": 1}, ws),
        _step([{"type": "scheduler_wakeup", "content": "Pressure triggered."}], "Release memory legally.", {"require_non_halt": True, "require_ops": ["memory_release"]}),
        _step([{"type": "human_input", "content": "Need evicted baseline unit-09."}], "Fault baseline.", {"require_non_halt": True, "require_ops": ["memory_fault"]}),
        _step([
            {"type": "validator_feedback", "content": "Previous proposal had an invalid memory op. Repair and continue."}
        ], "Demonstrate recovery from failure signal.", {"require_non_halt": True, "min_committed_actions": 1, "require_fault_use_recent": True}),
        _step([{"type": "human_input", "content": "Deliver recovered final output."}], "Emit corrected external answer.", {"require_external_output": True}),
    ]


TASK_BUILDERS: dict[str, tuple[Phase4Task, Any]] = {
    "multitask_benchmark": (Phase4Task("multitask_benchmark", "Multi-source task progression with contradiction reconciliation."), scenario_multitask_benchmark),
    "long_horizon_durability": (Phase4Task("long_horizon_durability", "Longer chain with multiple pressure/fault cycles."), scenario_long_horizon),
    "failure_recovery": (Phase4Task("failure_recovery", "Explicit recovery after validator-like failure signal."), scenario_failure_recovery),
}


def _token_totals(usage: dict[str, Any]) -> tuple[int, int, int]:
    prompt = int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0)
    completion = int(usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0)
    total = int(usage.get("total_tokens", prompt + completion) or (prompt + completion))
    return prompt, completion, total


def _summarize_log(path: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    valid = sum(1 for r in rows if r.get("ok"))
    required_misses = sum(len((r.get("scoreboard") or {}).get("required_ops_missing", [])) for r in rows)
    semantic_rejections = sum(int((r.get("scoreboard") or {}).get("semantic_rejection_count", 0)) for r in rows)
    content_filters = sum(1 for r in rows if "provider_content_filter" in (r.get("errors") or []))
    truncations = sum(1 for r in rows if "output_truncated_length" in (r.get("errors") or []))
    latency_total = sum(float(r.get("latency_s", 0.0) or 0.0) for r in rows)
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    for r in rows:
        p, c, t = _token_totals(r.get("usage", {}) or {})
        prompt_tokens += p
        completion_tokens += c
        total_tokens += t
    return {
        "steps": len(rows),
        "valid_steps": valid,
        "valid_rate": (valid / len(rows)) if rows else 0.0,
        "required_op_misses": required_misses,
        "semantic_rejections": semantic_rejections,
        "content_filters": content_filters,
        "truncations": truncations,
        "latency_total_s": latency_total,
        "latency_mean_s": (latency_total / len(rows)) if rows else 0.0,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4 OLMo-3 task realism evaluation")
    parser.add_argument("--model", default="allenai/olmo-3.1-32b-instruct")
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--seed-base", type=int, default=5000)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-token-profiles", nargs="+", type=int, default=[4000, 2000])
    parser.add_argument("--timeout", type=float, default=35.0)
    parser.add_argument("--request-hard-timeout", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--openrouter-api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--openrouter-base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--openrouter-app", default="pichay-phase4")
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/cognitive_transactions/phase4"))
    parser.add_argument("--tasks", nargs="*", default=None, help="Subset: multitask_benchmark long_horizon_durability failure_recovery")
    args = parser.parse_args()

    api_key = os.environ.get(args.openrouter_api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing API key env var: {args.openrouter_api_key_env}")

    task_names = list(TASK_BUILDERS.keys()) if not args.tasks else args.tasks
    for name in task_names:
        if name not in TASK_BUILDERS:
            raise RuntimeError(f"Unknown task: {name}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.out_dir / f"phase4_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    run_rows: list[dict[str, Any]] = []

    for max_tokens in args.max_token_profiles:
        for task_name in task_names:
            task_meta, builder = TASK_BUILDERS[task_name]
            scenario = builder()
            for rep in range(args.replicates):
                seed = args.seed_base + rep
                decode_config = {
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "max_tokens": max_tokens,
                    "seed": seed,
                }
                print(f"start task={task_name} max_tokens={max_tokens} rep={rep+1}/{args.replicates}", flush=True)
                log_path = harness.run_experiment(
                    provider="openrouter",
                    base_url=args.openrouter_base_url,
                    model=args.model,
                    timeout_s=args.timeout,
                    out_dir=run_dir,
                    decode_config=decode_config,
                    mode="treatment",
                    proposer="lmstudio",
                    teacher_trace_mode="synthetic",
                    openrouter_api_key=api_key,
                    max_steps=len(scenario),
                    openrouter_app_name=f"{args.openrouter_app}:{task_name}:max{max_tokens}:rep{rep+1}",
                    request_hard_timeout=args.request_hard_timeout,
                    max_retries=args.max_retries,
                    scenario_override=scenario,
                )
                summary = _summarize_log(log_path)
                run_rows.append(
                    {
                        "task": task_name,
                        "task_description": task_meta.description,
                        "replicate": rep + 1,
                        "seed": seed,
                        "max_tokens_profile": max_tokens,
                        "teacher_trace_mode": "synthetic",
                        "log_path": str(log_path),
                        **summary,
                    }
                )
                print(
                    f"done task={task_name} max_tokens={max_tokens} rep={rep+1} "
                    f"valid={summary['valid_steps']}/{summary['steps']} reqmiss={summary['required_op_misses']} "
                    f"semrej={summary['semantic_rejections']} tok={summary['total_tokens']}",
                    flush=True,
                )

    # Aggregate by task x max_tokens_profile
    aggregate_map: dict[tuple[str, int], dict[str, Any]] = {}
    for row in run_rows:
        key = (row["task"], row["max_tokens_profile"])
        a = aggregate_map.setdefault(
            key,
            {
                "task": row["task"],
                "task_description": row["task_description"],
                "max_tokens_profile": row["max_tokens_profile"],
                "replicates": 0,
                "steps_total": 0,
                "valid_steps_total": 0,
                "required_op_misses_total": 0,
                "semantic_rejections_total": 0,
                "content_filters_total": 0,
                "truncations_total": 0,
                "latency_total_s": 0.0,
                "prompt_tokens_total": 0,
                "completion_tokens_total": 0,
                "total_tokens_total": 0,
            },
        )
        a["replicates"] += 1
        a["steps_total"] += row["steps"]
        a["valid_steps_total"] += row["valid_steps"]
        a["required_op_misses_total"] += row["required_op_misses"]
        a["semantic_rejections_total"] += row["semantic_rejections"]
        a["content_filters_total"] += row["content_filters"]
        a["truncations_total"] += row["truncations"]
        a["latency_total_s"] += row["latency_total_s"]
        a["prompt_tokens_total"] += row["prompt_tokens"]
        a["completion_tokens_total"] += row["completion_tokens"]
        a["total_tokens_total"] += row["total_tokens"]

    aggregate_rows = []
    for a in aggregate_map.values():
        steps = a["steps_total"]
        a["valid_rate"] = (a["valid_steps_total"] / steps) if steps else 0.0
        a["latency_mean_s"] = (a["latency_total_s"] / steps) if steps else 0.0
        a["tokens_per_valid_step"] = (
            a["total_tokens_total"] / a["valid_steps_total"] if a["valid_steps_total"] else None
        )
        aggregate_rows.append(a)

    run_summary_path = run_dir / "phase4_run_summary.json"
    aggregate_path = run_dir / "phase4_aggregate.json"
    run_summary_path.write_text(json.dumps(run_rows, ensure_ascii=True, indent=2), encoding="utf-8")
    aggregate_path.write_text(json.dumps(sorted(aggregate_rows, key=lambda r: (r["task"], r["max_tokens_profile"])), ensure_ascii=True, indent=2), encoding="utf-8")

    manifest = {
        "experiment_type": "phase4",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "tasks": task_names,
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
            "max_token_profiles": args.max_token_profiles,
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
