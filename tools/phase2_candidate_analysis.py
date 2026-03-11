from __future__ import annotations

import argparse
import csv
import json
import os
import platform
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import sys

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import cognitive_step_harness as harness  # type: ignore
from repro_meta import get_git_meta, script_hashes  # type: ignore


DEFAULT_MODELS = [
    "inception/mercury-2",
    "moonshotai/kimi-k2.5",
    "meta-llama/llama-4-maverick",
    "allenai/olmo-3.1-32b-instruct",
]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _summarize_run(path: Path) -> dict[str, Any]:
    rows = _load_jsonl(path)
    if not rows:
        return {
            "log_path": str(path),
            "steps": 0,
            "valid_steps": 0,
        }

    valid_steps = sum(1 for r in rows if r.get("ok"))
    required_miss = 0
    memory_commits = 0
    halt_illegal = 0
    semantic_rejections = 0
    content_filters = 0
    http_errors = 0
    trunc = 0
    fault_use_checks = 0
    fault_use_failures = 0

    for r in rows:
        sb = r.get("scoreboard", {}) or {}
        required_miss += len(sb.get("required_ops_missing", []))
        memory_commits += int(sb.get("memory_action_count", 0))
        if not sb.get("halt_legal", True):
            halt_illegal += 1
        semantic_rejections += int(sb.get("semantic_rejection_count", 0))

        errs = r.get("errors", []) or []
        if "provider_content_filter" in errs:
            content_filters += 1
        if "output_truncated_length" in errs:
            trunc += 1
        if any(str(e).startswith("http_") for e in errs):
            http_errors += 1

        ta = r.get("task_assertion_errors", []) or []
        if any("faulted unit not used" in str(e) for e in ta):
            fault_use_checks += 1
            fault_use_failures += 1
        elif any((r.get("input_step", {}).get("task_assertions", {}) or {}).get("require_fault_use_recent") for _ in [0]):
            fault_use_checks += 1

    row0 = rows[0]
    return {
        "provider": row0.get("provider", "openrouter"),
        "model": row0.get("model"),
        "teacher_trace_mode": row0.get("teacher_trace_mode"),
        "mode": row0.get("mode"),
        "decode_config": row0.get("decode_config", {}),
        "openrouter_app": row0.get("openrouter_app"),
        "log_path": str(path),
        "steps": len(rows),
        "valid_steps": valid_steps,
        "required_op_misses": required_miss,
        "memory_action_commits": memory_commits,
        "halt_illegal_count": halt_illegal,
        "semantic_rejection_count": semantic_rejections,
        "content_filter_count": content_filters,
        "output_truncated_count": trunc,
        "http_error_steps": http_errors,
        "fault_use_checks": fault_use_checks,
        "fault_use_failures": fault_use_failures,
    }


def _aggregate(run_summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in run_summaries:
        buckets[(row["model"], row["teacher_trace_mode"])].append(row)

    out: list[dict[str, Any]] = []
    for (model, trace), rows in sorted(buckets.items()):
        n = len(rows)
        steps = sum(r["steps"] for r in rows)
        valid = sum(r["valid_steps"] for r in rows)
        req_miss = sum(r["required_op_misses"] for r in rows)
        mem = sum(r["memory_action_commits"] for r in rows)
        sem = sum(r["semantic_rejection_count"] for r in rows)
        cfilter = sum(r["content_filter_count"] for r in rows)
        trunc = sum(r["output_truncated_count"] for r in rows)
        http = sum(r["http_error_steps"] for r in rows)
        fault_checks = sum(r["fault_use_checks"] for r in rows)
        fault_fails = sum(r["fault_use_failures"] for r in rows)

        out.append(
            {
                "model": model,
                "teacher_trace_mode": trace,
                "replicates": n,
                "steps_total": steps,
                "valid_steps_total": valid,
                "valid_rate": (valid / steps) if steps else 0.0,
                "required_op_misses_total": req_miss,
                "memory_action_commits_total": mem,
                "semantic_rejection_total": sem,
                "content_filter_total": cfilter,
                "output_truncated_total": trunc,
                "http_error_steps_total": http,
                "fault_use_checks_total": fault_checks,
                "fault_use_failures_total": fault_fails,
                "fault_use_pass_rate": ((fault_checks - fault_fails) / fault_checks) if fault_checks else None,
            }
        )

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 candidate analysis runner")
    parser.add_argument("--provider", choices=["openrouter"], default="openrouter")
    parser.add_argument("--openrouter-api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--openrouter-base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--openrouter-app", default="pichay-phase2")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--teacher-traces", nargs="+", choices=["none", "synthetic"], default=["none", "synthetic"])
    parser.add_argument("--mode", choices=["treatment"], default="treatment")
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=4000)
    parser.add_argument("--timeout", type=float, default=35.0)
    parser.add_argument("--request-hard-timeout", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--seed-base", type=int, default=1000)
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/cognitive_transactions/phase2"))
    args = parser.parse_args()

    api_key = os.environ.get(args.openrouter_api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing API key env var: {args.openrouter_api_key_env}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.out_dir / f"phase2_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    run_rows: list[dict[str, Any]] = []

    for model in args.models:
        for trace in args.teacher_traces:
            for rep in range(args.replicates):
                seed = args.seed_base + rep
                decode_config = {
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "max_tokens": args.max_tokens,
                    "seed": seed,
                }
                print(f"start model={model} trace={trace} rep={rep+1}/{args.replicates}", flush=True)
                log_path = harness.run_experiment(
                    provider="openrouter",
                    base_url=args.openrouter_base_url,
                    model=model,
                    timeout_s=args.timeout,
                    out_dir=run_dir,
                    decode_config=decode_config,
                    mode=args.mode,
                    proposer="lmstudio",
                    teacher_trace_mode=trace,
                    openrouter_api_key=api_key,
                    max_steps=args.max_steps,
                    openrouter_app_name=f"{args.openrouter_app}:{args.mode}:{trace}:rep{rep+1}",
                    request_hard_timeout=args.request_hard_timeout,
                    max_retries=args.max_retries,
                )
                summary = _summarize_run(log_path)
                summary["replicate"] = rep + 1
                run_rows.append(summary)
                print(
                    f"done model={model} trace={trace} rep={rep+1} valid={summary.get('valid_steps')}/{summary.get('steps')} "
                    f"reqmiss={summary.get('required_op_misses')} fault_use_fail={summary.get('fault_use_failures')}",
                    flush=True,
                )

    aggregate = _aggregate(run_rows)

    run_json = run_dir / "phase2_run_summary.json"
    agg_json = run_dir / "phase2_aggregate.json"
    csv_path = run_dir / "phase2_aggregate.csv"

    with run_json.open("w", encoding="utf-8") as f:
        json.dump(run_rows, f, ensure_ascii=True, indent=2)
    with agg_json.open("w", encoding="utf-8") as f:
        json.dump(aggregate, f, ensure_ascii=True, indent=2)

    if aggregate:
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(aggregate[0].keys()))
            writer.writeheader()
            writer.writerows(aggregate)

    manifest = {
        "experiment_type": "phase2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "protocol_artifact": "olmo3_protocol_v1",
        "harness_version": "olmo3_protocol_v1",
        "models": args.models,
        "replicates": args.replicates,
        "teacher_traces": args.teacher_traces,
        "mode": args.mode,
        "max_steps": args.max_steps,
        "provider": args.provider,
        "openrouter_base_url": args.openrouter_base_url,
        "openrouter_app": args.openrouter_app,
        "openrouter_api_key_env": args.openrouter_api_key_env,
        "timeout": args.timeout,
        "request_hard_timeout": args.request_hard_timeout,
        "max_retries": args.max_retries,
        "decode_config": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "seed_base": args.seed_base,
        },
        "runtime": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
        "git": get_git_meta(Path(__file__).resolve().parent.parent),
        "script_hashes": script_hashes(
            [
                Path(__file__).resolve(),
                (Path(__file__).resolve().parent / "cognitive_step_harness.py").resolve(),
                (Path(__file__).resolve().parent / "repro_meta.py").resolve(),
            ]
        ),
        "run_summary_path": str(run_json),
        "aggregate_json_path": str(agg_json),
        "aggregate_csv_path": str(csv_path),
    }
    with (run_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=True, indent=2)

    print(f"wrote {run_json}")
    print(f"wrote {agg_json}")
    print(f"wrote {csv_path}")
    print(f"wrote {run_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
