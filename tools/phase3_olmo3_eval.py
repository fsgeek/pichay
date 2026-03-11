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


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _summarize(path: Path) -> dict[str, Any]:
    rows = _load_jsonl(path)
    if not rows:
        return {"log_path": str(path), "steps": 0, "valid_steps": 0}

    valid = sum(1 for r in rows if r.get("ok"))
    req_miss = 0
    mem = 0
    sem = 0
    trunc = 0
    cfilter = 0
    http_err = 0
    fault_checks = 0
    fault_fails = 0

    for r in rows:
        sb = r.get("scoreboard", {}) or {}
        req_miss += len(sb.get("required_ops_missing", []))
        mem += int(sb.get("memory_action_count", 0))
        sem += int(sb.get("semantic_rejection_count", 0))
        errs = r.get("errors", []) or []
        if "output_truncated_length" in errs:
            trunc += 1
        if "provider_content_filter" in errs:
            cfilter += 1
        if any(str(e).startswith("http_") for e in errs):
            http_err += 1
        checks = r.get("input_step", {}).get("task_assertions", {}).get("require_fault_use_recent")
        if checks:
            fault_checks += 1
            if any("faulted unit not used" in str(e) for e in (r.get("task_assertion_errors") or [])):
                fault_fails += 1

    r0 = rows[0]
    return {
        "model": r0.get("model"),
        "condition": r0.get("teacher_trace_mode"),
        "steps": len(rows),
        "valid_steps": valid,
        "valid_rate": valid / len(rows) if rows else 0.0,
        "required_op_misses": req_miss,
        "memory_action_commits": mem,
        "semantic_rejections": sem,
        "output_truncated_count": trunc,
        "content_filter_count": cfilter,
        "http_error_steps": http_err,
        "fault_use_checks": fault_checks,
        "fault_use_failures": fault_fails,
        "fault_use_pass_rate": ((fault_checks - fault_fails) / fault_checks) if fault_checks else None,
        "log_path": str(path),
    }


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[r["condition"]].append(r)

    out: list[dict[str, Any]] = []
    for cond, vals in sorted(groups.items()):
        rep = len(vals)
        steps = sum(v["steps"] for v in vals)
        valid = sum(v["valid_steps"] for v in vals)
        req_miss = sum(v["required_op_misses"] for v in vals)
        mem = sum(v["memory_action_commits"] for v in vals)
        sem = sum(v["semantic_rejections"] for v in vals)
        trunc = sum(v["output_truncated_count"] for v in vals)
        cfilter = sum(v["content_filter_count"] for v in vals)
        http_err = sum(v["http_error_steps"] for v in vals)
        fault_checks = sum(v["fault_use_checks"] for v in vals)
        fault_fails = sum(v["fault_use_failures"] for v in vals)
        out.append(
            {
                "condition": cond,
                "replicates": rep,
                "steps_total": steps,
                "valid_steps_total": valid,
                "valid_rate": (valid / steps) if steps else 0.0,
                "required_op_misses_total": req_miss,
                "memory_action_commits_total": mem,
                "semantic_rejections_total": sem,
                "output_truncated_total": trunc,
                "content_filter_total": cfilter,
                "http_error_steps_total": http_err,
                "fault_use_checks_total": fault_checks,
                "fault_use_failures_total": fault_fails,
                "fault_use_pass_rate": ((fault_checks - fault_fails) / fault_checks) if fault_checks else None,
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 OLMo-3 synthetic robustness + reduced-guidance ablation")
    parser.add_argument("--model", default="allenai/olmo-3.1-32b-instruct")
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=4000)
    parser.add_argument("--timeout", type=float, default=35.0)
    parser.add_argument("--request-hard-timeout", type=float, default=120.0)
    parser.add_argument("--seed-base", type=int, default=2000)
    parser.add_argument("--openrouter-api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--openrouter-base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--openrouter-app", default="pichay-phase3")
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/cognitive_transactions/phase3"))
    args = parser.parse_args()

    api_key = os.environ.get(args.openrouter_api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing API key env var: {args.openrouter_api_key_env}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.out_dir / f"phase3_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    conditions = [
        ("synthetic", harness._synthetic_teacher_trace(level="full")),
        ("synthetic_reduced", harness._synthetic_teacher_trace(level="reduced")),
    ]

    run_rows: list[dict[str, Any]] = []
    for cond_name, trace in conditions:
        for rep in range(args.replicates):
            seed = args.seed_base + rep
            decode_config = {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_tokens": args.max_tokens,
                "seed": seed,
            }
            print(f"start condition={cond_name} rep={rep+1}/{args.replicates}", flush=True)
            log_path = harness.run_experiment(
                provider="openrouter",
                base_url=args.openrouter_base_url,
                model=args.model,
                timeout_s=args.timeout,
                out_dir=run_dir,
                decode_config=decode_config,
                mode="treatment",
                proposer="lmstudio",
                teacher_trace_mode=cond_name,
                openrouter_api_key=api_key,
                max_steps=args.max_steps,
                openrouter_app_name=f"{args.openrouter_app}:treatment:{cond_name}:rep{rep+1}",
                request_hard_timeout=args.request_hard_timeout,
                teacher_trace_override=trace,
            )
            row = _summarize(log_path)
            row["replicate"] = rep + 1
            run_rows.append(row)
            print(
                f"done condition={cond_name} rep={rep+1} valid={row['valid_steps']}/{row['steps']} "
                f"reqmiss={row['required_op_misses']} fault_fail={row['fault_use_failures']}",
                flush=True,
            )

    agg = _aggregate(run_rows)

    run_json = run_dir / "phase3_run_summary.json"
    agg_json = run_dir / "phase3_aggregate.json"
    csv_path = run_dir / "phase3_aggregate.csv"

    run_json.write_text(json.dumps(run_rows, ensure_ascii=True, indent=2), encoding="utf-8")
    agg_json.write_text(json.dumps(agg, ensure_ascii=True, indent=2), encoding="utf-8")

    if agg:
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(agg[0].keys()))
            writer.writeheader()
            writer.writerows(agg)

    # Explicit ablation delta table.
    by_cond = {r["condition"]: r for r in agg}
    full = by_cond.get("synthetic")
    reduced = by_cond.get("synthetic_reduced")
    delta = None
    if full and reduced:
        delta = {
            "valid_rate_delta_reduced_minus_full": reduced["valid_rate"] - full["valid_rate"],
            "required_miss_delta_reduced_minus_full": reduced["required_op_misses_total"] - full["required_op_misses_total"],
            "fault_use_pass_delta_reduced_minus_full": (
                (reduced["fault_use_pass_rate"] or 0.0) - (full["fault_use_pass_rate"] or 0.0)
            ),
            "truncation_delta_reduced_minus_full": reduced["output_truncated_total"] - full["output_truncated_total"],
        }

    manifest = {
        "experiment_type": "phase3",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "protocol_artifact": "olmo3_protocol_v1",
        "harness_version": "olmo3_protocol_v1",
        "replicates": args.replicates,
        "conditions": ["synthetic", "synthetic_reduced"],
        "max_steps": args.max_steps,
        "provider": "openrouter",
        "openrouter_base_url": args.openrouter_base_url,
        "openrouter_app": args.openrouter_app,
        "openrouter_api_key_env": args.openrouter_api_key_env,
        "timeout": args.timeout,
        "request_hard_timeout": args.request_hard_timeout,
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
        "paths": {
            "run_summary": str(run_json),
            "aggregate": str(agg_json),
            "aggregate_csv": str(csv_path),
        },
        "ablation_delta": delta,
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )

    print(f"wrote {run_json}")
    print(f"wrote {agg_json}")
    print(f"wrote {csv_path}")
    print(f"wrote {run_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
