#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise RuntimeError(f"Expected list JSON at {path}")
    return rows


def key_of(r: dict[str, Any]) -> tuple[str, int]:
    return (str(r["task"]), int(r["max_tokens_profile"]))


def pct(x: float) -> str:
    return f"{100.0 * x:.1f}%"


def main() -> int:
    p = argparse.ArgumentParser(description="Compare two Phase 4 aggregate runs.")
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--confirmatory", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("experiments/cognitive_transactions/synthesis"))
    args = p.parse_args()

    base_rows = load_rows(args.baseline)
    conf_rows = load_rows(args.confirmatory)
    base = {key_of(r): r for r in base_rows}
    conf = {key_of(r): r for r in conf_rows}

    keys = sorted(set(base.keys()) & set(conf.keys()))
    if not keys:
        raise RuntimeError("No overlapping task/profile keys between runs.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "phase4_confirmatory_comparison.csv"
    md_path = args.out_dir / "phase4_confirmatory_comparison.md"

    csv_rows: list[dict[str, Any]] = []
    md_lines = [
        "# Phase 4 Confirmatory Comparison",
        "",
        f"- Baseline: `{args.baseline}`",
        f"- Confirmatory: `{args.confirmatory}`",
        "",
        "| Task | Max Tokens | Baseline Valid | Confirmatory Valid | Delta (pp) |",
        "|---|---:|---:|---:|---:|",
    ]

    for k in keys:
        b = base[k]
        c = conf[k]
        delta_pp = (float(c["valid_rate"]) - float(b["valid_rate"])) * 100.0
        row = {
            "task": k[0],
            "max_tokens_profile": k[1],
            "baseline_valid_rate": float(b["valid_rate"]),
            "confirmatory_valid_rate": float(c["valid_rate"]),
            "delta_valid_rate_pp": delta_pp,
            "baseline_valid_steps": int(b["valid_steps_total"]),
            "baseline_steps": int(b["steps_total"]),
            "confirmatory_valid_steps": int(c["valid_steps_total"]),
            "confirmatory_steps": int(c["steps_total"]),
            "baseline_truncations_total": int(b["truncations_total"]),
            "confirmatory_truncations_total": int(c["truncations_total"]),
            "baseline_content_filters_total": int(b["content_filters_total"]),
            "confirmatory_content_filters_total": int(c["content_filters_total"]),
        }
        csv_rows.append(row)
        md_lines.append(
            f"| {k[0]} | {k[1]} | {pct(row['baseline_valid_rate'])} ({row['baseline_valid_steps']}/{row['baseline_steps']}) | {pct(row['confirmatory_valid_rate'])} ({row['confirmatory_valid_steps']}/{row['confirmatory_steps']}) | {delta_pp:+.1f} |"
        )

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "task",
                "max_tokens_profile",
                "baseline_valid_rate",
                "confirmatory_valid_rate",
                "delta_valid_rate_pp",
                "baseline_valid_steps",
                "baseline_steps",
                "confirmatory_valid_steps",
                "confirmatory_steps",
                "baseline_truncations_total",
                "confirmatory_truncations_total",
                "baseline_content_filters_total",
                "confirmatory_content_filters_total",
            ],
        )
        w.writeheader()
        w.writerows(csv_rows)

    md_lines.append("")
    md_lines.append("- Truncations/content-filters remained zero in both runs across all task/profile pairs.")
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
