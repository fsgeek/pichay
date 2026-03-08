#!/usr/bin/env python3
"""Check headline paper numbers against a corpus snapshot JSON.

This is intentionally minimal and conservative:
- verifies headline literals if they appear in main.tex
- warns (does not edit) on mismatches
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def find_literal(text: str, pattern: str) -> bool:
    return re.search(pattern, text) is not None


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--paper", type=Path, default=Path("paper/main.tex"))
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument("--strict", action="store_true")
    args = p.parse_args()

    tex = args.paper.read_text(encoding="utf-8")
    snap = json.loads(args.snapshot.read_text(encoding="utf-8"))

    raw_sessions = snap["raw"]["sessions"]
    dedup_sessions = snap["dedup_by_content"]["sessions"]

    findings: list[str] = []

    # If paper explicitly states these values, compare them to snapshot.
    m = re.search(r"corpus comprises\s+([0-9\{\},]+)\s+sessions", tex)
    if m:
        raw = m.group(1).replace("{", "").replace("}", "").replace(",", "")
        try:
            stated = int(raw)
            if stated not in (raw_sessions, dedup_sessions):
                findings.append(
                    f"stated sessions={stated}, snapshot raw={raw_sessions}, dedup={dedup_sessions}"
                )
        except ValueError:
            findings.append(f"could not parse stated sessions literal: {m.group(1)}")

    # Non-inferiority internal consistency: protocol and table should match.
    proto = re.search(r"We select\s+([0-9]+)~sessions", tex)
    table = re.search(r"Non-inferiority evaluation:\s+([0-9]+)~sessions", tex)
    if proto and table and proto.group(1) != table.group(1):
        findings.append(
            f"non-inferiority mismatch: protocol={proto.group(1)} table={table.group(1)}"
        )

    if findings:
        print("Paper number check: FAIL")
        for f in findings:
            print(" -", f)
        return 1 if args.strict else 0

    print("Paper number check: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
