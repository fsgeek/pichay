#!/usr/bin/env python3
"""Reproducible corpus counting for Pichay paper.

Counts Claude Code JSONL sessions across multiple roots and optionally
reports deduplicated totals (by SHA-256 file hash). Also summarizes
Claude Desktop conversations.json as a separate schema.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SessionFile:
    path: Path
    size_bytes: int
    session_type: str


def classify_session(name: str) -> str:
    if name.startswith("agent-acompact"):
        return "compact"
    if name.startswith("agent-aprompt_suggestion"):
        return "prompt_suggestion"
    if name.startswith("agent-"):
        return "subagent"
    if name in ("history.jsonl", "pretty.jsonl", "scratch.jsonl"):
        return "other"
    return "main"


def find_jsonl_sessions(roots: list[Path], min_size: int) -> list[SessionFile]:
    out: list[SessionFile] = []
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*.jsonl"):
            sz = p.stat().st_size
            if sz < min_size:
                continue
            out.append(SessionFile(path=p, size_bytes=sz,
                                   session_type=classify_session(p.name)))
    return out


def dedup_by_content(files: list[SessionFile]) -> list[SessionFile]:
    seen: dict[str, SessionFile] = {}
    for sf in files:
        h = hashlib.sha256(sf.path.read_bytes()).hexdigest()
        if h not in seen:
            seen[h] = sf
    return list(seen.values())


def summarize(files: list[SessionFile]) -> dict:
    counts = Counter(sf.session_type for sf in files)
    return {
        "sessions": len(files),
        "by_type": dict(counts),
        "total_bytes": sum(sf.size_bytes for sf in files),
    }


def summarize_desktop(path: Path) -> dict:
    if not path.exists():
        return {"exists": False}
    arr = json.loads(path.read_text(encoding="utf-8"))
    msgs = sum(len(c.get("chat_messages", [])) for c in arr)
    return {
        "exists": True,
        "conversations": len(arr),
        "total_chat_messages": msgs,
        "first_created_at": arr[0].get("created_at") if arr else None,
        "last_created_at": arr[-1].get("created_at") if arr else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Count corpus sessions")
    parser.add_argument("--root", action="append", type=Path, required=True,
                        help="Claude Code root containing project dirs")
    parser.add_argument("--desktop", type=Path,
                        help="Path to Claude Desktop conversations.json")
    parser.add_argument("--min-size", type=int, default=10000,
                        help="Minimum JSONL file size in bytes")
    parser.add_argument("--json", action="store_true",
                        help="Emit JSON")
    args = parser.parse_args()

    raw = find_jsonl_sessions(args.root, args.min_size)
    dedup = dedup_by_content(raw)

    result = {
        "config": {
            "roots": [str(p) for p in args.root],
            "min_size": args.min_size,
        },
        "raw": summarize(raw),
        "dedup_by_content": summarize(dedup),
    }

    if args.desktop is not None:
        result["desktop"] = summarize_desktop(args.desktop)

    if args.json:
        print(json.dumps(result, indent=2))
        return

    print("Raw sessions:", result["raw"]["sessions"], result["raw"]["by_type"])
    print("Dedup sessions:", result["dedup_by_content"]["sessions"],
          result["dedup_by_content"]["by_type"])
    if "desktop" in result:
        print("Desktop:", result["desktop"])


if __name__ == "__main__":
    main()
