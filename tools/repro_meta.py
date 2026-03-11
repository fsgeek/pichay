from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def get_git_meta(repo_root: Path) -> dict[str, Any]:
    def run(cmd: list[str]) -> str:
        return subprocess.check_output(cmd, cwd=repo_root, text=True).strip()

    try:
        commit = run(["git", "rev-parse", "HEAD"])
    except Exception:
        commit = None

    try:
        dirty = bool(run(["git", "status", "--porcelain"]))
    except Exception:
        dirty = None

    try:
        branch = run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    except Exception:
        branch = None

    return {
        "commit": commit,
        "dirty": dirty,
        "branch": branch,
    }


def script_hashes(paths: list[Path]) -> dict[str, str]:
    out: dict[str, str] = {}
    for p in paths:
        try:
            out[str(p)] = file_sha256(p)
        except Exception:
            out[str(p)] = "<unavailable>"
    return out
