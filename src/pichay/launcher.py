from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass


@dataclass
class LaunchSpec:
    mode: str
    port: int
    extra_args: list[str]

    def command(self) -> list[str]:
        if self.mode == "claude":
            return ["claude", *self.extra_args]
        if self.mode == "codex":
            return ["codex", *self.extra_args]
        if self.mode == "gemini":
            raise RuntimeError(
                "Gemini adapter is not enabled in v1. Use --claude or --codex."
            )
        raise RuntimeError(f"unknown launch mode: {self.mode}")

    def env(self) -> dict[str, str]:
        env = os.environ.copy()
        base = f"http://127.0.0.1:{self.port}"
        if self.mode == "claude":
            env["ANTHROPIC_BASE_URL"] = base
        elif self.mode == "codex":
            env["OPENAI_BASE_URL"] = base
            env["OPENAI_API_BASE"] = base
        return env


def launch(spec: LaunchSpec) -> int:
    cmd = spec.command()
    env = spec.env()
    print(f"Launching: {' '.join(cmd)}", file=sys.stderr)
    try:
        proc = subprocess.run(cmd, env=env)
    except FileNotFoundError as e:
        raise RuntimeError(f"command not found: {cmd[0]}") from e
    return proc.returncode
