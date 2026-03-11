from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def _run(cmd: list[str], dry_run: bool) -> None:
    print("command:", " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def replay_phase2(manifest: dict, dry_run: bool) -> None:
    decode = manifest.get("decode_config", {})
    cmd = [
        "python3",
        "tools/phase2_candidate_analysis.py",
        "--provider",
        manifest.get("provider", "openrouter"),
        "--openrouter-api-key-env",
        manifest.get("openrouter_api_key_env", "OPENROUTER_API_KEY"),
        "--openrouter-base-url",
        manifest.get("openrouter_base_url", "https://openrouter.ai/api/v1"),
        "--openrouter-app",
        manifest.get("openrouter_app", "pichay-phase2"),
        "--replicates",
        str(manifest.get("replicates", 3)),
        "--mode",
        manifest.get("mode", "treatment"),
        "--max-steps",
        str(manifest.get("max_steps", 5)),
        "--temperature",
        str(decode.get("temperature", 0.0)),
        "--top-p",
        str(decode.get("top_p", 1.0)),
        "--max-tokens",
        str(decode.get("max_tokens", 4000)),
        "--timeout",
        str(manifest.get("timeout", 35.0)),
        "--request-hard-timeout",
        str(manifest.get("request_hard_timeout", 120.0)),
        "--max-retries",
        str(manifest.get("max_retries", 3)),
        "--seed-base",
        str(decode.get("seed_base", 1000)),
    ]

    models = manifest.get("models", []) or []
    if models:
        cmd.extend(["--models", *[str(x) for x in models]])

    traces = manifest.get("teacher_traces", []) or []
    if traces:
        cmd.extend(["--teacher-traces", *[str(x) for x in traces]])

    out_dir = Path(manifest.get("run_dir", "experiments/cognitive_transactions/phase2")).parent
    cmd.extend(["--out-dir", str(out_dir)])
    _run(cmd, dry_run)


def replay_phase3(manifest: dict, dry_run: bool) -> None:
    decode = manifest.get("decode_config", {})
    cmd = [
        "python3",
        "tools/phase3_olmo3_eval.py",
        "--model",
        manifest.get("model", "allenai/olmo-3.1-32b-instruct"),
        "--replicates",
        str(manifest.get("replicates", 3)),
        "--max-steps",
        str(manifest.get("max_steps", 5)),
        "--temperature",
        str(decode.get("temperature", 0.0)),
        "--top-p",
        str(decode.get("top_p", 1.0)),
        "--max-tokens",
        str(decode.get("max_tokens", 4000)),
        "--timeout",
        str(manifest.get("timeout", 35.0)),
        "--request-hard-timeout",
        str(manifest.get("request_hard_timeout", 120.0)),
        "--seed-base",
        str(decode.get("seed_base", 2000)),
        "--openrouter-api-key-env",
        manifest.get("openrouter_api_key_env", "OPENROUTER_API_KEY"),
        "--openrouter-base-url",
        manifest.get("openrouter_base_url", "https://openrouter.ai/api/v1"),
        "--openrouter-app",
        manifest.get("openrouter_app", "pichay-phase3"),
    ]
    out_dir = Path(manifest.get("paths", {}).get("run_summary", "experiments/cognitive_transactions/phase3")).parent.parent
    cmd.extend(["--out-dir", str(out_dir)])
    _run(cmd, dry_run)


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay experiment from manifest")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    exp = manifest.get("experiment_type")
    if not exp:
        # Backward-compat inference
        if "models" in manifest and "teacher_traces" in manifest:
            exp = "phase2"
        elif "model" in manifest and "conditions" in manifest:
            exp = "phase3"
    if exp == "phase2":
        replay_phase2(manifest, args.dry_run)
    elif exp == "phase3":
        replay_phase3(manifest, args.dry_run)
    else:
        raise RuntimeError(f"Unsupported or missing experiment_type: {exp}")


if __name__ == "__main__":
    main()
