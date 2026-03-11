from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import sys

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import cognitive_step_harness as harness  # type: ignore
from repro_meta import get_git_meta, script_hashes  # type: ignore


def main() -> None:
    out_dir = Path("experiments/cognitive_transactions/artifacts/olmo3_protocol_v1")
    out_dir.mkdir(parents=True, exist_ok=True)

    schema_dir = Path("experiments/cognitive_transactions/schemas")
    input_schema = json.loads((schema_dir / "cognitive_step_v1.input.json").read_text(encoding="utf-8"))
    output_schema = json.loads((schema_dir / "cognitive_step_v1.output.json").read_text(encoding="utf-8"))

    full_trace = harness._synthetic_teacher_trace(level="full")
    reduced_trace = harness._synthetic_teacher_trace(level="reduced")

    system_prompt = harness.SYSTEM_PROMPT

    (out_dir / "system_prompt.txt").write_text(system_prompt, encoding="utf-8")
    (out_dir / "teacher_trace_full.json").write_text(
        json.dumps(full_trace, ensure_ascii=True, indent=2), encoding="utf-8"
    )
    (out_dir / "teacher_trace_reduced.json").write_text(
        json.dumps(reduced_trace, ensure_ascii=True, indent=2), encoding="utf-8"
    )
    (out_dir / "input_schema.json").write_text(
        json.dumps(input_schema, ensure_ascii=True, indent=2), encoding="utf-8"
    )
    (out_dir / "output_schema.json").write_text(
        json.dumps(output_schema, ensure_ascii=True, indent=2), encoding="utf-8"
    )

    manifest = {
        "artifact": "olmo3_protocol_v1",
        "artifact_version": "v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_primary": "allenai/olmo-3.1-32b-instruct",
        "protocol": {
            "step_type": "cognitive_step_v1",
            "teacher_trace_variants": ["full", "reduced"],
            "assertions": [
                "require_non_halt",
                "min_committed_actions",
                "require_ops(memory_release,memory_fault)",
                "require_fault_use_recent",
                "require_external_output",
            ],
        },
        "files": {
            "system_prompt": str(out_dir / "system_prompt.txt"),
            "teacher_trace_full": str(out_dir / "teacher_trace_full.json"),
            "teacher_trace_reduced": str(out_dir / "teacher_trace_reduced.json"),
            "input_schema": str(out_dir / "input_schema.json"),
            "output_schema": str(out_dir / "output_schema.json"),
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
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2), encoding="utf-8"
    )

    print(f"wrote {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
