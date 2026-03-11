from __future__ import annotations

import argparse
import json
import os
import re
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx


SYSTEM_PROMPT = """You are a cognitive transaction engine.
Return ONLY JSON, no markdown, no prose outside JSON.

You must produce an object with keys:
- mutations: list
- memory: list
- internal_only: list
- external_outputs: list
- next_focus: list
- step_kind: one of [\"external\", \"internal\", \"fault_restore\", \"halt\"]

Every item in mutations and memory must include an 'op' string.
You must obey allowed_actions exactly. Do not invent operations.
Use stable ids like unit-001, unit-002.
When a mutation depends on prior evidence, include \"source_ids\": [\"unit-id\", ...].
"""


FAIL_SCHEMA = "schema_invalid"
FAIL_POLICY = "policy_invalid"
FAIL_GRAPH = "graph_invalid"
FAIL_MEMORY = "memory_invalid"
FAIL_DEP = "dependency_invalid"

STEP_KINDS = {"external", "internal", "fault_restore", "halt"}


@dataclass
class RunResult:
    ok: bool
    latency_s: float
    raw_content: str
    parsed: dict[str, Any] | None
    errors: list[str]
    usage: dict[str, Any]
    request_payload: dict[str, Any]
    response_status: int
    response_json: dict[str, Any] | None
    response_text: str
    started_at: str
    completed_at: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _extract_json(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if not text:
        return None
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        return None
    return None


def _build_contract(step: dict[str, Any]) -> dict[str, Any]:
    """Build an explicit legality contract for the current step."""
    assertions = step.get("task_assertions", {}) if isinstance(step.get("task_assertions"), dict) else {}
    workspace = step.get("workspace", {}) if isinstance(step.get("workspace"), dict) else {}
    resident_units = workspace.get("resident_units", []) if isinstance(workspace.get("resident_units"), list) else []
    evicted_handles = workspace.get("evicted_handles", []) if isinstance(workspace.get("evicted_handles"), list) else []
    resident_ids = [u.get("id") for u in resident_units if isinstance(u, dict) and isinstance(u.get("id"), str)]
    evicted_ids = [x for x in evicted_handles if isinstance(x, str)]

    return {
        "required_top_level": [
            "mutations",
            "memory",
            "internal_only",
            "external_outputs",
            "next_focus",
            "step_kind",
        ],
        "return_json_only": True,
        "step_constraints": {
            "halt_allowed": not bool(assertions.get("require_non_halt")),
            "required_ops": assertions.get("require_ops", []) if isinstance(assertions.get("require_ops"), list) else [],
            "memory_release_allowed_ids": resident_ids,
            "memory_fault_allowed_ids": evicted_ids,
        },
        "legality_rules": [
            "Do not emit step_kind=halt when halt_allowed is false.",
            "memory_release ids must be chosen only from memory_release_allowed_ids.",
            "memory_fault ids must be chosen only from memory_fault_allowed_ids.",
        ],
    }


def _workspace_unit_ids(workspace: dict[str, Any]) -> set[str]:
    resident = workspace.get("resident_units", [])
    ids: set[str] = set()
    if isinstance(resident, list):
        for unit in resident:
            if isinstance(unit, dict) and isinstance(unit.get("id"), str):
                ids.add(unit["id"])
    return ids


def _workspace_evicted_ids(workspace: dict[str, Any]) -> set[str]:
    evicted = workspace.get("evicted_handles", [])
    out: set[str] = set()
    if isinstance(evicted, list):
        for item in evicted:
            if isinstance(item, str):
                out.add(item)
    return out


def _reject(index: int, op: Any, cls: str, reason: str, bucket: str) -> dict[str, Any]:
    return {
        "bucket": bucket,
        "index": index,
        "op": op,
        "failure_class": cls,
        "reason": reason,
    }


def _validate_shape(output: dict[str, Any]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    required = (
        "mutations",
        "memory",
        "internal_only",
        "external_outputs",
        "next_focus",
        "step_kind",
    )
    for key in required:
        if key not in output:
            failures.append(_reject(-1, key, FAIL_SCHEMA, f"missing key: {key}", "top_level"))

    if failures:
        return failures

    if not isinstance(output["mutations"], list):
        failures.append(_reject(-1, "mutations", FAIL_SCHEMA, "mutations must be list", "top_level"))
    if not isinstance(output["memory"], list):
        failures.append(_reject(-1, "memory", FAIL_SCHEMA, "memory must be list", "top_level"))
    if not isinstance(output["internal_only"], list):
        failures.append(_reject(-1, "internal_only", FAIL_SCHEMA, "internal_only must be list", "top_level"))
    if not isinstance(output["external_outputs"], list):
        failures.append(_reject(-1, "external_outputs", FAIL_SCHEMA, "external_outputs must be list", "top_level"))
    if not isinstance(output["next_focus"], list):
        failures.append(_reject(-1, "next_focus", FAIL_SCHEMA, "next_focus must be list", "top_level"))
    step_kind = output.get("step_kind")
    if not isinstance(step_kind, str) or step_kind not in STEP_KINDS:
        failures.append(_reject(-1, "step_kind", FAIL_SCHEMA, "invalid step_kind", "top_level"))

    return failures


def _validate_actions(
    output: dict[str, Any],
    workspace_before: dict[str, Any],
    allowed_actions: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (accepted_mutations, accepted_memory, rejections)."""
    accepted_mutations: list[dict[str, Any]] = []
    accepted_memory: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []

    pre_units = _workspace_unit_ids(workspace_before)
    pre_evicted = _workspace_evicted_ids(workspace_before)

    def validate_item(item: Any, idx: int, bucket: str) -> dict[str, Any] | None:
        if not isinstance(item, dict):
            rejections.append(_reject(idx, item, FAIL_SCHEMA, f"{bucket}[{idx}] must be object", bucket))
            return None
        op = item.get("op")
        if not isinstance(op, str):
            rejections.append(_reject(idx, item, FAIL_SCHEMA, f"{bucket}[{idx}].op missing or not string", bucket))
            return None
        if op not in allowed_actions:
            rejections.append(_reject(idx, item, FAIL_POLICY, f"{op} not in allowed_actions", bucket))
            return None

        # Graph/dependency/memory checks (v1: no intra-step forward refs)
        if op == "create_unit":
            unit_id = item.get("id")
            if not isinstance(unit_id, str):
                rejections.append(_reject(idx, item, FAIL_SCHEMA, "create_unit requires string id", bucket))
                return None
            if unit_id in pre_units:
                rejections.append(_reject(idx, item, FAIL_GRAPH, f"create_unit id already exists: {unit_id}", bucket))
                return None
        elif op == "update_unit":
            unit_id = item.get("id")
            if not isinstance(unit_id, str):
                rejections.append(_reject(idx, item, FAIL_SCHEMA, "update_unit requires string id", bucket))
                return None
            if unit_id not in pre_units:
                if unit_id in {x.get("id") for x in accepted_mutations if isinstance(x, dict)}:
                    rejections.append(_reject(idx, item, FAIL_DEP, "forward reference to same-step created unit", bucket))
                else:
                    rejections.append(_reject(idx, item, FAIL_GRAPH, f"update references unknown unit: {unit_id}", bucket))
                return None
        elif op == "memory_release":
            ids = item.get("ids")
            if not isinstance(ids, list) or not all(isinstance(x, str) for x in ids):
                rejections.append(_reject(idx, item, FAIL_SCHEMA, "memory_release requires ids: list[str]", bucket))
                return None
            for unit_id in ids:
                if unit_id in {x.get("id") for x in accepted_mutations if isinstance(x, dict)}:
                    rejections.append(_reject(idx, item, FAIL_DEP, "release references same-step created unit", bucket))
                    return None
                if unit_id not in pre_units:
                    rejections.append(_reject(idx, item, FAIL_MEMORY, f"cannot release non-resident unit: {unit_id}", bucket))
                    return None
        elif op == "memory_fault":
            ids = item.get("ids")
            if not isinstance(ids, list) or not all(isinstance(x, str) for x in ids):
                rejections.append(_reject(idx, item, FAIL_SCHEMA, "memory_fault requires ids: list[str]", bucket))
                return None
            for unit_id in ids:
                if unit_id in {x.get("id") for x in accepted_mutations if isinstance(x, dict)}:
                    rejections.append(_reject(idx, item, FAIL_DEP, "fault references same-step created unit", bucket))
                    return None
                if unit_id not in pre_evicted:
                    rejections.append(_reject(idx, item, FAIL_MEMORY, f"cannot fault non-evicted unit: {unit_id}", bucket))
                    return None

        return item

    for idx, item in enumerate(output.get("mutations", [])):
        valid = validate_item(item, idx, "mutations")
        if valid is not None:
            accepted_mutations.append(valid)

    for idx, item in enumerate(output.get("memory", [])):
        valid = validate_item(item, idx, "memory")
        if valid is not None:
            accepted_memory.append(valid)

    return accepted_mutations, accepted_memory, rejections


def _reduce_workspace(
    workspace: dict[str, Any],
    mutations: list[dict[str, Any]],
    memory_ops: list[dict[str, Any]],
    next_focus: list[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Deterministic reducer. Applies validated ops in listed order."""
    next_ws = json.loads(json.dumps(workspace))
    resident = next_ws.setdefault("resident_units", [])
    evicted = next_ws.setdefault("evicted_handles", [])

    by_id = {
        u.get("id"): u
        for u in resident
        if isinstance(u, dict) and isinstance(u.get("id"), str)
    }

    committed: list[dict[str, Any]] = []

    for item in mutations:
        op = item.get("op")
        if op == "create_unit":
            unit_id = item["id"]
            unit = {
                "id": unit_id,
                "type": item.get("type", "note"),
                "content": item.get("content", ""),
            }
            resident.append(unit)
            by_id[unit_id] = unit
            committed_item = {"bucket": "mutations", "op": op, "id": unit_id, "applied": True}
            if isinstance(item.get("source_ids"), list):
                committed_item["source_ids"] = [s for s in item["source_ids"] if isinstance(s, str)]
            committed.append(committed_item)
        elif op == "update_unit":
            unit_id = item["id"]
            target = by_id.get(unit_id)
            if target is None:
                committed.append({"bucket": "mutations", "op": op, "id": unit_id, "applied": False})
                continue
            for key in ("type", "content", "status"):
                if key in item:
                    target[key] = item[key]
            committed_item = {"bucket": "mutations", "op": op, "id": unit_id, "applied": True}
            if isinstance(item.get("source_ids"), list):
                committed_item["source_ids"] = [s for s in item["source_ids"] if isinstance(s, str)]
            committed.append(committed_item)
        else:
            committed.append({"bucket": "mutations", "op": op, "applied": True})

    for item in memory_ops:
        op = item.get("op")
        if op == "memory_release":
            ids = item.get("ids", [])
            for unit_id in ids:
                if unit_id not in evicted:
                    evicted.append(unit_id)
                resident[:] = [u for u in resident if u.get("id") != unit_id]
                by_id.pop(unit_id, None)
            committed.append({"bucket": "memory", "op": op, "ids": ids, "applied": True})
        elif op == "memory_fault":
            ids = item.get("ids", [])
            for unit_id in ids:
                if unit_id in evicted:
                    evicted.remove(unit_id)
                if unit_id not in by_id:
                    unit = {"id": unit_id, "type": "faulted_unit", "content": "[faulted back]"}
                    resident.append(unit)
                    by_id[unit_id] = unit
            committed.append({"bucket": "memory", "op": op, "ids": ids, "applied": True})
        else:
            committed.append({"bucket": "memory", "op": op, "applied": True})

    valid_focus = [x for x in next_focus if isinstance(x, str) and x in by_id]
    next_ws["focus"] = valid_focus

    return next_ws, committed


def _lmstudio_chat(
    base_url: str,
    model: str,
    step: dict[str, Any],
    timeout_s: float,
    decode_config: dict[str, Any],
    teacher_trace: list[dict[str, Any]] | None,
) -> RunResult:
    schema = {
        "type": "object",
        "properties": {
            "mutations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"op": {"type": "string"}},
                    "required": ["op"],
                    "additionalProperties": True,
                },
            },
            "memory": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"op": {"type": "string"}},
                    "required": ["op"],
                    "additionalProperties": True,
                },
            },
            "internal_only": {"type": "array", "items": {"type": "object"}},
            "external_outputs": {"type": "array", "items": {"type": "object"}},
            "next_focus": {"type": "array", "items": {"type": "string"}},
            "step_kind": {"type": "string", "enum": sorted(STEP_KINDS)},
        },
        "required": [
            "mutations",
            "memory",
            "internal_only",
            "external_outputs",
            "next_focus",
            "step_kind",
        ],
        "additionalProperties": False,
    }

    payload = {
        "model": model,
        "temperature": decode_config["temperature"],
        "top_p": decode_config["top_p"],
        "max_tokens": decode_config["max_tokens"],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "cognitive_step_v1_output",
                "schema": schema,
            },
        },
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "task": "execute_cognitive_step_v1",
                        "step": step,
                        "teacher_trace": teacher_trace or [],
                        "contract": _build_contract(step),
                    },
                    ensure_ascii=True,
                ),
            },
        ],
    }

    seed = decode_config.get("seed")
    if seed is not None:
        payload["seed"] = seed

    url = base_url.rstrip("/") + "/v1/chat/completions"
    started_at = _utc_now()
    t0 = time.perf_counter()
    response_status = 0
    response_json: dict[str, Any] | None = None
    response_text = ""

    with httpx.Client(timeout=timeout_s) as client:
        resp = client.post(url, json=payload)
        response_status = resp.status_code
        response_text = resp.text
        try:
            data = resp.json()
            if isinstance(data, dict):
                response_json = data
        except json.JSONDecodeError:
            data = {}

    completed_at = _utc_now()
    latency = time.perf_counter() - t0

    if response_status >= 400:
        return RunResult(
            ok=False,
            latency_s=latency,
            raw_content="",
            parsed=None,
            errors=[f"http_{response_status}"],
            usage={},
            request_payload=payload,
            response_status=response_status,
            response_json=response_json,
            response_text=response_text,
            started_at=started_at,
            completed_at=completed_at,
        )

    usage = data.get("usage", {}) if isinstance(data, dict) else {}
    raw_content = ""
    finish_reason = None
    if isinstance(data, dict):
        choices = data.get("choices", [])
        if choices and isinstance(choices[0], dict):
            finish_reason = choices[0].get("finish_reason")
            msg = choices[0].get("message", {})
            if isinstance(msg, dict):
                content = msg.get("content", "")
                if isinstance(content, str):
                    raw_content = content
                elif isinstance(content, list):
                    parts = []
                    for item in content:
                        if isinstance(item, dict) and isinstance(item.get("text"), str):
                            parts.append(item["text"])
                    raw_content = "\n".join(parts)

    parsed = _extract_json(raw_content)
    errors = ["could not parse JSON output"] if parsed is None else []

    return RunResult(
        ok=not errors,
        latency_s=latency,
        raw_content=raw_content,
        parsed=parsed,
        errors=errors,
        usage=usage,
        request_payload=payload,
        response_status=response_status,
        response_json=response_json,
        response_text=response_text,
        started_at=started_at,
        completed_at=completed_at,
    )


def _openrouter_chat(
    base_url: str,
    api_key: str,
    app_name: str,
    model: str,
    step: dict[str, Any],
    timeout_s: float,
    hard_timeout_s: float,
    decode_config: dict[str, Any],
    teacher_trace: list[dict[str, Any]] | None,
) -> RunResult:
    schema = {
        "type": "object",
        "properties": {
            "mutations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"op": {"type": "string"}},
                    "required": ["op"],
                    "additionalProperties": True,
                },
            },
            "memory": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"op": {"type": "string"}},
                    "required": ["op"],
                    "additionalProperties": True,
                },
            },
            "internal_only": {"type": "array", "items": {"type": "object"}},
            "external_outputs": {"type": "array", "items": {"type": "object"}},
            "next_focus": {"type": "array", "items": {"type": "string"}},
            "step_kind": {"type": "string", "enum": sorted(STEP_KINDS)},
        },
        "required": [
            "mutations",
            "memory",
            "internal_only",
            "external_outputs",
            "next_focus",
            "step_kind",
        ],
        "additionalProperties": False,
    }

    payload = {
        "model": model,
        "temperature": decode_config["temperature"],
        "top_p": decode_config["top_p"],
        "max_tokens": decode_config["max_tokens"],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "cognitive_step_v1_output",
                "strict": True,
                "schema": schema,
            },
        },
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "task": "execute_cognitive_step_v1",
                        "step": step,
                        "teacher_trace": teacher_trace or [],
                        "contract": _build_contract(step),
                    },
                    ensure_ascii=True,
                ),
            },
        ],
    }

    seed = decode_config.get("seed")
    if seed is not None:
        payload["seed"] = seed

    url = base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-Title": app_name,
    }

    started_at = _utc_now()
    t0 = time.perf_counter()
    response_status = 0
    response_json: dict[str, Any] | None = None
    response_text = ""

    timeout = httpx.Timeout(
        connect=min(10.0, timeout_s),
        read=timeout_s,
        write=timeout_s,
        pool=min(10.0, timeout_s),
    )
    try:
        with httpx.Client(timeout=timeout) as client:
            with client.stream("POST", url, headers=headers, json=payload) as resp:
                response_status = resp.status_code
                chunks: list[bytes] = []
                for chunk in resp.iter_bytes():
                    chunks.append(chunk)
                    if time.perf_counter() - t0 > hard_timeout_s:
                        # Ensure we raise an error that is caught by the except block
                        raise httpx.TimeoutException(
                            f"hard timeout exceeded while reading response body ({hard_timeout_s}s)"
                        )
                response_text = b"".join(chunks).decode("utf-8", errors="replace")
                try:
                    data = json.loads(response_text)
                    if isinstance(data, dict):
                        response_json = data
                except json.JSONDecodeError:
                    data = {}
    except httpx.TimeoutException as e:
        completed_at = _utc_now()
        latency = time.perf_counter() - t0
        return RunResult(
            ok=False,
            latency_s=latency,
            raw_content="",
            parsed=None,
            errors=[f"request_timeout: {type(e).__name__}"],
            usage={},
            request_payload=payload,
            response_status=599,
            response_json=response_json,
            response_text=str(e),
            started_at=started_at,
            completed_at=completed_at,
        )
    except Exception as e:
        completed_at = _utc_now()
        latency = time.perf_counter() - t0
        return RunResult(
            ok=False,
            latency_s=latency,
            raw_content="",
            parsed=None,
            errors=[f"request_error: {type(e).__name__}: {e}"],
            usage={},
            request_payload=payload,
            response_status=599,
            response_json=response_json,
            response_text=str(e),
            started_at=started_at,
            completed_at=completed_at,
        )

    completed_at = _utc_now()
    latency = time.perf_counter() - t0

    if response_status >= 400:
        return RunResult(
            ok=False,
            latency_s=latency,
            raw_content="",
            parsed=None,
            errors=[f"http_{response_status}"],
            usage={},
            request_payload=payload,
            response_status=response_status,
            response_json=response_json,
            response_text=response_text,
            started_at=started_at,
            completed_at=completed_at,
        )

    usage = data.get("usage", {}) if isinstance(data, dict) else {}
    raw_content = ""
    finish_reason = None
    if isinstance(data, dict):
        choices = data.get("choices", [])
        if choices and isinstance(choices[0], dict):
            finish_reason = choices[0].get("finish_reason")
            msg = choices[0].get("message", {})
            if isinstance(msg, dict):
                content = msg.get("content", "")
                if isinstance(content, str):
                    raw_content = content
                elif isinstance(content, list):
                    parts = []
                    for item in content:
                        if isinstance(item, dict) and isinstance(item.get("text"), str):
                            parts.append(item["text"])
                    raw_content = "\n".join(parts)

    parsed = _extract_json(raw_content)
    errors = ["could not parse JSON output"] if parsed is None else []
    if finish_reason == "content_filter":
        errors.append("provider_content_filter")
    if finish_reason == "length":
        errors.append("output_truncated_length")
    return RunResult(
        ok=not errors,
        latency_s=latency,
        raw_content=raw_content,
        parsed=parsed,
        errors=errors,
        usage=usage,
        request_payload=payload,
        response_status=response_status,
        response_json=response_json,
        response_text=response_text,
        started_at=started_at,
        completed_at=completed_at,
    )


def _discover_openrouter_tool_models(
    base_url: str,
    api_key: str,
    include_regex: str | None = None,
    exclude_regex: str | None = None,
) -> list[str]:
    url = base_url.rstrip("/") + "/models"
    headers = {"Authorization": f"Bearer {api_key}"}
    with httpx.Client(timeout=60) as client:
        resp = client.get(url, headers=headers)
        resp.raise_for_status()
        data = resp.json()

    models: list[str] = []
    for item in data.get("data", []):
        if not isinstance(item, dict):
            continue
        model_id = item.get("id")
        if not isinstance(model_id, str):
            continue
        supported = item.get("supported_parameters", [])
        if not isinstance(supported, list):
            supported = []
        supported_set = {str(x) for x in supported}
        # Strict filter: explicit tools/function capability only.
        if "tools" not in supported_set and "tool_choice" not in supported_set:
            continue
        if include_regex and not re.search(include_regex, model_id):
            continue
        if exclude_regex and re.search(exclude_regex, model_id):
            continue
        models.append(model_id)
    return sorted(set(models))


def _classify_candidate(row: dict[str, Any]) -> str:
    valid = int(row.get("valid_steps", 0))
    total = int(row.get("total_steps", 0))
    req_miss = int(row.get("required_op_misses", 0))
    mem = int(row.get("memory_action_commits", 0))
    sem = int(row.get("semantic_rejection_count", 0))
    cfilter = int(row.get("content_filter_count", 0))
    if total == 0:
        return "drop"
    if valid == total and req_miss == 0 and mem >= 2 and sem == 0 and cfilter == 0:
        return "ready"
    if valid >= max(1, total - 1) and req_miss <= 1 and cfilter == 0:
        return "borderline"
    return "drop"


def _single_agent_paging_scenario(mode: str = "treatment") -> list[dict[str, Any]]:
    allowed = [
        "create_unit",
        "update_unit",
        "memory_release",
        "memory_fault",
        "memory_curate",
        "emit_text",
        "halt",
    ]

    ws0 = {
        "resident_units": [
            {"id": "unit-01", "type": "goal", "content": "Find root cause of failing benchmark"},
            {"id": "unit-02", "type": "evidence", "content": "Profiler trace v7"},
            {"id": "unit-03", "type": "tool_result", "content": "Large stack trace sample"},
            {"id": "unit-04", "type": "evidence", "content": "Regression starts after allocator patch"},
            {"id": "unit-05", "type": "summary", "content": "Old meeting notes"},
        ],
        "evicted_handles": ["unit-09"],
        "focus": ["unit-01", "unit-02"],
        "memory_budget_units": 3,
    }

    step1 = {
        "workspace": ws0,
        "events": [
            {"type": "human_input", "content": "Curate what matters and reduce memory load."}
        ],
        "policies": {
            "preserve_unresolved": True,
            "evict_tool_output_aggressively": True,
            "prefer_faults_over_compaction": True,
        },
        "allowed_actions": allowed,
        "goal_or_focus": "Curate essential hypothesis/evidence units.",
        "policy_header": {"experiment": "single_agent_paging", "mode": mode, "step": 1},
        "task_assertions": {
            "require_non_halt": True,
            "min_committed_actions": 1,
        },
    }

    step2_event = (
        "Memory pressure critical. Release at least one resident low-value unit now."
        if mode == "treatment"
        else "No memory pressure. Continue analysis and update one unit."
    )
    step2 = {
        "workspace": None,
        "events": [{"type": "scheduler_wakeup", "content": step2_event}],
        "policies": {
            "preserve_unresolved": True,
            "evict_tool_output_aggressively": mode == "treatment",
            "prefer_faults_over_compaction": True,
        },
        "allowed_actions": allowed,
        "goal_or_focus": "Manage memory while preserving active question.",
        "policy_header": {"experiment": "single_agent_paging", "mode": mode, "step": 2},
        "task_assertions": (
            {"require_non_halt": True, "require_ops": ["memory_release"]}
            if mode == "treatment"
            else {"require_non_halt": True, "min_committed_actions": 1}
        ),
    }

    step3 = {
        "workspace": None,
        "events": [
            {
                "type": "human_input",
                "content": (
                    "Answer requires the evicted baseline record unit-09. Fault it in and continue."
                    if mode == "treatment"
                    else "Continue reasoning and provide status update."
                ),
            }
        ],
        "policies": {
            "preserve_unresolved": True,
            "evict_tool_output_aggressively": mode == "treatment",
            "prefer_faults_over_compaction": True,
        },
        "allowed_actions": allowed,
        "goal_or_focus": "Recover necessary context and report progress.",
        "policy_header": {"experiment": "single_agent_paging", "mode": mode, "step": 3},
        "task_assertions": (
            {"require_ops": ["memory_fault"], "require_non_halt": True}
            if mode == "treatment"
            else {"require_non_halt": True}
        ),
    }

    step4 = {
        "workspace": None,
        "events": [
            {
                "type": "scheduler_wakeup",
                "content": (
                    "Use the restored unit to commit a concrete state update."
                    if mode == "treatment"
                    else "Commit a concrete state update."
                ),
            }
        ],
        "policies": {
            "preserve_unresolved": True,
            "evict_tool_output_aggressively": mode == "treatment",
            "prefer_faults_over_compaction": True,
        },
        "allowed_actions": allowed,
        "goal_or_focus": "Commit mutation based on currently available evidence.",
        "policy_header": {"experiment": "single_agent_paging", "mode": mode, "step": 4},
        "task_assertions": {
            "require_non_halt": True,
            "min_committed_actions": 1,
            "require_fault_use_recent": mode == "treatment",
        },
    }

    step5 = {
        "workspace": None,
        "events": [
            {"type": "human_input", "content": "Produce the user-facing status output and stop."}
        ],
        "policies": {
            "preserve_unresolved": True,
            "evict_tool_output_aggressively": mode == "treatment",
            "prefer_faults_over_compaction": True,
        },
        "allowed_actions": allowed,
        "goal_or_focus": "Emit external output and halt cleanly.",
        "policy_header": {"experiment": "single_agent_paging", "mode": mode, "step": 5},
        "task_assertions": {"require_external_output": True},
    }

    return [step1, step2, step3, step4, step5]


def _synthetic_teacher_trace(level: str = "full") -> list[dict[str, Any]]:
    scenario = _single_agent_paging_scenario(mode="treatment")
    trace: list[dict[str, Any]] = []
    for idx, step in enumerate(scenario, start=1):
        row: dict[str, Any] = {
            "step_index": idx,
            "events": step.get("events", []),
            "goal_or_focus": step.get("goal_or_focus"),
            "task_assertions": step.get("task_assertions", {}),
        }
        example = _mock_proposed_output(step, mode="treatment", step_index=idx)
        if level == "full":
            row["example_output"] = example
        elif level == "reduced":
            row["example_output"] = {
                "mutations": [{"op": x.get("op")} for x in example.get("mutations", []) if isinstance(x, dict)],
                "memory": [{"op": x.get("op")} for x in example.get("memory", []) if isinstance(x, dict)],
                "internal_only": [],
                "external_outputs": [{"type": "text"}] if example.get("external_outputs") else [],
                "next_focus": example.get("next_focus", [])[:1],
                "step_kind": example.get("step_kind"),
            }
            # Reduced guidance ablation: only first 3 exemplar steps.
            if idx > 3:
                continue
        else:
            raise ValueError(f"unknown synthetic trace level: {level}")
        trace.append(row)
    return trace


def _evaluate_task_assertions(
    step: dict[str, Any],
    proposed: dict[str, Any] | None,
    committed_actions: list[dict[str, Any]],
    step_index: int,
    recent_fault_ids: set[str] | None = None,
) -> list[str]:
    errors: list[str] = []
    assertions = step.get("task_assertions", {})
    if not isinstance(assertions, dict):
        return errors

    step_kind = (proposed or {}).get("step_kind")
    if assertions.get("require_non_halt") and step_kind == "halt":
        errors.append("policy_invalid: halt not allowed for this step")
    if step_index <= 2 and step_kind == "halt":
        errors.append("policy_invalid: early halt disallowed (steps 1-2)")

    if "min_committed_actions" in assertions:
        minimum = assertions.get("min_committed_actions")
        if isinstance(minimum, int) and len(committed_actions) < minimum:
            errors.append(f"policy_invalid: requires >= {minimum} committed actions")

    required_ops = assertions.get("require_ops")
    if isinstance(required_ops, list):
        committed_ops = {a.get("op") for a in committed_actions if isinstance(a, dict)}
        for op in required_ops:
            if op not in committed_ops:
                errors.append(f"policy_invalid: required committed op missing: {op}")

    if assertions.get("require_external_output"):
        external = (proposed or {}).get("external_outputs", [])
        if not isinstance(external, list) or len(external) == 0:
            errors.append("policy_invalid: external output required")

    if assertions.get("require_fault_use_recent"):
        required_ids = recent_fault_ids or set()
        if not required_ids:
            errors.append("policy_invalid: no recent fault ids available for fault-use check")
        else:
            used = False
            for action in committed_actions:
                if not isinstance(action, dict):
                    continue
                source_ids = action.get("source_ids", [])
                if isinstance(source_ids, list) and any(isinstance(s, str) and s in required_ids for s in source_ids):
                    used = True
                    break
            if not used:
                errors.append("policy_invalid: faulted unit not used by later committed mutation")

    return errors


def _step_scoreboard(
    step: dict[str, Any],
    proposed: dict[str, Any] | None,
    committed_actions: list[dict[str, Any]],
    task_assertion_errors: list[str],
    semantic_rejections: list[dict[str, Any]],
    step_index: int,
) -> dict[str, Any]:
    assertions = step.get("task_assertions", {})
    required_ops = assertions.get("require_ops", []) if isinstance(assertions, dict) else []
    committed_ops = [
        a.get("op")
        for a in committed_actions
        if isinstance(a, dict) and isinstance(a.get("op"), str)
    ]
    required_missing = [op for op in required_ops if op not in committed_ops]

    step_kind = (proposed or {}).get("step_kind")
    halt_illegal = False
    if isinstance(assertions, dict) and assertions.get("require_non_halt") and step_kind == "halt":
        halt_illegal = True
    if step_index <= 2 and step_kind == "halt":
        halt_illegal = True

    memory_action_count = len([a for a in committed_actions if a.get("bucket") == "memory"])
    return {
        "assertions_pass": len(task_assertion_errors) == 0,
        "required_ops_missing": required_missing,
        "halt_legal": not halt_illegal,
        "committed_action_count": len(committed_actions),
        "memory_action_count": memory_action_count,
        "semantic_rejection_count": len(semantic_rejections),
        "step_kind": step_kind,
        "external_output_count": len((proposed or {}).get("external_outputs", []))
        if isinstance((proposed or {}).get("external_outputs", []), list)
        else 0,
    }


def _mock_proposed_output(step: dict[str, Any], mode: str, step_index: int) -> dict[str, Any]:
    if mode == "treatment":
        if step_index == 1:
            return {
                "mutations": [{"op": "update_unit", "id": "unit-01", "content": "Hypothesis refined with allocator regression evidence"}],
                "memory": [],
                "internal_only": [{"note": "refined active hypothesis"}],
                "external_outputs": [],
                "next_focus": ["unit-01", "unit-04"],
                "step_kind": "internal",
            }
        if step_index == 2:
            return {
                "mutations": [],
                "memory": [{"op": "memory_release", "ids": ["unit-05"]}],
                "internal_only": [{"note": "released stale summary under pressure"}],
                "external_outputs": [],
                "next_focus": ["unit-01"],
                "step_kind": "internal",
            }
        if step_index == 3:
            return {
                "mutations": [],
                "memory": [{"op": "memory_fault", "ids": ["unit-09"]}],
                "internal_only": [{"note": "faulted baseline record for required comparison"}],
                "external_outputs": [],
                "next_focus": ["unit-09", "unit-01"],
                "step_kind": "fault_restore",
            }
        if step_index == 4:
            return {
                "mutations": [{
                    "op": "create_unit",
                    "id": "unit-10",
                    "type": "hypothesis",
                    "content": "Allocator patch likely introduces regression via fragmentation",
                    "source_ids": ["unit-09"],
                }],
                "memory": [],
                "internal_only": [{"note": "created derived hypothesis from faulted data"}],
                "external_outputs": [],
                "next_focus": ["unit-10", "unit-01"],
                "step_kind": "internal",
            }
        return {
            "mutations": [],
            "memory": [],
            "internal_only": [],
            "external_outputs": [{"type": "text", "content": "Root cause likely allocator patch regression; mitigation test queued."}],
            "next_focus": ["unit-10"],
            "step_kind": "halt",
        }

    # control
    if step_index == 1:
        return {
            "mutations": [{"op": "update_unit", "id": "unit-01", "content": "Narrowed investigation scope"}],
            "memory": [],
            "internal_only": [],
            "external_outputs": [],
            "next_focus": ["unit-01", "unit-02"],
            "step_kind": "internal",
        }
    if step_index == 2:
        return {
            "mutations": [{"op": "update_unit", "id": "unit-02", "content": "Trace supports allocator interaction"}],
            "memory": [],
            "internal_only": [],
            "external_outputs": [],
            "next_focus": ["unit-01", "unit-02"],
            "step_kind": "internal",
        }
    if step_index == 3:
        return {
            "mutations": [{"op": "create_unit", "id": "unit-11", "type": "summary", "content": "Current status stable; more evidence needed"}],
            "memory": [],
            "internal_only": [],
            "external_outputs": [],
            "next_focus": ["unit-11"],
            "step_kind": "external",
        }
    if step_index == 4:
        return {
            "mutations": [{"op": "update_unit", "id": "unit-11", "content": "Status refined with additional checks"}],
            "memory": [],
            "internal_only": [],
            "external_outputs": [],
            "next_focus": ["unit-11"],
            "step_kind": "internal",
        }
    return {
        "mutations": [],
        "memory": [],
        "internal_only": [],
        "external_outputs": [{"type": "text", "content": "Investigation active; allocator interaction remains top hypothesis."}],
        "next_focus": ["unit-11"],
        "step_kind": "halt",
    }


def _workspace_distortion(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    a_units = _workspace_unit_ids(a)
    b_units = _workspace_unit_ids(b)
    a_ev = _workspace_evicted_ids(a)
    b_ev = _workspace_evicted_ids(b)
    a_focus = a.get("focus", []) if isinstance(a.get("focus"), list) else []
    b_focus = b.get("focus", []) if isinstance(b.get("focus"), list) else []

    def jaccard(x: set[str], y: set[str]) -> float:
        denom = len(x | y)
        return 1.0 if denom == 0 else len(x & y) / denom

    resident_j = jaccard(a_units, b_units)
    evicted_j = jaccard(a_ev, b_ev)
    focus_match = a_focus == b_focus
    distortion = round(1.0 - ((resident_j + evicted_j + (1.0 if focus_match else 0.0)) / 3.0), 6)

    return {
        "resident_jaccard": round(resident_j, 6),
        "evicted_jaccard": round(evicted_j, 6),
        "focus_exact_match": focus_match,
        "distortion": distortion,
    }


def _load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def write_replay_distortion_report(candidate: Path, control: Path) -> Path:
    candidate_recs = _load_records(candidate)
    control_recs = _load_records(control)
    n = min(len(candidate_recs), len(control_recs))

    rows = []
    for i in range(n):
        c_ws = candidate_recs[i].get("workspace_after", {})
        k_ws = control_recs[i].get("workspace_after", {})
        metric = _workspace_distortion(c_ws, k_ws)
        rows.append({"step_index": i + 1, **metric})

    mean_distortion = round(sum(r["distortion"] for r in rows) / n, 6) if n else None
    report = {
        "candidate_log": str(candidate),
        "control_log": str(control),
        "compared_steps": n,
        "mean_distortion": mean_distortion,
        "per_step": rows,
        "generated_at": _utc_now(),
    }

    out_path = candidate.with_name(candidate.stem + "_replay_distortion.json")
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=True, indent=2)
    return out_path


def run_experiment(
    provider: str,
    base_url: str,
    model: str,
    timeout_s: float,
    out_dir: Path,
    decode_config: dict[str, Any],
    mode: str,
    proposer: str,
    teacher_trace_mode: str,
    openrouter_api_key: str | None = None,
    max_steps: int = 0,
    openrouter_app_name: str = "pichay-cognitive-step-v1",
    request_hard_timeout: float = 180.0,
    max_retries: int = 3,
    teacher_trace_override: list[dict[str, Any]] | None = None,
    scenario_override: list[dict[str, Any]] | None = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    out_path = out_dir / f"cognitive_step_v1_{mode}_{run_id}.jsonl"

    scenario = scenario_override if scenario_override is not None else _single_agent_paging_scenario(mode=mode)
    if max_steps > 0:
        scenario = scenario[:max_steps]
    teacher_trace: list[dict[str, Any]] | None = teacher_trace_override
    if teacher_trace is None:
        if teacher_trace_mode == "synthetic":
            teacher_trace = _synthetic_teacher_trace(level="full")
        elif teacher_trace_mode == "synthetic_reduced":
            teacher_trace = _synthetic_teacher_trace(level="reduced")
    workspace = scenario[0]["workspace"]
    lamport = 0
    prev_event_id: str | None = None
    recent_fault_ttl: dict[str, int] = {}

    with out_path.open("w", encoding="utf-8") as f:
        for i, step in enumerate(scenario, start=1):
            step = json.loads(json.dumps(step))
            if step.get("workspace") is None:
                step["workspace"] = workspace

            lamport += 1
            event_id = f"evt-{run_id}-{i:03d}"
            parent_refs = [prev_event_id] if prev_event_id else []
            step["clock"] = {"lamport": lamport, "session": run_id}
            step["event_id"] = event_id
            step["parent_refs"] = parent_refs

            print(
                f"model={model} trace={teacher_trace_mode} step={i}/{len(scenario)} request_start",
                flush=True,
            )
            
            result = None
            for attempt in range(max_retries):
                if proposer == "mock":
                    mock_payload = {
                        "model": "mock-proposer",
                        "step": step,
                        "decode_config": decode_config,
                    }
                    proposed = _mock_proposed_output(step, mode=mode, step_index=i)
                    now = _utc_now()
                    result = RunResult(
                        ok=True,
                        latency_s=0.0,
                        raw_content=json.dumps(proposed, ensure_ascii=True),
                        parsed=proposed,
                        errors=[],
                        usage={},
                        request_payload=mock_payload,
                        response_status=200,
                        response_json={"mock": True, "output": proposed},
                        response_text=json.dumps({"mock": True, "output": proposed}, ensure_ascii=True),
                        started_at=now,
                        completed_at=now,
                    )
                    break
                elif provider == "openrouter":
                    if not openrouter_api_key:
                        raise ValueError("openrouter_api_key is required for provider=openrouter")
                    result = _openrouter_chat(
                        base_url=base_url,
                        api_key=openrouter_api_key,
                        app_name=openrouter_app_name,
                        model=model,
                        step=step,
                        timeout_s=timeout_s,
                        hard_timeout_s=request_hard_timeout,
                        decode_config=decode_config,
                        teacher_trace=teacher_trace,
                    )
                else:
                    result = _lmstudio_chat(
                        base_url=base_url,
                        model=model,
                        step=step,
                        timeout_s=timeout_s,
                        decode_config=decode_config,
                        teacher_trace=teacher_trace,
                    )
                
                # Retry on timeout or 5xx error
                if result.ok or (result.response_status < 500 and "request_timeout" not in result.errors[0] and "request_error" not in result.errors[0]):
                    break
                
                if attempt < max_retries - 1:
                    wait_s = 2 ** attempt
                    print(f"  attempt {attempt+1} failed ({result.errors}), retrying in {wait_s}s...", flush=True)
                    time.sleep(wait_s)
            
            if result is None:
                 # Should not happen given the logic above
                 raise RuntimeError("Failed to get result from provider")


            proposed = result.parsed if isinstance(result.parsed, dict) else None
            shape_rejections: list[dict[str, Any]] = []
            accepted_mutations: list[dict[str, Any]] = []
            accepted_memory: list[dict[str, Any]] = []
            semantic_rejections: list[dict[str, Any]] = []

            workspace_after = workspace
            committed_actions: list[dict[str, Any]] = []
            validation_errors = list(result.errors)

            if proposed is not None:
                shape_rejections = _validate_shape(proposed)
                if not shape_rejections:
                    accepted_mutations, accepted_memory, semantic_rejections = _validate_actions(
                        proposed,
                        workspace_before=workspace,
                        allowed_actions=set(step.get("allowed_actions", [])),
                    )
                    next_focus = proposed.get("next_focus", [])
                    workspace_after, committed_actions = _reduce_workspace(
                        workspace=workspace,
                        mutations=accepted_mutations,
                        memory_ops=accepted_memory,
                        next_focus=next_focus if isinstance(next_focus, list) else [],
                    )
                else:
                    validation_errors.extend([f["reason"] for f in shape_rejections])
            task_assertion_errors = _evaluate_task_assertions(
                step=step,
                proposed=proposed,
                committed_actions=committed_actions,
                step_index=i,
                recent_fault_ids=set(recent_fault_ttl.keys()),
            )
            validation_errors.extend(task_assertion_errors)
            scoreboard = _step_scoreboard(
                step=step,
                proposed=proposed,
                committed_actions=committed_actions,
                task_assertion_errors=task_assertion_errors,
                semantic_rejections=semantic_rejections,
                step_index=i,
            )

            validated_actions = {
                "accepted": {
                    "mutations": accepted_mutations,
                    "memory": accepted_memory,
                },
                "rejected": shape_rejections + semantic_rejections,
            }

            ok = (
                result.response_status < 400
                and not validation_errors
                and not shape_rejections
                and len(semantic_rejections) == 0
            )

            record = {
                "run_id": run_id,
                "mode": mode,
                "step_index": i,
                "timestamp": _utc_now(),
                "started_at": result.started_at,
                "completed_at": result.completed_at,
                "event_id": event_id,
                "clock": {"lamport": lamport, "session": run_id},
                "parent_refs": parent_refs,
                "endpoint": base_url,
                "provider": provider,
                "openrouter_app": openrouter_app_name if provider == "openrouter" else None,
                "model": model,
                "proposer": proposer,
                "teacher_trace_mode": teacher_trace_mode,
                "decode_config": decode_config,
                "input_step": step,
                "request_payload": result.request_payload,
                "ok": ok,
                "errors": validation_errors,
                "latency_s": round(result.latency_s, 3),
                "usage": result.usage,
                "response_status": result.response_status,
                "response_json": result.response_json,
                "response_text": result.response_text,
                "raw_output": result.raw_content,
                "proposed_output": proposed,
                "proposed_actions": {
                    "mutations": (proposed or {}).get("mutations", []),
                    "memory": (proposed or {}).get("memory", []),
                },
                "validated_actions": validated_actions,
                "committed_actions": committed_actions,
                "task_assertion_errors": task_assertion_errors,
                "recent_fault_ids_before_step": sorted(recent_fault_ttl.keys()),
                "scoreboard": scoreboard,
                "internal_only": (proposed or {}).get("internal_only", []),
                "external_outputs": (proposed or {}).get("external_outputs", []),
                "step_kind": (proposed or {}).get("step_kind"),
                "workspace_before": workspace,
                "workspace_after": workspace_after,
            }
            f.write(json.dumps(record, ensure_ascii=True) + "\n")
            f.flush()

            # Track recent faults for downstream fault-use assertions.
            step_faults: set[str] = set()
            for action in committed_actions:
                if (
                    isinstance(action, dict)
                    and action.get("bucket") == "memory"
                    and action.get("op") == "memory_fault"
                    and isinstance(action.get("ids"), list)
                ):
                    for fid in action["ids"]:
                        if isinstance(fid, str):
                            step_faults.add(fid)
            # Decay existing fault visibility window (t+1 or t+2).
            decayed: dict[str, int] = {}
            for fid, ttl in recent_fault_ttl.items():
                if ttl - 1 > 0:
                    decayed[fid] = ttl - 1
            recent_fault_ttl = decayed
            for fid in step_faults:
                recent_fault_ttl[fid] = 2

            workspace = workspace_after
            prev_event_id = event_id

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Cognitive transaction v1 harness against LM Studio endpoint")
    parser.add_argument("--provider", choices=["lmstudio", "openrouter"], default="lmstudio")
    parser.add_argument("--base-url", default="http://192.168.111.125:1234", help="Provider base URL")
    parser.add_argument("--model", default="openai/gpt-oss-20b", help="Model ID loaded in LM Studio")
    parser.add_argument("--models", nargs="+", default=None, help="Optional batch run model IDs")
    parser.add_argument("--timeout", type=float, default=120.0, help="Per-step timeout seconds")
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/cognitive_transactions"))
    parser.add_argument("--mode", choices=["treatment", "control"], default="treatment")
    parser.add_argument("--proposer", choices=["lmstudio", "mock"], default="lmstudio")
    parser.add_argument("--teacher-trace", choices=["none", "synthetic", "synthetic_reduced"], default="none")
    parser.add_argument(
        "--teacher-trace-modes",
        nargs="+",
        choices=["none", "synthetic", "synthetic_reduced"],
        default=None,
    )
    parser.add_argument("--max-steps", type=int, default=0, help="0 means full scenario")
    parser.add_argument("--request-hard-timeout", type=float, default=180.0)
    parser.add_argument("--openrouter-api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--openrouter-base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--openrouter-app", default="pichay-cognitive-step-v1")
    parser.add_argument("--discover-openrouter-tool-models", action="store_true")
    parser.add_argument("--openrouter-include-regex", default=None)
    parser.add_argument("--openrouter-exclude-regex", default=None)
    parser.add_argument("--max-models", type=int, default=0, help="0 means no cap")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--compare-control-log",
        type=Path,
        default=None,
        help="Optional control run JSONL path for replay-distortion comparison",
    )
    args = parser.parse_args()

    decode_config = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
    }

    provider = args.provider
    base_url = args.openrouter_base_url if provider == "openrouter" else args.base_url
    openrouter_api_key = None
    if provider == "openrouter":
        openrouter_api_key = os.environ.get(args.openrouter_api_key_env)
        if not openrouter_api_key:
            raise RuntimeError(
                f"Missing OpenRouter API key in env var {args.openrouter_api_key_env}"
            )
        if args.proposer != "lmstudio":
            # Keep existing proposer switch semantics only for local flow.
            args.proposer = "lmstudio"

    model_list = [args.model] if not args.models else list(dict.fromkeys(args.models))
    if provider == "openrouter" and args.discover_openrouter_tool_models:
        discovered = _discover_openrouter_tool_models(
            base_url=base_url,
            api_key=openrouter_api_key,
            include_regex=args.openrouter_include_regex,
            exclude_regex=args.openrouter_exclude_regex,
        )
        if args.max_models > 0:
            discovered = discovered[: args.max_models]
        model_list = discovered
        print(f"discovered {len(model_list)} openrouter tool-capable model(s)", flush=True)
        if model_list:
            print("first models:", ", ".join(model_list[:10]), flush=True)
    elif args.max_models > 0:
        model_list = model_list[: args.max_models]
    trace_modes = [args.teacher_trace] if not args.teacher_trace_modes else list(dict.fromkeys(args.teacher_trace_modes))

    summary_rows: list[dict[str, Any]] = []
    for model in model_list:
        for teacher_trace_mode in trace_modes:
            print(
                f"starting model={model} trace={teacher_trace_mode} "
                f"provider={provider}",
                flush=True,
            )
            out_path = run_experiment(
                provider=provider,
                base_url=base_url,
                model=model,
                timeout_s=args.timeout,
                out_dir=args.out_dir,
                decode_config=decode_config,
                mode=args.mode,
                proposer=args.proposer,
                teacher_trace_mode=teacher_trace_mode,
                openrouter_api_key=openrouter_api_key,
                max_steps=args.max_steps,
                openrouter_app_name=(
                    f"{args.openrouter_app}:{args.mode}:{teacher_trace_mode}:{args.proposer}"
                    if provider == "openrouter"
                    else args.openrouter_app
                ),
                request_hard_timeout=args.request_hard_timeout,
            )
            print(f"wrote {out_path}", flush=True)

            ok = 0
            total = 0
            required_missing = 0
            memory_commits = 0
            halt_illegal = 0
            semantic_rejections = 0
            content_filters = 0
            with out_path.open("r", encoding="utf-8") as f:
                for line in f:
                    total += 1
                    rec = json.loads(line)
                    if rec.get("ok"):
                        ok += 1
                    sb = rec.get("scoreboard", {})
                    required_missing += len(sb.get("required_ops_missing", []))
                    memory_commits += int(sb.get("memory_action_count", 0))
                    semantic_rejections += int(sb.get("semantic_rejection_count", 0))
                    if "provider_content_filter" in (rec.get("errors") or []):
                        content_filters += 1
                    if not sb.get("halt_legal", True):
                        halt_illegal += 1
                    print(
                        f"{model} trace={teacher_trace_mode} step {rec['step_index']}: ok={rec['ok']} latency={rec['latency_s']}s "
                        f"errors={rec.get('errors', [])} rejected={len(rec.get('validated_actions', {}).get('rejected', []))} "
                        f"committed={sb.get('committed_action_count', 0)} mem={sb.get('memory_action_count', 0)} "
                        f"halt_legal={sb.get('halt_legal', True)}",
                        flush=True,
                    )
            print(f"{model} trace={teacher_trace_mode} summary: {ok}/{total} steps valid", flush=True)

            summary_rows.append(
                {
                    "provider": provider,
                    "model": model,
                    "teacher_trace_mode": teacher_trace_mode,
                    "valid_steps": ok,
                    "total_steps": total,
                    "required_op_misses": required_missing,
                    "memory_action_commits": memory_commits,
                    "halt_illegal_count": halt_illegal,
                    "semantic_rejection_count": semantic_rejections,
                    "content_filter_count": content_filters,
                    "log_path": str(out_path),
                }
            )

            if args.compare_control_log is not None and teacher_trace_mode == args.teacher_trace:
                report = write_replay_distortion_report(out_path, args.compare_control_log)
                print(f"wrote replay distortion report {report}", flush=True)

    if len(summary_rows) > 1:
        summary_path = args.out_dir / (
            datetime.now(timezone.utc).strftime("matrix_summary_%Y%m%dT%H%M%SZ") + ".json"
        )
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary_rows, f, ensure_ascii=True, indent=2)
        print(f"wrote matrix summary {summary_path}", flush=True)

        if provider == "openrouter":
            by_model: dict[str, dict[str, Any]] = defaultdict(dict)
            for row in summary_rows:
                by_model[row["model"]][row["teacher_trace_mode"]] = row

            leaderboard: list[dict[str, Any]] = []
            for model, traces in by_model.items():
                syn = traces.get("synthetic")
                if not syn:
                    continue
                none = traces.get("none")
                score = (
                    syn["valid_steps"] * 10
                    + syn["memory_action_commits"] * 2
                    - syn["required_op_misses"] * 3
                    - syn["semantic_rejection_count"] * 1
                    - syn["content_filter_count"] * 5
                )
                if none:
                    score += syn["valid_steps"] - none["valid_steps"]
                row = {
                    "model": model,
                    "score": score,
                    "classification": _classify_candidate(syn),
                    "synthetic_valid": f"{syn['valid_steps']}/{syn['total_steps']}",
                    "synthetic_required_op_misses": syn["required_op_misses"],
                    "synthetic_memory_action_commits": syn["memory_action_commits"],
                    "synthetic_semantic_rejections": syn["semantic_rejection_count"],
                    "synthetic_content_filters": syn["content_filter_count"],
                    "none_valid": (
                        f"{none['valid_steps']}/{none['total_steps']}" if none else None
                    ),
                    "synthetic_log_path": syn["log_path"],
                }
                leaderboard.append(row)

            leaderboard.sort(key=lambda x: x["score"], reverse=True)
            lb_path = args.out_dir / (
                datetime.now(timezone.utc).strftime("candidate_leaderboard_%Y%m%dT%H%M%SZ") + ".json"
            )
            with lb_path.open("w", encoding="utf-8") as f:
                json.dump(leaderboard, f, ensure_ascii=True, indent=2)
            print(f"wrote candidate leaderboard {lb_path}", flush=True)


if __name__ == "__main__":
    main()
