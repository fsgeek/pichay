# Pichay Code Review

Comprehensive review of `src/pichay/` and `tests/`.

---

## 1. proxy.py

### Unused imports

- **Line 27**: `os` is imported but never used in this file.
- **Line 29**: `time` is imported but never used. The pager module imports its own `time`.

### Bug: Streaming responses capture incomplete token usage

**Lines 431–445**: The streaming usage extraction scans lines in reverse. It finds the `message_delta` event first (appears last in the stream) and `break`s, never reaching the `message_start` event. The result: only `output_tokens` (from `message_delta`) is logged. `input_tokens`, `cache_creation_input_tokens`, and `cache_read_input_tokens` (from `message_start`) are all missed.

```python
for line in reversed(text.split("\n")):
    if "message_delta" in line and "usage" in line:
        if line.startswith("data: "):
            event_data = json.loads(line[6:])
            usage = event_data.get("usage", {})
            break  # ← exits before reaching message_start
    elif "message_start" in line and "usage" in line:
        ...
```

This means the eval framework (`eval.py`) will see `input_tokens=0` for every streaming API call (which is likely all of them, since Claude Code uses streaming). **This undermines the primary metric — token consumption.** The fix is to collect usage from both events and merge them.

### Minor: Temperature override logic

**Lines 240–247**: The two branches (`if ... and not thinking_enabled` / `elif ... and thinking_enabled`) could be simplified. They share the same guard `app.config.get("temperature_override") is not None` and just differ on whether thinking is enabled.

---

## 2. pager.py

### No issues with imports or dead code

All imports are used. The module is clean and well-structured.

### Edge case: `first_byte_ms` can be None from streaming logs

**Line 454 of proxy.py** logs `first_byte_ms: None` when no bytes are received. In `eval.py` **line 225**, `rec.get("first_byte_ms", 0)` returns `None` (key exists with value None, so the default isn't used). This `None` propagates to `TurnMetrics.first_byte_ms: int = 0`, violating the type annotation. Then at `eval.py` **line 308**:

```python
fb_times = [t.first_byte_ms for t in summary.turns if t.first_byte_ms > 0]
```

`None > 0` raises `TypeError` in Python 3. **This is a potential runtime crash** on streaming responses where no bytes were received (e.g., immediate error).

### Semantic note on `_eviction_key`

**Lines 233–248**: The fault-matching keys are reasonable, but `Grep`'s key uses `tool_input.get("path", ".")` while the actual Grep tool parameter might be omitted entirely. If the model re-issues a Grep with an explicit `.` path vs. the original having no path key, the keys would still match (both default to `.`), which is correct. No bug, just worth noting the implicit coupling to tool parameter naming.

---

## 3. eval.py

### Bug: `@property` fields missing from JSON output

**Lines 58–61, 112–133**: Key metrics like `effective_input_tokens`, `avg_input_tokens`, `avg_effective_input`, `n_squared_cost`, and `system_prompt_fraction` are all `@property` decorators on dataclasses. `asdict()` (used at **line 482** for `--json` output) does **not** include `@property` fields — only regular dataclass fields. This means the JSON output is missing these derived metrics that are central to the experiment's analysis.

### Bug: `first_byte_ms: None` causes TypeError

As noted above in pager.py section. The chain is:
1. `proxy.py:454` logs `first_byte_ms: None` for streaming responses
2. `eval.py:225` `rec.get("first_byte_ms", 0)` returns `None` (key present)
3. `eval.py:308` `t.first_byte_ms > 0` raises `TypeError: '>' not supported between instances of 'NoneType' and 'int'`

Fix: use `first_byte = rec.get("first_byte_ms") or 0`.

### Minor: `first_timestamp`/`last_timestamp` wall clock uses `datetime.fromisoformat`

**Lines 299–305**: Works correctly for the ISO format strings produced by the proxy. No issue, but worth noting there's no timezone awareness check — both timestamps include timezone info (UTC), so the subtraction is valid.

---

## 4. replay.py

### Clean module — no significant issues

Imports are all used. Logic is sound. The `copy.deepcopy` at **line 105** correctly prevents mutation of the original log data during replay.

### Minor duplication

The `_write_jsonl` helper is duplicated across all three test files (`test_replay.py:27`, `test_analyzer.py:36`, `test_trimmer.py:34`). Could be extracted to a shared conftest.py, but this is cosmetic.

---

## 5. analyzer.py

### Dead code: `_pct` helper

**Line 609**: `_pct` is defined and used within `print_analysis` (lines 552–606), so it's not dead code in this file.

### Duplicate detection only runs on turn 0

**Lines 503–508**: `_find_duplicate_skills` is only called for `turn_idx == 0`. This means if the skill list changes between turns (unlikely but possible), later duplicates are missed. The `duplicate_bytes` field then reports per-turn-0 waste, not session-level waste. The docstring and field name (`duplicate_bytes`) don't clarify this is a sample, not an aggregate.

### Tool definition bytes estimation is a median of per-request gaps

**Lines 416–439**: `_infer_tool_definition_bytes` takes the median gap between `total_request_bytes` and `(system_bytes + messages_bytes)`. This is reasonable but the gap also includes JSON envelope overhead (model name, max_tokens, etc.), so `tool_definition_bytes` is slightly overestimated.

---

## 6. trimmer.py

### Dead code: `_pct` helper never called

**Line 690**: `_pct(part, whole)` is defined but never called anywhere in `trimmer.py`. The `print_trim_report` function doesn't use it. This is dead code.

### Double computation in offline analysis

**Lines 544–551**: `_dedupe_skills_text` is called twice when duplicates are found — once to get counts, then again to get the deduped text:

```python
_, entries, dupes = _dedupe_skills_text(text)
if dupes > 0:
    new_text, _, _ = _dedupe_skills_text(text)
```

Since the function returns all three values, one call suffices.

### Potential over-stubbing: tool schema not validated before stub creation

**Line 346**: `_make_tool_stub` creates stubs for any tool definition passed to it. If the tool definition is already a stub (has `_STUB_SCHEMA`), it re-stubs it — no harm but wasteful. The `_trim_tools` method does check `tool.get("input_schema") == _STUB_SCHEMA` for the restoration path (**line 225**), so this is handled in practice.

---

## 7. __main__.py

### Unused imports

- **Line 27**: `signal` is imported but never used.

### Resource leak: file handles in subprocess.run

**Lines 188–203**: Three `subprocess.run` calls pass `stdout=open(...)` without closing the file handles:

```python
subprocess.run(
    ["git", "log", "--oneline", "-20"],
    cwd=project_dir,
    stdout=open(exp_dir / "git_log.txt", "w"),  # leaked handle
    stderr=subprocess.DEVNULL,
)
```

The file handle is created as an anonymous temporary and never closed. On CPython, the GC will close it eventually, but this is a resource leak. Should use a `with` block.

### Prompt truncation assumes >= 80 chars

**Line 148**: `args.prompt[:80]` works for any length, but the trailing `...` is always printed even for short prompts. Cosmetic issue.

### `shutil.rmtree` + `copytree(dirs_exist_ok=True)` is redundant

**Lines 181–182**: The directory is deleted then copied with `dirs_exist_ok=True`. The flag is unnecessary after rmtree, but harmless.

---

## 8. Cross-module consistency

### Naming conventions: consistent

All modules use lowercase with underscores for functions and variables. Dataclasses use PascalCase. Helper functions prefixed with `_`. Consistent throughout.

### Error handling patterns: inconsistent

- `proxy.py`: Catches broad `Exception` in multiple places (lines 383, 394, 444, 473), silently swallowing errors or logging minimal info.
- `eval.py`: `parse_proxy_log` silently skips malformed JSON lines (line 146–147). This is intentional for robustness but means corrupt data is invisible.
- `__main__.py`: Uses `sys.exit(1)` for fatal errors but catches `KeyboardInterrupt` and `FileNotFoundError` specifically (lines 168–172).

The silent error swallowing in the proxy is appropriate (don't crash the proxy on a logging failure), but the `except Exception` blocks should log more context.

### Shared helpers not factored out

- `_pct(part, whole)` appears in both `analyzer.py:609` and `trimmer.py:690`.
- `_write_jsonl` is duplicated across all three test files.
- `_content_hash` exists in both `analyzer.py:57` (Component.__post_init__) and `trimmer.py:411`, using the same sha256[:16] approach but as separate implementations.

---

## 9. Test coverage gaps

### Missing test files entirely

| Module | Test file | Status |
|--------|-----------|--------|
| `pager.py` | none | **No tests.** Core eviction engine is only indirectly tested via `test_replay.py`. |
| `eval.py` | none | **No tests.** The analysis framework (parsing, metric computation, cost estimation) has zero coverage. |
| `proxy.py` | none | **No tests.** The Flask app, streaming proxy, header stripping, and request forwarding are untested. |
| `__main__.py` | none | **No tests.** The experiment runner is untested. |
| `replay.py` | `test_replay.py` | Thorough. |
| `analyzer.py` | `test_analyzer.py` | Thorough. |
| `trimmer.py` | `test_trimmer.py` | Thorough. |

### Specific untested code paths

- **`pager.py`**: `_make_summary` for all tool types, `_content_size` with None input, `CompactionStats.reduction_pct` when `bytes_before=0`, `PageStore.summary()`, `PageStore.store()` logging path.
- **`eval.py`**: `analyze_run` pairing logic (request→response matching), wall clock computation, `print_comparison` side-by-side output, cost estimation formulas, `--compare` CLI path.
- **`proxy.py`**: `find_free_port` IPv6→IPv4 fallback, `measure_system_prompt` with absent/string/list system, streaming generator `finally` block, header stripping edge cases.
- **`__main__.py`**: `clean_env_for_subprocess`, `find_project_claude_dir`, the entire `run_experiment` flow.

### Potential false-pass tests

1. **`test_replay.py:TestMultiTurnEviction.test_apply_evictions_replaces_previously_evicted`** (line 302): Asserts `s.turns[1].bytes_saved > 0` but doesn't verify that the saving comes from BOTH the _apply_evictions replacement AND the new compaction. The test could pass even if _apply_evictions did nothing, as long as the new compaction alone saves bytes.

2. **`test_analyzer.py`**: Tests for `_infer_tool_definition_bytes` verify the median calculation works, but don't test whether the "gap" estimation actually approximates real tool definition sizes (no integration test with real-shaped data).

3. **`test_trimmer.py:TestToolTrimAfterUsage.test_restored_tools_count`** (line 167): The test verifies `restored_tools == 1` by sending a pre-stubbed tool definition in the second request. This correctly simulates what happens in production (where the API would send back whatever we gave it). Good test, no false-pass concern.

---

## 10. Summary of severity

### High (affects correctness of experiment data)
1. **Streaming token usage incomplete** (`proxy.py:431–445`): Input tokens not captured for streaming responses.
2. **Properties missing from JSON output** (`eval.py:482`): `asdict()` drops `@property` fields.
3. **`None > 0` TypeError** (`eval.py:308`): Crashes on streaming responses with no first byte.

### Medium (resource/quality issues)
4. **Resource leak** (`__main__.py:188–203`): Unclosed file handles in subprocess stdout.
5. **No tests for pager.py, eval.py, proxy.py** — core modules are untested.

### Low (cleanup)
6. **Unused imports**: `os`, `time` in `proxy.py`; `signal` in `__main__.py`.
7. **Dead code**: `_pct` in `trimmer.py:690`.
8. **Double computation**: `_dedupe_skills_text` called twice in `trimmer.py:544–546`.
9. **Duplicated helpers**: `_pct`, `_write_jsonl`, `_content_hash` across modules/tests.
