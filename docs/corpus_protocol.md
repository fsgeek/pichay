# Corpus Counting Protocol

This protocol defines how corpus counts are computed for the paper and how to reproduce them.

## Scope

Two Claude Code roots (JSONL schema):
- `~/.claude/projects`
- `~/projects/yanantin/tmp/ubuntu-vm.claude/projects`

One Claude Desktop source (different schema):
- `~/projects/yanantin/tmp/claude-desktop/conversations.json`

Desktop counts are reported separately and are not merged into Claude Code session counts.

## Session Definition (Claude Code)

A candidate session artifact is any `*.jsonl` file under the roots.

Session type classification by filename:
- `agent-acompact*` -> `compact`
- `agent-aprompt_suggestion*` -> `prompt_suggestion`
- `agent-*` -> `subagent`
- `history.jsonl`, `pretty.jsonl`, `scratch.jsonl` -> `other`
- otherwise -> `main` (UUID-style conversation files)

## Counting Modes

The paper should always state which mode is used:

1. `raw`: count all candidate files passing size filter.
2. `dedup_by_content`: content-hash dedup (`sha256(file bytes)`), keeping first occurrence.

## Size Filter

Default filter used for live recount snapshots:
- `min_size = 10000` bytes

Rationale: excludes tiny stubs/noise files while preserving conversation-bearing sessions.

## Canonical Tool

Use:
- `tools/corpus_counts.py`

Example:
```bash
python3 tools/corpus_counts.py \
  --root ~/.claude/projects \
  --root ~/projects/yanantin/tmp/ubuntu-vm.claude/projects \
  --desktop ~/projects/yanantin/tmp/claude-desktop/conversations.json \
  --min-size 10000 --json
```

## Snapshot Practice

For each paper revision, write one immutable snapshot artifact to `paper/data/`.

Suggested filename:
- `paper/data/corpus_snapshot_YYYYMMDD_live.json`

The manuscript should cite:
- snapshot timestamp (UTC)
- roots
- counting mode (`raw` or `dedup_by_content`)
- size filter

## Important Note on Drift

The live corpora continue to change. Reported paper numbers should come from a frozen cohort for that revision, not from rerunning live counts at read time.
