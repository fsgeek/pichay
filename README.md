# Pichay

Research code and paper artifacts for demand paging in LLM context windows.

## Gateway (v1)

`pichay` now starts the multi-provider gateway CLI.

Examples:

```bash
# Launch gateway + Claude CLI (random port)
uv run pichay --claude

# Launch gateway + Codex CLI (random port)
uv run pichay --codex

# Persistent gateway service only
uv run pichay --no-launch --port 8080
```

Observability endpoints:
- `GET /health`
- `GET /metrics` (Prometheus)
- `GET /api/sessions`
- `GET /api/events?window=24h`
- `GET /dashboard` (live browser UI)

## Reproduce Paper Counts (5 Minutes)

Run from repository root:

```bash
./tools/reproduce_paper_counts.sh
python3 tools/check_paper_numbers.py \
  --paper paper/main.tex \
  --snapshot paper/data/corpus_snapshot_20260307_live.json
```

What this does:
- Writes a timestamped live snapshot to `paper/data/corpus_snapshot_YYYYMMDD_live.json`
- Checks key manuscript literals against the snapshot and internal consistency rules

Canonical counting mode for paper claims:
- `dedup_by_content`
- `min_size=10000`

Reference protocol:
- `docs/corpus_protocol.md`

## Build Paper

```bash
latexmk -pdf paper/main.tex
```

Output:
- `paper/main.pdf`
