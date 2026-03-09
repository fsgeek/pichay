# Pichay

Research code and paper artifacts for demand paging in LLM context windows.

## Gateway

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

## Reproduce Paper Artifact (Recommended)

Run from repository root:

```bash
bash paper/reproduce_paper.sh
```

This performs:
- dependency sync
- figure regeneration (`paper/figures/cost_model.py`)
- strict headline-number check against latest snapshot (if present)
- clean LaTeX rebuild of `paper/main.pdf`

Detailed guide:
- `paper/PAPER_ARTIFACT.md`

## Reproduce/Refresh Paper Counts

Run from repository root:

```bash
./tools/reproduce_paper_counts.sh
uv run python tools/check_paper_numbers.py \
  --paper paper/main.tex \
  --snapshot paper/data/corpus_snapshot_YYYYMMDD_live.json \
  --strict
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
bash paper/reproduce_paper.sh
```

Output:
- `paper/main.pdf`
