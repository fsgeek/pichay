# Pichay

Context paging evaluation framework. Quechua for "to sweep, to clean."

Measures the effect of removing dead tool outputs from agentic LLM
context windows.

## Setup

```bash
uv sync
source .venv/bin/activate
```

Python 3.14. uv, not pip.

## Source Layout

```
src/pichay/
    proxy.py      — MITM proxy between Claude Code and Anthropic API
    pager.py      — FIFO eviction engine for stale tool results
    eval.py       — analysis framework (per-turn metrics, cost estimates)
    __main__.py   — experiment runner CLI
```

## Convention

Read existing code before writing new code. Match the coding style,
dataclass patterns, and naming conventions you find there.

## Experiment Data

`experiments/` contains scientific output — proxy logs, session
transcripts, evaluation summaries. Do not delete experiment data.
