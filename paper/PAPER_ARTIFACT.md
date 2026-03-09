# Paper Artifact Reproduction

This guide reproduces the `paper/main.pdf` build and verifies the key generated figures used in the manuscript.

## Scope

Deterministic from repository contents:
- LaTeX build for `paper/main.pdf`
- Figure regeneration from `paper/figures/cost_model.py`
- Optional numeric consistency check against latest frozen corpus snapshot in `paper/data/`

Not included by default:
- Recounting live corpus from local Claude/Desktop paths (machine-specific)
- Live model/API experiments

## Prerequisites

- `uv`
- TeX toolchain with `latexmk` and `pdflatex`

## One-command reproduction

From repo root:

```bash
bash paper/reproduce_paper.sh
```

This runs:
1. `uv sync --extra dev`
2. `uv run python paper/figures/cost_model.py --save --output-dir paper/figures`
3. Optional strict number check (if `paper/data/corpus_snapshot_*_live.json` exists):
   `uv run python tools/check_paper_numbers.py --paper paper/main.tex --snapshot <latest> --strict`
4. Clean LaTeX rebuild:
   `latexmk -C && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`

## Expected outputs

- `paper/main.pdf`
- `paper/main.bbl`
- `paper/figures/cost_model.pdf`
- `paper/figures/policy_gradient.pdf`
- `paper/figures/cumulative_attention.pdf`
- `paper/figures/fault_cost_fill.pdf`

## Optional: refresh live corpus snapshot (machine-specific)

If your local data paths are configured, you can regenerate a live snapshot:

```bash
bash tools/reproduce_paper_counts.sh
```

Then rerun:

```bash
uv run python tools/check_paper_numbers.py \
  --paper paper/main.tex \
  --snapshot paper/data/corpus_snapshot_<date>_live.json \
  --strict
```
