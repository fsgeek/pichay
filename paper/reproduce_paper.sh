#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PAPER_DIR="$ROOT_DIR/paper"

# Avoid inherited virtualenv mismatch warnings when called from another project.
unset VIRTUAL_ENV

cd "$ROOT_DIR"

echo "[1/5] Syncing dependencies..."
uv sync --extra dev

echo "[2/5] Regenerating paper figures..."
uv run python paper/figures/cost_model.py --save --output-dir paper/figures

echo "[3/5] Checking paper headline numbers against latest snapshot (if available)..."
SNAPSHOT="$(ls -1t paper/data/corpus_snapshot_*_live.json 2>/dev/null | head -n 1 || true)"
if [[ -n "$SNAPSHOT" ]]; then
  uv run python tools/check_paper_numbers.py --paper paper/main.tex --snapshot "$SNAPSHOT" --strict
  echo "  using snapshot: $SNAPSHOT"
else
  echo "  no corpus snapshot found under paper/data/; skipping numeric cross-check"
fi

echo "[4/5] Clean build..."
cd "$PAPER_DIR"
latexmk -C >/dev/null
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex >/dev/null

echo "[5/5] Verifying expected outputs..."
test -f "$PAPER_DIR/main.pdf"
test -f "$PAPER_DIR/main.bbl"
test -f "$PAPER_DIR/figures/cost_model.pdf"
test -f "$PAPER_DIR/figures/policy_gradient.pdf"
test -f "$PAPER_DIR/figures/cumulative_attention.pdf"
test -f "$PAPER_DIR/figures/fault_cost_fill.pdf"

echo
echo "Paper artifact reproduction complete."
echo "Generated/verified outputs:"
echo "  - paper/main.pdf"
echo "  - paper/main.bbl"
echo "  - paper/figures/cost_model.pdf"
echo "  - paper/figures/policy_gradient.pdf"
echo "  - paper/figures/cumulative_attention.pdf"
echo "  - paper/figures/fault_cost_fill.pdf"
