#!/usr/bin/env bash
set -euo pipefail

ROOT1="${1:-$HOME/.claude/projects}"
ROOT2="${2:-$HOME/projects/yanantin/tmp/ubuntu-vm.claude/projects}"
DESKTOP="${3:-$HOME/projects/yanantin/tmp/claude-desktop/conversations.json}"
MIN_SIZE="${MIN_SIZE:-10000}"
DATE_TAG="${DATE_TAG:-$(date -u +%Y%m%d)}"

OUT_DIR="/home/tony/projects/pichay/paper/data"
OUT_FILE="$OUT_DIR/corpus_snapshot_${DATE_TAG}_live.json"

mkdir -p "$OUT_DIR"

python3 /home/tony/projects/pichay/tools/corpus_counts.py \
  --root "$ROOT1" \
  --root "$ROOT2" \
  --desktop "$DESKTOP" \
  --min-size "$MIN_SIZE" \
  --json > "$OUT_FILE"

echo "Wrote snapshot: $OUT_FILE"
python3 - <<'PY' "$OUT_FILE"
import json, sys
p = sys.argv[1]
d = json.load(open(p, 'r', encoding='utf-8'))
print('raw.sessions =', d['raw']['sessions'])
print('raw.by_type =', d['raw']['by_type'])
print('dedup.sessions =', d['dedup_by_content']['sessions'])
print('dedup.by_type =', d['dedup_by_content']['by_type'])
if 'desktop' in d:
    print('desktop.conversations =', d['desktop'].get('conversations'))
PY
