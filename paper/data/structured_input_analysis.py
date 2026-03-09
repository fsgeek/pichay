"""
Analysis of Structured Input Sweep — 2026-03-08

Reads the JSONL results and produces summary statistics and
breakdowns suitable for the paper.
"""

import json
from collections import Counter
from pathlib import Path

DATA = Path(__file__).parent / "structured_input_sweep_20260308.jsonl"


def load_results():
    results = []
    with open(DATA) as f:
        for line in f:
            results.append(json.loads(line))
    return results


def analyze():
    results = load_results()
    ok = [r for r in results if r["status"] == "ok"]
    errors = [r for r in results if r["status"] == "error"]

    print(f"Total models attempted: {len(results)}")
    print(f"Successful responses:   {len(ok)}")
    print(f"Errors (timeout/API):   {len(errors)}")
    print()

    # Score distribution
    scores = [r["scores"]["total"] for r in ok]
    dist = Counter(scores)
    print("Score distribution:")
    for s in range(9):
        count = dist.get(s, 0)
        bar = "#" * count
        print(f"  {s}/8: {count:3d} {bar}")
    print()

    # Per-criterion pass rates
    criteria = [
        "has_gateway_response",
        "has_eviction_decisions",
        "references_tensor_ids",
        "reasons_about_age",
        "reasons_about_faults",
        "answers_human",
        "separates_concerns",
        "uses_structured_output",
    ]
    print("Per-criterion pass rates:")
    for c in criteria:
        passed = sum(1 for r in ok if r["scores"].get(c))
        pct = 100 * passed / len(ok) if ok else 0
        print(f"  {c:30s} {passed:3d}/{len(ok)} ({pct:.1f}%)")
    print()

    # Model family breakdown
    families = {}
    for r in ok:
        family = r["model_id"].split("/")[0]
        families.setdefault(family, []).append(r["scores"]["total"])

    print("By model family (avg score, count):")
    family_stats = []
    for family, fscores in sorted(families.items()):
        avg = sum(fscores) / len(fscores)
        family_stats.append((avg, family, len(fscores)))
    for avg, family, count in sorted(family_stats, reverse=True):
        print(f"  {family:35s} avg={avg:.1f}/8  n={count}")
    print()

    # Perfect scores
    perfect = [r for r in ok if r["scores"]["total"] == 8]
    print(f"Perfect 8/8 ({len(perfect)} models):")
    for r in sorted(perfect, key=lambda r: r["model_id"]):
        print(f"  {r['model_id']}")
    print()

    # Failures (0-2)
    poor = [r for r in ok if r["scores"]["total"] <= 2]
    print(f"Poor 0-2/8 ({len(poor)} models):")
    for r in sorted(poor, key=lambda r: r["scores"]["total"]):
        print(f"  {r['scores']['total']}/8  {r['model_id']}")


if __name__ == "__main__":
    analyze()
