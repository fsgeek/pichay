"""Cost simulation for context paging under the inverted cost model.

Computes three cost metrics from proxy JSONL logs:

1. Cumulative token cost: Σ n_t (what the API charges for)
2. Cumulative attention cost: Σ n_t² (proportional to actual compute)
3. Fault-adjusted cost: attention cost including extra inference
   passes from page faults at (n_t + |p|)²

The inverted cost model (Section 6.4 of the paper): keeping is
expensive, faulting is cheap. A page sitting in context for T turns
costs |p| · T tokens of processing. Faulting it back costs one extra
inference pass at the current context size — O(n²), not O(|p|).

This produces a counter-intuitive policy gradient:
  - Low fill: faults cheap → evict aggressively
  - High fill: faults expensive → evict conservatively

Usage:
    # Simulate cost on a proxy log (uses actual token counts)
    uv run python -m pichay.cost experiments/baseline_run2/logs/proxy_*.jsonl

    # Replay with eviction and compare
    uv run python -m pichay.cost --replay --age-threshold 4 \\
        experiments/baseline_run2/logs/proxy_*.jsonl

    # JSON output
    uv run python -m pichay.cost --json experiments/*/logs/proxy_*.jsonl
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

from pichay.eval import parse_proxy_log
from pichay.pager import PageStore, compact_messages
from pichay.replay import _apply_evictions


@dataclass
class TurnCost:
    """Cost metrics for a single API call."""

    turn: int
    timestamp: str = ""
    # Context size (effective input tokens)
    context_tokens: int = 0
    # Linear cost (what you pay for)
    token_cost: int = 0
    cumulative_token_cost: int = 0
    # Quadratic attention cost (proportional to compute)
    attention_cost: int = 0
    cumulative_attention_cost: int = 0
    # Eviction/fault info
    evictions: int = 0
    faults: int = 0
    fault_tokens: int = 0  # extra tokens from fault inference passes
    fault_attention_cost: int = 0  # extra attention cost from faults


@dataclass
class CostSummary:
    """Aggregate cost metrics for one simulation."""

    label: str
    log_path: str = ""
    total_turns: int = 0
    # Linear costs
    cumulative_token_cost: int = 0
    # Quadratic costs
    cumulative_attention_cost: int = 0
    # Fault overhead
    total_fault_attention_cost: int = 0
    total_faults: int = 0
    total_evictions: int = 0
    # Context size stats
    max_context_tokens: int = 0
    avg_context_tokens: float = 0.0
    # Per-turn data
    turns: list[TurnCost] = field(default_factory=list)


@dataclass
class CostComparison:
    """Side-by-side comparison of baseline vs managed costs."""

    baseline: CostSummary
    managed: CostSummary

    @property
    def token_savings_pct(self) -> float:
        if self.baseline.cumulative_token_cost == 0:
            return 0.0
        saved = self.baseline.cumulative_token_cost - self.managed.cumulative_token_cost
        return saved / self.baseline.cumulative_token_cost

    @property
    def attention_savings_pct(self) -> float:
        if self.baseline.cumulative_attention_cost == 0:
            return 0.0
        saved = self.baseline.cumulative_attention_cost - self.managed.cumulative_attention_cost
        return saved / self.baseline.cumulative_attention_cost

    @property
    def net_attention_savings_pct(self) -> float:
        """Attention savings after accounting for fault costs."""
        if self.baseline.cumulative_attention_cost == 0:
            return 0.0
        managed_total = (
            self.managed.cumulative_attention_cost
            + self.managed.total_fault_attention_cost
        )
        saved = self.baseline.cumulative_attention_cost - managed_total
        return saved / self.baseline.cumulative_attention_cost


def _effective_input(usage: dict) -> int:
    """Extract effective input tokens from a usage dict."""
    return (
        usage.get("input_tokens", 0)
        + usage.get("cache_creation_input_tokens", 0)
        + usage.get("cache_read_input_tokens", 0)
    )


def _detect_log_format(records: list[dict]) -> str:
    """Detect whether records are proxy format or native Claude Code format.

    Proxy format: {"type": "request", ...} / {"type": "response_stream", ...}
    Native format: {"type": "assistant", "message": {"usage": ...}, ...}
    """
    for rec in records[:10]:
        if rec.get("type") in ("request", "response_stream", "response"):
            return "proxy"
        if rec.get("type") == "assistant" and "message" in rec:
            return "native"
    return "unknown"


def _extract_native_turns(records: list[dict]) -> list[tuple[str, int]]:
    """Extract (timestamp, effective_input_tokens) from native Claude Code logs.

    Native logs have one record per assistant message, each with usage
    data embedded in message.usage. We extract the effective input
    tokens (input + cache_creation + cache_read) from each.
    """
    turns = []
    for rec in records:
        if rec.get("type") != "assistant":
            continue
        msg = rec.get("message", {})
        if not isinstance(msg, dict):
            continue
        usage = msg.get("usage", {})
        if not usage:
            continue
        n = _effective_input(usage)
        if n > 0:
            ts = rec.get("timestamp", "")
            turns.append((ts, n))
    return turns


def compute_baseline_cost(path: Path, label: str = "") -> CostSummary:
    """Compute cost metrics from a proxy or native Claude Code log.

    Accepts both formats:
    - Proxy JSONL: request/response pairs with usage in response
    - Native Claude Code JSONL: assistant messages with usage in message
    """
    records = parse_proxy_log(path)
    if not label:
        label = path.stem

    fmt = _detect_log_format(records)

    summary = CostSummary(label=label, log_path=str(path))
    cum_token = 0
    cum_attention = 0
    turn_num = 0
    context_sizes = []

    if fmt == "native":
        turn_data = _extract_native_turns(records)
        for ts, n in turn_data:
            turn_num += 1
            cum_token += n
            attention = n * n
            cum_attention += attention
            context_sizes.append(n)

            turn = TurnCost(
                turn=turn_num,
                timestamp=ts,
                context_tokens=n,
                token_cost=n,
                cumulative_token_cost=cum_token,
                attention_cost=attention,
                cumulative_attention_cost=cum_attention,
            )
            summary.turns.append(turn)
    else:
        # Proxy format: pair requests with responses
        pending_request = None
        for rec in records:
            if rec.get("type") == "request":
                pending_request = rec
                continue

            if rec.get("type") in ("response_stream", "response") and pending_request:
                turn_num += 1
                usage = rec.get("usage", {})
                n = _effective_input(usage)

                cum_token += n
                attention = n * n
                cum_attention += attention
                context_sizes.append(n)

                turn = TurnCost(
                    turn=turn_num,
                    timestamp=rec.get("timestamp", ""),
                    context_tokens=n,
                    token_cost=n,
                    cumulative_token_cost=cum_token,
                    attention_cost=attention,
                    cumulative_attention_cost=cum_attention,
                )
                summary.turns.append(turn)
                pending_request = None

    summary.total_turns = turn_num
    summary.cumulative_token_cost = cum_token
    summary.cumulative_attention_cost = cum_attention
    if context_sizes:
        summary.max_context_tokens = max(context_sizes)
        summary.avg_context_tokens = sum(context_sizes) / len(context_sizes)

    return summary


def simulate_managed_cost(
    path: Path,
    age_threshold: int = 4,
    min_size: int = 500,
    label: str = "",
) -> CostSummary:
    """Simulate cost with eviction, including fault overhead.

    Replays the proxy log with compaction, then estimates:
    - Reduced context size per turn (from eviction)
    - Extra inference passes from faults (at quadratic cost)

    Fault cost model: each fault triggers one additional inference
    pass over the full context (n + |p|)² ≈ n² for large n.
    """
    records = parse_proxy_log(path)
    if not label:
        label = f"{path.stem}_managed"

    summary = CostSummary(label=label, log_path=str(path))

    requests = [
        r for r in records
        if r.get("type") == "request" and "messages_full" in r
    ]
    # Pair with responses for actual token counts
    responses = []
    pending = None
    for rec in records:
        if rec.get("type") == "request":
            pending = rec
        elif rec.get("type") in ("response_stream", "response") and pending:
            responses.append(rec)
            pending = None

    if not requests:
        return summary

    page_store = PageStore()
    cum_token = 0
    cum_attention = 0
    total_fault_attention = 0
    context_sizes = []

    for turn_idx, req in enumerate(requests):
        messages = copy.deepcopy(req["messages_full"])

        # Apply previous evictions
        _apply_evictions(messages, page_store)

        # Detect faults before compaction
        faults = page_store.detect_faults(messages)

        # Run compaction
        stats = compact_messages(
            messages,
            age_threshold=age_threshold,
            min_size=min_size,
            page_store=page_store,
        )

        # Estimate managed context size:
        # Use the ratio of compacted/original bytes to scale the
        # actual token count from the API response.
        bytes_compacted = len(json.dumps(messages).encode("utf-8"))
        bytes_original = len(
            json.dumps(req["messages_full"]).encode("utf-8")
        )

        if turn_idx < len(responses):
            usage = responses[turn_idx].get("usage", {})
            n_baseline = _effective_input(usage)
        else:
            n_baseline = 0

        if bytes_original > 0 and n_baseline > 0:
            ratio = bytes_compacted / bytes_original
            n_managed = max(1, int(n_baseline * ratio))
        else:
            n_managed = n_baseline

        # Token cost (linear)
        cum_token += n_managed

        # Attention cost (quadratic) for the main inference
        attention = n_managed * n_managed
        cum_attention += attention

        # Fault cost: each fault is an extra inference pass
        # The fault restores |p| tokens into a context of size n_managed
        fault_attention = 0
        fault_tokens = 0
        for fault in faults:
            # Estimate restored page size from the fault
            p_size = getattr(fault, 'original_size', 0) or 500
            p_tokens = p_size // 4  # rough bytes-to-tokens
            n_with_fault = n_managed + p_tokens
            fault_attention += n_with_fault * n_with_fault
            fault_tokens += n_with_fault

        total_fault_attention += fault_attention
        context_sizes.append(n_managed)

        turn = TurnCost(
            turn=turn_idx + 1,
            timestamp=req.get("timestamp", ""),
            context_tokens=n_managed,
            token_cost=n_managed,
            cumulative_token_cost=cum_token,
            attention_cost=attention,
            cumulative_attention_cost=cum_attention,
            evictions=stats.evicted_count,
            faults=len(faults),
            fault_tokens=fault_tokens,
            fault_attention_cost=fault_attention,
        )
        summary.turns.append(turn)

    summary.total_turns = len(requests)
    summary.cumulative_token_cost = cum_token
    summary.cumulative_attention_cost = cum_attention
    summary.total_fault_attention_cost = total_fault_attention
    summary.total_faults = sum(t.faults for t in summary.turns)
    summary.total_evictions = sum(t.evictions for t in summary.turns)
    if context_sizes:
        summary.max_context_tokens = max(context_sizes)
        summary.avg_context_tokens = sum(context_sizes) / len(context_sizes)

    return summary


def compare(baseline: CostSummary, managed: CostSummary) -> CostComparison:
    """Build a comparison between baseline and managed runs."""
    return CostComparison(baseline=baseline, managed=managed)


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def _fmt_attention(n: int) -> str:
    """Format attention cost in human-readable units."""
    if n >= 1e12:
        return f"{n / 1e12:.2f}T"
    if n >= 1e9:
        return f"{n / 1e9:.2f}G"
    if n >= 1e6:
        return f"{n / 1e6:.2f}M"
    if n >= 1e3:
        return f"{n / 1e3:.1f}K"
    return str(n)


def print_cost_summary(s: CostSummary) -> None:
    """Print cost summary for one simulation."""
    print(f"\n{'=' * 60}")
    print(f"Cost: {s.label}")
    print(f"{'=' * 60}")
    print(f"Turns:                   {s.total_turns:>10,}")
    print(f"Cumulative tokens:       {s.cumulative_token_cost:>10,}")
    print(f"Cumulative attention:    {_fmt_attention(s.cumulative_attention_cost):>10s}")
    print(f"Max context (tokens):    {s.max_context_tokens:>10,}")
    print(f"Avg context (tokens):    {s.avg_context_tokens:>10,.0f}")

    if s.total_evictions > 0:
        print(f"Evictions:               {s.total_evictions:>10,}")
        print(f"Faults:                  {s.total_faults:>10,}")
        print(f"Fault attention cost:    {_fmt_attention(s.total_fault_attention_cost):>10s}")

    # Per-turn curve
    if len(s.turns) > 3:
        print(f"\nPer-turn context and attention:")
        print(f"  {'Turn':>4s}  {'Context':>10s}  {'Attention':>12s}  {'Cum Attn':>12s}  {'Evict':>5s}  {'Fault':>5s}")
        step = max(1, len(s.turns) // 20)
        for i, t in enumerate(s.turns):
            if i % step == 0 or i == len(s.turns) - 1:
                fault_str = ""
                if t.fault_attention_cost > 0:
                    fault_str = f"  +{_fmt_attention(t.fault_attention_cost)}"
                print(
                    f"  T{t.turn:>3d}  {t.context_tokens:>10,}  "
                    f"{_fmt_attention(t.attention_cost):>12s}  "
                    f"{_fmt_attention(t.cumulative_attention_cost):>12s}  "
                    f"{t.evictions:>5d}  {t.faults:>5d}{fault_str}"
                )


def print_comparison(comp: CostComparison) -> None:
    """Print side-by-side cost comparison."""
    b, m = comp.baseline, comp.managed

    print(f"\n{'=' * 65}")
    print("COST COMPARISON")
    print(f"{'=' * 65}")

    def row(label: str, bval: str, mval: str, delta: str = ""):
        print(f"  {label:<28s}  {bval:>14s}  {mval:>14s}  {delta:>10s}")

    row("", b.label, m.label, "delta")
    print(f"  {'-' * 28}  {'-' * 14}  {'-' * 14}  {'-' * 10}")

    row("Turns", str(b.total_turns), str(m.total_turns))
    row(
        "Cumulative tokens",
        f"{b.cumulative_token_cost:,}",
        f"{m.cumulative_token_cost:,}",
        f"{comp.token_savings_pct:+.1%}",
    )
    row(
        "Cumulative attention",
        _fmt_attention(b.cumulative_attention_cost),
        _fmt_attention(m.cumulative_attention_cost),
        f"{comp.attention_savings_pct:+.1%}",
    )
    row(
        "Fault attention overhead",
        "0",
        _fmt_attention(m.total_fault_attention_cost),
    )
    row(
        "Net attention (incl faults)",
        _fmt_attention(b.cumulative_attention_cost),
        _fmt_attention(m.cumulative_attention_cost + m.total_fault_attention_cost),
        f"{comp.net_attention_savings_pct:+.1%}",
    )
    row(
        "Max context",
        f"{b.max_context_tokens:,}",
        f"{m.max_context_tokens:,}",
    )
    row(
        "Avg context",
        f"{b.avg_context_tokens:,.0f}",
        f"{m.avg_context_tokens:,.0f}",
    )
    row("Evictions", "0", f"{m.total_evictions:,}")
    row("Faults", "0", f"{m.total_faults:,}")

    # Dollar cost estimate (Opus pricing)
    def _dollar(s: CostSummary) -> float:
        # Simplified: all effective input at $15/M
        return (s.cumulative_token_cost / 1e6) * 15.0

    b_cost = _dollar(b)
    m_cost = _dollar(m)
    print(f"\n  Est. input cost (Opus $15/M):")
    print(f"    Baseline:  ${b_cost:,.2f}")
    print(f"    Managed:   ${m_cost:,.2f}")
    if b_cost > 0:
        print(f"    Savings:   ${b_cost - m_cost:,.2f} ({(b_cost - m_cost) / b_cost:.1%})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Cost simulation under the inverted cost model"
    )
    parser.add_argument(
        "logs",
        type=Path,
        nargs="+",
        help="Proxy JSONL log file(s)",
    )
    parser.add_argument(
        "--replay",
        action="store_true",
        help="Also simulate managed cost with eviction and compare",
    )
    parser.add_argument(
        "--age-threshold",
        type=int,
        default=4,
        help="Evict tool results older than N turns (default: 4)",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=500,
        help="Don't evict results smaller than N bytes (default: 500)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )
    args = parser.parse_args()

    for log_path in args.logs:
        if not log_path.exists():
            print(f"Warning: {log_path} not found, skipping.", file=sys.stderr)
            continue

        baseline = compute_baseline_cost(log_path)

        if args.replay:
            managed = simulate_managed_cost(
                log_path,
                age_threshold=args.age_threshold,
                min_size=args.min_size,
            )
            comp = compare(baseline, managed)

            if args.json:
                out = {
                    "baseline": asdict(baseline),
                    "managed": asdict(managed),
                    "token_savings_pct": comp.token_savings_pct,
                    "attention_savings_pct": comp.attention_savings_pct,
                    "net_attention_savings_pct": comp.net_attention_savings_pct,
                }
                # Trim per-turn data for JSON
                for key in ("baseline", "managed"):
                    out[key]["turn_count"] = len(out[key].pop("turns"))
                print(json.dumps(out, indent=2))
            else:
                print_comparison(comp)
        else:
            if args.json:
                out = asdict(baseline)
                out["turn_count"] = len(out.pop("turns"))
                print(json.dumps(out, indent=2))
            else:
                print_cost_summary(baseline)


if __name__ == "__main__":
    main()
