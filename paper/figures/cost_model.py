#!/usr/bin/env python3
"""Visualize the inverted cost model for the Pichay paper.

Two figures:
1. Policy gradient — fault cost vs keep cost as a function of context fill.
   Shows the crossover where eviction policy should shift from aggressive
   to conservative.

2. Cumulative attention cost — baseline (no eviction) vs managed, from
   actual log data if provided, or synthetic 100-turn session if not.

Usage:
    # Theoretical curves only (no log data needed)
    uv run python paper/figures/cost_model.py

    # With actual log data overlaid
    uv run python paper/figures/cost_model.py --log experiments/baseline_run2/logs/proxy_*.jsonl

    # Save to PDF for paper
    uv run python paper/figures/cost_model.py --save
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WINDOW_SIZE = 200_000  # tokens
OBJECT_SIZE = 2_000    # representative eviction candidate (tokens)


def _fmt_tokens(x: float, _pos: int | None = None) -> str:
    if x >= 1e6:
        return f"{x / 1e6:.0f}M"
    if x >= 1e3:
        return f"{x / 1e3:.0f}K"
    return f"{x:.0f}"


def _fmt_cost(x: float, _pos: int | None = None) -> str:
    if x >= 1e12:
        return f"{x / 1e12:.1f}T"
    if x >= 1e9:
        return f"{x / 1e9:.1f}G"
    if x >= 1e6:
        return f"{x / 1e6:.1f}M"
    if x >= 1e3:
        return f"{x / 1e3:.1f}K"
    return f"{x:.0f}"


# ---------------------------------------------------------------------------
# Figure 1: Policy gradient
# ---------------------------------------------------------------------------

def plot_policy_gradient(ax: plt.Axes, object_size: int = OBJECT_SIZE) -> None:
    """Fault cost vs marginal keep cost as a function of context fill.

    Keep cost (per turn): the marginal contribution of |p| tokens to
    attention. d/dp (n+p)² ≈ 2np for p << n. This is what you save per
    turn by evicting the object.

    Fault cost: one additional full inference pass at (n + |p|)² ≈ n².
    This is what you pay if you evict and then need it back.
    """
    n = np.linspace(1_000, WINDOW_SIZE * 0.9, 500)
    p = object_size

    # Marginal keep cost per turn: 2 * n * p (derivative of (n+p)² - n²)
    keep_cost = 2 * n * p

    # Fault cost: full inference pass at n² (the additional API call)
    fault_cost = n * n

    # Normalize both to make the crossover visible
    scale = fault_cost.max()
    keep_norm = keep_cost / scale
    fault_norm = fault_cost / scale

    ax.plot(n, keep_norm, label=f"Keep cost / turn (|p|={_fmt_tokens(p)})",
            color="#2196F3", linewidth=2)
    ax.plot(n, fault_norm, label="Fault cost (one extra pass)",
            color="#F44336", linewidth=2)

    # Find crossover
    # keep_cost = fault_cost when 2np = n², i.e. n = 2p
    crossover_n = 2 * p
    crossover_y = crossover_n * crossover_n / scale
    ax.axvline(crossover_n, color="#888888", linestyle="--", alpha=0.5)
    ax.annotate(
        f"Crossover at n={_fmt_tokens(crossover_n)}",
        xy=(crossover_n, crossover_y),
        xytext=(crossover_n + WINDOW_SIZE * 0.08, crossover_y + 0.15),
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="#888888"),
    )

    # Shade regions
    ax.axvspan(0, crossover_n, alpha=0.05, color="#2196F3",
               label="Evict freely")
    ax.axvspan(crossover_n, WINDOW_SIZE * 0.9, alpha=0.05, color="#F44336",
               label="Evict conservatively")

    ax.set_xlabel("Context size (tokens)")
    ax.set_ylabel("Normalized cost")
    ax.set_title("Policy Gradient: When Eviction Becomes Risky")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(_fmt_tokens))
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)


# ---------------------------------------------------------------------------
# Figure 2: Cumulative attention cost over a session
# ---------------------------------------------------------------------------

def _synthetic_session(
    n_turns: int = 100,
    growth_per_turn: int = 1_500,
    eviction_target: int = 80_000,
    fault_rate: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic baseline vs managed attention curves.

    Returns (turns, baseline_cumulative, managed_cumulative).
    """
    turns = np.arange(1, n_turns + 1)

    # Baseline: context grows linearly, no eviction
    baseline_n = growth_per_turn * turns
    baseline_attention = np.cumsum(baseline_n ** 2)

    # Managed: context capped near eviction_target
    managed_n = np.minimum(baseline_n, eviction_target + growth_per_turn * 2)
    managed_attention = np.cumsum(managed_n ** 2)

    # Add fault overhead: fault_rate of turns trigger a full extra pass
    rng = np.random.default_rng(42)
    faults = rng.random(n_turns) < fault_rate
    fault_cost = np.where(faults, managed_n ** 2, 0)
    managed_attention = managed_attention + np.cumsum(fault_cost)

    return turns, baseline_attention, managed_attention


def plot_cumulative_attention(
    ax: plt.Axes,
    log_data: dict | None = None,
) -> None:
    """Cumulative attention cost: baseline vs managed.

    Uses log data if provided, otherwise synthetic session.
    """
    if log_data is not None:
        turns = np.array(log_data["turns"])
        baseline = np.array(log_data["baseline_cumulative"])
        managed = np.array(log_data["managed_cumulative"])
        source = "observed"
    else:
        turns, baseline, managed = _synthetic_session()
        source = "synthetic (100-turn session)"

    ax.plot(turns, baseline, label="Baseline (no eviction)",
            color="#F44336", linewidth=2)
    ax.plot(turns, managed, label="Managed (with faults)",
            color="#2196F3", linewidth=2)

    # Shade savings
    ax.fill_between(turns, managed, baseline, alpha=0.1, color="#4CAF50",
                     label="Attention savings")

    # Annotate final savings
    savings_pct = (1 - managed[-1] / baseline[-1]) * 100
    ax.annotate(
        f"{savings_pct:.0f}% savings",
        xy=(turns[-1], (baseline[-1] + managed[-1]) / 2),
        fontsize=11,
        fontweight="bold",
        color="#4CAF50",
        ha="right",
    )

    ax.set_xlabel("Turn")
    ax.set_ylabel("Cumulative attention cost")
    ax.set_title(f"Cumulative Attention Cost Over Session ({source})")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(_fmt_cost))
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)


# ---------------------------------------------------------------------------
# Figure 3: Fault cost by fill level (the counter-intuitive curve)
# ---------------------------------------------------------------------------

def plot_fault_cost_by_fill(ax: plt.Axes) -> None:
    """Show absolute fault cost at different fill levels.

    This is the key insight: a page fault at 20% fill costs 4% of max
    compute. At 80% fill, it costs 64%. Same operation, wildly different
    price.
    """
    fill_pct = np.linspace(5, 95, 100)
    n = (fill_pct / 100) * WINDOW_SIZE

    # Fault cost: n² (one extra inference pass)
    fault_cost = n ** 2
    max_cost = WINDOW_SIZE ** 2

    # Express as percentage of maximum possible fault cost
    fault_pct = (fault_cost / max_cost) * 100

    ax.plot(fill_pct, fault_pct, color="#F44336", linewidth=2.5)
    ax.fill_between(fill_pct, 0, fault_pct, alpha=0.1, color="#F44336")

    # Mark key points
    for pct in [20, 50, 85]:
        cost = (pct / 100) ** 2 * 100
        ax.plot(pct, cost, "o", color="#F44336", markersize=8)
        ax.annotate(
            f"{pct}% fill → {cost:.0f}% cost",
            xy=(pct, cost),
            xytext=(pct + 5, cost + 8),
            fontsize=9,
            arrowprops=dict(arrowstyle="->", color="#888888"),
        )

    ax.set_xlabel("Context fill (%)")
    ax.set_ylabel("Fault cost (% of maximum)")
    ax.set_title("Page Fault Cost Scales Quadratically with Fill Level")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Cost model figures for Pichay paper")
    parser.add_argument("--log", type=Path, nargs="*", help="Proxy log(s) for real data")
    parser.add_argument("--save", action="store_true", help="Save to PDF")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent,
                        help="Output directory for PDFs")
    args = parser.parse_args()

    # Load log data if provided
    log_data = None
    if args.log:
        # Import cost module for log parsing
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
        from pichay.cost import compute_baseline_cost, simulate_managed_cost, compare

        for log_path in args.log:
            if not log_path.exists():
                print(f"Warning: {log_path} not found", file=sys.stderr)
                continue
            baseline = compute_baseline_cost(log_path)
            managed = simulate_managed_cost(log_path)
            log_data = {
                "turns": [t.turn for t in baseline.turns],
                "baseline_cumulative": [t.cumulative_attention_cost for t in baseline.turns],
                "managed_cumulative": [
                    t.cumulative_attention_cost + sum(
                        mt.fault_attention_cost for mt in managed.turns[:i + 1]
                    )
                    for i, t in enumerate(managed.turns)
                ],
            }
            break  # use first valid log

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("The Inverted Cost Model", fontsize=14, fontweight="bold")

    plot_policy_gradient(axes[0])
    plot_cumulative_attention(axes[1], log_data)
    plot_fault_cost_by_fill(axes[2])

    plt.tight_layout()

    if args.save:
        out = args.output_dir / "cost_model.pdf"
        fig.savefig(out, bbox_inches="tight", dpi=300)
        print(f"Saved: {out}")

        # Also save individual figures for LaTeX \includegraphics
        for i, name in enumerate(["policy_gradient", "cumulative_attention", "fault_cost_fill"]):
            fig_single, ax_single = plt.subplots(figsize=(6, 4))
            if i == 0:
                plot_policy_gradient(ax_single)
            elif i == 1:
                plot_cumulative_attention(ax_single, log_data)
            else:
                plot_fault_cost_by_fill(ax_single)
            fig_single.tight_layout()
            single_out = args.output_dir / f"{name}.pdf"
            fig_single.savefig(single_out, bbox_inches="tight", dpi=300)
            print(f"Saved: {single_out}")
            plt.close(fig_single)
    else:
        plt.show()


if __name__ == "__main__":
    main()
