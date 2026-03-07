"""Paging policy configuration.

Defaults derived from corpus analysis (68 sessions, 36K turns,
427T attention units):
  - Median avg context: 95K tokens
  - 85% of sessions hit 100K+
  - 96.3% of attention cost comes from sessions >80K avg
  - Compaction events drop 70-80% of context (amnesia, not curation)
  - Fault cost at 80K: 6.4G attn, at 165K: 27.2G (4.2x more expensive)

Three-tier policy:
  - Below floor: no eviction needed, let context grow
  - Advisory: inform the model, suggest curation (cooperative)
  - Involuntary: pager evicts by policy (assertive)
  - Hard cap: aggressive eviction, only pinned content survives

These are starting points. Each session that runs through the gateway
generates data that can refine them. The long-term path is
crowdsourced calibration across instances.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class PagingPolicy:
    """Thresholds for context paging behavior.

    All token counts refer to effective input tokens
    (input + cache_creation + cache_read).
    """

    # Context window size (tokens)
    window_size: int = 200_000

    # Below this: no eviction, let context grow freely
    floor_tokens: int = 60_000

    # Above this: advisory — inform model, suggest curation
    advisory_tokens: int = 80_000

    # Above this: involuntary eviction by age/size policy
    involuntary_tokens: int = 100_000

    # Above this: aggressive eviction, survival requires pins
    hard_cap_tokens: int = 120_000

    # Eviction parameters
    age_threshold: int = 4       # evict tool results older than N turns
    min_evict_size: int = 500    # don't evict results smaller than N bytes

    @property
    def floor_pct(self) -> float:
        return self.floor_tokens / self.window_size

    @property
    def advisory_pct(self) -> float:
        return self.advisory_tokens / self.window_size

    @property
    def involuntary_pct(self) -> float:
        return self.involuntary_tokens / self.window_size

    @property
    def hard_cap_pct(self) -> float:
        return self.hard_cap_tokens / self.window_size

    def zone(self, context_tokens: int) -> str:
        """Return the policy zone for a given context size."""
        if context_tokens >= self.hard_cap_tokens:
            return "aggressive"
        if context_tokens >= self.involuntary_tokens:
            return "involuntary"
        if context_tokens >= self.advisory_tokens:
            return "advisory"
        return "normal"

    def to_dict(self) -> dict:
        return asdict(self)


# Module-level default
_default = PagingPolicy()


def get_policy() -> PagingPolicy:
    """Return the current paging policy."""
    return _default


def set_policy(policy: PagingPolicy) -> None:
    """Replace the current paging policy."""
    global _default
    _default = policy


def load_policy(
    window_size: int | None = None,
    floor_tokens: int | None = None,
    advisory_tokens: int | None = None,
    involuntary_tokens: int | None = None,
    hard_cap_tokens: int | None = None,
    age_threshold: int | None = None,
    min_evict_size: int | None = None,
) -> PagingPolicy:
    """Build a policy from explicit overrides, falling back to defaults.

    Call with no arguments to get the default policy. Pass any subset
    of parameters to override specific thresholds. This is the
    integration point for CLI args, env vars, config files, or
    eventually a database of crowdsourced values.
    """
    defaults = PagingPolicy()
    policy = PagingPolicy(
        window_size=window_size if window_size is not None else defaults.window_size,
        floor_tokens=floor_tokens if floor_tokens is not None else defaults.floor_tokens,
        advisory_tokens=advisory_tokens if advisory_tokens is not None else defaults.advisory_tokens,
        involuntary_tokens=involuntary_tokens if involuntary_tokens is not None else defaults.involuntary_tokens,
        hard_cap_tokens=hard_cap_tokens if hard_cap_tokens is not None else defaults.hard_cap_tokens,
        age_threshold=age_threshold if age_threshold is not None else defaults.age_threshold,
        min_evict_size=min_evict_size if min_evict_size is not None else defaults.min_evict_size,
    )
    set_policy(policy)
    return policy


def add_policy_args(parser) -> None:
    """Add paging policy arguments to an argparse parser."""
    group = parser.add_argument_group("paging policy")
    group.add_argument(
        "--window-size", type=int, default=None,
        help="Context window size in tokens (default: 200000)",
    )
    group.add_argument(
        "--floor-tokens", type=int, default=None,
        help="Below this: no eviction (default: 60000)",
    )
    group.add_argument(
        "--advisory-tokens", type=int, default=None,
        help="Above this: suggest curation to model (default: 80000)",
    )
    group.add_argument(
        "--involuntary-tokens", type=int, default=None,
        help="Above this: auto-evict by policy (default: 100000)",
    )
    group.add_argument(
        "--hard-cap-tokens", type=int, default=None,
        help="Above this: aggressive eviction (default: 120000)",
    )
    group.add_argument(
        "--age-threshold", type=int, default=None,
        help="Evict tool results older than N turns (default: 4)",
    )
    group.add_argument(
        "--min-evict-size", type=int, default=None,
        help="Don't evict results smaller than N bytes (default: 500)",
    )


def policy_from_args(args) -> PagingPolicy:
    """Build a PagingPolicy from parsed argparse args."""
    return load_policy(
        window_size=getattr(args, "window_size", None),
        floor_tokens=getattr(args, "floor_tokens", None),
        advisory_tokens=getattr(args, "advisory_tokens", None),
        involuntary_tokens=getattr(args, "involuntary_tokens", None),
        hard_cap_tokens=getattr(args, "hard_cap_tokens", None),
        age_threshold=getattr(args, "age_threshold", None),
        min_evict_size=getattr(args, "min_evict_size", None),
    )
