from __future__ import annotations

import json
import threading
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from prometheus_client import Counter, Histogram, generate_latest


REQ_TOTAL = Counter(
    "pichay_requests_total",
    "Total gateway requests",
    ["provider", "status"],
)

REQ_LATENCY_MS = Histogram(
    "pichay_request_latency_ms",
    "Gateway request latency milliseconds",
    ["provider"],
)

SHRINK_RATIO = Histogram(
    "pichay_shrink_ratio",
    "Outgoing/incoming payload ratio",
    ["provider"],
)

POLICY_CONFLICTS = Counter(
    "pichay_policy_conflicts_total",
    "Policy conflict resolution count",
    ["winner_stage", "loser_stage"],
)

ANOMALIES = Counter(
    "pichay_anomalies_total",
    "Data anomalies detected",
    ["kind"],
)

CACHE_READ = Counter(
    "pichay_cache_read_tokens_total",
    "Tokens read from provider cache",
    ["provider"],
)

CACHE_CREATE = Counter(
    "pichay_cache_create_tokens_total",
    "Tokens written to provider cache",
    ["provider"],
)

CACHE_MISS_EVENTS = Counter(
    "pichay_cache_miss_events_total",
    "Requests with zero cache read (unexpected misses)",
    ["provider"],
)


@dataclass
class SessionSummary:
    request_count: int = 0
    incoming_bytes: int = 0
    outgoing_bytes: int = 0


class Telemetry:
    def __init__(self, log_path: Path, hydration_window_seconds: int, max_events: int = 5000):
        self.log_path = log_path
        self.hydration_window_seconds = hydration_window_seconds
        self._lock = threading.Lock()
        self.events: deque[dict[str, Any]] = deque(maxlen=max_events)
        self.sessions: dict[str, SessionSummary] = defaultdict(SessionSummary)

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._hydrate()

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _hydrate(self) -> None:
        if not self.log_path.exists():
            return
        cutoff = datetime.now(timezone.utc).timestamp() - self.hydration_window_seconds
        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    ts = event.get("timestamp")
                    if not isinstance(ts, str):
                        continue
                    try:
                        epoch = datetime.fromisoformat(ts).timestamp()
                    except ValueError:
                        continue
                    if epoch < cutoff:
                        continue
                    self.events.append(event)
        except OSError:
            return

    def emit(self, event_type: str, **fields: Any) -> None:
        record = {
            "type": event_type,
            "timestamp": self._now(),
            **fields,
        }
        with self._lock:
            self.events.append(record)
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")

        if event_type == "policy_conflict_resolved":
            POLICY_CONFLICTS.labels(
                winner_stage=fields.get("winner_stage", "unknown"),
                loser_stage=fields.get("loser_stage", "unknown"),
            ).inc()
        if event_type == "anomaly":
            ANOMALIES.labels(kind=fields.get("kind", "unknown")).inc()

    def record_request(
        self,
        *,
        session_id: str,
        provider: str,
        status: int,
        incoming_bytes: int,
        outgoing_bytes: int,
        latency_ms: float,
        streaming: bool,
        model: str,
        request_id: str,
        duplication_score: float,
        usage: dict[str, Any] | None = None,
    ) -> None:
        shrink_ratio = (outgoing_bytes / incoming_bytes) if incoming_bytes > 0 else 1.0

        with self._lock:
            s = self.sessions[session_id]
            s.request_count += 1
            s.incoming_bytes += incoming_bytes
            s.outgoing_bytes += outgoing_bytes

        REQ_TOTAL.labels(provider=provider, status=str(status)).inc()
        REQ_LATENCY_MS.labels(provider=provider).observe(latency_ms)
        SHRINK_RATIO.labels(provider=provider).observe(shrink_ratio)

        # Cache analysis
        input_tokens = 0
        cache_read = 0
        cache_create = 0
        cache_read_pct = 0.0
        effective_tokens = 0
        if usage:
            input_tokens = usage.get("input_tokens", 0)
            cache_read = usage.get("cache_read_input_tokens", 0)
            cache_create = usage.get("cache_creation_input_tokens", 0)
            effective_tokens = input_tokens + cache_read + cache_create
            cache_read_pct = (cache_read / effective_tokens * 100) if effective_tokens > 0 else 0.0
            if cache_read:
                CACHE_READ.labels(provider=provider).inc(cache_read)
            if cache_create:
                CACHE_CREATE.labels(provider=provider).inc(cache_create)

        # Token-equivalent economics for subscription-style pricing:
        # cached read ~= 0.1x cost; miss ~= 1.0x cost.
        # A full miss therefore carries roughly a 0.9x premium on effective tokens.
        miss_penalty_tokens_est = 0.0
        if effective_tokens > 0 and cache_read == 0:
            miss_penalty_tokens_est = 0.9 * effective_tokens

        # Rough local savings signal from payload shrink (bytes to tokens @ ~4 bytes/token).
        size_saved_tokens_est = max(0.0, (incoming_bytes - outgoing_bytes) / 4.0)
        net_token_value_est = size_saved_tokens_est - miss_penalty_tokens_est

        self.emit(
            "request_metrics",
            request_id=request_id,
            session_id=session_id,
            provider=provider,
            model=model,
            status=status,
            incoming_bytes=incoming_bytes,
            outgoing_bytes=outgoing_bytes,
            shrink_ratio=shrink_ratio,
            duplication_score=duplication_score,
            latency_ms=latency_ms,
            streaming=streaming,
            input_tokens=input_tokens,
            cache_read_tokens=cache_read,
            cache_create_tokens=cache_create,
            effective_tokens=effective_tokens,
            cache_read_pct=round(cache_read_pct, 1),
            miss_penalty_tokens_est=round(miss_penalty_tokens_est, 1),
            size_saved_tokens_est=round(size_saved_tokens_est, 1),
            net_token_value_est=round(net_token_value_est, 1),
        )

        # Small increases are expected — Pichay injects tensor handles,
        # yuyay manifests, system reminders. Only flag when growth exceeds
        # 5% of incoming (suggesting duplicated message blocks, not injection).
        if incoming_bytes > 0 and outgoing_bytes > incoming_bytes:
            growth_pct = (outgoing_bytes - incoming_bytes) / incoming_bytes
            if growth_pct > 0.05:
                self.emit(
                    "anomaly",
                    kind="outgoing_growth_suspicious",
                    request_id=request_id,
                    session_id=session_id,
                    provider=provider,
                    incoming_bytes=incoming_bytes,
                    outgoing_bytes=outgoing_bytes,
                    growth_pct=round(growth_pct * 100, 1),
                    duplication_score=duplication_score,
                )

        # Flag unexpected cache misses: request had enough tokens to cache
        # but got zero cache reads. Skip first request per session (cold start).
        if usage and status == 200 and s.request_count > 1:
            if effective_tokens > 4096 and cache_read == 0 and cache_create > 0:
                CACHE_MISS_EVENTS.labels(provider=provider).inc()
                self.emit(
                    "cache_miss_unexpected",
                    request_id=request_id,
                    session_id=session_id,
                    provider=provider,
                    model=model,
                    effective_tokens=effective_tokens,
                    cache_create_tokens=cache_create,
                )

    def cost_summary(self, window_seconds: int | None = None) -> dict[str, Any]:
        events = self.recent_events(window_seconds)
        req = [e for e in events if e.get("type") == "request_metrics"]
        if not req:
            return {
                "requests": 0,
                "avg_cache_read_pct": 0.0,
                "zero_cache_read_requests": 0,
                "avg_effective_tokens": 0.0,
                "avg_miss_penalty_tokens_est": 0.0,
                "avg_size_saved_tokens_est": 0.0,
                "avg_net_token_value_est": 0.0,
            }
        n = len(req)
        return {
            "requests": n,
            "avg_cache_read_pct": round(sum(float(e.get("cache_read_pct", 0.0)) for e in req) / n, 2),
            "zero_cache_read_requests": sum(1 for e in req if float(e.get("cache_read_tokens", 0.0)) == 0.0),
            "avg_effective_tokens": round(sum(float(e.get("effective_tokens", 0.0)) for e in req) / n, 1),
            "avg_miss_penalty_tokens_est": round(sum(float(e.get("miss_penalty_tokens_est", 0.0)) for e in req) / n, 1),
            "avg_size_saved_tokens_est": round(sum(float(e.get("size_saved_tokens_est", 0.0)) for e in req) / n, 1),
            "avg_net_token_value_est": round(sum(float(e.get("net_token_value_est", 0.0)) for e in req) / n, 1),
        }

    def get_metrics(self) -> bytes:
        return generate_latest()

    def recent_events(self, window_seconds: int | None = None) -> list[dict[str, Any]]:
        if window_seconds is None:
            return list(self.events)
        cutoff = datetime.now(timezone.utc).timestamp() - window_seconds
        out: list[dict[str, Any]] = []
        for e in self.events:
            ts = e.get("timestamp")
            if not isinstance(ts, str):
                continue
            try:
                epoch = datetime.fromisoformat(ts).timestamp()
            except ValueError:
                continue
            if epoch >= cutoff:
                out.append(e)
        return out

    def session_summary(self) -> dict[str, dict[str, int]]:
        return {
            sid: {
                "request_count": s.request_count,
                "incoming_bytes": s.incoming_bytes,
                "outgoing_bytes": s.outgoing_bytes,
            }
            for sid, s in self.sessions.items()
        }
