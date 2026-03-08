from __future__ import annotations

from pichay.core.models import CanonicalRequest, PolicyContext
from pichay.core.policy import (
    PolicyConfig,
    apply_action,
    paging_stage,
    phantom_stage,
    trim_stage,
)


PRECEDENCE = {"phantom": 3, "paging": 2, "trim": 1}


class Pipeline:
    def __init__(self, cfg: PolicyConfig, emit_event):
        self.cfg = cfg
        self.emit_event = emit_event

    def run(self, req: CanonicalRequest) -> CanonicalRequest:
        ctx = PolicyContext()
        ctx, phantom_actions = phantom_stage(req, ctx)
        paging_actions = paging_stage(req, ctx, self.cfg)
        trim_actions = trim_stage(req, ctx, self.cfg)

        # Phantom only marks protections in v1.
        _ = phantom_actions

        taken_targets: set[str] = set()
        ordered = sorted(
            paging_actions + trim_actions,
            key=lambda a: PRECEDENCE.get(a.stage, 0),
            reverse=True,
        )

        for action in ordered:
            if action.target_id in ctx.protected_targets:
                self.emit_event(
                    "policy_conflict_resolved",
                    winner_stage="phantom",
                    loser_stage=action.stage,
                    loser_action=action.action,
                    target_id=action.target_id,
                    target_bytes=action.bytes,
                    duplication_score=action.duplication_score,
                    resolution_reason="phantom_protection",
                )
                continue
            if action.target_id in taken_targets:
                continue
            applied = apply_action(req, action)
            if applied:
                taken_targets.add(action.target_id)
                self.emit_event(
                    "policy_action_applied",
                    stage=action.stage,
                    action=action.action,
                    target_id=action.target_id,
                    target_bytes=action.bytes,
                    duplication_score=action.duplication_score,
                )

        return req
