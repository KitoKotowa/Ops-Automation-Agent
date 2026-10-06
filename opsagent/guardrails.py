"""Guardrail policy. The LLM *proposes* actions; this module decides what may run. Pure code, fully unit-testable.

Verdicts:  auto -> execute now | needs_approval -> human queue | blocked -> never executed
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .detectors import OpsIssue, TicketContext
from .models import ActionType as A
from .models import Decision, ProposedAction, TriageResult, Verdict

# ---- policy knobs
AUTO_REFUND_MAX = 20.00        # refunds above this need a human
AUTO_RESHIP_MAX = 40.00        # reships of orders above this value need a human
RUN_AUTO_MONEY_BUDGET = 100.00  # total $ the agent may auto-spend per run (circuit breaker)
MIN_AUTO_CONFIDENCE = 0.75

MONEY_ACTIONS = {A.refund, A.reship}
ACCOUNT_ACTIONS = {A.cancel_subscription, A.pause_subscription}
INTERNAL_ACTIONS = {A.create_internal_ticket, A.notify_slack, A.escalate_to_human}
ALWAYS_APPROVAL = {A.cancel_subscription}  # retention + hard to undo
UNVERIFIED_FLAGS = {"unknown_customer", "order_not_found", "order_not_owned"}
ESCALATION_FLAGS = {"chargeback_threat", "legal_threat"}


@dataclass
class Budget:
    remaining: float = RUN_AUTO_MONEY_BUDGET

    def try_spend(self, amount: float) -> bool:
        if amount <= self.remaining:
            self.remaining -= amount
            return True
        return False


@dataclass
class _Eval:
    verdict: Verdict = Verdict.auto
    rules: list[str] = field(default_factory=list)

    def at_least(self, v: Verdict, rule: str):
        order = [Verdict.auto, Verdict.needs_approval, Verdict.blocked]
        if order.index(v) > order.index(self.verdict):
            self.verdict = v
        self.rules.append(rule)


def _money_value(a: ProposedAction, order_totals: dict[str, float]) -> float:
    if a.type == A.refund:
        return a.amount or 0.0
    if a.type == A.reship:
        return order_totals.get(a.order_id or "", 0.0)
    return 0.0


def mandatory_actions(ctx: TicketContext, proposed: list[ProposedAction]) -> list[ProposedAction]:
    """Escalations the policy requires regardless of what the model proposed."""
    have = {(a.type, a.team) for a in proposed}
    extra: list[ProposedAction] = []

    def need(t, team, reason):
        if (t, team) not in have:
            have.add((t, team))
            extra.append(ProposedAction(type=t, team=team, order_id=ctx.ticket.order_id, reason=f"[policy] {reason}"))

    if ctx.flags & ESCALATION_FLAGS:
        need(A.escalate_to_human, "cx_lead", "Chargeback/legal language requires a CX lead.")
        need(A.notify_slack, "cx_lead", "Chargeback/legal risk on a ticket.")
    if "health_safety" in ctx.flags:
        need(A.escalate_to_human, "quality", "Health/safety mention requires human review.")
    if "prompt_injection" in ctx.flags:
        need(A.escalate_to_human, "cx_lead", "Ticket contains instructions aimed at the agent (possible abuse).")
    return extra


def evaluate_ticket(triage: TriageResult, ctx: TicketContext, budget: Budget) -> list[Decision]:
    actions = list(triage.proposed_actions) + mandatory_actions(ctx, triage.proposed_actions)
    owned = {o.order_id: o for o in ctx.customer_orders}
    totals = {oid: o.total for oid, o in owned.items()}
    decisions: list[Decision] = []
    seen: set[tuple] = set()
    replies: list[ProposedAction] = []

    for a in actions:
        if a.type == A.reply_to_customer:
            replies.append(a)  # evaluated last: depends on the other verdicts
            continue
        e = _Eval()
        key = (a.type, a.order_id, a.team)
        if key in seen:
            e.at_least(Verdict.blocked, "duplicate action in same ticket")
        seen.add(key)

        if a.type in MONEY_ACTIONS | ACCOUNT_ACTIONS:
            if "prompt_injection" in ctx.flags:
                e.at_least(Verdict.blocked, "prompt-injection detected: no money/account actions")
            if ctx.flags & {"unknown_customer"}:
                e.at_least(Verdict.blocked, "customer identity not verified")
            if ctx.flags & ESCALATION_FLAGS:
                e.at_least(Verdict.needs_approval, "chargeback/legal risk: human decides remedies")
            if triage.confidence < MIN_AUTO_CONFIDENCE:
                e.at_least(Verdict.needs_approval, f"model confidence {triage.confidence:.2f} < {MIN_AUTO_CONFIDENCE}")

        if a.type in MONEY_ACTIONS:
            order = owned.get(a.order_id or "")
            if order is None:
                e.at_least(Verdict.blocked, f"order {a.order_id or '(none)'} not verified as this customer's order")
            elif a.type == A.refund:
                if not a.amount or a.amount <= 0:
                    e.at_least(Verdict.blocked, "refund amount missing or non-positive")
                elif a.amount > order.total:
                    e.at_least(Verdict.blocked, f"refund ${a.amount:.2f} exceeds order total ${order.total:.2f}")
                elif a.amount > AUTO_REFUND_MAX:
                    e.at_least(Verdict.needs_approval, f"refund > ${AUTO_REFUND_MAX:.0f} auto limit")
            elif a.type == A.reship:
                if order.status != "delivered":
                    e.at_least(Verdict.blocked, f"cannot reship: order status is {order.status}")
                elif order.total > AUTO_RESHIP_MAX:
                    e.at_least(Verdict.needs_approval, f"reship value ${order.total:.2f} > ${AUTO_RESHIP_MAX:.0f} auto limit")

        if a.type in ALWAYS_APPROVAL:
            e.at_least(Verdict.needs_approval, f"{a.type.value} always requires approval")

        if e.verdict == Verdict.auto and a.type in MONEY_ACTIONS:
            if not budget.try_spend(_money_value(a, totals)):
                e.at_least(Verdict.needs_approval, "per-run auto-spend budget exhausted")
        if not e.rules:
            e.rules.append("within policy")
        decisions.append(Decision(action=a, verdict=e.verdict, rules=e.rules))

    for r in replies[:1]:
        e = _Eval()
        if len(replies) > 1:
            e.rules.append(f"{len(replies) - 1} extra reply action(s) dropped")
        if any(d.verdict == Verdict.needs_approval and d.action.type not in INTERNAL_ACTIONS for d in decisions):
            e.at_least(Verdict.needs_approval, "reply may reference a remedy that is pending approval")
        risky = ctx.flags & (ESCALATION_FLAGS | {"health_safety", "prompt_injection"} | UNVERIFIED_FLAGS)
        if risky:
            e.at_least(Verdict.needs_approval, f"sensitive ticket ({', '.join(sorted(risky))}): human reviews reply")
        if triage.confidence < MIN_AUTO_CONFIDENCE:
            e.at_least(Verdict.needs_approval, f"model confidence {triage.confidence:.2f} < {MIN_AUTO_CONFIDENCE}")
        if not triage.customer_reply_draft.strip():
            e.at_least(Verdict.blocked, "empty reply draft")
        if e.verdict == Verdict.auto:
            e.rules.append("within policy")
        decisions.append(Decision(action=r, verdict=e.verdict, rules=e.rules))
    return decisions


def evaluate_issue(issue: OpsIssue, order_totals: dict[str, float], budget: Budget) -> list[Decision]:
    """Detector-proposed actions (no LLM involved) go through the same money rules."""
    out = []
    for a in issue.actions:
        e = _Eval()
        if a.type == A.refund:
            total: Optional[float] = order_totals.get(a.order_id or "")
            if total is None or not a.amount or a.amount > total:
                e.at_least(Verdict.blocked, "refund not backed by a matching order")
            elif a.amount > AUTO_REFUND_MAX:
                e.at_least(Verdict.needs_approval, f"refund > ${AUTO_REFUND_MAX:.0f} auto limit")
            elif not budget.try_spend(a.amount):
                e.at_least(Verdict.needs_approval, "per-run auto-spend budget exhausted")
        if a.type in ALWAYS_APPROVAL | {A.reship}:
            e.at_least(Verdict.needs_approval, f"{a.type.value} from a detector requires approval")
        if not e.rules:
            e.rules.append("within policy")
        out.append(Decision(action=a, verdict=e.verdict, rules=e.rules))
    return out
