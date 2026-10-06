"""Orchestration: validate -> detect -> triage (LLM) -> guardrails -> act/queue -> report."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from .actions import Executor
from .data import Dataset, Rejected
from .detectors import OpsIssue, TicketContext, build_ticket_context, scan_orders
from .guardrails import Budget, evaluate_issue, evaluate_ticket
from .models import ActionType as A
from .models import Decision, ProposedAction, TriageResult, Verdict
from .triage import Triager, TriageError


@dataclass
class TicketOutcome:
    ctx: TicketContext
    triage: Optional[TriageResult]
    error: Optional[str]
    results: list[tuple[Decision, str]] = field(default_factory=list)


@dataclass
class IssueOutcome:
    issue: OpsIssue
    results: list[tuple[Decision, str]] = field(default_factory=list)


@dataclass
class RunReport:
    triager: str
    now: datetime
    rejected: list[tuple[Rejected, str]] = field(default_factory=list)
    issues: list[IssueOutcome] = field(default_factory=list)
    tickets: list[TicketOutcome] = field(default_factory=list)
    budget_left: float = 0.0

    def counts(self) -> dict[str, int]:
        c = {v.value: 0 for v in Verdict}
        for group in (self.issues, self.tickets):
            for o in group:
                for d, _ in o.results:
                    c[d.verdict.value] += 1
        return c


def _safe_triage(triager: Triager, ctx: TicketContext) -> tuple[Optional[TriageResult], Optional[str]]:
    try:
        return triager.triage(ctx), None
    except TriageError as e:
        return None, str(e)
    except Exception as e:  # never let one bad ticket crash the run
        return None, f"unexpected triage failure: {type(e).__name__}: {e}"


def run(ds: Dataset, triager: Triager, executor: Executor, now: datetime, workers: int = 4) -> RunReport:
    report = RunReport(triager=triager.name, now=now)
    budget = Budget()

    # 1. Invalid input: quarantine + hand to a human. Never guessed at, never sent to the LLM.
    for r in ds.rejected:
        a = ProposedAction(type=A.create_internal_ticket, team="cx_lead",
                           reason=f"Malformed {r.source} record quarantined: {'; '.join(r.errors)}")
        d = Decision(action=a, verdict=Verdict.auto, rules=["invalid input -> manual review"])
        out = executor.handle(d, {"source_id": f"{r.source}:{r.record_id}", "summary": f"Invalid {r.source} record {r.record_id}",
                                  "urgency": "low"})
        report.rejected.append((r, out))

    # 2. Proactive detectors over orders/subscriptions (no LLM).
    totals = {o.order_id: o.total for o in ds.orders.values()}
    for issue in scan_orders(ds, now):
        io = IssueOutcome(issue)
        for d in evaluate_issue(issue, totals, budget):
            ctx = {"source_id": issue.issue_id, "summary": issue.summary, "urgency": issue.severity,
                   "customer_id": issue.customer_id}
            io.results.append((d, executor.handle(d, ctx)))
        report.issues.append(io)

    # 3. Tickets: deterministic context first, then LLM triage (parallel), then guardrails + actions (sequential).
    contexts = [build_ticket_context(t, ds, now) for t in ds.tickets]
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        triaged = list(pool.map(lambda c: _safe_triage(triager, c), contexts))

    for ctx, (tri, err) in zip(contexts, triaged):
        to = TicketOutcome(ctx, tri, err)
        exec_ctx = {"source_id": ctx.ticket.ticket_id, "customer_email": ctx.ticket.customer_email,
                    "customer_id": ctx.customer.customer_id if ctx.customer else None}
        if tri is None:
            # Fail closed: no automated customer contact, a human gets the ticket.
            d = Decision(action=ProposedAction(type=A.escalate_to_human, team="cx_lead",
                                               reason=f"Automated triage failed: {err}"),
                         verdict=Verdict.auto, rules=["triage failed -> fail closed to human"])
            to.results.append((d, executor.handle(d, exec_ctx | {"summary": f"Triage failed: {err}", "urgency": "medium"})))
        else:
            exec_ctx |= {"summary": tri.summary, "urgency": tri.urgency, "reply": tri.customer_reply_draft}
            for d in evaluate_ticket(tri, ctx, budget):
                to.results.append((d, executor.handle(d, exec_ctx)))
        report.tickets.append(to)

    report.budget_left = budget.remaining
    return report
