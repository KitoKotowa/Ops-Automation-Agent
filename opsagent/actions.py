"""Simulated side effects. Every 'integration' (Slack, helpdesk, payments, email, subscriptions) appends JSON lines
to outbox/. Swapping a writer for a real API client is the only change needed to go live.

Safety properties:
- idempotency: each action has a key (e.g. refund:O1005); it can execute at most once across runs and sources.
- approval queue: needs_approval actions are persisted and only run via `opsagent approve <id>`.
- audit log: every decision (auto, queued, blocked, deduped) is recorded with the rules that produced it.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .models import ActionType as A
from .models import Decision, ProposedAction, Verdict

SINKS = {
    A.reply_to_customer: "emails_sent.jsonl",
    A.refund: "payments.jsonl",
    A.reship: "fulfillment_orders.jsonl",
    A.cancel_subscription: "subscriptions.jsonl",
    A.pause_subscription: "subscriptions.jsonl",
    A.create_internal_ticket: "helpdesk_tickets.jsonl",
    A.notify_slack: "slack.jsonl",
    A.escalate_to_human: "helpdesk_tickets.jsonl",
}


def action_key(a: ProposedAction, source_id: str, customer_id: Optional[str]) -> str:
    if a.type in (A.refund, A.reship):
        return f"{a.type.value}:{a.order_id}"
    if a.type in (A.cancel_subscription, A.pause_subscription):
        return f"{a.type.value}:{customer_id or source_id}"
    if a.type in (A.create_internal_ticket, A.notify_slack):
        return f"{a.type.value}:{a.team}:{a.order_id or source_id}"
    return f"{a.type.value}:{a.team or ''}:{source_id}"


class Executor:
    def __init__(self, outbox: Path, dry_run: bool = False):
        self.outbox = outbox
        self.dry_run = dry_run
        outbox.mkdir(parents=True, exist_ok=True)
        self.state_path = outbox / "state.json"
        self.state = (json.loads(self.state_path.read_text(encoding="utf-8")) if self.state_path.exists()
                      else {"executed_keys": [], "approvals": []})

    # ---------------------------------------------------------- persistence
    def _save(self):
        if not self.dry_run:
            self.state_path.write_text(json.dumps(self.state, indent=2), encoding="utf-8")

    def _append(self, filename: str, record: dict):
        if self.dry_run:
            return
        with open(self.outbox / filename, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")

    def audit(self, **record):
        self._append("audit.jsonl", {"ts": datetime.now(timezone.utc).isoformat(), **record})

    # ---------------------------------------------------------- core
    def _perform(self, a: ProposedAction, key: str, ctx: dict[str, Any], approved_by: Optional[str] = None) -> str:
        record = {"ts": datetime.now(timezone.utc).isoformat(), "idempotency_key": key, "action": a.type.value,
                  "source": ctx.get("source_id"), "order_id": a.order_id, "team": a.team, "reason": a.reason,
                  "approved_by": approved_by}
        if a.type == A.reply_to_customer:
            record |= {"to": ctx.get("customer_email"), "body": ctx.get("reply")}
            msg = f"email sent to {ctx.get('customer_email')}"
        elif a.type == A.refund:
            record |= {"amount": a.amount}
            msg = f"refund ${a.amount:.2f} on {a.order_id}"
        elif a.type == A.reship:
            msg = f"replacement order created for {a.order_id}"
        elif a.type in (A.cancel_subscription, A.pause_subscription):
            record |= {"customer_id": ctx.get("customer_id")}
            msg = f"{a.type.value.replace('_', ' ')} for {ctx.get('customer_id')}"
        elif a.type == A.notify_slack:
            channel = f"#ops-{a.team or 'general'}"
            record |= {"channel": channel, "text": f"[{ctx.get('source_id')}] {ctx.get('summary', '')} -> {a.reason}"}
            msg = f"Slack {channel}"
        else:  # internal ticket / escalation
            record |= {"queue": a.team or "cx_lead", "priority": ctx.get("urgency", "medium"),
                       "title": ctx.get("summary", a.reason)}
            msg = f"{'escalated' if a.type == A.escalate_to_human else 'ticket'} -> {a.team or 'cx_lead'}"
        self._append(SINKS[a.type], record)
        self.state["executed_keys"].append(key)
        self._save()
        return ("[dry-run] " if self.dry_run else "") + msg

    def handle(self, d: Decision, ctx: dict[str, Any]) -> str:
        """Apply one guardrail decision. Returns a human-readable outcome."""
        key = action_key(d.action, ctx["source_id"], ctx.get("customer_id"))
        base = {"source": ctx["source_id"], "action": d.action.type.value, "key": key,
                "verdict": d.verdict.value, "rules": d.rules}
        if d.verdict == Verdict.blocked:
            self.audit(**base, outcome="blocked")
            return "BLOCKED"
        if key in self.state["executed_keys"]:
            self.audit(**base, outcome="deduplicated")
            return f"skipped: already executed ({key})"
        pending = next((p for p in self.state["approvals"] if p["key"] == key and p["status"] == "pending"), None)
        if pending:
            self.audit(**base, outcome=f"already pending {pending['id']}")
            return f"skipped: already pending {pending['id']}"
        if d.verdict == Verdict.needs_approval:
            apr_id = f"APR-{len(self.state['approvals']) + 1:03d}"
            self.state["approvals"].append({
                "id": apr_id, "status": "pending", "key": key, "created": datetime.now(timezone.utc).isoformat(),
                "action": d.action.model_dump(mode="json"), "rules": d.rules, "context": ctx,
            })
            self._save()
            self.audit(**base, outcome=f"queued {apr_id}")
            return f"queued for approval {apr_id}"
        outcome = self._perform(d.action, key, ctx)
        self.audit(**base, outcome=outcome)
        return outcome

    # ---------------------------------------------------------- human approval
    def pending(self) -> list[dict]:
        return [p for p in self.state["approvals"] if p["status"] == "pending"]

    def decide(self, apr_id: str, approve: bool, who: str, note: str = "") -> str:
        p = next((p for p in self.state["approvals"] if p["id"] == apr_id), None)
        if p is None:
            raise KeyError(f"No approval request {apr_id}")
        if p["status"] != "pending":
            raise ValueError(f"{apr_id} is already {p['status']}")
        p |= {"status": "approved" if approve else "rejected", "decided_by": who, "note": note,
              "decided_at": datetime.now(timezone.utc).isoformat()}
        if not approve:
            self._save()
            self.audit(source=p["context"]["source_id"], key=p["key"], outcome=f"rejected by {who}", note=note)
            return f"{apr_id} rejected"
        if p["key"] in self.state["executed_keys"]:
            self._save()
            return f"{apr_id} approved, but {p['key']} was already executed; nothing to do"
        outcome = self._perform(ProposedAction.model_validate(p["action"]), p["key"], p["context"], approved_by=who)
        self.audit(source=p["context"]["source_id"], key=p["key"], outcome=f"approved by {who}: {outcome}")
        return f"{apr_id} approved -> {outcome}"
