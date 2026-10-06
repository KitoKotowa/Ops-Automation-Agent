"""Deterministic detection. Everything here is plain code: facts the LLM receives as ground truth,
risk flags the guardrails enforce, and ops issues found by scanning orders (no LLM needed)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

from .data import Dataset
from .models import ActionType, Customer, Order, ProposedAction, Ticket

# Thresholds (business policy, not model judgement)
UNSHIPPED_WARN = timedelta(days=3)
UNSHIPPED_CRITICAL = timedelta(days=6)
TRACKING_STALL = timedelta(days=5)
DUPLICATE_WINDOW = timedelta(minutes=15)
PAYMENT_RETRY_LIMIT = 3

RISK_PATTERNS: dict[str, re.Pattern] = {
    "prompt_injection": re.compile(
        r"ignore (all |any )?(previous|prior|above) instructions|admin mode|system prompt|policy no longer applies"
        r"|you are now|disregard (the|your) (rules|policy)", re.I),
    "chargeback_threat": re.compile(r"charge ?back|dispute (the|this) charge|call my bank|filing .* with my bank", re.I),
    "legal_threat": re.compile(r"\b(lawyers?|attorneys?|sue|suing|lawsuit|small claims|legal action|BBB)\b", re.I),
    "health_safety": re.compile(r"allerg|anaphyla|sick|hospital|injur|burn(ed|t)|mold|contaminat", re.I),
}


@dataclass
class TicketContext:
    ticket: Ticket
    customer: Optional[Customer]
    order: Optional[Order]
    customer_orders: list[Order]
    facts: list[str] = field(default_factory=list)
    flags: set[str] = field(default_factory=set)

    @property
    def verified_order_id(self) -> Optional[str]:
        return self.order.order_id if self.order and "order_not_owned" not in self.flags else None


@dataclass
class OpsIssue:
    issue_id: str
    kind: str
    severity: str  # low | medium | high | critical
    summary: str
    order_id: Optional[str]
    customer_id: Optional[str]
    actions: list[ProposedAction]


def _age(now: datetime, ts: datetime) -> str:
    d = now - ts
    return f"{d.days}d {d.seconds // 3600}h"


def find_duplicate_charges(orders: list[Order]) -> list[tuple[Order, Order]]:
    pairs = []
    by_cust: dict[str, list[Order]] = {}
    for o in orders:
        if o.status in ("paid", "in_transit", "delivered"):
            by_cust.setdefault(o.customer_id, []).append(o)
    for lst in by_cust.values():
        lst.sort(key=lambda o: o.created_at)
        for a, b in zip(lst, lst[1:]):
            if a.total == b.total and a.items == b.items and b.created_at - a.created_at <= DUPLICATE_WINDOW:
                pairs.append((a, b))
    return pairs


def build_ticket_context(t: Ticket, ds: Dataset, now: datetime) -> TicketContext:
    customer = ds.customers_by_email.get(t.customer_email)
    cust_orders = sorted(
        (o for o in ds.orders.values() if customer and o.customer_id == customer.customer_id),
        key=lambda o: o.created_at, reverse=True,
    )
    order = ds.orders.get(t.order_id) if t.order_id else None
    ctx = TicketContext(t, customer, order, cust_orders)
    text = f"{t.subject}\n{t.body}"

    for flag, pat in RISK_PATTERNS.items():
        if pat.search(text):
            ctx.flags.add(flag)

    if customer is None:
        ctx.flags.add("unknown_customer")
        ctx.facts.append(f"No customer record matches {t.customer_email}.")
    else:
        sub = customer.subscription
        ctx.facts.append(
            f"Customer {customer.customer_id} ({customer.name}), tier={customer.tier}, LTV=${customer.lifetime_value:.2f}, "
            f"subscription={sub.plan + ' / ' + sub.status if sub else 'none'}."
        )
        if sub and sub.status == "past_due":
            ctx.flags.add("past_due")
            ctx.facts.append(f"Subscription is past_due after {sub.failed_payment_attempts} failed payment attempts.")

    if t.order_id and order is None:
        ctx.flags.add("order_not_found")
        ctx.facts.append(f"Order {t.order_id} referenced in ticket does NOT exist in the order system.")
    elif order is not None:
        if customer is None or order.customer_id != customer.customer_id:
            ctx.flags.add("order_not_owned")
            ctx.facts.append(f"Order {order.order_id} does NOT belong to the ticket's customer.")
        else:
            ctx.facts.append(
                f"Order {order.order_id}: items={order.items}, total=${order.total:.2f}, status={order.status}, "
                f"placed {_age(now, order.created_at)} ago."
            )
            if order.status == "paid" and order.shipped_at is None and now - order.created_at > UNSHIPPED_WARN:
                ctx.flags.add("order_stuck")
                ctx.facts.append(f"Order {order.order_id} is paid but unshipped for {_age(now, order.created_at)} (SLA 3d).")
            if order.status == "in_transit" and order.last_tracking_scan and now - order.last_tracking_scan > TRACKING_STALL:
                ctx.flags.add("tracking_stalled")
                ctx.facts.append(f"Tracking for {order.order_id} has not updated in {_age(now, order.last_tracking_scan)}.")
            if order.delivered_at:
                ctx.facts.append(f"Order {order.order_id} was delivered {_age(now, order.delivered_at)} ago.")
            for a, b in find_duplicate_charges(cust_orders):
                if order.order_id in (a.order_id, b.order_id):
                    ctx.flags.add("duplicate_charge_verified")
                    ctx.facts.append(
                        f"Verified duplicate charge: {a.order_id} and {b.order_id}, both ${a.total:.2f}, "
                        f"{int((b.created_at - a.created_at).total_seconds() // 60)} min apart. Duplicate is {b.order_id}."
                    )
    if customer and not t.order_id and cust_orders:
        recent = ", ".join(f"{o.order_id} ({o.status}, ${o.total:.2f})" for o in cust_orders[:3])
        ctx.facts.append(f"No order referenced in ticket. Customer's recent orders: {recent}.")
    return ctx


def scan_orders(ds: Dataset, now: datetime) -> list[OpsIssue]:
    """Proactive checks over all orders/subscriptions — surfaces issues nobody has filed a ticket about yet."""
    issues: list[OpsIssue] = []

    def add(kind, severity, summary, order=None, customer_id=None, actions=()):
        issues.append(OpsIssue(f"ISS-{len(issues) + 1:03d}", kind, severity, summary,
                               order.order_id if order else None,
                               customer_id or (order.customer_id if order else None), list(actions)))

    for o in sorted(ds.orders.values(), key=lambda o: o.order_id):
        if o.status == "paid" and o.shipped_at is None and now - o.created_at > UNSHIPPED_WARN:
            crit = now - o.created_at > UNSHIPPED_CRITICAL
            act = [ProposedAction(type=ActionType.create_internal_ticket, team="fulfillment", order_id=o.order_id,
                                  reason=f"Paid but unshipped for {_age(now, o.created_at)}.")]
            if crit:
                act.append(ProposedAction(type=ActionType.notify_slack, team="fulfillment", order_id=o.order_id,
                                          reason="Unshipped beyond critical threshold (6d)."))
            add("order_unshipped", "high" if crit else "medium",
                f"{o.order_id} paid {_age(now, o.created_at)} ago and not shipped.", o, actions=act)
        if o.status == "in_transit" and o.last_tracking_scan and now - o.last_tracking_scan > TRACKING_STALL:
            add("tracking_stalled", "medium", f"{o.order_id} tracking silent for {_age(now, o.last_tracking_scan)}.", o,
                actions=[ProposedAction(type=ActionType.create_internal_ticket, team="fulfillment", order_id=o.order_id,
                                        reason="Open carrier trace for stalled shipment.")])

    for a, b in find_duplicate_charges(list(ds.orders.values())):
        add("duplicate_charge", "high",
            f"{a.order_id} and {b.order_id} look like a duplicate charge (${a.total:.2f}, same items, "
            f"{int((b.created_at - a.created_at).total_seconds() // 60)} min apart).", b,
            actions=[
                ProposedAction(type=ActionType.create_internal_ticket, team="billing", order_id=b.order_id,
                               reason="Verify duplicate and hold fulfillment of the duplicate order."),
                ProposedAction(type=ActionType.refund, amount=b.total, order_id=b.order_id,
                               reason="Refund the duplicate charge."),
            ])

    for c in ds.customers.values():
        s = c.subscription
        if s and s.status == "past_due" and s.failed_payment_attempts >= PAYMENT_RETRY_LIMIT:
            add("payment_retries_exhausted", "medium",
                f"{c.customer_id} ({c.name}) subscription past_due after {s.failed_payment_attempts} failed attempts.",
                customer_id=c.customer_id,
                actions=[ProposedAction(type=ActionType.notify_slack, team="retention",
                                        reason="Payment retries exhausted; involuntary churn risk.")])
    return issues
