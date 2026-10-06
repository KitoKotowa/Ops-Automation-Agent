"""Data models: inputs (orders/customers/tickets), LLM triage output, and guardrail decisions."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------- inputs


class Subscription(BaseModel):
    plan: str
    status: Literal["active", "paused", "past_due", "cancelled"]
    failed_payment_attempts: int = 0


class Customer(BaseModel):
    customer_id: str
    name: str
    email: str
    tier: Literal["standard", "vip"] = "standard"
    lifetime_value: float = 0.0
    subscription: Optional[Subscription] = None


class Order(BaseModel):
    order_id: str
    customer_id: str
    items: list[str]
    total: float
    status: Literal["paid", "in_transit", "delivered", "payment_failed", "refunded", "cancelled"]
    created_at: datetime
    shipped_at: Optional[datetime] = None
    delivered_at: Optional[datetime] = None
    last_tracking_scan: Optional[datetime] = None


MAX_BODY_CHARS = 4000


class Ticket(BaseModel):
    """A support ticket after validation. Raw tickets that fail this are quarantined, never triaged."""

    ticket_id: str = Field(min_length=1)
    customer_email: str
    order_id: Optional[str] = None
    channel: Literal["email", "chat", "phone", "social"] = "email"
    created_at: datetime
    subject: str = ""
    body: str

    @field_validator("customer_email")
    @classmethod
    def _email(cls, v: str) -> str:
        v = v.strip().lower()
        if "@" not in v or "." not in v.split("@")[-1]:
            raise ValueError("not a valid email address")
        return v

    @field_validator("body")
    @classmethod
    def _body(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("ticket body is empty")
        if len(v) > MAX_BODY_CHARS:
            raise ValueError(f"ticket body exceeds {MAX_BODY_CHARS} chars; route to a human instead of truncating")
        return v


# ---------------------------------------------------------------- LLM triage output (structured output schema)


class ActionType(str, Enum):
    reply_to_customer = "reply_to_customer"
    refund = "refund"
    reship = "reship"
    cancel_subscription = "cancel_subscription"
    pause_subscription = "pause_subscription"
    create_internal_ticket = "create_internal_ticket"
    notify_slack = "notify_slack"
    escalate_to_human = "escalate_to_human"


class ProposedAction(BaseModel):
    type: ActionType
    reason: str = Field(description="Why this action, citing facts from the context.")
    amount: Optional[float] = Field(None, description="USD amount; required for refund, null otherwise.")
    order_id: Optional[str] = Field(None, description="Order this action applies to, if any.")
    team: Optional[Literal["fulfillment", "billing", "retention", "cx_lead", "quality"]] = Field(
        None, description="Target team for create_internal_ticket / notify_slack / escalate_to_human."
    )


class TriageResult(BaseModel):
    category: Literal[
        "shipping_delay", "damaged_item", "wrong_item", "billing_issue", "refund_request",
        "cancel_request", "subscription_change", "product_question", "other",
    ]
    urgency: Literal["low", "medium", "high", "critical"]
    sentiment: Literal["positive", "neutral", "negative", "angry"]
    summary: str = Field(description="One or two sentences for an ops teammate.")
    evidence: list[str] = Field(description="Facts from the provided context that support the decision.")
    missing_info: list[str] = Field(description="Information needed but not present. Empty if none.")
    proposed_actions: list[ProposedAction]
    customer_reply_draft: str = Field(description="Reply to the customer. Never promise anything not in proposed_actions.")
    confidence: float = Field(description="0.0-1.0 confidence in this triage.")


# ---------------------------------------------------------------- guardrail decisions


class Verdict(str, Enum):
    auto = "auto"                      # safe to execute now
    needs_approval = "needs_approval"  # queued for a human
    blocked = "blocked"                # never executed


class Decision(BaseModel):
    action: ProposedAction
    verdict: Verdict
    rules: list[str] = Field(default_factory=list, description="Which guardrail rules produced this verdict.")
