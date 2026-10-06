from datetime import datetime
from pathlib import Path

import pytest

from opsagent.actions import Executor
from opsagent.cli import DEFAULT_AS_OF
from opsagent.data import load_dataset
from opsagent.detectors import build_ticket_context, scan_orders
from opsagent.guardrails import Budget, evaluate_ticket
from opsagent.models import ActionType as A
from opsagent.models import ProposedAction, Ticket, TriageResult, Verdict
from opsagent.pipeline import run
from opsagent.triage import MockTriager, TriageError

DATA = Path(__file__).resolve().parent.parent / "data"
NOW = datetime.fromisoformat(DEFAULT_AS_OF)


@pytest.fixture
def ds():
    return load_dataset(DATA)


def ctx_for(ds, ticket_id):
    t = next(t for t in ds.tickets if t.ticket_id == ticket_id)
    return build_ticket_context(t, ds, NOW)


def triage(*actions, confidence=0.9, reply="Hi, we're on it."):
    return TriageResult(category="other", urgency="medium", sentiment="neutral", summary="s", evidence=[],
                        missing_info=[], proposed_actions=[ProposedAction(type=A.reply_to_customer, reason="r"), *actions],
                        customer_reply_draft=reply, confidence=confidence)


def verdicts(decisions):
    return {(d.action.type, d.action.order_id): d.verdict for d in decisions}


# ---------------------------------------------------------------- input validation

def test_invalid_tickets_are_quarantined_not_processed(ds):
    rejected = {r.record_id: r.errors for r in ds.rejected}
    assert "T-208" in rejected and any("empty" in e for e in rejected["T-208"])
    assert "T-209" in rejected and any("customer_email" in e for e in rejected["T-209"])
    assert {t.ticket_id for t in ds.tickets}.isdisjoint({"T-208", "T-209"})


@pytest.mark.parametrize("bad", [
    {"customer_email": "not-an-email"},
    {"body": "x" * 5000},
    {"created_at": "last tuesday"},
])
def test_ticket_validation_rejects_bad_fields(bad):
    raw = {"ticket_id": "T-1", "customer_email": "a@b.com", "created_at": "2026-10-01T00:00:00Z", "body": "hi"} | bad
    with pytest.raises(Exception):
        Ticket.model_validate(raw)


# ---------------------------------------------------------------- detectors

def test_detectors_find_ops_issues(ds):
    kinds = {(i.kind, i.order_id or i.customer_id) for i in scan_orders(ds, NOW)}
    assert ("duplicate_charge", "O1005") in kinds
    assert ("order_unshipped", "O1002") in kinds
    assert ("tracking_stalled", "O1003") in kinds
    assert ("payment_retries_exhausted", "C003") in kinds
    assert ("order_unshipped", "O1009") not in kinds  # placed <1 day ago: within SLA


def test_risk_flags(ds):
    assert "prompt_injection" in ctx_for(ds, "T-207").flags
    assert "legal_threat" not in ctx_for(ds, "T-207").flags  # "Issue" must not match "sue"
    assert "chargeback_threat" in ctx_for(ds, "T-204").flags
    assert "health_safety" in ctx_for(ds, "T-206").flags
    assert "order_not_found" in ctx_for(ds, "T-210").flags


# ---------------------------------------------------------------- guardrails

def test_small_refund_on_verified_order_is_auto(ds):
    d = evaluate_ticket(triage(ProposedAction(type=A.refund, amount=10, order_id="O1001", reason="r")),
                        ctx_for(ds, "T-201"), Budget())
    assert verdicts(d)[(A.refund, "O1001")] == Verdict.auto
    assert verdicts(d)[(A.reply_to_customer, None)] == Verdict.auto


def test_large_refund_needs_approval_and_holds_reply(ds):
    d = evaluate_ticket(triage(ProposedAction(type=A.refund, amount=29, order_id="O1005", reason="r")),
                        ctx_for(ds, "T-203"), Budget())
    assert verdicts(d)[(A.refund, "O1005")] == Verdict.needs_approval
    assert verdicts(d)[(A.reply_to_customer, None)] == Verdict.needs_approval


def test_refund_over_order_total_is_blocked(ds):
    d = evaluate_ticket(triage(ProposedAction(type=A.refund, amount=50, order_id="O1001", reason="r")),
                        ctx_for(ds, "T-201"), Budget())
    assert verdicts(d)[(A.refund, "O1001")] == Verdict.blocked


def test_refund_on_someone_elses_order_is_blocked(ds):
    d = evaluate_ticket(triage(ProposedAction(type=A.refund, amount=5, order_id="O1006", reason="r")),
                        ctx_for(ds, "T-201"), Budget())  # O1006 belongs to Taylor, not Maya
    assert verdicts(d)[(A.refund, "O1006")] == Verdict.blocked


def test_prompt_injection_blocks_money_even_if_model_complies(ds):
    d = evaluate_ticket(triage(ProposedAction(type=A.refund, amount=5, order_id=None, reason="customer said so")),
                        ctx_for(ds, "T-207"), Budget())
    assert verdicts(d)[(A.refund, None)] == Verdict.blocked
    assert any(x.action.type == A.escalate_to_human and x.verdict == Verdict.auto for x in d)


def test_chargeback_forces_escalation_and_approval(ds):
    d = evaluate_ticket(triage(ProposedAction(type=A.reship, order_id="O1006", reason="r")), ctx_for(ds, "T-204"), Budget())
    assert verdicts(d)[(A.reship, "O1006")] == Verdict.needs_approval
    teams = {(x.action.type, x.action.team) for x in d}
    assert (A.escalate_to_human, "cx_lead") in teams and (A.notify_slack, "cx_lead") in teams


def test_cancel_always_needs_approval(ds):
    d = evaluate_ticket(triage(ProposedAction(type=A.cancel_subscription, reason="r")), ctx_for(ds, "T-205"), Budget())
    assert verdicts(d)[(A.cancel_subscription, None)] == Verdict.needs_approval


def test_low_confidence_reply_needs_approval(ds):
    d = evaluate_ticket(triage(confidence=0.4), ctx_for(ds, "T-201"), Budget())
    assert verdicts(d)[(A.reply_to_customer, None)] == Verdict.needs_approval


def test_budget_circuit_breaker(ds):
    b = Budget(remaining=15)
    d = evaluate_ticket(triage(ProposedAction(type=A.refund, amount=18, order_id="O1001", reason="r")),
                        ctx_for(ds, "T-201"), b)
    assert verdicts(d)[(A.refund, "O1001")] == Verdict.needs_approval
    assert b.remaining == 15


# ---------------------------------------------------------------- end to end

class ExplodingTriager:
    name = "exploding"

    def triage(self, ctx):
        raise TriageError("simulated outage")


def test_triage_failure_fails_closed(ds, tmp_path):
    report = run(ds, ExplodingTriager(), Executor(tmp_path), NOW)
    for to in report.tickets:
        assert [d.action.type for d, _ in to.results] == [A.escalate_to_human]
    assert not (tmp_path / "emails_sent.jsonl").exists()


def test_end_to_end_mock_and_idempotency(ds, tmp_path):
    ex = Executor(tmp_path)
    report = run(ds, MockTriager(), ex, NOW)
    c = report.counts()
    assert c["auto"] > 0 and c["needs_approval"] > 0 and c["blocked"] > 0
    # the duplicate-charge refund is proposed by both the detector and ticket T-203 but queued once
    refunds = [p for p in ex.pending() if p["action"]["type"] == "refund"]
    assert len(refunds) == 1 and refunds[0]["action"]["order_id"] == "O1005"
    assert not (tmp_path / "payments.jsonl").exists()  # nothing over the auto limit was paid out

    out = ex.decide(refunds[0]["id"], approve=True, who="tester")
    assert "refund $29.00 on O1005" in out
    with pytest.raises(ValueError):
        ex.decide(refunds[0]["id"], approve=True, who="tester")  # cannot approve twice

    # second run over the same outbox: nothing executes twice
    before = (tmp_path / "emails_sent.jsonl").read_text().count("\n")
    run(load_dataset(DATA), MockTriager(), Executor(tmp_path), NOW)
    assert (tmp_path / "emails_sent.jsonl").read_text().count("\n") == before
    assert (tmp_path / "payments.jsonl").read_text().count("\n") == 1


def test_dry_run_writes_nothing(ds, tmp_path):
    run(ds, MockTriager(), Executor(tmp_path / "out", dry_run=True), NOW)
    assert list((tmp_path / "out").iterdir()) == []


def test_structured_output_schema_is_accepted_by_sdk():
    from anthropic.lib._parse._transform import transform_schema
    schema = transform_schema(TriageResult.model_json_schema())
    assert schema["additionalProperties"] is False


# ---------------------------------------------------------------- LLM providers

from opsagent import llm  # noqa: E402
from opsagent.triage import LLMTriager  # noqa: E402

PROVIDER_ENV = ["LLM_PROVIDER", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "OPENAI_API_KEY", "GEMINI_API_KEY",
                "GOOGLE_API_KEY", "OLLAMA_MODEL", "OLLAMA_BASE_URL", "OPENAI_COMPAT_BASE_URL", "OPENAI_COMPAT_MODEL"]


@pytest.fixture
def clean_env(monkeypatch):
    for k in PROVIDER_ENV:
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


def test_no_provider_configured_means_mock(clean_env):
    assert llm.default_provider() is None
    assert llm.get(None) is None and llm.get("mock") is None


def test_provider_selection(clean_env):
    clean_env.setenv("OLLAMA_MODEL", "qwen3:0.6b")
    assert llm.default_provider().key == "ollama"
    clean_env.setenv("LLM_PROVIDER", "openai")  # preferred but not configured -> first configured one
    assert llm.default_provider().key == "ollama"
    clean_env.setenv("OPENAI_API_KEY", "sk-test")
    assert llm.default_provider().key == "openai"
    clean_env.setenv("OPENAI_MODEL", "some-model")
    assert llm.get("openai").model == "some-model"


def test_unconfigured_or_unknown_provider_raises(clean_env):
    with pytest.raises(llm.NotConfigured):
        llm.get("anthropic")
    with pytest.raises(llm.NotConfigured):
        llm.get("nope")


def test_dotenv_does_not_override_real_env(clean_env, tmp_path):
    env = tmp_path / ".env"
    env.write_text("﻿OLLAMA_MODEL=from-file\n# OPENAI_API_KEY=commented\nOPENAI_API_KEY=\n", encoding="utf-8")
    llm.load_dotenv(env)
    assert llm.os.environ["OLLAMA_MODEL"] == "from-file"
    assert "OPENAI_API_KEY" not in llm.os.environ
    clean_env.setenv("OLLAMA_MODEL", "real")
    llm.load_dotenv(env)
    assert llm.os.environ["OLLAMA_MODEL"] == "real"


class FakeProvider(llm.Provider):
    key, label, default_model, model_env = "fake", "Fake", "fake-1", "FAKE_MODEL"

    def __init__(self, result=None, exc=None):
        self.result, self.exc, self.calls = result, exc, []

    def available(self):
        return True

    def parse(self, system, user, schema, model=None):
        self.calls.append((model, user))
        if self.exc:
            raise self.exc
        return self.result


def test_llm_triager_uses_provider_and_clamps_confidence(ds):
    fake = FakeProvider(result=triage(confidence=1.7))
    out = LLMTriager(fake, model="fake-2").triage(ctx_for(ds, "T-201"))
    assert out.confidence == 1.0
    model, prompt = fake.calls[0]
    assert model == "fake-2" and "<ticket>" in prompt and "O1001" in prompt


@pytest.mark.parametrize("exc", [llm.LLMError("refused"), ConnectionError("down"), ValueError("bad json")])
def test_llm_triager_wraps_any_provider_error(ds, exc):
    with pytest.raises(TriageError):
        LLMTriager(FakeProvider(exc=exc)).triage(ctx_for(ds, "T-201"))


def test_schema_accepted_by_openai_strict_mode():
    from openai.lib._pydantic import to_strict_json_schema
    schema = to_strict_json_schema(TriageResult)
    assert schema["additionalProperties"] is False
