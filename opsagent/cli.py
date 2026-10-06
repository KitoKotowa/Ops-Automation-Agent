"""Command line interface.

  python -m opsagent run [--llm auto|claude|mock] [--dry-run] [--keep-state]
  python -m opsagent approvals
  python -m opsagent approve APR-001 [--by name] [--note "..."]
  python -m opsagent reject  APR-002 [--by name] [--note "..."]
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .actions import Executor
from .data import load_dataset
from .models import Verdict
from .pipeline import RunReport, run
from . import llm
from .triage import LLMTriager, MockTriager

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_AS_OF = "2026-10-06T09:00:00+00:00"  # fixed clock so the sample data tells the same story every run
VERDICT_STYLE = {Verdict.auto: "green", Verdict.needs_approval: "yellow", Verdict.blocked: "red"}

console = Console()


def _decisions_table(results) -> Table:
    t = Table(show_header=True, header_style="bold", expand=True, box=None, padding=(0, 1))
    t.add_column("action", style="cyan", no_wrap=True)
    t.add_column("verdict", no_wrap=True)
    t.add_column("why", ratio=3)
    t.add_column("outcome", ratio=2, overflow="fold")
    for d, outcome in results:
        a = d.action
        label = a.type.value + (f" ${a.amount:.2f}" if a.amount else "") + (f" {a.order_id}" if a.order_id else "") \
            + (f" @{a.team}" if a.team else "")
        style = VERDICT_STYLE[d.verdict]
        t.add_row(label, f"[{style}]{d.verdict.value}[/]", "; ".join(d.rules), outcome)
    return t


def render(report: RunReport):
    console.rule(f"[bold]Ops Automation Agent[/]  triage={report.triager}  as_of={report.now:%Y-%m-%d %H:%M}Z")

    if report.rejected:
        tbl = Table(title="Quarantined input (not processed by the agent)", expand=True, title_justify="left")
        tbl.add_column("record"); tbl.add_column("problems", ratio=3); tbl.add_column("handling", ratio=2)
        for r, out in report.rejected:
            tbl.add_row(f"{r.source}:{r.record_id}", "\n".join(r.errors), out)
        console.print(tbl)

    console.print("\n[bold]Proactive checks (orders & subscriptions, rule-based)[/]")
    for io in report.issues:
        i = io.issue
        console.print(Panel(_decisions_table(io.results), title=f"{i.issue_id} [{i.severity}] {i.kind}",
                            subtitle=i.summary, title_align="left", subtitle_align="left"))

    console.print("\n[bold]Support tickets (LLM triage + guardrails)[/]")
    for to in report.tickets:
        t = to.ctx.ticket
        lines = []
        if to.triage:
            tr = to.triage
            lines.append(f"[bold]{tr.category}[/] | urgency={tr.urgency} | sentiment={tr.sentiment} | confidence={tr.confidence:.2f}")
            lines.append(f"[italic]{tr.summary}[/]")
            if tr.missing_info:
                lines.append(f"[magenta]missing info:[/] {'; '.join(tr.missing_info)}")
        else:
            lines.append(f"[red]triage failed:[/] {to.error}")
        if to.ctx.flags:
            lines.append(f"[red]flags:[/] {', '.join(sorted(to.ctx.flags))}")
        body = Table.grid(expand=True)
        body.add_row("\n".join(lines))
        body.add_row(_decisions_table(to.results))
        if to.triage:
            body.add_row(f"[dim]reply draft:[/] [dim]{to.triage.customer_reply_draft.strip()}[/]")
        console.print(Panel(body, title=f"{t.ticket_id}  {t.customer_email}  order={t.order_id or '-'}",
                            subtitle=t.subject, title_align="left", subtitle_align="left"))

    c = report.counts()
    console.print(
        f"\n[bold]Summary:[/] [green]{c['auto']} auto-executed[/], [yellow]{c['needs_approval']} awaiting approval[/], "
        f"[red]{c['blocked']} blocked[/], {len(report.rejected)} input records quarantined. "
        f"Auto-spend budget left: ${report.budget_left:.2f}"
    )
    if c["needs_approval"]:
        console.print("Next: [bold]python -m opsagent approvals[/] then [bold]approve/reject <id>[/]")


def write_markdown(report: RunReport, path: Path):
    lines = [f"# Ops agent run - {report.now:%Y-%m-%d %H:%M}Z", "", f"Triage: {report.triager}", ""]
    c = report.counts()
    lines += [f"**{c['auto']} auto / {c['needs_approval']} need approval / {c['blocked']} blocked / "
              f"{len(report.rejected)} quarantined**", ""]
    for r, out in report.rejected:
        lines.append(f"- QUARANTINED `{r.source}:{r.record_id}`: {'; '.join(r.errors)} -> {out}")
    lines.append("\n## Proactive issues")
    for io in report.issues:
        lines.append(f"\n### {io.issue.issue_id} [{io.issue.severity}] {io.issue.kind}\n{io.issue.summary}\n")
        lines += [f"- `{d.action.type.value}` **{d.verdict.value}** ({'; '.join(d.rules)}) -> {o}" for d, o in io.results]
    lines.append("\n## Tickets")
    for to in report.tickets:
        t = to.ctx.ticket
        lines.append(f"\n### {t.ticket_id} - {t.subject}")
        if to.triage:
            tr = to.triage
            lines.append(f"{tr.category} / {tr.urgency} / confidence {tr.confidence:.2f}\n\n{tr.summary}\n")
            if tr.missing_info:
                lines.append(f"Missing: {'; '.join(tr.missing_info)}\n")
        else:
            lines.append(f"Triage failed: {to.error}\n")
        lines += [f"- `{d.action.type.value}` **{d.verdict.value}** ({'; '.join(d.rules)}) -> {o}" for d, o in to.results]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_triager(key: str, model: str | None = None):
    """key: 'auto' (LLM_PROVIDER or first configured provider, else mock), 'mock', or a provider key."""
    provider = llm.get(None if key == "auto" else key)
    if provider is None:
        if key == "auto":
            console.print("[yellow]No LLM provider configured in .env: using the offline mock triager.[/]")
        return MockTriager()
    return LLMTriager(provider, model)


def cmd_providers(args) -> int:
    default = llm.default_provider()
    tbl = Table(title="LLM providers (configure in .env)", title_justify="left")
    for col in ("key", "provider", "model", "status"):
        tbl.add_column(col)
    for p in llm.PROVIDERS.values():
        i = p.info()
        status = ("[green]ready[/]" if i["available"] else "[dim]not configured[/]") + (" (default)" if p is default else "")
        tbl.add_row(i["key"], i["label"], i["model"] or "-", status)
    tbl.add_row("mock", "Offline rules", "-", "[green]ready[/]" + ("" if default else " (default)"))
    console.print(tbl)
    return 0


def cmd_run(args) -> int:
    outbox = Path(args.outbox)
    if not args.keep_state and outbox.exists() and not args.dry_run:
        shutil.rmtree(outbox)  # demo convenience: each run starts from a clean outbox
    try:
        triager = make_triager(args.llm, args.model)
    except llm.NotConfigured as e:
        console.print(f"[red]{e}[/] Run [bold]python -m opsagent providers[/] to see what is configured.")
        return 2
    ds = load_dataset(Path(args.data))
    executor = Executor(outbox, dry_run=args.dry_run)
    with console.status(f"Triaging {len(ds.tickets)} tickets with {triager.name}..."):
        report = run(ds, triager, executor, datetime.fromisoformat(args.as_of), workers=args.workers)
    render(report)
    if not args.dry_run:
        write_markdown(report, outbox / "run_report.md")
        console.print(f"[dim]Outbox, audit log and report written to {outbox}/[/]")
    return 0


def cmd_approvals(args) -> int:
    ex = Executor(Path(args.outbox))
    pending = ex.pending()
    if not pending:
        console.print("No pending approvals.")
        return 0
    tbl = Table(title="Pending approvals", expand=True, title_justify="left")
    for col in ("id", "source", "action", "why it needs a human", "proposed because"):
        tbl.add_column(col)
    for p in pending:
        a = p["action"]
        label = a["type"] + (f" ${a['amount']:.2f}" if a.get("amount") else "") + (f" {a['order_id']}" if a.get("order_id") else "")
        tbl.add_row(p["id"], p["context"]["source_id"], label, "; ".join(p["rules"]), a["reason"])
    console.print(tbl)
    return 0


def cmd_decide(args, approve: bool) -> int:
    ex = Executor(Path(args.outbox))
    try:
        console.print(ex.decide(args.id, approve, args.by, args.note))
    except (KeyError, ValueError) as e:
        console.print(f"[red]{e}[/]")
        return 1
    return 0


def main(argv=None) -> int:
    llm.load_dotenv()
    p = argparse.ArgumentParser(prog="opsagent", description="Support/ops automation agent demo")
    p.add_argument("--outbox", default=str(ROOT / "outbox"))
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="process tickets + scan orders")
    r.add_argument("--data", default=str(ROOT / "data"))
    r.add_argument("--llm", choices=["auto", "mock", *llm.PROVIDERS], default="auto",
                   help="auto = LLM_PROVIDER from .env, else first configured provider, else mock")
    r.add_argument("--model", help="override the provider's model for this run")
    r.add_argument("--as-of", default=DEFAULT_AS_OF, help="ISO timestamp used as 'now'")
    r.add_argument("--dry-run", action="store_true", help="decide everything, write nothing")
    r.add_argument("--keep-state", action="store_true", help="don't clear outbox (tests idempotency across runs)")
    r.add_argument("--workers", type=int, default=4)
    r.set_defaults(func=cmd_run)

    sub.add_parser("providers", help="show which LLM providers are configured").set_defaults(func=cmd_providers)
    sub.add_parser("approvals", help="list pending approvals").set_defaults(func=cmd_approvals)
    for name, flag in (("approve", True), ("reject", False)):
        s = sub.add_parser(name, help=f"{name} a pending action")
        s.add_argument("id")
        s.add_argument("--by", default=os.environ.get("USERNAME") or os.environ.get("USER") or "operator")
        s.add_argument("--note", default="")
        s.set_defaults(func=lambda a, f=flag: cmd_decide(a, f))

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
