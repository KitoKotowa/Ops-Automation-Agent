"""Load and validate input data. Bad records are quarantined with a reason, never silently dropped."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .models import Customer, Order, Ticket


@dataclass
class Rejected:
    source: str
    record_id: str
    errors: list[str]
    raw: Any


@dataclass
class Dataset:
    customers: dict[str, Customer] = field(default_factory=dict)       # by customer_id
    customers_by_email: dict[str, Customer] = field(default_factory=dict)
    orders: dict[str, Order] = field(default_factory=dict)             # by order_id
    tickets: list[Ticket] = field(default_factory=list)
    rejected: list[Rejected] = field(default_factory=list)


def _load_json_list(path: Path) -> list:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SystemExit(f"Input file not found: {path}")
    except json.JSONDecodeError as e:
        raise SystemExit(f"{path} is not valid JSON: {e}")
    if not isinstance(data, list):
        raise SystemExit(f"{path} must contain a JSON list")
    return data


def _fmt(e: ValidationError) -> list[str]:
    return [f"{'.'.join(str(p) for p in err['loc']) or '<record>'}: {err['msg']}" for err in e.errors()]


def _validate(raw_list: list, model, source: str, id_field: str, ds: Dataset) -> list:
    out = []
    for raw in raw_list:
        rid = str(raw.get(id_field, "<missing id>")) if isinstance(raw, dict) else "<not an object>"
        try:
            out.append(model.model_validate(raw))
        except ValidationError as e:
            ds.rejected.append(Rejected(source, rid, _fmt(e), raw))
    return out


def load_dataset(data_dir: Path, extra_tickets: list | None = None) -> Dataset:
    """extra_tickets: raw ticket dicts (e.g. typed into the UI) validated exactly like the file's tickets."""
    ds = Dataset()
    for c in _validate(_load_json_list(data_dir / "customers.json"), Customer, "customers", "customer_id", ds):
        ds.customers[c.customer_id] = c
        ds.customers_by_email[c.email.lower()] = c
    for o in _validate(_load_json_list(data_dir / "orders.json"), Order, "orders", "order_id", ds):
        ds.orders[o.order_id] = o

    seen: set[str] = set()
    raw_tickets = _load_json_list(data_dir / "tickets.json") + list(extra_tickets or [])
    for t in _validate(raw_tickets, Ticket, "tickets", "ticket_id", ds):
        if t.ticket_id in seen:
            ds.rejected.append(Rejected("tickets", t.ticket_id, ["duplicate ticket_id"], t.model_dump(mode="json")))
            continue
        seen.add(t.ticket_id)
        ds.tickets.append(t)
    return ds
