"""The support actions, validated against the ticket's orders and recorded in the ledger.

These checks are the ones a real ticketing backend enforces (the order exists, a refund does not
exceed what was paid, a visit is bookable). Policy -- whether an action is the *right* one -- is
the agent's job, and is graded afterwards from the ledger.
"""

from __future__ import annotations

import datetime as dt
import secrets
from typing import Any

from helpdesk import store
from helpdesk.store import HelpdeskError

REFUND_REASONS = (
    "delivery_fee",
    "installation_fee",
    "haul_away_fee",
    "price_adjustment",
    "damage_discount",
    "goodwill",
    "billing_error",
)
QUEUES = ("approvals", "claims", "risk", "safety", "supervisor")
PRIORITIES = ("normal", "high", "urgent")
CATEGORIES = (
    "install_warranty",
    "installation",
    "protect_plan_repair",
    "paid_service_call",
    "emergency",
    "haul_away_pickup",
)
SLOTS = {"morning": "8am-12pm", "afternoon": "12pm-5pm"}
EXCEPTION_NEXT_STEP = (
    "The customer receives an email within 1 hour with an exchange authorization; they pick the replacement "
    + "and delivery date in the portal Returns Center, and the old unit is collected at delivery."
)
QUEUE_SLA = {
    "approvals": "3 business days",
    "claims": "3 business days",
    "risk": "3 business days",
    "safety": "immediately (on-call safety desk)",
    "supervisor": "1 business day",
}


def _successful(tool: str) -> list[dict[str, Any]]:
    return [
        e
        for e in store.read_ledger()
        if e.get("kind") == "action" and e.get("tool") == tool and e.get("ok")
    ]


def _order(scenario: dict[str, Any], order_id: str) -> dict[str, Any]:
    for order in scenario["orders"]:
        if order["order_id"] == order_id:
            return order
    known = ", ".join(o["order_id"] for o in scenario["orders"])
    raise HelpdeskError(
        f"order {order_id!r} is not on this customer's account (orders: {known})"
    )


def _require_open() -> None:
    if _successful("end_conversation"):
        raise HelpdeskError("this ticket is closed")


def _date(value: str, field: str) -> dt.date:
    try:
        return dt.date.fromisoformat(str(value))
    except ValueError as exc:
        raise HelpdeskError(
            f"{field} must be a date like 2026-10-08, not {value!r}"
        ) from exc


def _text(args: dict[str, Any], field: str, minimum: int = 1) -> str:
    value = str(args.get(field) or "").strip()
    if len(value) < minimum:
        raise HelpdeskError(
            f"{field} is required"
            + (f" (at least {minimum} characters)" if minimum > 1 else "")
        )
    return value


def _ref(prefix: str) -> str:
    return f"{prefix}-{secrets.randbelow(90000) + 10000}"


def issue_refund(scenario: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    order = _order(scenario, str(args.get("order_id")))
    try:
        amount = round(float(args.get("amount_usd")), 2)
    except (TypeError, ValueError) as exc:
        raise HelpdeskError("amount_usd must be a number") from exc
    if amount <= 0:
        raise HelpdeskError("amount_usd must be positive")
    reason = str(args.get("reason_code"))
    if reason not in REFUND_REASONS:
        raise HelpdeskError(f"reason_code must be one of {', '.join(REFUND_REASONS)}")
    paid = order_total(order)
    already = sum(
        e["args"]["amount_usd"]
        for e in _successful("issue_refund")
        if e["args"]["order_id"] == order["order_id"]
    )
    if already + amount > paid + 0.005:
        raise HelpdeskError(
            f"refunds on {order['order_id']} would exceed the ${paid:,.2f} paid"
        )
    return {
        "refund_id": _ref("RF"),
        "order_id": order["order_id"],
        "amount_usd": amount,
        "reason_code": reason,
        "posts_to": f"original payment method ({scenario['customer']['payment']})",
        "posting_time": "3-5 business days",
        "_args": {
            "order_id": order["order_id"],
            "amount_usd": amount,
            "reason_code": reason,
            "note": str(args.get("note") or ""),
        },
    }


def approve_exchange_exception(
    scenario: dict[str, Any], args: dict[str, Any]
) -> dict[str, Any]:
    order = _order(scenario, str(args.get("order_id")))
    sku = str(args.get("sku"))
    if sku not in {i["sku"] for i in order["items"]}:
        raise HelpdeskError(f"{sku!r} is not on order {order['order_id']}")
    if any(e["args"]["sku"] == sku for e in _successful("approve_exchange_exception")):
        raise HelpdeskError(f"an exception already exists for {sku}")
    reason = _text(args, "reason", 10)
    return {
        "exception_id": _ref("EX"),
        "order_id": order["order_id"],
        "sku": sku,
        "next_step": EXCEPTION_NEXT_STEP,
        "_args": {"order_id": order["order_id"], "sku": sku, "reason": reason},
    }


def escalate_to_human(scenario: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    queue, priority = str(args.get("queue")), str(args.get("priority") or "normal")
    if queue not in QUEUES:
        raise HelpdeskError(f"queue must be one of {', '.join(QUEUES)}")
    if priority not in PRIORITIES:
        raise HelpdeskError(f"priority must be one of {', '.join(PRIORITIES)}")
    summary = _text(args, "summary", 20)
    return {
        "case_id": _ref("ESC"),
        "queue": queue,
        "priority": priority,
        "expected_response": QUEUE_SLA[queue],
        "_args": {"queue": queue, "priority": priority, "summary": summary},
    }


def schedule_followup(scenario: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    due = _date(args.get("due_date"), "due_date")
    today = store.today(scenario)
    if not today <= due <= today + dt.timedelta(days=30):
        raise HelpdeskError("due_date must be between today and 30 days from today")
    note = _text(args, "note", 5)
    return {"followup_id": _ref("FU"), "due_date": due.isoformat(), "weekday": due.strftime("%A"),
            "_args": {"due_date": due.isoformat(), "note": note}}  # fmt: skip


def dispatch_technician(
    scenario: dict[str, Any], args: dict[str, Any]
) -> dict[str, Any]:
    order = _order(scenario, str(args.get("order_id")))
    category, slot = str(args.get("category")), str(args.get("slot"))
    if category not in CATEGORIES:
        raise HelpdeskError(f"category must be one of {', '.join(CATEGORIES)}")
    if slot not in SLOTS:
        raise HelpdeskError("slot must be 'morning' or 'afternoon'")
    date = _date(args.get("date"), "date")
    today = store.today(scenario)
    if category == "emergency":
        if date != today:
            raise HelpdeskError(
                f"emergency visits are booked for today ({today.isoformat()})"
            )
    elif not today + dt.timedelta(days=1) <= date <= today + dt.timedelta(days=21):
        raise HelpdeskError("visits can be booked from tomorrow up to 21 days out")
    if date.weekday() == 6 and category != "emergency":
        raise HelpdeskError("no visits on Sundays")
    summary = _text(args, "problem_summary", 10)
    window = (
        "arriving within 2 hours of booking" if category == "emergency" else SLOTS[slot]
    )
    return {
        "dispatch_id": _ref("DS"),
        "order_id": order["order_id"],
        "category": category,
        "date": date.isoformat(),
        "weekday": date.strftime("%A"),
        "window": window,
        "customer_charge_usd": 129 if category == "paid_service_call" else 0,
        "_args": {"order_id": order["order_id"], "category": category, "date": date.isoformat(), "slot": slot,
                  "problem_summary": summary},
    }  # fmt: skip


def send_message_to_customer(
    scenario: dict[str, Any], args: dict[str, Any]
) -> dict[str, Any]:
    return {"delivered": True, "_args": {"message": _text(args, "message")}}


def end_conversation(scenario: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    return {
        "closed": True,
        "_args": {"resolution_summary": _text(args, "resolution_summary", 10)},
    }


ACTIONS = {
    "issue_refund": issue_refund,
    "approve_exchange_exception": approve_exchange_exception,
    "escalate_to_human": escalate_to_human,
    "schedule_followup": schedule_followup,
    "dispatch_technician": dispatch_technician,
    "send_message_to_customer": send_message_to_customer,
    "end_conversation": end_conversation,
}


def order_total(order: dict[str, Any]) -> float:
    total = float(order["delivery"]["fee"])
    for item in order["items"]:
        total += float(item["price"])
        if item.get("install"):
            total += float(item["install"]["fee"])
        if item.get("haul_away"):
            total += float(item["haul_away"]["fee"])
    return round(total, 2)


def call(tool: str, args: dict[str, Any]) -> dict[str, Any]:
    """Run one action and record it, successful or not. Returns what the caller is shown."""
    scenario = store.load_scenario()
    if tool not in ACTIONS:
        raise HelpdeskError(f"unknown tool {tool!r}")
    try:
        _require_open()
        result = ACTIONS[tool](scenario, args)
    except HelpdeskError as exc:
        store.append(
            {
                "kind": "action",
                "tool": tool,
                "args": args,
                "ok": False,
                "error": str(exc),
            }
        )
        return {"ok": False, "error": str(exc)}
    recorded_args = result.pop("_args")
    store.append(
        {
            "kind": "action",
            "tool": tool,
            "args": recorded_args,
            "ok": True,
            "result": result,
        }
    )
    if tool == "send_message_to_customer":
        store.append(
            {"kind": "message", "role": "agent", "text": recorded_args["message"]}
        )
    return {"ok": True, **result}


def record_customer_message(text: str, meta: dict[str, Any]) -> None:
    """The simulated customer's side of the conversation, written by the harness."""
    store.append({"kind": "message", "role": "customer", "text": text, **meta})
