"""Building blocks for scenarios: rubric checks, reference actions and record helpers."""

from __future__ import annotations

from typing import Any

TODAY = "2026-10-06"

TIER_LIMITS = {"Standard": 250, "Silver": 250, "Gold": 500, "Platinum": 500}


# -- rubric helpers ------------------------------------------------------------------------


def refund(
    lo: float,
    hi: float,
    *,
    reasons: list[str] | None = None,
    w: int = 3,
    critical: bool = False,
) -> dict[str, Any]:
    scope = f" ({', '.join(reasons)})" if reasons else ""
    desc = (
        f"refunds{scope} total ${lo:.2f}"
        if lo == hi
        else f"refunds{scope} total ${lo:.2f}-${hi:.2f}"
    )
    return {"kind": "refund_total", "min": lo, "max": hi, "reasons": reasons, "weight": w,
            "critical": critical, "desc": desc}  # fmt: skip


def no_refund(*, w: int = 3, critical: bool = False) -> dict[str, Any]:
    return {"kind": "refund_total", "min": 0, "max": 0, "reasons": None, "weight": w,
            "critical": critical, "desc": "no refunds or credits issued"}  # fmt: skip


def exception(
    expected: bool, sku: str | None = None, *, w: int = 3, critical: bool = False
) -> dict[str, Any]:
    desc = (
        f"out-of-window exchange exception approved for {sku}"
        if expected
        else "no out-of-window exchange exception"
    )
    return {
        "kind": "exception",
        "expected": expected,
        "sku": sku,
        "weight": w,
        "critical": critical,
        "desc": desc,
    }


def escalation(
    expected: bool,
    queues: list[str] | None = None,
    priority: str | None = None,
    *,
    w: int = 3,
    critical: bool = False,
) -> dict[str, Any]:
    if expected:
        desc = f"escalated to {'/'.join(queues or ['any queue'])}" + (
            f" at {priority} priority" if priority else ""
        )
    else:
        desc = "no escalation to a human"
    return {"kind": "escalation", "expected": expected, "queues": queues, "priority": priority,
            "weight": w, "critical": critical, "desc": desc}  # fmt: skip


def escalation_only_in(queues: list[str], *, w: int = 1) -> dict[str, Any]:
    return {"kind": "escalation_only_in", "queues": queues, "weight": w, "critical": False,
            "desc": f"any escalation goes to {'/'.join(queues)} only"}  # fmt: skip


def dispatch(
    expected: bool,
    categories: list[str] | None = None,
    slots: list[tuple[str, str | None]] | None = None,
    *,
    w: int = 3,
    critical: bool = False,
) -> dict[str, Any]:
    if expected:
        when = ", ".join(f"{d} {s or 'any slot'}" for d, s in slots or []) or "any date"
        desc = f"{'/'.join(categories or ['any'])} visit booked for {when}"
    else:
        desc = "no technician or crew dispatched"
    return {"kind": "dispatch", "expected": expected, "categories": categories,
            "slots": [list(s) for s in slots] if slots else None, "weight": w,
            "critical": critical, "desc": desc}  # fmt: skip


def followup(due_from: str, due_to: str, *, w: int = 2) -> dict[str, Any]:
    return {"kind": "followup", "expected": True, "due_from": due_from, "due_to": due_to,
            "weight": w, "critical": False, "desc": f"follow-up due {due_from}..{due_to}"}  # fmt: skip


def followup_after_visit(*, w: int = 2) -> dict[str, Any]:
    return {"kind": "followup_after_dispatch", "min_business_days": 1, "max_business_days": 3,
            "weight": w, "critical": False,
            "desc": "follow-up 1-3 business days after the booked visit"}  # fmt: skip


def mentions(
    *groups: list[str], w: int = 1, first_message: bool = False, critical: bool = False
) -> dict[str, Any]:
    where = (
        "first message to the customer" if first_message else "messages to the customer"
    )
    desc = f"{where} mention " + " and ".join("(" + " | ".join(g) + ")" for g in groups)
    return {"kind": "mentions", "groups": [list(g) for g in groups], "first_message": first_message,
            "weight": w, "critical": critical, "desc": desc}  # fmt: skip


def never_mentions(*words: str, w: int = 2) -> dict[str, Any]:
    return {"kind": "not_mentions", "words": list(words), "weight": w, "critical": False,
            "desc": "messages never mention " + " / ".join(words)}  # fmt: skip


# -- reference action helpers ---------------------------------------------------------------


def a_refund(order_id: str, amount: float, reason: str, note: str) -> dict[str, Any]:
    return {
        "tool": "issue_refund",
        "args": {
            "order_id": order_id,
            "amount_usd": amount,
            "reason_code": reason,
            "note": note,
        },
    }


def a_exception(order_id: str, sku: str, reason: str) -> dict[str, Any]:
    return {
        "tool": "approve_exchange_exception",
        "args": {"order_id": order_id, "sku": sku, "reason": reason},
    }


def a_escalate(queue: str, priority: str, summary: str) -> dict[str, Any]:
    return {
        "tool": "escalate_to_human",
        "args": {"queue": queue, "priority": priority, "summary": summary},
    }


def a_dispatch(
    order_id: str, category: str, date: str, slot: str, summary: str
) -> dict[str, Any]:
    return {"tool": "dispatch_technician", "args": {"order_id": order_id, "category": category, "date": date,
                                                    "slot": slot, "problem_summary": summary}}  # fmt: skip


def a_message(message: str) -> dict[str, Any]:
    return {"tool": "send_message_to_customer", "args": {"message": message}}


def a_followup(due: str, note: str) -> dict[str, Any]:
    return {"tool": "schedule_followup", "args": {"due_date": due, "note": note}}


def item(
    sku: str,
    name: str,
    brand: str,
    price: float,
    *,
    install: dict[str, Any] | None = None,
    protect_plan: dict[str, Any] | None = None,
    haul_away: dict[str, Any] | None = None,
    current_price: float | None = None,
) -> dict[str, Any]:
    return {"sku": sku, "name": name, "brand": brand, "price": price, "install": install,
            "protect_plan": protect_plan, "haul_away": haul_away,
            "current_price": current_price if current_price is not None else price}  # fmt: skip


def customer(
    cid: str,
    name: str,
    tier: str,
    ltv: float,
    since: str,
    orders: int,
    *,
    refunds_180d: list[dict[str, Any]] | None = None,
    chargebacks: list[dict[str, Any]] | None = None,
    exceptions_12m: list[dict[str, Any]] | None = None,
    csat_history: str = "no surveys",
    notes: str = "",
    payment: str = "Visa ending 4417",
) -> dict[str, Any]:
    return {"customer_id": cid, "name": name, "tier": tier, "ltv": ltv, "since": since, "orders": orders,
            "refunds_180d": refunds_180d or [], "chargebacks": chargebacks or [],
            "exceptions_12m": exceptions_12m or [], "csat_history": csat_history, "notes": notes,
            "payment": payment}  # fmt: skip
