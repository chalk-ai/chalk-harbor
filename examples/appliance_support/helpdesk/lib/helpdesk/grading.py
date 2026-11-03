"""Grade a finished ticket against its policy rubric, from the action ledger alone.

A rubric is a list of weighted checks (built in ``scenarios.py``). The reward is the weighted
share of checks passed, or 0 when any check marked critical fails (a refund on a fraud ticket,
a safety emergency handled without safety instructions, ...).
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from helpdesk.costs import cost_of_service

_PRIORITY_RANK = {"normal": 0, "high": 1, "urgent": 2}
_VISIT_CATEGORIES = (
    "install_warranty",
    "installation",
    "protect_plan_repair",
    "paid_service_call",
    "emergency",
)


def add_business_days(day: dt.date, n: int) -> dt.date:
    while n > 0:
        day += dt.timedelta(days=1)
        if day.weekday() < 5:
            n -= 1
    return day


def successful_actions(ledger: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [e for e in ledger if e.get("kind") == "action" and e.get("ok")]


def _of(actions: list[dict[str, Any]], tool: str) -> list[dict[str, Any]]:
    return [a["args"] for a in actions if a["tool"] == tool]


def _agent_messages(ledger: list[dict[str, Any]]) -> list[str]:
    return [
        e["text"]
        for e in ledger
        if e.get("kind") == "message" and e.get("role") == "agent"
    ]


def _check(
    check: dict[str, Any], actions: list[dict[str, Any]], ledger: list[dict[str, Any]]
) -> tuple[bool, str]:
    kind = check["kind"]
    if kind == "refund_total":
        refunds = [
            r
            for r in _of(actions, "issue_refund")
            if not check.get("reasons") or r["reason_code"] in check["reasons"]
        ]
        total = round(sum(r["amount_usd"] for r in refunds), 2)
        return check["min"] - 0.005 <= total <= check[
            "max"
        ] + 0.005, f"refunded ${total:.2f}"
    if kind == "authority":
        total = round(sum(r["amount_usd"] for r in _of(actions, "issue_refund")), 2)
        return total <= check[
            "limit"
        ] + 0.005, f"refunded ${total:.2f} against a ${check['limit']} limit"
    if kind == "exception":
        granted = [e["sku"] for e in _of(actions, "approve_exchange_exception")]
        if check["expected"]:
            return (check.get("sku") in granted) if check.get("sku") else bool(
                granted
            ), f"exceptions: {granted or 'none'}"
        return not granted, f"exceptions: {granted or 'none'}"
    if kind == "escalation":
        escalations = _of(actions, "escalate_to_human")
        seen = [f"{e['queue']}/{e['priority']}" for e in escalations]
        if not check["expected"]:
            return not escalations, f"escalations: {seen or 'none'}"
        wanted = _PRIORITY_RANK.get(check.get("priority") or "normal", 0)
        ok = any(
            (not check.get("queues") or e["queue"] in check["queues"])
            and _PRIORITY_RANK[e["priority"]] >= wanted
            for e in escalations
        )
        return ok, f"escalations: {seen or 'none'}"
    if kind == "escalation_only_in":
        queues = [e["queue"] for e in _of(actions, "escalate_to_human")]
        return all(
            q in check["queues"] for q in queues
        ), f"escalations: {queues or 'none'}"
    if kind == "dispatch":
        visits = _of(actions, "dispatch_technician")
        seen = [f"{v['category']} {v['date']} {v['slot']}" for v in visits]
        if not check["expected"]:
            return not visits, f"dispatches: {seen or 'none'}"

        def matches(v: dict[str, Any]) -> bool:
            if check.get("categories") and v["category"] not in check["categories"]:
                return False
            if not check.get("slots"):
                return True
            return any(
                v["date"] == d and (s is None or v["slot"] == s)
                for d, s in check["slots"]
            )

        return any(matches(v) for v in visits), f"dispatches: {seen or 'none'}"
    if kind == "followup":
        due = [f["due_date"] for f in _of(actions, "schedule_followup")]
        return any(
            check["due_from"] <= d <= check["due_to"] for d in due
        ), f"follow-ups due: {due or 'none'}"
    if kind == "followup_after_dispatch":
        visits = [
            dt.date.fromisoformat(v["date"])
            for v in _of(actions, "dispatch_technician")
            if v["category"] in _VISIT_CATEGORIES
        ]
        due = [
            dt.date.fromisoformat(f["due_date"])
            for f in _of(actions, "schedule_followup")
        ]
        ok = any(
            add_business_days(v, check["min_business_days"])
            <= d
            <= add_business_days(v, check["max_business_days"])
            for v in visits
            for d in due
        )
        return (
            ok,
            f"visits: {[v.isoformat() for v in visits] or 'none'}; follow-ups due: {[d.isoformat() for d in due] or 'none'}",
        )
    if kind == "mentions":
        messages = _agent_messages(ledger)
        scope = messages[:1] if check.get("first_message") else messages
        text = "\n".join(scope).lower()
        missing = [
            g for g in check["groups"] if not any(word.lower() in text for word in g)
        ]
        return not missing, (
            "missing: " + "; ".join(" | ".join(g) for g in missing)
        ) if missing else "all mentioned"
    if kind == "not_mentions":
        text = "\n".join(_agent_messages(ledger)).lower()
        found = [w for w in check["words"] if w.lower() in text]
        return not found, f"mentioned: {found}" if found else "none mentioned"
    if kind == "messaged":
        n = len(_agent_messages(ledger))
        return n > 0, f"{n} messages to the customer"
    if kind == "closed":
        closed = bool(_of(actions, "end_conversation"))
        return closed, "conversation closed" if closed else "conversation never closed"
    raise ValueError(f"unknown check kind {kind!r}")


def grade(
    rubric: dict[str, Any], ledger: list[dict[str, Any]], scenario: dict[str, Any]
) -> dict[str, Any]:
    actions = successful_actions(ledger)
    results = []
    for check in rubric["checks"]:
        passed, detail = _check(check, actions, ledger)
        results.append({**check, "passed": passed, "detail": detail})
    total = sum(r["weight"] for r in results)
    earned = sum(r["weight"] for r in results if r["passed"])
    critical_failure = any(r["critical"] and not r["passed"] for r in results)
    cost = cost_of_service(actions, scenario)
    return {
        "ticket": rubric["ticket"],
        "reward": 0.0 if critical_failure else round(earned / total, 4),
        "critical_failure": critical_failure,
        "checks_passed": sum(r["passed"] for r in results),
        "checks_total": len(results),
        "checks": results,
        "cost_of_service_usd": cost["total_usd"],
        "cost_breakdown_usd": cost["breakdown_usd"],
        "reference_cost_usd": rubric["reference_cost_usd"],
        "failed_tool_calls": sum(
            1 for e in ledger if e.get("kind") == "action" and not e.get("ok")
        ),
        "customer_messages": sum(
            1
            for e in ledger
            if e.get("kind") == "message" and e.get("role") == "customer"
        ),
        "agent_messages": len(_agent_messages(ledger)),
    }
