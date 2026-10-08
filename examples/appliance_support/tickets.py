"""What the agent, the sandbox and the grader see of each ticket in ``scenarios.py``.

* ``render_instruction`` -- the agent's ticket prompt: the customer's message, a pre-fetched
  profile, the contact history and every order record.
* ``sealed`` -- the ticket's record plus the simulated customer's brief. It goes into the
  ticket's sandbox readable only by root, so the agent's tools cannot see it.
* ``rubric`` -- the policy checks the action ledger is graded against, and the reference cost.
* ``reference_steps`` -- the policy-correct resolution as helpdesk calls, for the self-checks.
"""


from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "helpdesk" / "lib"))

from helpdesk.costs import cost_of_service
from scenarios import TIER_LIMITS, TODAY

SERVICE_LABELS = {
    "standard": "Standard delivery",
    "white_glove": "White-glove delivery",
}


def money(value: float) -> str:
    return f"${value:,.2f}" if value % 1 else f"${value:,.0f}"


def check_weekday_labels(scenario: dict[str, Any]) -> None:
    # Windows are written like "Mon 2026-10-05 8am-12pm"; a wrong weekday would confuse the agent.
    for order in scenario["orders"]:
        parts = order["delivery"]["window"].split()
        if len(parts) >= 2 and parts[0] in (
            "Mon",
            "Tue",
            "Wed",
            "Thu",
            "Fri",
            "Sat",
            "Sun",
        ):
            actual = dt.date.fromisoformat(parts[1]).strftime("%a")
            assert actual == parts[0], (
                f"{scenario['id']}: {parts[1]} is a {actual}, not {parts[0]}"
            )


def render_instruction(s: dict[str, Any]) -> str:
    c = s["customer"]
    opened = dt.datetime.fromisoformat(s["opened_at"])
    lines = [
        f"# Ticket {s['ticket']} ({s['channel']})",
        "",
        f"Opened {opened:%A %Y-%m-%d at %H:%M} (today). The customer is waiting for your reply.",
        "",
        "## Customer's message",
        "",
        *[f"> {line}" if line else ">" for line in s["inbound"].splitlines()],
        "",
        "## Customer profile (pre-fetched)",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Name | {c['name']} ({c['customer_id']}) |",
        f"| Tier | {c['tier']} |",
        f"| Lifetime value | {money(c['ltv'])} across {c['orders']} orders |",
        f"| Customer since | {c['since']} |",
        "| Refunds/credits, last 180 days | "
        + (
            "; ".join(
                f"{r['date']} {money(r['amount'])} {r['reason']}"
                for r in c["refunds_180d"]
            )
            or "none"
        )
        + " |",
        "| Chargebacks | "
        + (
            "; ".join(
                f"{x['date']} {x['order']} {money(x['amount'])}"
                for x in c["chargebacks"]
            )
            or "none"
        )
        + " |",
        "| Out-of-window exceptions, last 12 months | "
        + (
            "; ".join(
                f"{x['date']} {x['id']} {x['detail']}" for x in c["exceptions_12m"]
            )
            or "none"
        )
        + " |",
        f"| Survey history | {c['csat_history']} |",
        f"| Payment method on file | {c['payment']} |",
    ]
    if c["notes"]:
        lines.append(f"| Notes | {c['notes']} |")
    lines.append(
        f"| Authorized users | {', '.join(c.get('authorized_users') or []) or 'none'} |"
    )
    if s.get("contact"):
        lines += ["", f"**Contacting us:** {s['contact']}"]
    lines += ["", "## Contact history", ""]
    if s["history"]:
        lines += [
            f"- {h['date']} {h['ticket']} ({h['channel']}): {h['summary']}"
            for h in s["history"]
        ]
    else:
        lines.append("No previous contacts.")
    for order in s["orders"]:
        d = order["delivery"]
        lines += [
            "",
            f"## Order {order['order_id']}",
            "",
            f"- Ordered {order['order_date']}; status: **{order['status']}**",
            f"- {SERVICE_LABELS[d['service']]} ({money(d['fee'])}); window {d['window']}; "
            + (
                f"delivered {d['delivered_on']}"
                if d["delivered_on"]
                else "not delivered yet"
            ),
        ]
        if d["events"]:
            lines.append("- Delivery events and driver notes:")
            lines += [f"  - {e}" for e in d["events"]]
        if order.get("payments"):
            lines.append("- Payment log:")
            lines += [f"  - {e}" for e in order["payments"]]
        lines += [
            "",
            "| SKU | Item | Paid | Current Larkspur price | Installation | ProtectPlan | Haul-away |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for item in order["items"]:
            install = item["install"]
            install_text = (
                f"{install['type']} {money(install['fee'])}, {install['status']}"
                + (f" {install['date']}" if install.get("date") else "")
                if install
                else "none"
            )
            plan = item["protect_plan"]
            plan_text = f"{plan['term']}, expires {plan['expires']}" if plan else "none"
            haul = item["haul_away"]
            haul_text = (
                f"{haul['units']} unit {money(haul['fee'])}, {haul['status']}"
                if haul
                else "none"
            )
            lines.append(
                f"| {item['sku']} | {item['name']} ({item['brand']}) | {money(item['price'])} | "
                f"{money(item['current_price'])} | {install_text} | {plan_text} | {haul_text} |"
            )
        from helpdesk.tools import order_total

        lines.append(f"\nOrder total paid: {money(order_total(order))}")
    lines += [
        "",
        "---",
        "",
        f"Resolve this ticket. Today is {dt.date.fromisoformat(TODAY):%A %Y-%m-%d}. Talk to the customer only "
        + "through `send_message_to_customer`, and call `end_conversation` when you are done.",
        "",
    ]
    return "\n".join(lines)


def rubric(s: dict[str, Any]) -> dict[str, Any]:
    limit = TIER_LIMITS[s["customer"]["tier"]]
    checks = [
        *s["checks"],
        {"kind": "authority", "limit": limit, "weight": 2, "critical": True,
         "desc": f"refunds stay within the ${limit} {s['customer']['tier']}-tier authority"},
        {"kind": "messaged", "weight": 1, "critical": False, "desc": "replied to the customer"},
        {"kind": "closed", "weight": 1, "critical": False, "desc": "closed the conversation"},
    ]  # fmt: skip
    reference_cost = cost_of_service(s["reference"], sealed(s))["total_usd"]
    return {
        "ticket": s["ticket"],
        "scenario": s["id"],
        "checks": [{"id": f"c{i + 1}", **check} for i, check in enumerate(checks)],
        "reference_actions": s["reference"],
        "reference_cost_usd": reference_cost,
    }


def sealed(s: dict[str, Any]) -> dict[str, Any]:
    """What the sandbox holds about a ticket: the record, plus the simulated customer's brief."""
    return {"ticket": s["ticket"], "scenario": s["id"], "today": TODAY, "channel": s["channel"],
            "customer": s["customer"], "history": s["history"], "orders": s["orders"],
            "inbound": s["inbound"], "sim": s["sim"]}  # fmt: skip


def reference_steps(s: dict[str, Any]) -> list[dict[str, Any]]:
    """The policy-correct resolution, as the helpdesk calls that carry it out, in order."""
    steps = list(s["reference"])
    if not any(a["tool"] == "send_message_to_customer" for a in steps):
        steps.insert(0, {"tool": "send_message_to_customer",
                         "args": {"message": "Thanks for reaching out to Larkspur. I've looked into your order and taken care of this for you."}})  # fmt: skip
    steps.append(
        {
            "tool": "end_conversation",
            "args": {"resolution_summary": f"Reference resolution: {s['brief']}"},
        }
    )
    return steps
