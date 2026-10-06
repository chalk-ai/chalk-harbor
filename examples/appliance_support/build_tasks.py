#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Generate the 30 Harbor tasks under ``tasks/`` from ``scenarios.py``.

    ./build_tasks.py

Every task gets an identical ``environment/`` -- the helpdesk backend, the knowledge base and
all 30 sealed customer briefs -- so all of them share one cached sandbox image; the task picks
its ticket through ``HELPDESK_TICKET`` in ``task.toml``. What differs per task is the agent's
``instruction.md``, the rubric in ``tests/`` (which reaches the sandbox only for grading), and
the oracle's ``solution/solve.sh``.
"""

from __future__ import annotations

import base64
import datetime as dt
import json
import shutil
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "helpdesk" / "lib"))

from helpdesk.costs import cost_of_service
from scenarios import SCENARIOS, TIER_LIMITS, TODAY

TASKS = HERE / "tasks"
SERVICE_LABELS = {
    "standard": "Standard delivery",
    "white_glove": "White-glove delivery",
}

DOCKERFILE = """\
FROM python:3.13-slim

# The agent's python/bash tools run as this unprivileged user; the helpdesk state and the
# simulated customers' briefs are root-only.
RUN useradd --create-home --shell /bin/bash agent
COPY helpdesk /opt/helpdesk
RUN chmod -R a+rX /opt/helpdesk \\
    && chmod 700 /opt/helpdesk/scenarios && chmod 600 /opt/helpdesk/scenarios/*.json \\
    && ln -s /opt/helpdesk/bin/helpdesk /usr/local/bin/helpdesk \\
    && install -d -m 700 /var/lib/helpdesk

WORKDIR /home/agent
"""

TEST_SH = """\
#!/bin/bash
mkdir -p /logs/verifier
helpdesk grade --rubric /tests/rubric.json --out /logs/verifier
"""


def money(value: float) -> str:
    return f"${value:,.2f}" if value % 1 else f"${value:,.0f}"


def _check_weekday_labels(scenario: dict[str, Any]) -> None:
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


def _call(tool: str, args: dict[str, Any]) -> str:
    encoded = base64.b64encode(json.dumps(args).encode()).decode()
    return f"helpdesk call {tool} --b64 {encoded}"


def solve_sh(s: dict[str, Any]) -> str:
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
    lines = [
        "#!/bin/bash",
        "# The policy-correct resolution, for Harbor's oracle agent.",
        "set -euo pipefail",
    ]
    for step in steps:
        lines.append(
            f"{_call(step['tool'], step['args'])} | tee -a /tmp/oracle.log | grep -q '\"ok\": true'"
        )
    return "\n".join(lines) + "\n"


def task_toml(s: dict[str, Any]) -> str:
    keywords = json.dumps(["customer-support", s["category"], s["difficulty"]])
    return f"""\
schema_version = "1.4"
artifacts = []

[task]
name = "chalk/larkspur-{s["id"]}"
version = "1.0.0"
description = {json.dumps(s["title"])}
authors = []
keywords = {keywords}

[metadata]
ticket = "{s["ticket"]}"
category = "{s["category"]}"
difficulty = "{s["difficulty"]}"

[verifier]
timeout_sec = 120.0
collect = []

[verifier.env]

[agent]
timeout_sec = 900.0

[environment]
network_mode = "no-network"
build_timeout_sec = 900.0
os = "linux"
cpus = 1
memory_mb = 2048
mcp_servers = []

[environment.env]
HELPDESK_TICKET = "{s["ticket"]}"

[solution.env]
"""


def build_environment(target: Path) -> None:
    helpdesk = target / "helpdesk"
    shutil.copytree(
        HERE / "helpdesk",
        helpdesk,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    sealed_dir = helpdesk / "scenarios"
    sealed_dir.mkdir()
    for s in SCENARIOS:
        (sealed_dir / f"{s['ticket']}.json").write_text(
            json.dumps(sealed(s), indent=1, sort_keys=True)
        )
    (target / "Dockerfile").write_text(DOCKERFILE)


def main() -> int:
    if TASKS.exists():
        shutil.rmtree(TASKS)
    for s in SCENARIOS:
        _check_weekday_labels(s)
        task = TASKS / s["id"]
        build_environment(task / "environment")
        (task / "instruction.md").write_text(render_instruction(s))
        (task / "task.toml").write_text(task_toml(s))
        (task / "tests").mkdir()
        (task / "tests" / "test.sh").write_text(TEST_SH)
        (task / "tests" / "test.sh").chmod(0o755)
        (task / "tests" / "rubric.json").write_text(json.dumps(rubric(s), indent=2))
        (task / "solution").mkdir()
        (task / "solution" / "solve.sh").write_text(solve_sh(s))
        (task / "solution" / "solve.sh").chmod(0o755)
    print(f"wrote {len(SCENARIOS)} tasks to {TASKS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
