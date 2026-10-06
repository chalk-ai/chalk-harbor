"""What each support action costs Larkspur (the same table as KB-121)."""

from __future__ import annotations

from typing import Any

EXCEPTION_RATE = 0.30
DISPATCH_COST = {"emergency": 260.0, "haul_away_pickup": 95.0}
DEFAULT_DISPATCH_COST = 145.0
ESCALATION_COST = {"urgent": 40.0}
DEFAULT_ESCALATION_COST = 25.0
FOLLOWUP_COST = 4.0


def _item_price(scenario: dict[str, Any], order_id: str, sku: str) -> float:
    for order in scenario["orders"]:
        if order["order_id"] == order_id:
            for item in order["items"]:
                if item["sku"] == sku:
                    return float(item["price"])
    return 0.0


def action_cost(action: dict[str, Any], scenario: dict[str, Any]) -> float:
    tool, args = action["tool"], action.get("args", {})
    if tool == "issue_refund":
        return float(args["amount_usd"])
    if tool == "approve_exchange_exception":
        return round(
            EXCEPTION_RATE * _item_price(scenario, args["order_id"], args["sku"]), 2
        )
    if tool == "dispatch_technician":
        return DISPATCH_COST.get(args["category"], DEFAULT_DISPATCH_COST)
    if tool == "escalate_to_human":
        return ESCALATION_COST.get(args["priority"], DEFAULT_ESCALATION_COST)
    if tool == "schedule_followup":
        return FOLLOWUP_COST
    return 0.0


def cost_of_service(
    actions: list[dict[str, Any]], scenario: dict[str, Any]
) -> dict[str, Any]:
    """Total cost of the successful actions, with a per-category breakdown."""
    breakdown: dict[str, float] = {}
    for action in actions:
        cost = action_cost(action, scenario)
        if cost:
            breakdown[action["tool"]] = round(
                breakdown.get(action["tool"], 0.0) + cost, 2
            )
    return {"total_usd": round(sum(breakdown.values()), 2), "breakdown_usd": breakdown}
