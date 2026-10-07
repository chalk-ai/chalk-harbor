"""Grade a case from its ledger: the decision against the hidden label, and what it cost."""

from __future__ import annotations

from typing import Any

DECISION_FOR_LABEL = {"fraud": "deny", "legit": "approve"}


def grade(rubric: dict[str, Any], ledger: list[dict[str, Any]]) -> dict[str, Any]:
    calls = [e for e in ledger if e.get("ok")]
    submitted = next(
        (e["result"] for e in calls if e["tool"] == "submit_decision"), None
    )
    expected = DECISION_FOR_LABEL[rubric["label"]]
    correct = submitted is not None and submitted["decision"] == expected
    cost = round(sum(e.get("cost_usd", 0.0) for e in calls), 2)
    paid = {
        tool: sum(1 for e in calls if e["tool"] == tool)
        for tool in ("deep_verification", "social_network_search")
    }
    return {
        "case_id": rubric["case_id"],
        "reward": 1.0 if correct else 0.0,
        "correct": correct,
        "expected_decision": expected,
        "decision": submitted["decision"] if submitted else None,
        "confidence": submitted["confidence"] if submitted else None,
        "analysis": submitted["analysis"] if submitted else None,
        "cost_usd": cost,
        "paid_calls": paid,
        "tool_calls": {
            tool: sum(1 for e in calls if e["tool"] == tool)
            for tool in {e["tool"] for e in calls}
        },
        "refused_calls": sum(1 for e in ledger if not e.get("ok")),
    }
