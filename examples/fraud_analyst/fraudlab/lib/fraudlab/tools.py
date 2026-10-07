"""The analyst's tools, metered and recorded in the case ledger."""

from __future__ import annotations

import json
import sqlite3
import time
from typing import Any

from fraudlab import store
from fraudlab.store import FraudlabError

PRICES_USD = {"deep_verification": 5.0, "social_network_search": 2.0}
MAX_SQL_ROWS = 200
SQL_TIMEOUT_SEC = 10.0
MIN_ANALYSIS_CHARS = 200


def _closed() -> bool:
    return any(
        e.get("tool") == "submit_decision" and e.get("ok") for e in store.read_ledger()
    )


def run_sql(args: dict[str, Any]) -> dict[str, Any]:
    query = str(args.get("query") or "").strip()
    if not query:
        raise FraudlabError("query is required")
    connection = sqlite3.connect(f"file:{store.WAREHOUSE}?mode=ro", uri=True)
    deadline = time.monotonic() + SQL_TIMEOUT_SEC
    connection.set_progress_handler(lambda: int(time.monotonic() > deadline), 10_000)
    try:
        cursor = connection.execute(query)
        columns = [d[0] for d in cursor.description or []]
        rows = cursor.fetchmany(MAX_SQL_ROWS + 1)
    except sqlite3.Error as exc:
        raise FraudlabError(f"SQL error: {exc}") from exc
    finally:
        connection.close()
    truncated = len(rows) > MAX_SQL_ROWS
    return {
        "columns": columns,
        "rows": [list(r) for r in rows[:MAX_SQL_ROWS]],
        "truncated": truncated,
    }


def chalk_query(args: dict[str, Any]) -> dict[str, Any]:
    account_id = str(args.get("account_id") or "")
    features = json.loads(store.FEATURES.read_text())
    if account_id not in features:
        raise FraudlabError(f"unknown account {account_id!r}")
    wanted = args.get("features") or list(features[account_id])
    unknown = [f for f in wanted if f not in features[account_id]]
    if unknown:
        raise FraudlabError(
            f"unknown features {unknown}; available: {sorted(features[account_id])}"
        )
    return {
        "account_id": account_id,
        "features": {f: features[account_id][f] for f in wanted},
    }


def deep_verification(args: dict[str, Any]) -> dict[str, Any]:
    account_id = str(args.get("account_id") or "")
    return {"account_id": account_id, **store.sealed(account_id)["deep_verification"]}


def social_network_search(args: dict[str, Any]) -> dict[str, Any]:
    account_id = str(args.get("account_id") or "")
    return {
        "account_id": account_id,
        **store.sealed(account_id)["social_network_search"],
    }


def submit_decision(args: dict[str, Any]) -> dict[str, Any]:
    decision = str(args.get("decision") or "").lower()
    if decision not in ("approve", "deny"):
        raise FraudlabError("decision must be 'approve' or 'deny'")
    analysis = str(args.get("analysis") or "").strip()
    if len(analysis) < MIN_ANALYSIS_CHARS:
        raise FraudlabError(
            f"analysis must justify the decision (at least {MIN_ANALYSIS_CHARS} characters)"
        )
    try:
        confidence = float(args.get("confidence"))
    except (TypeError, ValueError) as exc:
        raise FraudlabError("confidence must be a number between 0 and 1") from exc
    if not 0 <= confidence <= 1:
        raise FraudlabError("confidence must be between 0 and 1")
    return {
        "case_id": store.case_id(),
        "decision": decision,
        "confidence": confidence,
        "analysis": analysis,
    }


TOOLS = {
    "run_sql": run_sql,
    "chalk_query": chalk_query,
    "deep_verification": deep_verification,
    "social_network_search": social_network_search,
    "submit_decision": submit_decision,
}


def call(tool: str, args: dict[str, Any]) -> dict[str, Any]:
    """Run one tool, record it with its cost, and return what the analyst is shown."""
    if tool not in TOOLS:
        raise FraudlabError(f"unknown tool {tool!r}")
    cost = PRICES_USD.get(tool, 0.0)
    try:
        if _closed():
            raise FraudlabError("this case is closed")
        result = TOOLS[tool](args)
    except FraudlabError as exc:
        # A refused call is not billed.
        store.append(
            {
                "tool": tool,
                "args": args,
                "ok": False,
                "error": str(exc),
                "cost_usd": 0.0,
            }
        )
        return {"ok": False, "error": str(exc)}
    store.append(
        {"tool": tool, "args": args, "ok": True, "cost_usd": cost, "result": result}
    )
    return {"ok": True, "cost_usd": cost, **result}
