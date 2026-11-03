#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Print a run's per-ticket scores as a markdown table.

./summarize.py runs/<tag>.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SCORERS = {
    "policy": "larkspur-policy-compliance",
    "cost score": "larkspur-cost-of-service",
    "cost $": "cost_of_service_usd",
    "irate": "larkspur-customer-got-irate",
    "satisfied": "larkspur-customer-satisfied",
    "survey": "larkspur-csat-survey",
    "claims ok": "larkspur-agent-claims-accurate",
}


def _metadata(row: dict, scorer: str) -> dict:
    value = row.get(f"{scorer}_metadata")
    return json.loads(value) if isinstance(value, str) else value or {}


def main(path: str) -> int:
    run = json.loads(Path(path).read_text())
    rows = sorted(run["rows"], key=lambda r: r["task_name"])
    print(
        f"## {run['tag']}: {run['agent_model']} vs. {run['customer_model']} customers, judged by {run['judge_model']}\n"
    )
    print("| ticket | " + " | ".join(SCORERS) + " | ref $ | notes |")
    print("| --- |" + " ---: |" * (len(SCORERS) + 1) + " --- |")
    totals: dict[str, list[float]] = {name: [] for name in SCORERS}
    for row in rows:
        cells = []
        for name, scorer in SCORERS.items():
            value = row.get(f"{scorer}_value")
            if isinstance(value, (int, float)):
                totals[name].append(value)
                cells.append(f"{value:.0f}" if name == "cost $" else f"{value:.2f}")
            else:
                cells.append("-")
        output = json.loads(row["output"]) if row.get("output") else {}
        notes = _metadata(row, SCORERS["policy"]).get("failed_checks", [])
        notes += _metadata(row, SCORERS["claims ok"]).get("unbacked_claims", [])[:1]
        note = "; ".join(n.replace("|", "/") for n in notes)[:160]
        print(
            f"| `{row['task_name']}` | "
            + " | ".join(cells)
            + f" | {output.get('reference_cost_usd', '-')} | {note} |"
        )
    means = [
        f"{sum(v) / len(v):.0f}" if n == "cost $" else f"{sum(v) / len(v):.3f}"
        for n, v in totals.items()
        if v
    ]
    print("| **mean** | " + " | ".join(f"**{m}**" for m in means) + " | | |")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
