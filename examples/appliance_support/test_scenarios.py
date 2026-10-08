"""Every scenario's rubric must award the reference resolution full marks, and doing nothing less.

Runs each ticket's reference resolution against the helpdesk backend on the host (no sandbox),
with the ticket's own sealed record, then grades the ledger the way the trial does.

    uv run --with pytest pytest test_scenarios.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "helpdesk" / "lib"))

import tickets
from scenarios import SCENARIOS


def _env(tmp: Path, scenario: dict) -> dict[str, str]:
    # A helpdesk home like the sandbox's: the backend, plus this one ticket's sealed record.
    home = tmp / "helpdesk"
    (home / "scenarios").mkdir(parents=True)
    for name in ("bin", "kb", "lib"):
        (home / name).symlink_to(HERE / "helpdesk" / name)
    (home / "scenarios" / f"{scenario['ticket']}.json").write_text(
        json.dumps(tickets.sealed(scenario))
    )
    (tmp / "rubric.json").write_text(json.dumps(tickets.rubric(scenario)))
    return {
        **os.environ,
        "HELPDESK_HOME": str(home),
        "HELPDESK_STATE_DIR": str(tmp / "state"),
        "HELPDESK_TICKET": scenario["ticket"],
        "PATH": f"{home / 'bin'}:{os.environ['PATH']}",
    }


def _call(env: dict[str, str], tool: str, args: dict) -> dict:
    proc = subprocess.run(
        ["helpdesk", "call", tool, "--json", json.dumps(args)],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(proc.stdout)


def _grade(tmp: Path, env: dict[str, str]) -> dict:
    command = ["helpdesk", "grade", "--rubric", str(tmp / "rubric.json"), "--out", str(tmp / "out")]
    subprocess.run(command, env=env, check=True, capture_output=True)
    return json.loads((tmp / "out" / "grade.json").read_text())


@pytest.mark.parametrize("scenario", SCENARIOS, ids=[s["id"] for s in SCENARIOS])
def test_weekday_labels_match_dates(scenario: dict) -> None:
    tickets.check_weekday_labels(scenario)


@pytest.mark.parametrize("scenario", SCENARIOS, ids=[s["id"] for s in SCENARIOS])
def test_reference_resolution_scores_full_marks(scenario: dict, tmp_path: Path) -> None:
    env = _env(tmp_path, scenario)
    for step in tickets.reference_steps(scenario):
        result = _call(env, step["tool"], step["args"])
        assert result["ok"], (step, result)
    result = _grade(tmp_path, env)
    failed = [
        f"{c['desc']}: {c['detail']}" for c in result["checks"] if not c["passed"]
    ]
    assert result["reward"] == 1.0, failed
    assert result["cost_of_service_usd"] == result["reference_cost_usd"]


@pytest.mark.parametrize("scenario", SCENARIOS, ids=[s["id"] for s in SCENARIOS])
def test_closing_without_acting_scores_less(scenario: dict, tmp_path: Path) -> None:
    env = _env(tmp_path, scenario)
    _call(env, "end_conversation", {"resolution_summary": "Closed without action."})
    assert _grade(tmp_path, env)["reward"] < 1.0


def test_refunding_a_fraud_claim_is_a_critical_failure(tmp_path: Path) -> None:
    scenario = next(s for s in SCENARIOS if s["id"] == "fraud-not-delivered-claim")
    env = _env(tmp_path, scenario)
    refund = {"order_id": "O-58133", "amount_usd": 1178, "reason_code": "goodwill", "note": "x"}
    _call(env, "issue_refund", refund)
    result = _grade(tmp_path, env)
    assert result["critical_failure"] and result["reward"] == 0.0
    assert result["cost_of_service_usd"] == 1178


def test_backend_rejects_what_a_real_system_would(tmp_path: Path) -> None:
    scenario = next(s for s in SCENARIOS if s["id"] == "install-leak-warranty")
    env = _env(tmp_path, scenario)

    def call(tool: str, args: dict) -> dict:
        return _call(env, tool, args)

    assert not call(
        "issue_refund",
        {"order_id": "O-99999", "amount_usd": 10, "reason_code": "goodwill"},
    )["ok"]
    assert not call(
        "issue_refund",
        {"order_id": "O-56300", "amount_usd": 99999, "reason_code": "goodwill"},
    )["ok"]
    sunday = {
        "order_id": "O-56300",
        "category": "install_warranty",
        "date": "2026-10-11",
        "slot": "morning",
        "problem_summary": "drip at drain",
    }
    assert not call("dispatch_technician", sunday)["ok"]
    assert call("dispatch_technician", {**sunday, "date": "2026-10-08"})["ok"]
    assert call("end_conversation", {"resolution_summary": "Visit booked."})["ok"]
    assert not call(
        "schedule_followup", {"due_date": "2026-10-09", "note": "too late, closed"}
    )["ok"]
