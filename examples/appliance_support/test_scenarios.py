"""Every scenario's rubric must award the reference resolution full marks, and doing nothing less.

Runs each generated task's ``solution/solve.sh`` against the helpdesk backend on the host (no
sandbox), with the task's own sealed scenario, then grades the ledger the way the verifier does.

    ./build_tasks.py && uv run --with pytest pytest test_scenarios.py
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

from scenarios import SCENARIOS

TASKS = HERE / "tasks"


def _env(task: Path, state: Path, ticket: str) -> dict[str, str]:
    home = task / "environment" / "helpdesk"
    return {
        **os.environ,
        "HELPDESK_HOME": str(home),
        "HELPDESK_STATE_DIR": str(state),
        "HELPDESK_TICKET": ticket,
        "PATH": f"{home / 'bin'}:{os.environ['PATH']}",
    }


def _grade(task: Path, env: dict[str, str], out: Path) -> dict:
    command = [
        "helpdesk",
        "grade",
        "--rubric",
        str(task / "tests" / "rubric.json"),
        "--out",
        str(out),
    ]
    subprocess.run(command, env=env, check=True, capture_output=True)
    return json.loads((out / "grade.json").read_text())


@pytest.fixture(scope="module", autouse=True)
def built() -> None:
    subprocess.run(
        [sys.executable, str(HERE / "build_tasks.py")], check=True, capture_output=True
    )


@pytest.mark.parametrize("scenario", SCENARIOS, ids=[s["id"] for s in SCENARIOS])
def test_reference_resolution_scores_full_marks(scenario: dict, tmp_path: Path) -> None:
    task = TASKS / scenario["id"]
    env = _env(task, tmp_path / "state", scenario["ticket"])
    solve = subprocess.run(
        ["bash", str(task / "solution" / "solve.sh")],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert solve.returncode == 0, solve.stdout + solve.stderr
    result = _grade(task, env, tmp_path / "out")
    failed = [
        f"{c['desc']}: {c['detail']}" for c in result["checks"] if not c["passed"]
    ]
    assert result["reward"] == 1.0, failed
    assert result["cost_of_service_usd"] == result["reference_cost_usd"]


@pytest.mark.parametrize("scenario", SCENARIOS, ids=[s["id"] for s in SCENARIOS])
def test_closing_without_acting_scores_less(scenario: dict, tmp_path: Path) -> None:
    task = TASKS / scenario["id"]
    env = _env(task, tmp_path / "state", scenario["ticket"])
    subprocess.run(
        [
            "helpdesk",
            "call",
            "end_conversation",
            "--json",
            '{"resolution_summary": "Closed without action."}',
        ],
        env=env,
        check=True,
        capture_output=True,
    )
    assert _grade(task, env, tmp_path / "out")["reward"] < 1.0


def test_refunding_a_fraud_claim_is_a_critical_failure(tmp_path: Path) -> None:
    scenario = next(s for s in SCENARIOS if s["id"] == "fraud-not-delivered-claim")
    task = TASKS / scenario["id"]
    env = _env(task, tmp_path / "state", scenario["ticket"])
    subprocess.run(
        ["bash", str(task / "solution" / "solve.sh")],
        env=env,
        check=True,
        capture_output=True,
    )
    # The reference closes the ticket, so a refund after it is refused; issue one on a fresh ledger.
    env = _env(task, tmp_path / "state2", scenario["ticket"])
    refund = '{"order_id": "O-58133", "amount_usd": 1178, "reason_code": "goodwill", "note": "x"}'
    subprocess.run(
        ["helpdesk", "call", "issue_refund", "--json", refund],
        env=env,
        check=True,
        capture_output=True,
    )
    result = _grade(task, env, tmp_path / "out")
    assert result["critical_failure"] and result["reward"] == 0.0
    assert result["cost_of_service_usd"] == 1178


def test_backend_rejects_what_a_real_system_would(tmp_path: Path) -> None:
    scenario = next(s for s in SCENARIOS if s["id"] == "install-leak-warranty")
    task = TASKS / scenario["id"]
    env = _env(task, tmp_path / "state", scenario["ticket"])

    def call(tool: str, args: dict) -> dict:
        proc = subprocess.run(
            ["helpdesk", "call", tool, "--json", json.dumps(args)],
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(proc.stdout)

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
