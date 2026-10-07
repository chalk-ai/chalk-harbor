"""Every case's reference investigation scores 1.0 at its expected cost, and the wrong call scores 0.

./build_tasks.py && uv run --with pytest pytest test_cases.py
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from build_tasks import reference_steps
from population import generate

TASKS = HERE / "tasks"
CASES = generate()["cases"]


@pytest.fixture(scope="module")
def home(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The shared environment as the image build leaves it: data written in by materialize.py."""
    subprocess.run(
        [sys.executable, str(HERE / "build_tasks.py")], check=True, capture_output=True
    )
    home = tmp_path_factory.mktemp("image") / "fraudlab"
    shutil.copytree(
        TASKS / CASES[0]["case_id"].lower() / "environment" / "fraudlab", home
    )
    materialize = home / "gen" / "materialize.py"
    subprocess.run(
        [sys.executable, str(materialize), str(home)], check=True, capture_output=True
    )
    return home


def _env(home: Path, case: dict, state: Path) -> dict[str, str]:
    return {**os.environ, "FRAUDLAB_HOME": str(home), "FRAUDLAB_STATE_DIR": str(state),
            "FRAUDLAB_CASE": case["case_id"], "PATH": f"{home / 'bin'}:{os.environ['PATH']}"}  # fmt: skip


def _call(env: dict[str, str], tool: str, args: dict) -> dict:
    proc = subprocess.run(
        ["fraudlab", "call", tool, "--json", json.dumps(args)],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(proc.stdout)


def _grade(case: dict, env: dict[str, str], out: Path) -> dict:
    rubric = TASKS / case["case_id"].lower() / "tests" / "rubric.json"
    subprocess.run(
        ["fraudlab", "grade", "--rubric", str(rubric), "--out", str(out)],
        env=env,
        check=True,
        capture_output=True,
    )
    return json.loads((out / "grade.json").read_text())


@pytest.mark.parametrize("case", CASES, ids=[c["case_id"] for c in CASES])
def test_reference_investigation_is_right(
    home: Path, case: dict, tmp_path: Path
) -> None:
    env = _env(home, case, tmp_path / "state")
    solve = TASKS / case["case_id"].lower() / "solution" / "solve.sh"
    proc = subprocess.run(
        ["bash", str(solve)], env=env, capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    grade = _grade(case, env, tmp_path / "out")
    assert grade["reward"] == 1.0
    expected_cost = sum(
        {"deep_verification": 5.0, "social_network_search": 2.0}.get(t, 0.0)
        for t, _ in reference_steps(case)
    )
    assert grade["cost_usd"] == expected_cost


@pytest.mark.parametrize("case", CASES[:1] + CASES[-1:], ids=["fraud", "legit"])
def test_the_wrong_decision_scores_zero(home: Path, case: dict, tmp_path: Path) -> None:
    env = _env(home, case, tmp_path / "state")
    wrong = "approve" if case["label"] == "fraud" else "deny"
    result = _call(
        env,
        "submit_decision",
        {"decision": wrong, "confidence": 0.6, "analysis": "x" * 250},
    )
    assert result["ok"]
    assert _grade(case, env, tmp_path / "out")["reward"] == 0.0


def test_paid_tools_are_billed_and_refusals_are_not(home: Path, tmp_path: Path) -> None:
    case = CASES[0]
    env = _env(home, case, tmp_path / "state")
    assert (
        _call(env, "deep_verification", {"account_id": case["account_id"]})["cost_usd"]
        == 5.0
    )
    assert (
        _call(env, "social_network_search", {"account_id": case["account_id"]})[
            "cost_usd"
        ]
        == 2.0
    )
    assert not _call(env, "deep_verification", {"account_id": "A-0"})["ok"]
    assert not _call(
        env,
        "submit_decision",
        {"decision": "deny", "confidence": 0.9, "analysis": "too short"},
    )["ok"]
    assert _grade(case, env, tmp_path / "out")["cost_usd"] == 7.0


def test_sql_is_read_only_and_bounded(home: Path, tmp_path: Path) -> None:
    env = _env(home, CASES[0], tmp_path / "state")
    assert not _call(env, "run_sql", {"query": "DELETE FROM accounts"})["ok"]
    result = _call(env, "run_sql", {"query": "SELECT * FROM logins"})
    assert result["ok"] and result["truncated"] and len(result["rows"]) == 200
    statuses = _call(
        env,
        "run_sql",
        {
            "query": "SELECT DISTINCT status FROM accounts WHERE account_id IN (SELECT account_id FROM review_queue)"
        },
    )
    assert statuses["rows"] == [["pending_review"]]


def test_closed_case_refuses_more_work(home: Path, tmp_path: Path) -> None:
    case = CASES[0]
    env = _env(home, case, tmp_path / "state")
    assert _call(
        env,
        "submit_decision",
        {"decision": "deny", "confidence": 0.8, "analysis": "y" * 250},
    )["ok"]
    assert not _call(env, "deep_verification", {"account_id": case["account_id"]})["ok"]
