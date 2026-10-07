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
from population import HARD_ARCHETYPES, HARD_QUEUE, generate

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


HARD = [c for c in CASES if c["archetype"] in HARD_ARCHETYPES]


def _sql(env: dict[str, str], query: str) -> list[list]:
    result = _call(env, "run_sql", {"query": query})
    assert result["ok"], result
    return result["rows"]


def test_the_hard_tier_follows_the_standard_queue() -> None:
    assert [c["case_id"] for c in CASES[:40]] == [f"K-{5001 + i}" for i in range(40)]
    assert [c["archetype"] for c in CASES[40:]] == HARD_QUEUE
    assert sum(c["label"] == "fraud" for c in HARD) == len(HARD) // 2


@pytest.mark.parametrize("case", HARD, ids=[c["case_id"] for c in HARD])
def test_hard_cases_hinge_on_the_evidence_they_were_built_around(
    home: Path, case: dict, tmp_path: Path
) -> None:
    env = _env(home, case, tmp_path / "state")
    account, archetype = case["account_id"], case["archetype"]
    features = _call(env, "chalk_query", {"account_id": account})["features"]
    risk = features["account.risk_score"]
    # The production model points the wrong way on every hard case.
    assert risk < 0.5 if case["label"] == "fraud" else risk > 0.5
    if archetype == "payout_mule_link":
        statuses = _sql(
            env,
            f"SELECT DISTINCT a.status FROM transactions t1 JOIN transactions t2 ON t1.destination = t2.destination AND t2.account_id != t1.account_id JOIN accounts a ON a.account_id = t2.account_id WHERE t1.account_id = '{account}' AND t1.status = 'pending'",
        )
        assert statuses == [["closed_fraud"]]
        assert features["account.device_shared_account_count"] == 0
    elif archetype == "ato_quiet":
        # The owner's device and an unseen device are both active in the last two days, at home.
        recent = _sql(
            env,
            f"SELECT COUNT(DISTINCT device_id), MAX(vpn), GROUP_CONCAT(DISTINCT ip_country) FROM logins WHERE account_id = '{account}' AND ts >= '2026-10-05'",
        )
        assert recent == [[2, 0, "US"]]
    elif archetype == "bust_out":
        cards = _sql(
            env,
            f"SELECT COUNT(DISTINCT destination) FROM transactions WHERE account_id = '{account}' AND type = 'deposit' AND ts >= '2026-09-27'",
        )[0][0]
        assert cards >= 5
        assert _sql(
            env,
            f"SELECT COUNT(*) FROM transactions WHERE account_id = '{account}' AND type = 'payout' AND status = 'settled'",
        ) == [[0]]
    elif archetype == "legit_account_recovery":
        # The old device went silent before the new one appeared: a replacement, not a second user.
        spans = _sql(
            env,
            f"SELECT device_id, MIN(ts), MAX(ts) FROM logins WHERE account_id = '{account}' GROUP BY 1 ORDER BY 2",
        )
        assert len(spans) == 2 and spans[0][2] < spans[1][1]
    elif archetype == "legit_resold_device":
        # The fraudster's last use of the device predates this account.
        rows = _sql(
            env,
            f"SELECT a2.status, MAX(l2.ts), a1.created_at FROM logins l1 JOIN logins l2 ON l1.device_id = l2.device_id AND l2.account_id != l1.account_id JOIN accounts a1 ON a1.account_id = l1.account_id JOIN accounts a2 ON a2.account_id = l2.account_id WHERE l1.account_id = '{account}' GROUP BY l2.account_id",
        )
        assert (
            len(rows) == 1 and rows[0][0] == "closed_fraud" and rows[0][1] < rows[0][2]
        )
    elif archetype == "legit_vpn_privacy":
        assert _sql(
            env,
            f"SELECT COUNT(DISTINCT device_id), COUNT(DISTINCT ip_country), MIN(vpn) FROM logins WHERE account_id = '{account}'",
        ) == [[1, 1, 1]]
        prior = _sql(
            env,
            f"SELECT COUNT(*) FROM transactions p JOIN transactions h ON h.account_id = p.account_id AND h.destination = p.destination AND h.status = 'settled' WHERE p.account_id = '{account}' AND p.status = 'pending'",
        )
        assert prior[0][0] >= 3
