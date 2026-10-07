#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Generate the Kestrel Pay fraud-review tasks under ``tasks/`` from ``population.py``.

    ./build_tasks.py

One Harbor task per case in the review queue (40). Every task shares one ``environment/``: the
``fraudlab`` backend and the seeded population generator, which writes the warehouse (read-only
SQLite), the Chalk feature values and the paid tools' results (sealed root-only) at image build
time. All tasks therefore share one cached sandbox image; the task picks its case through
``FRAUDLAB_CASE``. The case's true label reaches the sandbox only with
``tests/rubric.json``, at grading time.
"""

from __future__ import annotations

import base64
import json
import shutil
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from population import generate

TASKS = HERE / "tasks"

DOCKERFILE = """\
FROM python:3.13-slim

# The analyst's python tool runs as this unprivileged user; the paid tools' results and the
# case ledger are root-only.
RUN useradd --create-home --shell /bin/bash agent
COPY fraudlab /opt/fraudlab
RUN python3 /opt/fraudlab/gen/materialize.py /opt/fraudlab \\
    && chmod -R a+rX /opt/fraudlab \\
    && chmod 700 /opt/fraudlab/sealed && chmod 600 /opt/fraudlab/sealed/*.json \\
    && ln -s /opt/fraudlab/bin/fraudlab /usr/local/bin/fraudlab \\
    && install -d -m 700 /var/lib/fraudlab

WORKDIR /home/agent
"""

TEST_SH = """\
#!/bin/bash
mkdir -p /logs/verifier
fraudlab grade --rubric /tests/rubric.json --out /logs/verifier
"""

DATA_DICTIONARY = """\
## What you can use

**Warehouse (SQL, free).** SQLite, read-only. Also readable from the python tool at
`/opt/fraudlab/data/warehouse.sqlite`.

| Table | Columns |
| --- | --- |
| `accounts` | account_id, created_at, full_name, email, phone, city, status (`active`, `closed_fraud`, `closed_voluntary`, `pending_review`) |
| `logins` | account_id, ts, device_id, ip_country, vpn (0/1) |
| `transactions` | txn_id, account_id, ts, type (`purchase`, `deposit`, `payout`), amount, destination, status (`settled`, `chargeback`, `pending`) |
| `review_queue` | case_id, account_id, opened_at, trigger, amount_usd |

**Chalk feature query (free).** Online features for any account: `account.age_days`,
`account.email_age_days`, `account.risk_score` (the production model, 0-1), `account.txn_count_30d`,
`account.chargeback_count_180d`, `account.distinct_devices_30d`, `account.distinct_countries_30d`,
`account.vpn_login_ratio_30d`, `account.password_reset_7d`, `account.new_payout_destination_7d`,
`account.device_shared_account_count`, `account.max_prior_payout_usd`.

**Python sandbox (free).** Python 3.13, standard library, no network.

**Deep identity verification ($5 per call).** Document and SSN/name/DOB match, liveness,
synthetic-identity score, phone tenure, recent SIM swap, address history.

**Social network search ($2 per call).** Identities linked to the account across the fraud
consortium (shared phone, device, email pattern, SSN), with their status at other institutions.

Paid checks are billed to the case. Use them when the free evidence leaves the decision open.
"""


def render_instruction(case: dict[str, Any], today: str) -> str:
    return f"""# Fraud review {case["case_id"]}

Today is {today}. A payout on account **{case["account_id"]}** is held for manual review.

| Field | Value |
| --- | --- |
| Case | {case["case_id"]} |
| Account | {case["account_id"]} |
| Opened | {case["opened_at"]} |
| Alert | {case["trigger"]} |
| Amount held | ${case["amount_usd"]:,.2f} |

Investigate the account and decide whether to **approve** the payout (legitimate) or **deny** it
(fraud). Then call `submit_decision` with your decision, your confidence, and an analysis a
reviewer can audit: the evidence you found for and against fraud, how you weighed it, and why the
decision follows.

{DATA_DICTIONARY}"""


def _call(tool: str, args: dict[str, Any]) -> str:
    return f"fraudlab call {tool} --b64 {base64.b64encode(json.dumps(args).encode()).decode()}"


def reference_steps(case: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """What a careful analyst checks for this kind of case, ending in the right decision."""
    account = case["account_id"]
    steps: list[tuple[str, dict[str, Any]]] = [("chalk_query", {"account_id": account})]
    archetype = case["archetype"]
    if archetype in ("ring_member", "legit_household"):
        steps.append(
            (
                "run_sql",
                {
                    "query": f"SELECT l2.account_id, a.status FROM logins l1 JOIN logins l2 ON l1.device_id = l2.device_id AND l2.account_id != l1.account_id JOIN accounts a ON a.account_id = l2.account_id WHERE l1.account_id = '{account}' GROUP BY 1, 2"
                },
            )
        )
    if archetype in ("account_takeover", "legit_traveler"):
        steps.append(
            (
                "run_sql",
                {
                    "query": f"SELECT device_id, ip_country, vpn, MIN(ts), MAX(ts), COUNT(*) FROM logins WHERE account_id = '{account}' GROUP BY 1, 2, 3"
                },
            )
        )
    if archetype in (
        "synthetic_identity",
        "legit_thin_file",
        "account_takeover",
        "legit_traveler",
    ):
        steps.append(("deep_verification", {"account_id": account}))
    if archetype in ("ring_member", "legit_household"):
        steps.append(("social_network_search", {"account_id": account}))
    decision = "deny" if case["label"] == "fraud" else "approve"
    analysis = (
        f"Reference resolution for a {archetype.replace('_', ' ')} case. Evidence for and against fraud was checked "
        + "with the feature store, the warehouse and, where the free evidence left the decision open, the paid "
        + f"checks. The decision is to {decision} the payout of ${case['amount_usd']:,.2f} on {account}."
    )
    steps.append(
        (
            "submit_decision",
            {"decision": decision, "confidence": 0.9, "analysis": analysis},
        )
    )
    return steps


def solve_sh(case: dict[str, Any]) -> str:
    lines = [
        "#!/bin/bash",
        "# A reasonable investigation, for Harbor's oracle agent.",
        "set -euo pipefail",
    ]
    lines += [
        f"{_call(tool, args)} | grep -q '\"ok\": true'"
        for tool, args in reference_steps(case)
    ]
    return "\n".join(lines) + "\n"


def task_toml(case: dict[str, Any]) -> str:
    return f"""\
schema_version = "1.4"
artifacts = []

[task]
name = "chalk/kestrel-{case["case_id"].lower()}"
version = "1.0.0"
description = "Fraud review of a held payout: approve or deny, with an auditable analysis"
authors = []
keywords = ["fraud", "case-analysis"]

[metadata]
case_id = "{case["case_id"]}"

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
FRAUDLAB_CASE = "{case["case_id"]}"

[solution.env]
"""


def build_environment(target: Path) -> None:
    fraudlab = target / "fraudlab"
    shutil.copytree(
        HERE / "fraudlab",
        fraudlab,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copy(HERE / "population.py", fraudlab / "gen" / "population.py")
    (target / "Dockerfile").write_text(DOCKERFILE)


def main() -> int:
    population = generate()
    if TASKS.exists():
        shutil.rmtree(TASKS)
    TASKS.mkdir()
    for case in population["cases"]:
        task = TASKS / case["case_id"].lower()
        build_environment(task / "environment")
        (task / "instruction.md").write_text(
            render_instruction(case, population["today"])
        )
        (task / "task.toml").write_text(task_toml(case))
        (task / "tests").mkdir()
        (task / "tests" / "test.sh").write_text(TEST_SH)
        (task / "tests" / "test.sh").chmod(0o755)
        rubric = {k: case[k] for k in ("case_id", "account_id", "archetype", "label")}
        (task / "tests" / "rubric.json").write_text(json.dumps(rubric, indent=2))
        (task / "solution").mkdir()
        (task / "solution" / "solve.sh").write_text(solve_sh(case))
        (task / "solution" / "solve.sh").chmod(0o755)
    print(f"wrote {len(population['cases'])} tasks to {TASKS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
