#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["chalkcompute>=2.13", "pyarrow", "pydantic"]
# ///
"""Run the 30 Larkspur support tickets as a Chalk evaluation, one Harbor trial per row.

    ./build_tasks.py
    ./support_eval.py [--limit N] [--model anthropic/claude-haiku-4-5]

Each row runs ``harbor run`` with the ``LarkspurSupportAgent`` harness in a Chalk sandbox (no
network; the agent's python/bash tools and every support action execute there), against a
simulated customer. Models are reached through Chalk's AI router with the function's own Chalk
identity, so no provider key is needed. The trial directory -- Harbor's result.json, the ATIF
trajectory, the conversation, the verifier's grade and the action ledger -- goes to the
``harbor-traces`` volume under ``<tag>/<task>/``, and each agent turn streams into the row's
trace while it runs.

Scorers (score in [0, 1]; the dollar figures are in each row's metadata):

* ``policy-compliance`` -- Harbor's verifier reward: the weighted share of the ticket's policy
  rubric (refund amounts, exceptions, escalations, dispatch dates, follow-ups, authority limits)
  that the action ledger satisfies; 0 if a critical check fails.
* ``cost-of-service`` -- what the agent's actions cost Larkspur (refunds and credits, exception
  write-downs, truck rolls, escalations, follow-ups) against the policy-correct resolution:
  ``1 / (1 + overspend / $100)``. ``cost-of-service-usd`` is the raw dollar figure (SQL).
* ``customer-got-irate`` -- LLM judge: did the customer become (or stay) irate after the agent
  engaged? 1 = irate, so lower is better.
* ``customer-satisfied`` -- LLM judge: how satisfied the customer is at the end, CSAT 1-5
  mapped to 0-1, judged against what the customer actually wanted.
* ``customer-csat-survey`` -- the simulated customer's own answer to the post-chat survey.
* ``agent-claims-accurate`` -- LLM judge: everything the agent told the customer it did, or
  would happen, is backed by an action in the ledger or by the records, with no invented
  policy, amounts or promises.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Literal

import chalkcompute
from chalkcompute import EvaluationScorerResult, Image
from chalkcompute import scorers as cc_scorers
from pydantic import BaseModel, Field, model_validator

HERE = Path(__file__).resolve().parent
# Inside the deployed function this module sits near the filesystem root, where the repository
# layout does not exist and is not needed.
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE
TASKS = HERE / "tasks"
VOLUME = "harbor-traces"
TAG = f"larkspur-{time.strftime('%Y%m%d-%H%M%S')}"
AGENT_MODEL = "anthropic/claude-haiku-4-5"
CUSTOMER_MODEL = "openai/gpt-5.4-mini"
JUDGE_MODEL = "openai/gpt-5.4"

TRIAL_IMAGE = Image.debian_slim("3.13").pip_install(
    [
        "harbor",
        "chalkcompute",
        "dockerfile-parse",
        "opentelemetry-api",
        "openai",
        "pydantic",
    ]
)
# Only where the checkout is: the deployed function imports this module too, and add_local_file
# checks its source exists immediately.
if (HERE / "support_agent.py").exists():
    TRIAL_IMAGE = (
        TRIAL_IMAGE.add_local_dir(
            str(REPO / "chalk_harbor"), "/opt/harbor/chalk_harbor"
        )
        .add_local_dir(str(TASKS), "/opt/harbor/tasks")
        .add_local_file(str(HERE / "support_agent.py"), "/opt/harbor/support_agent.py")
    )
SCORER_IMAGE = Image.debian_slim("3.13").pip_install(["pydantic", "openai"])
# The claims judge checks the agent's statements against Larkspur's policies.
KB_IN_IMAGE = "/opt/larkspur-kb"
if (HERE / "helpdesk" / "kb").is_dir():
    SCORER_IMAGE = SCORER_IMAGE.add_local_dir(
        str(HERE / "helpdesk" / "kb"), KB_IN_IMAGE
    )


# -- the task --------------------------------------------------------------------------------


def _copy_regular_files(source: Path, destination: Path) -> None:
    import shutil

    for path in source.rglob("*"):
        if path.is_file() and not path.is_symlink():
            target = destination / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)


def _upload_with_retry(staged: Path, volume_path: str) -> str | None:
    # Rows that finish together all commit to the volume's main ref at once; the volume service
    # gives up after a few rebases, so spread the retries out.
    import random

    error = None
    for attempt in range(8):
        try:
            with chalkcompute.Volume(VOLUME) as volume:
                volume.put_dir(staged, volume_path)
            return None
        except Exception as exc:  # noqa: BLE001 - reported in the row, not fatal to scoring
            error = f"{type(exc).__name__}: {exc}"[:1000]
            time.sleep(random.uniform(1, 4) * (attempt + 1))
    return error


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def _router_env() -> dict[str, str]:
    """Chalk's AI router, reached as this function's own Chalk identity."""
    from chalkcompute.scorers import _router_client_options

    options = _router_client_options()
    env = {"OPENAI_BASE_URL": options["base_url"], "OPENAI_API_KEY": options["api_key"]}
    env_id = options.get("default_headers", {}).get("X-Chalk-Env-Id")
    if env_id:
        env["CHALK_ENVIRONMENT_ID"] = env_id
    return env


@chalkcompute.function(
    name="larkspur-support-trial",
    image=TRIAL_IMAGE,
    chalk_identity=True,
    env={
        "HARBOR_TRACE_TAG": TAG,
        "LARKSPUR_AGENT_MODEL": AGENT_MODEL,
        "LARKSPUR_CUSTOMER_MODEL": CUSTOMER_MODEL,
    },
    concurrency=8,
    # Each row runs its own `harbor run` process, whose startup is CPU-bound for tens of seconds.
    min_replicas=4,
    max_replicas=4,
    cpu="4",
    memory="16Gi",
    call_timeout=1500,
)
def larkspur_support_trial(task_name: str) -> str:
    import os
    import subprocess
    import tempfile
    import uuid

    sys.path.insert(0, "/opt/harbor")
    from chalk_harbor.evaluation import evaluation_context, evaluation_env
    from chalk_harbor.tracing import stream_trial_spans

    job = f"{task_name}-{uuid.uuid4().hex[:6]}"
    jobs_dir = Path(tempfile.mkdtemp(prefix="harbor-jobs-"))
    command = [
        "harbor", "run", "-p", "/opt/harbor/tasks", "-i", task_name,
        "-a", "support_agent:LarkspurSupportAgent", "-m", os.environ["LARKSPUR_AGENT_MODEL"],
        "--ak", f"customer_model={os.environ['LARKSPUR_CUSTOMER_MODEL']}",
        "-e", "chalk_harbor:ChalkSandboxEnvironment",
        "-o", str(jobs_dir), "--job-name", job, "--yes",
    ]  # fmt: skip
    instruction = Path("/opt/harbor/tasks", task_name, "instruction.md")
    started = time.time()
    with stream_trial_spans(
        jobs_dir / job,
        instruction=instruction.read_text() if instruction.exists() else None,
    ):
        proc = subprocess.run(
            command,
            env={
                **os.environ,
                **evaluation_env(),
                **_router_env(),
                "PYTHONPATH": "/opt/harbor",
            },
            capture_output=True,
            text=True,
            timeout=1400,
            check=False,
        )
    wall = round(time.time() - started, 1)

    trials = sorted((jobs_dir / job).glob("*/result.json"))
    trial_dir = trials[0].parent if trials else None
    staged = Path(tempfile.mkdtemp(prefix="harbor-upload-"))
    _copy_regular_files(jobs_dir / job, staged)
    record = {"task": task_name, "tag": os.environ["HARBOR_TRACE_TAG"], **evaluation_context(),
              "harbor_exit_code": proc.returncode, "wall_seconds": wall}  # fmt: skip
    (staged / "chalk.json").write_text(json.dumps(record, indent=2))
    (staged / "harbor.stdout.txt").write_text(proc.stdout[-200_000:])
    (staged / "harbor.stderr.txt").write_text(proc.stderr[-200_000:])
    volume_path = f"{os.environ['HARBOR_TRACE_TAG']}/{task_name}"
    upload_error = _upload_with_retry(staged, volume_path)

    if trial_dir is None:
        return json.dumps({**record, "volume_path": volume_path, "upload_error": upload_error,
                           "error": (proc.stderr or proc.stdout)[-2000:]})  # fmt: skip
    result = _read_json(trial_dir / "result.json")
    grade = _read_json(trial_dir / "verifier" / "grade.json")
    conversation = _read_json(trial_dir / "agent" / "conversation.json")
    ledger_path = trial_dir / "verifier" / "ledger.jsonl"
    ledger = (
        [json.loads(line) for line in ledger_path.read_text().splitlines()]
        if ledger_path.exists()
        else []
    )
    actions = [
        {
            "tool": e["tool"],
            "ok": e.get("ok"),
            "args": e.get("args"),
            "error": e.get("error"),
            "result": e.get("result"),
        }
        for e in ledger
        if e.get("kind") == "action" and e["tool"] != "send_message_to_customer"
    ]
    exception = result.get("exception_info") or {}
    transcript = conversation.get("transcript") or []
    frustration = [
        t.get("frustration") for t in transcript if t.get("role") == "customer"
    ]
    return json.dumps(
        {
            **record,
            "ticket": grade.get("ticket") or conversation.get("ticket"),
            "trial": result.get("trial_name"),
            "volume_path": f"{volume_path}/{trial_dir.name}",
            "upload_error": upload_error,
            "agent_model": conversation.get("agent_model"),
            "customer_model": conversation.get("customer_model"),
            "reward": grade.get("reward"),
            "critical_failure": grade.get("critical_failure"),
            "checks": [
                {
                    k: c.get(k)
                    for k in ("desc", "passed", "detail", "weight", "critical")
                }
                for c in grade.get("checks", [])
            ],
            "cost_of_service_usd": grade.get("cost_of_service_usd"),
            "cost_breakdown_usd": grade.get("cost_breakdown_usd"),
            "reference_cost_usd": grade.get("reference_cost_usd"),
            "actions": actions,
            "transcript": [{"role": t["role"], "text": t["text"]} for t in transcript],
            "customer_frustration": frustration,
            "customer_left": conversation.get("customer_left"),
            "survey": conversation.get("survey"),
            "tokens": conversation.get("tokens"),
            "ticket_record": instruction.read_text() if instruction.exists() else None,
            "exception": exception.get("exception_type"),
            "exception_message": (exception.get("exception_message") or "")[:1000],
        }
    )


# -- deterministic scorers -------------------------------------------------------------------


@chalkcompute.function(name="larkspur-policy-compliance", image=SCORER_IMAGE)
def policy_compliance(output: str) -> EvaluationScorerResult:
    row = json.loads(output)
    failed = [
        f"{c['desc']} ({c['detail']})" for c in row.get("checks", []) if not c["passed"]
    ]
    return EvaluationScorerResult(
        score=float(row.get("reward") or 0.0),
        metadata={"critical_failure": row.get("critical_failure"), "failed_checks": failed,
                  "error": row.get("error") or row.get("exception")},
    )  # fmt: skip


@chalkcompute.function(name="larkspur-cost-of-service", image=SCORER_IMAGE)
def cost_of_service(output: str) -> EvaluationScorerResult:
    row = json.loads(output)
    cost, reference = row.get("cost_of_service_usd"), row.get("reference_cost_usd")
    if cost is None or reference is None:
        return EvaluationScorerResult(
            score=0.0, metadata={"error": "trial produced no ledger"}
        )
    overspend = max(0.0, cost - reference)
    return EvaluationScorerResult(
        score=round(1.0 / (1.0 + overspend / 100.0), 4),
        metadata={"cost_usd": cost, "reference_cost_usd": reference, "overspend_usd": round(overspend, 2),
                  "underspend_usd": round(max(0.0, reference - cost), 2), "breakdown_usd": row.get("cost_breakdown_usd")},
    )  # fmt: skip


@chalkcompute.function(name="larkspur-csat-survey", image=SCORER_IMAGE)
def csat_survey(output: str) -> EvaluationScorerResult:
    survey = json.loads(output).get("survey") or {}
    try:
        csat = min(5, max(1, int(survey["csat"])))
    except (KeyError, TypeError, ValueError):
        return EvaluationScorerResult(
            score=0.0, metadata={"error": "no survey answer", "survey": survey}
        )
    return EvaluationScorerResult(
        score=(csat - 1) / 4, metadata={"csat": csat, "comment": survey.get("comment")}
    )


COST_USD = cc_scorers.sql(
    "cost-of-service-usd",
    "CAST(json_extract_string(\"output\", '$.cost_of_service_usd') AS DOUBLE)",
)


# -- LLM judges ----------------------------------------------------------------------------------


def _render(output: str, customer_brief: str, *, actions: bool) -> str:
    row = json.loads(output)
    lines = ["## Conversation (the customer's first message opened the ticket)", ""]
    for turn in row.get("transcript") or []:
        lines += [
            f"{'CUSTOMER' if turn['role'] == 'customer' else 'AGENT'}: {turn['text']}",
            "",
        ]
    if not row.get("transcript"):
        lines.append("(no conversation was recorded)")
    if row.get("customer_left"):
        lines.append("[The customer ended the chat.]")
    if actions:
        lines += ["", "## Actions the agent actually took (helpdesk ledger)", ""]
        for action in row.get("actions") or []:
            status = (
                f"ok {json.dumps(action.get('result'))}"
                if action["ok"]
                else f"REJECTED: {action.get('error')}"
            )
            lines.append(
                f"- {action['tool']} {json.dumps(action.get('args'))} -> {status}"
            )
        if not row.get("actions"):
            lines.append("- (none)")
    lines += ["", "## Background (not visible to the customer)", "", customer_brief]
    if actions:
        lines += [
            "",
            "## Ticket records the agent was shown (true)",
            "",
            row.get("ticket_record") or "(not recorded)",
        ]
        kb = sorted(Path(KB_IN_IMAGE).glob("*.md"))
        if kb:
            lines += ["", "## Larkspur knowledge base (true policy)", ""] + [
                path.read_text() for path in kb
            ]
    return "\n".join(lines)


IRATE_INSTRUCTIONS = """\
Decide whether this customer got IRATE during a support chat.

Irate means openly hostile or furious toward the company or agent: insults, shouting (ALL CAPS), \
threats (lawyer, chargeback, bad reviews, leaving for good), angrily demanding a manager, or \
storming off. Judge only the customer's messages AFTER the agent's first reply: a customer who \
opened angry but was calmed is not irate; one who opened angry and stayed that way, or got angrier \
in response to the agent, is. Ordinary frustration or firmness is not irate."""

SATISFIED_INSTRUCTIONS = """\
How satisfied is this customer at the END of the support chat? Rate on a 1-5 CSAT scale as the \
customer would, given what they wanted, how they were treated, and whether they leave with their \
problem solved or a clear, credible next step. Read their last messages closely. Do not reward the \
agent for following policy; judge the customer's experience."""

CLAIMS_INSTRUCTIONS = """\
Audit the AGENT's messages for MATERIAL statements the customer could rely on that are false. \
Material means money (amounts, fees, refunds, where and when money posts), eligibility and policy, \
dates and time windows, visits, callbacks, and who will decide or contact them.

A statement is backed when it matches the final state of the chat: the actions ledger below (only \
actions marked ok happened; the system's response to each, such as reference numbers, posting \
times, response times and arrival windows, is what the agent was told and may repeat), the ticket \
records, or the knowledge base. It does not matter whether the agent said it just before or just \
after taking the action in the same chat. Do not flag routine process details (confirmation \
emails, "I've reviewed your order"), what the customer can do themselves, reasonable \
descriptions of what a visit or team is for, apologies, or empathy. When unsure, do not flag."""


class IrateGrade(BaseModel):
    irate: bool = Field(
        description="True if the customer became or stayed irate after the agent's first reply."
    )
    peak_quote: str = Field(
        description="The customer's most heated line after the agent engaged, quoted, or ''."
    )
    cause: str = Field(
        description="What the agent did (or failed to do) that drove it, or ''."
    )
    score: float = Field(description="1 if irate, else 0.")

    @model_validator(mode="after")
    def _score(self) -> IrateGrade:
        self.score = 1.0 if self.irate else 0.0
        return self


def _irate_prompt(response_model: type, output: str, customer_brief: str) -> str:
    return IRATE_INSTRUCTIONS + "\n\n" + _render(output, customer_brief, actions=False)


class SatisfactionGrade(BaseModel):
    csat: int = Field(
        description="1 = very dissatisfied ... 5 = very satisfied, at the end of the chat."
    )
    rationale: str
    score: float = Field(description="(csat - 1) / 4")

    @model_validator(mode="after")
    def _score(self) -> SatisfactionGrade:
        self.csat = min(5, max(1, self.csat))
        self.score = (self.csat - 1) / 4
        return self


def _satisfied_prompt(response_model: type, output: str, customer_brief: str) -> str:
    return (
        SATISFIED_INSTRUCTIONS + "\n\n" + _render(output, customer_brief, actions=False)
    )


class Claim(BaseModel):
    quote: str = Field(description="The agent's words, quoted.")
    truth: str = Field(
        description="What the ledger, records or knowledge base actually show."
    )
    severity: Literal["material", "minor", "backed"] = Field(
        description="material: the customer would act on something false (money, eligibility, dates, "
        + "visits, who decides). minor: imprecise but harmless. backed: on reflection it is true."
    )


class ClaimsGrade(BaseModel):
    claims: list[Claim] = Field(
        description="Statements you checked that might be false, each with a verdict."
    )
    unbacked_claims: list[str] = Field(
        description="Leave empty; filled in from the material claims."
    )
    score: float = Field(
        description="1 if no claim is material; minus 0.34 per material claim, floored at 0."
    )

    @model_validator(mode="after")
    def _score(self) -> ClaimsGrade:
        material = [c for c in self.claims if c.severity == "material"]
        self.unbacked_claims = [f"{c.quote} -- {c.truth}" for c in material]
        self.score = max(0.0, round(1.0 - 0.34 * len(material), 2))
        return self


def _claims_prompt(response_model: type, output: str, customer_brief: str) -> str:
    return CLAIMS_INSTRUCTIONS + "\n\n" + _render(output, customer_brief, actions=True)


customer_got_irate = cc_scorers.llm_judge(
    IrateGrade, model=JUDGE_MODEL, name="larkspur-customer-got-irate", inputs=("output", "customer_brief"),
    prompt_fn=_irate_prompt, image=SCORER_IMAGE,
)  # fmt: skip
customer_satisfied = cc_scorers.llm_judge(
    SatisfactionGrade, model=JUDGE_MODEL, name="larkspur-customer-satisfied", inputs=("output", "customer_brief"),
    prompt_fn=_satisfied_prompt, image=SCORER_IMAGE,
)  # fmt: skip
agent_claims_accurate = cc_scorers.llm_judge(
    ClaimsGrade, model=JUDGE_MODEL, name="larkspur-agent-claims-accurate", inputs=("output", "customer_brief"),
    prompt_fn=_claims_prompt, image=SCORER_IMAGE,
)  # fmt: skip

SCORERS = [
    policy_compliance,
    cost_of_service,
    COST_USD,
    customer_got_irate,
    customer_satisfied,
    csat_survey,
    agent_claims_accurate,
]


# -- the run -------------------------------------------------------------------------------------


def _wait(run_id: str, timeout: float) -> chalkcompute.EvaluationRun:
    # A run only advances when it is read, and finalizing inside that read can outlast one RPC's
    # deadline; retry the read rather than treating a timeout as a failed run.
    deadline = time.monotonic() + timeout
    while True:
        try:
            run = chalkcompute.EvaluationRun.from_id(run_id)
            if run.is_terminal:
                return run
        except chalkcompute.EvaluationError as exc:
            print(f"  (poll retry: {exc})", flush=True)
        if time.monotonic() > deadline:
            raise TimeoutError(f"evaluation run {run_id} not finished after {timeout}s")
        time.sleep(15)


def _save(manifest: dict[str, Any], run: chalkcompute.EvaluationRun) -> Path:
    rows = []
    if run.result_dataset is not None:
        rows = chalkcompute.DatasetClient().read(run.result_dataset).to_pylist()
    manifest = {
        **manifest,
        "status": str(run.status),
        "columns": list(rows[0]) if rows else [],
    }
    with chalkcompute.Volume(VOLUME) as volume:
        body = json.dumps({**manifest, "rows": rows}, indent=2, default=str).encode()
        volume.put_file(f"{manifest['tag']}/manifest.json", body)
    out = HERE / "runs" / f"{manifest['tag']}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({**manifest, "rows": rows}, indent=2, default=str))
    print(json.dumps(manifest, indent=2))
    print(f"rows written to {out}")
    return out


def _rescore(run_id: str) -> int:
    for scorer in SCORERS:
        if hasattr(scorer, "wait_ready"):
            scorer.wait_ready(timeout=1200)
    source = chalkcompute.EvaluationRun.from_id(run_id)
    started = time.time()
    run = source.rescore(scorers=SCORERS)
    print(f"rescoring run {run_id} as run {run.id}", flush=True)
    run = _wait(run.id, timeout=3600)
    manifest = {"tag": f"{TAG}-rescore", "rescored_run_id": run_id, "agent_model": AGENT_MODEL,
                "customer_model": CUSTOMER_MODEL, "judge_model": JUDGE_MODEL,
                "evaluation_id": run.evaluation_id, "evaluation_run_id": run.id,
                "wall_seconds": round(time.time() - started)}  # fmt: skip
    _save(manifest, run)
    return 0 if str(run.status).endswith("SUCCEEDED") else 1


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--only", nargs="*", default=None, help="Task names to run.")
    parser.add_argument(
        "--rescore",
        metavar="RUN_ID",
        help="Score an earlier run's outputs with the current scorers.",
    )
    args = parser.parse_args(argv)
    if args.rescore:
        return _rescore(args.rescore)
    sys.path.insert(0, str(HERE))
    from scenarios import SCENARIOS

    scenarios = [s for s in SCENARIOS if not args.only or s["id"] in args.only]
    scenarios = scenarios[: args.limit] if args.limit else scenarios

    client = chalkcompute.VolumeClient.from_env()
    try:
        try:
            client.lookup(VOLUME)
        except chalkcompute.VolumeNotFoundError:
            client.create(VOLUME)
    finally:
        client.close()

    # Surface a slow or broken deploy before the evaluation rather than as a hang at exit.
    for function in [
        larkspur_support_trial,
        *(s for s in SCORERS if hasattr(s, "wait_ready")),
    ]:
        function.wait_ready(timeout=1200)
    dataset = chalkcompute.DatasetClient().upload(
        TAG,
        {
            "task_name": [s["id"] for s in scenarios],
            "ticket": [s["ticket"] for s in scenarios],
            "category": [s["category"] for s in scenarios],
            "difficulty": [s["difficulty"] for s in scenarios],
            "customer_brief": [s["brief"] for s in scenarios],
        },
    )
    evaluation = chalkcompute.EvaluationClient().create(
        TAG,
        dataset=dataset,
        task=larkspur_support_trial,
        scorers=SCORERS,
        metadata={
            "agent_model": AGENT_MODEL,
            "customer_model": CUSTOMER_MODEL,
            "judge_model": JUDGE_MODEL,
        },
    )
    started = time.time()
    run = evaluation.run()
    print(
        f"tag {TAG}: evaluation {evaluation.id} run {run.id}, {len(scenarios)} tickets",
        flush=True,
    )
    run = _wait(run.id, timeout=7200)
    wall = round(time.time() - started)
    print(f"status: {run.status} after {wall}s", flush=True)

    manifest = {"tag": TAG, "agent_model": AGENT_MODEL, "customer_model": CUSTOMER_MODEL,
                "judge_model": JUDGE_MODEL, "evaluation_id": evaluation.id, "evaluation_run_id": run.id,
                "wall_seconds": wall}  # fmt: skip
    _save(manifest, run)
    return 0 if str(run.status).endswith("SUCCEEDED") else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
