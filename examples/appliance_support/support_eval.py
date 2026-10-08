#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["chalkcompute>=2.13", "pyarrow"]
# ///
"""Run the 60 Larkspur support tickets as a Chalk evaluation, one sandboxed trial per row.

    ./support_eval.py [--limit N] [--only <id> ...]
    ./support_eval.py --rescore <run-id>

Each row starts a Chalk sandbox with no network, holding the helpdesk backend and the ticket's
sealed record, and runs the support agent against a simulated customer: the agent loop runs in
the trial function, and the agent's python/bash tools and every support action execute in the
sandbox. When the agent is done, the sandbox's ledger is graded against the ticket's rubric.
Models are reached through Chalk's AI router with the function's own Chalk identity, so no
provider key is needed.

Scorers (score in [0, 1] unless noted; the details are in each row's metadata):

* ``policy_compliance`` -- the weighted share of the ticket's policy rubric (refund amounts,
  exceptions, escalations, dispatch dates, follow-ups, authority limits) that the action ledger
  satisfies; 0 if a critical check fails.
* ``cost_of_service`` -- what the agent's actions cost Larkspur (refunds and credits, exception
  write-downs, truck rolls, escalations, follow-ups) against the policy-correct resolution:
  ``1 / (1 + overspend / $100)``. ``cost_of_service_usd`` is the raw dollar figure, as a SQL
  expression.
* ``customer_got_irate`` -- built-in ``jev`` judge (a cheap model, escalating to JUDGE_MODEL): did
  the customer become (or stay) irate after the agent engaged? 1 = irate, so lower is better.
* ``customer_satisfied`` -- LLM judge: how satisfied the customer is at the end, CSAT 1-5
  mapped to 0-1, judged against what the customer actually wanted.
* ``csat_survey`` -- the simulated customer's own answer to the post-chat survey.
* ``agent_claims_accurate`` -- LLM judge: everything the agent told the customer it did, or
  would happen, is backed by an action in the ledger or by the records, with no invented
  policy, amounts or promises.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import chalkcompute
from chalkcompute import EvaluationScorerResult, Image, NetworkPolicy, Sandbox, scorers
from openai import OpenAI

import tickets
from scenarios import SCENARIOS
from support_agent import HELPDESK, SANDBOX_USER, SupportAgent, grade_ticket

if TYPE_CHECKING:
    from pydantic import BaseModel

HERE = Path(__file__).resolve().parent
VOLUME = "larkspur-traces"
TAG = f"larkspur-{time.strftime('%Y%m%d-%H%M%S')}"
AGENT_MODEL = "anthropic/claude-haiku-4-5"
CUSTOMER_MODEL = "openai/gpt-5.4-mini"
JUDGE_MODEL = "openai/gpt-5.4"
# Models are reached through Chalk's AI router, an OpenAI-compatible endpoint that takes the
# function's own Chalk identity (chalk_identity=True) as its key, so no provider key is needed.
CHALK_API_SERVER = "https://api.staging.chalk.ai"


def _run_metadata() -> dict[str, Any]:
    """The evaluation run's metadata; a post-training run names the policy as `agent_model`."""
    context = {k.lower(): v for k, v in chalkcompute.get_call_context().items()}
    run_id = context.get("x-chalk-evaluation-run-id")
    return dict(chalkcompute.EvaluationRun.from_id(run_id).metadata or {}) if run_id else {}


def _volume_path(metadata: dict[str, Any], run_tag: str, task_name: str) -> str:
    # A post-training's rollouts share one dataset, and with it one run_tag; keep each
    # iteration's samples apart. Run metadata numbers arrive as doubles.
    if not metadata.get("post_training_id"):
        return f"{run_tag}/{task_name}"
    whole = lambda v: str(int(v)) if isinstance(v, float) and v.is_integer() else str(v)  # noqa: E731
    return "/".join([run_tag, f"posttrain-{metadata['post_training_id']}",
                     f"iter-{whole(metadata.get('iteration', 0))}",
                     f"sample-{whole(metadata.get('sample', 0))}", task_name])  # fmt: skip


# -- the task --------------------------------------------------------------------------------


@chalkcompute.function(image=Image.debian_slim("3.13").pip_install(["openai"]))
def larkspur_support_trial(task_name: str, run_tag: str, instruction: str, scenario: str, rubric: str) -> str:
    sandbox = Sandbox(
        image=Image.debian_slim("3.13")
        .run_commands(
            f"useradd --create-home --shell /bin/bash {SANDBOX_USER}",
            "install -d -m 700 /opt/helpdesk/scenarios /var/lib/helpdesk",
        )
        .add_local_dir(str(HELPDESK / "bin"), "/opt/helpdesk/bin")
        .add_local_dir(str(HELPDESK / "lib"), "/opt/helpdesk/lib", exclude=["__pycache__"])
        .add_local_dir(str(HELPDESK / "kb"), "/opt/helpdesk/kb"),
        env={"HELPDESK_TICKET": json.loads(scenario)["ticket"]},
        network_policy=NetworkPolicy(),  # no routes: no egress at all
    ).run()
    agent = SupportAgent(
        sandbox,
        model=_run_metadata().get("agent_model") or AGENT_MODEL,
        customer_model=_run_metadata().get("customer_model") or CUSTOMER_MODEL,
        client=OpenAI(base_url="https://api.staging.chalk.ai/v1/router"),
    )
    # The customer's persona and the rubric's facts; root-only, out of the agent's reach.
    sandbox.fs.write_bytes(f"/opt/helpdesk/scenarios/{json.loads(scenario)['ticket']}.json", scenario.encode())
    agent.run(instruction)
    sandbox.terminate()
    chat = agent.result()

    # The post-trainer reads each sample's trajectory from the volume.
    grade, ledger = grade_ticket(sandbox, json.loads(rubric))
    volume_path = _volume_path(_run_metadata(), run_tag, task_name)
    with chalkcompute.Volume(VOLUME) as volume:
        volume.put_file(f"{volume_path}/agent/trajectory.json", json.dumps(chat["trajectory"]))
        volume.put_file(f"{volume_path}/verifier/grade.json", json.dumps(grade))

    transcript = chat["transcript"]
    return json.dumps(
        {
            "task": task_name,
            "volume_path": volume_path,
            "ticket": grade.get("ticket"),
            "agent_model": chat["agent_model"],
            "customer_model": chat["customer_model"],
            "reward": grade.get("reward"),
            "critical_failure": grade.get("critical_failure"),
            "checks": [
                {k: c.get(k) for k in ("desc", "passed", "detail", "weight", "critical")}
                for c in grade.get("checks", [])
            ],
            "cost_of_service_usd": grade.get("cost_of_service_usd"),
            "cost_breakdown_usd": grade.get("cost_breakdown_usd"),
            "reference_cost_usd": grade.get("reference_cost_usd"),
            "actions": [
                {k: e.get(k) for k in ("tool", "ok", "args", "error", "result")}
                for e in ledger
                if e.get("kind") == "action" and e["tool"] != "send_message_to_customer"
            ],
            "transcript": [{"role": t["role"], "text": t["text"]} for t in transcript],
            "customer_frustration": [t.get("frustration") for t in transcript if t["role"] == "customer"],
            "customer_left": chat["customer_left"],
            "survey": chat["survey"],
            "tokens": chat["tokens"],
            "ticket_record": instruction,
            "error": type(error).__name__ if error else None,
            "error_message": str(error)[:1000] if error else None,
        }
    )  # fmt: skip


# -- deterministic scorers -------------------------------------------------------------------


@chalkcompute.function
def policy_compliance(output: str) -> EvaluationScorerResult:
    row = json.loads(output)
    failed = [f"{c['desc']} ({c['detail']})" for c in row.get("checks", []) if not c["passed"]]
    return EvaluationScorerResult(
        score=float(row.get("reward") or 0.0),
        metadata={"critical_failure": row.get("critical_failure"), "failed_checks": failed,
                  "error": row.get("error")},
    )  # fmt: skip


@chalkcompute.function
def cost_of_service(output: str) -> EvaluationScorerResult:
    row = json.loads(output)
    cost, reference = row.get("cost_of_service_usd"), row.get("reference_cost_usd")
    if cost is None or reference is None:
        return EvaluationScorerResult(score=0.0, metadata={"error": "trial produced no ledger"})
    overspend = max(0.0, cost - reference)
    return EvaluationScorerResult(
        score=round(1.0 / (1.0 + overspend / 100.0), 4),
        metadata={"cost_usd": cost, "reference_cost_usd": reference, "overspend_usd": round(overspend, 2),
                  "underspend_usd": round(max(0.0, reference - cost), 2), "breakdown_usd": row.get("cost_breakdown_usd")},
    )  # fmt: skip


@chalkcompute.function
def csat_survey(output: str) -> EvaluationScorerResult:
    survey = json.loads(output).get("survey") or {}
    try:
        csat = min(5, max(1, int(survey["csat"])))
    except (KeyError, TypeError, ValueError):
        return EvaluationScorerResult(score=0.0, metadata={"error": "no survey answer", "survey": survey})
    return EvaluationScorerResult(score=(csat - 1) / 4, metadata={"csat": csat, "comment": survey.get("comment")})


# -- LLM judges ----------------------------------------------------------------------------------


def _render(output: str, customer_brief: str, *, actions: bool) -> str:
    row = json.loads(output)
    lines = ["## Conversation (the customer's first message opened the ticket)", ""]
    for turn in row.get("transcript") or []:
        lines += [f"{'CUSTOMER' if turn['role'] == 'customer' else 'AGENT'}: {turn['text']}", ""]
    if not row.get("transcript"):
        lines.append("(no conversation was recorded)")
    if row.get("customer_left"):
        lines.append("[The customer ended the chat.]")
    if actions:
        lines += ["", "## Actions the agent actually took (helpdesk ledger)", ""]
        for action in row.get("actions") or []:
            status = f"ok {json.dumps(action.get('result'))}" if action["ok"] else f"REJECTED: {action.get('error')}"
            lines.append(f"- {action['tool']} {json.dumps(action.get('args'))} -> {status}")
        if not row.get("actions"):
            lines.append("- (none)")
    lines += ["", "## Background (not visible to the customer)", "", customer_brief]
    if actions:
        lines += ["", "## Ticket records the agent was shown (true)", "",
                  row.get("ticket_record") or "(not recorded)"]  # fmt: skip
        kb = sorted((HERE / "helpdesk" / "kb").glob("*.md"))
        if kb:
            lines += ["", "## Larkspur knowledge base (true policy)", ""] + [path.read_text() for path in kb]
    return "\n".join(lines)


def _judge(grade_model: type[BaseModel], prompt: str) -> EvaluationScorerResult:
    """Ask the judge model for a ``grade_model``; its ``score`` is the scorer's score."""
    client = OpenAI(
        base_url=f"{CHALK_API_SERVER}/v1/router",
        api_key=Path(os.environ["CHALK_WEB_IDENTITY_TOKEN_FILE"]).read_text().strip(),
        default_headers={"X-Chalk-Env-Id": os.environ["CHALK_ENVIRONMENT_ID"]},
    )
    message = (
        client.chat.completions.parse(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format=grade_model,
        )
        .choices[0]
        .message
    )
    if message.parsed is None:
        raise RuntimeError(f"the judge returned no grade ({message.refusal or 'empty reply'})")
    grade = message.parsed.model_dump()
    return EvaluationScorerResult(score=grade.pop("score"), metadata={**grade, "judge_model": JUDGE_MODEL})


@chalkcompute.function(image=Image.debian_slim("3.13").pip_install(["openai", "pydantic"]), chalk_identity=True)
def customer_satisfied(output: str, customer_brief: str) -> EvaluationScorerResult:
    from grades import SatisfactionGrade

    return _judge(
        SatisfactionGrade,
        """\
How satisfied is this customer at the END of the support chat? Rate on a 1-5 CSAT scale as the \
customer would, given what they wanted, how they were treated, and whether they leave with their \
problem solved or a clear, credible next step. Read their last messages closely. Do not reward the \
agent for following policy; judge the customer's experience.

"""
        + _render(output, customer_brief, actions=False),
    )


@chalkcompute.function(image=Image.debian_slim("3.13").pip_install(["openai", "pydantic"]), chalk_identity=True)
def agent_claims_accurate(output: str, customer_brief: str) -> EvaluationScorerResult:
    from grades import ClaimsGrade

    return _judge(
        ClaimsGrade,
        """\
Audit the AGENT's messages for MATERIAL statements the customer could rely on that are false. \
Material means money (amounts, fees, refunds, where and when money posts), eligibility and policy, \
dates and time windows, visits, callbacks, and who will decide or contact them.

A statement is backed when it matches the final state of the chat: the actions ledger below (only \
actions marked ok happened; the system's response to each, such as reference numbers, posting \
times, response times and arrival windows, is what the agent was told and may repeat), the ticket \
records, or the knowledge base. It does not matter whether the agent said it just before or just \
after taking the action in the same chat. Do not flag routine process details (confirmation \
emails, "I've reviewed your order"), what the customer can do themselves, reasonable \
descriptions of what a visit or team is for, apologies, or empathy. When unsure, do not flag.

"""
        + _render(output, customer_brief, actions=True),
    )


SCORERS = [
    policy_compliance,
    cost_of_service,
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
    manifest = {**manifest, "status": str(run.status), "columns": list(rows[0]) if rows else []}
    with chalkcompute.Volume(VOLUME) as volume:
        body = json.dumps({**manifest, "rows": rows}, indent=2, default=str).encode()
        volume.put_file(f"{manifest['tag']}/manifest.json", body)
    out = HERE / "runs" / f"{manifest['tag']}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({**manifest, "rows": rows}, indent=2, default=str))
    print(json.dumps(manifest, indent=2))
    print(f"rows written to {out}")
    return out


def main(argv: list[str]) -> int:
    scenarios = SCENARIOS
    parser = argparse.ArgumentParser()
    parser.add_argument("--rescore", metavar="RUN_ID", help="Score an earlier run's outputs with the current scorers.")
    args = parser.parse_args(argv)
    if args.rescore:
        chalkcompute.EvaluationRun.from_id(run_id).rescore(scorers=SCORERS)

    # Everything a trial needs travels in its row, so the trial function ships no ticket data.
    dataset = chalkcompute.DatasetClient().upload(
        TAG,
        {
            "task_name": [s["id"] for s in scenarios],
            "run_tag": [TAG] * len(scenarios),
            "instruction": [tickets.render_instruction(s) for s in scenarios],
            "scenario": [json.dumps(tickets.sealed(s)) for s in scenarios],
            "rubric": [json.dumps(tickets.rubric(s)) for s in scenarios],
            "ticket": [s["ticket"] for s in scenarios],
            "category": [s["category"] for s in scenarios],
            "difficulty": [s["difficulty"] for s in scenarios],
            "customer_brief": [s["brief"] for s in scenarios],
        },
    )
    evaluation = chalkcompute.Evaluation.create(
        name=f"larkspur-{time.strftime('%Y%m%d-%H%M%S')}",
        dataset=dataset,
        task=larkspur_support_trial,
        scorers=[
            csat_survey,
            scorers.sql(
                "cost_of_service_usd",
                "CAST(json_extract_string(output, '$.cost_of_service_usd') AS DOUBLE)",
            ),
            scorers.jev(
                name="customer_got_irate",
                labels={"irate": 1.0, "not_irate": 0.0},
                rubric="""Decide whether this customer got IRATE during a support chat.
                    Irate means openly hostile or furious toward the company or agent: insults, shouting (ALL CAPS), \
                    threats (lawyer, chargeback, bad reviews, leaving for good), angrily demanding a manager, or \
                    storming off. Judge only the customer's messages AFTER the agent's first reply: a customer who \
                    opened angry but was calmed is not irate; one who opened angry and stayed that way, or got angrier \
                    in response to the agent, is. Ordinary frustration or firmness is not irate.""",
                model="openai/gpt-5.4-mini",
                fallback_model="openai/gpt-5.4",
                inputs=("output", "customer_brief"),
            ),
            customer_satisfied,
            agent_claims_accurate,
            policy_compliance,
            cost_of_service,
        ],
        metadata={"agent_model": AGENT_MODEL, "customer_model": CUSTOMER_MODEL, "judge_model": JUDGE_MODEL},
    ).run()

    #
    # To take this to production:
    #
    # chalkcompute.RemoteFunction.from_name("larkspur_support_trial").remote(
    #     task_name="...",
    #     run_tag="...",
    #     instruction="...",
    #     scenario="...",
    #     rubric="...",
    # )


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
