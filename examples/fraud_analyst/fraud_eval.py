#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["chalkcompute>=2.13", "pyarrow", "pydantic"]
# ///
"""Run Kestrel Pay fraud reviews as a Chalk evaluation, one Harbor trial per case.

    ./build_tasks.py
    ./fraud_eval.py                 # the 12 demo cases (6 fraud, 6 legitimate)
    ./fraud_eval.py --hard          # the 12 hard-tier cases
    ./fraud_eval.py --all           # all 52 cases in the review queue
    ./fraud_eval.py --rescore <run-id>

Each row runs ``harbor run`` with the ``FraudAnalystAgent`` harness in a Chalk sandbox (no network;
SQL, Chalk features, python and the metered paid checks execute there). Models go through Chalk's
AI router as the function's own identity. Each trial's record goes to the ``harbor-traces`` volume
under ``<tag>/<case>/``, and its turns stream into the row's trace.

Scorers (score in [0, 1]; dollars and the judge's notes are in each row's metadata):

* ``kestrel-decision-correct`` -- approve or deny against the case's hidden label.
* ``kestrel-analysis-quality`` -- LLM judge of the submitted analysis, blind to the label: is every
  claim grounded in evidence the analyst actually retrieved, is the reasoning coherent, does it weigh
  counter-evidence, does the decision follow, and was the spend proportionate.
* ``kestrel-investigation-cost`` -- paid-check spend, ``1 / (1 + dollars / 7)``: $0 scores 1, both
  checks once ($7) scores 0.5.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

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
TAG = f"kestrel-{time.strftime('%Y%m%d-%H%M%S')}"
AGENT_MODEL = "anthropic/claude-haiku-4-5"
JUDGE_MODEL = "openai/gpt-5.4"
# Two of each fraud archetype and six legitimate cases across the four legitimate archetypes.
DEMO_CASES = ["K-5001", "K-5002", "K-5008", "K-5009", "K-5015", "K-5016",
              "K-5021", "K-5027", "K-5028", "K-5032", "K-5033", "K-5037"]  # fmt: skip

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
if (HERE / "analyst_agent.py").exists():
    TRIAL_IMAGE = (
        TRIAL_IMAGE.add_local_dir(
            str(REPO / "chalk_harbor"), "/opt/harbor/chalk_harbor"
        )
        .add_local_dir(str(TASKS), "/opt/harbor/tasks")
        .add_local_file(str(HERE / "analyst_agent.py"), "/opt/harbor/analyst_agent.py")
    )
SCORER_IMAGE = Image.debian_slim("3.13").pip_install(["pydantic", "openai"])
# Scorers handle one row at a time by default.
SCORER_SCALE = {"concurrency": 32, "min_replicas": 1, "max_replicas": 2}


def _copy_regular_files(source: Path, destination: Path) -> None:
    import shutil

    for path in source.rglob("*"):
        if path.is_file() and not path.is_symlink():
            target = destination / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)


def _upload_with_retry(staged: Path, volume_path: str) -> None:
    # Rows that finish together all commit to the volume's main ref at once; spread the retries out.
    # Runs after the row has returned, so a final failure can only be logged.
    import random

    error = None
    for attempt in range(8):
        try:
            with chalkcompute.Volume(VOLUME) as volume:
                volume.put_dir(staged, volume_path)
            return
        except Exception as exc:  # noqa: BLE001 - logged; the record is not needed to score
            error = f"{type(exc).__name__}: {exc}"[:1000]
            time.sleep(random.uniform(1, 4) * (attempt + 1))
    print(f"upload of {volume_path} failed: {error}", flush=True)


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
    name="kestrel-fraud-trial",
    image=TRIAL_IMAGE,
    chalk_identity=True,
    # Nothing run-specific in the spec (the run's tag arrives per row), so runs reuse the deployment.
    env={"KESTREL_AGENT_MODEL": AGENT_MODEL},
    # `concurrency` caps calls across all replicas. Each wave of rows (8, 16, 32) lands on one replica
    # as a batch, and every trial's `harbor run` startup is CPU-bound, so replicas are few and large.
    concurrency=64,
    min_replicas=2,
    max_replicas=2,
    cpu="16",
    memory="32Gi",
    call_timeout=1500,
)
def kestrel_fraud_trial(task_name: str, run_tag: str) -> str:
    import os
    import subprocess
    import tempfile
    import threading
    import uuid

    sys.path.insert(0, "/opt/harbor")
    from chalk_harbor.evaluation import evaluation_context, evaluation_env
    from chalk_harbor.tracing import stream_trial_spans

    job = f"{task_name}-{uuid.uuid4().hex[:6]}"
    jobs_dir = Path(tempfile.mkdtemp(prefix="harbor-jobs-"))
    command = [
        "harbor", "run", "-p", "/opt/harbor/tasks", "-i", task_name,
        "-a", "analyst_agent:FraudAnalystAgent", "-m", os.environ["KESTREL_AGENT_MODEL"],
        "-e", "chalk_harbor:ChalkSandboxEnvironment",
        "-o", str(jobs_dir), "--job-name", job, "--yes",
    ]  # fmt: skip
    harbor_env = {
        **os.environ, **evaluation_env(), **_router_env(), "PYTHONPATH": "/opt/harbor",
        # This process streams the trial's spans itself; OTel inside `harbor run` stalls at exit.
        "OTEL_SDK_DISABLED": "true", "OTEL_TRACES_EXPORTER": "none", "HARBOR_TELEMETRY": "0",
    }  # fmt: skip
    instruction = Path("/opt/harbor/tasks", task_name, "instruction.md")
    started = time.time()
    with stream_trial_spans(
        jobs_dir / job,
        instruction=instruction.read_text() if instruction.exists() else None,
    ):
        proc = subprocess.run(
            command,
            env=harbor_env,
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
    record = {"task": task_name, "tag": run_tag, **evaluation_context(), "harbor_exit_code": proc.returncode,
              "wall_seconds": wall}  # fmt: skip
    (staged / "chalk.json").write_text(json.dumps(record, indent=2))
    (staged / "harbor.stderr.txt").write_text(proc.stderr[-200_000:])
    volume_path = f"{run_tag}/{task_name}"
    threading.Thread(
        target=_upload_with_retry,
        args=(staged, volume_path),
        name=f"upload-{task_name}",
    ).start()

    if trial_dir is None:
        return json.dumps(
            {
                **record,
                "volume_path": volume_path,
                "error": (proc.stderr or proc.stdout)[-2000:],
            }
        )
    result = _read_json(trial_dir / "result.json")
    grade = _read_json(trial_dir / "verifier" / "grade.json")
    investigation = _read_json(trial_dir / "agent" / "investigation.json")
    job_result = _read_json(jobs_dir / job / "result.json")
    try:
        record["process_startup_seconds"] = round(
            datetime.fromisoformat(job_result["started_at"]).timestamp() - started, 1
        )
    except (KeyError, TypeError, ValueError):
        pass
    exception = result.get("exception_info") or {}
    return json.dumps(
        {
            **record,
            "case_id": grade.get("case_id"),
            "trial": result.get("trial_name"),
            "volume_path": f"{volume_path}/{trial_dir.name}",
            "agent_model": investigation.get("agent_model"),
            "decision": grade.get("decision"),
            "expected_decision": grade.get("expected_decision"),
            "correct": grade.get("correct"),
            "confidence": grade.get("confidence"),
            "analysis": grade.get("analysis"),
            "cost_usd": grade.get("cost_usd"),
            "paid_calls": grade.get("paid_calls"),
            "tool_calls": [
                {
                    k: c.get(k)
                    for k in ("tool", "args", "ok", "cost_usd", "result_preview")
                }
                for c in investigation.get("tool_calls", [])
            ],
            "tokens": investigation.get("tokens"),
            "exception": exception.get("exception_type"),
            "exception_message": (exception.get("exception_message") or "")[:1000],
        }
    )


@chalkcompute.function(
    name="kestrel-decision-correct", image=SCORER_IMAGE, **SCORER_SCALE
)
def decision_correct(output: str) -> EvaluationScorerResult:
    row = json.loads(output)
    return EvaluationScorerResult(
        score=1.0 if row.get("correct") else 0.0,
        metadata={"decision": row.get("decision"), "expected": row.get("expected_decision"),
                  "confidence": row.get("confidence"), "error": row.get("error") or row.get("exception")},
    )  # fmt: skip


@chalkcompute.function(
    name="kestrel-investigation-cost", image=SCORER_IMAGE, **SCORER_SCALE
)
def investigation_cost(output: str) -> EvaluationScorerResult:
    row = json.loads(output)
    cost = row.get("cost_usd")
    if cost is None:
        return EvaluationScorerResult(
            score=0.0, metadata={"error": "the trial produced no ledger"}
        )
    return EvaluationScorerResult(
        score=round(1.0 / (1.0 + cost / 7.0), 4),
        metadata={"cost_usd": cost, "paid_calls": row.get("paid_calls")},
    )


QUALITY_INSTRUCTIONS = """\
You review a fraud analyst's case write-up for a payments company. The analyst investigated a held
payout with tools and submitted APPROVE or DENY with an analysis. You do not know the right answer
and must not guess it; judge the analysis itself against the evidence the analyst actually
retrieved (the tool calls and their results below).

Rate each criterion from 1 (poor) to 5 (excellent):
- evidence_grounding: every factual claim matches a tool result; no invented, misread or
  overstated evidence. Unsupported inferences count against it.
- coherence: a clear chain from evidence to decision, with no contradictions.
- counter_evidence: weighs the strongest evidence against its own conclusion, rather than ignoring it.
- decision_support: the decision and stated confidence follow from the evidence presented.
- proportionality: investigation depth and paid-check spend ($5 deep verification, $2 network
  search) fit the case: paid checks used when they could change the decision, not by reflex."""


class QualityGrade(BaseModel):
    evidence_grounding: int = Field(description="1-5")
    coherence: int = Field(description="1-5")
    counter_evidence: int = Field(description="1-5")
    decision_support: int = Field(description="1-5")
    proportionality: int = Field(description="1-5")
    issues: list[str] = Field(
        description="Specific problems: an unsupported claim, a misread value, an ignored signal."
    )
    score: float = Field(
        description="Mean of the five ratings, mapped from 1-5 to 0-1."
    )

    @model_validator(mode="after")
    def _score(self) -> QualityGrade:
        ratings = [
            self.evidence_grounding,
            self.coherence,
            self.counter_evidence,
            self.decision_support,
            self.proportionality,
        ]
        self.score = round(
            (sum(min(5, max(1, r)) for r in ratings) / len(ratings) - 1) / 4, 4
        )
        return self


def _quality_prompt(response_model: type, output: str) -> str:
    row = json.loads(output)
    lines = [QUALITY_INSTRUCTIONS, "", "## Tool calls and results", ""]
    for index, call in enumerate(row.get("tool_calls") or [], start=1):
        if call.get("tool") == "submit_decision":
            continue
        status = "" if call.get("ok") else " (FAILED)"
        lines += [f"{index}. {call.get('tool')}{status} ${call.get('cost_usd') or 0:.2f} args={json.dumps(call.get('args'))}",
                  f"   result: {call.get('result_preview')}"]  # fmt: skip
    lines += ["", f"Total paid-check spend: ${row.get('cost_usd') or 0:.2f}", "",
              f"## Submitted decision: {str(row.get('decision')).upper()} (confidence {row.get('confidence')})", "",
              row.get("analysis") or "(no analysis was submitted)"]  # fmt: skip
    return "\n".join(lines)


analysis_quality = cc_scorers.llm_judge(
    QualityGrade, model=JUDGE_MODEL, name="kestrel-analysis-quality", inputs=("output",),
    prompt_fn=_quality_prompt, image=SCORER_IMAGE, **SCORER_SCALE,
)  # fmt: skip

SCORERS = [decision_correct, analysis_quality, investigation_cost]


def _wait(run_id: str, timeout: float) -> chalkcompute.EvaluationRun:
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
        time.sleep(10)


def _save(manifest: dict[str, Any], run: chalkcompute.EvaluationRun) -> Path:
    rows = (
        chalkcompute.DatasetClient().read(run.result_dataset).to_pylist()
        if run.result_dataset is not None
        else []
    )
    manifest = {**manifest, "status": str(run.status)}
    with chalkcompute.Volume(VOLUME) as volume:
        volume.put_file(
            f"{manifest['tag']}/manifest.json",
            json.dumps({**manifest, "rows": rows}, indent=2, default=str).encode(),
        )
    out = HERE / "runs" / f"{manifest['tag']}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({**manifest, "rows": rows}, indent=2, default=str))
    print(json.dumps(manifest, indent=2))
    print(f"rows written to {out}")
    return out


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--all", action="store_true", help="All 52 cases instead of the 12 demo cases."
    )
    parser.add_argument(
        "--hard",
        action="store_true",
        help="The 12 hard-tier cases instead of the 12 demo cases.",
    )
    parser.add_argument(
        "--rescore",
        metavar="RUN_ID",
        help="Score an earlier run's outputs with the current scorers.",
    )
    args = parser.parse_args(argv)
    for scorer in SCORERS:
        scorer.wait_ready(timeout=1200)
    if args.rescore:
        started = time.time()
        run = _wait(
            chalkcompute.EvaluationRun.from_id(args.rescore)
            .rescore(scorers=SCORERS)
            .id,
            timeout=3600,
        )
        _save({"tag": f"{TAG}-rescore", "rescored_run_id": args.rescore, "judge_model": JUDGE_MODEL,
               "evaluation_run_id": run.id, "wall_seconds": round(time.time() - started)}, run)  # fmt: skip
        return 0 if str(run.status).endswith("SUCCEEDED") else 1

    sys.path.insert(0, str(HERE))
    from population import HARD_ARCHETYPES, generate

    cases = [
        c
        for c in generate()["cases"]
        if args.all
        or (
            c["archetype"] in HARD_ARCHETYPES
            if args.hard
            else c["case_id"] in DEMO_CASES
        )
    ]
    client = chalkcompute.VolumeClient.from_env()
    try:
        try:
            client.lookup(VOLUME)
        except chalkcompute.VolumeNotFoundError:
            client.create(VOLUME)
    finally:
        client.close()
    kestrel_fraud_trial.wait_ready(timeout=1200)
    dataset = chalkcompute.DatasetClient().upload(
        TAG,
        {
            "task_name": [c["case_id"].lower() for c in cases],
            "run_tag": [TAG] * len(cases),
            "archetype": [c["archetype"] for c in cases],
            "label": [c["label"] for c in cases],
        },
    )
    evaluation = chalkcompute.EvaluationClient().create(
        TAG, dataset=dataset, task=kestrel_fraud_trial, scorers=SCORERS,
        metadata={"agent_model": AGENT_MODEL, "judge_model": JUDGE_MODEL},
    )  # fmt: skip
    started = time.time()
    run = evaluation.run()
    print(
        f"tag {TAG}: evaluation {evaluation.id} run {run.id}, {len(cases)} cases",
        flush=True,
    )
    run = _wait(run.id, timeout=3600)
    wall = round(time.time() - started)
    print(f"status: {run.status} after {wall}s", flush=True)
    _save({"tag": TAG, "agent_model": AGENT_MODEL, "judge_model": JUDGE_MODEL, "evaluation_id": evaluation.id,
           "evaluation_run_id": run.id, "wall_seconds": wall}, run)  # fmt: skip
    return 0 if str(run.status).endswith("SUCCEEDED") else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
