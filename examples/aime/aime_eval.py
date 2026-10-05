#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["chalkcompute", "pyarrow"]
# ///
"""Run Harbor's aime@1.0 (60 tasks) as a Chalk evaluation, keeping Harbor's native records.

    harbor download aime@1.0 -o /tmp/aime && mv /tmp/aime/aime tasks
    ./aime_eval.py [--limit N]

One dataset row per AIME task. Each row runs one Harbor trial (terminus-2 agent, a cheap
model, a Chalk sandbox as the environment) and then:

* uploads the trial directory exactly as Harbor wrote it -- result.json, the ATIF
  agent/trajectory.json, agent and verifier logs -- to the ``harbor-traces`` volume under
  ``<tag>/<task>/``, next to a ``chalk.json`` naming the evaluation, run and row session;
* streams the trial as spans into the row's session while it runs, one per agent turn and
  command, each tagged with the evaluation and run, so the Chalk trace shows it live.

The model key comes from the environment's OPENAI_API_KEY Chalk secret, injected by name.
At the end the script writes ``<tag>/manifest.json`` (evaluation and run ids, per-task
reward) so the volume alone is enough to rebuild the comparison elsewhere.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import chalkcompute
from chalkcompute import EvaluationScorerResult, Image, Secret

HERE = Path(__file__).resolve().parent
# The image is built from the local checkout; inside the deployed function this module sits
# near the filesystem root, where the repository layout does not exist and is not needed.
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE
TASKS = HERE / "tasks"
VOLUME = "harbor-traces"
AGENT = "terminus-2"
TAG = f"aime-{time.strftime('%Y%m%d-%H%M%S')}"

IMAGE = (
    Image.debian_slim("3.13")
    .pip_install(["harbor", "chalkcompute", "dockerfile-parse", "opentelemetry-api"])
    .add_local_dir(str(REPO / "chalk_harbor"), "/opt/harbor/chalk_harbor")
    .add_local_dir(str(TASKS), "/opt/harbor/tasks")
)


def _copy_regular_files(source: Path, destination: Path) -> None:
    # Volume uploads reject symlinks and special files; Harbor's trial dirs hold neither in
    # practice, but a stray one must not lose the whole record.
    import shutil

    for path in source.rglob("*"):
        if path.is_file() and not path.is_symlink():
            target = destination / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)


def _upload_with_retry(staged: Path, volume_path: str) -> str | None:
    """Upload a row's record, returning the last error if every attempt failed.

    Rows that finish together all commit to the volume's ``main`` ref at once, and the volume
    service gives up on a commit after a few rebases; spreading the retries out lets them land.
    """
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


@chalkcompute.function(
    name="harbor-aime-trial",
    image=IMAGE,
    chalk_identity=True,
    secrets=[Secret.from_chalk_env("OPENAI_API_KEY")],
    env={"HARBOR_TRACE_TAG": TAG, "HARBOR_MODEL": "openai/gpt-5-mini"},
    concurrency=32,
    # Each row runs its own `harbor run` process, whose startup alone is CPU-bound for tens of
    # seconds; 8 rows per 4-CPU replica keeps that from dominating the trial.
    min_replicas=4,
    max_replicas=4,
    cpu="4",
    memory="16Gi",
    call_timeout=900,
)
def harbor_aime_trial(task_name: str) -> str:
    import os
    import subprocess
    import tempfile
    import uuid

    sys.path.insert(0, "/opt/harbor")
    from chalk_harbor.tracing import stream_trial_spans

    job = f"{task_name}-{uuid.uuid4().hex[:6]}"
    jobs_dir = Path(tempfile.mkdtemp(prefix="harbor-jobs-"))
    command = [
        "harbor", "run", "-p", "/opt/harbor/tasks", "-i", task_name,
        "-a", AGENT, "-m", os.environ["HARBOR_MODEL"],
        "-e", "chalk_harbor:ChalkSandboxEnvironment",
        # AIME allows 50 minutes per phase; cap the agent at 4 so the suite stays near 5.
        "--agent-timeout-multiplier", "0.08",
        "-o", str(jobs_dir), "--job-name", job, "--yes",
    ]  # fmt: skip
    instruction = Path("/opt/harbor/tasks", task_name, "instruction.md")
    started = time.time()
    # Each agent turn reaches the row's trace as soon as Harbor records it, tagged with the
    # evaluation and run, so a trial can be watched while the evaluation is still running.
    with stream_trial_spans(
        jobs_dir / job,
        instruction=instruction.read_text() if instruction.exists() else None,
    ):
        proc = subprocess.run(
            command,
            env={**os.environ, "PYTHONPATH": "/opt/harbor"},
            capture_output=True,
            text=True,
            timeout=840,
            check=False,
        )
    wall = round(time.time() - started, 1)
    trials = sorted((jobs_dir / job).glob("*/result.json"))
    context = dict(chalkcompute.get_call_context())
    chalk_record = {
        "task": task_name,
        "tag": os.environ["HARBOR_TRACE_TAG"],
        "evaluation_id": context.get("x-chalk-evaluation-id"),
        "evaluation_run_id": context.get("x-chalk-evaluation-run-id"),
        "session_id": chalkcompute._call_context._session_id_from_headers(context)
        or chalkcompute._call_context._session_id_from_row_metadata(
            chalkcompute.get_row_metadata()
        ),
        "harbor_exit_code": proc.returncode,
        "wall_seconds": wall,
    }

    # The job dir is the native record even when the trial failed to produce a result.
    staged = Path(tempfile.mkdtemp(prefix="harbor-upload-"))
    _copy_regular_files(jobs_dir / job, staged)
    (staged / "chalk.json").write_text(json.dumps(chalk_record, indent=2))
    (staged / "harbor.stdout.txt").write_text(proc.stdout[-200_000:])
    (staged / "harbor.stderr.txt").write_text(proc.stderr[-200_000:])
    # The task itself, so the volume alone holds the problem and its answer key.
    task_dir = Path("/opt/harbor/tasks", task_name)
    for name in ("instruction.md", "tests/test_outputs.py"):
        if (task_dir / name).exists():
            (staged / "task" / name).parent.mkdir(parents=True, exist_ok=True)
            (staged / "task" / name).write_text((task_dir / name).read_text())
    volume_path = f"{os.environ['HARBOR_TRACE_TAG']}/{task_name}"
    upload_error = _upload_with_retry(staged, volume_path)

    if not trials:
        return json.dumps(
            {
                **chalk_record,
                "volume_path": volume_path,
                "upload_error": upload_error,
                "error": (proc.stderr or proc.stdout)[-2000:],
            }
        )
    result = json.loads(trials[0].read_text())
    rewards = (result.get("verifier_result") or {}).get("rewards") or {}
    exception = result.get("exception_info") or {}
    agent_result = result.get("agent_result") or {}
    return json.dumps(
        {
            **chalk_record,
            "trial": result.get("trial_name"),
            "volume_path": f"{volume_path}/{trials[0].parent.name}",
            "upload_error": upload_error,
            "reward": rewards.get("reward"),
            "rewards": rewards,
            "exception": exception.get("exception_type"),
            "exception_message": (exception.get("exception_message") or "")[:1000],
            "input_tokens": agent_result.get("n_input_tokens"),
            "output_tokens": agent_result.get("n_output_tokens"),
            "cost_usd": agent_result.get("cost_usd"),
        }
    )


@chalkcompute.function(name="harbor-aime-reward")
def harbor_aime_reward(output: str) -> EvaluationScorerResult:
    trial = json.loads(output)
    reward = trial.get("reward")
    return EvaluationScorerResult(
        score=float(reward) if reward is not None else 0.0, metadata=trial
    )


def _wait(run_id: str, timeout: float) -> chalkcompute.EvaluationRun:
    # A run only advances when it is read, and finalizing inside that read can outlast one
    # RPC's deadline; retry the read rather than treating a timeout as a failed run.
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


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)
    tasks = sorted(p.name for p in TASKS.iterdir() if (p / "task.toml").exists())
    tasks = tasks[: args.limit] if args.limit else tasks

    client = chalkcompute.VolumeClient.from_env()
    try:
        try:
            client.lookup(VOLUME)
        except chalkcompute.VolumeNotFoundError:
            client.create(VOLUME)
    finally:
        client.close()

    # Surface a slow or broken deploy before the evaluation rather than as a hang at exit.
    for function in (harbor_aime_trial, harbor_aime_reward):
        function.wait_ready(timeout=900)
    dataset = chalkcompute.DatasetClient().upload(TAG, {"task_name": tasks})
    evaluation = chalkcompute.EvaluationClient().create(
        TAG, dataset=dataset, task=harbor_aime_trial, scorers=[harbor_aime_reward]
    )
    started = time.time()
    run = evaluation.run()
    print(
        f"tag {TAG}: evaluation {evaluation.id} run {run.id}, {len(tasks)} tasks",
        flush=True,
    )
    run = _wait(run.id, timeout=3600)
    wall = round(time.time() - started)
    print(f"status: {run.status} after {wall}s", flush=True)

    rows = []
    if run.result_dataset is not None:
        table = chalkcompute.DatasetClient().read(run.result_dataset)
        output_column = next(
            c for c in table.column_names if c.endswith("_output") or c == "output"
        )
        for row in table.to_pylist():
            try:
                rows.append(json.loads(row[output_column]))
            except (TypeError, ValueError):
                rows.append({"task": row.get("task_name"), "raw": row[output_column]})
    manifest = {
        "tag": TAG,
        "dataset": "aime@1.0",
        "agent": AGENT,
        "model": "openai/gpt-5-mini",
        "evaluation_id": evaluation.id,
        "evaluation_run_id": run.id,
        "status": str(run.status),
        "wall_seconds": wall,
        "rows": rows,
    }
    with chalkcompute.Volume(VOLUME) as volume:
        volume.put_file(f"{TAG}/manifest.json", json.dumps(manifest, indent=2).encode())
    print(json.dumps({k: v for k, v in manifest.items() if k != "rows"}, indent=2))
    solved = sum(1 for r in rows if r.get("reward") == 1)
    print(f"solved {solved}/{len(tasks)}")
    return 0 if str(run.status).endswith("SUCCEEDED") else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
