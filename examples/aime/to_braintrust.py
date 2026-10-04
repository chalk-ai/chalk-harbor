#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["chalkcompute", "braintrust"]
# ///
"""Load one aime_eval.py run from the harbor-traces volume into a Braintrust experiment.

    ./to_braintrust.py <tag> [--project harbor-aime] [--jsonl out.jsonl]

Reads ``<tag>/manifest.json`` and every ``<tag>/<task>/<trial>/`` Harbor wrote, and logs one
experiment row per task with the same reward the Chalk evaluation scored. The trial record
becomes the row's span tree, built from the same files and timestamps as the Chalk trace:
Harbor's phases, an LLM span per agent turn, and a tool span per command.

With ``BRAINTRUST_API_KEY`` set it uploads. Without it, or with ``--jsonl``, it writes the rows
to a file instead, so the mapping can be checked before anything leaves the machine.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

import chalkcompute

VOLUME = "harbor-traces"
PHASES = ("environment_setup", "agent_setup", "agent_execution", "verifier")


def _seconds(timestamp: str | None) -> float | None:
    return datetime.fromisoformat(timestamp).timestamp() if timestamp else None


def _read_json(volume: chalkcompute.Volume, path: str) -> Any | None:
    try:
        return json.loads(volume.read_file(path))
    except Exception:  # noqa: BLE001 - a missing optional file is normal
        return None


def _read_text(volume: chalkcompute.Volume, path: str) -> str | None:
    try:
        return volume.read_file(path).decode(errors="replace")
    except Exception:  # noqa: BLE001
        return None


def load_trials(tag: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return the run manifest and one record per task, read back from the volume."""
    with chalkcompute.Volume(VOLUME) as volume:
        manifest = _read_json(volume, f"{tag}/manifest.json") or {"tag": tag}
        files = [info.path for info in volume.listdir(f"{tag}/")]
        trials: list[dict[str, Any]] = []
        for chalk_path in sorted(p for p in files if p.endswith("/chalk.json")):
            task_dir = str(PurePosixPath(chalk_path).parent)
            chalk = _read_json(volume, chalk_path) or {}
            results = sorted(
                p for p in files
                if p.startswith(task_dir + "/") and p.endswith("/result.json")
                and p.count("/") == task_dir.count("/") + 2
            )  # fmt: skip
            trial_dir = str(PurePosixPath(results[0]).parent) if results else None
            trials.append(
                {
                    "task": chalk.get("task") or PurePosixPath(task_dir).name,
                    "chalk": chalk,
                    "volume_path": trial_dir or task_dir,
                    "result": _read_json(volume, results[0]) if results else None,
                    "trajectory": (
                        _read_json(volume, f"{trial_dir}/agent/trajectory.json")
                        if trial_dir
                        else None
                    ),
                    "instruction": _read_text(volume, f"{task_dir}/task/instruction.md")
                    or _local_task_file(chalk.get("task"), "instruction.md"),
                    "answer_key": _read_text(
                        volume, f"{task_dir}/task/tests/test_outputs.py"
                    )
                    or _local_task_file(chalk.get("task"), "tests/test_outputs.py"),
                    "verifier_stdout": (
                        _read_text(volume, f"{trial_dir}/verifier/test-stdout.txt")
                        if trial_dir
                        else None
                    ),
                }
            )
    return manifest, trials


def _local_task_file(task: str | None, name: str) -> str | None:
    # Runs before the eval staged task files on the volume fall back to the local task copy.
    path = Path(__file__).resolve().parent / "tasks" / (task or "") / name
    return path.read_text() if task and path.exists() else None


def _expected(answer_key: str | None) -> str | None:
    match = re.search(r'EXPECTED_ANSWER\s*=\s*"(\d+)"', answer_key or "")
    return match.group(1) if match else None


def _final_message(trajectory: dict[str, Any] | None) -> str | None:
    for step in reversed((trajectory or {}).get("steps") or []):
        if step.get("source") == "agent" and step.get("message"):
            return step["message"]
    return None


_WRITE_ANSWER = re.compile(r"(\d{1,3})(?:\\n)?[\"']?\s*>+\s*/app/answer\.txt")
_CAT_ANSWER = re.compile(r"cat /app/answer\.txt\s*\n\s*(\d{1,3})\s*$", re.MULTILINE)


def _answer(trajectory: dict[str, Any] | None) -> str | None:
    """What the agent left in /app/answer.txt: the last value it wrote, else the last one it read back."""
    written = read_back = None
    for step in (trajectory or {}).get("steps") or []:
        for call in step.get("tool_calls") or []:
            keystrokes = str((call.get("arguments") or {}).get("keystrokes") or "")
            for match in _WRITE_ANSWER.finditer(keystrokes):
                written = match.group(1)
        for result in (step.get("observation") or {}).get("results") or []:
            for match in _CAT_ANSWER.finditer(str(result.get("content") or "")):
                read_back = match.group(1)
    return written or read_back


def row_for(trial: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    result = trial["result"] or {}
    rewards = (result.get("verifier_result") or {}).get("rewards") or {}
    agent_result = result.get("agent_result") or {}
    exception = result.get("exception_info") or {}
    trajectory = trial["trajectory"]
    reward = rewards.get("reward")
    return {
        "id": f"{manifest.get('tag')}:{trial['task']}",
        "input": {"task": trial["task"], "problem": trial["instruction"]},
        "output": {
            "answer": _answer(trajectory),
            "final_message": _final_message(trajectory),
        },
        "expected": {"answer": _expected(trial["answer_key"])},
        "scores": {"reward": float(reward) if reward is not None else 0.0},
        "error": exception.get("exception_type"),
        "metadata": {
            "dataset": manifest.get("dataset", "aime@1.0"),
            "agent": manifest.get("agent"),
            "model": manifest.get("model"),
            "harbor_trial": result.get("trial_name"),
            "chalk_evaluation_id": manifest.get("evaluation_id"),
            "chalk_evaluation_run_id": manifest.get("evaluation_run_id"),
            "chalk_session_id": trial["chalk"].get("session_id"),
            "volume_path": f"{VOLUME}:{trial['volume_path']}",
            "exception_message": exception.get("exception_message"),
        },
        "metrics": {
            k: v
            for k, v in {
                "prompt_tokens": agent_result.get("n_input_tokens"),
                "completion_tokens": agent_result.get("n_output_tokens"),
                "cached_tokens": agent_result.get("n_cache_tokens"),
                "cost_usd": agent_result.get("cost_usd"),
            }.items()
            if v is not None
        },
        "spans": spans_for(result, trajectory),
    }


def spans_for(
    result: dict[str, Any], trajectory: dict[str, Any] | None
) -> list[dict[str, Any]]:
    """Harbor's phases and the agent's turns as a flat list with parent names."""
    spans: list[dict[str, Any]] = []
    for phase in PHASES:
        block = result.get(phase) or {}
        if block.get("started_at"):
            spans.append(
                {
                    "name": phase,
                    "type": "task",
                    "parent": None,
                    "start": _seconds(block["started_at"]),
                    "end": _seconds(block.get("finished_at")),
                }
            )
    steps = (trajectory or {}).get("steps") or []
    times = [_seconds(step.get("timestamp")) for step in steps]
    execution = result.get("agent_execution") or {}
    end = _seconds(execution.get("finished_at"))
    model = ((trajectory or {}).get("agent") or {}).get("model_name") or "llm"
    for index, step in enumerate(steps):
        if step.get("source") != "agent":
            continue
        previous = next((t for t in reversed(times[:index]) if t), times[index])
        following = next((t for t in times[index + 1 :] if t), end)
        metrics = step.get("metrics") or {}
        spans.append(
            {
                "name": step.get("model_name") or model,
                "type": "llm",
                "parent": "agent_execution",
                "start": previous,
                "end": times[index] or previous,
                "output": step.get("message"),
                "metrics": {
                    k: v
                    for k, v in {
                        "prompt_tokens": metrics.get("prompt_tokens"),
                        "completion_tokens": metrics.get("completion_tokens"),
                        "cost_usd": metrics.get("cost_usd"),
                    }.items()
                    if v is not None
                },
            }
        )
        observations = {
            r.get("source_call_id"): r.get("content")
            for r in ((step.get("observation") or {}).get("results") or [])
        }
        for call in step.get("tool_calls") or []:
            spans.append(
                {
                    "name": call.get("function_name") or "tool",
                    "type": "tool",
                    "parent": "agent_execution",
                    "start": times[index],
                    "end": following,
                    "input": call.get("arguments"),
                    "output": observations.get(call.get("tool_call_id")),
                }
            )
    return spans


def upload(rows: list[dict[str, Any]], manifest: dict[str, Any], project: str) -> str:
    import braintrust

    experiment = braintrust.init(
        project=project,
        experiment=manifest.get("tag"),
        metadata={k: v for k, v in manifest.items() if k != "rows"},
    )
    for row in rows:
        start = min((s["start"] for s in row["spans"] if s.get("start")), default=None)
        end = max((s["end"] for s in row["spans"] if s.get("end")), default=None)
        root = experiment.start_span(
            name=row["input"]["task"], type="eval", start_time=start
        )
        root.log(
            id=row["id"],
            input=row["input"],
            output=row["output"],
            expected=row["expected"],
            scores=row["scores"],
            metadata=row["metadata"],
            metrics=row["metrics"],
            **({"error": row["error"]} if row["error"] else {}),
        )
        phases: dict[str, Any] = {}
        for span in row["spans"]:
            parent = phases.get(span["parent"], root) if span["parent"] else root
            child = parent.start_span(
                name=span["name"], type=span["type"], start_time=span["start"]
            )
            child.log(
                **{
                    k: span[k]
                    for k in ("input", "output", "metrics")
                    if span.get(k) is not None
                }
            )
            if span["parent"] is None:
                phases[span["name"]] = child
            else:
                child.end(end_time=span["end"])
        for phase in phases.values():
            phase.end(
                end_time=next(
                    s["end"] for s in row["spans"] if phases.get(s["name"]) is phase
                )
            )
        root.end(end_time=end)
    summary = experiment.summarize()
    return summary.experiment_url or ""


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("tag")
    parser.add_argument("--project", default="harbor-aime")
    parser.add_argument("--jsonl")
    args = parser.parse_args(argv)

    manifest, trials = load_trials(args.tag)
    rows = [row_for(trial, manifest) for trial in trials]
    solved = sum(1 for r in rows if r["scores"]["reward"] == 1.0)
    print(f"{args.tag}: {len(rows)} tasks, {solved} solved", flush=True)
    if args.jsonl or not os.environ.get("BRAINTRUST_API_KEY"):
        path = args.jsonl or f"{args.tag}.braintrust.jsonl"
        with open(path, "w") as out:
            for row in rows:
                out.write(json.dumps(row) + "\n")
        print(f"wrote {path} (set BRAINTRUST_API_KEY to upload)")
        return 0
    print("experiment:", upload(rows, manifest, args.project))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
