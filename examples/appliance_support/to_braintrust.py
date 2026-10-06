#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["chalkcompute", "braintrust"]
# ///
"""Load one support_eval.py run from the harbor-traces volume into a Braintrust experiment.

    BRAINTRUST_API_KEY=... ./to_braintrust.py <tag> [--project larkspur-support]
    ./to_braintrust.py <tag> --jsonl out.jsonl      # no upload: write the rows locally

Reads ``<tag>/manifest.json`` (the evaluation's rows, with every scorer's value and metadata)
and the trial each row's Harbor run wrote under ``<tag>/<task>/``. Each ticket becomes one
experiment row: the ticket the agent saw as input, the conversation and the actions it took as
output, the policy-correct resolution as expected, and the same scores the Chalk evaluation
recorded. The trial becomes the row's span tree, from the same files and timestamps as the
Chalk trace: Harbor's phases, an LLM span per agent turn and a tool span per tool call, where
``send_message_to_customer`` spans carry the simulated customer's replies.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from typing import Any

import chalkcompute

VOLUME = "harbor-traces"
PHASES = ("environment_setup", "agent_setup", "agent_execution", "verifier")
# Chalk scorer -> Braintrust score name.
SCORES = {
    "larkspur-policy-compliance": "policy_compliance",
    "larkspur-cost-of-service": "cost_of_service",
    "larkspur-customer-got-irate": "customer_got_irate",
    "larkspur-customer-satisfied": "customer_satisfied",
    "larkspur-csat-survey": "csat_survey",
    "larkspur-agent-claims-accurate": "agent_claims_accurate",
}


def _seconds(timestamp: str | None) -> float | None:
    if not timestamp:
        return None
    moment = datetime.fromisoformat(timestamp)
    # Harbor writes naive UTC timestamps; the agent writes aware ones.
    return (moment if moment.tzinfo else moment.replace(tzinfo=UTC)).timestamp()


def _read_json(volume: chalkcompute.Volume, path: str) -> Any | None:
    try:
        return json.loads(volume.read_file(path))
    except Exception:  # noqa: BLE001 - a missing optional file is normal
        return None


def _metadata(row: dict[str, Any], scorer: str) -> dict[str, Any]:
    value = row.get(f"{scorer}_metadata")
    return json.loads(value) if isinstance(value, str) else value or {}


def load(tag: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """The run manifest, and per ticket its evaluation row plus the trial files."""
    with chalkcompute.Volume(VOLUME) as volume:
        manifest = _read_json(volume, f"{tag}/manifest.json")
        if not manifest:
            raise SystemExit(f"{VOLUME}:{tag}/manifest.json not found")
        tickets = []
        for row in manifest["rows"]:
            output = json.loads(row["output"]) if row.get("output") else {}
            trial_dir = output.get("volume_path")
            tickets.append(
                {
                    "row": row,
                    "output": output,
                    "result": _read_json(volume, f"{trial_dir}/result.json") if trial_dir else None,
                    "trajectory": _read_json(volume, f"{trial_dir}/agent/trajectory.json") if trial_dir else None,
                    "grade": _read_json(volume, f"{trial_dir}/verifier/grade.json") if trial_dir else None,
                }
            )  # fmt: skip
    return manifest, tickets


def row_for(ticket: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    row, output, result = ticket["row"], ticket["output"], ticket["result"] or {}
    grade = ticket["grade"] or {}
    scores = {
        name: float(row[f"{scorer}_value"])
        for scorer, name in SCORES.items()
        if row.get(f"{scorer}_value") is not None
    }
    exception = result.get("exception_info") or {}
    tokens = output.get("tokens") or {}
    return {
        "id": f"{manifest['tag']}:{row['task_name']}",
        "input": {
            "task": row["task_name"],
            "ticket": row.get("ticket"),
            "category": row.get("category"),
            "difficulty": row.get("difficulty"),
            "ticket_record": output.get("ticket_record"),
        },
        "output": {
            "transcript": output.get("transcript"),
            "actions": output.get("actions"),
            "survey": output.get("survey"),
            "customer_left": output.get("customer_left"),
        },
        "expected": {
            "resolution": row.get("customer_brief"),
            "policy_checks": [c.get("desc") for c in grade.get("checks", [])],
        },
        "scores": scores,
        "error": exception.get("exception_type") or output.get("error"),
        "metadata": {
            "agent_model": output.get("agent_model"),
            "customer_model": output.get("customer_model"),
            "judge_model": manifest.get("judge_model"),
            "harbor_reward": output.get("reward"),
            "critical_failure": output.get("critical_failure"),
            "failed_checks": _metadata(row, "larkspur-policy-compliance").get(
                "failed_checks"
            ),
            "cost_usd": output.get("cost_of_service_usd"),
            "reference_cost_usd": output.get("reference_cost_usd"),
            "cost_breakdown_usd": output.get("cost_breakdown_usd"),
            "customer_frustration": output.get("customer_frustration"),
            "irate_judgement": _metadata(row, "larkspur-customer-got-irate"),
            "satisfaction_judgement": _metadata(row, "larkspur-customer-satisfied"),
            "unbacked_claims": _metadata(row, "larkspur-agent-claims-accurate").get(
                "unbacked_claims"
            ),
            "harbor_trial": result.get("trial_name"),
            "chalk_evaluation_id": manifest.get("evaluation_id"),
            "chalk_evaluation_run_id": manifest.get("evaluation_run_id"),
            "chalk_session_id": output.get("session_id"),
            "volume_path": f"{VOLUME}:{output.get('volume_path')}",
        },
        "metrics": {
            k: v
            for k, v in {
                "prompt_tokens": tokens.get("prompt"),
                "completion_tokens": tokens.get("completion"),
                "cached_tokens": tokens.get("cached"),
                "trial_seconds": output.get("wall_seconds"),
            }.items()
            if v is not None
        },
        "spans": spans_for(result, ticket["trajectory"]),
    }


def spans_for(
    result: dict[str, Any], trajectory: dict[str, Any] | None
) -> list[dict[str, Any]]:
    """Harbor's phases and the agent's turns as a flat list with parent names."""
    spans: list[dict[str, Any]] = []
    for phase in PHASES:
        block = result.get(phase) or {}
        if block.get("started_at"):
            spans.append({"name": phase, "type": "task", "parent": None,
                          "start": _seconds(block["started_at"]), "end": _seconds(block.get("finished_at"))})  # fmt: skip
    steps = (trajectory or {}).get("steps") or []
    times = [_seconds(step.get("timestamp")) for step in steps]
    end = _seconds((result.get("agent_execution") or {}).get("finished_at"))
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
                "output": {"message": step.get("message"),
                           "tool_calls": [c.get("function_name") for c in step.get("tool_calls") or []]},
                "metrics": {k: v for k, v in {"prompt_tokens": metrics.get("prompt_tokens"),
                                               "completion_tokens": metrics.get("completion_tokens")}.items() if v is not None},
            }
        )  # fmt: skip
        observations = {
            r.get("source_call_id"): r.get("content")
            for r in ((step.get("observation") or {}).get("results") or [])
        }
        for call in step.get("tool_calls") or []:
            spans.append({"name": call.get("function_name") or "tool", "type": "tool", "parent": "agent_execution",
                          "start": times[index], "end": following, "input": call.get("arguments"),
                          "output": observations.get(call.get("tool_call_id"))})  # fmt: skip
    return spans


def upload(rows: list[dict[str, Any]], manifest: dict[str, Any], project: str) -> str:
    import braintrust

    experiment = braintrust.init(
        project=project,
        experiment=manifest["tag"],
        metadata={k: v for k, v in manifest.items() if k not in ("rows", "columns")},
    )
    for row in rows:
        start = min((s["start"] for s in row["spans"] if s.get("start")), default=None)
        end = max((s["end"] for s in row["spans"] if s.get("end")), default=None)
        root = experiment.start_span(
            name=row["input"]["task"], type="eval", start_time=start
        )
        root.log(input=row["input"], output=row["output"], expected=row["expected"],
                 scores=row["scores"], metadata=row["metadata"], metrics=row["metrics"],
                 **({"error": row["error"]} if row["error"] else {}))  # fmt: skip
        phases: dict[str, Any] = {}
        phase_ends: dict[str, float | None] = {}
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
                phase_ends[span["name"]] = span["end"]
            else:
                child.end(end_time=span["end"])
        for name, phase in phases.items():
            phase.end(end_time=phase_ends[name])
        root.end(end_time=end)
    return experiment.summarize().experiment_url or ""


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("tag")
    parser.add_argument("--project", default="larkspur-support")
    parser.add_argument("--jsonl")
    args = parser.parse_args(argv)

    manifest, tickets = load(args.tag)
    rows = [row_for(ticket, manifest) for ticket in tickets]
    perfect = sum(1 for r in rows if r["scores"].get("policy_compliance") == 1.0)
    print(
        f"{args.tag}: {len(rows)} tickets, {perfect} with a perfect policy score",
        flush=True,
    )
    if args.jsonl or not os.environ.get("BRAINTRUST_API_KEY"):
        path = args.jsonl or f"{args.tag}.braintrust.jsonl"
        with open(path, "w") as out:
            out.writelines(json.dumps(row) + "\n" for row in rows)
        print(f"wrote {path} (set BRAINTRUST_API_KEY to upload)")
        return 0
    print("experiment:", upload(rows, manifest, args.project))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
