#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["chalkcompute", "braintrust"]
# ///
"""Load one support_eval.py run from the larkspur-traces volume into a Braintrust experiment.

    BRAINTRUST_API_KEY=... ./to_braintrust.py <tag> [--project larkspur-support]
    ./to_braintrust.py <tag> --jsonl out.jsonl      # no upload: write the rows locally

Reads ``<tag>/manifest.json`` (the evaluation's rows, with every scorer's value and metadata)
and the trajectory and grade each trial wrote under ``<tag>/<task>/``. Each ticket becomes one
experiment row: the ticket the agent saw as input, the conversation and the actions it took as
output, the policy-correct resolution as expected, and the same scores the Chalk evaluation
recorded. The agent's ATIF trajectory becomes the row's spans: an LLM span per agent turn and
a tool span per tool call, where ``send_message_to_customer`` spans carry the customer's replies.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from typing import Any

import chalkcompute

VOLUME = "larkspur-traces"
SCORES = (
    "policy_compliance",
    "cost_of_service",
    "customer_got_irate",
    "customer_satisfied",
    "csat_survey",
    "agent_claims_accurate",
)


def _seconds(timestamp: str | None) -> float | None:
    if not timestamp:
        return None
    moment = datetime.fromisoformat(timestamp)
    # Older records have naive UTC timestamps; current ones are aware.
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
                    "trajectory": _read_json(volume, f"{trial_dir}/agent/trajectory.json") if trial_dir else None,
                    "grade": _read_json(volume, f"{trial_dir}/verifier/grade.json") if trial_dir else None,
                }
            )  # fmt: skip
    return manifest, tickets


def row_for(ticket: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    row, output = ticket["row"], ticket["output"]
    grade = ticket["grade"] or {}
    scores = {
        name: float(row[f"{name}_value"]) for name in SCORES if row.get(f"{name}_value") is not None
    }
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
        "error": output.get("error"),
        "metadata": {
            "agent_model": output.get("agent_model"),
            "customer_model": output.get("customer_model"),
            "judge_model": manifest.get("judge_model"),
            "reward": output.get("reward"),
            "critical_failure": output.get("critical_failure"),
            "failed_checks": _metadata(row, "policy_compliance").get(
                "failed_checks"
            ),
            "cost_usd": output.get("cost_of_service_usd"),
            "reference_cost_usd": output.get("reference_cost_usd"),
            "cost_breakdown_usd": output.get("cost_breakdown_usd"),
            "customer_frustration": output.get("customer_frustration"),
            "irate_judgement": _metadata(row, "customer_got_irate"),
            "satisfaction_judgement": _metadata(row, "customer_satisfied"),
            "unbacked_claims": _metadata(row, "agent_claims_accurate").get(
                "unbacked_claims"
            ),
            "chalk_evaluation_id": manifest.get("evaluation_id"),
            "chalk_evaluation_run_id": manifest.get("evaluation_run_id"),
            "volume_path": f"{VOLUME}:{output.get('volume_path')}",
        },
        "metrics": {
            k: v
            for k, v in {
                "prompt_tokens": tokens.get("prompt"),
                "completion_tokens": tokens.get("completion"),
                "cached_tokens": tokens.get("cached"),
            }.items()
            if v is not None
        },
        "spans": spans_for(ticket["trajectory"]),
    }


def spans_for(trajectory: dict[str, Any] | None) -> list[dict[str, Any]]:
    """The agent's turns: an LLM span per model turn and a tool span per tool call."""
    spans: list[dict[str, Any]] = []
    steps = (trajectory or {}).get("steps") or []
    times = [_seconds(step.get("timestamp")) for step in steps]
    model = ((trajectory or {}).get("agent") or {}).get("model_name") or "llm"
    for index, step in enumerate(steps):
        if step.get("source") != "agent":
            continue
        previous = next((t for t in reversed(times[:index]) if t), times[index])
        following = next((t for t in times[index + 1 :] if t), times[index])
        metrics = step.get("metrics") or {}
        spans.append(
            {
                "name": step.get("model_name") or model,
                "type": "llm",
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
            spans.append({"name": call.get("function_name") or "tool", "type": "tool",
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
        for span in row["spans"]:
            child = root.start_span(
                name=span["name"], type=span["type"], start_time=span["start"]
            )
            child.log(
                **{
                    k: span[k]
                    for k in ("input", "output", "metrics")
                    if span.get(k) is not None
                }
            )
            child.end(end_time=span["end"])
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
