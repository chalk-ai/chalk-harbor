import itertools
import json
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from chalk_harbor.tracing import emit_trial_spans, stream_trial_spans

ATTRIBUTES = {"chalk.evaluation.id": "eval-1", "chalk.evaluation.run_id": "run-1"}

# The global provider can only be set once per process; tests share it and clear the exporter.
_EXPORTER = InMemorySpanExporter()
_PROVIDER = TracerProvider()
_PROVIDER.add_span_processor(SimpleSpanProcessor(_EXPORTER))
trace.set_tracer_provider(_PROVIDER)


@pytest.fixture
def exporter() -> InMemorySpanExporter:
    _EXPORTER.clear()
    return _EXPORTER


def _step(index: int, source: str, second: int, tool_calls: int) -> dict[str, Any]:
    calls = [
        {
            "tool_call_id": f"call-{index}-{n}",
            "function_name": "bash_command",
            "arguments": {"keystrokes": f"echo {index}-{n}"},
        }
        for n in range(tool_calls)
    ]
    return {
        "step_id": index,
        "source": source,
        "timestamp": f"2026-10-04T22:00:{second:02d}",
        "message": f"step {index}",
        "metrics": {"prompt_tokens": 100, "completion_tokens": 20, "cost_usd": 0.001},
        "tool_calls": calls,
        "observation": {
            "results": [
                {"source_call_id": call["tool_call_id"], "content": "ok"}
                for call in calls
            ]
        },
    }


TRAJECTORY = {
    "agent": {"name": "terminus-2", "model_name": "openai/gpt-5-mini"},
    "steps": [
        _step(0, "user", 10, 0),
        _step(1, "agent", 20, 2),
        _step(2, "agent", 30, 1),
        _step(3, "agent", 40, 0),
    ],
}
RESULT = {
    "task_name": "aime_1",
    "trial_name": "aime_1__abc",
    "started_at": "2026-10-04T22:00:00",
    "finished_at": "2026-10-04T22:00:50",
    "agent_info": {"name": "terminus-2"},
    "verifier_result": {"rewards": {"reward": 1.0}},
    "environment_setup": {
        "started_at": "2026-10-04T22:00:01",
        "finished_at": "2026-10-04T22:00:04",
    },
    "agent_setup": {
        "started_at": "2026-10-04T22:00:04",
        "finished_at": "2026-10-04T22:00:06",
    },
    "agent_execution": {
        "started_at": "2026-10-04T22:00:10",
        "finished_at": "2026-10-04T22:00:45",
    },
    "verifier": {
        "started_at": "2026-10-04T22:00:45",
        "finished_at": "2026-10-04T22:00:49",
    },
}


def _write_trial(trial_dir: Path, steps: int, finished: bool) -> None:
    (trial_dir / "agent").mkdir(parents=True, exist_ok=True)
    (trial_dir / "config.json").write_text("{}")
    trajectory = dict(TRAJECTORY, steps=TRAJECTORY["steps"][:steps])
    (trial_dir / "agent" / "trajectory.json").write_text(json.dumps(trajectory))
    if finished:
        (trial_dir / "result.json").write_text(json.dumps(RESULT))


def _shape(
    spans: list[ReadableSpan],
) -> list[tuple[str, int | None, int | None, str | None]]:
    names = {span.context.span_id: span.name for span in spans}
    return sorted(
        (
            span.name,
            span.start_time,
            span.end_time,
            names.get(span.parent.span_id) if span.parent else None,
        )
        for span in spans
        # The streamed trial span starts at launch and agent execution at the first step;
        # both are compared separately.
        if span.name not in ("harbor.trial", "harbor.agent_execution")
    )


def test_replay_builds_the_trial_tree(
    tmp_path: Path, exporter: InMemorySpanExporter
) -> None:
    _write_trial(tmp_path, len(TRAJECTORY["steps"]), finished=True)
    emit_trial_spans(tmp_path, instruction="problem", attributes=ATTRIBUTES)

    spans = exporter.get_finished_spans()
    names = sorted(span.name for span in spans)
    assert names.count("openai/gpt-5-mini") == 3
    assert names.count("bash_command") == 3
    assert {"harbor.trial", "harbor.agent_execution", "harbor.verifier"} <= set(names)
    # Each agent step is a turn under agent execution, holding its model call and tools.
    by_id = {span.context.span_id: span for span in spans}
    turns = sorted(
        (span for span in spans if span.name.startswith("turn ")),
        key=lambda span: span.start_time,
    )
    assert [turn.name for turn in turns] == ["turn 1", "turn 2", "turn 3"]
    assert {by_id[turn.parent.span_id].name for turn in turns} == {
        "harbor.agent_execution"
    }
    for span in spans:
        if span.name in ("openai/gpt-5-mini", "bash_command"):
            assert by_id[span.parent.span_id].name.startswith("turn ")
    # Step stamps only: a turn runs from the previous stamp to its own, so turns never
    # overlap, and its spans are marked as step-timed.
    for earlier, later in itertools.pairwise(turns):
        assert earlier.end_time <= later.start_time
    llm = next(span for span in spans if span.name == "openai/gpt-5-mini")
    assert llm.attributes["harbor.timing"] == "step"
    verifier = next(span for span in spans if span.name == "harbor.verifier")
    assert verifier.attributes["harbor.reward.reward"] == 1.0
    assert all(span.attributes["chalk.evaluation.run_id"] == "run-1" for span in spans)


def test_streaming_exports_each_step_while_the_trial_runs(
    tmp_path: Path, exporter: InMemorySpanExporter
) -> None:
    replay_dir = tmp_path / "replay"
    _write_trial(replay_dir, len(TRAJECTORY["steps"]), finished=True)
    emit_trial_spans(replay_dir, instruction="problem", attributes=ATTRIBUTES)
    replayed = list(exporter.get_finished_spans())
    exporter.clear()

    job_dir = tmp_path / "job"
    trial_dir = job_dir / "aime_1__abc"
    exported_after_step: list[list[str]] = []

    def run_trial() -> None:
        for steps in range(1, len(TRAJECTORY["steps"]) + 1):
            _write_trial(trial_dir, steps, finished=False)
            time.sleep(0.3)
            exported_after_step.append(
                sorted(span.name for span in exporter.get_finished_spans())
            )
        _write_trial(trial_dir, len(TRAJECTORY["steps"]), finished=True)

    tracer = trace.get_tracer("test")
    with (
        tracer.start_as_current_span("session") as session,
        stream_trial_spans(
            job_dir, instruction="problem", attributes=ATTRIBUTES, poll_seconds=0.05
        ),
    ):
        trial = threading.Thread(target=run_trial)
        trial.start()
        trial.join()

    # Each turn lands as soon as its step is recorded with its tool results; nothing waits
    # for the trial to end.
    assert exported_after_step[0] == []
    assert sorted(exported_after_step[1]) == [
        "bash_command",
        "bash_command",
        "openai/gpt-5-mini",
        "turn 1",
    ]
    assert exported_after_step[2].count("bash_command") == 3
    assert exported_after_step[3].count("openai/gpt-5-mini") == 3

    streamed = [
        span for span in exporter.get_finished_spans() if span.name != "session"
    ]
    assert _shape(streamed) == _shape(replayed)
    root = next(span for span in streamed if span.name == "harbor.trial")
    replayed_root = next(span for span in replayed if span.name == "harbor.trial")
    assert root.end_time == replayed_root.end_time
    assert dict(root.attributes) == dict(replayed_root.attributes)
    assert root.parent.span_id == session.get_span_context().span_id
    assert all(
        span.attributes["chalk.evaluation.run_id"] == "run-1" for span in streamed
    )


def test_streaming_ends_every_span_when_the_trial_leaves_no_record(
    tmp_path: Path, exporter: InMemorySpanExporter
) -> None:
    with stream_trial_spans(tmp_path / "job", attributes=ATTRIBUTES, poll_seconds=0.05):
        pass

    assert [span.name for span in exporter.get_finished_spans()] == ["harbor.trial"]


def _iso(second: float) -> str:
    return f"2026-10-04T22:00:{second:06.3f}"


def test_recorded_timing_gives_each_call_its_own_interval(
    tmp_path: Path, exporter: InMemorySpanExporter
) -> None:
    # An agent that stamps a step when the model replies, and records exact timing.
    agent_step = {
        "step_id": 1,
        "source": "agent",
        "timestamp": _iso(13),
        "message": "checking",
        "extra": {"llm_started_at": _iso(10), "llm_finished_at": _iso(13)},
        "tool_calls": [
            {
                "tool_call_id": "a",
                "function_name": "run_sql",
                "arguments": {},
                "extra": {"started_at": _iso(13.1), "finished_at": _iso(15)},
            },
            {"tool_call_id": "b", "function_name": "run_python", "arguments": {}},
        ],
        "observation": {
            "results": [
                {"source_call_id": "a", "content": "rows"},
                {
                    "source_call_id": "b",
                    "content": "ok",
                    "extra": {"started_at": _iso(15), "finished_at": _iso(18)},
                },
            ]
        },
    }
    trajectory = {
        "agent": {"model_name": "anthropic/claude-haiku-4-5"},
        "steps": [_step(0, "user", 9, 0), agent_step],
    }
    (tmp_path / "agent").mkdir()
    (tmp_path / "agent" / "trajectory.json").write_text(json.dumps(trajectory))
    (tmp_path / "result.json").write_text(json.dumps(RESULT))
    emit_trial_spans(tmp_path, attributes=ATTRIBUTES)

    spans = {span.name: span for span in exporter.get_finished_spans()}
    second = 1_000_000_000
    base = spans["turn 1"].start_time - 10 * second
    interval = lambda name: (
        round((spans[name].start_time - base) / second, 1),
        round((spans[name].end_time - base) / second, 1),
    )
    assert interval("anthropic/claude-haiku-4-5") == (10, 13)
    assert interval("run_sql") == (13.1, 15)
    assert interval("run_python") == (15, 18)
    assert interval("turn 1") == (10, 18)
    assert "harbor.timing" not in spans["anthropic/claude-haiku-4-5"].attributes


def test_streaming_waits_for_tool_results_written_after_the_step(
    tmp_path: Path, exporter: InMemorySpanExporter
) -> None:
    job_dir = tmp_path / "job"
    trial_dir = job_dir / "trial"
    (trial_dir / "agent").mkdir(parents=True)
    (trial_dir / "config.json").write_text("{}")
    step = _step(1, "agent", 20, 1)
    pending = dict(step, observation=None)
    trajectory_path = trial_dir / "agent" / "trajectory.json"
    seen: list[list[str]] = []

    def run_trial() -> None:
        trajectory_path.write_text(
            json.dumps(dict(TRAJECTORY, steps=[_step(0, "user", 10, 0), pending]))
        )
        time.sleep(0.3)
        seen.append(sorted(span.name for span in exporter.get_finished_spans()))
        trajectory_path.write_text(
            json.dumps(dict(TRAJECTORY, steps=[_step(0, "user", 10, 0), step]))
        )
        time.sleep(0.3)
        seen.append(sorted(span.name for span in exporter.get_finished_spans()))

    with stream_trial_spans(job_dir, attributes=ATTRIBUTES, poll_seconds=0.05):
        trial = threading.Thread(target=run_trial)
        trial.start()
        trial.join()

    # Recorded before its tool ran, the step waits; once rewritten with the result, its turn
    # lands with the tool's output.
    assert seen[0] == []
    assert seen[1] == ["bash_command", "openai/gpt-5-mini", "turn 1"]
    tool = next(s for s in exporter.get_finished_spans() if s.name == "bash_command")
    assert tool.attributes["output.value"] == "ok"
