import json
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from opentelemetry import baggage, trace
from opentelemetry import context as otel_context
from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from chalk_harbor.tracing import emit_trial_spans, stream_trial_spans

ATTRIBUTES = {"chalk.evaluation.id": "eval-1", "chalk.evaluation.run_id": "run-1"}

# The global provider can only be set once per process; tests share it and clear the exporter.
_EXPORTER = InMemorySpanExporter()
_PROVIDER = TracerProvider()
_PROVIDER.add_span_processor(SimpleSpanProcessor(_EXPORTER))


class _BaggageAtStart(SpanProcessor):
    """Records the ``session.id`` baggage current where each span starts, as chalkcompute reads it."""

    def __init__(self) -> None:
        super().__init__()
        self.sessions: dict[int, object] = {}

    def on_start(
        self, span: ReadableSpan, parent_context: otel_context.Context | None = None
    ) -> None:
        self.sessions[span.context.span_id] = baggage.get_baggage("session.id")


_BAGGAGE = _BaggageAtStart()
_PROVIDER.add_span_processor(_BAGGAGE)
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

    # Each agent turn's LLM span lands when the turn is recorded, and its tool spans when the
    # next step starts; nothing waits for the trial to end.
    assert exported_after_step[0] == []
    assert exported_after_step[1] == ["openai/gpt-5-mini"]
    assert exported_after_step[2].count("bash_command") == 2
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


def test_streamed_spans_start_in_the_callers_context(
    tmp_path: Path, exporter: InMemorySpanExporter
) -> None:
    # chalkcompute stamps a span with the row's session from the baggage current where the span
    # starts; spans the poller emits on its own thread must see the caller's.
    job_dir = tmp_path / "job"
    token = otel_context.attach(baggage.set_baggage("session.id", "run-1:row-7"))
    try:
        with stream_trial_spans(
            job_dir, instruction="problem", attributes=ATTRIBUTES, poll_seconds=0.05
        ):
            _write_trial(
                job_dir / "aime_1__abc", len(TRAJECTORY["steps"]), finished=False
            )
            time.sleep(0.3)
            streamed_before_end = {
                span.context.span_id for span in exporter.get_finished_spans()
            }
            _write_trial(
                job_dir / "aime_1__abc", len(TRAJECTORY["steps"]), finished=True
            )
    finally:
        otel_context.detach(token)

    assert streamed_before_end, "the poller emitted nothing while the trial ran"
    spans = exporter.get_finished_spans()
    assert {span.name for span in spans} >= {
        "harbor.trial",
        "openai/gpt-5-mini",
        "bash_command",
    }
    assert {_BAGGAGE.sessions[span.context.span_id] for span in spans} == {
        "run-1:row-7"
    }
