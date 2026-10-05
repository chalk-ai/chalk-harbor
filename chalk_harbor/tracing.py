"""Harbor trials as OpenInference spans, replayed afterwards or streamed while they run.

Harbor emits no telemetry of its own, and the agent runs inside the trial's sandbox, so
nothing it does reaches the tracer of the process that launched the trial. What Harbor
does keep is a record: ``result.json`` (phase timings, rewards, exceptions) and, for agents
that support it, ``agent/trajectory.json`` in ATIF (every model turn and tool call, with
timestamps, token counts, and cost). This module turns that record into spans with the
original timestamps, under whatever span is current. Called from a Chalk evaluation task,
that is the row's session, so the row's trace shows the trial.

Span tree::

    harbor.trial                      AGENT  instruction in, final message out, reward
    ├── harbor.environment_setup      CHAIN
    ├── harbor.agent_setup            CHAIN
    ├── harbor.agent_execution        CHAIN
    │   ├── <model>                   LLM    one per agent turn: message, tokens, cost
    │   └── <tool name>               TOOL   one per tool call: arguments in, observation out
    └── harbor.verifier               CHAIN  rewards

A turn's LLM span runs from the previous step to the turn's own timestamp (the time the
model took to produce it); its tool spans run from that timestamp to the next step's (the
time the tools took before the agent continued). ATIF records no finer timing than that.

``emit_trial_spans`` replays a finished trial. ``stream_trial_spans`` wraps a running one:
Harbor rewrites the trajectory after every step, so each turn's LLM span is exported as soon
as the turn is recorded and its tool spans as soon as the next step starts, while the
enclosing spans end, and therefore arrive, when the trial does. A trace viewer that hangs
spans with a missing parent under a synthetic root can show the agent's progress live.

Inside a Chalk evaluation, every span also carries the evaluation and run it belongs to
(``evaluation_attributes``), so a run's trials can be found while the run is still going.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.trace import Span, Status, StatusCode

_TRACER_NAME = "chalk_harbor"
# Attribute values are capped so one huge observation cannot blow the span size limit.
_MAX_VALUE_CHARS = 16_000
_SETUP_PHASES = ("environment_setup", "agent_setup")
# Call metadata a Chalk evaluation attaches to every task and scorer call, and the span
# attribute each is recorded as.
_EVALUATION_HEADERS = {
    "x-chalk-evaluation-id": "chalk.evaluation.id",
    "x-chalk-evaluation-run-id": "chalk.evaluation.run_id",
}


def evaluation_attributes() -> dict[str, str]:
    """The evaluation and run the current Chalk function call serves, as span attributes.

    Empty outside a Chalk evaluation, or where chalkcompute is not installed.
    """
    try:
        import chalkcompute
    except ImportError:
        return {}
    call_context = chalkcompute.get_call_context()
    return {
        attribute: call_context[header]
        for header, attribute in _EVALUATION_HEADERS.items()
        if call_context.get(header)
    }


def emit_trial_spans(
    trial_dir: Path | str,
    *,
    instruction: str | None = None,
    attributes: Mapping[str, Any] | None = None,
) -> None:
    """Emit spans for the finished Harbor trial recorded in ``trial_dir`` under the current span.

    ``attributes`` are added to every span; they default to ``evaluation_attributes()``.
    """
    trial_dir = Path(trial_dir)
    result = json.loads((trial_dir / "result.json").read_text())
    trajectory = _read_json(trial_dir / "agent" / "trajectory.json")
    emitter = _Emitter(
        otel_context.get_current(),
        evaluation_attributes() if attributes is None else dict(attributes),
    )
    root = emitter.start_root(_ns(result.get("started_at")), instruction)
    emitter.finish(root, result, trajectory, emitted_steps=0, execution=None)


@contextmanager
def stream_trial_spans(
    job_dir: Path | str,
    *,
    instruction: str | None = None,
    attributes: Mapping[str, Any] | None = None,
    poll_seconds: float = 1.0,
) -> Iterator[None]:
    """Stream spans for the single Harbor trial that runs under ``job_dir`` inside the block.

    Wrap the ``harbor run`` that writes into ``job_dir`` (its ``-o``/``--job-name``
    directory). Spans hang off the span current when the block is entered, and
    ``attributes`` default to ``evaluation_attributes()`` read at that moment. Leaving the
    block, even by an exception, reads the trial's final record and ends every span.
    """
    streamer = _TrialStreamer(
        Path(job_dir),
        instruction,
        evaluation_attributes() if attributes is None else dict(attributes),
        poll_seconds,
    )
    streamer.start()
    try:
        yield
    finally:
        streamer.finish()


class _Emitter:
    """Builds the span tree under one parent context, stamping shared attributes on each span."""

    def __init__(
        self, parent: otel_context.Context, attributes: dict[str, Any]
    ) -> None:
        super().__init__()
        self._parent = parent
        self._attributes = attributes
        self._tracer = trace.get_tracer(_TRACER_NAME)

    def span(self, name: str, parent: Span | None, start_time: int | None) -> Span:
        context = self._parent if parent is None else trace.set_span_in_context(parent)
        span = self._tracer.start_span(name, context=context, start_time=start_time)
        _set(span, self._attributes)
        return span

    def start_root(self, start_time: int | None, instruction: str | None) -> Span:
        root = self.span("harbor.trial", None, start_time)
        _set(root, {"openinference.span.kind": "AGENT", "input.value": instruction})
        return root

    def start_execution(self, root: Span, start_time: int | None) -> Span:
        execution = self.span("harbor.agent_execution", root, start_time)
        _set(execution, {"openinference.span.kind": "CHAIN"})
        return execution

    def emit_llm_span(
        self,
        execution: Span,
        trajectory: dict[str, Any],
        times: list[int | None],
        index: int,
    ) -> None:
        step = trajectory["steps"][index]
        agent = trajectory.get("agent") or {}
        previous = next((t for t in reversed(times[:index]) if t), times[index])
        metrics = step.get("metrics") or {}
        model = step.get("model_name") or agent.get("model_name") or "llm"
        llm = self.span(model, execution, previous)
        _set(
            llm,
            {
                "openinference.span.kind": "LLM",
                "llm.model_name": model,
                "llm.provider": model.split("/", 1)[0] if "/" in model else None,
                "llm.token_count.prompt": metrics.get("prompt_tokens"),
                "llm.token_count.completion": metrics.get("completion_tokens"),
                "llm.token_count.prompt_details.cache_read": metrics.get(
                    "cached_tokens"
                ),
                "llm.token_count.total": _sum(
                    metrics.get("prompt_tokens"), metrics.get("completion_tokens")
                ),
                "llm.cost.total": metrics.get("cost_usd"),
                "output.value": step.get("message") or None,
                "harbor.reasoning": step.get("reasoning_content"),
            },
        )
        llm.end(end_time=times[index] or previous)

    def emit_tool_spans(
        self,
        execution: Span,
        step: dict[str, Any],
        start_time: int | None,
        end_time: int | None,
    ) -> None:
        observations = {
            result.get("source_call_id"): result.get("content")
            for result in ((step.get("observation") or {}).get("results") or [])
        }
        for call in step.get("tool_calls") or []:
            tool = self.span(call.get("function_name") or "tool", execution, start_time)
            _set(
                tool,
                {
                    "openinference.span.kind": "TOOL",
                    "tool.name": call.get("function_name"),
                    "tool_call.id": call.get("tool_call_id"),
                    "input.value": json.dumps(call.get("arguments"), default=str),
                    "input.mime_type": "application/json",
                    "output.value": _text(observations.get(call.get("tool_call_id"))),
                },
            )
            tool.end(end_time=end_time)

    def finish(
        self,
        root: Span,
        result: dict[str, Any] | None,
        trajectory: dict[str, Any] | None,
        *,
        emitted_steps: int,
        execution: Span | None,
    ) -> None:
        """Emit everything not yet emitted from the final record, then end every open span.

        Steps before ``emitted_steps`` already have their LLM span, and all but the last of
        them their tool spans. A streamed trial passes its open ``execution`` span; a
        replayed one passes None.
        """
        result = result or {}
        rewards = (result.get("verifier_result") or {}).get("rewards") or {}
        exception = result.get("exception_info")
        _set(
            root,
            {
                "harbor.task": result.get("task_name"),
                "harbor.trial": result.get("trial_name"),
                "harbor.agent": (result.get("agent_info") or {}).get("name"),
                "harbor.reward": rewards.get("reward"),
                "output.value": _final_message(trajectory),
            },
        )
        if exception:
            root.set_status(Status(StatusCode.ERROR, exception.get("exception_type")))
            _set(root, {"harbor.exception": exception.get("exception_message")})

        for phase in _SETUP_PHASES:
            self._emit_phase(root, result.get(phase), phase, {})

        block = result.get("agent_execution") or {}
        end = _ns(block.get("finished_at"))
        if execution is None and (block.get("started_at") or trajectory is not None):
            execution = self.start_execution(root, _ns(block.get("started_at")))
        if execution is not None:
            steps = (trajectory or {}).get("steps") or []
            times = [_ns(step.get("timestamp")) for step in steps]
            for index in range(len(steps)):
                if steps[index].get("source") != "agent":
                    continue
                if index >= emitted_steps:
                    self.emit_llm_span(execution, trajectory, times, index)
                if index + 1 >= emitted_steps:
                    following = next((t for t in times[index + 1 :] if t), end)
                    self.emit_tool_spans(
                        execution, steps[index], times[index], following
                    )
            execution.end(end_time=end)

        self._emit_phase(
            root,
            result.get("verifier"),
            "verifier",
            {f"harbor.reward.{name}": value for name, value in rewards.items()},
        )
        root.end(end_time=_ns(result.get("finished_at")))

    def _emit_phase(
        self,
        root: Span,
        block: dict[str, Any] | None,
        phase: str,
        attributes: Mapping[str, Any],
    ) -> None:
        if not block or not block.get("started_at"):
            return
        span = self.span(f"harbor.{phase}", root, _ns(block["started_at"]))
        _set(span, {"openinference.span.kind": "CHAIN", **attributes})
        span.end(end_time=_ns(block.get("finished_at")))


class _TrialStreamer:
    """Tails a running trial's trajectory and emits each step's spans as it is recorded.

    The trial and agent-execution spans stay open until ``finish``; every LLM and tool span
    under them ends, and so is exported, as soon as its timing is known.
    """

    def __init__(
        self,
        job_dir: Path,
        instruction: str | None,
        attributes: dict[str, Any],
        poll_seconds: float,
    ) -> None:
        super().__init__()
        self._job_dir = job_dir
        self._poll_seconds = poll_seconds
        self._emitter = _Emitter(otel_context.get_current(), attributes)
        self._root = self._emitter.start_root(time.time_ns(), instruction)
        self._execution: Span | None = None
        # Steps whose LLM span is emitted; the tool spans of every step but the last of
        # these are emitted too, since each needs the following step's start as its end.
        self._emitted_steps = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="harbor-trial-spans", daemon=True
        )

    def start(self) -> None:
        self._thread.start()

    def finish(self) -> None:
        self._stop.set()
        self._thread.join()
        trial_dir = self._trial_dir()
        result = _read_json(trial_dir / "result.json") if trial_dir else None
        trajectory = (
            _read_json(trial_dir / "agent" / "trajectory.json") if trial_dir else None
        )
        with self._lock:
            try:
                self._emitter.finish(
                    self._root,
                    result,
                    trajectory,
                    emitted_steps=self._emitted_steps,
                    execution=self._execution,
                )
            except Exception:  # noqa: BLE001 - tracing must never fail the trial it describes
                if self._execution is not None:
                    self._execution.end()
                self._root.end()

    def _run(self) -> None:
        while not self._stop.wait(self._poll_seconds):
            try:
                self._poll()
            except Exception:  # noqa: BLE001, S112 - one bad poll must not end the stream
                continue

    def _trial_dir(self) -> Path | None:
        return next((path.parent for path in self._job_dir.glob("*/config.json")), None)

    def _poll(self) -> None:
        trial_dir = self._trial_dir()
        if trial_dir is None:
            return
        # Read mid-rewrite, the file fails to parse; the next poll gets the whole of it.
        trajectory = _read_json(trial_dir / "agent" / "trajectory.json")
        steps = (trajectory or {}).get("steps") or []
        if not steps:
            return
        times = [_ns(step.get("timestamp")) for step in steps]
        with self._lock:
            if self._execution is None:
                # The first step is the instruction handed to the agent, so it marks the
                # start of agent execution as closely as anything recorded mid-trial.
                self._execution = self._emitter.start_execution(self._root, times[0])
            for index in range(self._emitted_steps, len(steps)):
                if index > 0 and steps[index - 1].get("source") == "agent":
                    self._emitter.emit_tool_spans(
                        self._execution,
                        steps[index - 1],
                        times[index - 1],
                        times[index],
                    )
                if steps[index].get("source") == "agent":
                    self._emitter.emit_llm_span(
                        self._execution, trajectory, times, index
                    )
            self._emitted_steps = len(steps)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _final_message(trajectory: dict[str, Any] | None) -> str | None:
    if trajectory is None:
        return None
    for step in reversed(trajectory.get("steps") or []):
        if step.get("source") == "agent" and step.get("message"):
            return step["message"]
    return None


def _set(span: Span, attributes: Mapping[str, Any]) -> None:
    for key, value in attributes.items():
        if value is None:
            continue
        if isinstance(value, str):
            value = value[:_MAX_VALUE_CHARS]
        elif not isinstance(value, (bool, int, float)):
            value = json.dumps(value, default=str)[:_MAX_VALUE_CHARS]
        span.set_attribute(key, value)


def _text(value: Any) -> str | None:
    if value is None or isinstance(value, str):
        return value
    return json.dumps(value, default=str)


def _sum(*values: int | None) -> int | None:
    present = [v for v in values if v is not None]
    return sum(present) if present else None


def _ns(timestamp: str | None) -> int | None:
    if not timestamp:
        return None
    return int(datetime.fromisoformat(timestamp).timestamp() * 1e9)


__all__ = ["emit_trial_spans", "evaluation_attributes", "stream_trial_spans"]
