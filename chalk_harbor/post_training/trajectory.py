"""Rebuild the chat an agent had with its model from a Harbor ATIF trajectory.

A function-calling agent such as the Larkspur harness keeps an OpenAI-style message list and
records each model call as an ATIF step. The mapping back, one step at a time:

* ``agent.extra.system_prompt`` -> a leading ``system`` message (ATIF has no system step for
  a prompt the harness supplies; ``source: "system"`` steps also become ``system`` messages).
* ``source: "user"`` step -> ``user`` message with the step's text (the task instruction, or a
  harness nudge).
* ``source: "agent"`` step -> ``assistant`` message: the step's text as ``content`` and each
  ATIF tool call as an OpenAI ``tool_calls`` entry ``{id, type: "function", function: {name,
  arguments}}``. ATIF stores ``arguments`` parsed; they stay a dict, so the chat template
  re-serializes them with ``tojson``, which matches how vLLM's tool parser re-emits them.
  Then one ``tool`` message per tool call, in call order, with the observation result whose
  ``source_call_id`` is that call's id (empty when the step recorded none). Results that
  answer no call follow as ``user`` messages.
* ``agent.tool_definitions`` -> the ``tools`` list for the chat template.

Assistant messages carry ``"train": True`` when the policy generated them: agent steps that
made a model call (``llm_call_count`` is not 0) and are not copied context. Chat templates
ignore the extra key; ``masking`` reads it.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

TRAIN_KEY = "train"


@dataclass(frozen=True)
class AgentChat:
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None


class TrajectoryError(ValueError):
    """A trajectory that cannot be turned into a chat."""


def _text(content: Any) -> str:
    """ATIF content is a string or a list of content parts; keep the text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text") or ""
            for part in content
            if isinstance(part, Mapping) and part.get("type") == "text"
        )
    return str(content)


def _tool_call(call: Mapping[str, Any]) -> dict[str, Any]:
    arguments = call.get("arguments")
    if arguments is None:
        arguments = {}
    return {
        "id": call["tool_call_id"],
        "type": "function",
        "function": {"name": call["function_name"], "arguments": arguments},
    }


def _is_policy_step(step: Mapping[str, Any]) -> bool:
    return step.get("llm_call_count") != 0 and not step.get("is_copied_context")


def atif_to_chat(trajectory: Mapping[str, Any]) -> AgentChat:
    """The agent's chat and tool schema from an ATIF trajectory (see the module docstring)."""
    steps = trajectory.get("steps")
    if not isinstance(steps, list) or not steps:
        raise TrajectoryError("trajectory has no steps")
    agent = trajectory.get("agent") or {}
    extra = agent.get("extra") or {}
    messages: list[dict[str, Any]] = []
    system_prompt = extra.get("system_prompt")
    if isinstance(system_prompt, str) and system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    for step in steps:
        source = step.get("source")
        text = _text(step.get("message"))
        if source in ("system", "user"):
            messages.append({"role": source, "content": text})
            continue
        if source != "agent":
            raise TrajectoryError(f"step {step.get('step_id')} has source {source!r}")
        calls = step.get("tool_calls") or []
        assistant: dict[str, Any] = {
            "role": "assistant",
            "content": text,
            TRAIN_KEY: _is_policy_step(step),
        }
        if calls:
            assistant["tool_calls"] = [_tool_call(call) for call in calls]
        messages.append(assistant)

        results = (step.get("observation") or {}).get("results") or []
        by_call: dict[str, Any] = {}
        unanswered: list[Any] = []
        call_ids = {call["tool_call_id"] for call in calls}
        for result in results:
            call_id = result.get("source_call_id")
            if call_id in call_ids and call_id not in by_call:
                by_call[call_id] = result
            else:
                unanswered.append(result)
        for call in calls:
            result = by_call.get(call["tool_call_id"]) or {}
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call["tool_call_id"],
                    "name": call["function_name"],
                    "content": _text(result.get("content")),
                }
            )
        for result in unanswered:
            messages.append({"role": "user", "content": _text(result.get("content"))})

    if not any(m["role"] == "assistant" and m.get(TRAIN_KEY) for m in messages):
        raise TrajectoryError("trajectory has no model-generated turn")
    tools = agent.get("tool_definitions")
    return AgentChat(messages=messages, tools=list(tools) if tools else None)


@dataclass(frozen=True)
class TrajectoryRef:
    """Where a row's trajectory is: inline in the output, or a path in a volume."""

    inline: Mapping[str, Any] | None
    volume_path: str | None


def trajectory_ref(output: Any, *, path_field: str, file: str) -> TrajectoryRef | None:
    """Locate the trajectory a task output points at.

    The output is a JSON object (or its string). A ``trajectory`` key holds the trajectory
    itself; otherwise ``path_field`` names the trial directory in the trajectory volume, as
    the Larkspur task's ``volume_path`` (``<tag>/<task>/<trial>``) does.
    """
    if isinstance(output, str):
        try:
            output = json.loads(output)
        except ValueError:
            return None
    if not isinstance(output, Mapping):
        return None
    inline = output.get("trajectory")
    if isinstance(inline, Mapping):
        return TrajectoryRef(inline=inline, volume_path=None)
    directory = output.get(path_field)
    if not isinstance(directory, str) or not directory:
        return None
    return TrajectoryRef(
        inline=None, volume_path=f"{directory.rstrip('/')}/{file.lstrip('/')}"
    )
