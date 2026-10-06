from collections.abc import Callable
from typing import Any

import pytest

from chalk_harbor.post_training.trajectory import (
    TrajectoryError,
    atif_to_chat,
    trajectory_ref,
)


def test_atif_rebuilds_the_agents_chat(
    make_trajectory: Callable[[str], dict[str, Any]],
) -> None:
    chat = atif_to_chat(make_trajectory("Refunded."))

    assert [m["role"] for m in chat.messages] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
        "user",
        "assistant",
        "tool",
        "tool",
    ]
    assert chat.messages[0]["content"].startswith("You are a Tier-1")
    assert chat.messages[2] == {
        "role": "assistant",
        "content": "",
        "train": True,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "search_knowledge_base",
                    "arguments": {"query": "missed delivery refund"},
                },
            }
        ],
    }
    assert chat.messages[3] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "name": "search_knowledge_base",
        "content": "Refund the delivery fee.",
    }
    assert chat.messages[4] == {
        "role": "assistant",
        "content": "I will help.",
        "train": True,
    }
    assert [m["tool_call_id"] for m in chat.messages[7:]] == ["call_2", "call_3"]
    assert chat.tools is not None
    assert [t["function"]["name"] for t in chat.tools] == [
        "search_knowledge_base",
        "send_message_to_customer",
        "end_conversation",
    ]


def test_results_are_matched_to_calls_by_id_not_order() -> None:
    trajectory = {
        "steps": [
            {"step_id": 1, "source": "user", "message": "hi"},
            {
                "step_id": 2,
                "source": "agent",
                "message": "",
                "tool_calls": [
                    {"tool_call_id": "a", "function_name": "f", "arguments": {}},
                    {"tool_call_id": "b", "function_name": "g", "arguments": {}},
                ],
                "observation": {
                    "results": [
                        {"source_call_id": "b", "content": "from g"},
                        {"source_call_id": None, "content": "system note"},
                    ]
                },
            },
        ]
    }

    messages = atif_to_chat(trajectory).messages

    assert messages[2:] == [
        {"role": "tool", "tool_call_id": "a", "name": "f", "content": ""},
        {"role": "tool", "tool_call_id": "b", "name": "g", "content": "from g"},
        {"role": "user", "content": "system note"},
    ]


def test_steps_without_a_model_call_are_context_not_training() -> None:
    trajectory = {
        "steps": [
            {"step_id": 1, "source": "user", "message": "hi"},
            {"step_id": 2, "source": "agent", "message": "canned", "llm_call_count": 0},
            {
                "step_id": 3,
                "source": "agent",
                "message": [{"type": "text", "text": "real"}],
            },
        ]
    }

    messages = atif_to_chat(trajectory).messages

    assert [(m["content"], m.get("train")) for m in messages[1:]] == [
        ("canned", False),
        ("real", True),
    ]


def test_a_trajectory_without_model_turns_is_rejected() -> None:
    with pytest.raises(TrajectoryError):
        atif_to_chat({"steps": [{"step_id": 1, "source": "user", "message": "hi"}]})


def test_larkspur_output_points_into_the_traces_volume() -> None:
    output = (
        '{"volume_path": "larkspur-x/missed-delivery-first/missed__abc", "reward": 1}'
    )

    ref = trajectory_ref(output, path_field="volume_path", file="agent/trajectory.json")

    assert ref is not None
    assert ref.inline is None
    assert (
        ref.volume_path
        == "larkspur-x/missed-delivery-first/missed__abc/agent/trajectory.json"
    )


def test_an_inline_trajectory_wins_and_bad_outputs_have_none() -> None:
    inline = {"steps": []}

    assert (
        trajectory_ref(
            {"trajectory": inline, "volume_path": "x"},
            path_field="volume_path",
            file="t",
        ).inline
        == inline
    )  # type: ignore[union-attr]
    assert trajectory_ref("not json", path_field="volume_path", file="t") is None
    assert trajectory_ref({"error": "boom"}, path_field="volume_path", file="t") is None
