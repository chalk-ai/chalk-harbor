from collections.abc import Callable
from typing import Any

import pytest

from chalk_harbor.post_training.masking import (
    TemplateMismatchError,
    generated_spans,
    tokenize_chat,
)
from chalk_harbor.post_training.trajectory import AgentChat, atif_to_chat


def _masked_text(tokenizer: Any, input_ids: list[int], mask: list[int]) -> list[str]:
    """The decoded runs of consecutive masked tokens."""
    runs: list[str] = []
    current: list[int] = []
    for token, keep in zip(input_ids, mask, strict=True):
        if keep:
            current.append(token)
        elif current:
            runs.append(tokenizer.decode(current))
            current = []
    if current:
        runs.append(tokenizer.decode(current))
    return runs


def test_only_the_policys_turns_are_trained(
    tiny_tokenizer: Any, make_trajectory: Callable[[str], dict[str, Any]]
) -> None:
    chat = atif_to_chat(make_trajectory("Your fee is refunded."))

    tokens = tokenize_chat(tiny_tokenizer, chat, 100_000)

    assert _masked_text(tiny_tokenizer, tokens.input_ids, tokens.loss_mask) == [
        '<tool_call>\n{"name": "search_knowledge_base", "arguments": '
        + '{"query": "missed delivery refund"}}\n</tool_call><|im_end|>',
        "I will help.<|im_end|>",
        "Let me tell them.\n"
        + '<tool_call>\n{"name": "send_message_to_customer", "arguments": '
        + '{"message": "Your fee is refunded."}}\n</tool_call>\n'
        + '<tool_call>\n{"name": "end_conversation", "arguments": '
        + '{"resolution_summary": "refunded fee"}}\n</tool_call><|im_end|>',
    ]
    assert not tokens.truncated


def test_the_tokens_are_the_full_rendering_with_tools(
    tiny_tokenizer: Any, make_trajectory: Callable[[str], dict[str, Any]]
) -> None:
    chat = atif_to_chat(make_trajectory("ok"))

    tokens = tokenize_chat(tiny_tokenizer, chat, 100_000)
    text = tiny_tokenizer.decode(tokens.input_ids)

    assert text.startswith("<|im_start|>system\nYou are a Tier-1")
    tool_block = text.split("<tools>\n")[1].split("\n</tools>")[0]
    assert '"name": "search_knowledge_base"' in tool_block
    assert (
        "<|im_start|>user\n<tool_response>\nRefund the delivery fee.\n</tool_response>"
        in text
    )
    assert "<|im_start|>user\nThe customer cannot see that text.<|im_end|>" in text


def test_truncation_keeps_the_beginning(
    tiny_tokenizer: Any, make_trajectory: Callable[[str], dict[str, Any]]
) -> None:
    chat = atif_to_chat(make_trajectory("ok"))
    full = tokenize_chat(tiny_tokenizer, chat, 100_000)

    short = tokenize_chat(tiny_tokenizer, chat, len(full.input_ids) - 40)

    assert short.truncated
    assert short.input_ids == full.input_ids[: len(short.input_ids)]
    assert short.loss_mask == full.loss_mask[: len(short.input_ids)]


def test_a_template_that_rewrites_history_is_rejected(tiny_tokenizer: Any) -> None:
    # Drops every assistant turn but the last, as reasoning-stripping templates do.
    tiny_tokenizer.chat_template = (
        "{% for m in messages %}{% if m.role != 'assistant' or loop.last %}"
        + "{{ m.role }}: {{ m.content }}\n{% endif %}{% endfor %}"
        + "{% if add_generation_prompt %}assistant: {% endif %}"
    )
    chat = AgentChat(
        messages=[
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b", "train": True},
            {"role": "user", "content": "c"},
            {"role": "assistant", "content": "d", "train": True},
        ],
        tools=None,
    )

    with pytest.raises(TemplateMismatchError):
        generated_spans(tiny_tokenizer, chat)
