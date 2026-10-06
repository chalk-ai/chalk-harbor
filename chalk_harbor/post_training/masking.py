"""Tokenize an agent chat with the policy's chat template and mark the tokens it generated.

The whole chat is rendered once with ``apply_chat_template(messages, tools=...)`` and
tokenized once, so the model scores exactly the token stream a fresh rendering produces. The
loss mask comes from character spans: an assistant turn's generated text is what rendering
through that turn adds to rendering everything before it with the generation prompt. That is
exact for templates that only append (Qwen3's instruct template does, at assistant
boundaries); a template that rewrites earlier turns, such as one that drops past reasoning,
fails the prefix check and the chat is rejected rather than mis-masked.

The trailing whitespace a template puts after the end-of-turn token is not generated, so it
is not trained; the end-of-turn token is.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from chalk_harbor.post_training.trajectory import TRAIN_KEY, AgentChat


class TemplateMismatchError(ValueError):
    """The chat template does not render the chat as a sequence of appended turns."""


@dataclass(frozen=True)
class TokenizedChat:
    input_ids: list[int]
    # 1 where the policy generated the token, 0 elsewhere; aligned with input_ids.
    loss_mask: list[int]
    truncated: bool

    @property
    def generated_tokens(self) -> int:
        return sum(self.loss_mask)


def _render(
    tokenizer: Any,
    messages: Sequence[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    add_generation_prompt: bool,
) -> str:
    return tokenizer.apply_chat_template(
        [{k: v for k, v in m.items() if k != TRAIN_KEY} for m in messages],
        tools=tools,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


def generated_spans(
    tokenizer: Any, chat: AgentChat
) -> tuple[str, list[tuple[int, int]]]:
    """The rendered chat and the ``[start, end)`` character spans the policy generated."""
    messages = chat.messages
    full = _render(tokenizer, messages, chat.tools, False)
    spans: list[tuple[int, int]] = []
    for i, message in enumerate(messages):
        if message["role"] != "assistant" or not message.get(TRAIN_KEY):
            continue
        if i == 0:
            raise TemplateMismatchError("the chat opens with an assistant turn")
        before = _render(tokenizer, messages[:i], chat.tools, True)
        through = _render(tokenizer, messages[: i + 1], chat.tools, False)
        if not through.startswith(before) or not full.startswith(through):
            raise TemplateMismatchError(
                f"rendering message {i} does not extend the rendering before it"
            )
        generated = through[len(before) :].rstrip()
        if generated:
            spans.append((len(before), len(before) + len(generated)))
    return full, spans


def tokenize_chat(tokenizer: Any, chat: AgentChat, max_tokens: int) -> TokenizedChat:
    """Token ids and loss mask for a chat, truncated to ``max_tokens`` from the end.

    Truncation keeps the beginning: later turns are conditioned on earlier ones, so the
    prefix is what can still be scored correctly.
    """
    text, spans = generated_spans(tokenizer, chat)
    encoding = tokenizer(
        text, add_special_tokens=False, return_offsets_mapping=True, truncation=False
    )
    input_ids = list(encoding["input_ids"])
    offsets = encoding["offset_mapping"]
    mask = [0] * len(input_ids)
    span_index = 0
    for t, (start, end) in enumerate(offsets):
        if end <= start:
            continue
        while span_index < len(spans) and spans[span_index][1] <= start:
            span_index += 1
        if span_index == len(spans):
            break
        span_start, span_end = spans[span_index]
        # A token straddling a span edge mixes generated and template text; leave it out.
        if start >= span_start and end <= span_end:
            mask[t] = 1
    truncated = len(input_ids) > max_tokens
    return TokenizedChat(
        input_ids=input_ids[:max_tokens],
        loss_mask=mask[:max_tokens],
        truncated=truncated,
    )
