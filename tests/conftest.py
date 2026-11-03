import json
from collections.abc import Callable
from typing import Any

import pytest

# Qwen3-4B-Instruct-2507's chat template, verbatim, so masking is tested against the
# template the policy is trained with (without downloading the model).
QWEN3_INSTRUCT_TEMPLATE = """\
{%- if tools %}
    {{- '<|im_start|>system\\n' }}
    {%- if messages[0].role == 'system' %}
        {{- messages[0].content + '\\n\\n' }}
    {%- endif %}
    {{- "# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>" }}
    {%- for tool in tools %}
        {{- "\\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\"name\\": <function-name>, \\"arguments\\": <args-json-object>}\\n</tool_call><|im_end|>\\n" }}
{%- else %}
    {%- if messages[0].role == 'system' %}
        {{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}
    {%- endif %}
{%- endif %}
{%- for message in messages %}
    {%- if message.content is string %}
        {%- set content = message.content %}
    {%- else %}
        {%- set content = '' %}
    {%- endif %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) %}
        {{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>' + '\\n' }}
    {%- elif message.role == "assistant" %}
        {{- '<|im_start|>' + message.role + '\\n' + content }}
        {%- if message.tool_calls %}
            {%- for tool_call in message.tool_calls %}
                {%- if (loop.first and content) or (not loop.first) %}
                    {{- '\\n' }}
                {%- endif %}
                {%- if tool_call.function %}
                    {%- set tool_call = tool_call.function %}
                {%- endif %}
                {{- '<tool_call>\\n{"name": "' }}
                {{- tool_call.name }}
                {{- '", "arguments": ' }}
                {%- if tool_call.arguments is string %}
                    {{- tool_call.arguments }}
                {%- else %}
                    {{- tool_call.arguments | tojson }}
                {%- endif %}
                {{- '}\\n</tool_call>' }}
            {%- endfor %}
        {%- endif %}
        {{- '<|im_end|>\\n' }}
    {%- elif message.role == "tool" %}
        {%- if loop.first or (messages[loop.index0 - 1].role != "tool") %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\\n<tool_response>\\n' }}
        {{- content }}
        {{- '\\n</tool_response>' }}
        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
            {{- '<|im_end|>\\n' }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\\n' }}
{%- endif %}"""

SYSTEM_PROMPT = "You are a Tier-1 customer support agent at Larkspur."
TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "Search Larkspur's knowledge base.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_message_to_customer",
            "description": "Send a chat message to the customer.",
            "parameters": {
                "type": "object",
                "properties": {"message": {"type": "string"}},
                "required": ["message"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "end_conversation",
            "description": "Close the ticket.",
            "parameters": {
                "type": "object",
                "properties": {"resolution_summary": {"type": "string"}},
                "required": ["resolution_summary"],
            },
        },
    },
]


def larkspur_trajectory(reply: str) -> dict[str, Any]:
    """An ATIF trajectory shaped exactly as the Larkspur harness writes one.

    Built with Harbor's own models, the way ``support_agent.LarkspurSupportAgent._write``
    does: a user step with the ticket, an agent step with a knowledge-base search, a nudge
    after a turn with no tool call, a customer message and an ``end_conversation``.
    """
    from harbor.models.trajectories import (
        Agent,
        FinalMetrics,
        Observation,
        ObservationResult,
        Step,
        ToolCall,
        Trajectory,
    )

    model = "posttrain/Qwen/Qwen3-4B-Instruct-2507"
    steps = [
        Step(step_id=1, source="user", message="Ticket: my dryer never arrived."),
        Step(
            step_id=2,
            source="agent",
            model_name=model,
            message="",
            tool_calls=[
                ToolCall(
                    tool_call_id="call_1",
                    function_name="search_knowledge_base",
                    arguments={"query": "missed delivery refund"},
                )
            ],
            observation=Observation(
                results=[
                    ObservationResult(
                        source_call_id="call_1", content="Refund the delivery fee."
                    )
                ]
            ),
        ),
        Step(step_id=3, source="agent", model_name=model, message="I will help."),
        Step(
            step_id=4,
            source="user",
            message="The customer cannot see that text.",
        ),
        Step(
            step_id=5,
            source="agent",
            model_name=model,
            message="Let me tell them.",
            tool_calls=[
                ToolCall(
                    tool_call_id="call_2",
                    function_name="send_message_to_customer",
                    arguments={"message": reply},
                ),
                ToolCall(
                    tool_call_id="call_3",
                    function_name="end_conversation",
                    arguments={"resolution_summary": "refunded fee"},
                ),
            ],
            observation=Observation(
                results=[
                    ObservationResult(source_call_id="call_2", content="Customer: ok"),
                    ObservationResult(source_call_id="call_3", content='{"ok": true}'),
                ]
            ),
        ),
    ]
    trajectory = Trajectory(
        session_id="larkspur",
        agent=Agent(
            name="larkspur-support",
            version="1.0.0",
            model_name=model,
            tool_definitions=TOOLS,
            extra={"system_prompt": SYSTEM_PROMPT, "customer_model": "openai/x"},
        ),
        steps=steps,
        final_metrics=FinalMetrics(total_steps=len(steps)),
    )
    return json.loads(trajectory.model_dump_json(exclude_none=True))


@pytest.fixture
def make_trajectory() -> Callable[[str], dict[str, Any]]:
    return larkspur_trajectory


@pytest.fixture
def tiny_tokenizer() -> Any:
    """A byte-level tokenizer (one token per byte) with Qwen's special tokens and template."""
    pytest.importorskip("transformers")
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    backend = Tokenizer(
        models.BPE(vocab={c: i for i, c in enumerate(alphabet)}, merges=[])
    )
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False, use_regex=False
    )
    backend.decoder = decoders.ByteLevel()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        eos_token="<|im_end|>",
        pad_token="<|endoftext|>",
        additional_special_tokens=["<|im_start|>"],
    )
    tokenizer.chat_template = QWEN3_INSTRUCT_TEMPLATE
    return tokenizer
