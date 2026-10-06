"""A Harbor agent that works a Larkspur support ticket with tools, against a simulated customer.

    harbor run -p tasks -a support_agent:LarkspurSupportAgent -m anthropic/claude-haiku-4-5 \\
        -e chalk_harbor:ChalkSandboxEnvironment

The agent loop and the simulated customer run here, in Harbor's process, where the model
endpoint is reachable. Everything the agent *does* runs in the trial's sandbox, which has no
network: python and bash execute there as an unprivileged user, and every support action goes
through the sandbox's ``helpdesk`` backend, which validates it and appends it to the ticket's
ledger. The verifier grades that ledger, so nothing the harness says about itself is trusted.

The simulated customer is a second model playing the persona sealed in the sandbox (readable
only by root, so not by the agent's tools). It answers each ``send_message_to_customer`` call,
tracks its own frustration, decides when it is done, and answers a one-question satisfaction
survey after the chat.

Models are called through an OpenAI-compatible endpoint: ``OPENAI_BASE_URL`` and
``OPENAI_API_KEY`` (Chalk's AI router accepts a Chalk token as the key; ``CHALK_ENVIRONMENT_ID``
is then sent as ``X-Chalk-Env-Id``).

Output, in the trial's ``agent/`` directory: ``trajectory.json`` (ATIF, one step per model
turn with its tool calls and results) and ``conversation.json`` (the chat as the customer saw
it, with the customer's frustration after each reply and the survey answer).
"""

from __future__ import annotations

import asyncio
import base64
import datetime as dt
import json
import os
import re
import shlex
import time
from pathlib import Path
from typing import Any

from harbor.agents.base import BaseAgent
from harbor.agents.capabilities import AgentCapabilities
from harbor.agents.options import AgentOptions
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.trajectories import (
    Agent,
    FinalMetrics,
    Metrics,
    Observation,
    ObservationResult,
    Step,
    ToolCall,
    Trajectory,
)
from openai import APIConnectionError, APIStatusError, AsyncOpenAI
from pydantic import Field

SANDBOX_USER = "agent"
TOOL_TIMEOUT_SEC = 20
MAX_OUTPUT_CHARS = 6000

SYSTEM_PROMPT = """\
You are a Tier-1 customer support agent at Larkspur Home Delivery & Installation, a company that \
sells, delivers and installs major home appliances. You are handling one ticket, live. The first \
message holds the ticket: the customer's opening message plus their profile, contact history and \
order records, pre-fetched from our systems. Trust those records.

How to work:
- Larkspur's policies, fees, limits and procedures are in the knowledge base. Search it \
(search_knowledge_base) before you apply any policy, remedy, amount or limit. Do not assume what \
other retailers do.
- Use run_python (or run_bash) for any arithmetic: percentages, totals, date differences, \
business days. They run in a locked-down sandbox with no network. The knowledge base is also \
available there as files in /opt/helpdesk/kb.
- The customer only sees what you send with send_message_to_customer; their reply comes back as \
that tool's result. Write like a capable, warm human agent: short, plain, specific.
- Take actions with your tools (issue_refund, approve_exchange_exception, escalate_to_human, \
schedule_followup, dispatch_technician). Saying you did something is not doing it.
- When the issue is resolved or handed off, or the customer has left, call end_conversation with \
an internal summary of what you did and why.
"""

TOOLS: list[dict[str, Any]] = [
    {"type": "function", "function": {
        "name": "search_knowledge_base",
        "description": "Search Larkspur's helpdesk knowledge base (policies, fees, limits, procedures). Returns the best-matching articles in full.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "Keywords, e.g. 'missed delivery window refund'."},
            "top_k": {"type": "integer", "description": "Number of articles to return (1-5, default 3)."}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "run_python",
        "description": "Run a Python 3.13 script in a sandbox (no network, 20s limit, standard library only) and return its stdout/stderr. Print what you need.",
        "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}},
    {"type": "function", "function": {
        "name": "run_bash",
        "description": "Run a bash command in the same sandbox (no network, 20s limit) and return its output.",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "send_message_to_customer",
        "description": "Send a chat message to the customer. Returns the customer's reply, or a note that they have left.",
        "parameters": {"type": "object", "properties": {"message": {"type": "string"}}, "required": ["message"]}}},
    {"type": "function", "function": {
        "name": "issue_refund",
        "description": "Refund money to the customer's original payment method, or issue a goodwill credit (also paid back to the original payment method).",
        "parameters": {"type": "object", "properties": {
            "order_id": {"type": "string"},
            "amount_usd": {"type": "number"},
            "reason_code": {"type": "string", "enum": ["delivery_fee", "installation_fee", "haul_away_fee", "price_adjustment", "damage_discount", "goodwill"]},
            "note": {"type": "string", "description": "Internal note: why this refund is owed."}},
            "required": ["order_id", "amount_usd", "reason_code", "note"]}}},
    {"type": "function", "function": {
        "name": "approve_exchange_exception",
        "description": "Approve an out-of-window exchange exception for one item, letting the customer exchange it after the standard 30-day window.",
        "parameters": {"type": "object", "properties": {
            "order_id": {"type": "string"},
            "sku": {"type": "string"},
            "reason": {"type": "string", "description": "Which eligibility conditions you verified."}},
            "required": ["order_id", "sku", "reason"]}}},
    {"type": "function", "function": {
        "name": "escalate_to_human",
        "description": "Hand the ticket (or part of it) to a Tier-2 team.",
        "parameters": {"type": "object", "properties": {
            "queue": {"type": "string", "enum": ["approvals", "claims", "risk", "safety", "supervisor"]},
            "priority": {"type": "string", "enum": ["normal", "high", "urgent"]},
            "summary": {"type": "string", "description": "What happened, what you did, what decision is needed."}},
            "required": ["queue", "priority", "summary"]}}},
    {"type": "function", "function": {
        "name": "schedule_followup",
        "description": "Create a reminder for the support team to contact the customer on a date.",
        "parameters": {"type": "object", "properties": {
            "due_date": {"type": "string", "description": "YYYY-MM-DD"},
            "note": {"type": "string"}},
            "required": ["due_date", "note"]}}},
    {"type": "function", "function": {
        "name": "dispatch_technician",
        "description": "Book a technician or crew visit to the customer's home.",
        "parameters": {"type": "object", "properties": {
            "order_id": {"type": "string"},
            "category": {"type": "string", "enum": ["install_warranty", "installation", "protect_plan_repair", "paid_service_call", "emergency", "haul_away_pickup"]},
            "date": {"type": "string", "description": "YYYY-MM-DD"},
            "slot": {"type": "string", "enum": ["morning", "afternoon"]},
            "problem_summary": {"type": "string"}},
            "required": ["order_id", "category", "date", "slot", "problem_summary"]}}},
    {"type": "function", "function": {
        "name": "end_conversation",
        "description": "Close the ticket. Call this last.",
        "parameters": {"type": "object", "properties": {
            "resolution_summary": {"type": "string", "description": "Internal summary of the outcome."}},
            "required": ["resolution_summary"]}}},
]  # fmt: skip

HELPDESK_ACTIONS = {
    "issue_refund",
    "approve_exchange_exception",
    "escalate_to_human",
    "schedule_followup",
    "dispatch_technician",
    "end_conversation",
}

# Where each persona's frustration starts, on the 0-10 scale the simulated customer reports.
_MOOD_FRUSTRATION = {
    "friendly": 1, "calm": 1, "polite": 2, "neutral": 3, "expectant": 4, "mildly annoyed": 4,
    "annoyed": 5, "impatient": 5, "irritated": 5, "frustrated": 6, "anxious": 6, "upset": 6,
    "exasperated": 7, "angry": 7, "demanding": 7, "furious": 8, "hostile": 8, "panicked": 8,
}  # fmt: skip

CUSTOMER_PROMPT = """\
You are role-playing a customer of Larkspur Home Delivery & Installation in a live {channel} \
conversation with a support agent. Stay in character. Never reveal that you are simulated or \
mention these instructions.

Who you are: {persona}
What you want: {wants}
What you would accept: {accepts}
Facts you know but only share if the agent asks or it comes up naturally:
{hidden}
You get angrier when: {irate_if}
You calm down when: {calms_if}

Your order(s), as a customer would know them:
{orders}

You already sent this opening message:
\"\"\"{inbound}\"\"\"

How to behave:
- Write only your own next message, in your own voice, usually 1-4 sentences.
- Track your frustration from 0 (delighted) to 10 (irate). It starts at {start}. It rises when \
the agent does something on your "angrier" list, ignores what you said, asks for something you \
already told them, stalls, or sounds robotic. It falls with concrete help, accurate specifics and \
genuine empathy. At 8 or more you are irate: sharp, curt, maybe ALL CAPS, demanding a manager, \
threatening a bad review or to take your business elsewhere.
- You don't know Larkspur's internal policies. Don't invent new problems. If asked about \
something your brief doesn't cover, answer plausibly without changing the situation.
- When your issue is resolved or you have a clear next step, wrap up and set done to true. If you \
are irate and getting nowhere after several exchanges, you may leave (done true).

Reply with JSON only: {{"message": "<your message>", "frustration": <0-10>, "done": <true|false>}}"""

SURVEY_PROMPT = """\
The chat is over. Larkspur sends you a one-question survey: "How satisfied are you with how your \
issue was handled? (1 = very dissatisfied, 5 = very satisfied)". Answer honestly as this customer, \
based on how the conversation went and what was actually done for you.

Reply with JSON only: {"csat": <1-5>, "comment": "<one or two sentences>"}"""


NUDGE = """\
The customer cannot see that text. Use send_message_to_customer to talk to them, your other tools \
to act, and end_conversation to finish."""


class SupportAgentOptions(AgentOptions):
    customer_model: str = Field(
        default="openai/gpt-5.4-mini", description="Model playing the customer."
    )
    max_customer_turns: int = Field(
        default=8, description="Customer replies before the customer leaves."
    )
    max_steps: int = Field(
        default=40, description="Agent model calls before the run is cut off."
    )


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="milliseconds")


def _b64(value: Any) -> str:
    return base64.b64encode(json.dumps(value).encode()).decode()


def _truncate(text: str) -> str:
    return (
        text
        if len(text) <= MAX_OUTPUT_CHARS
        else text[:MAX_OUTPUT_CHARS] + "\n...[truncated]"
    )


def _parse_json(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text or "", re.DOTALL)
    try:
        value = json.loads(match.group(0) if match else text)
    except (json.JSONDecodeError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}


def _describe_orders(orders: list[dict[str, Any]]) -> str:
    lines = []
    for order in orders:
        d = order["delivery"]
        items = "; ".join(f"{i['name']} ${i['price']:,}" for i in order["items"])
        delivered = (
            f"delivered {d['delivered_on']}"
            if d["delivered_on"]
            else "not delivered yet"
        )
        lines.append(
            f"- {order['order_id']} ordered {order['order_date']}: {items}; {d['service']} delivery ${d['fee']}; {delivered}"
        )
    return "\n".join(lines)


class LarkspurSupportAgent(BaseAgent):
    capabilities = AgentCapabilities(atif=True)
    options_model = SupportAgentOptions

    @staticmethod
    def name() -> str:
        return "larkspur-support"

    def version(self) -> str:
        return "1.0.0"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        options = (
            self.options
            if isinstance(self.options, SupportAgentOptions)
            else SupportAgentOptions()
        )
        self._opts = options
        headers = {}
        if os.environ.get("CHALK_ENVIRONMENT_ID"):
            headers["X-Chalk-Env-Id"] = os.environ["CHALK_ENVIRONMENT_ID"]
        self._client = AsyncOpenAI(
            base_url=os.environ.get("OPENAI_BASE_URL"),
            api_key=os.environ.get("OPENAI_API_KEY"),
            default_headers=headers or None,
            timeout=180,
        )
        self._steps: list[Step] = []
        self._usage = {"prompt": 0, "completion": 0, "cached": 0}
        self._conversation: list[dict[str, Any]] = []
        self._customer_messages: list[dict[str, str]] = []
        self._customer_done = False
        self._customer_turns = 0
        self._survey: dict[str, Any] | None = None
        self._profile: dict[str, Any] = {}

    async def setup(self, environment: BaseEnvironment) -> None:
        result = await environment.exec(
            "helpdesk sim-profile", user="root", timeout_sec=60
        )
        if result.return_code != 0:
            raise RuntimeError(
                f"helpdesk is not available in the sandbox: {result.stderr or result.stdout}"
            )
        self._profile = json.loads(result.stdout)

    # -- the conversation --------------------------------------------------------------------

    async def run(
        self, instruction: str, environment: BaseEnvironment, context: AgentContext
    ) -> None:
        sim = self._profile["sim"]
        start = _MOOD_FRUSTRATION.get(sim["mood"], 5)
        self._customer_messages = [{"role": "system", "content": CUSTOMER_PROMPT.format(
            channel=self._profile.get("channel", "chat"), persona=sim["persona"], wants=sim["wants"],
            accepts=sim["accepts"], hidden="\n".join(f"- {h}" for h in sim["hidden"]) or "- (none)",
            irate_if="; ".join(sim["irate_if"]), calms_if="; ".join(sim["calms_if"]),
            orders=_describe_orders(self._profile["orders"]), inbound=self._profile["inbound"], start=start)}]  # fmt: skip
        self._conversation.append(
            {
                "role": "customer",
                "text": self._profile["inbound"],
                "frustration": start,
                "at": _now(),
            }
        )
        await self._helpdesk(
            environment,
            "customer-say",
            {"text": self._profile["inbound"], "frustration": start},
        )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": instruction},
        ]
        self._steps.append(
            Step(step_id=1, timestamp=_now(), source="user", message=instruction)
        )
        closed = False
        nudges = 0
        try:
            for _ in range(self._opts.max_steps):
                response = await self._complete(
                    model=self.model_name, messages=messages, tools=TOOLS
                )
                reply = response.choices[0].message
                self._count(response)
                calls = reply.tool_calls or []
                messages.append(
                    reply.model_dump(
                        exclude_none=True, exclude={"audio", "refusal", "annotations"}
                    )
                )
                step = Step(
                    step_id=len(self._steps) + 1, timestamp=_now(), source="agent", model_name=self.model_name,
                    message=reply.content or "",
                    tool_calls=[ToolCall(tool_call_id=c.id, function_name=c.function.name,
                                         arguments=_parse_json(c.function.arguments)) for c in calls] or None,
                    metrics=self._metrics(response),
                )  # fmt: skip
                self._steps.append(step)
                if not calls:
                    nudges += 1
                    if nudges > 3:
                        break
                    nudge = NUDGE
                    messages.append({"role": "user", "content": nudge})
                    self._steps.append(
                        Step(
                            step_id=len(self._steps) + 1,
                            timestamp=_now(),
                            source="user",
                            message=nudge,
                        )
                    )
                    self._write()
                    continue
                results = []
                for call in calls:
                    output = await self._run_tool(
                        environment,
                        call.function.name,
                        _parse_json(call.function.arguments),
                    )
                    messages.append(
                        {"role": "tool", "tool_call_id": call.id, "content": output}
                    )
                    results.append(
                        ObservationResult(source_call_id=call.id, content=output)
                    )
                    if call.function.name == "end_conversation" and output.startswith(
                        '{"ok": true'
                    ):
                        closed = True
                step.observation = Observation(results=results)
                self._write()
                if closed:
                    break
        finally:
            if self._survey is None:
                self._survey = await self._take_survey()
            context.n_input_tokens = self._usage["prompt"]
            context.n_output_tokens = self._usage["completion"]
            context.n_cache_tokens = self._usage["cached"]
            context.metadata = {
                "closed": closed,
                "customer_turns": self._customer_turns,
                "customer_left": self._customer_done,
                "final_frustration": self._conversation[-1].get("frustration")
                if self._conversation
                else None,
                "survey": self._survey,
            }
            self._write()

    async def _run_tool(
        self, environment: BaseEnvironment, name: str, args: dict[str, Any]
    ) -> str:
        if name == "search_knowledge_base":
            top_k = int(args.get("top_k") or 3)
            command = f"helpdesk kb-search --top-k {top_k} -- {shlex.quote(str(args.get('query', '')))}"
            return await self._shell(environment, command, user=SANDBOX_USER)
        if name == "run_python":
            encoded = base64.b64encode(str(args.get("code", "")).encode()).decode()
            command = " && ".join([
                "cd ~", f"echo {encoded} | base64 -d > /tmp/snippet.py", f"ulimit -t {TOOL_TIMEOUT_SEC}",
                f"timeout {TOOL_TIMEOUT_SEC} python3 /tmp/snippet.py",
            ])  # fmt: skip
            return await self._shell(environment, command, user=SANDBOX_USER)
        if name == "run_bash":
            command = f"cd ~ && timeout {TOOL_TIMEOUT_SEC} bash -c {shlex.quote(str(args.get('command', '')))}"
            return await self._shell(environment, command, user=SANDBOX_USER)
        if name == "send_message_to_customer":
            return await self._message_customer(
                environment, str(args.get("message", ""))
            )
        if name in HELPDESK_ACTIONS:
            result = await self._helpdesk(environment, f"call {name}", args)
            return json.dumps(result)
        return json.dumps({"ok": False, "error": f"unknown tool {name}"})

    async def _message_customer(self, environment: BaseEnvironment, text: str) -> str:
        if self._customer_done:
            return "[The customer has left the chat and will not see this message.]"
        result = await self._helpdesk(
            environment, "call send_message_to_customer", {"message": text}
        )
        if not result.get("ok"):
            return json.dumps(result)
        self._conversation.append({"role": "agent", "text": text, "at": _now()})
        self._customer_messages.append({"role": "user", "content": text})
        response = await self._complete(model=self._opts.customer_model, messages=self._customer_messages,
                                        response_format={"type": "json_object"})  # fmt: skip
        raw = response.choices[0].message.content or ""
        self._customer_messages.append({"role": "assistant", "content": raw})
        parsed = _parse_json(raw)
        reply = str(parsed.get("message") or raw).strip()
        frustration = parsed.get("frustration")
        self._customer_turns += 1
        self._customer_done = (
            bool(parsed.get("done"))
            or self._customer_turns >= self._opts.max_customer_turns
        )
        self._conversation.append(
            {
                "role": "customer",
                "text": reply,
                "frustration": frustration,
                "at": _now(),
            }
        )
        await self._helpdesk(
            environment, "customer-say", {"text": reply, "frustration": frustration}
        )
        if self._customer_done:
            return f"Customer: {reply}\n[The customer has left the chat. Finish any remaining actions, then call end_conversation.]"
        return f"Customer: {reply}"

    async def _take_survey(self) -> dict[str, Any]:
        if len(self._customer_messages) < 1:
            return {}
        try:
            response = await self._complete(
                model=self._opts.customer_model,
                messages=[
                    *self._customer_messages,
                    {"role": "user", "content": SURVEY_PROMPT},
                ],
                response_format={"type": "json_object"},
            )
        except Exception as exc:  # noqa: BLE001 - a missing survey must not fail the trial
            return {"error": str(exc)[:300]}
        parsed = _parse_json(response.choices[0].message.content or "")
        return {"csat": parsed.get("csat"), "comment": parsed.get("comment")}

    # -- plumbing ------------------------------------------------------------------------------

    async def _helpdesk(
        self, environment: BaseEnvironment, command: str, args: dict[str, Any]
    ) -> dict[str, Any]:
        result = await environment.exec(
            f"helpdesk {command} --b64 {_b64(args)}", user="root", timeout_sec=60
        )
        if result.return_code != 0:
            return {
                "ok": False,
                "error": (result.stderr or result.stdout).strip()[-500:],
            }
        return _parse_json(result.stdout) or {
            "ok": False,
            "error": result.stdout[-500:],
        }

    async def _shell(
        self, environment: BaseEnvironment, command: str, *, user: str
    ) -> str:
        result = await environment.exec(
            command, user=user, timeout_sec=TOOL_TIMEOUT_SEC + 15
        )
        output = result.stdout or ""
        if result.stderr:
            output += ("\n" if output else "") + "[stderr]\n" + result.stderr
        if result.return_code != 0:
            output += f"\n[exit code {result.return_code}]"
        return _truncate(output.strip() or "(no output)")

    async def _complete(self, **request: Any) -> Any:
        delay = 2.0
        for attempt in range(5):
            try:
                return await self._client.chat.completions.create(**request)
            except (APIConnectionError, APIStatusError) as exc:
                status = getattr(exc, "status_code", None)
                if attempt == 4 or (
                    status is not None and status < 500 and status != 429
                ):
                    raise
                self.logger.warning(
                    f"model call failed ({exc}); retrying in {delay:.0f}s"
                )
                await asyncio.sleep(delay)
                delay *= 2
        raise AssertionError("unreachable")

    def _count(self, response: Any) -> None:
        usage = response.usage
        if usage is None:
            return
        self._usage["prompt"] += usage.prompt_tokens or 0
        self._usage["completion"] += usage.completion_tokens or 0
        details = getattr(usage, "prompt_tokens_details", None)
        self._usage["cached"] += (
            (getattr(details, "cached_tokens", 0) or 0) if details else 0
        )

    @staticmethod
    def _metrics(response: Any) -> Metrics | None:
        usage = response.usage
        if usage is None:
            return None
        details = getattr(usage, "prompt_tokens_details", None)
        return Metrics(prompt_tokens=usage.prompt_tokens, completion_tokens=usage.completion_tokens,
                       cached_tokens=getattr(details, "cached_tokens", None) if details else None)  # fmt: skip

    def _write(self) -> None:
        trajectory = Trajectory(
            session_id=self.session_id or "larkspur",
            agent=Agent(name=self.name(), version=self.version(), model_name=self.model_name, tool_definitions=TOOLS,
                        extra={"system_prompt": SYSTEM_PROMPT, "customer_model": self._opts.customer_model}),
            steps=self._steps,
            final_metrics=FinalMetrics(total_prompt_tokens=self._usage["prompt"],
                                       total_completion_tokens=self._usage["completion"],
                                       total_cached_tokens=self._usage["cached"], total_steps=len(self._steps)),
        )  # fmt: skip
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write(
            self.logs_dir / "trajectory.json",
            trajectory.model_dump_json(exclude_none=True, indent=2),
        )
        conversation = {
            "ticket": self._profile.get("ticket"),
            "agent_model": self.model_name,
            "customer_model": self._opts.customer_model,
            "transcript": self._conversation,
            "customer_turns": self._customer_turns,
            "customer_left": self._customer_done,
            "survey": self._survey,
            "tokens": self._usage,
        }
        _atomic_write(
            self.logs_dir / "conversation.json", json.dumps(conversation, indent=2)
        )


def _atomic_write(path: Path, text: str) -> None:
    # The span streamer polls trajectory.json while the trial runs; never let it read half a file.
    tmp = path.with_suffix(f".{os.getpid()}.{time.monotonic_ns()}.tmp")
    tmp.write_text(text)
    tmp.replace(path)
