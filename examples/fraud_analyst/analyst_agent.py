"""A Harbor agent that works a Kestrel Pay fraud case with tools and submits approve or deny.

    harbor run -p tasks -a analyst_agent:FraudAnalystAgent -m anthropic/claude-haiku-4-5 \\
        -e chalk_harbor:ChalkSandboxEnvironment

The loop runs here, in Harbor's process, where the model endpoint is reachable; every tool runs
in the trial's sandbox, which has no network. Python executes there as an unprivileged user, and
SQL, Chalk feature queries, the paid checks and the final decision go through the sandbox's
``fraudlab`` backend, which meters and records them in the case ledger.

Models are reached through an OpenAI-compatible endpoint (``OPENAI_BASE_URL``,
``OPENAI_API_KEY``; with Chalk's AI router, ``CHALK_ENVIRONMENT_ID`` is sent as
``X-Chalk-Env-Id``). Output in the trial's ``agent/``: ``trajectory.json`` (ATIF) and
``investigation.json`` (each tool call with its cost, and the submitted decision).
"""

from __future__ import annotations

import asyncio
import base64
import datetime as dt
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from harbor.agents.base import BaseAgent
from harbor.agents.capabilities import AgentCapabilities
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

SANDBOX_USER = "agent"
TOOL_TIMEOUT_SEC = 20
MAX_OUTPUT_CHARS = 8000
MAX_STEPS = 30

SYSTEM_PROMPT = """\
You are a fraud case analyst at Kestrel Pay, a payments company. Each case is a customer payout \
held for review. Investigate it with your tools and decide whether to approve it (legitimate) or \
deny it (fraud).

How to work:
- Start with the free evidence: the Chalk features and the warehouse (SQL). Look for both fraud \
signals and legitimate explanations, and compare the account with its own history and with \
accounts it is connected to.
- The paid checks cost real money: deep identity verification is $5 a call, the social network \
search $2. Use them when the free evidence leaves the decision genuinely open, not by reflex.
- Use run_python for anything that needs computation.
- Finish by calling submit_decision. The analysis is read by a human reviewer: state the decision, \
the evidence on both sides with the specific values you found, how you weighed it, what you \
checked and why, and any residual uncertainty. Do not claim evidence you did not retrieve.
"""

TOOLS: list[dict[str, Any]] = [
    {"type": "function", "function": {
        "name": "run_sql",
        "description": "Run a read-only SQLite query against the warehouse (accounts, logins, transactions, review_queue). Free. Returns up to 200 rows.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "chalk_query",
        "description": "Fetch online Chalk features for an account. Free. Omit `features` for all of them.",
        "parameters": {"type": "object", "properties": {
            "account_id": {"type": "string"},
            "features": {"type": "array", "items": {"type": "string"}}},
            "required": ["account_id"]}}},
    {"type": "function", "function": {
        "name": "run_python",
        "description": "Run a Python 3.13 script in the sandbox (no network, 20s, standard library; the warehouse is at /opt/fraudlab/data/warehouse.sqlite). Free. Print what you need.",
        "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}},
    {"type": "function", "function": {
        "name": "deep_verification",
        "description": "Deep identity verification, a fresh check of the person requesting the payout: document and SSN/name/DOB match, liveness (a live selfie matched to the ID document), synthetic-identity score, phone tenure, recent SIM swap, address history. Costs $5 per call.",
        "parameters": {"type": "object", "properties": {"account_id": {"type": "string"}}, "required": ["account_id"]}}},
    {"type": "function", "function": {
        "name": "social_network_search",
        "description": "Identities linked to an account across the fraud consortium (shared phone, device, email pattern, SSN) and their status at other institutions. Costs $2 per call.",
        "parameters": {"type": "object", "properties": {"account_id": {"type": "string"}}, "required": ["account_id"]}}},
    {"type": "function", "function": {
        "name": "submit_decision",
        "description": "Close the case with your decision. Call this last.",
        "parameters": {"type": "object", "properties": {
            "decision": {"type": "string", "enum": ["approve", "deny"]},
            "confidence": {"type": "number", "description": "0 to 1"},
            "analysis": {"type": "string", "description": "The auditable analysis justifying the decision (markdown)."}},
            "required": ["decision", "confidence", "analysis"]}}},
]  # fmt: skip

BACKEND_TOOLS = {
    "run_sql",
    "chalk_query",
    "deep_verification",
    "social_network_search",
    "submit_decision",
}
NUDGE = """\
Nothing happens until you call a tool. Investigate with your tools, then call submit_decision."""


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="milliseconds")


def _parse_json(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text or "", re.DOTALL)
    try:
        value = json.loads(match.group(0) if match else text)
    except (json.JSONDecodeError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}


def _truncate(text: str) -> str:
    return (
        text
        if len(text) <= MAX_OUTPUT_CHARS
        else text[:MAX_OUTPUT_CHARS] + "\n...[truncated]"
    )


class FraudAnalystAgent(BaseAgent):
    capabilities = AgentCapabilities(atif=True)

    @staticmethod
    def name() -> str:
        return "kestrel-fraud-analyst"

    def version(self) -> str:
        return "1.0.0"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        headers = (
            {"X-Chalk-Env-Id": os.environ["CHALK_ENVIRONMENT_ID"]}
            if os.environ.get("CHALK_ENVIRONMENT_ID")
            else None
        )
        self._client = AsyncOpenAI(base_url=os.environ.get("OPENAI_BASE_URL"), api_key=os.environ.get("OPENAI_API_KEY"),
                                   default_headers=headers, timeout=180)  # fmt: skip
        self._steps: list[Step] = []
        self._usage = {"prompt": 0, "completion": 0, "cached": 0}
        self._calls: list[dict[str, Any]] = []
        self._decision: dict[str, Any] | None = None

    async def setup(self, environment: BaseEnvironment) -> None:
        result = await environment.exec("fraudlab ledger", user="root", timeout_sec=60)
        if result.return_code != 0:
            raise RuntimeError(
                f"fraudlab is not available in the sandbox: {result.stderr or result.stdout}"
            )

    async def run(
        self, instruction: str, environment: BaseEnvironment, context: AgentContext
    ) -> None:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": instruction},
        ]
        self._steps.append(
            Step(step_id=1, timestamp=_now(), source="user", message=instruction)
        )
        nudges = 0
        try:
            for _ in range(MAX_STEPS):
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
                step = Step(step_id=len(self._steps) + 1, timestamp=_now(), source="agent", model_name=self.model_name,
                            message=reply.content or "",
                            tool_calls=[ToolCall(tool_call_id=c.id, function_name=c.function.name,
                                                 arguments=_parse_json(c.function.arguments)) for c in calls] or None,
                            metrics=self._metrics(response))  # fmt: skip
                self._steps.append(step)
                if not calls:
                    nudges += 1
                    if nudges > 3:
                        break
                    messages.append({"role": "user", "content": NUDGE})
                    self._steps.append(
                        Step(
                            step_id=len(self._steps) + 1,
                            timestamp=_now(),
                            source="user",
                            message=NUDGE,
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
                step.observation = Observation(results=results)
                self._write()
                if self._decision is not None:
                    break
        finally:
            context.n_input_tokens = self._usage["prompt"]
            context.n_output_tokens = self._usage["completion"]
            context.n_cache_tokens = self._usage["cached"]
            context.metadata = {
                "decision": self._decision,
                "tool_cost_usd": self._cost(),
            }
            self._write()

    async def _run_tool(
        self, environment: BaseEnvironment, name: str, args: dict[str, Any]
    ) -> str:
        started = time.monotonic()
        if name == "run_python":
            encoded = base64.b64encode(str(args.get("code", "")).encode()).decode()
            command = " && ".join(["cd ~", f"echo {encoded} | base64 -d > /tmp/snippet.py",
                                   f"timeout {TOOL_TIMEOUT_SEC} python3 /tmp/snippet.py"])  # fmt: skip
            result = await environment.exec(
                command, user=SANDBOX_USER, timeout_sec=TOOL_TIMEOUT_SEC + 15
            )
            output = (result.stdout or "") + (
                f"\n[stderr]\n{result.stderr}" if result.stderr else ""
            )
            if result.return_code != 0:
                output += f"\n[exit code {result.return_code}]"
            output = _truncate(output.strip() or "(no output)")
            record = {
                "tool": name,
                "args": args,
                "ok": result.return_code == 0,
                "cost_usd": 0.0,
            }
        elif name in BACKEND_TOOLS:
            encoded = base64.b64encode(json.dumps(args).encode()).decode()
            result = await environment.exec(
                f"fraudlab call {name} --b64 {encoded}", user="root", timeout_sec=60
            )
            payload = _parse_json(result.stdout) or {
                "ok": False,
                "error": (result.stderr or result.stdout)[-500:],
            }
            output = _truncate(json.dumps(payload))
            record = {
                "tool": name,
                "args": args,
                "ok": bool(payload.get("ok")),
                "cost_usd": float(payload.get("cost_usd") or 0.0),
            }
            if name == "submit_decision" and payload.get("ok"):
                self._decision = {
                    k: payload.get(k) for k in ("decision", "confidence", "analysis")
                }
        else:
            output = json.dumps({"ok": False, "error": f"unknown tool {name}"})
            record = {"tool": name, "args": args, "ok": False, "cost_usd": 0.0}
        record.update(
            result_preview=output[:2000], seconds=round(time.monotonic() - started, 2)
        )
        self._calls.append(record)
        if record["cost_usd"]:
            output += (
                f"\n[billed ${record['cost_usd']:.2f}; case total ${self._cost():.2f}]"
            )
        return output

    def _cost(self) -> float:
        return round(sum(c["cost_usd"] for c in self._calls if c["ok"]), 2)

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
                await asyncio.sleep(delay)
                delay *= 2
        raise AssertionError("unreachable")

    def _count(self, response: Any) -> None:
        usage = response.usage
        if usage is not None:
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
        return Metrics(
            prompt_tokens=usage.prompt_tokens, completion_tokens=usage.completion_tokens
        )

    def _write(self) -> None:
        trajectory = Trajectory(
            session_id=self.session_id or "kestrel",
            agent=Agent(name=self.name(), version=self.version(), model_name=self.model_name, tool_definitions=TOOLS,
                        extra={"system_prompt": SYSTEM_PROMPT}),
            steps=self._steps,
            final_metrics=FinalMetrics(total_prompt_tokens=self._usage["prompt"], total_completion_tokens=self._usage["completion"],
                                       total_cached_tokens=self._usage["cached"], total_steps=len(self._steps)),
        )  # fmt: skip
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write(
            self.logs_dir / "trajectory.json",
            trajectory.model_dump_json(exclude_none=True, indent=2),
        )
        investigation = {"agent_model": self.model_name, "decision": self._decision, "tool_cost_usd": self._cost(),
                         "tool_calls": self._calls, "tokens": self._usage}  # fmt: skip
        _atomic_write(
            self.logs_dir / "investigation.json", json.dumps(investigation, indent=2)
        )


def _atomic_write(path: Path, text: str) -> None:
    # The span streamer polls trajectory.json while the trial runs; never let it read half a file.
    tmp = path.with_suffix(f".{os.getpid()}.{time.monotonic_ns()}.tmp")
    tmp.write_text(text)
    tmp.replace(path)
