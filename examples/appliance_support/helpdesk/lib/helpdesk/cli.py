"""``helpdesk``: the command-line face of the Larkspur helpdesk inside a scenario sandbox.

helpdesk kb-search "<query>" [--top-k N]     anyone: search the knowledge base
helpdesk kb-list                             anyone: list knowledge-base articles
helpdesk call <tool> --b64 <json>            support platform only: run a support action
helpdesk customer-say --b64 <json>           support platform only: record the customer's reply
helpdesk sim-profile                         support platform only: the simulated customer's brief
helpdesk ledger                              support platform only: everything done on the ticket
helpdesk grade --rubric R --out DIR          verifier only: grade the ticket, write reward.json
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

from helpdesk import kb, store, tools
from helpdesk.store import HelpdeskError


def _payload(args: argparse.Namespace) -> dict:
    raw = base64.b64decode(args.b64).decode() if args.b64 else (args.json or "{}")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise HelpdeskError("arguments must be a JSON object")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="helpdesk")
    sub = parser.add_subparsers(dest="command", required=True)
    search = sub.add_parser("kb-search")
    search.add_argument("query", nargs="+")
    search.add_argument("--top-k", type=int, default=3)
    sub.add_parser("kb-list")
    call = sub.add_parser("call")
    call.add_argument("tool")
    call.add_argument("--b64")
    call.add_argument("--json")
    say = sub.add_parser("customer-say")
    say.add_argument("--b64")
    say.add_argument("--json")
    sub.add_parser("sim-profile")
    sub.add_parser("ledger")
    grade = sub.add_parser("grade")
    grade.add_argument("--rubric", required=True)
    grade.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    try:
        if args.command == "kb-search":
            print(
                kb.render(kb.search(" ".join(args.query), max(1, min(args.top_k, 5))))
            )
        elif args.command == "kb-list":
            for article in kb.load_articles():
                print(f"{article.id}  {article.title}")
        elif args.command == "call":
            print(json.dumps(tools.call(args.tool, _payload(args))))
        elif args.command == "customer-say":
            payload = _payload(args)
            tools.record_customer_message(str(payload.pop("text")), payload)
            print(json.dumps({"ok": True}))
        elif args.command == "sim-profile":
            scenario = store.load_scenario()
            print(
                json.dumps(
                    {
                        k: scenario[k]
                        for k in (
                            "ticket",
                            "today",
                            "customer",
                            "orders",
                            "inbound",
                            "sim",
                        )
                    }
                )
            )
        elif args.command == "ledger":
            for entry in store.read_ledger():
                print(json.dumps(entry))
        elif args.command == "grade":
            from helpdesk.grading import grade as grade_ticket

            rubric = json.loads(Path(args.rubric).read_text())
            ledger = store.read_ledger()
            result = grade_ticket(rubric, ledger, store.load_scenario())
            out = Path(args.out)
            out.mkdir(parents=True, exist_ok=True)
            (out / "grade.json").write_text(json.dumps(result, indent=2))
            (out / "ledger.jsonl").write_text(
                "".join(json.dumps(e) + "\n" for e in ledger)
            )
            rewards = {
                "reward": result["reward"],
                "critical_failure": int(result["critical_failure"]),
                "cost_of_service_usd": result["cost_of_service_usd"],
                "reference_cost_usd": result["reference_cost_usd"],
            }
            (out / "reward.json").write_text(json.dumps(rewards))
            print(
                json.dumps(
                    {
                        k: result[k]
                        for k in (
                            "reward",
                            "checks_passed",
                            "checks_total",
                            "cost_of_service_usd",
                        )
                    }
                )
            )
    except HelpdeskError as exc:
        print(f"helpdesk: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
