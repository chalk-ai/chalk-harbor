"""``fraudlab``: the case backend's command line inside the sandbox.

fraudlab call <tool> --b64 <json>      case platform only: run a metered tool
fraudlab ledger                        case platform only: everything done on the case
fraudlab grade --rubric R --out DIR    verifier only: grade the case, write reward.json
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

from fraudlab import store, tools
from fraudlab.store import FraudlabError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fraudlab")
    sub = parser.add_subparsers(dest="command", required=True)
    call = sub.add_parser("call")
    call.add_argument("tool")
    call.add_argument("--b64")
    call.add_argument("--json")
    sub.add_parser("ledger")
    grade = sub.add_parser("grade")
    grade.add_argument("--rubric", required=True)
    grade.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "call":
            raw = (
                base64.b64decode(args.b64).decode() if args.b64 else (args.json or "{}")
            )
            print(json.dumps(tools.call(args.tool, json.loads(raw)), default=str))
        elif args.command == "ledger":
            for entry in store.read_ledger():
                print(json.dumps(entry))
        elif args.command == "grade":
            from fraudlab.grading import grade as grade_case

            ledger = store.read_ledger()
            result = grade_case(json.loads(Path(args.rubric).read_text()), ledger)
            out = Path(args.out)
            out.mkdir(parents=True, exist_ok=True)
            (out / "grade.json").write_text(json.dumps(result, indent=2))
            (out / "ledger.jsonl").write_text(
                "".join(json.dumps(e) + "\n" for e in ledger)
            )
            (out / "reward.json").write_text(
                json.dumps({"reward": result["reward"], "cost_usd": result["cost_usd"]})
            )
            print(
                json.dumps(
                    {
                        k: result[k]
                        for k in ("reward", "decision", "expected_decision", "cost_usd")
                    }
                )
            )
    except FraudlabError as exc:
        print(f"fraudlab: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
