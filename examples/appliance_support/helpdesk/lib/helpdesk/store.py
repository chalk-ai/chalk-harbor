"""Where a scenario's data lives, and the append-only ledger of everything done on its ticket."""

from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
from typing import Any

HOME = Path(os.environ.get("HELPDESK_HOME", "/opt/helpdesk"))
STATE_DIR = Path(os.environ.get("HELPDESK_STATE_DIR", "/var/lib/helpdesk"))
KB_DIR = HOME / "kb"
SCENARIO_DIR = HOME / "scenarios"


class HelpdeskError(Exception):
    """A request the helpdesk refuses; the message is shown to the caller."""


def ticket() -> str:
    value = os.environ.get("HELPDESK_TICKET")
    if not value:
        raise HelpdeskError(
            "HELPDESK_TICKET is not set: this sandbox has no active ticket"
        )
    return value


def load_scenario() -> dict[str, Any]:
    path = SCENARIO_DIR / f"{ticket()}.json"
    try:
        return json.loads(path.read_text())
    except PermissionError as exc:
        raise HelpdeskError("only the support platform can use this command") from exc
    except FileNotFoundError as exc:
        raise HelpdeskError(f"unknown ticket {ticket()}") from exc


def today(scenario: dict[str, Any]) -> dt.date:
    return dt.date.fromisoformat(scenario["today"])


def _ledger_path() -> Path:
    return STATE_DIR / "ledger.jsonl"


def read_ledger() -> list[dict[str, Any]]:
    path = _ledger_path()
    try:
        text = path.read_text()
    except FileNotFoundError:
        return []
    except PermissionError as exc:
        raise HelpdeskError("only the support platform can use this command") from exc
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def append(entry: dict[str, Any]) -> dict[str, Any]:
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        entries = read_ledger()
        record = {
            "seq": len(entries) + 1,
            "at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
            **entry,
        }
        with _ledger_path().open("a") as handle:
            handle.write(json.dumps(record) + "\n")
    except PermissionError as exc:
        raise HelpdeskError("only the support platform can use this command") from exc
    return record
