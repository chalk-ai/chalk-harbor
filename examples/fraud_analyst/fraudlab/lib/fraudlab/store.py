"""Where a case's data lives, and the append-only ledger of everything done on it."""

from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
from typing import Any

HOME = Path(os.environ.get("FRAUDLAB_HOME", "/opt/fraudlab"))
STATE_DIR = Path(os.environ.get("FRAUDLAB_STATE_DIR", "/var/lib/fraudlab"))
WAREHOUSE = HOME / "data" / "warehouse.sqlite"
FEATURES = HOME / "data" / "features.json"
SEALED_DIR = HOME / "sealed"


class FraudlabError(Exception):
    """A request the backend refuses; the message is shown to the caller."""


def case_id() -> str:
    value = os.environ.get("FRAUDLAB_CASE")
    if not value:
        raise FraudlabError("FRAUDLAB_CASE is not set: this sandbox has no active case")
    return value


def read_ledger() -> list[dict[str, Any]]:
    try:
        text = (STATE_DIR / "ledger.jsonl").read_text()
    except FileNotFoundError:
        return []
    except PermissionError as exc:
        raise FraudlabError("only the case platform can use this command") from exc
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def append(entry: dict[str, Any]) -> dict[str, Any]:
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        record = {
            "seq": len(read_ledger()) + 1,
            "at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
            **entry,
        }
        with (STATE_DIR / "ledger.jsonl").open("a") as handle:
            handle.write(json.dumps(record) + "\n")
    except PermissionError as exc:
        raise FraudlabError("only the case platform can use this command") from exc
    return record


def sealed(account_id: str) -> dict[str, Any]:
    try:
        return json.loads((SEALED_DIR / f"{account_id}.json").read_text())
    except PermissionError as exc:
        raise FraudlabError("only the case platform can use this command") from exc
    except FileNotFoundError as exc:
        raise FraudlabError(f"unknown account {account_id!r}") from exc
