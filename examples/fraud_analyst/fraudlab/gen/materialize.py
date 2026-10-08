"""Write the case data into a fraudlab home: the warehouse, the Chalk features, the sealed results.

    python3 materialize.py /opt/fraudlab

Runs at image build time next to a copy of ``population.py``, so the image carries a few KB of
seeded generator instead of megabytes of data (an image build accepts at most 128 KiB of
Dockerfile). The population is seeded, so every build writes the same data, and the labels that
build_tasks.py puts in each task's rubric describe exactly these accounts.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from population import generate


def write_warehouse(path: Path, tables: dict[str, list[dict[str, Any]]]) -> None:
    path.unlink(missing_ok=True)
    connection = sqlite3.connect(path)
    for name, rows in tables.items():
        columns = list(rows[0])
        connection.execute(f"CREATE TABLE {name} ({', '.join(columns)})")
        connection.executemany(
            f"INSERT INTO {name} VALUES ({', '.join('?' for _ in columns)})",
            [[row[c] for c in columns] for row in rows],
        )
    connection.commit()
    connection.execute("VACUUM")
    connection.close()


def materialize(home: Path) -> None:
    population = generate()
    (home / "data").mkdir(parents=True, exist_ok=True)
    write_warehouse(home / "data" / "warehouse.sqlite", population["tables"])
    (home / "data" / "features.json").write_text(
        json.dumps(population["features"], sort_keys=True)
    )
    (home / "sealed").mkdir(exist_ok=True)
    for account_id, results in population["sealed"].items():
        (home / "sealed" / f"{account_id}.json").write_text(
            json.dumps(results, sort_keys=True)
        )


if __name__ == "__main__":
    materialize(Path(sys.argv[1]))
