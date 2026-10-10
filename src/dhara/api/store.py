"""Job and facts store (DuckDB, one file). The API process is the only writer; access is locked.

Why DuckDB: a single embedded file, no server to run, and it can query the per-job GeoParquet
outputs directly. The workload is one writer (this process) and a handful of readers.
"""

import json
import threading
from pathlib import Path
from typing import Any

import duckdb

from dhara.timeutil import now_utc

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
  id VARCHAR PRIMARY KEY,
  kind VARCHAR,              -- 'analysis' | 'flowpath'
  created_utc VARCHAR,
  updated_utc VARCHAR,
  status VARCHAR,            -- queued | running | done | failed
  stage VARCHAR,
  progress DOUBLE,
  request JSON,
  summary JSON,
  error VARCHAR
);
CREATE TABLE IF NOT EXISTS facts (
  job_id VARCHAR, key VARCHAR, value VARCHAR, unit VARCHAR, text VARCHAR, source VARCHAR
);
"""


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._con = duckdb.connect(str(path))
        self._lock = threading.Lock()
        with self._lock:
            self._con.execute(SCHEMA)
            # Jobs interrupted by a restart can never finish: mark them failed honestly.
            self._con.execute(
                "UPDATE jobs SET status='failed', error='interrupted by server restart' "
                "WHERE status IN ('queued','running')"
            )

    def create(self, job_id: str, kind: str, request: dict) -> None:
        t = now_utc()
        with self._lock:
            self._con.execute(
                "INSERT INTO jobs VALUES (?, ?, ?, ?, 'queued', 'queued', 0, ?, NULL, NULL)",
                [job_id, kind, t, t, json.dumps(request)],
            )

    def update(self, job_id: str, **fields: Any) -> None:
        fields["updated_utc"] = now_utc()
        cols = ", ".join(f"{k} = ?" for k in fields)
        vals = [json.dumps(v) if k in ("summary", "request") else v for k, v in fields.items()]
        with self._lock:
            self._con.execute(f"UPDATE jobs SET {cols} WHERE id = ?", [*vals, job_id])

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            cur = self._con.execute("SELECT * FROM jobs WHERE id = ?", [job_id])
            row = cur.fetchone()
            names = [d[0] for d in cur.description]
        return _row(names, row) if row else None

    def recent(self, limit: int = 50) -> list[dict]:
        with self._lock:
            cur = self._con.execute("SELECT * FROM jobs ORDER BY created_utc DESC LIMIT ?", [limit])
            rows = cur.fetchall()
            names = [d[0] for d in cur.description]
        return [_row(names, r) for r in rows]

    def put_facts(self, job_id: str, facts: dict) -> None:
        rows = [
            [job_id, k, json.dumps(v["value"], default=str), v["unit"], v["text"], v["source"]]
            for k, v in facts.items()
        ]
        with self._lock:
            self._con.execute("DELETE FROM facts WHERE job_id = ?", [job_id])
            if rows:
                self._con.executemany("INSERT INTO facts VALUES (?, ?, ?, ?, ?, ?)", rows)

    def facts(self, job_id: str) -> list[dict]:
        with self._lock:
            cur = self._con.execute(
                "SELECT key, value, unit, text, source FROM facts WHERE job_id = ?", [job_id]
            )
            rows = cur.fetchall()
        return [dict(zip(("key", "value", "unit", "text", "source"), r, strict=True)) for r in rows]


def _row(names: list[str], row: tuple) -> dict:
    d = dict(zip(names, row, strict=True))
    for k in ("request", "summary"):
        if isinstance(d.get(k), str):
            d[k] = json.loads(d[k])
    return d
