"""SQLite trace logger: one row per request."""

import hashlib
import sqlite3
from pathlib import Path

from gpt_efficient.schemas import TraceRow

_COLUMNS = list(TraceRow.model_fields)


def query_hash(query: str) -> str:
    return hashlib.sha256(query.encode("utf-8")).hexdigest()[:16]


class TraceLogger:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            cols = ", ".join(f"{c} {'PRIMARY KEY' if c == 'id' else ''}" for c in _COLUMNS)
            conn.execute(f"CREATE TABLE IF NOT EXISTS traces ({cols})")
            # Trace DBs from earlier milestones lack newer columns; add them in place.
            existing = {r[1] for r in conn.execute("PRAGMA table_info(traces)")}
            for c in _COLUMNS:
                if c not in existing:
                    conn.execute(f"ALTER TABLE traces ADD COLUMN {c}")

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def log(self, row: TraceRow) -> None:
        data = row.model_dump(mode="json")
        placeholders = ", ".join("?" for _ in _COLUMNS)
        with self._connect() as conn:
            conn.execute(
                f"INSERT INTO traces ({', '.join(_COLUMNS)}) VALUES ({placeholders})",
                [data[c] for c in _COLUMNS],
            )

    def get(self, trace_id: str) -> TraceRow | None:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM traces WHERE id = ?", [trace_id]).fetchone()
        return TraceRow.model_validate(dict(row)) if row else None

    def all(self) -> list[TraceRow]:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM traces ORDER BY ts").fetchall()
        return [TraceRow.model_validate(dict(r)) for r in rows]
