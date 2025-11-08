"""Backwards-compatible database helpers expected by the legacy tests."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from fastapi import HTTPException

# Default location of the bundled StatsBomb SQLite database. Tests may monkeypatch
# this path to point at a temporary fixture database.
_DB_PATH = Path(__file__).resolve().parents[3] / "data/statsbomb-db.sqlite"


@contextmanager
def get_db() -> Iterator[sqlite3.Connection]:
    """Yield a SQLite connection configured with dictionary-style rows."""

    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(_DB_PATH)
        connection.row_factory = sqlite3.Row
        yield connection
    except sqlite3.Error as exc:  # pragma: no cover - defensive failure path
        raise HTTPException(status_code=500, detail="Database error") from exc
    finally:
        if connection is not None:
            connection.close()


__all__ = ["get_db", "_DB_PATH"]
