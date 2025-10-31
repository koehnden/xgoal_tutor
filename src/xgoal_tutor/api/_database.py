"""SQLite helpers for the API layer."""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

_DB_ENV_VAR = "XGOAL_TUTOR_DB_PATH"


def _default_db_path() -> Path:
    root = Path(__file__).resolve().parents[2]
    return root / "data" / "xgoal-db.sqlite"


def get_database_path() -> Path:
    """Return the path to the application SQLite database."""

    override = os.getenv(_DB_ENV_VAR)
    if override:
        return Path(override)
    return _default_db_path()


@contextmanager
def get_connection() -> Iterator[sqlite3.Connection]:
    """Yield a SQLite connection with sensible defaults."""

    path = get_database_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)
    try:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        yield conn
    finally:
        conn.close()
