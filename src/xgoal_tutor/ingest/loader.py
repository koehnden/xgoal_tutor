from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Sequence

from .reader import MutableEvent, read_statsbomb_events
from .schema import initialise_schema
from .writer import StatsBombSQLiteWriter


def load_statsbomb_events(events_path: Path, db_path: Path) -> None:
    events = read_statsbomb_events(events_path)
    _write_to_database(events, db_path)


def _write_to_database(events: Sequence[MutableEvent], db_path: Path) -> None:
    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        initialise_schema(connection)
        StatsBombSQLiteWriter(connection).write(events)
        connection.commit()


__all__ = ["load_statsbomb_events"]
