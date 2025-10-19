from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Sequence

from xgoal_tutor.ingest.reader import MutableEvent, read_statsbomb_events
from xgoal_tutor.ingest.schema import initialise_schema
from xgoal_tutor.ingest.writer import StatsBombSQLiteWriter


def load_statsbomb_events(
    events_path: Path | str, db_path: Path | str, *, show_progress: bool = False
) -> None:
    events = read_statsbomb_events(events_path)
    _write_to_database(events, Path(db_path), show_progress=show_progress)


def _write_to_database(
    events: Sequence[MutableEvent], db_path: Path, *, show_progress: bool = False
) -> None:
    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        initialise_schema(connection)
        StatsBombSQLiteWriter(connection).write(events, show_progress=show_progress)
        connection.commit()


__all__ = ["load_statsbomb_events"]
