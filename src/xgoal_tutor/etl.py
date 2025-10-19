from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from .ingest import load_statsbomb_events


def load_match_events(events_path: Path, db_path: Path) -> None:
    load_statsbomb_events(events_path, db_path)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="xgoal-tutor-etl",
        description="Load a StatsBomb events JSON file into a SQLite database.",
    )
    parser.add_argument("events_path", type=Path, help="Path to the StatsBomb events JSON file")
    parser.add_argument(
        "database_path", type=Path, help="Path to the SQLite database that will receive the data"
    )
    args = parser.parse_args(argv)

    load_match_events(args.events_path, args.database_path)


__all__ = ["load_match_events", "main"]


if __name__ == "__main__":
    main()
