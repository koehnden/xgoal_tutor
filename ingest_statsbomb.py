from pathlib import Path

from xgoal_tutor.etl import load_match_events


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        prog="ingest-statsbomb",
        description="Load a StatsBomb events JSON file into a SQLite database.",
    )
    parser.add_argument(
        "events_path",
        help="Path or URL of the StatsBomb events JSON file",
    )
    parser.add_argument(
        "database_path", type=Path, help="Path to the SQLite database that will receive the data"
    )
    args = parser.parse_args()

    load_match_events(args.events_path, args.database_path)


if __name__ == "__main__":
    main()
