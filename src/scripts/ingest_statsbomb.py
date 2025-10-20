from __future__ import annotations

import argparse
import contextlib
import sqlite3
import sys
import tempfile
from pathlib import Path
from typing import Iterable, Iterator, List, Optional
from urllib.parse import urlparse

from xgoal_tutor.etl import load_match_events
from xgoal_tutor.etl.download_helper import download_github_directory_jsons


def is_url(s: str) -> bool:
    try:
        parts = urlparse(s)
        return parts.scheme in {"http", "https"}
    except Exception:
        return False

def _parse_github_tree_url(url: str) -> Optional[tuple[str, str, str, str]]:
    """
    Parse: https://github.com/<owner>/<repo>/tree/<ref>/<path...>
    Returns (owner, repo, ref, subpath) or None.
    """
    p = urlparse(url)
    if p.netloc != "github.com":
        return None
    parts = [seg for seg in p.path.split("/") if seg]
    if len(parts) >= 4 and parts[2] == "tree":
        owner, repo = parts[0], parts[1]
        ref = parts[3]
        subpath = "/".join(parts[4:]) if len(parts) > 4 else ""
        return owner, repo, ref, subpath
    return None


def iter_event_files(input_arg: str) -> Iterator[Path]:
    """
    Yields Path objects for JSON files to process, given:
      - a single local file
      - a local directory (recursively finds *.json)
      - a GitHub tree URL (lists all *.json within)
    """
    gh = _parse_github_tree_url(input_arg)
    if gh:
        owner, repo, ref, subpath = gh
        if not subpath:
            subpath = "data/events"
        yield from download_github_directory_jsons(owner, repo, ref, subpath)
        return

    p = Path(input_arg).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Input not found: {p}")

    if p.is_file():
        if p.suffix.lower() != ".json":
            raise ValueError(f"Expected a JSON file, got: {p.name}")
        yield p
        return

    if p.is_dir():
        for fp in sorted(p.rglob("*.json")):
            yield fp
        return

    raise ValueError(f"Unsupported input: {input_arg}")


def ingest(inputs: List[str], db_path: Path, stop_on_error: bool = False) -> None:
    processed = 0
    failures: List[str] = []

    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        for input_arg in inputs:
            for events_path in iter_event_files(input_arg):
                try:
                    load_match_events(events_path, db_path, connection=connection)
                    processed += 1
                except Exception as exc:
                    msg = f"✗ Failed for {events_path}: {exc}"
                    print(msg, file=sys.stderr)
                    failures.append(msg)
                    if stop_on_error:
                        raise
                finally:
                    if events_path.name.startswith("events_") and events_path.parent == Path(tempfile.gettempdir()):
                        with contextlib.suppress(Exception):
                            events_path.unlink(missing_ok=True)

    print(f"\nDone. Files processed: {processed}. Database: {db_path}")
    if failures:
        print("Some files failed:", *failures, sep="\n- ")


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="xgoal-ingest",
        description=(
            "Fetch StatsBomb open-data event files (local dir/file, or GitHub directory URL) "
            "and load into SQLite."
        ),
    )
    parser.add_argument(
        "input",
        nargs="+",
        help=(
            "Path(s) to a JSON file, a directory containing JSON files, "
            "or a GitHub directory URL like "
            "'https://github.com/statsbomb/open-data/tree/master/data/events'."
        ),
    )
    parser.add_argument(
        "-o", "--database",
        default="xgoal-db.sqlite",
        type=Path,
        help="SQLite database path to write to (default: ./xgoal_tutor.sqlite)"
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately if any file fails to load (default: continue)."
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    db_path: Path = args.database.expanduser().resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    ingest(args.input, db_path, stop_on_error=args.stop_on_error)


if __name__ == "__main__":
    main()
