from __future__ import annotations

import argparse
import contextlib
import logging
import sqlite3
import sys
import tempfile
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Tuple
from urllib.parse import urlparse

from xgoal_tutor.etl import load_match_events
from xgoal_tutor.etl.download_helper import (
    download_github_directory_jsons,
    materialize_github_subtrees,
)


logger = logging.getLogger(__name__)

DEFAULT_STATS_BOMB_ROOT = "https://github.com/statsbomb/open-data/tree/master/data"


def is_url(value: str) -> bool:
    try:
        parts = urlparse(value)
        return parts.scheme in {"http", "https"}
    except Exception:
        return False


def _parse_github_tree_url(url: str) -> Optional[tuple[str, str, str, str]]:
    """Parse a GitHub tree URL into its components."""

    parsed = urlparse(url)
    if parsed.netloc != "github.com":
        return None

    parts = [segment for segment in parsed.path.split("/") if segment]
    if len(parts) >= 4 and parts[2] == "tree":
        owner, repo = parts[0], parts[1]
        ref = parts[3]
        subpath = "/".join(parts[4:]) if len(parts) > 4 else ""
        return owner, repo, ref, subpath
    return None


def iter_event_files(input_arg: str) -> Iterator[Path]:
    """Yield event JSON files from the provided input."""

    gh = _parse_github_tree_url(input_arg)
    if gh:
        owner, repo, ref, subpath = gh
        normalized = subpath.strip("/") if subpath else ""

        if normalized in {"", "data"}:
            raise ValueError(
                "GitHub dataset roots must be materialised before iteration"
            )

        if not normalized:
            subpath = "data/events"
        yield from download_github_directory_jsons(owner, repo, ref, subpath)
        return

    path = Path(input_arg).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path}")

    if path.is_file():
        if path.suffix.lower() != ".json":
            raise ValueError(f"Expected a JSON file, got: {path.name}")
        yield path
        return

    if path.is_dir():
        for file_path in sorted(path.rglob("*.json")):
            yield file_path
        return

    raise ValueError(f"Unsupported input: {input_arg}")


def ingest(
    inputs: List[str],
    db_path: Path,
    stop_on_error: bool = False,
    limit: Optional[int] = None,
    statsbomb_root: Optional[str] = None,
) -> None:
    processed = 0
    attempted = 0
    failures: List[str] = []
    reached_limit = False

    if limit is not None and limit <= 0:
        raise ValueError("limit must be a positive integer when provided")

    effective_inputs = list(inputs)
    if not effective_inputs and statsbomb_root:
        effective_inputs = [statsbomb_root]
    elif not effective_inputs:
        raise ValueError("At least one input path or URL must be provided")

    expanded_inputs, temp_dirs = _expand_inputs(effective_inputs)

    try:
        with sqlite3.connect(db_path) as connection:
            connection.row_factory = sqlite3.Row
            for input_arg in expanded_inputs:
                for events_path in iter_event_files(input_arg):
                    if limit is not None and attempted >= limit:
                        reached_limit = True
                        break

                    attempted += 1
                    try:
                        load_match_events(events_path, db_path, connection=connection)
                        processed += 1
                    except Exception as exc:  # pragma: no cover - defensive logging
                        msg = f"✗ Failed for {events_path}: {exc}"
                        logger.error(msg, file=sys.stderr)
                        failures.append(msg)
                        if stop_on_error:
                            raise
                    finally:
                        if events_path.name.startswith("events_") and events_path.parent == Path(tempfile.gettempdir()):
                            with contextlib.suppress(Exception):
                                events_path.unlink(missing_ok=True)
                if reached_limit:
                    break

        logger.info(f"\nDone. Files processed: {processed}. Database: {db_path}")
        if reached_limit and limit is not None:
            logger.info("Reached requested limit of %s file(s); stopping early.", limit)
        if failures:
            logger.warning("Some files failed:", *failures, sep="\n- ")
    finally:
        for tmp in temp_dirs:
            with contextlib.suppress(Exception):
                tmp.cleanup()


def _expand_inputs(inputs: List[str]) -> Tuple[List[str], List[tempfile.TemporaryDirectory]]:
    expanded: List[str] = []
    temp_dirs: List[tempfile.TemporaryDirectory] = []

    for raw in inputs:
        input_arg = str(raw)
        gh = _parse_github_tree_url(input_arg)
        if gh:
            owner, repo, ref, subpath = gh
            normalized = subpath.strip("/") if subpath else ""

            if normalized in {"", "data"}:
                tmpdir = tempfile.TemporaryDirectory(prefix="statsbomb-data-")
                temp_dirs.append(tmpdir)
                dest_root = Path(tmpdir.name)
                materialize_github_subtrees(
                    owner,
                    repo,
                    ref,
                    ("data/events", "data/lineups"),
                    dest_root,
                )
                expanded.append(str(dest_root / "data" / "events"))
                continue

        path = Path(input_arg).expanduser().resolve()
        if path.is_dir() and (path / "events").is_dir():
            expanded.append(str(path / "events"))
            continue

        expanded.append(input_arg)

    return expanded, temp_dirs


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
        nargs="*",
        help=(
            "Path(s) to a JSON file, a directory containing JSON files, or a GitHub "
            "directory URL. When omitted, --statsbomb-root is used."
        ),
    )
    parser.add_argument(
        "-o",
        "--database",
        default="xgoal-db.sqlite",
        type=Path,
        help="SQLite database path to write to (default: ./xgoal_tutor.sqlite)",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately if any file fails to load (default: continue).",
    )
    parser.add_argument(
        "-n",
        "--limit",
        type=int,
        default=None,
        help=(
            "Process at most N event files. Useful for quick dry-runs of the pipeline "
            "with a small subset of data."
        ),
    )
    parser.add_argument(
        "--statsbomb-root",
        default=DEFAULT_STATS_BOMB_ROOT,
        help=(
            "Base path (local directory or GitHub tree URL) for StatsBomb open-data. "
            "When no inputs are provided, events will be read from <root>/events and "
            "matching lineups from <root>/lineups."
        ),
    )

    args = parser.parse_args(list(argv) if argv is not None else None)

    db_path: Path = args.database.expanduser().resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    ingest(
        args.input,
        db_path,
        stop_on_error=args.stop_on_error,
        limit=args.limit,
        statsbomb_root=args.statsbomb_root,
    )


if __name__ == "__main__":
    main()
