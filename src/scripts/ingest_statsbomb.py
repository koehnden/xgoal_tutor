from __future__ import annotations

import argparse
import contextlib
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import tempfile
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Tuple
from urllib.parse import urlparse

from xgoal_tutor.etl import load_match_events
from xgoal_tutor.etl.download_helper import download_github_directory_jsons


logger = logging.getLogger(__name__)


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


def _default_worker_count() -> int:
    cpu_count = os.cpu_count() or 1
    return max(4, min(32, cpu_count * 2))


def _process_event_file(events_path: Path, db_path: Path) -> Tuple[bool, Path, Optional[Exception]]:
    try:
        load_match_events(events_path, db_path)
        return True, events_path, None
    except Exception as exc:  # pragma: no cover - propagated to caller
        return False, events_path, exc
    finally:
        if events_path.name.startswith("events_") and events_path.parent == Path(tempfile.gettempdir()):
            with contextlib.suppress(Exception):
                events_path.unlink(missing_ok=True)


def ingest(inputs: List[str], db_path: Path, stop_on_error: bool = False) -> None:
    processed = 0
    failures: List[str] = []

    futures = {}
    with ThreadPoolExecutor(max_workers=_default_worker_count()) as executor:
        for input_arg in inputs:
            for events_path in iter_event_files(input_arg):
                future = executor.submit(_process_event_file, events_path, db_path)
                futures[future] = events_path

        for future in as_completed(futures):
            try:
                success, events_path, exc = future.result()
            except Exception as unexpected:
                msg = f"✗ Failed for {futures[future]}: {unexpected}"
                logger.error(msg)
                failures.append(msg)
                if stop_on_error:
                    raise
                continue

            if success:
                processed += 1
                continue

            msg = f"✗ Failed for {events_path}: {exc}"
            logger.error(msg)
            failures.append(msg)
            if stop_on_error and exc is not None:
                raise exc

    logger.info("Done. Files processed: %s. Database: %s", processed, db_path)
    if failures:
        logger.error("Some files failed:")
        for failure in failures:
            logger.error("- %s", failure)


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

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    ingest(args.input, db_path, stop_on_error=args.stop_on_error)


if __name__ == "__main__":
    main()
