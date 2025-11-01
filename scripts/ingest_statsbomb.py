from __future__ import annotations

import argparse
import contextlib
import io
import logging
import sqlite3
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

from tqdm import tqdm

from xgoal_tutor.etl import load_match_events
from xgoal_tutor.etl.download_helper import _list_events_with_trees_api, _raw_url
from xgoal_tutor.etl.http_helper import _get_bytes


logger = logging.getLogger(__name__)

UNPROCESSED_ROOT = Path(__file__).resolve().parents[1] / "data" / "unprocessed"
EVENTS_DIR = UNPROCESSED_ROOT / "events"
LINEUPS_DIR = UNPROCESSED_ROOT / "lineups"

for directory in (EVENTS_DIR, LINEUPS_DIR):
    directory.mkdir(parents=True, exist_ok=True)


def _relative_to_folder(path: str, folder: str) -> Path:
    parts = Path(path).parts
    for index, part in enumerate(parts):
        if part == folder:
            remainder = parts[index + 1 :]
            if remainder:
                return Path(*remainder)
            break
    # Fallback to the filename when the folder marker is missing or empty
    return Path(parts[-1]) if parts else Path(path)


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


def _download_github_dataset(
    base_url: str, limit: Optional[int]
) -> Tuple[List[Path], List[Path]]:
    gh = _parse_github_tree_url(base_url)
    if not gh:
        raise ValueError(
            "GitHub base URL must follow the format "
            "'https://github.com/<owner>/<repo>/tree/<ref>/<path>'."
        )

    owner, repo, ref, subpath = gh
    subpath = subpath.strip("/")
    events_subpath = "/".join([p for p in [subpath, "events"] if p])

    event_repo_paths = _list_events_with_trees_api(owner, repo, ref, events_subpath)
    if not event_repo_paths:
        return _download_dataset_from_zip(owner, repo, ref, subpath, limit)

    event_repo_paths.sort()
    if limit is not None:
        event_repo_paths = event_repo_paths[:limit]

    local_event_paths: List[Path] = []
    new_event_paths: List[Path] = []

    for rel_path in tqdm(event_repo_paths, desc="Downloading events", unit="file"):
        relative_event = _relative_to_folder(rel_path, "events")
        local_path = EVENTS_DIR / relative_event
        local_event_paths.append(local_path)

        if not local_path.exists():
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_bytes(_get_bytes(_raw_url(owner, repo, ref, rel_path)))
            new_event_paths.append(local_path)

        lineup_rel = rel_path.replace("/events/", "/lineups/", 1)
        if lineup_rel != rel_path:
            lineup_relative = _relative_to_folder(lineup_rel, "lineups")
            lineup_local = LINEUPS_DIR / lineup_relative
            if not lineup_local.exists():
                try:
                    lineup_local.parent.mkdir(parents=True, exist_ok=True)
                    lineup_local.write_bytes(
                        _get_bytes(_raw_url(owner, repo, ref, lineup_rel))
                    )
                except Exception as exc:  # pragma: no cover - network failures
                    logger.debug("No lineup found for %s: %s", rel_path, exc)

    return local_event_paths, new_event_paths


def _download_dataset_from_zip(
    owner: str, repo: str, ref: str, subpath: str, limit: Optional[int]
) -> Tuple[List[Path], List[Path]]:
    zip_url = f"https://codeload.github.com/{owner}/{repo}/zip/refs/heads/{ref}"
    data = _get_bytes(zip_url)

    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        prefix = f"{repo}-{ref}/"
        base_parts = [p for p in subpath.strip("/").split("/") if p]
        events_prefix = prefix + "/".join(base_parts + ["events"]).rstrip("/") + "/"
        lineups_prefix = prefix + "/".join(base_parts + ["lineups"]).rstrip("/") + "/"

        event_names = [
            name
            for name in archive.namelist()
            if name.startswith(events_prefix) and name.endswith(".json")
        ]

        if not event_names:
            raise RuntimeError(
                f"No event files found below '{events_prefix}' in repository ZIP"
            )

        event_names.sort()
        if limit is not None:
            event_names = event_names[:limit]

        lineup_names = {
            name
            for name in archive.namelist()
            if name.startswith(lineups_prefix) and name.endswith(".json")
        }

        local_event_paths: List[Path] = []
        new_event_paths: List[Path] = []

        for event_name in tqdm(event_names, desc="Extracting events", unit="file"):
            relative = event_name[len(prefix) :]
            relative_path = _relative_to_folder(relative, "events")
            local_path = EVENTS_DIR / relative_path
            local_event_paths.append(local_path)

            if not local_path.exists():
                local_path.parent.mkdir(parents=True, exist_ok=True)
                local_path.write_bytes(archive.read(event_name))
                new_event_paths.append(local_path)

            lineup_name = event_name.replace("/events/", "/lineups/", 1)
            if lineup_name in lineup_names:
                lineup_relative = lineup_name[len(prefix) :]
                lineup_path = LINEUPS_DIR / _relative_to_folder(lineup_relative, "lineups")
                lineup_local = lineup_path
                if not lineup_local.exists():
                    lineup_local.parent.mkdir(parents=True, exist_ok=True)
                    lineup_local.write_bytes(archive.read(lineup_name))

    return local_event_paths, new_event_paths


def _collect_local_event_files(base_path: Path, limit: Optional[int]) -> List[Path]:
    if base_path.is_file():
        if base_path.suffix.lower() != ".json":
            raise ValueError(f"Expected a JSON file, got: {base_path.name}")
        return [base_path]

    if not base_path.exists():
        raise FileNotFoundError(f"Input not found: {base_path}")

    events_dir = base_path
    if base_path.is_dir() and base_path.name != "events":
        candidate = base_path / "events"
        if candidate.exists():
            events_dir = candidate

    if not events_dir.is_dir():
        raise ValueError(
            "Local input must be an events directory, a file, or a directory "
            "containing an 'events' subdirectory."
        )

    event_paths = sorted(events_dir.rglob("*.json"))
    if limit is not None:
        event_paths = event_paths[:limit]

    return event_paths


def iter_event_files(base_input: str, limit: Optional[int]) -> Tuple[List[Path], List[Path]]:
    gh = _parse_github_tree_url(base_input)
    if gh:
        return _download_github_dataset(base_input, limit)

    paths = _collect_local_event_files(Path(base_input).expanduser().resolve(), limit)
    return paths, []


def ingest(event_paths: Sequence[Path], db_path: Path, stop_on_error: bool = False) -> None:
    processed = 0
    failures: List[str] = []

    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        for events_path in event_paths:
            try:
                load_match_events(events_path, db_path, connection=connection)
                processed += 1
            except Exception as exc:
                msg = f"✗ Failed for {events_path}: {exc}"
                logger.error(msg, file=sys.stderr)
                failures.append(msg)
                if stop_on_error:
                    raise
            finally:
                if events_path.name.startswith("events_") and events_path.parent == Path(tempfile.gettempdir()):
                    with contextlib.suppress(Exception):
                        events_path.unlink(missing_ok=True)

    logger.info(f"\nDone. Files processed: {processed}. Database: {db_path}")
    if failures:
        logger.warning("Some files failed:", *failures, sep="\n- ")


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="xgoal-ingest",
        description=(
            "Fetch StatsBomb open-data event files (local dir/file, or GitHub directory URL) "
            "and load into SQLite."
        ),
    )
    parser.add_argument(
        "--input",
        default="https://github.com/statsbomb/open-data/tree/master/data",
        help=(
            "Base path containing 'events' and 'lineups' directories. Supports "
            "a local directory or a GitHub directory URL like "
            "'https://github.com/statsbomb/open-data/tree/master/data'."
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
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Optionally limit the number of event files to download and ingest. "
            "Useful for quick test runs."
        ),
    )
    parser.add_argument(
        "--ingest-only-new",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "When enabled (default), ingest only the files downloaded during this run. "
            "Disable to ingest every available event file."
        ),
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be a positive integer")

    db_path: Path = args.database.expanduser().resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    event_paths: List[Path]
    new_event_paths: List[Path]
    try:
        event_paths, new_event_paths = iter_event_files(args.input, args.limit)
    except Exception:
        logger.exception("Failed to resolve input dataset")
        raise

    if args.ingest_only_new:
        ingest_paths = new_event_paths
        description = "newly downloaded"
    else:
        ingest_paths = event_paths
        description = "available"

    if not ingest_paths:
        logger.info("No %s event files to ingest.", description)
        return

    ingest(ingest_paths, db_path, stop_on_error=args.stop_on_error)


if __name__ == "__main__":
    main()
