from __future__ import annotations

import argparse
import contextlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Iterable, Iterator, List, Optional
from urllib.parse import urlparse
from urllib.request import urlopen, Request

from src.xgoal_tutor.etl import load_match_events


def is_url(s: str) -> bool:
    try:
        parts = urlparse(s)
        return parts.scheme in {"http", "https"}
    except Exception:
        return False


def iter_event_files(input_arg: str) -> Iterator[Path]:
    """
    Yields Path objects for JSON files to process, given:
      - a single file path
      - a directory path (recursively finds *.json)
      - a URL to a single JSON file
    """
    if is_url(input_arg):
        yield _download_to_tempfile(input_arg)
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


def _download_to_tempfile(url: str) -> Path:
    """
    Streams a URL to a NamedTemporaryFile and returns the file path.
    Verifies the payload looks like a JSON array (StatsBomb events format).
    """
    # GitHub tip: you can pass the "raw" URL from the repo. This works for any HTTPS.
    req = Request(url, headers={"User-Agent": "xgoal-tutor/ingest"})
    with contextlib.closing(urlopen(req)) as resp:
        if resp.status != 200:
            raise RuntimeError(f"Failed to download {url} (HTTP {resp.status})")

        # Write to a temp file
        tmp = tempfile.NamedTemporaryFile(prefix="events_", suffix=".json", delete=False)
        tmp_path = Path(tmp.name)
        try:
            chunk = resp.read()  # events files are small enough; if huge, stream in chunks
            tmp.write(chunk)
            tmp.flush()
        finally:
            tmp.close()

    # Quick sanity check: must be a JSON array
    try:
        raw = tmp_path.read_text(encoding="utf-8")
        data = json.loads(raw)
        if not isinstance(data, list):
            raise ValueError("Downloaded file is not a JSON array.")
    except Exception:
        # Clean up the bad temp file
        with contextlib.suppress(Exception):
            tmp_path.unlink(missing_ok=True)
        raise

    return tmp_path


def ingest(inputs: List[str], db_path: Path, stop_on_error: bool = False) -> None:
    processed = 0
    failures: List[str] = []

    for input_arg in inputs:
        for events_path in iter_event_files(input_arg):
            try:
                print(f"→ Loading events from {events_path} into {db_path} …")
                load_match_events(events_path, db_path)
                processed += 1
            except Exception as exc:
                msg = f"✗ Failed for {events_path}: {exc}"
                print(msg, file=sys.stderr)
                failures.append(msg)
                if stop_on_error:
                    raise
            finally:
                # Remove temp downloads only (we recognize our own tempfile pattern)
                if events_path.name.startswith("events_") and events_path.parent == Path(tempfile.gettempdir()):
                    with contextlib.suppress(Exception):
                        events_path.unlink(missing_ok=True)

    print(f"\nDone. Files processed: {processed}. Database: {db_path}")
    if failures:
        print("Some files failed:", *failures, sep="\n- ")


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="xgoal-ingest",
        description="Fetch StatsBomb open-data event files (local dir/file or URL) and load into SQLite."
    )
    parser.add_argument(
        "input",
        nargs="+",
        help=(
            "Path(s) to a JSON file, a directory containing JSON files (searched recursively), "
            "or a direct URL to a single JSON file (e.g. a GitHub raw link)."
        ),
    )
    parser.add_argument(
        "-o", "--database",
        default="xgoal_tutor.sqlite",
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
