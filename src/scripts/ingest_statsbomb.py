from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import shutil
import ssl
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Iterable, Iterator, List, Optional
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from tqdm import tqdm

from xgoal_tutor.etl import load_match_events


# ---------- HTTP helpers (simple & robust) ----------

def _ssl_context() -> ssl.SSLContext:
    """Prefer certifi’s CA bundle; fall back to system defaults."""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()

def _headers(extra: Optional[dict] = None) -> dict:
    hdrs = {"User-Agent": "xgoal-tutor/ingest"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        hdrs["Authorization"] = f"Bearer {token}"
    if extra:
        hdrs.update(extra)
    return hdrs

def _get_bytes(url: str) -> bytes:
    """GET bytes with urllib; on error, try curl once (helps on macOS)."""
    req = Request(url, headers=_headers())
    try:
        with contextlib.closing(urlopen(req, context=_ssl_context(), timeout=30)) as resp:
            if resp.status != 200:
                raise RuntimeError(f"HTTP {resp.status} for {url}")
            return resp.read()
    except Exception:
        curl = shutil.which("curl")
        if curl:
            try:
                return subprocess.check_output(
                    [curl, "-fsSL", "--retry", "3", url],
                    stderr=subprocess.STDOUT,
                )
            except subprocess.CalledProcessError:
                pass
        raise

def _get_json(url: str):
    data = _get_bytes(url)
    return json.loads(data.decode("utf-8"))


# ---------- Input detection ----------

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


# ---------- Download helpers ----------

def _download_to_tempfile(url: str) -> Path:
    raw = _get_bytes(url)
    tmp = tempfile.NamedTemporaryFile(prefix="events_", suffix=".json", delete=False)
    tmp_path = Path(tmp.name)
    try:
        tmp.write(raw)
        tmp.flush()
    finally:
        tmp.close()
    data = json.loads(tmp_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array from {url}")
    return tmp_path

def _raw_url(owner: str, repo: str, ref: str, path: str) -> str:
    return f"https://raw.githubusercontent.com/{owner}/{repo}/{ref}/{path.lstrip('/')}"

def _list_events_with_trees_api(owner: str, repo: str, ref: str, subpath: str) -> list[str]:
    """
    Git Trees API (recursive) → list all blob paths; filter to subpath/*.json
    GET /repos/{owner}/{repo}/git/trees/{ref}?recursive=1
    """
    api = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{ref}?recursive=1"
    tree = _get_json(api)
    if not isinstance(tree, dict) or "tree" not in tree:
        raise RuntimeError(f"Unexpected response from Git Trees API: {api}")

    base = subpath.strip("/").rstrip("/") + "/"
    paths = [
        e["path"]
        for e in tree.get("tree", [])
        if e.get("type") == "blob"
        and e.get("path", "").startswith(base)
        and e["path"].endswith(".json")
    ]

    if tree.get("truncated"):
        return []
    return paths

def _download_github_directory_jsons(owner: str, repo: str, ref: str, subpath: str) -> Iterator[Path]:
    """
    Preferred: list with Trees API, then sequentially download each raw JSON.
    Fallback: download repo ZIP and extract only subpath/*.json.
    """
    try:
        paths = _list_events_with_trees_api(owner, repo, ref, subpath)
        if paths:
            for rel in tqdm(paths, desc="Downloading events", unit="file"):
                yield _download_to_tempfile(_raw_url(owner, repo, ref, rel))
            return
    except Exception as e:
        print(f"[warn] Trees API listing failed; trying ZIP fallback: {e}", file=sys.stderr)

    zip_url = f"https://codeload.github.com/{owner}/{repo}/zip/refs/heads/{ref}"
    data = _get_bytes(zip_url)
    zf = zipfile.ZipFile(io.BytesIO(data))
    prefix = f"{repo}-{ref}/" + subpath.strip("/").rstrip("/") + "/"

    names = [n for n in zf.namelist() if n.startswith(prefix) and n.endswith(".json")]
    for name in tqdm(names, desc="Extracting events", unit="file"):
        tmp = tempfile.NamedTemporaryFile(prefix="events_", suffix=".json", delete=False)
        with tmp:
            tmp.write(zf.read(name))
        yield Path(tmp.name)


# ---------- File iterator for all input modes ----------

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
        yield from _download_github_directory_jsons(owner, repo, ref, subpath)
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


# ---------- Main ingest loop (sequential, clear) ----------

def ingest(inputs: List[str], db_path: Path, stop_on_error: bool = False) -> None:
    processed = 0
    failures: List[str] = []

    for input_arg in inputs:
        for events_path in iter_event_files(input_arg):
            try:
                load_match_events(events_path, db_path)
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


# ---------- CLI ----------

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
