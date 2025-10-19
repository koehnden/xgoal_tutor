from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
import ssl
import warnings
from urllib.error import URLError
from urllib.parse import ParseResult, urlparse
from urllib.request import Request, urlopen
from typing import Any, Iterable, MutableMapping

try:  # pragma: no cover - optional dependency resolution
    import certifi
except ImportError:  # pragma: no cover - fallback if certifi missing
    certifi = None

MutableEvent = MutableMapping[str, Any]


def read_statsbomb_events(events_path: Path | str) -> list[MutableEvent]:
    """Read and normalise StatsBomb events from a file or directory."""
    raw_payloads = _read_resources(events_path)
    events: list[MutableEvent] = []

    for raw in raw_payloads:
        data = _parse_json(raw)
        events.extend(_normalise_events(data))

    return events


def _read_resources(events_path: Path | str) -> list[str]:
    parsed = urlparse(str(events_path))

    if parsed.scheme in {"http", "https"}:
        return list(_read_http_resources(parsed))

    path = Path(events_path)
    if not path.exists():
        raise FileNotFoundError(f"Event file not found: {path}")

    if path.is_dir():
        json_files = sorted(p for p in path.iterdir() if p.suffix.lower() == ".json")
        if not json_files:
            raise FileNotFoundError(
                f"No JSON event files found in directory: {path}")
        return [p.read_text(encoding="utf-8") for p in json_files]

    return [path.read_text(encoding="utf-8")]


def _read_http_resources(parsed: ParseResult) -> Iterable[str]:
    if _is_github_tree(parsed):
        owner, repo, ref, directory = _parse_github_tree(parsed)
        yield from _fetch_github_directory_files(owner, repo, ref, directory)
        return

    url_str = parsed.geturl()
    if not url_str.lower().endswith(".json"):
        raise ValueError(
            "Remote StatsBomb resources must point to a JSON file or a GitHub "
            f"directory. Received: {url_str}",
        )

    yield _download_url(url_str)


def _download_url(url: str) -> str:
    try:
        with _open_url(url) as response:
            status = getattr(response, "status", 200)
            if status != 200:
                raise FileNotFoundError(
                    f"Could not download StatsBomb events from {url} (HTTP {status})"
                )
            return response.read().decode("utf-8")
    except URLError as error:  # pragma: no cover - requires network failure
        raise ConnectionError(
            "Could not download StatsBomb events from "
            f"{url}: {error.reason}"
        ) from error


def _is_github_tree(parsed: ParseResult) -> bool:
    return parsed.netloc.lower() == "github.com" and "/tree/" in parsed.path


def _parse_github_tree(parsed: ParseResult) -> tuple[str, str, str, str]:
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 5 or parts[2] != "tree":
        raise ValueError(
            "GitHub URLs must follow the pattern /<owner>/<repo>/tree/<ref>/<path>"
        )

    owner, repo, _, ref, *directory = parts
    if not directory:
        raise ValueError("GitHub tree URLs must point to a directory")

    return owner, repo, ref, "/".join(directory)


def _fetch_github_directory_files(
    owner: str, repo: str, ref: str, directory: str, *, _root: str | None = None
) -> list[str]:
    root = directory if _root is None else _root
    listing = _list_github_directory(owner, repo, ref, directory)

    entries: list[tuple[str, dict[str, Any]]] = []
    for item in listing:
        if not isinstance(item, dict):
            continue

        item_type = item.get("type")
        path = item.get("path")
        if not isinstance(path, str):
            continue

        if item_type == "file":
            download_url = item.get("download_url")
            name = item.get("name", "")
            if isinstance(download_url, str) and name.lower().endswith(".json"):
                entries.append((path, {"kind": "file", "url": download_url}))
        elif item_type == "dir":
            entries.append((path, {"kind": "dir", "path": path}))

    payloads: list[str] = []
    for _, entry in sorted(entries, key=lambda item: item[0]):
        if entry.get("kind") == "file":
            payloads.append(_download_url(entry["url"]))
        elif entry.get("kind") == "dir":
            payloads.extend(
                _fetch_github_directory_files(
                    owner, repo, ref, entry["path"], _root=root
                )
            )

    if payloads:
        return payloads

    if _root is None:
        raise FileNotFoundError(
            "No JSON files found in GitHub directory "
            f"https://github.com/{owner}/{repo}/tree/{ref}/{directory}"
        )

    return []


def _list_github_directory(owner: str, repo: str, ref: str, directory: str) -> list[Any]:
    api_url = (
        f"https://api.github.com/repos/{owner}/{repo}/contents/{directory}?ref={ref}"
    )
    request = Request(api_url, headers={"Accept": "application/vnd.github.v3+json"})

    try:
        with _open_url(request) as response:
            status = getattr(response, "status", 200)
            if status != 200:
                raise FileNotFoundError(
                    "Could not list StatsBomb events from "
                    f"https://github.com/{owner}/{repo}/tree/{ref}/{directory} "
                    f"(HTTP {status})"
                )
            listing = json.loads(response.read().decode("utf-8"))
    except URLError as error:  # pragma: no cover - requires network failure
        raise ConnectionError(
            "Could not list StatsBomb events from "
            f"https://github.com/{owner}/{repo}/tree/{ref}/{directory}: {error.reason}"
        ) from error

    if not isinstance(listing, list):
        raise ValueError("Unexpected response when listing GitHub directory contents")

    return listing


def _parse_json(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError("StatsBomb event files must contain valid JSON") from error


def _normalise_events(data: Any) -> list[MutableEvent]:
    if not isinstance(data, list):
        raise ValueError("StatsBomb event files must contain a JSON array")

    normalised: list[MutableEvent] = []
    for item in data:
        if not isinstance(item, MutableMapping):
            raise ValueError("Each event must be represented as a JSON object")
        normalised.append(dict(item))
    return normalised


__all__ = ["MutableEvent", "read_statsbomb_events"]


@lru_cache(maxsize=1)
def _ssl_context() -> ssl.SSLContext:
    if certifi is not None:
        return ssl.create_default_context(cafile=certifi.where())
    return ssl.create_default_context()


@lru_cache(maxsize=1)
def _insecure_ssl_context() -> ssl.SSLContext:
    return ssl._create_unverified_context()  # type: ignore[attr-defined]


def _open_url(resource: str | Request):
    try:
        return urlopen(resource, context=_ssl_context())  # nosec: B310
    except URLError as error:
        if _is_certificate_error(error):
            warnings.warn(
                "Falling back to an unverified HTTPS connection to download "
                "StatsBomb events because certificate verification failed. "
                "Install system certificates to restore secure downloads.",
                RuntimeWarning,
                stacklevel=2,
            )
            return urlopen(resource, context=_insecure_ssl_context())  # nosec: B310
        raise


def _is_certificate_error(error: URLError) -> bool:
    reason = getattr(error, "reason", None)
    if isinstance(reason, ssl.SSLCertVerificationError):
        return True
    if isinstance(reason, ssl.SSLError):
        return _contains_certificate_error(str(reason))
    if isinstance(reason, str):
        return _contains_certificate_error(reason)
    return False


def _contains_certificate_error(message: str) -> bool:
    normalised = message.upper().replace(" ", "_")
    return "CERTIFICATE_VERIFY_FAILED" in normalised
