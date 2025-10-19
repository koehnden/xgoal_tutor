from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import ParseResult, urlparse
from urllib.request import Request, urlopen
from typing import Any, Iterable, MutableMapping

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
    with urlopen(url) as response:  # nosec: B310 - trusted input validated above
        status = getattr(response, "status", 200)
        if status != 200:
            raise FileNotFoundError(
                f"Could not download StatsBomb events from {url} (HTTP {status})"
            )
        return response.read().decode("utf-8")


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
    owner: str, repo: str, ref: str, directory: str
) -> list[str]:
    api_url = (
        f"https://api.github.com/repos/{owner}/{repo}/contents/{directory}?ref={ref}"
    )
    request = Request(api_url, headers={"Accept": "application/vnd.github.v3+json"})

    with urlopen(request) as response:  # nosec: B310 - trusted input validated above
        status = getattr(response, "status", 200)
        if status != 200:
            raise FileNotFoundError(
                "Could not list StatsBomb events from "
                f"https://github.com/{owner}/{repo}/tree/{ref}/{directory} "
                f"(HTTP {status})"
            )
        listing = json.loads(response.read().decode("utf-8"))

    if not isinstance(listing, list):
        raise ValueError("Unexpected response when listing GitHub directory contents")

    download_urls = sorted(
        item.get("download_url")
        for item in listing
        if isinstance(item, dict)
        and item.get("type") == "file"
        and isinstance(item.get("download_url"), str)
        and item.get("name", "").lower().endswith(".json")
    )

    if not download_urls:
        raise FileNotFoundError(
            "No JSON files found in GitHub directory "
            f"https://github.com/{owner}/{repo}/tree/{ref}/{directory}"
        )

    return [_download_url(url) for url in download_urls]


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
