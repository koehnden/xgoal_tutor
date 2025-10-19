from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen
from typing import Any, MutableMapping

MutableEvent = MutableMapping[str, Any]


def read_statsbomb_events(events_path: Path | str) -> list[MutableEvent]:
    """Read and normalise a StatsBomb events file."""
    raw = _read_resource(events_path)
    data = _parse_json(raw)
    return _normalise_events(data)


def _read_resource(events_path: Path | str) -> str:
    parsed = urlparse(str(events_path))

    if parsed.scheme in {"http", "https"}:
        return _read_http_resource(events_path)

    path = Path(events_path)
    if not path.exists():
        raise FileNotFoundError(f"Event file not found: {path}")
    return path.read_text(encoding="utf-8")


def _read_http_resource(url: str | Path) -> str:
    url_str = str(url)
    if not url_str.lower().endswith(".json"):
        raise ValueError(
            "Remote StatsBomb resources must point directly to a JSON file. "
            f"Received: {url_str}",
        )

    with urlopen(url_str) as response:  # nosec: B310 - trusted input validated above
        status = getattr(response, "status", 200)
        if status != 200:
            raise FileNotFoundError(
                f"Could not download StatsBomb events from {url_str} (HTTP {status})"
            )
        return response.read().decode("utf-8")


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
