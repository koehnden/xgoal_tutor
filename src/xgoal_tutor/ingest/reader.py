from __future__ import annotations

import json
from pathlib import Path
from typing import Any, MutableMapping

MutableEvent = MutableMapping[str, Any]


def read_statsbomb_events(events_path: Path) -> list[MutableEvent]:
    """Read and normalise a StatsBomb events file."""
    raw = _read_file(events_path)
    data = _parse_json(raw)
    return _normalise_events(data)


def _read_file(events_path: Path) -> str:
    if not events_path.exists():
        raise FileNotFoundError(f"Event file not found: {events_path}")
    return events_path.read_text(encoding="utf-8")


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
