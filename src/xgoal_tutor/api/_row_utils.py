"""Utility helpers for working with SQLite rows."""

from __future__ import annotations

from typing import Any, Dict, Optional

import sqlite3


def row_value(row: sqlite3.Row, key: str) -> Any:
    """Safely extract a column value from a row."""

    try:
        return row[key]
    except (KeyError, IndexError):  # pragma: no cover - defensive guardrail
        return None


def as_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def int_to_bool(value: Any, *, default: Optional[bool] = None) -> Optional[bool]:
    if value is None:
        return default
    try:
        return bool(int(value))
    except (TypeError, ValueError):
        return default


def team_payload(team_id: Any, name: Optional[str], short_name: Optional[str] = None) -> Dict[str, Any]:
    return {
        "id": str(team_id) if team_id is not None else None,
        "name": name,
        "short_name": short_name,
    }


__all__ = [
    "row_value",
    "as_int",
    "as_float",
    "int_to_bool",
    "team_payload",
]
