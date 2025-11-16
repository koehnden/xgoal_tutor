"""Shared helpers for LLM-related modules."""

from __future__ import annotations

from typing import Optional


def as_int(value: Optional[object]) -> Optional[int]:
    """Best-effort conversion to :class:`int`."""

    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def as_bool(value: Optional[object]) -> bool:
    """Best-effort conversion to :class:`bool`."""

    if isinstance(value, bool):
        return value
    if value is None:
        return False
    try:
        return bool(int(value))
    except (TypeError, ValueError):
        text = str(value).strip().lower()
        return text in {"true", "t", "yes", "y"}
