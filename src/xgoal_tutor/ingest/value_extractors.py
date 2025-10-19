from __future__ import annotations

from typing import Any, Mapping, MutableMapping, Optional, Sequence, Tuple

MutableEvent = MutableMapping[str, Any]


def get_nested_str(data: Any, path: Sequence[str]) -> Optional[str]:
    value = get_nested_value(data, path)
    return get_str(value)


def get_nested_int(data: Any, path: Sequence[str]) -> Optional[int]:
    value = get_nested_value(data, path)
    return get_int(value)


def get_nested_value(data: Any, path: Sequence[str]) -> Any:
    current = data
    for key in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def get_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    return str(value)


def get_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def get_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def bool_to_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(bool(value))
    return None


def extract_location(value: Any) -> Tuple[Optional[float], Optional[float]]:
    if isinstance(value, Sequence):
        coords = list(value)
        coords.extend([None, None])
        return get_float(coords[0]), get_float(coords[1])
    return None, None


def extract_end_location(value: Any) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if isinstance(value, Sequence):
        coords = list(value)
        coords.extend([None, None, None])
        return get_float(coords[0]), get_float(coords[1]), get_float(coords[2])
    return None, None, None


__all__ = [
    "MutableEvent",
    "get_nested_str",
    "get_nested_int",
    "get_nested_value",
    "get_str",
    "get_int",
    "get_float",
    "bool_to_int",
    "extract_location",
    "extract_end_location",
]
