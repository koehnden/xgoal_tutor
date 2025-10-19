from __future__ import annotations

import json
from typing import Any, Mapping, MutableMapping, Optional, Sequence, Tuple

from xgoal_tutor.ingest.value_extractors import (
    MutableEvent,
    bool_to_int,
    extract_end_location,
    extract_location,
    get_float,
    get_int,
    get_nested_int,
    get_nested_str,
    get_str,
)

EventRow = Tuple[
    str,
    Optional[int],
    Optional[int],
    Optional[int],
    Optional[int],
    Optional[int],
    Optional[int],
    Optional[int],
    Optional[float],
    Optional[str],
    Optional[str],
    Optional[str],
    Optional[int],
    Optional[int],
    Optional[float],
    str,
]

ShotRow = Tuple[
    str,
    Optional[int],
    Optional[int],
    Optional[int],
    Optional[int],
    Optional[int],
    Optional[int],
    Optional[int],
    Optional[int],
    Optional[float],
    Optional[str],
    Optional[str],
    Optional[float],
    Optional[float],
    Optional[float],
    Optional[float],
    Optional[float],
    Optional[str],
    Optional[str],
    Optional[str],
    Optional[str],
    Optional[str],
    Optional[str],
    Optional[float],
    Optional[int],
    Optional[int],
    Optional[int],
    Optional[int],
    Optional[int],
    Optional[int],
    Optional[int],
    Optional[int],
    Optional[int],
    Optional[int],
    Optional[int],
    Optional[int],
    Optional[int],
    Optional[int],
    Optional[int],
    Optional[int],
]

FreezeFrameRow = Tuple[
    str,
    Optional[int],
    Optional[str],
    Optional[str],
    Optional[int],
    Optional[int],
    Optional[float],
    Optional[float],
]


def build_event_rows(events: Sequence[MutableEvent]) -> list[EventRow]:
    rows: list[EventRow] = []
    for event in events:
        event_id = get_str(event.get("id"))
        if not event_id:
            continue
        rows.append(_build_event_row(event_id, event))
    return rows


def build_shot_rows(
    events: Sequence[MutableEvent],
) -> tuple[list[ShotRow], list[FreezeFrameRow]]:
    rows: list[ShotRow] = []
    freeze_frames: list[FreezeFrameRow] = []
    events_by_id = _events_by_id(events)

    for event in events:
        if get_nested_str(event, ("type", "name")) != "Shot":
            continue

        shot_data = event.get("shot")
        if not isinstance(shot_data, Mapping):
            continue

        shot_id = get_str(event.get("id"))
        if not shot_id:
            continue

        rows.append(_build_shot_row(shot_id, event, shot_data, events_by_id))
        freeze_frames.extend(
            _build_freeze_frame_rows(shot_id, shot_data.get("freeze_frame"))
        )

    return rows, freeze_frames


def _build_event_row(event_id: str, event: MutableEvent) -> EventRow:
    return (
        event_id,
        get_int(event.get("match_id")),
        get_nested_int(event, ("team", "id")),
        get_nested_int(event, ("player", "id")),
        get_nested_int(event, ("opponent", "id")),
        get_int(event.get("possession")),
        get_int(event.get("period")),
        get_int(event.get("minute")),
        get_float(event.get("second")),
        get_str(event.get("timestamp")),
        get_nested_str(event, ("type", "name")),
        get_nested_str(event, ("play_pattern", "name")),
        bool_to_int(event.get("under_pressure")),
        bool_to_int(event.get("counterpress")),
        get_float(event.get("duration")),
        json.dumps(event),
    )


def _build_shot_row(
    shot_id: str,
    event: MutableEvent,
    shot_data: Mapping[str, Any],
    events_by_id: Mapping[str, MutableEvent],
) -> ShotRow:
    location = extract_location(event.get("location"))
    end_location = extract_end_location(shot_data.get("end_location"))
    shot_type = get_nested_str(shot_data, ("type", "name"))
    assist_type = _derive_assist_type(shot_data.get("key_pass_id"), events_by_id)
    outcome = get_nested_str(shot_data, ("outcome", "name"))

    return (
        shot_id,
        get_int(event.get("match_id")),
        get_nested_int(event, ("team", "id")),
        get_nested_int(event, ("opponent", "id")),
        get_nested_int(event, ("player", "id")),
        get_int(event.get("possession")),
        get_nested_int(event, ("possession_team", "id")),
        get_int(event.get("period")),
        get_int(event.get("minute")),
        get_float(event.get("second")),
        get_str(event.get("timestamp")),
        get_nested_str(event, ("play_pattern", "name")),
        location[0],
        location[1],
        end_location[0],
        end_location[1],
        end_location[2],
        outcome,
        get_nested_str(shot_data, ("body_part", "name")),
        get_nested_str(shot_data, ("technique", "name")),
        shot_type,
        assist_type,
        get_str(shot_data.get("key_pass_id")),
        get_float(shot_data.get("statsbomb_xg")),
        bool_to_int(shot_data.get("first_time")),
        bool_to_int(shot_data.get("one_on_one")),
        bool_to_int(shot_data.get("open_goal")),
        bool_to_int(shot_data.get("follows_dribble")),
        bool_to_int(shot_data.get("deflected")),
        bool_to_int(shot_data.get("aerial_won")),
        bool_to_int(shot_data.get("rebound")),
        bool_to_int(event.get("under_pressure")),
        bool_to_int(_is_set_piece(shot_type)),
        bool_to_int(shot_type == "Corner"),
        bool_to_int(shot_type == "Free Kick"),
        bool_to_int(shot_type == "Penalty"),
        bool_to_int(shot_type == "Throw-in"),
        bool_to_int(shot_type == "Kick Off"),
        bool_to_int(outcome == "Own Goal"),
        bool_to_int(bool(shot_data.get("freeze_frame"))),
        _freeze_frame_count(shot_data.get("freeze_frame")),
    )


def _build_freeze_frame_rows(
    shot_id: str, freeze_frames: Any
) -> list[FreezeFrameRow]:
    rows: list[FreezeFrameRow] = []
    if not isinstance(freeze_frames, Sequence):
        return rows

    for entity in freeze_frames:
        if not isinstance(entity, Mapping):
            continue
        location_xy = extract_location(entity.get("location"))
        rows.append(
            (
                shot_id,
                get_nested_int(entity, ("player", "id")),
                get_nested_str(entity, ("player", "name")),
                get_nested_str(entity, ("position", "name")),
                bool_to_int(entity.get("teammate")),
                bool_to_int(entity.get("keeper")),
                location_xy[0],
                location_xy[1],
            )
        )
    return rows


def _events_by_id(events: Sequence[MutableEvent]) -> Mapping[str, MutableEvent]:
    indexed: dict[str, MutableEvent] = {}
    for event in events:
        event_id = get_str(event.get("id"))
        if event_id:
            indexed[event_id] = event
    return indexed


def _derive_assist_type(
    key_pass_id: Any, events_by_id: Mapping[str, MutableEvent]
) -> Optional[str]:
    if not key_pass_id:
        return None

    key_pass = events_by_id.get(str(key_pass_id))
    if not key_pass:
        return None

    pass_data = key_pass.get("pass")
    if not isinstance(pass_data, Mapping):
        return None

    if pass_data.get("cross"):
        return "Cross"

    assist_type = get_nested_str(pass_data, ("type", "name"))
    if assist_type:
        return assist_type

    if pass_data.get("switch"):
        return "Switch"

    if pass_data.get("through_ball"):
        return "Through Ball"

    if get_nested_str(pass_data, ("height", "name")) == "High" and pass_data.get(
        "aerial_won"
    ):
        return "Aerial"

    return None


def _is_set_piece(shot_type: Optional[str]) -> bool:
    if not shot_type:
        return False
    return shot_type != "Open Play"


def _freeze_frame_count(value: Any) -> Optional[int]:
    if isinstance(value, Sequence):
        return len(value)
    return 0


__all__ = [
    "EventRow",
    "ShotRow",
    "FreezeFrameRow",
    "build_event_rows",
    "build_shot_rows",
]
