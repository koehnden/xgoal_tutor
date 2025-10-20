from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

from xgoal_tutor.etl.schema import CREATE_INDEX_STATEMENTS, CREATE_TABLE_STATEMENTS

MutableEvent = MutableMapping[str, Any]


def load_match_events(
    events_path: Path, db_path: Path, *, connection: Optional[sqlite3.Connection] = None
) -> None:
    events = _read_event_file(events_path)
    match_id_override = _derive_match_id_from_path(events_path)
    if match_id_override is not None:
        for event in events:
            if event.get("match_id") is None:
                event["match_id"] = match_id_override

    match_teams = _derive_match_teams(events)

    if connection is None:
        with sqlite3.connect(db_path) as owned_connection:
            _load_into_connection(owned_connection, events, match_teams)
            owned_connection.commit()
    else:
        _load_into_connection(connection, events, match_teams)


def _load_into_connection(
    connection: sqlite3.Connection,
    events: Sequence[MutableEvent],
    match_teams: Mapping[int, Set[int]],
) -> None:
    connection.row_factory = sqlite3.Row
    _initialise_schema(connection)
    _insert_events(connection, events, match_teams)
    _insert_shots_and_freeze_frames(connection, events, match_teams)


def _read_event_file(events_path: Path) -> List[MutableEvent]:
    if not events_path.exists():
        raise FileNotFoundError(f"Event file not found: {events_path}")

    raw = events_path.read_text(encoding="utf-8")
    data = json.loads(raw)
    if not isinstance(data, list):
        raise ValueError("StatsBomb event files must contain a JSON array")

    normalised: List[MutableEvent] = []
    for item in data:
        if not isinstance(item, MutableMapping):
            raise ValueError("Each event must be represented as a JSON object")
        normalised.append(dict(item))
    return normalised


def _derive_match_id_from_path(events_path: Path) -> Optional[int]:
    """Derive a stable match identifier from the filename when absent in data."""

    stem = events_path.stem
    if not stem:
        return None

    candidates = [stem]
    digits_only = "".join(ch for ch in stem if ch.isdigit())
    if digits_only and digits_only != stem:
        candidates.append(digits_only)

    for candidate in candidates:
        try:
            return int(candidate)
        except ValueError:
            continue

    digest = hashlib.sha1(stem.encode("utf-8")).digest()
    hashed = int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)
    return hashed


def _initialise_schema(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA foreign_keys = ON;")

    for statement in CREATE_TABLE_STATEMENTS:
        connection.execute(statement)

    for statement in CREATE_INDEX_STATEMENTS:
        connection.execute(statement)


def _insert_events(
    connection: sqlite3.Connection,
    events: Sequence[MutableEvent],
    match_teams: Mapping[int, Set[int]],
) -> None:
    event_rows = []
    for event in events:
        event_id = str(event.get("id"))
        if not event_id:
            continue
        match_id = _get_int(event.get("match_id"))
        team_id = _get_nested_int(event, ("team", "id"))
        opponent_team_id = _get_nested_int(event, ("opponent", "id"))
        if opponent_team_id is None:
            opponent_team_id = _infer_opponent_team_id(match_id, team_id, match_teams)
        row = (
            event_id,
            match_id,
            team_id,
            _get_nested_int(event, ("player", "id")),
            opponent_team_id,
            _get_int(event.get("possession")),
            _get_int(event.get("period")),
            _get_int(event.get("minute")),
            _get_float(event.get("second")),
            event.get("timestamp"),
            _get_nested_str(event, ("type", "name")),
            _get_nested_str(event, ("play_pattern", "name")),
            _bool_to_int(event.get("under_pressure")),
            _bool_to_int(event.get("counterpress")),
            _get_float(event.get("duration")),
            json.dumps(event),
        )
        event_rows.append(row)

    connection.executemany(
        """
        INSERT OR REPLACE INTO events (
            event_id, match_id, team_id, player_id, opponent_team_id, possession,
            period, minute, second, timestamp, type, play_pattern, under_pressure,
            counterpress, duration, raw_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        event_rows,
    )


SHOT_COLUMNS: Tuple[str, ...] = (
    "shot_id",
    "match_id",
    "team_id",
    "opponent_team_id",
    "player_id",
    "possession",
    "possession_team_id",
    "period",
    "minute",
    "second",
    "timestamp",
    "play_pattern",
    "start_x",
    "start_y",
    "end_x",
    "end_y",
    "end_z",
    "outcome",
    "body_part",
    "technique",
    "shot_type",
    "assist_type",
    "key_pass_id",
    "statsbomb_xg",
    "first_time",
    "one_on_one",
    "open_goal",
    "follows_dribble",
    "deflected",
    "aerial_won",
    "rebound",
    "under_pressure",
    "is_set_piece",
    "is_corner",
    "is_free_kick",
    "is_penalty",
    "is_throw_in",
    "is_kick_off",
    "is_own_goal",
    "freeze_frame_available",
    "freeze_frame_count",
)


def _insert_shots_and_freeze_frames(
    connection: sqlite3.Connection,
    events: Sequence[MutableEvent],
    match_teams: Mapping[int, Set[int]],
) -> None:
    events_by_id = {str(event.get("id")): event for event in events if event.get("id")}

    shot_rows = []
    freeze_frame_rows: List[
        Tuple[str, Optional[int], Optional[str], Optional[str], int, int, Optional[float], Optional[float]]
    ] = []

    for event in events:
        if not _is_shot_event(event):
            continue

        shot_row = _build_shot_row(event, events_by_id, match_teams)
        if shot_row:
            shot_rows.append(shot_row)
            freeze_frame_rows.extend(_build_freeze_frame_rows(event))

    _bulk_insert_shots(connection, shot_rows)
    _bulk_insert_freeze_frames(connection, freeze_frame_rows)


def _is_shot_event(event: MutableEvent) -> bool:
    return _get_nested_str(event, ("type", "name")) == "Shot"


def _build_shot_row(
    event: MutableEvent,
    events_by_id: Mapping[str, MutableEvent],
    match_teams: Mapping[int, Set[int]],
) -> Optional[Tuple[Any, ...]]:
    shot = event.get("shot")
    if not isinstance(shot, Mapping):
        return None

    shot_id_value = event.get("id")
    if not shot_id_value:
        return None
    shot_id = str(shot_id_value)
    location = _extract_location(event.get("location"))
    end_location = _extract_end_location(shot.get("end_location"))
    shot_type = _get_nested_str(shot, ("type", "name"))
    assist_type = _derive_assist_type(shot.get("key_pass_id"), events_by_id)
    outcome = _get_nested_str(shot, ("outcome", "name"))

    match_id = _get_int(event.get("match_id"))
    team_id = _get_nested_int(event, ("team", "id"))
    opponent_team_id = _resolve_opponent_team(event, match_teams)

    freeze_frame_entries = shot.get("freeze_frame")
    freeze_frame_count = _freeze_frame_count(freeze_frame_entries)

    return (
        shot_id,
        match_id,
        team_id,
        opponent_team_id,
        _get_nested_int(event, ("player", "id")),
        _get_int(event.get("possession")),
        _get_nested_int(event, ("possession_team", "id")),
        _get_int(event.get("period")),
        _get_int(event.get("minute")),
        _get_float(event.get("second")),
        event.get("timestamp"),
        _get_nested_str(event, ("play_pattern", "name")),
        location[0],
        location[1],
        end_location[0],
        end_location[1],
        end_location[2],
        outcome,
        _get_nested_str(shot, ("body_part", "name")),
        _get_nested_str(shot, ("technique", "name")),
        shot_type,
        assist_type,
        _get_str(shot.get("key_pass_id")),
        _get_float(shot.get("statsbomb_xg")),
        _bool_to_int(shot.get("first_time")),
        _bool_to_int(shot.get("one_on_one")),
        _bool_to_int(shot.get("open_goal")),
        _bool_to_int(shot.get("follows_dribble")),
        _bool_to_int(shot.get("deflected")),
        _bool_to_int(shot.get("aerial_won")),
        _bool_to_int(shot.get("rebound")),
        _bool_to_int(event.get("under_pressure")),
        _bool_to_int(_is_set_piece(shot_type)),
        _bool_to_int(shot_type == "Corner"),
        _bool_to_int(shot_type == "Free Kick"),
        _bool_to_int(shot_type == "Penalty"),
        _bool_to_int(shot_type == "Throw-in"),
        _bool_to_int(shot_type == "Kick Off"),
        _bool_to_int(outcome == "Own Goal"),
        _bool_to_int(freeze_frame_count > 0),
        freeze_frame_count,
    )


def _resolve_opponent_team(
    event: MutableEvent, match_teams: Mapping[int, Set[int]]
) -> Optional[int]:
    match_id = _get_int(event.get("match_id"))
    team_id = _get_nested_int(event, ("team", "id"))
    opponent_team_id = _get_nested_int(event, ("opponent", "id"))
    if opponent_team_id is not None:
        return opponent_team_id
    return _infer_opponent_team_id(match_id, team_id, match_teams)


def _freeze_frame_count(entries: Any) -> int:
    if isinstance(entries, Sequence) and not isinstance(entries, (str, bytes, bytearray)):
        return len(entries)
    return 0


def _build_freeze_frame_rows(
    event: MutableEvent,
) -> List[Tuple[str, Optional[int], Optional[str], Optional[str], int, int, Optional[float], Optional[float]]]:
    shot = event.get("shot")
    if not isinstance(shot, Mapping):
        return []

    shot_id_value = event.get("id")
    if not shot_id_value:
        return []
    shot_id = str(shot_id_value)

    freeze_frame_entries = shot.get("freeze_frame")
    if not isinstance(freeze_frame_entries, Sequence):
        return []
    rows = []
    for entity in freeze_frame_entries:
        if not isinstance(entity, Mapping):
            continue
        position_name = _get_nested_str(entity, ("position", "name"))
        rows.append(
            (
                shot_id,
                _get_nested_int(entity, ("player", "id")),
                _get_nested_str(entity, ("player", "name")),
                position_name,
                1 if entity.get("teammate") else 0,
                _keeper_flag(entity, position_name),
                *_extract_location(entity.get("location")),
            )
        )
    return rows


def _keeper_flag(entity: Mapping[str, Any], position_name: Optional[str]) -> int:
    if entity.get("keeper") is True:
        return 1
    if position_name == "Goalkeeper":
        return 1
    return 0


def _bulk_insert_shots(connection: sqlite3.Connection, shots: Iterable[Tuple[Any, ...]]) -> None:
    shot_list = list(shots)
    if not shot_list:
        return

    placeholders = ", ".join("?" for _ in SHOT_COLUMNS)
    connection.executemany(
        f"""
        INSERT OR REPLACE INTO shots (
            {', '.join(SHOT_COLUMNS)}
        ) VALUES ({placeholders})
        """,
        shot_list,
    )


def _bulk_insert_freeze_frames(
    connection: sqlite3.Connection,
    freeze_frames: Iterable[
        Tuple[str, Optional[int], Optional[str], Optional[str], int, int, Optional[float], Optional[float]]
    ],
) -> None:
    freeze_frame_list = list(freeze_frames)
    if not freeze_frame_list:
        return

    connection.executemany(
        """
        INSERT INTO freeze_frames (
            shot_id, player_id, player_name, position_name, teammate, keeper, x, y
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        freeze_frame_list,
    )


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

    assist_type = _get_nested_str(pass_data, ("type", "name"))
    if assist_type:
        return assist_type

    if pass_data.get("switch"):
        return "Switch"

    if pass_data.get("through_ball"):
        return "Through Ball"

    if _get_nested_str(pass_data, ("height", "name")) == "High" and pass_data.get("aerial_won"):
        return "Aerial"

    return None


def _is_set_piece(shot_type: Optional[str]) -> bool:
    if not shot_type:
        return False
    return shot_type != "Open Play"


def _extract_location(value: Any) -> Tuple[Optional[float], Optional[float]]:
    if isinstance(value, Sequence):
        coords = list(value)
        coords.extend([None, None])
        return _get_float(coords[0]), _get_float(coords[1])
    return None, None


def _extract_end_location(value: Any) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if isinstance(value, Sequence):
        coords = list(value)
        coords.extend([None, None, None])
        return _get_float(coords[0]), _get_float(coords[1]), _get_float(coords[2])
    return None, None, None


def _get_nested_str(data: Any, path: Sequence[str]) -> Optional[str]:
    current = data
    for key in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return _get_str(current)


def _get_nested_int(data: Any, path: Sequence[str]) -> Optional[int]:
    value = _get_nested_value(data, path)
    return _get_int(value)


def _get_nested_value(data: Any, path: Sequence[str]) -> Any:
    current = data
    for key in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _derive_match_teams(events: Sequence[MutableEvent]) -> Dict[int, Set[int]]:
    match_teams: Dict[int, Set[int]] = {}
    for event in events:
        match_id = _get_int(event.get("match_id"))
        team_id = _get_nested_int(event, ("team", "id"))
        if match_id is None or team_id is None:
            continue
        match_teams.setdefault(match_id, set()).add(team_id)
    return match_teams


def _infer_opponent_team_id(
    match_id: Optional[int], team_id: Optional[int], match_teams: Mapping[int, Set[int]]
) -> Optional[int]:
    if match_id is None or team_id is None:
        return None
    teams = match_teams.get(match_id)
    if not teams or team_id not in teams or len(teams) < 2:
        return None
    opponents = [other for other in teams if other != team_id]
    if len(opponents) == 1:
        return opponents[0]
    return None


def _get_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    return str(value)


def _get_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _get_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bool_to_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(bool(value))
    return None


def main(
    argv: Optional[Sequence[str]] = None,
    loader=load_match_events,
) -> None:
    parser = argparse.ArgumentParser(
        prog="xgoal-tutor-etl",
        description="Load a StatsBomb events JSON file into a SQLite database.",
    )
    parser.add_argument("events_path", type=Path, help="Path to the StatsBomb events JSON file")
    parser.add_argument(
        "database_path", type=Path, help="Path to the SQLite database that will receive the data"
    )
    args = parser.parse_args(argv)

    loader(args.events_path, args.database_path)


__all__ = ["load_match_events", "main"]


if __name__ == "__main__":
    main()
