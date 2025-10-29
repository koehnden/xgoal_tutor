from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

from xgoal_tutor.etl.schema import CREATE_INDEX_STATEMENTS, CREATE_TABLE_STATEMENTS, SHOT_COLUMNS

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
    teams, players, matches = _collect_dimension_data(events)
    _insert_dimension_tables(connection, teams, players, matches)
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


def _collect_dimension_data(
    events: Sequence[MutableEvent],
) -> Tuple[Dict[int, str], Dict[int, str], Dict[int, Tuple[Optional[int], Optional[int], Optional[str], Optional[str]]]]:
    teams: Dict[int, str] = {}
    players: Dict[int, str] = {}
    match_team_names: Dict[int, Dict[int, str]] = {}
    match_team_order: Dict[int, List[int]] = {}

    for event in events:
        match_id = _get_int(event.get("match_id"))

        def register_team(entity: Any) -> None:
            if not isinstance(entity, Mapping):
                return
            team_id = _get_int(entity.get("id"))
            name = _get_str(entity.get("name"))
            if team_id is None or not name:
                return
            teams.setdefault(team_id, name)
            if match_id is None:
                return
            match_team_names.setdefault(match_id, {})[team_id] = name
            order = match_team_order.setdefault(match_id, [])
            if team_id not in order:
                order.append(team_id)

        def register_player(entity: Any) -> None:
            if not isinstance(entity, Mapping):
                return
            player_id = _get_int(entity.get("id"))
            name = _get_str(entity.get("name"))
            if player_id is None or not name:
                return
            players.setdefault(player_id, name)

        def visit(value: Any, key: Optional[str]) -> None:
            if isinstance(value, Mapping):
                key_lower = key.lower() if isinstance(key, str) else ""
                if key_lower == "opponent" or (key_lower and "team" in key_lower):
                    register_team(value)
                if (
                    key_lower
                    and (
                        "player" in key_lower
                        or key_lower in {"recipient", "replacement"}
                    )
                ):
                    register_player(value)
                for child_key, child_value in value.items():
                    visit(child_value, child_key)
            elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
                for item in value:
                    visit(item, None)

        visit(event, None)

    matches: Dict[int, Tuple[Optional[int], Optional[int], Optional[str], Optional[str]]] = {}
    for match_id, teams_by_id in match_team_names.items():
        if not teams_by_id:
            continue
        order = match_team_order.get(match_id, [])
        home_team_id = order[0] if len(order) >= 1 else None
        away_team_id = order[1] if len(order) >= 2 else None
        matches[match_id] = (
            home_team_id,
            away_team_id,
            teams_by_id.get(home_team_id) if home_team_id is not None else None,
            teams_by_id.get(away_team_id) if away_team_id is not None else None,
        )

    return teams, players, matches


def _insert_dimension_tables(
    connection: sqlite3.Connection,
    teams: Mapping[int, str],
    players: Mapping[int, str],
    matches: Mapping[int, Tuple[Optional[int], Optional[int], Optional[str], Optional[str]]],
) -> None:
    _bulk_upsert(
        connection,
        "teams",
        ("team_id", "team_name"),
        ((team_id, name) for team_id, name in teams.items()),
    )
    _bulk_upsert(
        connection,
        "players",
        ("player_id", "player_name"),
        ((player_id, name) for player_id, name in players.items()),
    )
    _bulk_upsert(
        connection,
        "matches",
        ("match_id", "home_team_id", "away_team_id", "home_team_name", "away_team_name"),
        (
            (
                match_id,
                home_team_id,
                away_team_id,
                home_team_name,
                away_team_name,
            )
            for match_id, (home_team_id, away_team_id, home_team_name, away_team_name) in matches.items()
        ),
    )


def _bulk_upsert(
    connection: sqlite3.Connection,
    table: str,
    columns: Sequence[str],
    rows: Iterable[Sequence[Optional[Any]]],
) -> None:
    row_list = [tuple(row) for row in rows if row and row[0] is not None]
    if not row_list:
        return

    placeholders = ", ".join("?" for _ in columns)
    conflict_column = columns[0]
    assignments = ", ".join(f"{column} = excluded.{column}" for column in columns[1:])
    on_conflict = f" ON CONFLICT({conflict_column}) DO"
    if assignments:
        on_conflict += f" UPDATE SET {assignments}"
    else:
        on_conflict += " NOTHING"

    connection.executemany(
        f"""
        INSERT INTO {table} ({', '.join(columns)})
        VALUES ({placeholders}){on_conflict}
        """,
        row_list,
    )


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
