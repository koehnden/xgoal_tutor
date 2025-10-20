from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

MutableEvent = MutableMapping[str, Any]


def load_match_events(events_path: Path, db_path: Path) -> None:
    events = _read_event_file(events_path)
    match_id_override = _derive_match_id_from_path(events_path)
    if match_id_override is not None:
        for event in events:
            if event.get("match_id") is None:
                event["match_id"] = match_id_override

    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        _initialise_schema(connection)
        _insert_events(connection, events)
        _insert_shots_and_freeze_frames(connection, events)
        connection.commit()


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

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS events (
            event_id TEXT PRIMARY KEY,
            match_id INTEGER,
            team_id INTEGER,
            player_id INTEGER,
            opponent_team_id INTEGER,
            possession INTEGER,
            period INTEGER,
            minute INTEGER,
            second REAL,
            timestamp TEXT,
            type TEXT,
            play_pattern TEXT,
            under_pressure INTEGER,
            counterpress INTEGER,
            duration REAL,
            raw_json TEXT NOT NULL
        );
        """
    )

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS shots (
            shot_id TEXT PRIMARY KEY,
            match_id INTEGER,
            team_id INTEGER,
            opponent_team_id INTEGER,
            player_id INTEGER,
            possession INTEGER,
            possession_team_id INTEGER,
            period INTEGER,
            minute INTEGER,
            second REAL,
            timestamp TEXT,
            play_pattern TEXT,
            start_x REAL,
            start_y REAL,
            end_x REAL,
            end_y REAL,
            end_z REAL,
            outcome TEXT,
            body_part TEXT,
            technique TEXT,
            shot_type TEXT,
            assist_type TEXT,
            key_pass_id TEXT,
            statsbomb_xg REAL,
            first_time INTEGER,
            one_on_one INTEGER,
            open_goal INTEGER,
            follows_dribble INTEGER,
            deflected INTEGER,
            aerial_won INTEGER,
            rebound INTEGER,
            under_pressure INTEGER,
            is_set_piece INTEGER,
            is_corner INTEGER,
            is_free_kick INTEGER,
            is_penalty INTEGER,
            is_throw_in INTEGER,
            is_kick_off INTEGER,
            is_own_goal INTEGER,
            freeze_frame_available INTEGER,
            freeze_frame_count INTEGER
        );
        """
    )

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS freeze_frames (
            freeze_frame_id INTEGER PRIMARY KEY AUTOINCREMENT,
            shot_id TEXT NOT NULL,
            player_id INTEGER,
            player_name TEXT,
            position_name TEXT,
            teammate INTEGER NOT NULL,
            keeper INTEGER NOT NULL,
            x REAL,
            y REAL,
            FOREIGN KEY (shot_id) REFERENCES shots(shot_id) ON DELETE CASCADE
        );
        """
    )

    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_events_match ON events(match_id);
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_shots_match ON shots(match_id);
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_freeze_frames_shot ON freeze_frames(shot_id);
        """
    )


def _insert_events(connection: sqlite3.Connection, events: Sequence[MutableEvent]) -> None:
    teams_by_match = _map_teams_by_match(events)
    event_rows = []
    for event in events:
        event_id = str(event.get("id"))
        if not event_id:
            continue
        match_id = _get_int(event.get("match_id"))
        team_id = _get_nested_int(event, ("team", "id"))
        row = (
            event_id,
            match_id,
            team_id,
            _get_nested_int(event, ("player", "id")),
            _infer_opponent_team_id(match_id, team_id, teams_by_match),
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
    connection: sqlite3.Connection, events: Sequence[MutableEvent]
) -> None:
    teams_by_match = _map_teams_by_match(events)
    shots = []
    freeze_frames: List[Tuple[str, Optional[int], Optional[str], Optional[str], int, int, Optional[float], Optional[float]]] = []
    events_by_id = {str(event.get("id")): event for event in events if event.get("id")}

    for event in events:
        if _get_nested_str(event, ("type", "name")) != "Shot":
            continue

        shot = event.get("shot")
        if not isinstance(shot, Mapping):
            continue

        shot_id = str(event.get("id"))
        location = _extract_location(event.get("location"))
        end_location = _extract_end_location(shot.get("end_location"))
        shot_type = _get_nested_str(shot, ("type", "name"))
        assist_type = _derive_assist_type(shot.get("key_pass_id"), events_by_id)
        outcome = _get_nested_str(shot, ("outcome", "name"))

        match_id = _get_int(event.get("match_id"))
        team_id = _get_nested_int(event, ("team", "id"))
        shots.append(
            (
                shot_id,
                match_id,
                team_id,
                _infer_opponent_team_id(match_id, team_id, teams_by_match),
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
                _bool_to_int(bool(shot.get("freeze_frame"))),
                len(shot.get("freeze_frame")) if isinstance(shot.get("freeze_frame"), Sequence) else 0,
            )
        )

        freeze_frame_entries = shot.get("freeze_frame")
        if isinstance(freeze_frame_entries, Sequence):
            for entity in freeze_frame_entries:
                if not isinstance(entity, Mapping):
                    continue
                position_name = _get_nested_str(entity, ("position", "name"))
                keeper_flag = 1 if (entity.get("keeper") is True or position_name == "Goalkeeper") else 0
                teammate_flag = 1 if entity.get("teammate") else 0
                location_xy = _extract_location(entity.get("location"))
                freeze_frames.append(
                    (
                        shot_id,
                        _get_nested_int(entity, ("player", "id")),
                        _get_nested_str(entity, ("player", "name")),
                        position_name,
                        teammate_flag,
                        keeper_flag,
                        location_xy[0],
                        location_xy[1],
                    )
                )

    connection.executemany(
        """
        INSERT OR REPLACE INTO shots (
            shot_id, match_id, team_id, opponent_team_id, player_id, possession,
            possession_team_id, period, minute, second, timestamp, play_pattern,
            start_x, start_y, end_x, end_y, end_z, outcome, body_part, technique,
            shot_type, assist_type, key_pass_id, statsbomb_xg, first_time, one_on_one,
            open_goal, follows_dribble, deflected, aerial_won, rebound,
            under_pressure, is_set_piece, is_corner, is_free_kick, is_penalty,
            is_throw_in, is_kick_off, is_own_goal, freeze_frame_available,
            freeze_frame_count
        ) VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
        """,
        shots,
    )

    connection.executemany(
        """
        INSERT INTO freeze_frames (
            shot_id, player_id, player_name, position_name, teammate, keeper, x, y
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        freeze_frames,
    )


def _map_teams_by_match(events: Sequence[MutableEvent]) -> Dict[int, Set[int]]:
    teams_by_match: Dict[int, Set[int]] = {}
    for event in events:
        match_id = _get_int(event.get("match_id"))
        team_id = _get_nested_int(event, ("team", "id"))
        if match_id is None or team_id is None:
            continue
        teams_by_match.setdefault(match_id, set()).add(team_id)
    return teams_by_match


def _infer_opponent_team_id(
    match_id: Optional[int], team_id: Optional[int], teams_by_match: Mapping[int, Set[int]]
) -> Optional[int]:
    if match_id is None or team_id is None:
        return None
    teams = teams_by_match.get(match_id)
    if not teams or len(teams) != 2:
        return None
    opponents = [candidate for candidate in teams if candidate != team_id]
    if len(opponents) != 1:
        return None
    return opponents[0]


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




def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="xgoal-tutor-etl",
        description="Load a StatsBomb events JSON file into a SQLite database.",
    )
    parser.add_argument("events_path", type=Path, help="Path to the StatsBomb events JSON file")
    parser.add_argument(
        "database_path", type=Path, help="Path to the SQLite database that will receive the data"
    )
    args = parser.parse_args(argv)

    load_match_events(args.events_path, args.database_path)


__all__ = ["load_match_events", "main"]


if __name__ == "__main__":
    main()
