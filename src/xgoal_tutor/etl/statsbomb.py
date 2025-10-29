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
    teams, players, matches = _collect_reference_data(events, match_teams)

    if connection is None:
        with sqlite3.connect(db_path) as owned_connection:
            _load_into_connection(
                owned_connection,
                events,
                match_teams,
                teams,
                players,
                matches,
            )
            owned_connection.commit()
    else:
        _load_into_connection(connection, events, match_teams, teams, players, matches)


def _load_into_connection(
    connection: sqlite3.Connection,
    events: Sequence[MutableEvent],
    match_teams: Mapping[int, Set[int]],
    teams: Mapping[int, str],
    players: Mapping[int, str],
    matches: Mapping[int, Mapping[str, Any]],
) -> None:
    connection.row_factory = sqlite3.Row
    _initialise_schema(connection)
    _insert_reference_data(connection, teams, players, matches)
    _insert_events(connection, events, match_teams)
    scorelines = _compute_shot_scoreboard(events, matches, match_teams)
    _insert_shots_and_freeze_frames(connection, events, match_teams, scorelines)


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


def _collect_reference_data(
    events: Sequence[MutableEvent], match_teams: Mapping[int, Set[int]]
) -> Tuple[Dict[int, str], Dict[int, str], Dict[int, Dict[str, Any]]]:
    teams: Dict[int, str] = {}
    players: Dict[int, str] = {}
    matches: Dict[int, Dict[str, Any]] = {}

    for event in events:
        _record_team(teams, event.get("team"))
        _record_team(teams, event.get("opponent"))
        _record_team(teams, event.get("possession_team"))

        _record_player(players, event.get("player"))

        pass_data = event.get("pass")
        if isinstance(pass_data, Mapping):
            _record_player(players, pass_data.get("recipient"))

        shot_data = event.get("shot")
        if isinstance(shot_data, Mapping):
            freeze_frame = shot_data.get("freeze_frame")
            if isinstance(freeze_frame, Sequence):
                for frame in freeze_frame:
                    if isinstance(frame, Mapping):
                        _record_player(players, frame.get("player"))

        match_id = _get_int(event.get("match_id"))
        if match_id is None:
            continue

        match_record = matches.setdefault(
            match_id,
            {
                "match_id": match_id,
                "home_team_id": None,
                "away_team_id": None,
                "home_team_name": None,
                "away_team_name": None,
                "competition_name": None,
                "season_name": None,
                "match_date": None,
                "venue": None,
                "match_label": None,
                "_team_names": {},
            },
        )

        competition_name = _get_nested_str(event, ("competition", "name"))
        if competition_name and not match_record["competition_name"]:
            match_record["competition_name"] = competition_name

        season_name = _get_nested_str(event, ("season", "name"))
        if season_name and not match_record["season_name"]:
            match_record["season_name"] = season_name

        match_date = event.get("match_date") or _get_nested_str(event, ("match", "match_date"))
        if match_date and not match_record["match_date"]:
            match_record["match_date"] = _get_str(match_date)

        venue = event.get("venue")
        if venue and not match_record["venue"]:
            if isinstance(venue, Mapping):
                match_record["venue"] = _get_nested_str(venue, ("name",))
            else:
                match_record["venue"] = _get_str(venue)

        stadium_name = _get_nested_str(event, ("stadium", "name"))
        if stadium_name and not match_record["venue"]:
            match_record["venue"] = stadium_name

        for key, team_slot in (("home_team", "home"), ("away_team", "away")):
            team_info = event.get(key)
            if isinstance(team_info, Mapping):
                team_id = _get_int(team_info.get("id"))
                team_name = _get_str(team_info.get("name"))
                if team_slot == "home" and team_id is not None and match_record["home_team_id"] is None:
                    match_record["home_team_id"] = team_id
                    if team_name:
                        match_record["home_team_name"] = team_name
                if team_slot == "away" and team_id is not None and match_record["away_team_id"] is None:
                    match_record["away_team_id"] = team_id
                    if team_name:
                        match_record["away_team_name"] = team_name

        for team_mapping in (event.get("team"), event.get("opponent"), event.get("possession_team")):
            if isinstance(team_mapping, Mapping):
                team_id = _get_int(team_mapping.get("id"))
                if team_id is None:
                    continue
                team_name = _get_str(team_mapping.get("name"))
                if team_name:
                    match_record["_team_names"][team_id] = team_name

    _finalise_match_info(matches, match_teams, teams)

    return teams, players, matches


def _record_team(store: MutableMapping[int, str], data: Any) -> None:
    if not isinstance(data, Mapping):
        return
    team_id = _get_int(data.get("id"))
    name = _get_str(data.get("name"))
    if team_id is None or not name:
        return
    store.setdefault(team_id, name)


def _record_player(store: MutableMapping[int, str], data: Any) -> None:
    if not isinstance(data, Mapping):
        return
    player_id = _get_int(data.get("id"))
    name = _get_str(data.get("name"))
    if player_id is None or not name:
        return
    store.setdefault(player_id, name)


def _finalise_match_info(
    matches: Dict[int, Dict[str, Any]],
    match_teams: Mapping[int, Set[int]],
    teams: Mapping[int, str],
) -> None:
    for match_id, record in matches.items():
        team_names: Dict[int, str] = record.get("_team_names", {})
        for team_id in match_teams.get(match_id, set()):
            if team_id not in team_names and team_id in teams:
                team_names[team_id] = teams[team_id]

        if record.get("home_team_id") is None or record.get("away_team_id") is None:
            ordered = sorted(team_names)
            if ordered:
                if record.get("home_team_id") is None:
                    record["home_team_id"] = ordered[0]
                if record.get("away_team_id") is None and len(ordered) > 1:
                    record["away_team_id"] = ordered[1]

        if record.get("home_team_name") is None and record.get("home_team_id") is not None:
            record["home_team_name"] = team_names.get(record["home_team_id"])

        if record.get("away_team_name") is None and record.get("away_team_id") is not None:
            record["away_team_name"] = team_names.get(record["away_team_id"])

        if not record.get("match_label") and record.get("home_team_name") and record.get("away_team_name"):
            record["match_label"] = f"{record['home_team_name']} vs {record['away_team_name']}"

        record.pop("_team_names", None)


def _insert_reference_data(
    connection: sqlite3.Connection,
    teams: Mapping[int, str],
    players: Mapping[int, str],
    matches: Mapping[int, Mapping[str, Any]],
) -> None:
    _insert_teams(connection, teams)
    _insert_players(connection, players)
    _insert_matches(connection, matches)


def _insert_teams(connection: sqlite3.Connection, teams: Mapping[int, str]) -> None:
    if not teams:
        return
    connection.executemany(
        "INSERT OR REPLACE INTO teams (team_id, team_name) VALUES (?, ?)",
        [(team_id, name) for team_id, name in teams.items()],
    )


def _insert_players(connection: sqlite3.Connection, players: Mapping[int, str]) -> None:
    if not players:
        return
    connection.executemany(
        "INSERT OR REPLACE INTO players (player_id, player_name) VALUES (?, ?)",
        [(player_id, name) for player_id, name in players.items()],
    )


def _insert_matches(connection: sqlite3.Connection, matches: Mapping[int, Mapping[str, Any]]) -> None:
    if not matches:
        return

    rows = [
        (
            match_id,
            match.get("home_team_id"),
            match.get("away_team_id"),
            match.get("home_team_name"),
            match.get("away_team_name"),
            match.get("competition_name"),
            match.get("season_name"),
            match.get("match_date"),
            match.get("venue"),
            match.get("match_label"),
        )
        for match_id, match in matches.items()
    ]

    connection.executemany(
        """
        INSERT OR REPLACE INTO matches (
            match_id,
            home_team_id,
            away_team_id,
            home_team_name,
            away_team_name,
            competition_name,
            season_name,
            match_date,
            venue,
            match_label
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
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


def _compute_shot_scoreboard(
    events: Sequence[MutableEvent],
    matches: Mapping[int, Mapping[str, Any]],
    match_teams: Mapping[int, Set[int]],
) -> Dict[str, Tuple[Optional[int], Optional[int]]]:
    team_scores: Dict[int, Dict[int, int]] = {
        match_id: {team_id: 0 for team_id in team_ids}
        for match_id, team_ids in match_teams.items()
    }
    match_sides: Dict[int, Tuple[Optional[int], Optional[int]]] = {
        match_id: (
            _get_int(match_info.get("home_team_id")),
            _get_int(match_info.get("away_team_id")),
        )
        for match_id, match_info in matches.items()
    }

    scorelines: Dict[str, Tuple[Optional[int], Optional[int]]] = {}

    for event in events:
        if not _is_shot_event(event):
            continue

        match_id = _get_int(event.get("match_id"))
        if match_id is None:
            continue

        shot_id_value = event.get("id")
        if not shot_id_value:
            continue

        shot_id = str(shot_id_value)
        scores = team_scores.setdefault(match_id, {})

        team_id = _get_nested_int(event, ("team", "id"))
        if team_id is not None and team_id not in scores:
            scores[team_id] = 0

        outcome = _get_nested_str(event, ("shot", "outcome", "name"))
        if outcome == "Goal" and team_id is not None:
            scores[team_id] = scores.get(team_id, 0) + 1
        elif outcome == "Own Goal":
            opponent = _resolve_opponent_team(event, match_teams)
            target = opponent if opponent is not None else team_id
            if target is not None:
                scores[target] = scores.get(target, 0) + 1

        home_id, away_id = match_sides.get(match_id, (None, None))
        if (home_id is None or away_id is None) and match_teams.get(match_id):
            ordered = sorted(match_teams[match_id])
            if home_id is None and ordered:
                home_id = ordered[0]
            if away_id is None and len(ordered) > 1:
                away_id = ordered[1]
            match_sides[match_id] = (home_id, away_id)

        home_score = scores.get(home_id, 0) if home_id is not None else None
        away_score = scores.get(away_id, 0) if away_id is not None else None

        scorelines[shot_id] = (home_score, away_score)

    return scorelines


def _insert_shots_and_freeze_frames(
    connection: sqlite3.Connection,
    events: Sequence[MutableEvent],
    match_teams: Mapping[int, Set[int]],
    scorelines: Mapping[str, Tuple[Optional[int], Optional[int]]],
) -> None:
    events_by_id = {str(event.get("id")): event for event in events if event.get("id")}

    shot_rows = []
    freeze_frame_rows: List[
        Tuple[str, Optional[int], Optional[str], Optional[str], int, int, Optional[float], Optional[float]]
    ] = []

    for event in events:
        if not _is_shot_event(event):
            continue

        shot_row = _build_shot_row(event, events_by_id, match_teams, scorelines)
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
    scorelines: Mapping[str, Tuple[Optional[int], Optional[int]]],
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
    score_home, score_away = scorelines.get(shot_id, (None, None))

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
        _bool_to_int(outcome == "Goal"),
        score_home,
        score_away,
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
