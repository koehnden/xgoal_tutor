"""Utilities for building structured prompts for xGoal explanations."""

from __future__ import annotations

import math
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from xgoal_tutor.prompts import load_template
from xgoal_tutor.llm.utils import as_bool, as_int


@dataclass
class _FreezeFrameEntry:
    """Lightweight representation of a freeze-frame player."""

    player_id: Optional[int]
    player_name: Optional[str]
    position_name: Optional[str]
    teammate: bool
    keeper: bool
    x: Optional[float]
    y: Optional[float]


@contextmanager
def _row_factory(connection: sqlite3.Connection) -> Iterable[None]:
    """Temporarily ensure :class:`sqlite3.Row` objects are returned."""

    original = connection.row_factory
    connection.row_factory = sqlite3.Row
    try:
        yield
    finally:  # pragma: no cover - defensive reset
        connection.row_factory = original


@dataclass
class _MatchMetadata:
    home: str
    score_home: str
    score_away: str
    away: str
    competition: str
    season: str
    home_team_id: Optional[int]
    away_team_id: Optional[int]


@dataclass
class _EventMetadata:
    period: str
    minute: int
    second: int
    play_pattern: str


@dataclass
class _ShooterMetadata:
    name: str
    team_name: str
    position: str
    body_part: str
    technique: str
    start_x: float
    start_y: float


def _row_get(row: sqlite3.Row, key: str, default: Optional[object] = None) -> Optional[object]:
    """Safely access a column from a SQLite row."""

    try:
        if key in row.keys():
            return row[key]
    except AttributeError:  # pragma: no cover - defensive fallback
        pass
    return default




def build_xgoal_prompt(
    connection: sqlite3.Connection,
    shot_id: str,
    *,
    feature_block: Sequence[str],
    context_block: Optional[str] = None,
    template_name: str = "xgoal_offense_prompt.md",
) -> str:
    """Construct the structured prompt for a given shot identifier."""

    with _row_factory(connection):
        shot_row = connection.execute(
            """
            SELECT
                s.*, p.player_name AS shooter_name,
                t.team_name AS shooter_team_name,
                o.team_name AS opponent_team_name
            FROM shots AS s
            LEFT JOIN players AS p ON p.player_id = s.player_id
            LEFT JOIN teams AS t ON t.team_id = s.team_id
            LEFT JOIN teams AS o ON o.team_id = s.opponent_team_id
            WHERE s.shot_id = ?
            """,
            (shot_id,),
        ).fetchone()

    if shot_row is None:
        raise ValueError(f"Unknown shot_id: {shot_id}")

    match_meta = _collect_match_metadata(connection, shot_row)
    event_meta = _collect_event_metadata(shot_row)
    freeze_frame_entries: Optional[List[_FreezeFrameEntry]] = None
    if _table_exists(connection, "freeze_frames"):
        freeze_frame_entries = _load_freeze_frames(connection, shot_row["shot_id"])

    shooter_meta = _collect_shooter_metadata(
        connection, shot_row, freeze_frame_entries or []
    )
    goalkeeper_text, support_line, pressure_line = _format_freeze_frame_blocks(
        freeze_frame_entries or [], shot_row, shooter_meta
    )
    scoreline_before, scoreline_after = _build_scorelines(shot_row, match_meta)
    shot_outcome = _format_shot_outcome(
        scoreline_before,
        scoreline_after,
        match_meta,
        is_goal=_row_get(shot_row, "is_goal"),
    )

    xg_raw = _row_get(shot_row, "statsbomb_xg", 0.0)
    xg = float(xg_raw or 0.0)
    features = [line for line in feature_block if line]
    features = features[:10]
    feature_text = "\n".join(features) if features else "none"

    context_section = context_block.strip() if context_block else ""
    if context_section:
        feature_text = f"{feature_text}\n{context_section}" if feature_text else context_section

    template = load_template(template_name)
    prompt = template.render(
        {
            "home": match_meta.home,
            "score_home": match_meta.score_home,
            "score_away": match_meta.score_away,
            "away": match_meta.away,
            "competition": match_meta.competition,
            "season": match_meta.season,
            "period": event_meta.period,
            "minute": event_meta.minute,
            "second": event_meta.second,
            "play_pattern": event_meta.play_pattern,
            "shooter_name": shooter_meta.name,
            "team_name": shooter_meta.team_name,
            "shooter_position": shooter_meta.position,
            "body_part": shooter_meta.body_part,
            "technique": shooter_meta.technique,
            "start_x": shooter_meta.start_x,
            "start_y": shooter_meta.start_y,
            "gk_line": goalkeeper_text,
            "attack_support_line": support_line,
            "pressure_line": pressure_line,
            "xg": xg,
            "feature_block": feature_text,
            "shot_outcome": shot_outcome,
        }
    )

    return prompt


def _collect_match_metadata(
    connection: sqlite3.Connection, shot_row: sqlite3.Row
) -> _MatchMetadata:
    match_id = shot_row["match_id"]

    home_team: Optional[str] = None
    away_team: Optional[str] = None
    competition: Optional[str] = None
    season: Optional[str] = None
    home_team_id: Optional[int] = None
    away_team_id: Optional[int] = None

    if match_id is not None:
        with _row_factory(connection):
            if _table_exists(connection, "matches"):
                select_fields = [
                    "m.home_team_id",
                    "m.away_team_id",
                    "home.team_name AS home_team",
                    "away.team_name AS away_team",
                ]
                joins = [
                    "LEFT JOIN teams AS home ON home.team_id = m.home_team_id",
                    "LEFT JOIN teams AS away ON away.team_id = m.away_team_id",
                ]

                has_competition_column = _table_has_column(
                    connection, "matches", "competition_name"
                )
                has_season_column = _table_has_column(
                    connection, "matches", "season_name"
                )

                if _table_exists(connection, "competitions"):
                    competition_select = (
                        "COALESCE(comp.competition_name, m.competition_name) AS competition"
                        if has_competition_column
                        else "comp.competition_name AS competition"
                    )
                    select_fields.append(competition_select)
                    joins.append(
                        "LEFT JOIN competitions AS comp ON comp.competition_id = m.competition_id"
                    )
                elif has_competition_column:
                    select_fields.append("m.competition_name AS competition")
                else:
                    select_fields.append("NULL AS competition")

                if _table_exists(connection, "seasons"):
                    season_select = (
                        "COALESCE(seas.season_name, m.season_name) AS season"
                        if has_season_column
                        else "seas.season_name AS season"
                    )
                    select_fields.append(season_select)
                    joins.append(
                        "LEFT JOIN seasons AS seas ON seas.season_id = m.season_id"
                    )
                elif has_season_column:
                    select_fields.append("m.season_name AS season")
                else:
                    select_fields.append("NULL AS season")

                query = " ".join(
                    [
                        "SELECT",
                        ", ".join(select_fields),
                        "FROM matches AS m",
                        *joins,
                        "WHERE m.match_id = ?",
                    ]
                )

                match_row = connection.execute(query, (match_id,)).fetchone()
            else:
                match_row = None
    else:
        match_row = None

    if match_row is not None:
        home_team = match_row["home_team"] or home_team
        away_team = match_row["away_team"] or away_team
        competition = match_row["competition"] or competition
        season = match_row["season"] or season
        home_team_id = as_int(_row_get(match_row, "home_team_id")) or home_team_id
        away_team_id = as_int(_row_get(match_row, "away_team_id")) or away_team_id

    home_team = home_team or _row_get(shot_row, "shooter_team_name") or "unknown"
    away_team = away_team or _row_get(shot_row, "opponent_team_name") or "unknown"

    score_home = _row_get(shot_row, "score_home")
    score_away = _row_get(shot_row, "score_away")

    return _MatchMetadata(
        home=str(home_team),
        score_home=str(score_home) if score_home is not None else "?",
        score_away=str(score_away) if score_away is not None else "?",
        away=str(away_team),
        competition=str(competition) if competition else "unknown",
        season=str(season) if season else "unknown",
        home_team_id=home_team_id,
        away_team_id=away_team_id,
    )


def _collect_event_metadata(shot_row: sqlite3.Row) -> _EventMetadata:
    period_value = _row_get(shot_row, "period")
    period = str(period_value) if period_value is not None else "?"
    minute = int(_row_get(shot_row, "minute", 0) or 0)
    second_value = int(round(float(_row_get(shot_row, "second", 0.0) or 0.0)))
    play_pattern = (_row_get(shot_row, "play_pattern") or "unknown")
    return _EventMetadata(
        period=period,
        minute=minute,
        second=second_value,
        play_pattern=play_pattern,
    )


def _collect_shooter_metadata(
    connection: sqlite3.Connection,
    shot_row: sqlite3.Row,
    freeze_frames: Sequence[_FreezeFrameEntry],
) -> _ShooterMetadata:
    shooter_name = (_row_get(shot_row, "shooter_name") or "unknown")
    team_name = (_row_get(shot_row, "shooter_team_name") or "unknown")
    body_part = (_row_get(shot_row, "body_part") or "unknown")
    technique = (_row_get(shot_row, "technique") or "unknown")
    start_x = float(_row_get(shot_row, "start_x", 0.0) or 0.0)
    start_y = float(_row_get(shot_row, "start_y", 0.0) or 0.0)

    shooter_position: Optional[str] = None
    player_id = _row_get(shot_row, "player_id")
    shooter_name = _row_get(shot_row, "shooter_name")
    shooter_entry = _resolve_shooter_entry(
        freeze_frames, player_id, shooter_name, start_x, start_y
    )

    if shooter_entry is not None:
        if shooter_entry.position_name:
            shooter_position = shooter_entry.position_name
        if shooter_entry.x is not None:
            start_x = float(shooter_entry.x)
        if shooter_entry.y is not None:
            start_y = float(shooter_entry.y)

    if not shooter_position:
        shooter_position = (
            _lookup_lineup_position(connection, shot_row, player_id) or "unknown"
        )

    return _ShooterMetadata(
        name=shooter_name,
        team_name=team_name,
        position=shooter_position,
        body_part=body_part,
        technique=technique,
        start_x=start_x,
        start_y=start_y,
    )


def _resolve_shooter_entry(
    entries: Sequence[_FreezeFrameEntry],
    player_id: Optional[int],
    shooter_name: Optional[str],
    fallback_x: float,
    fallback_y: float,
) -> Optional[_FreezeFrameEntry]:
    if player_id is not None:
        for entry in entries:
            if entry.player_id == player_id:
                return entry

    if shooter_name:
        for entry in entries:
            if entry.player_name and entry.player_name == shooter_name:
                return entry

    closest_entry: Optional[_FreezeFrameEntry] = None
    closest_distance = float("inf")
    for entry in entries:
        if not entry.teammate or entry.keeper:
            continue
        if entry.x is None or entry.y is None:
            continue
        distance = math.hypot(entry.x - fallback_x, entry.y - fallback_y)
        if distance < closest_distance:
            closest_distance = distance
            closest_entry = entry

    return closest_entry


def _lookup_lineup_position(
    connection: sqlite3.Connection,
    shot_row: sqlite3.Row,
    player_id: Optional[int],
) -> Optional[str]:
    match_id = _row_get(shot_row, "match_id")
    if player_id is None or match_id is None:
        return None
    if not _table_exists(connection, "match_lineups"):
        return None

    with _row_factory(connection):
        row = connection.execute(
            """
            SELECT position_name
            FROM match_lineups
            WHERE match_id = ? AND player_id = ? AND position_name IS NOT NULL
            ORDER BY is_starter DESC, sort_order ASC
            LIMIT 1
            """,
            (match_id, player_id),
        ).fetchone()

    if row is None:
        return None

    position_name = row["position_name"]
    return str(position_name) if position_name else None


def _format_freeze_frame_blocks(
    entries: Sequence[_FreezeFrameEntry],
    shot_row: sqlite3.Row,
    shooter_meta: _ShooterMetadata,
) -> Tuple[str, str, str]:
    if not entries:
        return ("unknown", "none", "none")

    goalkeeper_text = _build_goalkeeper_text(entries)
    shooter_id = _row_get(shot_row, "player_id")
    support_line = _build_support_line(
        entries, shooter_id, shooter_meta.start_x, shooter_meta.start_y
    )
    pressure_line = _build_pressure_line(entries, shooter_meta.start_x, shooter_meta.start_y)

    return goalkeeper_text, support_line, pressure_line


def _load_freeze_frames(
    connection: sqlite3.Connection, shot_id: str
) -> List[_FreezeFrameEntry]:
    with _row_factory(connection):
        rows = connection.execute(
            """
            SELECT player_id, player_name, position_name, teammate, keeper, x, y
            FROM freeze_frames
            WHERE shot_id = ?
            """,
            (shot_id,),
        ).fetchall()

    return [
        _FreezeFrameEntry(
            player_id=row["player_id"],
            player_name=row["player_name"],
            position_name=row["position_name"],
            teammate=bool(row["teammate"]),
            keeper=bool(row["keeper"]),
            x=row["x"],
            y=row["y"],
        )
        for row in rows
    ]


def _build_goalkeeper_text(entries: Sequence[_FreezeFrameEntry]) -> str:
    keepers = [entry for entry in entries if entry.keeper]
    if not keepers:
        return "unknown"

    name = next((k.player_name for k in keepers if k.player_name), "unknown")

    xs = [k.x for k in keepers if k.x is not None]
    ys = [k.y for k in keepers if k.y is not None]
    if not xs or not ys:
        return "unknown"

    avg_x = sum(xs) / len(xs)
    avg_y = sum(ys) / len(ys)
    depth = 120 - avg_x
    offset = avg_y - 40

    return (
        f"{name} at x={avg_x:.1f}, y={avg_y:.1f}\n"
        f"    (depth={depth:.1f}m from goal-line, offset={offset:.1f}m)"
    )


def _build_support_line(
    entries: Sequence[_FreezeFrameEntry],
    shooter_id: Optional[int],
    shooter_x: float,
    shooter_y: float,
) -> str:
    teammates: List[Tuple[str, float, float]] = []
    for entry in entries:
        if not entry.teammate or entry.player_id == shooter_id:
            continue
        if entry.x is None or entry.y is None:
            continue
        distance = math.hypot(entry.x - shooter_x, entry.y - shooter_y)
        if distance <= 18:
            bearing = _bearing_to_goal(shooter_x, shooter_y, entry.x, entry.y)
            name = entry.player_name or "unknown"
            teammates.append((name, distance, bearing))

    teammates.sort(key=lambda item: item[1])
    formatted = [
        f"{name}({dist:.1f}m @ {bearing:+.0f}°)" for name, dist, bearing in teammates[:3]
    ]

    return ", ".join(formatted) if formatted else "none"


def _build_pressure_line(
    entries: Sequence[_FreezeFrameEntry], shooter_x: float, shooter_y: float
) -> str:
    defenders_close: List[Tuple[str, float, float]] = []
    defenders_cone: List[Tuple[str, float, float]] = []

    for entry in entries:
        if entry.teammate or entry.keeper:
            continue
        if entry.x is None or entry.y is None:
            continue

        distance = math.hypot(entry.x - shooter_x, entry.y - shooter_y)
        bearing = _bearing_to_goal(shooter_x, shooter_y, entry.x, entry.y)
        name = entry.player_name or "unknown"

        if distance <= 15:
            defenders_close.append((name, distance, bearing))
        elif _in_shot_cone(shooter_x, shooter_y, entry.x, entry.y):
            defenders_cone.append((name, distance, bearing))

    defenders_close.sort(key=lambda item: item[1])
    defenders_cone.sort(key=lambda item: item[1])

    selected: List[Tuple[str, float, float]] = []
    seen: set[Tuple[str, float]] = set()

    for group in (defenders_close, defenders_cone):
        for name, dist, bearing in group:
            key = (name, dist)
            if key in seen:
                continue
            selected.append((name, dist, bearing))
            seen.add(key)
            if len(selected) >= 6:
                break
        if len(selected) >= 6:
            break

    formatted = [
        f"{name}({dist:.1f}m @ {bearing:+.0f}°)" for name, dist, bearing in selected[:4]
    ]

    return ", ".join(formatted) if formatted else "none"


def _build_scorelines(
    shot_row: sqlite3.Row, match_meta: _MatchMetadata
) -> Tuple[Optional[Dict[str, int]], Optional[Dict[str, int]]]:
    score_home = as_int(_row_get(shot_row, "score_home"))
    score_away = as_int(_row_get(shot_row, "score_away"))
    if score_home is None or score_away is None:
        return None, None

    after = {"home": score_home, "away": score_away}
    before: Optional[Dict[str, int]] = {"home": score_home, "away": score_away}

    if not as_bool(_row_get(shot_row, "is_goal")):
        return before, after

    if match_meta.home_team_id is None or match_meta.away_team_id is None:
        return None, after

    scoring_team = _resolve_scoring_team(shot_row, match_meta)
    if scoring_team == match_meta.home_team_id and before is not None:
        before["home"] = max(before["home"] - 1, 0)
    elif scoring_team == match_meta.away_team_id and before is not None:
        before["away"] = max(before["away"] - 1, 0)
    else:
        before = None

    return before, after


def _resolve_scoring_team(
    shot_row: sqlite3.Row, match_meta: _MatchMetadata
) -> Optional[int]:
    team_id = as_int(_row_get(shot_row, "team_id"))
    opponent_team_id = as_int(_row_get(shot_row, "opponent_team_id"))

    if as_bool(_row_get(shot_row, "is_own_goal")):
        if opponent_team_id is not None:
            return opponent_team_id
        home_id = match_meta.home_team_id
        away_id = match_meta.away_team_id
        if team_id == home_id:
            return away_id
        if team_id == away_id:
            return home_id
        return opponent_team_id

    return team_id


def _format_shot_outcome(
    scoreline_before: Optional[Dict[str, int]],
    scoreline_after: Optional[Dict[str, int]],
    match_meta: _MatchMetadata,
    *,
    is_goal: Optional[object],
) -> str:
    goal_known = is_goal is not None
    goal_flag = as_bool(is_goal)
    score_text = _format_scoreline(scoreline_after)

    if scoreline_before is None or scoreline_after is None:
        if goal_flag:
            return f"Goal ({score_text})" if score_text else "Goal"
        if goal_known:
            return f"No goal ({score_text})" if score_text else "No goal"
        return f"Outcome unknown ({score_text})" if score_text else "Outcome unknown"

    home_before = scoreline_before.get("home")
    home_after = scoreline_after.get("home")
    away_before = scoreline_before.get("away")
    away_after = scoreline_after.get("away")

    values = (home_before, home_after, away_before, away_after)
    if any(value is None for value in values):
        if goal_flag:
            return f"Goal ({score_text})" if score_text else "Goal"
        if goal_known:
            return f"No goal ({score_text})" if score_text else "No goal"
        return f"Outcome unknown ({score_text})" if score_text else "Outcome unknown"

    home_delta = home_after - home_before
    away_delta = away_after - away_before

    if home_delta > 0 and away_delta == 0:
        team = match_meta.home or "home team"
        return f"Goal for {team} ({score_text})" if score_text else f"Goal for {team}"
    if away_delta > 0 and home_delta == 0:
        team = match_meta.away or "away team"
        return f"Goal for {team} ({score_text})" if score_text else f"Goal for {team}"
    if home_delta == 0 and away_delta == 0:
        return f"No goal ({score_text})" if score_text else "No goal"

    change = "Score changed"
    return f"{change} ({score_text})" if score_text else change


def _format_scoreline(scoreline: Optional[Dict[str, int]]) -> str:
    if not scoreline:
        return ""
    home = scoreline.get("home")
    away = scoreline.get("away")
    if home is None or away is None:
        return ""
    return f"{home}–{away}"


def _bearing_to_goal(sx: float, sy: float, px: float, py: float) -> float:
    goal_vector = (120.0 - sx, 40.0 - sy)
    player_vector = (px - sx, py - sy)

    if abs(goal_vector[0]) < 1e-6 and abs(goal_vector[1]) < 1e-6:
        goal_angle = 0.0
    else:
        goal_angle = math.degrees(math.atan2(goal_vector[1], goal_vector[0]))

    if abs(player_vector[0]) < 1e-6 and abs(player_vector[1]) < 1e-6:
        player_angle = goal_angle
    else:
        player_angle = math.degrees(math.atan2(player_vector[1], player_vector[0]))

    bearing = player_angle - goal_angle
    while bearing <= -180:
        bearing += 360
    while bearing > 180:
        bearing -= 360
    return bearing


def _in_shot_cone(sx: float, sy: float, px: float, py: float) -> bool:
    a = (sx, sy)
    b = (120.0, 36.0)
    c = (120.0, 44.0)

    v0 = (c[0] - a[0], c[1] - a[1])
    v1 = (b[0] - a[0], b[1] - a[1])
    v2 = (px - a[0], py - a[1])

    dot00 = v0[0] * v0[0] + v0[1] * v0[1]
    dot01 = v0[0] * v1[0] + v0[1] * v1[1]
    dot02 = v0[0] * v2[0] + v0[1] * v2[1]
    dot11 = v1[0] * v1[0] + v1[1] * v1[1]
    dot12 = v1[0] * v2[0] + v1[1] * v2[1]

    denom = dot00 * dot11 - dot01 * dot01
    if denom == 0:
        return False

    inv_denom = 1 / denom
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom

    return u >= 0 and v >= 0 and (u + v) <= 1


def _table_exists(connection: sqlite3.Connection, table_name: str) -> bool:
    cursor = connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,)
    )
    return cursor.fetchone() is not None


def _table_has_column(
    connection: sqlite3.Connection, table_name: str, column_name: str
) -> bool:
    try:
        cursor = connection.execute(f"PRAGMA table_info({table_name})")
    except sqlite3.Error:  # pragma: no cover - defensive guard
        return False
    return any(row[1] == column_name for row in cursor.fetchall())

