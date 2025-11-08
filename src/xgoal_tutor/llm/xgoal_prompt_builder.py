"""Utilities for building structured prompts for xGoal explanations."""

from __future__ import annotations

import math
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

from xgoal_tutor.prompts import load_template


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


def build_xgoal_prompt(
    connection: sqlite3.Connection,
    shot_id: str,
    *,
    feature_block: Sequence[str],
    context_block: Optional[str] = None,
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
    shooter_meta = _collect_shooter_metadata(connection, shot_row)
    goalkeeper_text, support_line, pressure_line = _format_freeze_frame_blocks(
        connection, shot_row
    )

    xg = float(shot_row["statsbomb_xg"] or 0.0)
    features = [line for line in feature_block if line]
    features = features[:5]
    feature_text = "\n".join(features) if features else "none"

    context_section = context_block.strip() if context_block else ""
    if context_section:
        feature_text = f"{feature_text}\n{context_section}" if feature_text else context_section

    template = load_template("xgoal_prompt.txt")
    prompt = template.format(
        home=match_meta.home,
        score_home=match_meta.score_home,
        score_away=match_meta.score_away,
        away=match_meta.away,
        competition=match_meta.competition,
        season=match_meta.season,
        period=event_meta.period,
        minute=event_meta.minute,
        second=event_meta.second,
        play_pattern=event_meta.play_pattern,
        shooter_name=shooter_meta.name,
        team_name=shooter_meta.team_name,
        shooter_position=shooter_meta.position,
        body_part=shooter_meta.body_part,
        technique=shooter_meta.technique,
        start_x=shooter_meta.start_x,
        start_y=shooter_meta.start_y,
        gk_line=goalkeeper_text,
        attack_support_line=support_line,
        pressure_line=pressure_line,
        xg=xg,
        feature_block=feature_text,
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

    if match_id is not None:
        with _row_factory(connection):
            if _table_exists(connection, "matches"):
                match_row = connection.execute(
                    """
                    SELECT m.home_team_id, m.away_team_id,
                           home.team_name AS home_team,
                           away.team_name AS away_team,
                           comp.competition_name AS competition,
                           seas.season_name AS season
                    FROM matches AS m
                    LEFT JOIN teams AS home ON home.team_id = m.home_team_id
                    LEFT JOIN teams AS away ON away.team_id = m.away_team_id
                    LEFT JOIN competitions AS comp ON comp.competition_id = m.competition_id
                    LEFT JOIN seasons AS seas ON seas.season_id = m.season_id
                    WHERE m.match_id = ?
                    """,
                    (match_id,),
                ).fetchone()
            else:
                match_row = None
    else:
        match_row = None

    if match_row is not None:
        home_team = match_row["home_team"] or home_team
        away_team = match_row["away_team"] or away_team
        competition = match_row["competition"] or competition
        season = match_row["season"] or season

    home_team = home_team or shot_row["shooter_team_name"] or "unknown"
    away_team = away_team or shot_row["opponent_team_name"] or "unknown"

    score_home = shot_row["score_home"]
    score_away = shot_row["score_away"]

    return _MatchMetadata(
        home=str(home_team),
        score_home=str(score_home) if score_home is not None else "?",
        score_away=str(score_away) if score_away is not None else "?",
        away=str(away_team),
        competition=str(competition) if competition else "unknown",
        season=str(season) if season else "unknown",
    )


def _collect_event_metadata(shot_row: sqlite3.Row) -> _EventMetadata:
    period = str(shot_row["period"]) if shot_row["period"] is not None else "?"
    minute = int(shot_row["minute"] or 0)
    second_value = int(round(float(shot_row["second"] or 0.0)))
    play_pattern = shot_row["play_pattern"] or "unknown"
    return _EventMetadata(
        period=period,
        minute=minute,
        second=second_value,
        play_pattern=play_pattern,
    )


def _collect_shooter_metadata(
    connection: sqlite3.Connection, shot_row: sqlite3.Row
) -> _ShooterMetadata:
    shooter_name = shot_row["shooter_name"] or "unknown"
    team_name = shot_row["shooter_team_name"] or "unknown"
    body_part = shot_row["body_part"] or "unknown"
    technique = shot_row["technique"] or "unknown"
    start_x = float(shot_row["start_x"] or 0.0)
    start_y = float(shot_row["start_y"] or 0.0)

    shooter_position = "unknown"
    player_id = shot_row["player_id"]
    with _row_factory(connection):
        if player_id is not None and _table_exists(connection, "freeze_frames"):
            row = connection.execute(
                """
                SELECT position_name
                FROM freeze_frames
                WHERE shot_id = ? AND teammate = 1 AND player_id = ? AND position_name IS NOT NULL
                ORDER BY freeze_frame_id ASC
                LIMIT 1
                """,
                (shot_row["shot_id"], player_id),
            ).fetchone()
            if row is not None and row["position_name"]:
                shooter_position = row["position_name"]

    return _ShooterMetadata(
        name=shooter_name,
        team_name=team_name,
        position=shooter_position,
        body_part=body_part,
        technique=technique,
        start_x=start_x,
        start_y=start_y,
    )


def _format_freeze_frame_blocks(
    connection: sqlite3.Connection, shot_row: sqlite3.Row
) -> Tuple[str, str, str]:
    if not _table_exists(connection, "freeze_frames"):
        return ("unknown", "none", "none")

    entries = _load_freeze_frames(connection, shot_row["shot_id"])

    goalkeeper_text = _build_goalkeeper_text(entries)
    support_line = _build_support_line(entries, shot_row)
    pressure_line = _build_pressure_line(entries, shot_row)

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
    entries: Sequence[_FreezeFrameEntry], shot_row: sqlite3.Row
) -> str:
    start_x = float(shot_row["start_x"] or 0.0)
    start_y = float(shot_row["start_y"] or 0.0)
    shooter_id = shot_row["player_id"]

    teammates: List[Tuple[str, float, float]] = []
    for entry in entries:
        if not entry.teammate or entry.player_id == shooter_id:
            continue
        if entry.x is None or entry.y is None:
            continue
        distance = math.hypot(entry.x - start_x, entry.y - start_y)
        if distance <= 18:
            bearing = _bearing_to_goal(start_x, start_y, entry.x, entry.y)
            name = entry.player_name or "unknown"
            teammates.append((name, distance, bearing))

    teammates.sort(key=lambda item: item[1])
    formatted = [
        f"{name}({dist:.1f}m @ {bearing:+.0f}°)" for name, dist, bearing in teammates[:3]
    ]

    return ", ".join(formatted) if formatted else "none"


def _build_pressure_line(
    entries: Sequence[_FreezeFrameEntry], shot_row: sqlite3.Row
) -> str:
    start_x = float(shot_row["start_x"] or 0.0)
    start_y = float(shot_row["start_y"] or 0.0)

    defenders_close: List[Tuple[str, float, float]] = []
    defenders_cone: List[Tuple[str, float, float]] = []

    for entry in entries:
        if entry.teammate or entry.keeper:
            continue
        if entry.x is None or entry.y is None:
            continue

        distance = math.hypot(entry.x - start_x, entry.y - start_y)
        bearing = _bearing_to_goal(start_x, start_y, entry.x, entry.y)
        name = entry.player_name or "unknown"

        if distance <= 15:
            defenders_close.append((name, distance, bearing))
        elif _in_shot_cone(start_x, start_y, entry.x, entry.y):
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

