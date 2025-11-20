"""Query helpers for match shot feature retrieval."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import sqlite3
from fastapi import HTTPException

from xgoal_tutor.api.database import get_db
from xgoal_tutor.api._row_utils import as_float, as_int, int_to_bool, row_value
from xgoal_tutor.api.models import ShotFeatures


def _score_dict(home: Any, away: Any) -> Dict[str, int] | None:
    home_score = as_int(home)
    away_score = as_int(away)
    if home_score is None or away_score is None:
        return None
    return {"home": home_score, "away": away_score}


def _resolve_scoring_team(row: sqlite3.Row, match_info: sqlite3.Row) -> int | None:
    team_id = as_int(row_value(row, "team_id"))
    opponent_team_id = as_int(row_value(row, "opponent_team_id"))

    if int_to_bool(row_value(row, "is_own_goal"), default=False):
        if opponent_team_id is not None:
            return opponent_team_id
        home_id = as_int(row_value(match_info, "home_team_id"))
        away_id = as_int(row_value(match_info, "away_team_id"))
        if team_id == home_id:
            return away_id
        if team_id == away_id:
            return home_id
        return opponent_team_id

    return team_id


def _build_scorelines(row: sqlite3.Row, match_info: sqlite3.Row) -> Dict[str, Dict[str, int] | None]:
    after = _score_dict(row_value(row, "score_home"), row_value(row, "score_away"))
    if after is None:
        return {"before": None, "after": None}

    before = {"home": after["home"], "away": after["away"]}
    if int_to_bool(row_value(row, "is_goal"), default=False):
        scoring_team = _resolve_scoring_team(row, match_info)
        home_id = as_int(row_value(match_info, "home_team_id"))
        away_id = as_int(row_value(match_info, "away_team_id"))

        if scoring_team == home_id and before["home"] is not None:
            before["home"] = max(before["home"] - 1, 0)
        elif scoring_team == away_id and before["away"] is not None:
            before["away"] = max(before["away"] - 1, 0)

    return {"before": before, "after": after}


def _map_period(value: Any) -> str | None:
    mapping = {1: "1H", 2: "2H", 3: "ET1", 4: "ET2", 5: "PEN"}
    period = as_int(value)
    if period is None:
        return None
    return mapping.get(period, "Unknown")


def _map_result(value: Any) -> str | None:
    if value is None:
        return None

    normalised = str(value).strip().lower()
    mapping = {
        "goal": "Goal",
        "saved": "Saved",
        "saved to post": "Saved",
        "saved off target": "Saved",
        "saved to keeper": "Saved",
        "blocked": "Blocked",
        "off t": "Off Target",
        "off target": "Off Target",
        "post": "Post",
        "bar": "Bar",
        "own goal": "Own Goal",
    }

    return mapping.get(normalised, "Unknown")


def _row_to_features(row: sqlite3.Row) -> Dict[str, Any]:
    return {
        "shot_id": str(row_value(row, "shot_id")) if row_value(row, "shot_id") is not None else None,
        "match_id": str(row_value(row, "match_id")) if row_value(row, "match_id") is not None else None,
        "start_x": as_float(row_value(row, "start_x")),
        "start_y": as_float(row_value(row, "start_y")),
        "is_set_piece": int_to_bool(row_value(row, "is_set_piece"), default=False) or False,
        "is_corner": int_to_bool(row_value(row, "is_corner"), default=False) or False,
        "is_free_kick": int_to_bool(row_value(row, "is_free_kick"), default=False) or False,
        "first_time": int_to_bool(row_value(row, "first_time")),
        "under_pressure": int_to_bool(row_value(row, "under_pressure")),
        "body_part": row_value(row, "body_part"),
        "ff_keeper_x": as_float(row_value(row, "ff_keeper_x")),
        "ff_keeper_y": as_float(row_value(row, "ff_keeper_y")),
        "ff_opponents": as_float(row_value(row, "ff_opponents")),
        "freeze_frame_available": as_int(row_value(row, "freeze_frame_available")),
        "ff_keeper_count": as_int(row_value(row, "ff_keeper_count")),
        "one_on_one": int_to_bool(row_value(row, "one_on_one")),
        "open_goal": int_to_bool(row_value(row, "open_goal")),
        "follows_dribble": int_to_bool(row_value(row, "follows_dribble")),
        "deflected": int_to_bool(row_value(row, "deflected")),
        "aerial_won": int_to_bool(row_value(row, "aerial_won")),
        "pass_under_pressure": int_to_bool(row_value(row, "pass_under_pressure")),
        "pass_height": row_value(row, "pass_height"),
        "pass_is_cross": int_to_bool(row_value(row, "pass_is_cross")),
        "pass_is_through_ball": int_to_bool(row_value(row, "pass_is_through_ball")),
        "pass_is_cutback": int_to_bool(row_value(row, "pass_is_cutback")),
        "pass_is_switch": int_to_bool(row_value(row, "pass_is_switch")),
        "assist_type": row_value(row, "assist_type"),
    }


def list_match_shot_features(match_id: str) -> Dict[str, Any]:
    with get_db() as connection:
        try:
            match_row = connection.execute(
                """
                SELECT
                    match_id,
                    home_team_id,
                    away_team_id,
                    home_team_name,
                    away_team_name
                FROM matches
                WHERE match_id = ?
                """,
                (match_id,),
            ).fetchone()
        except sqlite3.Error as exc:  # pragma: no cover - defensive guardrail
            raise HTTPException(status_code=500, detail="Database error") from exc

        if match_row is None:
            raise HTTPException(status_code=404, detail="Not found")

        try:
            shot_rows: List[sqlite3.Row] = connection.execute(
                """
                WITH ff AS (
                    SELECT
                        shot_id,
                        SUM(CASE WHEN teammate = 0 THEN 1 ELSE 0 END) AS ff_opponents,
                        SUM(CASE WHEN keeper = 1 THEN 1 ELSE 0 END) AS ff_keeper_count,
                        AVG(CASE WHEN keeper = 1 THEN x END) AS ff_keeper_x,
                        AVG(CASE WHEN keeper = 1 THEN y END) AS ff_keeper_y
                    FROM freeze_frames
                    GROUP BY shot_id
                )
                SELECT
                    s.shot_id,
                    s.match_id,
                    s.team_id,
                    s.opponent_team_id,
                    s.player_id,
                    s.period,
                    s.minute,
                    s.second,
                    s.outcome,
                    s.score_home,
                    s.score_away,
                    s.start_x,
                    s.start_y,
                    s.is_set_piece,
                    s.is_corner,
                    s.is_free_kick,
                    s.first_time,
                    s.under_pressure,
                    s.body_part,
                    s.freeze_frame_available,
                    s.freeze_frame_count,
                    s.one_on_one,
                    s.open_goal,
                    s.follows_dribble,
                    s.deflected,
                    s.aerial_won,
                    s.is_goal,
                    s.is_own_goal,
                    s.assist_type,
                    ff.ff_opponents,
                    ff.ff_keeper_count,
                    ff.ff_keeper_x,
                    ff.ff_keeper_y,
                    kp.under_pressure AS pass_under_pressure,
                    json_extract(kp.raw_json, '$.pass.height.name') AS pass_height,
                    json_extract(kp.raw_json, '$.pass.cross') AS pass_is_cross,
                    json_extract(kp.raw_json, '$.pass.through_ball') AS pass_is_through_ball,
                    json_extract(kp.raw_json, '$.pass.cut_back') AS pass_is_cutback,
                    json_extract(kp.raw_json, '$.pass.switch') AS pass_is_switch,
                    p.player_name,
                    COALESCE(t.team_name, CASE WHEN s.team_id = ? THEN ? WHEN s.team_id = ? THEN ? ELSE NULL END) AS team_name
                FROM shots s
                LEFT JOIN ff ON ff.shot_id = s.shot_id
                LEFT JOIN events kp ON kp.event_id = s.key_pass_id
                LEFT JOIN players p ON p.player_id = s.player_id
                LEFT JOIN teams t ON t.team_id = s.team_id
                WHERE s.match_id = ?
                ORDER BY s.period, s.minute, s.second, s.shot_id
                """,
                (
                    row_value(match_row, "home_team_id"),
                    row_value(match_row, "home_team_name"),
                    row_value(match_row, "away_team_id"),
                    row_value(match_row, "away_team_name"),
                    match_id,
                ),
            ).fetchall()
        except sqlite3.Error as exc:  # pragma: no cover - defensive guardrail
            raise HTTPException(status_code=500, detail="Database error") from exc

    if not shot_rows:
        raise HTTPException(status_code=404, detail="Not found")

    items: List[Dict[str, Any]] = []
    for row in shot_rows:
        scorelines = _build_scorelines(row, match_row)
        period = _map_period(row_value(row, "period"))
        result = _map_result(row_value(row, "outcome"))
        minute = as_int(row_value(row, "minute"))
        second_value = row_value(row, "second")
        second = as_int(second_value) if second_value is not None else None

        team_id = row_value(row, "team_id")
        shooter = {
            "player_id": str(row_value(row, "player_id")) if row_value(row, "player_id") is not None else None,
            "player_name": row_value(row, "player_name"),
            "team_id": str(team_id) if team_id is not None else None,
            "team_name": row_value(row, "team_name"),
        }
        if not any(value is not None for value in shooter.values()):
            shooter = None

        features = _row_to_features(row)

        items.append(
            {
                "period": period,
                "minute": minute,
                "second": second,
                "result": result,
                "scoreline_before": scorelines["before"],
                "scoreline_after": scorelines["after"],
                "shooter": shooter,
                "features": features,
            }
        )

    return {"items": items}


def load_shot_features_by_ids(shot_ids: Sequence[str]) -> Dict[str, ShotFeatures]:
    if not shot_ids:
        return {}

    with get_db() as connection:
        try:
            placeholders = ",".join("?" for _ in shot_ids)
            shot_rows: List[sqlite3.Row] = connection.execute(
                f"""
                WITH ff AS (
                    SELECT
                        shot_id,
                        SUM(CASE WHEN teammate = 0 THEN 1 ELSE 0 END) AS ff_opponents,
                        SUM(CASE WHEN keeper = 1 THEN 1 ELSE 0 END) AS ff_keeper_count,
                        AVG(CASE WHEN keeper = 1 THEN x END) AS ff_keeper_x,
                        AVG(CASE WHEN keeper = 1 THEN y END) AS ff_keeper_y
                    FROM freeze_frames
                    WHERE shot_id IN ({placeholders})
                    GROUP BY shot_id
                )
                SELECT
                    s.shot_id,
                    s.match_id,
                    s.start_x,
                    s.start_y,
                    s.is_set_piece,
                    s.is_corner,
                    s.is_free_kick,
                    s.first_time,
                    s.under_pressure,
                    s.body_part,
                    s.freeze_frame_available,
                    s.freeze_frame_count,
                    s.one_on_one,
                    s.open_goal,
                    s.follows_dribble,
                    s.deflected,
                    s.aerial_won,
                    s.assist_type,
                    ff.ff_opponents,
                    ff.ff_keeper_count,
                    ff.ff_keeper_x,
                    ff.ff_keeper_y,
                    kp.under_pressure AS pass_under_pressure,
                    json_extract(kp.raw_json, '$.pass.height.name') AS pass_height,
                    json_extract(kp.raw_json, '$.pass.cross') AS pass_is_cross,
                    json_extract(kp.raw_json, '$.pass.through_ball') AS pass_is_through_ball,
                    json_extract(kp.raw_json, '$.pass.cut_back') AS pass_is_cutback,
                    json_extract(kp.raw_json, '$.pass.switch') AS pass_is_switch
                FROM shots s
                LEFT JOIN ff ON ff.shot_id = s.shot_id
                LEFT JOIN events kp ON kp.event_id = s.key_pass_id
                WHERE s.shot_id IN ({placeholders})
                """,
                tuple(shot_ids) + tuple(shot_ids),
            ).fetchall()
        except sqlite3.Error as exc:  # pragma: no cover - defensive guardrail
            raise HTTPException(status_code=500, detail="Database error") from exc

    features: Dict[str, ShotFeatures] = {}
    for row in shot_rows:
        feature_values = _row_to_features(row)
        shot_id = feature_values.get("shot_id")
        if shot_id is None:
            continue
        features[shot_id] = ShotFeatures(**feature_values)

    return features


__all__ = ["list_match_shot_features", "load_shot_features_by_ids"]
