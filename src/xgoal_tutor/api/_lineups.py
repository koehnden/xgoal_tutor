"""Query helpers for match lineup retrieval."""

from __future__ import annotations

from typing import Any, Dict, List

import sqlite3
from fastapi import HTTPException

from xgoal_tutor.api._database import get_db
from xgoal_tutor.api._row_utils import as_int, int_to_bool, row_value, team_payload


def get_match_lineups(match_id: str) -> Dict[str, Any]:
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
            lineup_rows: List[sqlite3.Row] = connection.execute(
                """
                SELECT
                    team_id,
                    player_id,
                    player_name,
                    jersey_number,
                    position_name,
                    is_starter,
                    sort_order
                FROM match_lineups
                WHERE match_id = ?
                ORDER BY team_id, is_starter DESC, COALESCE(sort_order, 0), player_name
                """,
                (match_id,),
            ).fetchall()
        except sqlite3.Error as exc:  # pragma: no cover - defensive guardrail
            raise HTTPException(status_code=500, detail="Database error") from exc

    home_team_id = row_value(match_row, "home_team_id")
    away_team_id = row_value(match_row, "away_team_id")
    home_name = row_value(match_row, "home_team_name")
    away_name = row_value(match_row, "away_team_name")

    home_lineup = {"starters": [], "bench": []}
    away_lineup = {"starters": [], "bench": []}

    for row in lineup_rows:
        team_id = row_value(row, "team_id")
        is_starter = int_to_bool(row_value(row, "is_starter"), default=False) or False
        jersey_number = as_int(row_value(row, "jersey_number"))
        position_name = row_value(row, "position_name")
        sort_order = as_int(row_value(row, "sort_order"))

        player_entry = {
            "player": {
                "id": str(row_value(row, "player_id")) if row_value(row, "player_id") is not None else None,
                "name": row_value(row, "player_name"),
                "jersey_number": jersey_number,
                "team_id": str(team_id) if team_id is not None else None,
                "position": position_name,
            },
            "is_starter": is_starter,
            "jersey_number": jersey_number,
            "position_name": position_name,
            "sort_order": sort_order,
        }

        target = home_lineup if team_id == home_team_id else away_lineup
        collection = target["starters" if is_starter else "bench"]
        collection.append(player_entry)

    return {
        "home": {
            "team": team_payload(home_team_id, home_name),
            "starters": home_lineup["starters"],
            "bench": home_lineup["bench"],
        },
        "away": {
            "team": team_payload(away_team_id, away_name),
            "starters": away_lineup["starters"],
            "bench": away_lineup["bench"],
        },
    }


__all__ = ["get_match_lineups"]
