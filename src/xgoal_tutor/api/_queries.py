"""Data access helpers for match and player information."""

from __future__ import annotations

import sqlite3
from typing import Dict, Iterable, List, Optional, Sequence

from xgoal_tutor.api._database import get_connection
from xgoal_tutor.api.models import MatchInfo, PlayerInfo, ShotRow


def fetch_match(match_id: str) -> Optional[MatchInfo]:
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT match_id, home_team_id, away_team_id, home_team_name, away_team_name,
                   competition_name, season_name, venue, match_date, match_label
              FROM matches
             WHERE match_id = ?
            """,
            (match_id,),
        ).fetchone()
    if row is None:
        return None
    match_identifier = row["match_id"]
    home_identifier = row["home_team_id"]
    away_identifier = row["away_team_id"]

    return MatchInfo(
        match_id=str(match_identifier) if match_identifier is not None else str(match_id),
        home_team_id=str(home_identifier) if home_identifier is not None else None,
        away_team_id=str(away_identifier) if away_identifier is not None else None,
        home_team_name=row["home_team_name"] or "Home Team",
        away_team_name=row["away_team_name"] or "Away Team",
        competition=row["competition_name"],
        season=row["season_name"],
        venue=row["venue"],
        match_date=row["match_date"],
        label=row["match_label"],
    )


def fetch_match_players(match_id: str) -> Dict[str, PlayerInfo]:
    """Return available lineup metadata for the requested match."""

    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT player_id, player_name, team_id, jersey_number, position_name
              FROM match_lineups
             WHERE match_id = ?
            """,
            (match_id,),
        ).fetchall()
    players: Dict[str, PlayerInfo] = {}
    for row in rows:
        player_id = str(row["player_id"])
        players[player_id] = PlayerInfo(
            player_id=player_id,
            name=row["player_name"] or f"Player {player_id}",
            team_id=str(row["team_id"]) if row["team_id"] is not None else None,
            jersey_number=row["jersey_number"],
            position=row["position_name"],
        )
    return players


def fetch_shots(match_id: str) -> List[ShotRow]:
    with get_connection() as conn:
        shot_rows = conn.execute(
            """
            SELECT s.shot_id,
                   s.match_id,
                   s.team_id,
                   s.opponent_team_id,
                   s.player_id,
                   p.player_name,
                   ml.jersey_number,
                   ml.position_name,
                   s.period,
                   s.minute,
                   s.second,
                   s.is_goal,
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
                   s.one_on_one,
                   s.open_goal,
                   s.follows_dribble,
                   s.deflected,
                   s.aerial_won,
                   s.freeze_frame_available
              FROM shots s
         LEFT JOIN players p ON p.player_id = s.player_id
         LEFT JOIN match_lineups ml
                ON ml.match_id = s.match_id AND ml.player_id = s.player_id
             WHERE s.match_id = ?
          ORDER BY s.period, s.minute, s.second, s.shot_id
            """,
            (match_id,),
        ).fetchall()

        ff_map: Dict[str, sqlite3.Row] = {}
        shot_ids = [row["shot_id"] for row in shot_rows]
        if shot_ids:
            placeholders = ",".join("?" for _ in shot_ids)
            aggregate_query = f"""
                SELECT shot_id,
                       SUM(CASE WHEN keeper = 1 THEN 1 ELSE 0 END) AS keeper_count,
                       MAX(CASE WHEN keeper = 1 THEN x END) AS keeper_x,
                       MAX(CASE WHEN keeper = 1 THEN y END) AS keeper_y,
                       SUM(CASE WHEN teammate = 0 THEN 1 ELSE 0 END) AS opponent_count
                  FROM freeze_frames
                 WHERE shot_id IN ({placeholders})
              GROUP BY shot_id
            """
            rows_ff = conn.execute(aggregate_query, shot_ids).fetchall()
            ff_map = {row["shot_id"]: row for row in rows_ff}

    results: List[ShotRow] = []
    for row in shot_rows:
        agg = ff_map.get(row["shot_id"], {})
        results.append(
            ShotRow(
                shot_id=str(row["shot_id"]),
                match_id=str(row["match_id"]),
                team_id=str(row["team_id"]) if row["team_id"] is not None else None,
                opponent_team_id=str(row["opponent_team_id"]) if row["opponent_team_id"] is not None else None,
                player_id=str(row["player_id"]) if row["player_id"] is not None else None,
                player_name=row["player_name"],
                jersey_number=row["jersey_number"],
                position_name=row["position_name"],
                period=row["period"],
                minute=row["minute"],
                second=row["second"],
                is_goal=row["is_goal"],
                score_home=row["score_home"],
                score_away=row["score_away"],
                start_x=row["start_x"],
                start_y=row["start_y"],
                is_set_piece=row["is_set_piece"],
                is_corner=row["is_corner"],
                is_free_kick=row["is_free_kick"],
                first_time=row["first_time"],
                under_pressure=row["under_pressure"],
                body_part=row["body_part"],
                one_on_one=row["one_on_one"],
                open_goal=row["open_goal"],
                follows_dribble=row["follows_dribble"],
                deflected=row["deflected"],
                aerial_won=row["aerial_won"],
                freeze_frame_available=row["freeze_frame_available"],
                ff_keeper_x=agg.get("keeper_x"),
                ff_keeper_y=agg.get("keeper_y"),
                ff_keeper_count=agg.get("keeper_count"),
                ff_opponents=agg.get("opponent_count"),
            )
        )
    return results


def player_participated(match_id: str, player_id: str) -> bool:
    """Return True if the player appears in lineups or events for the match."""

    with get_connection() as conn:
        row = conn.execute(
            "SELECT 1 FROM match_lineups WHERE match_id = ? AND player_id = ? LIMIT 1",
            (match_id, player_id),
        ).fetchone()
        if row is not None:
            return True
        row = conn.execute(
            "SELECT 1 FROM shots WHERE match_id = ? AND player_id = ? LIMIT 1",
            (match_id, player_id),
        ).fetchone()
    return row is not None


def match_exists(match_id: str) -> bool:
    with get_connection() as conn:
        row = conn.execute("SELECT 1 FROM matches WHERE match_id = ?", (match_id,)).fetchone()
    return row is not None
