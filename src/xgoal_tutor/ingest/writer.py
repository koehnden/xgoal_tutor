from __future__ import annotations

import sqlite3
from typing import Sequence

from xgoal_tutor.ingest.reader import MutableEvent
from xgoal_tutor.ingest.rows import build_event_rows, build_shot_rows


class StatsBombSQLiteWriter:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def write(self, events: Sequence[MutableEvent]) -> None:
        event_rows = build_event_rows(events)
        shot_rows, freeze_frame_rows = build_shot_rows(events)

        self._insert_events(event_rows)
        self._insert_shots(shot_rows)
        self._insert_freeze_frames(freeze_frame_rows)

    def _insert_events(self, rows: Sequence[tuple]) -> None:
        self._connection.executemany(
            """
            INSERT OR REPLACE INTO events (
                event_id, match_id, team_id, player_id, opponent_team_id, possession,
                period, minute, second, timestamp, type, play_pattern, under_pressure,
                counterpress, duration, raw_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

    def _insert_shots(self, rows: Sequence[tuple]) -> None:
        self._connection.executemany(
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
            rows,
        )

    def _insert_freeze_frames(self, rows: Sequence[tuple]) -> None:
        if not rows:
            return
        self._connection.executemany(
            """
            INSERT INTO freeze_frames (
                shot_id, player_id, player_name, position_name, teammate, keeper, x, y
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )


__all__ = ["StatsBombSQLiteWriter"]
