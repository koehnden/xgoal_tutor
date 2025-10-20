from __future__ import annotations


CREATE_TABLE_STATEMENTS = (
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
    """,
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
    """,
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
    """,
)


CREATE_INDEX_STATEMENTS = (
    """
    CREATE INDEX IF NOT EXISTS idx_events_match ON events(match_id);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_shots_match ON shots(match_id);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_freeze_frames_shot ON freeze_frames(shot_id);
    """,
)


__all__ = [
    "CREATE_INDEX_STATEMENTS",
    "CREATE_TABLE_STATEMENTS",
]


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