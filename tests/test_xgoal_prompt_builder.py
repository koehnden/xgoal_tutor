"""Unit tests for the xGoal prompt builder."""

from __future__ import annotations

import sqlite3

import pytest

from xgoal_tutor.llm.xgoal_prompt_builder import build_xgoal_prompt


def _setup_base_schema(connection: sqlite3.Connection) -> None:
    cursor = connection.cursor()
    cursor.executescript(
        """
        CREATE TABLE shots (
            shot_id TEXT PRIMARY KEY,
            match_id INTEGER,
            team_id INTEGER,
            opponent_team_id INTEGER,
            player_id INTEGER,
            period INTEGER,
            minute INTEGER,
            second REAL,
            play_pattern TEXT,
            start_x REAL,
            start_y REAL,
            body_part TEXT,
            technique TEXT,
            statsbomb_xg REAL,
            score_home INTEGER,
            score_away INTEGER,
            is_goal INTEGER,
            is_own_goal INTEGER
        );

        CREATE TABLE players (
            player_id INTEGER PRIMARY KEY,
            player_name TEXT
        );

        CREATE TABLE teams (
            team_id INTEGER PRIMARY KEY,
            team_name TEXT
        );

        CREATE TABLE matches (
            match_id INTEGER PRIMARY KEY,
            home_team_id INTEGER,
            away_team_id INTEGER,
            competition_id INTEGER,
            season_id INTEGER
        );

        CREATE TABLE competitions (
            competition_id INTEGER PRIMARY KEY,
            competition_name TEXT
        );

        CREATE TABLE seasons (
            season_id INTEGER PRIMARY KEY,
            season_name TEXT
        );

        CREATE TABLE freeze_frames (
            freeze_frame_id INTEGER PRIMARY KEY AUTOINCREMENT,
            shot_id TEXT NOT NULL,
            player_id INTEGER,
            player_name TEXT,
            position_name TEXT,
            teammate INTEGER NOT NULL,
            keeper INTEGER NOT NULL,
            x REAL,
            y REAL
        );
        """
    )


def test_prompt_builder_formats_full_prompt() -> None:
    connection = sqlite3.connect(":memory:")
    _setup_base_schema(connection)

    connection.executescript(
        """
        INSERT INTO teams(team_id, team_name) VALUES (1, 'Attacking FC'), (2, 'Defensive SC');
        INSERT INTO players(player_id, player_name) VALUES (9, 'Jordan Smith');
        INSERT INTO competitions VALUES (1, 'Champions League');
        INSERT INTO seasons VALUES (5, '2023/24');
        INSERT INTO matches VALUES (100, 1, 2, 1, 5);
        INSERT INTO shots(
            shot_id, match_id, team_id, opponent_team_id, player_id,
            period, minute, second, play_pattern, start_x, start_y,
            body_part, technique, statsbomb_xg, score_home, score_away,
            is_goal, is_own_goal
        ) VALUES (
            'shot-1', 100, 1, 2, 9, 1, 23, 12.4, 'open_play',
            102.0, 38.0, 'right foot', 'volley', 0.348, 1, 0,
            1, 0
        );
        """
    )

    freeze_frames = [
        ("shot-1", 9, "Jordan Smith", "Striker", 1, 0, 102.0, 38.0),
        ("shot-1", 21, "Alex Wing", "Winger", 1, 0, 110.0, 34.5),
        ("shot-1", 6, "Pat Anchor", "Midfielder", 1, 0, 105.0, 41.0),
        ("shot-1", 1, "Keeper One", "Goalkeeper", 0, 1, 118.5, 40.5),
        ("shot-1", 5, "Centre Back", "Defender", 0, 0, 109.0, 38.5),
        ("shot-1", 4, "Full Back", "Defender", 0, 0, 106.0, 33.0),
        ("shot-1", 3, "Holding Mid", "Midfielder", 0, 0, 100.0, 38.0),
    ]
    connection.executemany(
        """
        INSERT INTO freeze_frames(shot_id, player_id, player_name, position_name, teammate, keeper, x, y)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        freeze_frames,
    )

    feature_block = ["↑ close range", "↑ free between lines", "↓ tight angle"]
    prompt = build_xgoal_prompt(
        connection,
        "shot-1",
        feature_block=feature_block,
        context_block="Set up by a quick one-two on the left.\n",
    )

    assert prompt.startswith("You are a football analyst translating xGoal probability model outputs")
    assert "Jordan Smith" in prompt
    assert "Match: Attacking FC 1–0 Defensive SC | Champions League 2023/24" in prompt
    assert "Event: 1’ 23:12 | pattern=open_play" in prompt
    assert "Shooter: Jordan Smith (Attacking FC), pos=Striker" in prompt
    assert "GK: Keeper One at x=118.5, y=40.5" in prompt
    assert "Attack support: Pat Anchor" in prompt
    assert "Alex Wing" in prompt
    assert "Centre Back" in prompt
    assert "Model xG: 0.348" in prompt
    assert (
        "Top factors (↑ raises xG, ↓ lowers xG) from logistic coefficients and raw feature values:" in prompt
    )
    assert "Set up by a quick one-two on the left." in prompt
    assert "Outcome: Goal for Attacking FC (1–0)" in prompt


def test_prompt_builder_handles_missing_freeze_frame() -> None:
    connection = sqlite3.connect(":memory:")
    _setup_base_schema(connection)

    connection.executescript(
        """
        INSERT INTO teams(team_id, team_name) VALUES (1, 'Attacking FC');
        INSERT INTO players(player_id, player_name) VALUES (9, 'Jordan Smith');
        INSERT INTO shots(
            shot_id, match_id, team_id, opponent_team_id, player_id,
            period, minute, second, play_pattern, start_x, start_y,
            body_part, technique, statsbomb_xg, score_home, score_away,
            is_goal, is_own_goal
        ) VALUES (
            'shot-2', 100, 1, NULL, 9, 2, 55, 48.9, 'fast_break',
            98.0, 40.0, 'left foot', 'open', 0.412, 2, 1,
            0, 0
        );
        """
    )

    prompt = build_xgoal_prompt(
        connection,
        "shot-2",
        feature_block=["↑ transition"],
    )

    assert "Attack support: none" in prompt
    assert "Pressure: none" in prompt
    assert "GK: unknown" in prompt
    assert "Outcome: No goal (2–1)" in prompt


def test_prompt_builder_uses_unknown_names_and_cone_defender() -> None:
    connection = sqlite3.connect(":memory:")
    _setup_base_schema(connection)

    connection.executescript(
        """
        INSERT INTO teams VALUES (1, 'Attacking FC'), (2, 'Defensive SC');
        INSERT INTO players VALUES (9, 'Jordan Smith');
        INSERT INTO matches VALUES (200, 1, 2, NULL, NULL);
        INSERT INTO shots(
            shot_id, match_id, team_id, opponent_team_id, player_id,
            period, minute, second, play_pattern, start_x, start_y,
            body_part, technique, statsbomb_xg, score_home, score_away,
            is_goal, is_own_goal
        ) VALUES (
            'shot-3', 200, 1, 2, 9, 1, 10, 5.0, 'open_play',
            95.0, 30.0, NULL, NULL, 0.250, NULL, NULL,
            0, 0
        );
        """
    )

    freeze_frames = [
        ("shot-3", 9, None, None, 1, 0, 95.0, 30.0),
        ("shot-3", 33, None, None, 0, 0, 119.0, 36.5),
    ]
    connection.executemany(
        """
        INSERT INTO freeze_frames(shot_id, player_id, player_name, position_name, teammate, keeper, x, y)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        freeze_frames,
    )

    prompt = build_xgoal_prompt(
        connection,
        "shot-3",
        feature_block=["↑ shot on target"],
    )

    assert "pos=unknown" in prompt
    assert "Pressure: unknown" in prompt  # defender included via cone rule


def test_prompt_builder_truncates_feature_block() -> None:
    connection = sqlite3.connect(":memory:")
    _setup_base_schema(connection)

    connection.executescript(
        """
        INSERT INTO teams VALUES (1, 'Attacking FC');
        INSERT INTO players VALUES (9, 'Jordan Smith');
        INSERT INTO shots(
            shot_id, match_id, team_id, opponent_team_id, player_id,
            period, minute, second, play_pattern, start_x, start_y,
            body_part, technique, statsbomb_xg, score_home, score_away,
            is_goal, is_own_goal
        ) VALUES (
            'shot-4', NULL, 1, NULL, 9, 1, 1, 0.1, 'open_play',
            110.0, 39.0, 'right foot', 'open', 0.123, NULL, NULL,
            0, 0
        );
        """
    )

    features = [
        "↑ close range",
        "↑ central lane",
        "↑ first time",
        "↓ tight angle",
        "↓ heavy pressure",
        "↑ keeper caught",
    ]

    prompt = build_xgoal_prompt(connection, "shot-4", feature_block=features)

    feature_lines = [line for line in prompt.splitlines() if line.startswith("↑") or line.startswith("↓")]
    assert feature_lines == features[:5]


def test_prompt_builder_requires_valid_shot() -> None:
    connection = sqlite3.connect(":memory:")
    _setup_base_schema(connection)

    with pytest.raises(ValueError):
        build_xgoal_prompt(connection, "missing", feature_block=["↑ test"])

