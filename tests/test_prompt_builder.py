from __future__ import annotations

import sqlite3

import pytest

from xgoal_tutor.llm.xgoal_prompt_builder import build_xgoal_prompt


def _init_schema(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE shots (
            shot_id TEXT PRIMARY KEY,
            match_id INTEGER,
            player_id INTEGER,
            team_id INTEGER,
            opponent_team_id INTEGER,
            start_x REAL,
            start_y REAL,
            body_part TEXT,
            technique TEXT,
            period INTEGER,
            minute INTEGER,
            second REAL,
            play_pattern TEXT,
            statsbomb_xg REAL,
            is_goal INTEGER,
            is_own_goal INTEGER,
            score_home INTEGER,
            score_away INTEGER
        )
        """
    )
    cur.execute("CREATE TABLE players (player_id INTEGER PRIMARY KEY, player_name TEXT)")
    cur.execute("CREATE TABLE teams (team_id INTEGER PRIMARY KEY, team_name TEXT)")
    cur.execute(
        """
        CREATE TABLE matches (
            match_id INTEGER PRIMARY KEY,
            home_team_id INTEGER,
            away_team_id INTEGER,
            competition_name TEXT,
            season_name TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE freeze_frames (
            shot_id TEXT,
            player_id INTEGER,
            player_name TEXT,
            position_name TEXT,
            teammate INTEGER,
            keeper INTEGER,
            x REAL,
            y REAL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE match_lineups (
            match_id INTEGER,
            player_id INTEGER,
            position_name TEXT,
            is_starter INTEGER,
            sort_order INTEGER
        )
        """
    )
    conn.commit()


def _seed_minimal_data(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute("INSERT INTO teams(team_id, team_name) VALUES (1,'Home FC'), (2,'Away FC')")
    cur.execute("INSERT INTO players(player_id, player_name) VALUES (9,'Striker'), (1,'Goalkeeper')")
    cur.execute(
        "INSERT INTO matches(match_id, home_team_id, away_team_id, competition_name, season_name) VALUES (?,?, ?, ?, ?)",
        (100, 1, 2, "Friendly", "2024/25"),
    )
    cur.execute(
        """
        INSERT INTO shots(
            shot_id, match_id, player_id, team_id, opponent_team_id,
            start_x, start_y, body_part, technique,
            period, minute, second, play_pattern,
            statsbomb_xg, is_goal, is_own_goal, score_home, score_away
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "S1", 100, 9, 1, 2,
            110.0, 40.0, "Right Foot", "Normal",
            1, 10, 30.0, "Open Play",
            0.25, 0, 0, 0, 0,
        ),
    )
    # Freeze frame: GK and two players
    cur.execute(
        "INSERT INTO freeze_frames(shot_id, player_id, player_name, position_name, teammate, keeper, x, y) VALUES (?,?,?,?,?,?,?,?)",
        ("S1", 1, "Goalkeeper", "Goalkeeper", 0, 1, 116.0, 40.0),
    )
    cur.execute(
        "INSERT INTO freeze_frames(shot_id, player_id, player_name, position_name, teammate, keeper, x, y) VALUES (?,?,?,?,?,?,?,?)",
        ("S1", 9, "Striker", "Center Forward", 1, 0, 110.0, 40.0),
    )
    cur.execute(
        "INSERT INTO match_lineups(match_id, player_id, position_name, is_starter, sort_order) VALUES (?,?,?,?,?)",
        (100, 9, "Center Forward", 1, 1),
    )
    conn.commit()


def test_build_xgoal_prompt_minimum_happy_path() -> None:
    conn = sqlite3.connect(":memory:")
    _init_schema(conn)
    _seed_minimal_data(conn)

    prompt = build_xgoal_prompt(
        conn,
        "S1",
        feature_block=["- dist_sb: 10.0 (↓)", "- angle_deg_sb: 25.0 (↑)"],
        context_block="Coach note: be calmer in the box.",
    )

    # Check for key sections and fields rendered into the prompt
    assert "Match: Home FC" in prompt and "Away FC" in prompt
    assert "Shooter: Striker (Home FC)" in prompt
    assert "Model xG: 0.250" in prompt
    assert "Top factors" in prompt
    assert "Coach note" in prompt
    assert "GK:" in prompt and "Attack support:" in prompt and "Pressure:" in prompt


def test_build_xgoal_prompt_unknown_shot() -> None:
    conn = sqlite3.connect(":memory:")
    _init_schema(conn)
    with pytest.raises(ValueError):
        build_xgoal_prompt(conn, "UNKNOWN", feature_block=["- a: 1"], context_block=None)
