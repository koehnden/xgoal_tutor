import math

import pandas as pd
import pytest

from xgoal_tutor.api.models import LogisticRegressionModel, ShotFeatures, ShotPrediction

services = pytest.importorskip("xgoal_tutor.api.services")

if getattr(services, "__STUB__", False):
    pytest.skip("xgoal_tutor.api.services stubbed", allow_module_level=True)

from xgoal_tutor.api.services import (
    _build_xgoal_prompts,
    _format_feature_block,
    _compute_teammate_context,
    build_feature_dataframe,
    calculate_probabilities,
    generate_shot_predictions,
    group_predictions_by_match,
)


def test_calculate_probabilities_matches_manual_logistic():
    shots = [
        ShotFeatures(
            shot_id="s1",
            match_id="m1",
            start_x=100.0,
            start_y=35.0,
            is_set_piece=True,
            ff_keeper_x=120.0,
            ff_keeper_y=40.0,
        )
    ]
    features = build_feature_dataframe(shots)
    model = LogisticRegressionModel(
        intercept=-0.4,
        coefficients={"dist_sb": -0.05, "angle_deg_sb": 0.025, "is_set_piece": 0.3},
    )

    probabilities, contributions = calculate_probabilities(features, model)

    aligned = features.reindex(columns=list(model.coefficients.keys()), fill_value=0.0)
    linear_score = float(aligned.mul(pd.Series(model.coefficients), axis=1).sum(axis=1) + model.intercept)
    expected_probability = 1.0 / (1.0 + math.exp(-linear_score))

    assert probabilities.iloc[0] == pytest.approx(expected_probability)
    assert contributions.iloc[0]["dist_sb"] == pytest.approx(
        aligned.iloc[0]["dist_sb"] * model.coefficients["dist_sb"]
    )
    assert contributions.iloc[0]["is_set_piece"] == pytest.approx(
        aligned.iloc[0]["is_set_piece"] * model.coefficients["is_set_piece"]
    )


def test_generate_shot_predictions_returns_sorted_reason_codes():
    shots = [
        ShotFeatures(
            shot_id="shot-a",
            match_id="match-123",
            start_x=96.0,
            start_y=40.0,
            is_set_piece=True,
            under_pressure=True,
            ff_keeper_x=118.0,
            ff_keeper_y=42.0,
        )
    ]
    model = LogisticRegressionModel(
        intercept=-0.2,
        coefficients={
            "dist_sb": -0.04,
            "angle_deg_sb": 0.03,
            "is_set_piece": 0.5,
            "under_pressure": -0.25,
        },
    )

    predictions, contributions = generate_shot_predictions(shots, model)

    assert len(predictions) == 1
    prediction = predictions[0]
    assert prediction.shot_id == "shot-a"
    assert prediction.match_id == "match-123"
    assert 0.0 <= prediction.xg <= 1.0

    sorted_by_contrib = contributions.iloc[0].abs().sort_values(ascending=False)
    returned_features = [code.feature for code in prediction.reason_codes]
    assert returned_features == list(sorted_by_contrib.index[: len(returned_features)])


def test_build_xgoal_prompts_render_markdown_template(monkeypatch):
    import sqlite3
    from contextlib import contextmanager

    connection = sqlite3.connect(":memory:")
    connection.executescript(
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
        CREATE TABLE players (player_id INTEGER PRIMARY KEY, player_name TEXT);
        CREATE TABLE teams (team_id INTEGER PRIMARY KEY, team_name TEXT);
        CREATE TABLE freeze_frames (
            freeze_frame_id INTEGER PRIMARY KEY AUTOINCREMENT,
            shot_id TEXT,
            player_id INTEGER,
            player_name TEXT,
            position_name TEXT,
            teammate INTEGER,
            keeper INTEGER,
            x REAL,
            y REAL
        );
        """
    )

    connection.executescript(
        """
        INSERT INTO teams(team_id, team_name) VALUES (1, 'Home FC'), (2, 'Away FC');
        INSERT INTO players(player_id, player_name) VALUES (9, 'Jordan Smith');
        INSERT INTO shots(
            shot_id, match_id, team_id, opponent_team_id, player_id,
            period, minute, second, play_pattern, start_x, start_y,
            body_part, technique, statsbomb_xg, score_home, score_away,
            is_goal, is_own_goal
        ) VALUES (
            'shot-1', 10, 1, 2, 9, 1, 12, 30.0, 'open_play', 102.0, 38.0,
            'right foot', 'volley', 0.321, 1, 0,
            0, 0
        );
        INSERT INTO shots(
            shot_id, match_id, team_id, opponent_team_id, player_id,
            period, minute, second, play_pattern, start_x, start_y,
            body_part, technique, statsbomb_xg, score_home, score_away,
            is_goal, is_own_goal
        ) VALUES (
            'shot-2', 10, 1, 2, 9, 1, 16, 45.0, 'fast_break', 108.0, 40.0,
            'right foot', 'open', 0.210, 1, 0,
            0, 0
        );
        INSERT INTO freeze_frames(shot_id, player_id, player_name, position_name, teammate, keeper, x, y)
        VALUES
            ('shot-1', 9, 'Jordan Smith', 'Striker', 1, 0, 102.0, 38.0),
            ('shot-1', 30, 'Goalkeeper', 'Goalkeeper', 0, 1, 118.0, 40.0),
            ('shot-2', 9, 'Jordan Smith', 'Striker', 1, 0, 108.0, 40.0),
            ('shot-2', 31, 'Goalkeeper', 'Goalkeeper', 0, 1, 119.0, 39.5);
        """
    )

    @contextmanager
    def fake_get_db():
        try:
            yield connection
        finally:
            pass

    monkeypatch.setattr(services, "get_db", fake_get_db)

    shots = [
        ShotFeatures(shot_id="shot-1", match_id="10", start_x=102.0, start_y=38.0),
        ShotFeatures(shot_id="shot-2", match_id="10", start_x=108.0, start_y=40.0),
    ]
    predictions = [
        ShotPrediction(shot_id="shot-1", match_id="10", xg=0.32, reason_codes=[]),
        ShotPrediction(shot_id="shot-2", match_id="10", xg=0.21, reason_codes=[]),
    ]
    contributions = pd.DataFrame(
        [
            {"dist_sb": -0.30, "angle_deg_sb": 0.15},
            {"dist_sb": -0.10, "angle_deg_sb": 0.08},
        ]
    )

    prompts = _build_xgoal_prompts(shots, predictions, contributions)
    assert len(prompts) == 2
    assert all("You are a football analyst" in prompt for prompt in prompts)
    assert all("Top factors" in prompt for prompt in prompts)
    assert all("(raw value:" in prompt for prompt in prompts)
    assert all("---" not in prompt for prompt in prompts)

    connection.close()


def test_format_feature_block_orders_by_absolute_value():
    row = pd.Series({"dist_sb": -0.2, "angle_deg_sb": 0.05, "is_corner": 0.0})
    raw = pd.Series({"dist_sb": 10.5, "angle_deg_sb": 25.0, "is_corner": 0.0})

    lines = _format_feature_block(row, raw)

    assert lines[0].startswith("↓ dist_sb")
    assert "raw value:10.5" in lines[0]
    assert all("is_corner" not in line for line in lines)


def test_group_predictions_by_match_excludes_missing_ids():
    predictions = [
        ShotPrediction(shot_id="a", match_id="match-1", xg=0.1, reason_codes=[]),
        ShotPrediction(shot_id="b", match_id=None, xg=0.2, reason_codes=[]),
        ShotPrediction(shot_id="c", match_id="match-1", xg=0.3, reason_codes=[]),
        ShotPrediction(shot_id="d", match_id="match-2", xg=0.4, reason_codes=[]),
    ]

    grouped = group_predictions_by_match(predictions)

    assert set(grouped.keys()) == {"match-1", "match-2"}
    assert len(grouped["match-1"]) == 2
    assert all(isinstance(item, ShotPrediction) for item in grouped["match-1"])


def test_compute_teammate_context_scores_teammates(monkeypatch):
    import sqlite3
    from contextlib import contextmanager

    connection = sqlite3.connect(":memory:")
    connection.executescript(
        """
        CREATE TABLE shots (
            shot_id TEXT PRIMARY KEY,
            match_id TEXT,
            team_id TEXT,
            player_id TEXT,
            start_x REAL,
            start_y REAL
        );
        CREATE TABLE freeze_frames (
            freeze_frame_id INTEGER PRIMARY KEY AUTOINCREMENT,
            shot_id TEXT,
            player_id TEXT,
            player_name TEXT,
            teammate INTEGER,
            keeper INTEGER,
            x REAL,
            y REAL
        );
        """
    )

    connection.execute(
        "INSERT INTO shots (shot_id, match_id, team_id, player_id, start_x, start_y) VALUES (?, ?, ?, ?, ?, ?)",
        ("shot-ctx", "match-ctx", "team-1", "player-10", 100.0, 40.0),
    )

    freeze_frames = [
        ("shot-ctx", "player-10", "Shooter", 1, 0, 100.0, 40.0),
        ("shot-ctx", "player-11", "Finisher", 1, 0, 112.0, 40.0),
        ("shot-ctx", "player-12", "Support", 1, 0, 96.0, 35.0),
        ("shot-ctx", "player-99", "Goalkeeper", 0, 1, 118.0, 40.0),
        ("shot-ctx", "player-20", "Defender", 0, 0, 114.0, 38.0),
    ]

    connection.executemany(
        "INSERT INTO freeze_frames (shot_id, player_id, player_name, teammate, keeper, x, y) VALUES (?, ?, ?, ?, ?, ?, ?)",
        freeze_frames,
    )

    @contextmanager
    def fake_get_db():
        try:
            yield connection
        finally:
            pass

    monkeypatch.setattr(services, "get_db", fake_get_db)

    shot = ShotFeatures(
        shot_id="shot-ctx",
        match_id="match-ctx",
        start_x=100.0,
        start_y=40.0,
        is_set_piece=False,
        ff_keeper_x=118.0,
        ff_keeper_y=40.0,
    )

    model = LogisticRegressionModel(
        intercept=-0.1,
        coefficients={
            "dist_sb": -0.05,
            "angle_deg_sb": 0.06,
            "ff_opponents": -0.02,
            "is_set_piece": -0.01,
            "gk_depth_sb": -0.01,
            "gk_offset_sb": -0.02,
        },
    )

    shooter_prediction, _ = generate_shot_predictions([shot], model)

    context = _compute_teammate_context([shot], shooter_prediction, model)[0]

    assert context.teammate_name_with_max_xgoal == "Finisher"
    assert context.team_mate_in_better_position_count == 1
    assert context.max_teammate_xgoal_diff is not None
    assert context.max_teammate_xgoal_diff < 0
    assert len(context.teammate_scoring_potential) == 2
    assert context.teammate_scoring_potential[0]["player_name"] == "Finisher"
    assert context.teammate_scoring_potential[0]["xg"] is not None


def test_is_offside_handles_positions():
    from xgoal_tutor.api.models import FreezeFramePlayer
    from xgoal_tutor.api.services import is_offside

    shooter_x = 100.0
    opponents = [
        FreezeFramePlayer(player_id="o1", player_name="Def1", teammate=False, keeper=False, x=110.0, y=40.0),
        FreezeFramePlayer(player_id="o2", player_name="Def2", teammate=False, keeper=False, x=112.0, y=38.0),
    ]
    keeper = FreezeFramePlayer(player_id="ok", player_name="Keeper", teammate=False, keeper=True, x=116.0, y=40.0)

    forward = FreezeFramePlayer(
        player_id="t1", player_name="Forward", teammate=True, keeper=False, x=114.0, y=40.0
    )
    trailing = FreezeFramePlayer(
        player_id="t2", player_name="Mid", teammate=True, keeper=False, x=95.0, y=40.0
    )
    inside_line = FreezeFramePlayer(
        player_id="t3", player_name="Wing", teammate=True, keeper=False, x=111.0, y=40.0
    )

    assert is_offside(forward, shooter_x, opponents, keeper) is True
    assert is_offside(trailing, shooter_x, opponents, keeper) is False
    assert is_offside(inside_line, shooter_x, opponents, keeper) is False


def test_is_offside_examples_cover_ball_and_defender_lines():
    from xgoal_tutor.api.models import FreezeFramePlayer
    from xgoal_tutor.api.services import is_offside

    shooter_x = 102.0
    opponents = [
        FreezeFramePlayer(player_id="o1", player_name="Def1", teammate=False, keeper=False, x=110.0, y=40.0),
        FreezeFramePlayer(player_id="o2", player_name="Def2", teammate=False, keeper=False, x=107.0, y=39.0),
    ]
    keeper = FreezeFramePlayer(player_id="ok", player_name="Keeper", teammate=False, keeper=True, x=116.0, y=40.0)

    offside_teammate = FreezeFramePlayer(
        player_id="t1", player_name="Forward", teammate=True, keeper=False, x=118.0, y=40.0
    )
    defended_teammate = FreezeFramePlayer(
        player_id="t2", player_name="Mid", teammate=True, keeper=False, x=109.0, y=42.0
    )
    deeper_teammate = FreezeFramePlayer(
        player_id="t3", player_name="Wing", teammate=True, keeper=False, x=98.0, y=41.0
    )

    assert is_offside(offside_teammate, shooter_x, opponents, keeper) is True
    assert is_offside(defended_teammate, shooter_x, opponents, keeper) is False
    assert is_offside(deeper_teammate, shooter_x, opponents, keeper) is False


def test_offside_teammate_receives_zero_xg(monkeypatch):
    import sqlite3
    from contextlib import contextmanager

    connection = sqlite3.connect(":memory:")
    connection.executescript(
        """
        CREATE TABLE shots (
            shot_id TEXT PRIMARY KEY,
            match_id TEXT,
            team_id TEXT,
            player_id TEXT,
            start_x REAL,
            start_y REAL
        );
        CREATE TABLE freeze_frames (
            freeze_frame_id INTEGER PRIMARY KEY AUTOINCREMENT,
            shot_id TEXT,
            player_id TEXT,
            player_name TEXT,
            teammate INTEGER,
            keeper INTEGER,
            x REAL,
            y REAL
        );
        """
    )

    connection.execute(
        "INSERT INTO shots (shot_id, match_id, team_id, player_id, start_x, start_y) VALUES (?, ?, ?, ?, ?, ?)",
        ("shot-off", "match-off", "team-1", "player-10", 100.0, 40.0),
    )

    freeze_frames = [
        ("shot-off", "player-10", "Shooter", 1, 0, 100.0, 40.0),
        ("shot-off", "player-11", "Offside", 1, 0, 114.0, 40.0),
        ("shot-off", "player-12", "Onside", 1, 0, 101.0, 42.0),
        ("shot-off", "player-99", "Goalkeeper", 0, 1, 118.0, 40.0),
        ("shot-off", "player-20", "Defender", 0, 0, 112.0, 39.0),
        ("shot-off", "player-21", "Defender2", 0, 0, 108.0, 41.0),
    ]

    connection.executemany(
        "INSERT INTO freeze_frames (shot_id, player_id, player_name, teammate, keeper, x, y) VALUES (?, ?, ?, ?, ?, ?, ?)",
        freeze_frames,
    )

    @contextmanager
    def fake_get_db():
        try:
            yield connection
        finally:
            pass

    monkeypatch.setattr(services, "get_db", fake_get_db)

    shot = ShotFeatures(
        shot_id="shot-off",
        match_id="match-off",
        start_x=100.0,
        start_y=40.0,
        is_set_piece=False,
        ff_keeper_x=118.0,
        ff_keeper_y=40.0,
    )

    model = LogisticRegressionModel(
        intercept=-0.1,
        coefficients={
            "dist_sb": -0.05,
            "angle_deg_sb": 0.06,
            "ff_opponents": -0.02,
            "is_set_piece": -0.01,
            "gk_depth_sb": -0.01,
            "gk_offset_sb": -0.02,
        },
    )

    shooter_prediction, _ = generate_shot_predictions([shot], model)
    context = _compute_teammate_context([shot], shooter_prediction, model)[0]

    offside_entry = next(entry for entry in context.teammate_scoring_potential if entry["player_name"] == "Offside")
    onside_entry = next(entry for entry in context.teammate_scoring_potential if entry["player_name"] == "Onside")

    assert offside_entry["xg"] == 0
    assert onside_entry["xg"] is not None and onside_entry["xg"] > 0
    assert context.team_mate_in_better_position_count >= 0


def test_adjust_potential_xgoal_with_passlane_reduces_blocked_lane():
    from xgoal_tutor.api.services import (
        FreezeFramePlayer,
        adjust_potential_xgoal_with_passlane,
    )

    shooter = FreezeFramePlayer(player_id="s", player_name="Shooter", teammate=True, keeper=False, x=100.0, y=40.0)
    teammate = FreezeFramePlayer(player_id="t", player_name="Runner", teammate=True, keeper=False, x=112.0, y=40.0)
    opponents = [
        FreezeFramePlayer(player_id="d1", player_name="Def1", teammate=False, keeper=False, x=106.0, y=40.0),
        FreezeFramePlayer(player_id="d2", player_name="Def2", teammate=False, keeper=False, x=90.0, y=35.0),
    ]

    adjusted = adjust_potential_xgoal_with_passlane(0.25, shooter.x, shooter.y, teammate, opponents)

    expected = 0.25 / (1 + math.exp(3.0))
    assert adjusted == pytest.approx(expected)


def test_adjust_potential_xgoal_with_passlane_keeps_open_lane():
    from xgoal_tutor.api.services import (
        FreezeFramePlayer,
        adjust_potential_xgoal_with_passlane,
    )

    shooter = FreezeFramePlayer(player_id="s", player_name="Shooter", teammate=True, keeper=False, x=100.0, y=40.0)
    teammate = FreezeFramePlayer(player_id="t", player_name="Runner", teammate=True, keeper=False, x=112.0, y=40.0)
    opponents = [
        FreezeFramePlayer(player_id="d1", player_name="Def1", teammate=False, keeper=False, x=104.0, y=46.0),
        FreezeFramePlayer(player_id="d2", player_name="Def2", teammate=False, keeper=False, x=90.0, y=35.0),
    ]

    adjusted = adjust_potential_xgoal_with_passlane(0.25, shooter.x, shooter.y, teammate, opponents)

    assert adjusted == pytest.approx(0.25, rel=1e-4)


def test_adjust_potential_xgoal_with_passlane_handles_missing_coordinates():
    from xgoal_tutor.api.services import (
        FreezeFramePlayer,
        adjust_potential_xgoal_with_passlane,
    )

    shooter = FreezeFramePlayer(player_id="s", player_name="Shooter", teammate=True, keeper=False, x=100.0, y=40.0)
    teammate = FreezeFramePlayer(player_id="t", player_name="Runner", teammate=True, keeper=False, x=112.0, y=40.0)
    opponents = [
        FreezeFramePlayer(player_id="d1", player_name="Def1", teammate=False, keeper=False, x=None, y=40.0),
        FreezeFramePlayer(player_id="d2", player_name="Def2", teammate=False, keeper=False, x=90.0, y=None),
    ]

    adjusted = adjust_potential_xgoal_with_passlane(0.4, shooter.x, shooter.y, teammate, opponents)

    assert adjusted == pytest.approx(0.4)
