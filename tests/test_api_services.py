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
