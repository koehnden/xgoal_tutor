import math

import pandas as pd
import pytest

from xgoal_tutor.api.models import LogisticRegressionModel, ShotFeatures, ShotPrediction
from xgoal_tutor.api.services import (
    build_event_inputs,
    build_feature_dataframe,
    build_llm_prompt,
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


def test_build_event_inputs_and_prompt_for_multiple_events():
    shots = [
        ShotFeatures(
            shot_id="shot-1",
            match_id="match-9",
            start_x=101.0,
            start_y=38.0,
            is_corner=True,
            body_part="Head",
        ),
        ShotFeatures(
            shot_id="shot-2",
            match_id="match-9",
            start_x=110.0,
            start_y=32.0,
            is_free_kick=True,
            under_pressure=True,
        ),
    ]
    predictions = [
        ShotPrediction(shot_id="shot-1", match_id="match-9", xg=0.42, reason_codes=[]),
        ShotPrediction(shot_id="shot-2", match_id="match-9", xg=0.18, reason_codes=[]),
    ]
    contributions = pd.DataFrame(
        [
            {"dist_sb": -0.3, "angle_deg_sb": 0.2},
            {"dist_sb": -0.15, "angle_deg_sb": 0.05, "is_free_kick": 0.1},
        ]
    )

    events = build_event_inputs(shots, predictions, contributions)
    assert len(events) == 2
    assert events[0].context is not None
    assert "Body part: Head" in events[0].context
    assert "Traits: Corner" in events[0].context
    assert events[1].contributions["is_free_kick"] == pytest.approx(0.1)

    prompt = build_llm_prompt(events, match_metadata={"teams": {"home": "Home", "away": "Away"}})
    assert "You will receive multiple shot events" in prompt
    assert "Player: shot-1" in prompt
    assert "Player: shot-2" in prompt


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
