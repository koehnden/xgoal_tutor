from typing import Any, Dict, Tuple

import pytest

import importlib

app_module = importlib.import_module("xgoal_tutor.api.app")
from xgoal_tutor.api.models import (
    LogisticRegressionModel,
    ShotFeatures,
    ShotPredictionRequest,
    ShotPredictionWithPromptRequest,
)
from xgoal_tutor.api.services import generate_shot_predictions
from fastapi import HTTPException


class DummyLLM:
    def __init__(self) -> None:
        self.calls: list[Dict[str, Any]] = []

    def generate(self, prompt: str, model: str | None = None, **kwargs: Any) -> Tuple[str, str]:
        self.calls.append({"prompt": prompt, "model": model, "options": kwargs})
        return (" stub explanation ", model or "dummy-model")


@pytest.fixture
def llm_stub() -> DummyLLM:
    dummy = DummyLLM()
    original = getattr(app_module, "_LLM_CLIENT", None)
    app_module._LLM_CLIENT = dummy
    app_module._MATCH_CACHE.clear()
    try:
        yield dummy
    finally:
        if original is not None:
            app_module._LLM_CLIENT = original


def _build_shot_payload(**overrides: Any) -> Dict[str, Any]:
    payload = {
        "shot_id": "shot-101",
        "match_id": "match-abc",
        "start_x": 102.0,
        "start_y": 34.0,
        "is_set_piece": False,
        "under_pressure": True,
        "ff_keeper_x": 118.0,
        "ff_keeper_y": 40.0,
    }
    payload.update(overrides)
    return payload


def _model_payload() -> Dict[str, Any]:
    return {
        "intercept": -0.3,
        "coefficients": {
            "dist_sb": -0.04,
            "angle_deg_sb": 0.03,
            "is_set_piece": 0.2,
            "under_pressure": -0.15,
        },
    }


def test_predict_shots_endpoint_returns_predictions_and_caches(llm_stub: DummyLLM):
    shot_payload = _build_shot_payload()
    model_payload = _model_payload()

    request = ShotPredictionRequest(
        shots=[ShotFeatures(**shot_payload)],
        model=LogisticRegressionModel(**model_payload),
    )

    response = app_module.predict_shots(request)

    assert response.explanation == "stub explanation"
    assert response.llm_model == "dummy-model"
    assert len(response.shots) == 1

    expected_predictions, _ = generate_shot_predictions(request.shots, request.model)
    assert response.shots[0].xg == pytest.approx(expected_predictions[0].xg)

    assert llm_stub.calls
    assert "You are an analyst" in llm_stub.calls[0]["prompt"]

    cached = app_module.get_match_shots(shot_payload["match_id"])
    assert cached.shots[0].shot_id == shot_payload["shot_id"]


def test_predict_shots_with_prompt_uses_custom_prompt(llm_stub: DummyLLM):
    shot_payload = _build_shot_payload(shot_id="shot-202")
    model_payload = _model_payload()
    prompt_text = "Custom tactical analysis"

    request = ShotPredictionWithPromptRequest(
        shots=[ShotFeatures(**shot_payload)],
        model=LogisticRegressionModel(**model_payload),
        prompt=prompt_text,
    )

    response = app_module.predict_shots_with_prompt(request)

    assert response.explanation == "stub explanation"
    assert llm_stub.calls[-1]["prompt"] == prompt_text


def test_predict_shots_rejects_empty_shot_list(llm_stub: DummyLLM):
    request = ShotPredictionRequest(
        shots=[],
        model=LogisticRegressionModel(**_model_payload()),
    )

    with pytest.raises(HTTPException) as excinfo:
        app_module.predict_shots(request)

    assert excinfo.value.status_code == 400
    assert excinfo.value.detail == "At least one shot must be provided"
    assert not llm_stub.calls
