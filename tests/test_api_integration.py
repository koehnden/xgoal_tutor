from __future__ import annotations

import importlib
from typing import Any, Dict

import pytest
from starlette.testclient import TestClient


@pytest.fixture()
def api_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    # Ensure the FastAPI app uses the stub LLM client created in services.create_llm_client
    # by setting the environment flag before importing the app module.
    monkeypatch.setenv("XGOAL_TUTOR_STUB_LLM", "1")
    app_module = importlib.import_module("xgoal_tutor.api.app")
    importlib.reload(app_module)
    if not callable(app_module.app):
        pytest.skip("FastAPI app is not ASGI-callable in this environment")
    client = TestClient(app_module.app)
    return client


def _shot(min_overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
    data: Dict[str, Any] = {
        "shot_id": "S1",
        "match_id": "M1",
        "start_x": 108.0,
        "start_y": 38.0,
        "under_pressure": True,
    }
    if min_overrides:
        data.update(min_overrides)
    return data


def test_post_predict_shots_returns_stubbed_explanations(api_client: TestClient) -> None:
    from xgoal_tutor.api.models import DEFAULT_PRIMARY_MODEL

    payload = {
        "shots": [_shot()],
    }

    resp = api_client.post("/predict_shots", json=payload)
    assert resp.status_code == 200
    body = resp.json()

    assert body["llm_model"] == DEFAULT_PRIMARY_MODEL
    assert isinstance(body["shots"], list) and len(body["shots"]) == 1
    shot = body["shots"][0]
    assert shot["shot_id"] == "S1"
    assert shot["explanation"] == "stub explanation"
    assert 0.0 <= float(shot["xg"]) <= 1.0


def test_offense_and_defense_endpoints_work_with_stub_llm(api_client: TestClient) -> None:
    for path in ("/offense/predict_shots", "/defense/predict_shots"):
        resp = api_client.post(path, json={"shots": [_shot({"shot_id": f"{path}-S2"})]})
        assert resp.status_code == 200
        data = resp.json()
        assert data["shots"][0]["explanation"] == "stub explanation"


def test_rejects_empty_shot_list(api_client: TestClient) -> None:
    resp = api_client.post("/predict_shots", json={"shots": []})
    assert resp.status_code == 400
    body = resp.json()
    assert body["detail"] == "At least one shot must be provided"


def test_llm_model_override_accepted(api_client: TestClient) -> None:
    # Use the allowed fallback model name from the API models module
    from xgoal_tutor.api.models import DEFAULT_FALLBACK_MODELS

    model_override = list(DEFAULT_FALLBACK_MODELS)[0]
    resp = api_client.post(
        "/predict_shots",
        json={"shots": [_shot({"shot_id": "S3"})], "llm_model": model_override},
    )
    assert resp.status_code == 200
    assert resp.json()["llm_model"] == model_override
