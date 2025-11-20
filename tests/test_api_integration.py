from __future__ import annotations

import importlib
import sqlite3
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable

import pytest


@pytest.fixture()
def api_client(monkeypatch: pytest.MonkeyPatch) -> Any:
    # Ensure the package under test is importable when the repository isn't installed.
    project_src = Path(__file__).resolve().parents[1] / "src"
    sys.path.insert(0, str(project_src))

    # Build a minimal in-memory database so prompt generation can read shot rows
    # without relying on the real StatsBomb dataset.
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row

    connection.executescript(
        """
        CREATE TABLE players (
            player_id INTEGER PRIMARY KEY,
            player_name TEXT
        );

        CREATE TABLE teams (
            team_id INTEGER PRIMARY KEY,
            team_name TEXT
        );

        CREATE TABLE shots (
            shot_id TEXT PRIMARY KEY,
            match_id TEXT,
            player_id INTEGER,
            period INTEGER,
            minute INTEGER,
            second REAL,
            play_pattern TEXT,
            score_home INTEGER,
            score_away INTEGER,
            start_x REAL,
            start_y REAL,
            is_goal INTEGER,
            body_part TEXT,
            technique TEXT,
            team_id INTEGER,
            opponent_team_id INTEGER,
            statsbomb_xg REAL
        );
        """
    )

    connection.executemany(
        "INSERT INTO players (player_id, player_name) VALUES (?, ?)",
        [
            (10, "Player One"),
            (11, "Player Two"),
            (12, "Player Three"),
        ],
    )

    connection.executemany(
        "INSERT INTO teams (team_id, team_name) VALUES (?, ?)",
        [
            (1, "Home FC"),
            (2, "Away FC"),
        ],
    )

    connection.executemany(
        """
        INSERT INTO shots (
            shot_id,
            match_id,
            player_id,
            period,
            minute,
            second,
            play_pattern,
            score_home,
            score_away,
            start_x,
            start_y,
            is_goal,
            body_part,
            technique,
            team_id,
            opponent_team_id,
            statsbomb_xg
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                "S1",
                "M1",
                10,
                1,
                12,
                30.0,
                "Open Play",
                1,
                0,
                108.0,
                38.0,
                0,
                "Right Foot",
                "Normal",
                1,
                2,
                0.25,
            ),
            (
                "/offense/predict_shots-S2",
                "M2",
                11,
                2,
                45,
                10.5,
                "Open Play",
                0,
                0,
                100.0,
                40.0,
                0,
                "Left Foot",
                "Volley",
                1,
                2,
                0.15,
            ),
            (
                "/defense/predict_shots-S2",
                "M2",
                11,
                2,
                46,
                5.5,
                "Open Play",
                0,
                0,
                102.0,
                35.0,
                0,
                "Left Foot",
                "Volley",
                1,
                2,
                0.18,
            ),
            (
                "S3",
                "M3",
                12,
                1,
                5,
                2.0,
                "Corner",
                0,
                0,
                95.0,
                30.0,
                0,
                "Head",
                "Header",
                2,
                1,
                0.12,
            ),
        ],
    )

    @contextmanager
    def stub_get_db() -> Iterable[sqlite3.Connection]:
        try:
            yield connection
        finally:
            pass

    # Ensure the FastAPI app uses the stub LLM client created in services.create_llm_client
    # by setting the environment flag before importing the app module.
    monkeypatch.setenv("XGOAL_TUTOR_STUB_LLM", "1")
    monkeypatch.setattr("xgoal_tutor.api.database.get_db", stub_get_db)
    monkeypatch.setattr("xgoal_tutor.api.services.get_db", stub_get_db)

    app_module = importlib.import_module("xgoal_tutor.api.app")
    importlib.reload(app_module)

    class _Response:
        def __init__(self, status_code: int, payload: Dict[str, Any]):
            self.status_code = status_code
            self._payload = payload

        def json(self) -> Dict[str, Any]:
            return self._payload

    class _SyncClient:
        def post(self, url: str, json: Dict[str, Any] | None = None) -> _Response:  # type: ignore[override]
            from fastapi import HTTPException
            from xgoal_tutor.api.models import ShotPredictionRequest

            handler_map = {
                "/predict_shots": app_module.predict_shots,
                "/offense/predict_shots": app_module.offense_predict_shots,
                "/defense/predict_shots": app_module.defense_predict_shots,
            }

            handler = handler_map.get(url)
            if handler is None:
                return _Response(404, {"detail": "Not found"})

            try:
                request_model = ShotPredictionRequest(**(json or {}))
                from xgoal_tutor.api.models import ShotFeatures

                request_model.shots = [
                    shot if isinstance(shot, ShotFeatures) else ShotFeatures(**shot)
                    for shot in request_model.shots
                ]
                result = handler(request_model)
                if hasattr(result, "model_dump"):
                    payload = result.model_dump()
                    if isinstance(payload, dict) and "shots" in payload:
                        payload["shots"] = [
                            shot.model_dump() if hasattr(shot, "model_dump") else shot
                            for shot in payload["shots"]
                        ]
                else:
                    payload = result.dict()  # type: ignore[attr-defined]
                return _Response(200, payload)  # type: ignore[arg-type]
            except HTTPException as exc:  # pragma: no cover - maps FastAPI-style errors
                detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
                return _Response(exc.status_code, {"detail": detail})

    return _SyncClient()


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
