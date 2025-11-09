import sqlite3
import sys
from collections import defaultdict
from types import ModuleType
from typing import Any, Dict, Tuple

import pytest

import importlib
import importlib.util

# Provide a lightweight stub for xgoal_tutor.api.services so tests do not depend on
# optional heavy dependencies such as pandas/numpy when the real module is absent.
if "xgoal_tutor.api.services" not in sys.modules:  # pragma: no cover - import side effect
    try:
        services_spec = importlib.util.find_spec("xgoal_tutor.api.services")
    except ModuleNotFoundError:  # pragma: no cover - package has heavy deps
        services_spec = None

    if services_spec is None:
        services_stub = ModuleType("xgoal_tutor.api.services")

        class _DummyContributions:
            def __init__(self, count: int) -> None:
                self._count = count

            def iloc(self, index: int) -> Dict[str, float]:
                return {}

            def __len__(self) -> int:
                return self._count

        def create_llm_client() -> Any:
            class _StubLLM:
                def generate(self, prompt: str, model: str | None = None, **kwargs: Any) -> Tuple[str, str]:
                    return (" stub explanation ", model or "dummy-model")

            return _StubLLM()

        def generate_llm_explanation(
            client: Any,
            shots: Any,
            predictions: Any,
            contributions: Any,
            *,
            llm_model: str | None = None,
            prompt_override: str | None = None,
        ) -> Tuple[str, str]:
            prompt = prompt_override or "You are a football analyst"
            return client.generate(prompt, model=llm_model)

        def generate_shot_predictions(shots: Any, model: Any) -> Tuple[list, _DummyContributions]:
            from xgoal_tutor.api.models import ShotPrediction

            predictions = [
                ShotPrediction(
                    shot_id=getattr(shot, "shot_id", None),
                    match_id=getattr(shot, "match_id", None),
                    xg=0.25 + 0.05 * index,
                    reason_codes=[],
                )
                for index, shot in enumerate(shots)
            ]
            return predictions, _DummyContributions(len(predictions))

        def group_predictions_by_match(predictions: Any) -> Dict[str, list]:
            grouped: Dict[str, list] = defaultdict(list)
            for prediction in predictions:
                match_id = getattr(prediction, "match_id", None)
                if match_id:
                    grouped[match_id].append(prediction)
            return grouped

        services_stub.create_llm_client = create_llm_client
        services_stub.generate_llm_explanation = generate_llm_explanation
        services_stub.generate_shot_predictions = generate_shot_predictions
        services_stub.group_predictions_by_match = group_predictions_by_match
        services_stub.__STUB__ = True
        sys.modules["xgoal_tutor.api.services"] = services_stub

services_module = importlib.import_module("xgoal_tutor.api.services")
USING_SERVICES_STUB = getattr(services_module, "__STUB__", False)

_DATABASE_MODULE_NAMES = []
try:
    database_spec = importlib.util.find_spec("xgoal_tutor.api.database")
except ModuleNotFoundError:  # pragma: no cover - package missing optional deps
    database_spec = None

if database_spec is not None:
    _DATABASE_MODULE_NAMES.append("xgoal_tutor.api.database")
DATABASE_MODULES = [importlib.import_module(name) for name in _DATABASE_MODULE_NAMES]

app_module = importlib.import_module("xgoal_tutor.api.app")
from xgoal_tutor.api.models import (
    LogisticRegressionModel,
    ShotFeatures,
    ShotPredictionRequest,
    ShotPredictionWithPromptRequest,
)
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
        "shot_id": "shot-1",
        "match_id": "match-1",
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


def test_predict_shots_endpoint_returns_predictions_and_caches(
    llm_stub: DummyLLM, seeded_match_database: Dict[str, str]
) -> None:
    shot_payload = _build_shot_payload()
    model_payload = _model_payload()

    request = ShotPredictionRequest(
        shots=[ShotFeatures(**shot_payload)],
        model=LogisticRegressionModel(**model_payload),
    )

    response = app_module.predict_shots(request)

    assert response.llm_model == "dummy-model"
    assert len(response.shots) == 1

    if USING_SERVICES_STUB:
        assert response.explanation == "stub explanation"
        assert response.shots[0].xg == pytest.approx(0.25)
    else:
        assert response.explanation.strip()
        expected_predictions, _ = services_module.generate_shot_predictions(
            [ShotFeatures(**shot_payload)],
            LogisticRegressionModel(**model_payload),
        )
        assert response.shots[0].xg == pytest.approx(expected_predictions[0].xg)

    assert llm_stub.calls
    if USING_SERVICES_STUB:
        assert "You are a football analyst" in llm_stub.calls[0]["prompt"]
    else:
        assert llm_stub.calls[0]["prompt"].strip()

    cached = app_module._MATCH_CACHE[shot_payload["match_id"]]
    assert cached.shots[0].shot_id == shot_payload["shot_id"]


def test_predict_shots_with_prompt_uses_custom_prompt(llm_stub: DummyLLM):
    if not hasattr(app_module, "predict_shots_with_prompt"):
        pytest.skip("predict_shots_with_prompt endpoint not implemented")

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


@pytest.fixture
def seeded_match_database(tmp_path, monkeypatch) -> Dict[str, str]:
    db_path = tmp_path / "xgoal-db.sqllite"
    connection = sqlite3.connect(db_path)
    try:
        connection.executescript(
            """
            CREATE TABLE matches (
                match_id TEXT PRIMARY KEY,
                match_date TEXT,
                competition_name TEXT,
                season_name TEXT,
                venue TEXT,
                home_team_id TEXT,
                away_team_id TEXT,
                home_team_name TEXT,
                away_team_name TEXT,
                match_label TEXT
            );
            CREATE TABLE teams (
                team_id TEXT PRIMARY KEY,
                team_name TEXT,
                short_name TEXT
            );
            CREATE TABLE match_lineups (
                match_id TEXT,
                team_id TEXT,
                player_id TEXT,
                player_name TEXT,
                jersey_number INTEGER,
                position_name TEXT,
                is_starter INTEGER,
                sort_order INTEGER
            );
            CREATE TABLE players (
                player_id TEXT PRIMARY KEY,
                player_name TEXT
            );
            CREATE TABLE shots (
                shot_id TEXT PRIMARY KEY,
                match_id TEXT,
                team_id TEXT,
                opponent_team_id TEXT,
                player_id TEXT,
                period INTEGER,
                minute INTEGER,
                second INTEGER,
                outcome TEXT,
                score_home INTEGER,
                score_away INTEGER,
                start_x REAL,
                start_y REAL,
                is_set_piece INTEGER,
                is_corner INTEGER,
                is_free_kick INTEGER,
                first_time INTEGER,
                under_pressure INTEGER,
                body_part TEXT,
                freeze_frame_available INTEGER,
                freeze_frame_count INTEGER,
                one_on_one INTEGER,
                open_goal INTEGER,
                follows_dribble INTEGER,
                deflected INTEGER,
                aerial_won INTEGER,
                is_goal INTEGER,
                is_own_goal INTEGER
            );
            CREATE TABLE freeze_frames (
                freeze_frame_id INTEGER PRIMARY KEY AUTOINCREMENT,
                shot_id TEXT,
                player_id TEXT,
                player_name TEXT,
                position_name TEXT,
                teammate INTEGER,
                keeper INTEGER,
                x REAL,
                y REAL
            );
            """
        )

        connection.executemany(
            "INSERT INTO teams (team_id, team_name, short_name) VALUES (?, ?, ?)",
            [
                ("1", "Home FC", "HFC"),
                ("2", "Away FC", "AFC"),
                ("3", "Team C", None),
                ("4", "Team D", None),
            ],
        )

        connection.executemany(
            "INSERT INTO matches (match_id, match_date, competition_name, season_name, venue, home_team_id, away_team_id, home_team_name, away_team_name, match_label) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    "match-1",
                    "2025-09-21T19:00:00Z",
                    "Friendly",
                    "2025/26",
                    "Camp Nou",
                    "1",
                    "2",
                    "Home FC",
                    "Away FC",
                    None,
                ),
                (
                    "match-2",
                    "2025-09-22",
                    "League",
                    "2025/26",
                    "Stadium Two",
                    "3",
                    "4",
                    "Team C",
                    "Team D",
                    "Team C vs Team D",
                ),
            ],
        )

        connection.executemany(
            "INSERT INTO match_lineups (match_id, team_id, player_id, player_name, jersey_number, position_name, is_starter, sort_order) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                ("match-1", "1", "player-1", "Striker", 9, "FW", 1, 1),
                ("match-1", "1", "player-2", "Midfielder", 8, "MF", 0, 12),
                ("match-1", "2", "player-3", "Defender", 5, "DF", 1, 1),
                ("match-1", "2", "player-4", "Goalkeeper", 1, "GK", 0, 18),
            ],
        )

        connection.executemany(
            "INSERT INTO players (player_id, player_name) VALUES (?, ?)",
            [
                ("player-1", "Striker"),
                ("player-3", "Defender"),
            ],
        )

        connection.execute(
            "INSERT INTO shots (shot_id, match_id, team_id, opponent_team_id, player_id, period, minute, second, outcome, score_home, score_away, start_x, start_y, is_set_piece, is_corner, is_free_kick, first_time, under_pressure, body_part, freeze_frame_available, freeze_frame_count, one_on_one, open_goal, follows_dribble, deflected, aerial_won, is_goal, is_own_goal) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "shot-1",
                "match-1",
                "1",
                "2",
                "player-1",
                1,
                15,
                30,
                "Goal",
                1,
                0,
                104.2,
                35.7,
                0,
                0,
                0,
                1,
                1,
                "Right Foot",
                1,
                3,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
            ),
        )

        connection.executemany(
            "INSERT INTO freeze_frames (shot_id, teammate, keeper, x, y) VALUES (?, ?, ?, ?, ?)",
            [
                ("shot-1", 0, 0, 110.0, 30.0),
                ("shot-1", 1, 1, 120.0, 40.0),
            ],
        )

        connection.commit()
    finally:
        connection.close()

    for module in DATABASE_MODULES:
        monkeypatch.setattr(module, "_DB_PATH", db_path)

    return {"match_id": "match-1", "second_match_id": "match-2"}


def test_list_matches_supports_pagination(seeded_match_database: Dict[str, str]) -> None:
    page_one = app_module.list_matches(page=1, page_size=1)

    assert page_one["page"] == 1
    assert page_one["page_size"] == 1
    assert page_one["total"] == 2
    assert len(page_one["items"]) == 1

    first_item = page_one["items"][0]
    assert first_item["id"] == seeded_match_database["match_id"]
    assert first_item["label"] == "Home FC – Away FC (2025-09-21)"
    assert first_item["kickoff_utc"] == "2025-09-21T19:00:00Z"
    assert first_item["home_team"]["name"] == "Home FC"
    assert first_item["away_team"]["name"] == "Away FC"

    page_two = app_module.list_matches(page=2, page_size=1)
    assert page_two["page"] == 2
    assert page_two["page_size"] == 1
    assert page_two["total"] == 2
    assert len(page_two["items"]) == 1
    assert page_two["items"][0]["id"] == seeded_match_database["second_match_id"]


def test_get_match_lineups_returns_home_and_away_groups(seeded_match_database: Dict[str, str]) -> None:
    response = app_module.get_match_lineups(seeded_match_database["match_id"])

    assert response["home"]["team"]["name"] == "Home FC"
    assert response["away"]["team"]["name"] == "Away FC"

    home_starters = response["home"]["starters"]
    assert len(home_starters) == 1
    assert home_starters[0]["is_starter"] is True
    assert home_starters[0]["player"]["name"] == "Striker"
    assert home_starters[0]["jersey_number"] == 9

    away_bench = response["away"]["bench"]
    assert len(away_bench) == 1
    assert away_bench[0]["is_starter"] is False
    assert away_bench[0]["player"]["name"] == "Goalkeeper"


def test_list_match_shot_features_returns_contextualised_payload(seeded_match_database: Dict[str, str]) -> None:
    response = app_module.list_match_shot_features(seeded_match_database["match_id"])

    assert len(response["items"]) == 1
    shot = response["items"][0]

    assert shot["period"] == "1H"
    assert shot["minute"] == 15
    assert shot["second"] == 30
    assert shot["result"] == "Goal"
    assert shot["scoreline_before"] == {"home": 0, "away": 0}
    assert shot["scoreline_after"] == {"home": 1, "away": 0}

    shooter = shot["shooter"]
    assert shooter["player_id"] == "player-1"
    assert shooter["player_name"] == "Striker"
    assert shooter["team_id"] == "1"
    assert shooter["team_name"] == "Home FC"

    features = shot["features"]
    assert features["shot_id"] == "shot-1"
    assert features["match_id"] == seeded_match_database["match_id"]
    assert features["start_x"] == pytest.approx(104.2)
    assert features["start_y"] == pytest.approx(35.7)
    assert features["is_set_piece"] is False
    assert features["is_corner"] is False
    assert features["is_free_kick"] is False
    assert features["first_time"] is True
    assert features["under_pressure"] is True
    assert features["body_part"] == "Right Foot"
    assert features["ff_keeper_x"] == pytest.approx(120.0)
    assert features["ff_keeper_y"] == pytest.approx(40.0)
    assert features["ff_opponents"] == pytest.approx(1.0)
    assert features["freeze_frame_available"] == 1
    assert features["ff_keeper_count"] == 1
    assert features["one_on_one"] is False
    assert features["open_goal"] is False
    assert features["follows_dribble"] is False
    assert features["deflected"] is False
    assert features["aerial_won"] is False


def test_list_match_shot_features_missing_match_raises_404(seeded_match_database: Dict[str, str]) -> None:
    with pytest.raises(HTTPException) as excinfo:
        app_module.list_match_shot_features("missing")

    assert excinfo.value.status_code == 404


def test_list_match_shot_features_without_shots_raises_404(seeded_match_database: Dict[str, str]) -> None:
    with pytest.raises(HTTPException) as excinfo:
        app_module.list_match_shot_features(seeded_match_database["second_match_id"])

    assert excinfo.value.status_code == 404
