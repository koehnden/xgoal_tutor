"""Integration tests for async prediction endpoints."""

from __future__ import annotations

import sys
import uuid
from datetime import datetime, timezone
from types import ModuleType
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

# Mock Celery before imports
if "celery" not in sys.modules:
    celery_module = ModuleType("celery")

    class AsyncResult:
        def __init__(self, task_id: str, state: str = "PENDING", result: Any = None, info: Any = None):
            self.id = task_id
            self.state = state
            self.result = result
            self.info = info

    celery_module.result = ModuleType("celery.result")
    celery_module.result.AsyncResult = AsyncResult

    class Celery:
        def __init__(self, *args: Any, **kwargs: Any):
            self.conf = MagicMock()
            self.conf.update = MagicMock()

        def autodiscover_tasks(self, packages: Any) -> None:
            pass

        def task(self, *args: Any, **kwargs: Any):
            def decorator(func):
                func.apply_async = MagicMock(return_value=MagicMock(id=str(uuid.uuid4())))
                func.delay = MagicMock()
                return func
            return decorator

    celery_module.Celery = Celery
    sys.modules["celery"] = celery_module
    sys.modules["celery.result"] = celery_module.result

# Import after mocking
from xgoal_tutor.api.models import (
    JobStatus,
    PredictionJobResponse,
    ShotFeatures,
    ShotPredictionRequest,
)

# Import app module
import importlib
app_module = importlib.import_module("xgoal_tutor.api.app")


def _build_shot_payload(**overrides: Any) -> Dict[str, Any]:
    """Helper to build shot feature payload."""
    payload = {
        "shot_id": "shot-async-1",
        "match_id": "match-async-1",
        "start_x": 102.0,
        "start_y": 34.0,
        "is_set_piece": False,
        "under_pressure": True,
        "ff_keeper_x": 118.0,
        "ff_keeper_y": 40.0,
    }
    payload.update(overrides)
    return payload


class TestAsyncOffensePredictions:
    """Tests for /offense/predict_shots async endpoint."""

    @patch("xgoal_tutor.api.app.predict_shots_offense_task")
    def test_offense_predict_shots_returns_generation_id(self, mock_task):
        """Test that offense endpoint returns 202 with generation_id."""
        # Setup mock
        mock_task.apply_async = MagicMock()

        # Build request
        shot_payload = _build_shot_payload()
        request = ShotPredictionRequest(
            shots=[ShotFeatures(**shot_payload)],
        )

        # Call endpoint
        response = app_module.offense_predict_shots(request)

        # Assertions
        assert isinstance(response, PredictionJobResponse)
        assert response.status == JobStatus.QUEUED
        assert response.generation_id is not None
        assert len(response.generation_id) > 0
        assert isinstance(response.created_at, datetime)

        # Verify task was enqueued
        mock_task.apply_async.assert_called_once()
        call_kwargs = mock_task.apply_async.call_args
        assert "task_id" in call_kwargs[1]
        assert call_kwargs[1]["task_id"] == response.generation_id

    @patch("xgoal_tutor.api.app.predict_shots_offense_task")
    def test_offense_predict_shots_validates_empty_shots(self, mock_task):
        """Test that offense endpoint rejects empty shot list."""
        from fastapi import HTTPException

        request = ShotPredictionRequest(shots=[])

        with pytest.raises(HTTPException) as exc_info:
            app_module.offense_predict_shots(request)

        assert exc_info.value.status_code == 400
        assert "at least one shot" in str(exc_info.value.detail).lower()

    @patch("xgoal_tutor.api.app.predict_shots_offense_task")
    def test_offense_predict_shots_serializes_model_data(self, mock_task):
        """Test that model data is properly serialized for Celery."""
        from xgoal_tutor.api.models import DEFAULT_LOGISTIC_REGRESSION_MODEL, LogisticRegressionModel

        mock_task.apply_async = MagicMock()

        shot_payload = _build_shot_payload()
        custom_model = LogisticRegressionModel(
            intercept=-1.5,
            coefficients={"dist_sb": -0.1, "angle_deg_sb": 0.05}
        )

        request = ShotPredictionRequest(
            shots=[ShotFeatures(**shot_payload)],
            model=custom_model,
        )

        response = app_module.offense_predict_shots(request)

        # Verify model was serialized
        call_args = mock_task.apply_async.call_args
        shots_data, model_data, llm_model = call_args[1]["args"]
        assert model_data is not None
        assert model_data["intercept"] == -1.5
        assert "dist_sb" in model_data["coefficients"]


class TestAsyncDefensePredictions:
    """Tests for /defense/predict_shots async endpoint."""

    @patch("xgoal_tutor.api.app.predict_shots_defense_task")
    def test_defense_predict_shots_returns_generation_id(self, mock_task):
        """Test that defense endpoint returns 202 with generation_id."""
        mock_task.apply_async = MagicMock()

        shot_payload = _build_shot_payload(shot_id="shot-defense-1")
        request = ShotPredictionRequest(
            shots=[ShotFeatures(**shot_payload)],
        )

        response = app_module.defense_predict_shots(request)

        assert isinstance(response, PredictionJobResponse)
        assert response.status == JobStatus.QUEUED
        assert response.generation_id is not None
        assert isinstance(response.created_at, datetime)

        mock_task.apply_async.assert_called_once()

    @patch("xgoal_tutor.api.app.predict_shots_defense_task")
    def test_defense_predict_shots_with_llm_model_override(self, mock_task):
        """Test that LLM model can be overridden."""
        mock_task.apply_async = MagicMock()

        shot_payload = _build_shot_payload()
        request = ShotPredictionRequest(
            shots=[ShotFeatures(**shot_payload)],
            llm_model="mistral:7b-instruct-q4_0",
        )

        response = app_module.defense_predict_shots(request)

        # Verify LLM model was passed
        call_args = mock_task.apply_async.call_args
        shots_data, model_data, llm_model = call_args[1]["args"]
        assert llm_model == "mistral:7b-instruct-q4_0"


class TestPredictionStatusPolling:
    """Tests for GET /predict_shots status polling endpoint."""

    @patch("xgoal_tutor.api.celery_app.celery_app.AsyncResult")
    def test_get_prediction_status_queued(self, mock_async_result_class):
        """Test polling returns QUEUED status."""
        generation_id = str(uuid.uuid4())

        # Mock pending task
        mock_result = MagicMock()
        mock_result.state = "PENDING"
        mock_result.info = {"queued": True}  # Make info truthy to avoid 404
        mock_async_result_class.return_value = mock_result

        response = app_module.get_prediction_status(generation_id)

        assert response.generation_id == generation_id
        assert response.status == JobStatus.QUEUED
        assert response.result is None
        assert response.error_message is None

    @patch("xgoal_tutor.api.celery_app.celery_app.AsyncResult")
    def test_get_prediction_status_running(self, mock_async_result_class):
        """Test polling returns RUNNING status."""
        generation_id = str(uuid.uuid4())

        mock_result = MagicMock()
        mock_result.state = "STARTED"
        mock_async_result_class.return_value = mock_result

        response = app_module.get_prediction_status(generation_id)

        assert response.generation_id == generation_id
        assert response.status == JobStatus.RUNNING
        assert response.result is None

    @patch("xgoal_tutor.api.celery_app.celery_app.AsyncResult")
    def test_get_prediction_status_completed(self, mock_async_result_class):
        """Test polling returns COMPLETED status with results."""
        from xgoal_tutor.api.models import DEFAULT_PRIMARY_MODEL

        generation_id = str(uuid.uuid4())

        # Mock successful result
        mock_result = MagicMock()
        mock_result.state = "SUCCESS"
        mock_result.result = {
            "shots": [
                {
                    "shot_id": "shot-1",
                    "match_id": "match-1",
                    "xg": 0.35,
                    "reason_codes": [],
                    "explanation": "Test explanation",
                }
            ],
            "llm_model": DEFAULT_PRIMARY_MODEL,
        }
        mock_async_result_class.return_value = mock_result

        response = app_module.get_prediction_status(generation_id)

        assert response.generation_id == generation_id
        assert response.status == JobStatus.COMPLETED
        assert response.result is not None
        assert len(response.result.shots) == 1
        assert response.result.shots[0].xg == 0.35
        assert response.result.shots[0].explanation == "Test explanation"
        assert response.result.llm_model == DEFAULT_PRIMARY_MODEL

    @patch("xgoal_tutor.api.celery_app.celery_app.AsyncResult")
    def test_get_prediction_status_failed(self, mock_async_result_class):
        """Test polling returns FAILED status with error message."""
        generation_id = str(uuid.uuid4())

        mock_result = MagicMock()
        mock_result.state = "FAILURE"
        mock_result.info = Exception("LLM service unavailable")
        mock_async_result_class.return_value = mock_result

        response = app_module.get_prediction_status(generation_id)

        assert response.generation_id == generation_id
        assert response.status == JobStatus.FAILED
        assert response.error_message is not None
        assert "LLM service unavailable" in response.error_message

    @patch("xgoal_tutor.api.celery_app.celery_app.AsyncResult")
    def test_get_prediction_status_not_found(self, mock_async_result_class):
        """Test polling returns 404 for non-existent job."""
        from fastapi import HTTPException

        generation_id = str(uuid.uuid4())

        mock_result = MagicMock()
        mock_result.state = "PENDING"
        mock_result.info = None
        mock_async_result_class.return_value = mock_result

        with pytest.raises(HTTPException) as exc_info:
            app_module.get_prediction_status(generation_id)

        assert exc_info.value.status_code == 404


class TestBackwardCompatibility:
    """Test that /predict_shots alias still works."""

    @patch("xgoal_tutor.api.app.predict_shots_offense_task")
    def test_predict_shots_alias_works(self, mock_task):
        """Test /predict_shots endpoint delegates to offense endpoint."""
        mock_task.apply_async = MagicMock()

        shot_payload = _build_shot_payload()
        request = ShotPredictionRequest(
            shots=[ShotFeatures(**shot_payload)],
        )

        response = app_module.predict_shots(request)

        assert isinstance(response, PredictionJobResponse)
        assert response.status == JobStatus.QUEUED
        mock_task.apply_async.assert_called_once()
