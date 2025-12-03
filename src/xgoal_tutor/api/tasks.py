"""Celery tasks for async xGoal prediction processing."""

from __future__ import annotations

from typing import Any, Dict, List

from xgoal_tutor.api.celery_app import celery_app
from xgoal_tutor.api.models import (
    DEFAULT_LOGISTIC_REGRESSION_MODEL,
    DEFAULT_PRIMARY_MODEL,
    LogisticRegressionModel,
    ShotFeatures,
    ShotPrediction,
)
from xgoal_tutor.api.services import (
    create_llm_client,
    generate_llm_explanation,
    generate_shot_predictions,
    _apply_teammate_context,
)


@celery_app.task(bind=True, name="xgoal_tutor.predict_shots_offense")
def predict_shots_offense_task(
    self,
    shots_data: List[Dict[str, Any]],
    model_data: Dict[str, Any] | None = None,
    llm_model: str | None = None,
) -> Dict[str, Any]:
    """
    Async task for offense-focused shot predictions.

    Parameters
    ----------
    shots_data : List[Dict[str, Any]]
        List of shot feature dictionaries
    model_data : Dict[str, Any] | None
        Optional logistic regression model parameters
    llm_model : str | None
        Optional LLM model override

    Returns
    -------
    Dict[str, Any]
        Serialized ShotPredictionResponse
    """
    try:
        # Parse shots from dictionaries
        shots = [ShotFeatures(**shot_dict) for shot_dict in shots_data]

        # Parse model if provided
        if model_data:
            model = LogisticRegressionModel(**model_data)
        else:
            model = LogisticRegressionModel(**DEFAULT_LOGISTIC_REGRESSION_MODEL.model_dump())

        # Generate predictions
        predictions, contributions = generate_shot_predictions(shots, model)
        predictions = _apply_teammate_context(shots, predictions, model)

        # Generate LLM explanations with offense template
        llm_client = create_llm_client()
        llm_responses, model_used = generate_llm_explanation(
            llm_client,
            shots,
            predictions,
            contributions,
            llm_model=llm_model,
            prompt_template_name="xgoal_offense_prompt.md",
        )

        resolved_llm_model = model_used or llm_model or DEFAULT_PRIMARY_MODEL

        # Attach explanations to predictions
        predictions_with_explanations: List[ShotPrediction] = []
        for index, prediction in enumerate(predictions):
            data = prediction.model_dump()
            data["explanation"] = llm_responses[index]
            predictions_with_explanations.append(ShotPrediction(**data))

        # Return serialized response
        return {
            "shots": [pred.model_dump() for pred in predictions_with_explanations],
            "llm_model": resolved_llm_model,
        }

    except Exception as exc:
        # Re-raise to mark task as failed
        raise exc


@celery_app.task(bind=True, name="xgoal_tutor.predict_shots_defense")
def predict_shots_defense_task(
    self,
    shots_data: List[Dict[str, Any]],
    model_data: Dict[str, Any] | None = None,
    llm_model: str | None = None,
) -> Dict[str, Any]:
    """
    Async task for defense-focused shot predictions.

    Parameters
    ----------
    shots_data : List[Dict[str, Any]]
        List of shot feature dictionaries
    model_data : Dict[str, Any] | None
        Optional logistic regression model parameters
    llm_model : str | None
        Optional LLM model override

    Returns
    -------
    Dict[str, Any]
        Serialized ShotPredictionResponse
    """
    try:
        # Parse shots from dictionaries
        shots = [ShotFeatures(**shot_dict) for shot_dict in shots_data]

        # Parse model if provided
        if model_data:
            model = LogisticRegressionModel(**model_data)
        else:
            model = LogisticRegressionModel(**DEFAULT_LOGISTIC_REGRESSION_MODEL.model_dump())

        # Generate predictions
        predictions, contributions = generate_shot_predictions(shots, model)
        predictions = _apply_teammate_context(shots, predictions, model)

        # Generate LLM explanations with defense template
        llm_client = create_llm_client()
        llm_responses, model_used = generate_llm_explanation(
            llm_client,
            shots,
            predictions,
            contributions,
            llm_model=llm_model,
            prompt_template_name="xgoal_defense_prompt.md",
        )

        resolved_llm_model = model_used or llm_model or DEFAULT_PRIMARY_MODEL

        # Attach explanations to predictions
        predictions_with_explanations: List[ShotPrediction] = []
        for index, prediction in enumerate(predictions):
            data = prediction.model_dump()
            data["explanation"] = llm_responses[index]
            predictions_with_explanations.append(ShotPrediction(**data))

        # Return serialized response
        return {
            "shots": [pred.model_dump() for pred in predictions_with_explanations],
            "llm_model": resolved_llm_model,
        }

    except Exception as exc:
        # Re-raise to mark task as failed
        raise exc
