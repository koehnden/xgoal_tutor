"""FastAPI service exposing xGoal logistic regression inference endpoints."""

from __future__ import annotations

from typing import Any, Dict

from fastapi import FastAPI, HTTPException

from xgoal_tutor.api.models import (
    ShotPredictionRequest,
    ShotPredictionResponse,
    ShotPredictionWithPromptRequest,
)
from xgoal_tutor.api.services import (
    create_llm_client,
    generate_llm_explanation,
    generate_shot_predictions,
    group_predictions_by_match,
)

app = FastAPI(title="xGoal Inference Service", version="1.0.0")


_LLM_CLIENT = create_llm_client()
_MATCH_CACHE: Dict[str, ShotPredictionResponse] = {}


@app.get("/matches")
def list_matches() -> Dict[str, Any]:
    """Return the collection of matches available to the tutor UI."""

    raise HTTPException(status_code=501, detail="Listing matches is not yet implemented")


@app.get("/matches/{match_id}/lineups")
def get_match_lineups(match_id: str) -> Dict[str, Any]:
    """Return the starting lineups for the requested match."""

    raise HTTPException(status_code=501, detail="Match lineup retrieval is not yet implemented")


@app.post("/matches/{match_id}/summary")
def generate_match_summary(match_id: str) -> Dict[str, Any]:
    """Produce a tactical summary for the provided match for each team "home_team" and "away team" based on
    individual shot summaries from `predict_shots` and summaries for players from (generate_player_summary).
    Output includes "positives", "improvements", "best_player", "improve_player".
    """

    raise HTTPException(status_code=501, detail="Match summary generation is not yet implemented")


@app.post("/matches/{match_id}/players/{player_id}/summary")
def generate_match_player_summary(match_id: str, player_id: str) -> Dict[str, Any]:
    """Produce a tactical summary for the provided player for the match based on
    individual shot summaries from `predict_shots` where the player was involved. Output includes "positives"
    and "improvements".
    """

    raise HTTPException(status_code=501, detail="Match player summary generation is not yet implemented")



@app.post("/predict_shots", response_model=ShotPredictionResponse)
def predict_shots(payload: ShotPredictionRequest) -> ShotPredictionResponse:
    if not payload.shots:
        raise HTTPException(status_code=400, detail="At least one shot must be provided")

    predictions, contributions = generate_shot_predictions(payload.shots, payload.model)

    try:
        llm_response, model_used = generate_llm_explanation(
            _LLM_CLIENT,
            payload.shots,
            predictions,
            contributions,
            llm_model=payload.llm_model,
        )
    except RuntimeError as exc:  # pragma: no cover - network error path
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    response = ShotPredictionResponse(shots=predictions, explanation=llm_response.strip(), llm_model=model_used)

    for match_id, match_predictions in group_predictions_by_match(predictions).items():
        cached_response = ShotPredictionResponse(
            shots=list(match_predictions),
            explanation=response.explanation,
            llm_model=response.llm_model,
        )
        _MATCH_CACHE[match_id] = cached_response

    return response


@app.post("/predict_shots_with_prompt", response_model=ShotPredictionResponse)
def predict_shots_with_prompt(
    payload: ShotPredictionWithPromptRequest,
) -> ShotPredictionResponse:
    if not payload.shots:
        raise HTTPException(status_code=400, detail="At least one shot must be provided")

    predictions, contributions = generate_shot_predictions(payload.shots, payload.model)

    try:
        llm_response, model_used = generate_llm_explanation(
            _LLM_CLIENT,
            payload.shots,
            predictions,
            contributions,
            llm_model=payload.llm_model,
            prompt_override=payload.prompt,
        )
    except RuntimeError as exc:  # pragma: no cover - network error path
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    response = ShotPredictionResponse(
        shots=predictions,
        explanation=llm_response.strip(),
        llm_model=model_used,
    )

    for match_id, match_predictions in group_predictions_by_match(predictions).items():
        cached_response = ShotPredictionResponse(
            shots=list(match_predictions),
            explanation=response.explanation,
            llm_model=response.llm_model,
        )
        _MATCH_CACHE[match_id] = cached_response

    return response


@app.get("/match/{match_id}/shots", response_model=ShotPredictionResponse)
def get_match_shots(match_id: str) -> ShotPredictionResponse:
    """Return cached shot predictions for a single match."""

    cached = _MATCH_CACHE.get(match_id)
    if cached is None:
        raise HTTPException(status_code=404, detail="No cached predictions for match")
    return cached


@app.get("/shots/{shot_id}")
def get_shot_detail(shot_id: str) -> Dict[str, Any]:
    """Return full detail for a specific shot, including positions and explanations."""

    raise HTTPException(status_code=501, detail="Shot detail retrieval is not yet implemented")
