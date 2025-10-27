"""FastAPI service exposing xGoal logistic regression inference endpoints."""

from __future__ import annotations

from typing import Dict, List

from fastapi import FastAPI, HTTPException

from xgoal_tutor.api.models import (
    ShotPrediction,
    ShotPredictionRequest,
    ShotPredictionResponse,
)
from xgoal_tutor.api.services import (
    build_feature_dataframe,
    build_summary,
    calculate_probabilities,
    create_llm_client,
    format_reason_codes,
    group_predictions_by_match,
)

app = FastAPI(title="xGoal Inference Service", version="1.0.0")


_LLM_CLIENT = create_llm_client()
_MATCH_CACHE: Dict[str, ShotPredictionResponse] = {}


@app.post("/predict_shots", response_model=ShotPredictionResponse)
def predict_shots(payload: ShotPredictionRequest) -> ShotPredictionResponse:
    if not payload.shots:
        raise HTTPException(status_code=400, detail="At least one shot must be provided")

    feature_frame = build_feature_dataframe(payload.shots)
    probabilities, contributions = calculate_probabilities(feature_frame, payload.model)

    predictions: List[ShotPrediction] = []
    for index, shot in enumerate(payload.shots):
        row_contrib = contributions.iloc[index]
        reason_codes = format_reason_codes(feature_frame.iloc[index], row_contrib, payload.model)
        predictions.append(
            ShotPrediction(
                shot_id=shot.shot_id,
                match_id=shot.match_id,
                xg=float(probabilities.iloc[index]),
                reason_codes=reason_codes,
            )
        )

    summary = build_summary(predictions)
    llm_prompt = f"{payload.prompt.strip()}\n\nShot analytics:\n{summary}" if summary else payload.prompt

    try:
        llm_response, model_used = _LLM_CLIENT.generate(llm_prompt, model=payload.llm_model)
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


@app.get("/match/{match_id}/shots", response_model=ShotPredictionResponse)
def get_match_shots(match_id: str) -> ShotPredictionResponse:
    cached = _MATCH_CACHE.get(match_id)
    if cached is None:
        raise HTTPException(status_code=404, detail="No cached predictions for match")
    return cached
