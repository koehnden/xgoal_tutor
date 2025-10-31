"""FastAPI service exposing xGoal logistic regression inference endpoints."""

from __future__ import annotations
import logging
from typing import Any, Dict, Optional
from fastapi import FastAPI, HTTPException
from fastapi.requests import Request
from fastapi.responses import Response
from uuid import uuid4

from xgoal_tutor.api.models import (
    ShotPredictionRequest,
    ShotPredictionResponse,
)
from xgoal_tutor.api.services import (
    create_llm_client,
    generate_llm_explanation,
    generate_shot_predictions,
    group_predictions_by_match,
)
from xgoal_tutor.api._jobs import get_job, get_latest_job, insert_job
from xgoal_tutor.api._queries import match_exists, player_participated
from xgoal_tutor.api.tasks import queue_match_summary, queue_player_summary

app = FastAPI(title="xGoal Inference Service", version="1.0.0")

_LLM_CLIENT = create_llm_client()
_MATCH_CACHE: Dict[str, ShotPredictionResponse] = {}
_LOGGER = logging.getLogger(__name__)


@app.get("/matches")
def list_matches() -> Dict[str, Any]:
    """
    List matches available to the tutor UI (paged).

    Expected response shape (see swagger.yaml for full schema):
    {
      "items": [
        {
          "id": "SB-0001",
          "label": "FC Barcelona – Chelsea FC (2025-09-21)",
          "kickoff_utc": "2025-09-21T19:00:00Z",
          "competition": "Friendly",
          "season": "2025/26",
          "home_team": {"id": "...", "name": "...", "short_name": "..."},
          "away_team": {"id": "...", "name": "...", "short_name": "..."},
          "venue": "..."
        }
      ],
      "page": 1,
      "page_size": 100,
      "total": 2
    }

    Notes
    -----
    * Wire up pagination params (page, page_size) and return a paged listing for Streamlit dropdowns.
    """
    raise HTTPException(status_code=501, detail="Listing matches is not yet implemented")


@app.get("/matches/{match_id}/lineups")
def get_match_lineups(match_id: str) -> Dict[str, Any]:
    """
    Return the starting lineups for the requested match (home & away).

    Expected response shape (see swagger.yaml for full schema):
    {
      "home": {
        "team": {"id": "...", "name": "...", "short_name": "..."},
        "starters": [{"player": {...}, "is_starter": true, "position_name": "...", "sort_order": 1}],
        "bench": [{"player": {...}, "is_starter": false, "sort_order": 18}]
      },
      "away": {
        "team": {"id": "...", "name": "...", "short_name": "..."},
        "starters": [...],
        "bench": [...]
      }
    }

    Notes
    -----
    * Provide jersey numbers and position names where available to support the Streamlit lineup view.
    """
    raise HTTPException(status_code=501, detail="Match lineup retrieval is not yet implemented")


@app.post("/matches/{match_id}/summary")
def generate_match_summary(match_id: str, request: Request, response: Response) -> Dict[str, Any]:
    if not match_exists(match_id):
        raise HTTPException(status_code=404, detail="Match not found")

    generation_id = uuid4().hex
    try:
        job = insert_job(generation_id, kind="match", match_id=match_id)
        queue_match_summary(generation_id, match_id)
    except Exception as exc:  # pragma: no cover - defensive branch for DB/broker issues
        _LOGGER.exception("Failed to enqueue match summary for %s", match_id)
        raise HTTPException(status_code=500, detail="Internal server error") from exc

    status_url = str(request.url_for("get_match_summary_status", match_id=match_id)) + f"?generation_id={generation_id}"
    response.headers["Location"] = status_url
    response.status_code = 202
    return {
        "generation_id": generation_id,
        "match_id": match_id,
        "status": job.status,
        "status_url": status_url,
        "enqueued_at": job.created_at,
    }


@app.get("/matches/{match_id}/summary")
def get_match_summary_status(match_id: str, generation_id: Optional[str] = None) -> Dict[str, Any]:
    if generation_id:
        job = get_job(generation_id)
    else:
        job = get_latest_job("match", match_id)

    if job is None or job.kind != "match" or job.match_id != match_id:
        raise HTTPException(status_code=404, detail="Match summary job not found")

    result = job.result if job.status == "done" else None
    return {
        "generation_id": job.generation_id,
        "match_id": job.match_id,
        "status": job.status,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
        "expires_at": job.expires_at,
        "result": result,
        "error_message": job.error_message,
    }


@app.post("/matches/{match_id}/players/{player_id}/summary")
def generate_match_player_summary(
    match_id: str,
    player_id: str,
    request: Request,
    response: Response,
) -> Dict[str, Any]:
    if not match_exists(match_id):
        raise HTTPException(status_code=404, detail="Match not found")
    if not player_participated(match_id, player_id):
        raise HTTPException(status_code=404, detail="Player not found for match")

    generation_id = uuid4().hex
    try:
        job = insert_job(generation_id, kind="player", match_id=match_id, player_id=player_id)
        queue_player_summary(generation_id, match_id, player_id)
    except Exception as exc:  # pragma: no cover - defensive branch for DB/broker issues
        _LOGGER.exception("Failed to enqueue player summary for %s/%s", match_id, player_id)
        raise HTTPException(status_code=500, detail="Internal server error") from exc

    status_url = str(
        request.url_for(
            "get_match_player_summary_status",
            match_id=match_id,
            player_id=player_id,
        )
    ) + f"?generation_id={generation_id}"
    response.headers["Location"] = status_url
    response.status_code = 202
    return {
        "generation_id": generation_id,
        "match_id": match_id,
        "player_id": player_id,
        "status": job.status,
        "status_url": status_url,
        "enqueued_at": job.created_at,
    }


@app.get("/matches/{match_id}/players/{player_id}/summary")
def get_match_player_summary_status(
    match_id: str,
    player_id: str,
    generation_id: Optional[str] = None,
) -> Dict[str, Any]:
    if generation_id:
        job = get_job(generation_id)
    else:
        job = get_latest_job("player", match_id, player_id)

    if (
        job is None
        or job.kind != "player"
        or job.match_id != match_id
        or job.player_id != player_id
    ):
        raise HTTPException(status_code=404, detail="Player summary job not found")

    result = job.result if job.status == "done" else None
    return {
        "generation_id": job.generation_id,
        "match_id": job.match_id,
        "player_id": job.player_id,
        "status": job.status,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
        "expires_at": job.expires_at,
        "result": result,
        "error_message": job.error_message,
    }


@app.post("/predict_shots", response_model=ShotPredictionResponse)
def predict_shots(payload: ShotPredictionRequest) -> ShotPredictionResponse:
    """
    Synchronous xG prediction with a managed prompt (and optional override via the request model).

    Notes
    -----
    * This endpoint remains implemented as before.
    * If your request model now has `prompt_override`, you may later thread it into
      `generate_llm_explanation(..., prompt_override=payload.prompt_override)`—but do not change
      the behavior here until you wire up the model/schema accordingly.
    """
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


@app.get("/match/{match_id}/shots")
def list_match_shot_features(match_id: str) -> Dict[str, Any]:
    """
    List shots for a match (features + light context; no predictions).

    Description
    -----------
    Returns the **features** required to call `/predict_shots` plus light UI context:
    - period, minute, second
    - result (Goal/Saved/Off Target/Blocked/Post/Bar/Own Goal/Unknown)
    - scoreline_before {home, away}, scoreline_after {home, away}
    - shooter {player_id, player_name, team_id, team_name}
    - features (ShotFeatures)

    Expected response shape (see swagger.yaml for full schema):
    {
      "items": [
        {
          "period": "2H",
          "minute": 78,
          "second": 4,
          "result": "Goal",
          "scoreline_before": {"home": 1, "away": 1},
          "scoreline_after": {"home": 2, "away": 1},
          "shooter": {"player_id": "...", "player_name": "...", "team_id": "...", "team_name": "..."},
          "features": {
            "shot_id": "...",
            "match_id": "...",
            "start_x": 108.5,
            "start_y": 38.0,
            "is_set_piece": false,
            "is_corner": false,
            "is_free_kick": false,
            "first_time": true,
            "under_pressure": true,
            "body_part": "Right Foot",
            "ff_keeper_x": 117.5,
            "ff_keeper_y": 42.0,
            "ff_opponents": 3,
            "freeze_frame_available": 1,
            "ff_keeper_count": 1,
            "one_on_one": false,
            "open_goal": false,
            "follows_dribble": false,
            "deflected": false,
            "aerial_won": false
          }
        }
      ]
    }

    Notes
    -----
    * This endpoint intentionally returns no predictions or explanations; the client should pipe the
      `features` array into `POST /predict_shots` as needed.
    """
    raise HTTPException(status_code=501, detail="Match shot feature listing is not yet implemented")
