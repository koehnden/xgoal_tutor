"""FastAPI service exposing xGoal logistic regression inference endpoints."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException

try:  # pragma: no cover - support lightweight FastAPI stubs in tests
    from fastapi import Query
except (ImportError, AttributeError):  # pragma: no cover - fallback for simplified fastapi
    def Query(default: Any = ..., **_: Any) -> Any:  # type: ignore
        return default

from xgoal_tutor.api.models import (
    DEFAULT_LOGISTIC_REGRESSION_MODEL,
    DEFAULT_PRIMARY_MODEL,
    LogisticRegressionModel,
    ShotFeatures,
    ShotPrediction,
    ShotPredictionRequest,
    ShotPredictionResponse,
)
from xgoal_tutor.api.services import (
    create_llm_client,
    generate_llm_explanation,
    generate_shot_predictions,
    group_predictions_by_match,
)

from xgoal_tutor.api._lineups import get_match_lineups as load_match_lineups
from xgoal_tutor.api._matches import list_matches as fetch_matches
from xgoal_tutor.api._shots import (
    list_match_shot_features as fetch_match_shots,
    load_shot_features_by_ids,
)

app = FastAPI(title="xGoal Inference Service", version="1.0.0")

_LLM_CLIENT = create_llm_client()
_MATCH_CACHE: Dict[str, ShotPredictionResponse] = {}


@app.get("/matches")
def list_matches(
    page: int = Query(1, ge=1), page_size: int = Query(100, ge=1, le=200)
) -> Dict[str, Any]:
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
    return fetch_matches(page=page, page_size=page_size)


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
    return load_match_lineups(match_id)


@app.post("/matches/{match_id}/summary")
def generate_match_summary(match_id: str) -> Dict[str, Any]:
    """
    Enqueue asynchronous tactical summary generation for the match.

    Description
    -----------
    Produces team-level insights for both sides ("home_team", "away_team") using shot-level
    predictions/explanations and player summaries. Output (when ready) includes:
    - positives (list of {explanation, evidence_shot_ids})
    - improvements (list of {explanation, evidence_shot_ids})
    - best_player ({player, explanation, evidence_shot_ids})
    - improve_player ({player, explanation, evidence_shot_ids})

    Returns
    -------
    202 Accepted with body:
    {
      "generation_id": "...",
      "match_id": "...",
      "status": "queued" | "running",
      "status_url": "http://.../matches/{match_id}/summary?generation_id=...",
      "enqueued_at": "RFC3339"
    }

    Notes
    -----
    * This is a placeholder; integrate your job queue and return 202 accordingly.
    """
    raise HTTPException(status_code=501, detail="Match summary generation is not yet implemented")


@app.get("/matches/{match_id}/summary")
def get_match_summary_status(match_id: str, generation_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Poll the status of a match summary job.

    Parameters
    ----------
    match_id : str
        Identifier of the match whose summary job should be queried.
    generation_id : Optional[str]
        Optional job identifier. If omitted, return the latest job for this match (if any).

    Returns
    -------
    200 OK with:
    {
      "generation_id": "...",
      "match_id": "...",
      "status": "queued" | "running" | "done" | "failed",
      "created_at": "RFC3339",
      "updated_at": "RFC3339",
      "expires_at": "RFC3339 | null",
      "result": {  # present iff status == "done"
        "result": { "home_team": {...}, "away_team": {...}, "score_home": 2, "score_away": 1, "final": true },
        "home_insights": {...},  # uses `explanation` fields consistently
        "away_insights": {...},
        "generated_at": "RFC3339",
        "llm_model": "..."
      },
      "error_message": "..." | null
    }

    Notes
    -----
    * Placeholder: look up your job store and return 404 if no job exists.
    """
    raise HTTPException(status_code=501, detail="Match summary status retrieval is not yet implemented")


@app.post("/matches/{match_id}/players/{player_id}/summary")
def generate_match_player_summary(match_id: str, player_id: str) -> Dict[str, Any]:
    """
    Enqueue asynchronous player summary generation for the given match/player.

    Description
    -----------
    Produces player-level insights based on shot-level predictions/explanations where the player
    was involved. Output (when ready) includes:
    - positives (list of {explanation, evidence_shot_ids})
    - improvements (list of {explanation, evidence_shot_ids})

    Returns
    -------
    202 Accepted with body:
    {
      "generation_id": "...",
      "match_id": "...",
      "player_id": "...",
      "status": "queued" | "running",
      "status_url": "http://.../matches/{match_id}/players/{player_id}/summary?generation_id=...",
      "enqueued_at": "RFC3339"
    }

    Notes
    -----
    * Placeholder: integrate queue/enqueue logic and return 202; currently 501.
    """
    raise HTTPException(status_code=501, detail="Match player summary generation is not yet implemented")


@app.get("/matches/{match_id}/players/{player_id}/summary")
def get_match_player_summary_status(match_id: str, player_id: str, generation_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Poll the status of a player match summary job.

    Parameters
    ----------
    match_id : str
        Match identifier.
    player_id : str
        Player identifier within the match context.
    generation_id : Optional[str]
        Optional job identifier. If omitted, return the latest job for (match_id, player_id) if any.

    Returns
    -------
    200 OK with:
    {
      "generation_id": "...",
      "match_id": "...",
      "player_id": "...",
      "status": "queued" | "running" | "done" | "failed",
      "created_at": "RFC3339",
      "updated_at": "RFC3339",
      "expires_at": "RFC3339 | null",
      "result": {   # present iff status == "done"
        "player": {...},
        "team": {...},
        "positives": [{"explanation": "...", "evidence_shot_ids": ["..."]}],
        "improvements": [{"explanation": "...", "evidence_shot_ids": ["..."]}],
        "generated_at": "RFC3339",
        "llm_model": "..."
      },
      "error_message": "..." | null
    }

    Notes
    -----
    * Placeholder: read from job store and return 404 if not found.
    """
    raise HTTPException(status_code=501, detail="Match player summary status retrieval is not yet implemented")


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
    shots: List[ShotFeatures] = list(payload.shots)
    if payload.shot_ids:
        fetched = load_shot_features_by_ids(payload.shot_ids)
        missing = [shot_id for shot_id in payload.shot_ids if shot_id not in fetched]
        if missing:
            missing_list = ", ".join(sorted(missing))
            raise HTTPException(status_code=404, detail=f"Shot IDs not found: {missing_list}")
        shots.extend(fetched[shot_id] for shot_id in payload.shot_ids)

    if not shots:
        raise HTTPException(status_code=400, detail="At least one shot must be provided")

    model = payload.model or DEFAULT_LOGISTIC_REGRESSION_MODEL
    if model is DEFAULT_LOGISTIC_REGRESSION_MODEL:
        model = LogisticRegressionModel(**DEFAULT_LOGISTIC_REGRESSION_MODEL.model_dump())
    predictions, contributions = generate_shot_predictions(shots, model)

    try:
        llm_responses, model_used = generate_llm_explanation(
            _LLM_CLIENT,
            shots,
            predictions,
            contributions,
            llm_model=payload.llm_model,
        )
    except RuntimeError as exc:  # pragma: no cover - network error path
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    resolved_llm_model = model_used or payload.llm_model or DEFAULT_PRIMARY_MODEL

    predictions_with_explanations: List[ShotPrediction] = []
    for index, prediction in enumerate(predictions):
        if hasattr(prediction, "model_dump"):
            data = prediction.model_dump()
        else:
            data = prediction.dict()  # type: ignore[attr-defined]
        data["explanation"] = llm_responses[index]
        predictions_with_explanations.append(ShotPrediction(**data))

    response = ShotPredictionResponse(
        shots=predictions_with_explanations,
        llm_model=resolved_llm_model,
    )

    for match_id, match_predictions in group_predictions_by_match(predictions_with_explanations).items():
        cached_response = ShotPredictionResponse(
            shots=[
                ShotPrediction(
                    **(
                        prediction.model_dump()
                        if hasattr(prediction, "model_dump")
                        else prediction.dict()  # type: ignore[attr-defined]
                    )
                )
                for prediction in match_predictions
            ],
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
    return fetch_match_shots(match_id)
