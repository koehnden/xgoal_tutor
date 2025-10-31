"""Shared data structures for the xGoal inference API.

This module hosts both the public Pydantic schemas surfaced by the HTTP layer and
lightweight dataclasses that the asynchronous orchestration code relies on.
Centralising the dataclasses here keeps import relationships straightforward and
avoids sprinkling duplicate type declarations throughout the API package.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Dataclasses used internally by the asynchronous job pipeline
# ---------------------------------------------------------------------------


@dataclass
class JobRecord:
    """Structured representation of a stored background job."""

    generation_id: str
    kind: str
    match_id: str
    player_id: Optional[str]
    status: str
    created_at: str
    updated_at: str
    expires_at: Optional[str]
    result: Optional[Dict[str, object]]
    error_message: Optional[str]


@dataclass
class MatchInfo:
    """Key metadata describing a football match."""

    match_id: str
    home_team_id: str
    away_team_id: str
    home_team_name: str
    away_team_name: str
    competition: Optional[str]
    season: Optional[str]
    venue: Optional[str]
    match_date: Optional[str]
    label: Optional[str]


@dataclass
class PlayerInfo:
    """Lineup metadata for a participant in a match."""

    player_id: str
    name: str
    team_id: Optional[str]
    jersey_number: Optional[int]
    position: Optional[str]


@dataclass
class ShotRow:
    """Row returned from the shots table enriched with freeze-frame aggregates."""

    shot_id: str
    match_id: str
    team_id: Optional[str]
    opponent_team_id: Optional[str]
    player_id: Optional[str]
    player_name: Optional[str]
    jersey_number: Optional[int]
    position_name: Optional[str]
    period: Optional[int]
    minute: Optional[int]
    second: Optional[float]
    is_goal: Optional[int]
    score_home: Optional[int]
    score_away: Optional[int]
    start_x: float
    start_y: float
    is_set_piece: Optional[int]
    is_corner: Optional[int]
    is_free_kick: Optional[int]
    first_time: Optional[int]
    under_pressure: Optional[int]
    body_part: Optional[str]
    one_on_one: Optional[int]
    open_goal: Optional[int]
    follows_dribble: Optional[int]
    deflected: Optional[int]
    aerial_won: Optional[int]
    freeze_frame_available: Optional[int]
    ff_keeper_x: Optional[float]
    ff_keeper_y: Optional[float]
    ff_keeper_count: Optional[int]
    ff_opponents: Optional[int]


@dataclass
class ShotContext:
    """Container linking raw shot data with its prediction."""

    row: ShotRow
    prediction: "ShotPrediction"


class LogisticRegressionModel(BaseModel):
    """Parameters describing a trained logistic regression model."""

    intercept: float = Field(..., description="Intercept term of the logistic regression model")
    coefficients: Dict[str, float] = Field(
        ..., description="Mapping from feature name to coefficient value"
    )

    @field_validator("coefficients")
    @classmethod
    def _ensure_non_empty(cls, value: Dict[str, float]) -> Dict[str, float]:
        if not value:
            raise ValueError("At least one coefficient must be provided")
        return value


class ShotFeatures(BaseModel):
    """Minimal set of fields required to compute xGoal features."""

    shot_id: Optional[str] = Field(
        default=None,
        description="Identifier of the shot within the match or dataset",
    )
    match_id: Optional[str] = Field(
        default=None,
        description="Identifier of the match the shot belongs to",
    )
    start_x: float
    start_y: float
    is_set_piece: Optional[bool] = False
    is_corner: Optional[bool] = False
    is_free_kick: Optional[bool] = False
    first_time: Optional[bool] = False
    under_pressure: Optional[bool] = False
    body_part: Optional[str] = None
    ff_keeper_x: Optional[float] = None
    ff_keeper_y: Optional[float] = None
    ff_opponents: Optional[float] = None
    freeze_frame_available: Optional[int] = None
    ff_keeper_count: Optional[int] = None
    one_on_one: Optional[bool] = None
    open_goal: Optional[bool] = None
    follows_dribble: Optional[bool] = None
    deflected: Optional[bool] = None
    aerial_won: Optional[bool] = None

    class Config:
        extra = "forbid"


class ReasonCode(BaseModel):
    """Feature contribution that influenced the xG estimate."""

    feature: str
    value: float
    coefficient: float
    contribution: float


class ShotPrediction(BaseModel):
    """xG estimate and contributing factors for a single shot."""

    shot_id: Optional[str]
    match_id: Optional[str]
    xg: float
    reason_codes: List[ReasonCode]


class ShotPredictionRequest(BaseModel):
    """Payload accepted by the /predict_shots endpoint."""

    shots: List[ShotFeatures]
    model: LogisticRegressionModel
    llm_model: Optional[str] = Field(
        default=None,
        description="Optional override for the Ollama model to use for explanations",
    )

    @field_validator("llm_model")
    @classmethod
    def _validate_llm_model(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        if value not in ALLOWED_LLM_MODELS:
            raise ValueError(
                "llm_model must be one of: " + ", ".join(sorted(ALLOWED_LLM_MODELS))
            )
        return value


class ShotPredictionWithPromptRequest(ShotPredictionRequest):
    """Payload for /predict_shots_with_prompt allowing manual prompt overrides."""

    prompt: str = Field(
        ..., description="Custom instruction block sent directly to the language model"
    )


class ShotPredictionResponse(BaseModel):
    """Response returned by /predict_shots."""

    shots: List[ShotPrediction]
    explanation: str
    llm_model: str


# Allowed models are declared here to keep validation next to the request model.
DEFAULT_PRIMARY_MODEL = "qwen2.5:7b-instruct-q4_0"
DEFAULT_FALLBACK_MODELS: tuple[str, ...] = ("mistral:7b-instruct-q4_0",)
ALLOWED_LLM_MODELS = {DEFAULT_PRIMARY_MODEL, *DEFAULT_FALLBACK_MODELS}
