"""Pydantic models used by the xGoal inference API."""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


def _list_field(*, description: str):
    field = Field(default_factory=list, description=description)
    if getattr(field, "default_factory", None) is None:
        field = Field(default=list(), description=description)
    return field


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
    has_cutback: Optional[bool] = None
    cutback_target_x: Optional[float] = None
    cutback_target_y: Optional[float] = None

    class Config:
        extra = "forbid"


class ReasonCode(BaseModel):
    """Feature contribution that influenced the xG estimate."""

    feature: str
    value: float
    coefficient: float
    contribution: float


class ShotPrediction(BaseModel):
    """xG estimate, contributing factors, and explanation for a single shot."""

    shot_id: Optional[str]
    match_id: Optional[str]
    xg: float
    reason_codes: List[ReasonCode]
    team_mate_in_better_position_count: Optional[int] = Field(
        default=None, description="Count of teammates whose simulated xG exceeds the shooter"
    )
    max_teammate_xgoal_diff: Optional[float] = Field(
        default=None,
        description="Shooter xG minus the best simulated teammate xG",
    )
    teammate_name_with_max_xgoal: Optional[str] = Field(
        default=None, description="Name of the teammate with the highest simulated xG"
    )
    explanation: Optional[str] = Field(
        default=None,
        description="Natural-language explanation generated for this shot",
    )


class ShotPredictionRequest(BaseModel):
    """Payload accepted by the /predict_shots endpoint."""

    shots: List[ShotFeatures] = _list_field(
        description="Inline shot feature payloads to score immediately",
    )
    shot_ids: List[str] = _list_field(
        description="Identifiers of existing shots to retrieve features for before scoring",
    )
    model: Optional[LogisticRegressionModel] = Field(
        default=None,
        description=(
            "Optional logistic regression model overriding the default baseline parameters"
        ),
    )
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
    llm_model: str


# Allowed models are declared here to keep validation next to the request model.
DEFAULT_PRIMARY_MODEL = "qwen2.5:7b-instruct-q4_0"
DEFAULT_FALLBACK_MODELS: tuple[str, ...] = ("mistral:7b-instruct-q4_0",)
ALLOWED_LLM_MODELS = {DEFAULT_PRIMARY_MODEL, *DEFAULT_FALLBACK_MODELS}


DEFAULT_LOGISTIC_REGRESSION_MODEL = LogisticRegressionModel(
    intercept=-1.3291,
    coefficients={
        "dist_sb": -0.0793,
        "angle_deg_sb": 0.0521,
        "is_set_piece": 0.1845,
        "is_corner": 0.0623,
        "is_free_kick": 0.2714,
        "first_time": 0.0896,
        "under_pressure": -0.1287,
        "is_header": -0.2458,
        "gk_depth_sb": -0.0143,
        "gk_offset_sb": -0.0098,
        "ff_opponents": -0.0431,
        "one_on_one": 0.6142,
        "open_goal": 1.1421,
        "follows_dribble": 0.1475,
        "deflected": -0.1218,
        "aerial_won": -0.0836,
        "has_cutback": 0.0225,
        "first_time_miss": -0.0582,
        "one_on_one_miss": -0.0419,
        "open_goal_miss": -0.0527,
        "follows_dribble_miss": -0.0375,
        "deflected_miss": -0.0331,
        "aerial_won_miss": -0.0249,
        "under_pressure_miss": -0.0294,
    },
)
