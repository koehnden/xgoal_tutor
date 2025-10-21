"""Pydantic schemas used by the FastAPI service."""

from __future__ import annotations

import json
from typing import Dict, Iterable, List, Mapping, Optional

from pydantic import BaseModel, Field, model_validator


class ShotInput(BaseModel):
    """Minimal features required for a shot."""

    x: float = Field(..., description="Normalized horizontal coordinate of the shot.")
    y: float = Field(..., description="Normalized vertical coordinate of the shot.")
    flags: Dict[str, bool] = Field(
        default_factory=dict,
        description="Arbitrary boolean flags providing context for the shot.",
    )


class PredictShotsRequest(BaseModel):
    """Payload for the /predict_shots endpoint."""

    shots: List[ShotInput] = Field(..., description="Shots to evaluate with the xGoal model.")
    match_id: Optional[str] = Field(
        default=None,
        description=(
            "Optional identifier allowing the predictions to be cached and later "
            "retrieved via the match endpoint."
        ),
    )


class ShotPredictionResponse(BaseModel):
    """Prediction details for a single shot."""

    x: float
    y: float
    flags: Dict[str, bool] = Field(default_factory=dict)
    xg: float = Field(..., ge=0.0, description="Expected goals value for the shot.")
    reason_codes: List[str] = Field(
        default_factory=list,
        description="Model-produced reason codes that justify the prediction.",
    )
    explanation: str = Field(
        default="",
        description="Optional natural-language explanation for the prediction.",
    )


class PredictShotsResponse(BaseModel):
    """Response payload used by the prediction endpoints."""

    match_id: Optional[str] = Field(
        default=None,
        description="Identifier of the match if the predictions are cached.",
    )
    predictions: List[ShotPredictionResponse] = Field(
        default_factory=list,
        description="Per-shot predictions returned by the service.",
    )


class ShotCSVRow(BaseModel):
    """Schema representing a row of user-provided CSV shot data."""

    x: float
    y: float
    flags: Dict[str, bool] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _parse_flags(cls, data: Mapping[str, object] | None) -> Dict[str, object]:
        if data is None:
            return {}
        mutable_data: Dict[str, object] = dict(data)
        raw_flags = mutable_data.get("flags")
        if raw_flags in (None, ""):
            mutable_data["flags"] = {}
            return mutable_data
        if isinstance(raw_flags, dict):
            mutable_data["flags"] = {str(key): bool(value) for key, value in raw_flags.items()}
            return mutable_data
        if isinstance(raw_flags, str):
            text = raw_flags.strip()
            if not text:
                mutable_data["flags"] = {}
                return mutable_data
            try:
                flags = json.loads(text)
            except json.JSONDecodeError as exc:  # pragma: no cover - defensive branch
                raise ValueError("Flags column must contain a JSON object.") from exc
            if not isinstance(flags, dict):
                raise ValueError("Flags column must contain a JSON object.")
            mutable_data["flags"] = {str(key): bool(value) for key, value in flags.items()}
            return mutable_data
        raise ValueError("Unsupported flags format; use a JSON object or leave blank.")


class CSVImportError(BaseModel):
    """Error details returned when CSV validation fails."""

    row: int = Field(..., description="CSV row number (1-indexed) where validation failed.")
    errors: Iterable[Dict[str, object]] = Field(
        ..., description="Pydantic validation errors for the offending row."
    )


__all__ = [
    "CSVImportError",
    "PredictShotsRequest",
    "PredictShotsResponse",
    "ShotCSVRow",
    "ShotInput",
    "ShotPredictionResponse",
]
