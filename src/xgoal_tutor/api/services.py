"""Core prediction utilities used by the xGoal inference API."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from xgoal_tutor.api.models import (
    DEFAULT_FALLBACK_MODELS,
    DEFAULT_PRIMARY_MODEL,
    LogisticRegressionModel,
    ReasonCode,
    ShotFeatures,
    ShotPrediction,
)
from xgoal_tutor.llm.client import OllamaConfig, OllamaLLM
from xgoal_tutor.modeling.feature_engineering import build_feature_matrix


_BASE_SUMMARY_PROMPT = (
    "You are xGoal Tutor, an assistant who explains expected goals (xG) "
    "predictions for football shots. Craft a concise narrative for coaches, "
    "highlighting why each shot's probability looks the way it does."
)


def create_llm_client() -> OllamaLLM:
    """Initialise an Ollama client with sensible defaults."""

    config = OllamaConfig(
        primary_model=DEFAULT_PRIMARY_MODEL,
        fallback_models=DEFAULT_FALLBACK_MODELS,
    )
    return OllamaLLM(config)


def build_feature_dataframe(shots: Iterable[ShotFeatures]) -> pd.DataFrame:
    """Convert incoming shot payloads into the engineered feature matrix."""

    raw_records = [shot.model_dump() for shot in shots]
    frame = pd.DataFrame(raw_records)
    return build_feature_matrix(frame)


def calculate_probabilities(
    features: pd.DataFrame, model: LogisticRegressionModel
) -> Tuple[pd.Series, pd.DataFrame]:
    """Apply the logistic regression model to the provided feature matrix."""

    coefficients = model.coefficients
    ordered_columns = list(coefficients.keys())
    aligned = features.reindex(columns=ordered_columns, fill_value=0.0)
    coef_vector = pd.Series(coefficients)
    linear_score = aligned.mul(coef_vector, axis=1).sum(axis=1) + model.intercept
    probabilities = 1.0 / (1.0 + np.exp(-linear_score))
    contributions = aligned.mul(coef_vector, axis=1)
    return probabilities, contributions


def format_reason_codes(
    row_values: pd.Series,
    contributions: pd.Series,
    model: LogisticRegressionModel,
    max_reasons: int = 3,
) -> List[ReasonCode]:
    """Summarise the most influential features for a prediction."""

    reason_codes: List[ReasonCode] = []
    sorted_features = contributions.abs().sort_values(ascending=False)
    for feature in sorted_features.index:
        contribution = contributions[feature]
        if math.isclose(contribution, 0.0, abs_tol=1e-9):
            continue
        value_raw = row_values.get(feature, 0.0)
        value = float(0.0 if pd.isna(value_raw) else value_raw)
        coefficient = float(model.coefficients.get(feature, 0.0))
        reason_codes.append(
            ReasonCode(
                feature=feature,
                value=value,
                coefficient=coefficient,
                contribution=float(contribution),
            )
        )
        if len(reason_codes) >= max_reasons:
            break
    return reason_codes


def build_summary(predictions: Iterable[ShotPrediction]) -> str:
    """Create a short textual overview of the model outputs."""

    lines: List[str] = []
    for index, prediction in enumerate(predictions, start=1):
        shot_label = prediction.shot_id or f"#{index}"
        top_features = ", ".join(
            f"{reason.feature} ({reason.contribution:+.3f})" for reason in prediction.reason_codes
        )
        lines.append(
            f"Shot {shot_label}: xG={prediction.xg:.3f}{'; ' + top_features if top_features else ''}"
        )
    return "\n".join(lines)


def build_llm_prompt(summary: str) -> str:
    """Combine the fixed instruction prompt with the per-request summary."""

    if summary:
        return f"{_BASE_SUMMARY_PROMPT}\n\nShot analytics:\n{summary}"
    return _BASE_SUMMARY_PROMPT


def group_predictions_by_match(
    predictions: Iterable[ShotPrediction],
) -> Dict[str, List[ShotPrediction]]:
    """Partition predictions by match identifier for caching."""

    grouped: Dict[str, List[ShotPrediction]] = defaultdict(list)
    for prediction in predictions:
        if prediction.match_id:
            grouped[prediction.match_id].append(prediction)
    return grouped
