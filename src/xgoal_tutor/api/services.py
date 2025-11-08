"""Core prediction utilities used by the xGoal inference API."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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
from xgoal_tutor.api._database import get_db
from xgoal_tutor.llm.client import OllamaConfig, OllamaLLM
from xgoal_tutor.llm.xgoal_prompt_builder import build_xgoal_prompt
from xgoal_tutor.modeling.feature_engineering import build_feature_matrix


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


def generate_shot_predictions(
    shots: Sequence[ShotFeatures], model: LogisticRegressionModel
) -> Tuple[List[ShotPrediction], pd.DataFrame]:
    """Produce model predictions and contribution breakdowns for the provided shots."""

    feature_frame = build_feature_dataframe(shots)
    probabilities, contributions = calculate_probabilities(feature_frame, model)

    predictions: List[ShotPrediction] = []
    for index, shot in enumerate(shots):
        row_contrib = contributions.iloc[index]
        reason_codes = format_reason_codes(feature_frame.iloc[index], row_contrib, model)
        predictions.append(
            ShotPrediction(
                shot_id=shot.shot_id,
                match_id=shot.match_id,
                xg=float(probabilities.iloc[index]),
                reason_codes=reason_codes,
            )
        )

    return predictions, contributions


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


def generate_llm_explanation(
    client: OllamaLLM,
    shots: Sequence[ShotFeatures],
    predictions: Sequence[ShotPrediction],
    contributions: pd.DataFrame,
    *,
    llm_model: Optional[str] = None,
    prompt_override: Optional[str] = None,
) -> Tuple[str, str]:
    """Create an explanation for the predictions via the configured LLM."""

    prompt: str
    if prompt_override is not None:
        prompt = prompt_override
    else:
        prompt = _build_xgoal_prompt(shots, predictions, contributions)

    return client.generate(prompt, model=llm_model)


def group_predictions_by_match(
    predictions: Iterable[ShotPrediction],
) -> Dict[str, List[ShotPrediction]]:
    """Partition predictions by match identifier for caching."""

    grouped: Dict[str, List[ShotPrediction]] = defaultdict(list)
    for prediction in predictions:
        if prediction.match_id:
            grouped[prediction.match_id].append(prediction)
    return grouped


def _build_xgoal_prompt(
    shots: Sequence[ShotFeatures],
    predictions: Sequence[ShotPrediction],
    contributions: pd.DataFrame,
) -> str:
    """Render the xGoal Markdown prompt(s) for the provided shots."""

    if len(shots) != len(predictions):
        raise ValueError("Shots and predictions length mismatch")

    if len(contributions) < len(shots):
        raise ValueError("Missing contribution rows for the provided shots")

    prompts: List[str] = []
    with get_db() as connection:
        for index, (shot, prediction) in enumerate(zip(shots, predictions)):
            shot_id = shot.shot_id or prediction.shot_id
            if not shot_id:
                raise ValueError("Each shot must include a shot_id to build prompts")

            contribution_row = contributions.iloc[index]
            feature_block = _format_feature_block(contribution_row)

            prompts.append(
                build_xgoal_prompt(
                    connection,
                    str(shot_id),
                    feature_block=feature_block,
                )
            )

    if len(prompts) == 1:
        return prompts[0]

    separator = "\n\n---\n\n"
    intro = (
        "You will receive multiple shot prompts below. Provide a distinct explanation for each, "
        "respecting every block's guidance."
    )
    return f"{intro}\n\n" + separator.join(prompts)


def _format_feature_block(contribution_row: pd.Series, limit: int = 5) -> List[str]:
    """Convert a contribution row into formatted prompt bullet points."""

    sortable = (
        contribution_row.dropna()
        if hasattr(contribution_row, "dropna")
        else pd.Series(contribution_row)
    )

    if not isinstance(sortable, pd.Series):
        sortable = pd.Series(sortable)

    sortable = sortable[[col for col in sortable.index if not str(col).startswith("__")]]
    ordered = sortable.reindex(sortable.abs().sort_values(ascending=False).index)

    lines: List[str] = []
    for feature, value in ordered.iloc[:limit].items():
        if math.isclose(float(value), 0.0, abs_tol=1e-9):
            continue
        arrow = "↑" if value > 0 else "↓"
        lines.append(f"{arrow} {feature} ({value:+.3f})")

    return lines
