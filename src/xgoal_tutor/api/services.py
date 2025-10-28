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
from xgoal_tutor.llm.client import OllamaConfig, OllamaLLM
from xgoal_tutor.llm.pipeline import EventExplanationInput, ExplanationPipeline
from xgoal_tutor.modeling.feature_engineering import build_feature_matrix


class _PromptOnlyLLM:
    """Stub LLM used solely for accessing prompt templates from the pipeline."""

    def generate(self, prompt: str, options: Optional[Dict[str, object]] = None) -> Tuple[str, str]:
        raise RuntimeError("Prompt-only pipeline cannot generate completions")


_PROMPT_PIPELINE = ExplanationPipeline(_PromptOnlyLLM())


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

    if prompt_override is None:
        events = build_event_inputs(shots, predictions, contributions)
        prompt = build_llm_prompt(events)
    else:
        prompt = prompt_override

    return client.generate(prompt, model=llm_model)


def build_event_inputs(
    shots: Sequence[ShotFeatures],
    predictions: Sequence[ShotPrediction],
    contributions: pd.DataFrame,
) -> List[EventExplanationInput]:
    """Translate predictions into the structured inputs expected by the LLM pipeline."""

    events: List[EventExplanationInput] = []
    for index, (shot, prediction) in enumerate(zip(shots, predictions)):
        event_id = prediction.shot_id or f"shot-{index + 1}"
        contribution_row = contributions.iloc[index]
        contribution_map = {
            str(feature): float(value)
            for feature, value in contribution_row.items()
            if not pd.isna(value)
        }
        context = _summarise_shot_context(event_id, shot)
        team_name = prediction.match_id or "Unknown Team"
        player_name = prediction.shot_id or event_id
        events.append(
            EventExplanationInput(
                event_id=event_id,
                minute=0,
                second=0,
                team=team_name,
                player=player_name,
                xg=prediction.xg,
                contributions=contribution_map,
                context=context or None,
            )
        )
    return events


def build_llm_prompt(
    events: Sequence[EventExplanationInput],
    *,
    match_metadata: Optional[Dict[str, object]] = None,
) -> str:
    """Compose the final prompt using the LLM pipeline's event templates."""

    if not events:
        raise ValueError("At least one event is required to build the LLM prompt")

    metadata = match_metadata or {}
    event_prompts = [
        _PROMPT_PIPELINE._build_event_prompt(metadata, event) for event in events
    ]

    if len(event_prompts) == 1:
        return event_prompts[0]

    header = (
        "You will receive multiple shot events. Provide a concise explanation for each, "
        "separating your answers with blank lines and following the guidance in every block."
    )
    return f"{header}\n\n" + "\n\n".join(event_prompts)


def group_predictions_by_match(
    predictions: Iterable[ShotPrediction],
) -> Dict[str, List[ShotPrediction]]:
    """Partition predictions by match identifier for caching."""

    grouped: Dict[str, List[ShotPrediction]] = defaultdict(list)
    for prediction in predictions:
        if prediction.match_id:
            grouped[prediction.match_id].append(prediction)
    return grouped


def _summarise_shot_context(event_id: str, shot: ShotFeatures) -> str:
    """Create a compact textual description of shot metadata for the LLM."""

    parts = [f"Shot ID: {event_id}"]
    if shot.match_id:
        parts.append(f"Match: {shot.match_id}")
    parts.append(f"Start location: ({shot.start_x:.1f}, {shot.start_y:.1f})")

    if shot.body_part:
        parts.append(f"Body part: {shot.body_part}")

    bool_features = {
        "is_set_piece": "Set piece",
        "is_corner": "Corner",
        "is_free_kick": "Free kick",
        "first_time": "First-time finish",
        "under_pressure": "Under pressure",
        "one_on_one": "One-on-one",
        "open_goal": "Open goal",
        "follows_dribble": "After dribble",
        "deflected": "Deflected",
        "aerial_won": "Aerial duel won",
    }

    flags = [label for attr, label in bool_features.items() if getattr(shot, attr, False)]
    if flags:
        parts.append("Traits: " + ", ".join(flags))

    extra_numeric = {
        "ff_keeper_x": shot.ff_keeper_x,
        "ff_keeper_y": shot.ff_keeper_y,
        "ff_opponents": shot.ff_opponents,
        "ff_keeper_count": shot.ff_keeper_count,
        "freeze_frame_available": shot.freeze_frame_available,
    }

    numeric_parts = [
        f"{name.replace('_', ' ').title()}: {value}"
        for name, value in extra_numeric.items()
        if value is not None
    ]
    parts.extend(numeric_parts)

    return " | ".join(parts)
