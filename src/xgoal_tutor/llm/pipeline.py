"""Prompt building utilities for local xGoal model explanations."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Dict, List, Sequence

from xgoal_tutor.llm.client import OllamaLLM
from xgoal_tutor.llm.models import (
    EventExplanationInput,
    EventExplanationResult,
    ExplanationOutput,
)
from xgoal_tutor.llm.prompts import (
    build_event_prompt,
    build_match_summary_prompt,
    build_player_summary_prompt,
    build_team_summary_prompt,
)


def normalize_feature_contributions(raw: Any) -> Dict[str, float]:
    """Convert assorted model output formats into a feature-to-value mapping.

    The xGoal models can emit plain dictionaries, coefficient lists, or SHAP payloads.
    This helper keeps the pipeline agnostic by normalising these structures into a
    single ``Dict[str, float]`` representation.
    """

    if isinstance(raw, Mapping):
        if "feature_names" in raw and "shap_values" in raw:
            return _parse_shap_payload(raw)
        return {str(key): float(value) for key, value in raw.items()}

    if isinstance(raw, Iterable) and not isinstance(raw, (str, bytes)):
        entries: List[Mapping[str, Any]] = []
        for entry in raw:
            if isinstance(entry, Mapping):
                entries.append(entry)
            else:
                raise ValueError("List contributions must contain mappings with feature keys")

        if entries and all("feature" in entry for entry in entries):
            contributions: Dict[str, float] = {}
            for entry in entries:
                feature = str(entry["feature"])
                value = None
                for key in ("coefficient", "value", "importance", "contribution"):
                    if key in entry:
                        value = entry[key]
                        break
                if value is None:
                    raise ValueError(
                        "Coefficient-style contributions must include a numeric value field",
                    )
                contributions[feature] = float(value)
            return contributions

    raise ValueError("Unsupported contribution format; expected dict, SHAP payload, or list")


def _parse_shap_payload(payload: Mapping[str, Any]) -> Dict[str, float]:
    feature_names = payload.get("feature_names")
    shap_values = payload.get("shap_values")

    if not isinstance(feature_names, Iterable) or isinstance(feature_names, (str, bytes)):
        raise ValueError("SHAP payload must include a feature_names iterable")

    vector = _normalise_shap_values(shap_values)
    feature_list = [str(name) for name in feature_names]

    if len(vector) != len(feature_list):
        raise ValueError("Number of SHAP values does not match feature_names length")

    contributions = {name: float(value) for name, value in zip(feature_list, vector)}

    expected_value = payload.get("expected_value")
    if expected_value is not None:
        contributions["__expected_value__"] = float(expected_value)

    base_values = payload.get("base_values")
    if isinstance(base_values, Iterable) and not isinstance(base_values, (str, bytes)):
        base_values = next(iter(base_values), None)
    if base_values is not None:
        contributions["__base_value__"] = float(base_values)

    return contributions


def _normalise_shap_values(values: Any) -> List[float]:
    if isinstance(values, Mapping):
        values = values.get("values") or values.get("data")

    if isinstance(values, Iterable) and not isinstance(values, (str, bytes)):
        values_list = list(values)
        if values_list and isinstance(values_list[0], Iterable) and not isinstance(
            values_list[0], (str, bytes)
        ):
            # ``shap`` often returns ``[[...]]`` even for a single sample
            return [float(v) for v in list(values_list[0])]
        return [float(v) for v in values_list]

    raise ValueError("Unable to interpret SHAP values array")


class ExplanationPipeline:
    """Generate natural language explanations for xGoal model outputs."""

    def __init__(
        self,
        llm: OllamaLLM,
        *,
        top_features: int = 6,
        temperature: float = 0.3,
    ) -> None:
        self._llm = llm
        self._top_features = top_features
        self._temperature = temperature

    def run(
        self,
        match_metadata: Dict[str, object],
        events: Sequence[EventExplanationInput],
    ) -> ExplanationOutput:
        models_used: List[str] = []
        event_results: List[EventExplanationResult] = []

        for event in events:
            prompt = build_event_prompt(
                match_metadata,
                event,
                top_features=self._top_features,
            )
            text, model_used = self._llm.generate(
                prompt,
                options={"temperature": self._temperature},
            )
            models_used.append(model_used)
            event_results.append(
                EventExplanationResult(
                    event=event,
                    explanation=text.strip(),
                    model_used=model_used,
                )
            )

        match_summary_prompt = build_match_summary_prompt(match_metadata, event_results)
        match_summary, model_used = self._llm.generate(
            match_summary_prompt,
            options={"temperature": self._temperature},
        )
        models_used.append(model_used)

        player_summary_prompt = build_player_summary_prompt(match_metadata, event_results)
        player_summaries, model_used = self._llm.generate(
            player_summary_prompt,
            options={"temperature": self._temperature},
        )
        models_used.append(model_used)

        team_summary_prompt = build_team_summary_prompt(match_metadata, event_results)
        team_summaries, model_used = self._llm.generate(
            team_summary_prompt,
            options={"temperature": self._temperature},
        )
        models_used.append(model_used)

        return ExplanationOutput(
            match_summary=match_summary.strip(),
            player_summaries=player_summaries.strip(),
            team_summaries=team_summaries.strip(),
            event_explanations=event_results,
            models_used=models_used,
        )

    # Prompt builders are now defined in ``prompts.py``.
