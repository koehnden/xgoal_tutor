"""Prompt building utilities for local xGoal model explanations."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from xgoal_tutor.llm.client import OllamaLLM


@dataclass
class EventExplanationInput:
    """Structured information for a single model output to explain."""

    event_id: str
    minute: int
    second: int
    team: str
    player: str
    xg: float
    contributions: Dict[str, float]
    context: Optional[str] = None


@dataclass
class EventExplanationResult:
    """Resulting natural-language explanation for an event."""

    event: EventExplanationInput
    explanation: str
    model_used: str


@dataclass
class ExplanationOutput:
    """Aggregated explanations for a full match."""

    match_summary: str
    player_summaries: str
    team_summaries: str
    event_explanations: List[EventExplanationResult]
    models_used: Sequence[str]


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
            prompt = self._build_event_prompt(match_metadata, event)
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

        match_summary_prompt = self._build_match_summary_prompt(match_metadata, event_results)
        match_summary, model_used = self._llm.generate(
            match_summary_prompt,
            options={"temperature": self._temperature},
        )
        models_used.append(model_used)

        player_summary_prompt = self._build_player_summary_prompt(match_metadata, event_results)
        player_summaries, model_used = self._llm.generate(
            player_summary_prompt,
            options={"temperature": self._temperature},
        )
        models_used.append(model_used)

        team_summary_prompt = self._build_team_summary_prompt(match_metadata, event_results)
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

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------
    def _build_event_prompt(
        self,
        match_metadata: Dict[str, object],
        event: EventExplanationInput,
    ) -> str:
        feature_block = self._format_features(event.contributions)
        context_block = f"Additional context: {event.context}\n" if event.context else ""
        teams = match_metadata.get("teams", {})
        home = teams.get("home", "Home Team")
        away = teams.get("away", "Away Team")
        competition = match_metadata.get("competition", "")
        season = match_metadata.get("season", "")

        header_lines: List[str] = [
            "You are an analyst translating xGoal model outputs into plain football language.",
            "Explain how the listed features influenced the shot's expected goals.",
            "Focus on tactical insight and avoid repeating raw numbers verbatim.",
            "Keep the explanation under 120 words and speak to a coach or tactics nerd.",
        ]
        header = "\n".join(header_lines)

        prompt = (
            f"{header}\n\n"
            f"Match: {home} vs {away} ({competition} {season}).\n"
            f"Event time: {event.minute:02d}:{event.second:02d}.\n"
            f"Player: {event.player} ({event.team}).\n"
            f"Model xG: {event.xg:.3f}.\n"
            f"{context_block}"
            "Feature contributions (positive increases xG, negative lowers it):\n"
            f"{feature_block}\n"
            "Provide a concise explanation referencing the most influential factors."
        )
        return prompt

    def _build_match_summary_prompt(
        self,
        match_metadata: Dict[str, object],
        events: Sequence[EventExplanationResult],
    ) -> str:
        lines: List[str] = []
        for result in events:
            event = result.event
            lines.append(
                f"- {event.minute:02d}:{event.second:02d} {event.team} - {event.player}: {result.explanation}"
            )

        teams = match_metadata.get("teams", {})
        home = teams.get("home", "Home Team")
        away = teams.get("away", "Away Team")
        competition = match_metadata.get("competition", "Competition")
        season = match_metadata.get("season", "")

        prompt = (
            "You are preparing a post-match tactical summary for coaching staff.\n"
            "Using the event explanations below, highlight the overall attacking patterns, key"
            " tactical themes, and how chance quality evolved across the match."
            " Provide 3-4 short paragraphs.\n\n"
            f"Match: {home} vs {away} ({competition} {season}).\n"
            "Important attacking moments:\n"
            + "\n".join(lines)
        )
        return prompt

    def _build_player_summary_prompt(
        self,
        match_metadata: Dict[str, object],
        events: Sequence[EventExplanationResult],
    ) -> str:
        per_player: Dict[str, List[str]] = {}
        for result in events:
            per_player.setdefault(result.event.player, []).append(
                f"{result.event.minute:02d}:{result.event.second:02d} - xG {result.event.xg:.3f}: {result.explanation}"
            )

        lines = []
        for player, notes in sorted(per_player.items()):
            joined = "\n".join(f"    {note}" for note in notes)
            lines.append(f"Player: {player}\n{joined}")

        prompt = (
            "You are writing player-facing feedback for an attacking coach. Summarise each player's"
            " involvement using the provided notes. Be specific about movement, decision making,"
            " and shot quality. Use bullet lists per player and keep the tone constructive.\n\n"
            + "\n\n".join(lines)
        )
        return prompt

    def _build_team_summary_prompt(
        self,
        match_metadata: Dict[str, object],
        events: Sequence[EventExplanationResult],
    ) -> str:
        per_team: Dict[str, List[str]] = {}
        for result in events:
            per_team.setdefault(result.event.team, []).append(
                f"{result.event.minute:02d}:{result.event.second:02d} {result.event.player}: {result.explanation}"
            )

        teams = match_metadata.get("teams", {})
        home = teams.get("home", "Home Team")
        away = teams.get("away", "Away Team")

        lines = []
        for team, notes in per_team.items():
            joined = "\n".join(f"    {note}" for note in notes)
            lines.append(f"Team: {team}\n{joined}")

        prompt = (
            "Provide a tactical summary for each team covering chance creation patterns, build-up"
            " tendencies, and finishing quality. Use the notes below and relate them to broader"
            " team-level trends. Aim for 2-3 bullet points per team.\n\n"
            f"Fixture: {home} vs {away}.\n\n"
            + "\n\n".join(lines)
        )
        return prompt

    def _format_features(self, contributions: Dict[str, float]) -> str:
        sorted_items = sorted(
            contributions.items(),
            key=lambda item: abs(item[1]),
            reverse=True,
        )
        lines = []
        for feature, value in sorted_items[: self._top_features]:
            if value > 0:
                direction = "higher xG"
            elif value < 0:
                direction = "lower xG"
            else:
                direction = "neutral"
            lines.append(f"- {feature}: {value:+.3f} ({direction})")
        return "\n".join(lines)
