"""Data structures used by the language model explanation pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence


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
