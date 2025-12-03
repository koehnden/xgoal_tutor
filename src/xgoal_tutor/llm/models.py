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


@dataclass
class FreezeFrameEntry:
    player_id: Optional[int]
    player_name: Optional[str]
    position_name: Optional[str]
    teammate: bool
    keeper: bool
    x: Optional[float]
    y: Optional[float]


@dataclass
class MoveSimulationContext:
    block: str
    gain: Optional[float]
    heading_label: Optional[str]


@dataclass
class MatchMetadata:
    home: str
    score_home: str
    score_away: str
    away: str
    competition: str
    season: str
    home_team_id: Optional[int]
    away_team_id: Optional[int]


@dataclass
class EventMetadata:
    period: str
    minute: int
    second: int
    play_pattern: str


@dataclass
class ShooterMetadata:
    name: str
    team_name: str
    position: str
    body_part: str
    technique: str
    start_x: float
    start_y: float
