"""Data models for the xGoal Tutor Streamlit UI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

DEFAULT_PITCH_LENGTH = 105.0
DEFAULT_PITCH_WIDTH = 68.0


@dataclass(frozen=True)
class Match:
    """Basic metadata describing a match."""

    match_id: str
    date: str
    competition: str
    home_team: str
    away_team: str


@dataclass(frozen=True)
class Goal:
    """A goal within a match."""

    goal_id: str
    minute: int
    scorer: str
    assist: Optional[str]


@dataclass(frozen=True)
class Pitch:
    """Pitch dimensions in meters."""

    length_m: float = DEFAULT_PITCH_LENGTH
    width_m: float = DEFAULT_PITCH_WIDTH


@dataclass(frozen=True)
class Player:
    """Player positioning information."""

    name: str
    team: str
    role: str
    x: float
    y: float


@dataclass(frozen=True)
class Explanation:
    """Shot explanation details."""

    xgoal_probability: float
    explanation: str
    pitch: Pitch
    players: List[Player]


__all__ = [
    "DEFAULT_PITCH_LENGTH",
    "DEFAULT_PITCH_WIDTH",
    "Explanation",
    "Goal",
    "Match",
    "Pitch",
    "Player",
]
