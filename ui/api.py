"""API access functions for the xGoal Tutor UI."""

from __future__ import annotations

from typing import List, Optional

import requests
import streamlit as st

from ui.models import (
    DEFAULT_PITCH_LENGTH,
    DEFAULT_PITCH_WIDTH,
    Explanation,
    Goal,
    Match,
    Pitch,
    Player,
)

REQUEST_TIMEOUT_SECONDS = 6


def _request_json(url: str) -> dict | list:
    """Perform a GET request with timeout and return JSON payload."""

    response = requests.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()


def _convert_match(payload: dict) -> Match:
    return Match(
        match_id=str(payload.get("match_id", "")),
        date=str(payload.get("date", "")),
        competition=str(payload.get("competition", "")),
        home_team=str(payload.get("home_team", "")),
        away_team=str(payload.get("away_team", "")),
    )


def _convert_goal(payload: dict) -> Goal:
    return Goal(
        goal_id=str(payload.get("goal_id", "")),
        minute=int(payload.get("minute", 0)),
        scorer=str(payload.get("scorer", "Unknown")),
        assist=payload.get("assist"),
    )


def _convert_pitch(payload: Optional[dict]) -> Pitch:
    if not isinstance(payload, dict):
        return Pitch()
    length = float(payload.get("length_m", DEFAULT_PITCH_LENGTH))
    width = float(payload.get("width_m", DEFAULT_PITCH_WIDTH))
    return Pitch(length_m=length, width_m=width)


def _convert_player(payload: dict, pitch: Pitch) -> Player:
    x = float(payload.get("x", 0.0))
    y = float(payload.get("y", 0.0))
    clamped_x = max(0.0, min(x, pitch.length_m))
    clamped_y = max(0.0, min(y, pitch.width_m))
    return Player(
        name=str(payload.get("name", "Unknown")),
        team=str(payload.get("team", "Unknown Team")),
        role=str(payload.get("role", "unknown")),
        x=clamped_x,
        y=clamped_y,
    )


@st.cache_data(ttl=300, show_spinner=False)
def fetch_matches(base_url: str) -> List[Match]:
    """Fetch the list of matches from the API."""

    url = f"{base_url}/api/matches"
    payload = _request_json(url)
    if not isinstance(payload, list):
        raise ValueError("Unexpected response when fetching matches")
    matches = [_convert_match(item) for item in payload]
    return [match for match in matches if match.match_id]


@st.cache_data(ttl=300, show_spinner=False)
def fetch_goals(base_url: str, match_id: str) -> List[Goal]:
    """Fetch the goals for a given match."""

    url = f"{base_url}/api/matches/{match_id}/goals"
    payload = _request_json(url)
    if not isinstance(payload, list):
        raise ValueError("Unexpected response when fetching goals")
    goals = [_convert_goal(item) for item in payload]
    return [goal for goal in goals if goal.goal_id]


def fetch_explanation(base_url: str, match_id: str, goal_id: str) -> Explanation:
    """Fetch the shot explanation for a goal."""

    url = f"{base_url}/api/matches/{match_id}/goals/{goal_id}/explanation"
    payload = _request_json(url)
    if not isinstance(payload, dict):
        raise ValueError("Unexpected response when fetching explanation")
    pitch = _convert_pitch(payload.get("pitch"))
    players_payload = payload.get("players", [])
    players = [_convert_player(item, pitch) for item in players_payload if isinstance(item, dict)]
    return Explanation(
        xgoal_probability=float(payload.get("xgoal_probability", 0.0)),
        explanation=str(payload.get("explanation", "")),
        pitch=pitch,
        players=players,
    )


__all__ = [
    "REQUEST_TIMEOUT_SECONDS",
    "fetch_explanation",
    "fetch_goals",
    "fetch_matches",
]
