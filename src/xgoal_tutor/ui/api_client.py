"""Utility functions for communicating with the xGoal Tutor backend API."""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Mapping, Optional

import requests
import streamlit as st

DEFAULT_TIMEOUT_SECONDS = 15


def get_api_base_url() -> str:
    """Return the configured API base URL with sensible default."""
    return os.getenv("XGOAL_API_BASE_URL", "http://localhost:8000").rstrip("/")


def _build_url(base_url: str, path: str) -> str:
    return f"{base_url}{path if path.startswith('/') else '/' + path}"


@st.cache_data(show_spinner=False)
def list_matches(base_url: str, *, page_size: int = 100) -> List[Dict[str, Any]]:
    """Fetch all matches using pagination and return them as a list."""
    matches: List[Dict[str, Any]] = []
    page = 1
    while True:
        params = {"page": page, "page_size": page_size}
        response = requests.get(
            _build_url(base_url, "/matches"), params=params, timeout=DEFAULT_TIMEOUT_SECONDS
        )
        response.raise_for_status()
        payload: Dict[str, Any] = response.json()
        items: Iterable[Dict[str, Any]] = payload.get("items", [])
        matches.extend(items)
        total = payload.get("total")
        if total is not None and len(matches) >= int(total):
            break
        if not items:
            break
        page += 1
    return matches


@st.cache_data(show_spinner=False)
def get_match_lineups(base_url: str, match_id: str) -> Dict[str, Any]:
    """Fetch lineups for the provided match."""
    response = requests.get(
        _build_url(base_url, f"/matches/{match_id}/lineups"), timeout=DEFAULT_TIMEOUT_SECONDS
    )
    response.raise_for_status()
    return response.json()


@st.cache_data(show_spinner=False)
def get_match_shots(base_url: str, match_id: str) -> Dict[str, Any]:
    """Return the list of shots (features + context) for the given match."""
    response = requests.get(
        _build_url(base_url, f"/match/{match_id}/shots"),
        timeout=DEFAULT_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    payload: Any = response.json()
    if isinstance(payload, dict) and "items" in payload:
        return payload
    if isinstance(payload, list):
        return {"items": payload}
    return {"items": []}


def enqueue_match_summary(base_url: str, match_id: str) -> Dict[str, Any]:
    """Trigger asynchronous generation of a match summary."""
    response = requests.post(
        _build_url(base_url, f"/matches/{match_id}/summary"),
        timeout=DEFAULT_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def get_match_summary_status(
    base_url: str, match_id: str, *, generation_id: Optional[str] = None
) -> Dict[str, Any]:
    """Retrieve the current status for a match summary generation job."""
    params: Dict[str, Any] = {}
    if generation_id:
        params["generation_id"] = generation_id
    response = requests.get(
        _build_url(base_url, f"/matches/{match_id}/summary"),
        params=params or None,
        timeout=DEFAULT_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def enqueue_player_summary(base_url: str, match_id: str, player_id: str) -> Dict[str, Any]:
    """Trigger asynchronous generation of a player summary for a match."""
    response = requests.post(
        _build_url(base_url, f"/matches/{match_id}/players/{player_id}/summary"),
        timeout=DEFAULT_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def get_player_summary_status(
    base_url: str, match_id: str, player_id: str, *, generation_id: Optional[str] = None
) -> Dict[str, Any]:
    """Retrieve the current status for a player summary generation job."""
    params: Dict[str, Any] = {}
    if generation_id:
        params["generation_id"] = generation_id
    response = requests.get(
        _build_url(base_url, f"/matches/{match_id}/players/{player_id}/summary"),
        params=params or None,
        timeout=DEFAULT_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def predict_shots(base_url: str, payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Call the synchronous shot prediction endpoint."""
    response = requests.post(
        _build_url(base_url, "/predict_shots"),
        json=payload,
        timeout=DEFAULT_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()
