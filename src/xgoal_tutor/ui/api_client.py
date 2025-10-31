"""Utility functions for communicating with the xGoal Tutor backend API."""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Optional

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


def generate_match_summary(base_url: str, match_id: str) -> Dict[str, Any]:
    """Trigger match summary generation and return the resulting payload."""
    response = requests.post(
        _build_url(base_url, f"/matches/{match_id}/summary"), timeout=DEFAULT_TIMEOUT_SECONDS
    )
    response.raise_for_status()
    return response.json()


@st.cache_data(show_spinner=False)
def get_match_shots(
    base_url: str, match_id: str, *, include: Optional[str] = None, goals_only: bool = False
) -> List[Dict[str, Any]]:
    """Return the list of shots for the given match."""
    params: Dict[str, Any] = {"page_size": 200, "page": 1}
    if include:
        params["include"] = include
    if goals_only:
        params["goals_only"] = str(goals_only).lower()
    shots: List[Dict[str, Any]] = []
    while True:
        response = requests.get(
            _build_url(base_url, f"/matches/{match_id}/shots"),
            params=params,
            timeout=DEFAULT_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload: Dict[str, Any] = response.json()
        items: Iterable[Dict[str, Any]] = payload.get("items", [])
        shots.extend(items)
        total = payload.get("total")
        if total is not None and len(shots) >= int(total):
            break
        if not items:
            break
        params["page"] = params.get("page", 1) + 1
    return shots


@st.cache_data(show_spinner=False)
def get_shot_detail(base_url: str, shot_id: str) -> Dict[str, Any]:
    """Fetch detail for a shot including positions and explanations."""
    params = {"include": "positions,model_prediction,explanation"}
    response = requests.get(
        _build_url(base_url, f"/shots/{shot_id}"), params=params, timeout=DEFAULT_TIMEOUT_SECONDS
    )
    response.raise_for_status()
    return response.json()
