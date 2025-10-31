"""Utility functions for communicating with the xGoal Tutor backend API."""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional

import requests
import streamlit as st

DEFAULT_TIMEOUT_SECONDS = 15
DEFAULT_POLL_INTERVAL_SECONDS = 1.0
DEFAULT_POLL_TIMEOUT_SECONDS = 120.0


def get_api_base_url() -> str:
    """Return the configured API base URL with sensible default."""
    return os.getenv("XGOAL_API_BASE_URL", "http://localhost:8000").rstrip("/")


def _build_url(base_url: str, path: str) -> str:
    return f"{base_url}{path if path.startswith('/') else '/' + path}"


def _poll_until_complete(
    request_func,
    *,
    poll_interval: float = DEFAULT_POLL_INTERVAL_SECONDS,
    timeout: float = DEFAULT_POLL_TIMEOUT_SECONDS,
) -> Mapping[str, Any]:
    """Poll a callable returning job metadata until completion."""

    start_time = time.monotonic()
    while True:
        payload: Mapping[str, Any] = request_func()
        status = str(payload.get("status", "")).lower()
        if status == "done":
            return payload
        if status == "failed":
            message = payload.get("error_message") or "Job failed"
            raise RuntimeError(message)
        if (time.monotonic() - start_time) > timeout:
            raise TimeoutError("Job did not finish before the timeout elapsed")
        time.sleep(poll_interval)


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


def generate_match_summary(
    base_url: str,
    match_id: str,
    *,
    poll_interval: float = DEFAULT_POLL_INTERVAL_SECONDS,
    timeout: float = DEFAULT_POLL_TIMEOUT_SECONDS,
) -> Mapping[str, Any]:
    """Trigger match summary generation and return the resulting payload."""

    response = requests.post(
        _build_url(base_url, f"/matches/{match_id}/summary"), timeout=DEFAULT_TIMEOUT_SECONDS
    )
    response.raise_for_status()
    submission = response.json()
    generation_id = submission.get("generation_id")

    def _fetch_status() -> Mapping[str, Any]:
        params = {"generation_id": generation_id} if generation_id else None
        status_response = requests.get(
            _build_url(base_url, f"/matches/{match_id}/summary"),
            params=params,
            timeout=DEFAULT_TIMEOUT_SECONDS,
        )
        status_response.raise_for_status()
        return status_response.json()

    status_payload = _poll_until_complete(
        _fetch_status, poll_interval=poll_interval, timeout=timeout
    )
    result = status_payload.get("result")
    if not isinstance(result, Mapping):
        raise RuntimeError("Summary job completed without a result payload")
    return result


def generate_player_summary(
    base_url: str,
    match_id: str,
    player_id: str,
    *,
    poll_interval: float = DEFAULT_POLL_INTERVAL_SECONDS,
    timeout: float = DEFAULT_POLL_TIMEOUT_SECONDS,
) -> Mapping[str, Any]:
    """Trigger player summary generation for a match and return the result."""

    response = requests.post(
        _build_url(base_url, f"/matches/{match_id}/players/{player_id}/summary"),
        timeout=DEFAULT_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    submission = response.json()
    generation_id = submission.get("generation_id")

    def _fetch_status() -> Mapping[str, Any]:
        params = {"generation_id": generation_id} if generation_id else None
        status_response = requests.get(
            _build_url(base_url, f"/matches/{match_id}/players/{player_id}/summary"),
            params=params,
            timeout=DEFAULT_TIMEOUT_SECONDS,
        )
        status_response.raise_for_status()
        return status_response.json()

    status_payload = _poll_until_complete(
        _fetch_status, poll_interval=poll_interval, timeout=timeout
    )
    result = status_payload.get("result")
    if not isinstance(result, Mapping):
        raise RuntimeError("Player summary job completed without a result payload")
    return result


@st.cache_data(show_spinner=False)
def get_match_shots(base_url: str, match_id: str) -> List[Dict[str, Any]]:
    """Return the list of shot feature payloads for the given match."""

    response = requests.get(
        _build_url(base_url, f"/match/{match_id}/shots"), timeout=DEFAULT_TIMEOUT_SECONDS
    )
    response.raise_for_status()
    payload: Dict[str, Any] = response.json()
    items: Iterable[Dict[str, Any]] = payload.get("items", [])
    return list(items)


def predict_shots(
    base_url: str,
    *,
    shots: Iterable[Mapping[str, Any]],
    model: Mapping[str, Any],
    llm_model: Optional[str] = None,
) -> Mapping[str, Any]:
    """Call the synchronous prediction endpoint for the provided shots."""

    payload: Dict[str, Any] = {"shots": list(shots), "model": dict(model)}
    if llm_model:
        payload["llm_model"] = llm_model

    response = requests.post(
        _build_url(base_url, "/predict_shots"), json=payload, timeout=DEFAULT_TIMEOUT_SECONDS
    )
    response.raise_for_status()
    return response.json()
