"""Configuration helpers for the Streamlit UI."""

from __future__ import annotations

import os

DEFAULT_BASE_URL = "http://localhost:8000"
ENV_VAR_BASE_URL = "XGOAL_API_BASE_URL"


def get_base_url() -> str:
    """Return the API base URL, defaulting to localhost when unset."""

    return os.environ.get(ENV_VAR_BASE_URL, DEFAULT_BASE_URL).rstrip("/")


__all__ = ["DEFAULT_BASE_URL", "ENV_VAR_BASE_URL", "get_base_url"]
