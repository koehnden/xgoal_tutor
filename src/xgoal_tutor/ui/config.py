"""Configuration helpers for the Streamlit UI."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import streamlit as st

LOGREG_MODEL_JSON_ENV = "XGOAL_LOGREG_MODEL_JSON"
LOGREG_MODEL_PATH_ENV = "XGOAL_LOGREG_MODEL_PATH"
LLM_MODEL_ENV = "XGOAL_UI_LLM_MODEL"


def _coerce_float(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value)
    raise TypeError(f"Cannot convert {value!r} to float")


def _normalise_model(payload: Mapping[str, Any]) -> Dict[str, Any]:
    intercept_raw = payload.get("intercept")
    coefficients_raw = payload.get("coefficients")
    if coefficients_raw is None:
        raise ValueError("coefficients are required")
    if not isinstance(coefficients_raw, Mapping):
        raise TypeError("coefficients must be a mapping of feature -> value")

    intercept = _coerce_float(intercept_raw)
    coefficients: Dict[str, float] = {}
    for feature, value in coefficients_raw.items():
        coefficients[str(feature)] = _coerce_float(value)

    if not coefficients:
        raise ValueError("At least one coefficient must be provided")

    return {"intercept": intercept, "coefficients": coefficients}


def _load_json(value: str) -> Mapping[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise ValueError("Could not parse logistic model JSON") from exc
    if not isinstance(parsed, Mapping):
        raise ValueError("Logistic model JSON must describe an object")
    return parsed


def _load_model_from_path(path_value: str) -> Optional[Dict[str, Any]]:
    path = Path(path_value)
    if not path.exists():
        return None
    data = path.read_text(encoding="utf-8")
    return _normalise_model(_load_json(data))


@st.cache_resource(show_spinner=False)
def get_default_logistic_model() -> Optional[Dict[str, Any]]:
    """Return the configured logistic regression model if available."""

    secret_value: Any = st.secrets.get("logistic_model") if hasattr(st, "secrets") else None
    if secret_value:
        if isinstance(secret_value, str):
            payload = _load_json(secret_value)
        elif isinstance(secret_value, Mapping):
            payload = secret_value
        else:
            raise TypeError("st.secrets['logistic_model'] must be a mapping or JSON string")
        return _normalise_model(payload)

    json_env = os.getenv(LOGREG_MODEL_JSON_ENV)
    if json_env:
        return _normalise_model(_load_json(json_env))

    path_env = os.getenv(LOGREG_MODEL_PATH_ENV)
    if path_env:
        return _load_model_from_path(path_env)

    return None


@st.cache_resource(show_spinner=False)
def get_llm_model_override() -> Optional[str]:
    """Return the configured LLM model override if provided."""

    value = os.getenv(LLM_MODEL_ENV)
    if value and value.strip():
        return value.strip()
    secret = st.secrets.get("llm_model") if hasattr(st, "secrets") else None
    if isinstance(secret, str) and secret.strip():
        return secret.strip()
    return None
