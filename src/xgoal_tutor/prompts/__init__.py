"""Prompt templates for xGoal tutor LLM interactions."""

from __future__ import annotations

from importlib import resources
from typing import Dict

_TEMPLATE_CACHE: Dict[str, str] = {}


def load_template(name: str) -> str:
    """Load a prompt template by filename, caching the result."""

    if name not in _TEMPLATE_CACHE:
        with resources.files(__name__).joinpath(name).open("r", encoding="utf-8") as file:
            _TEMPLATE_CACHE[name] = file.read()
    return _TEMPLATE_CACHE[name]
