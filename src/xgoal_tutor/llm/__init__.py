"""Utilities for interacting with locally served language models via Ollama."""

from xgoal_tutor.llm.client import OllamaConfig, OllamaLLM
from xgoal_tutor.llm.pipeline import (
    ExplanationOutput,
    ExplanationPipeline,
    EventExplanationInput,
    normalize_feature_contributions,
)

__all__ = [
    "OllamaConfig",
    "OllamaLLM",
    "ExplanationPipeline",
    "EventExplanationInput",
    "ExplanationOutput",
    "normalize_feature_contributions",
]
