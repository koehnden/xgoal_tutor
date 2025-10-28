"""Utilities for interacting with locally served language models via Ollama."""

from xgoal_tutor.llm.client import OllamaConfig, OllamaLLM
from xgoal_tutor.llm.models import ExplanationOutput, EventExplanationInput
from xgoal_tutor.llm.pipeline import ExplanationPipeline, normalize_feature_contributions

__all__ = [
    "OllamaConfig",
    "OllamaLLM",
    "EventExplanationInput",
    "ExplanationOutput",
    "ExplanationPipeline",
    "normalize_feature_contributions",
]
