"""Minimal HTTP client for interacting with a local Ollama server."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Sequence
from urllib import error, request
from urllib.parse import urljoin


@dataclass
class OllamaConfig:
    """Configuration for running prompts against a local Ollama server."""

    primary_model: str
    fallback_models: Sequence[str] = field(default_factory=tuple)
    host: Optional[str] = None
    default_options: Dict[str, object] = field(default_factory=dict)


class OllamaLLM:
    """Utility to send prompts to a locally running Ollama instance."""

    def __init__(self, config: OllamaConfig):
        self._config = config
        self._models: Sequence[str] = (config.primary_model, *config.fallback_models)
        self._default_options: Dict[str, object] = dict(config.default_options)
        base_host = config.host or "http://localhost:11434"
        self._endpoint = urljoin(base_host if base_host.endswith("/") else f"{base_host}/", "api/generate")

    @property
    def models(self) -> Sequence[str]:
        """Return the configured model preference order."""

        return self._models

    def generate(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        options: Optional[Dict[str, object]] = None,
    ) -> tuple[str, str]:
        """Generate a response using the configured models.

        Returns a tuple of (text, model_used).
        """

        options = options or {}
        models_to_try: Iterable[str]
        if model:
            models_to_try = (model,)
        else:
            models_to_try = self._models

        last_error: Optional[Exception] = None
        for model_name in models_to_try:
            try:
                return self._generate_once(model_name, prompt, options)
            except Exception as exc:  # pragma: no cover - network/ollama errors
                last_error = exc

        raise RuntimeError("All configured Ollama models failed to respond") from last_error

    def _generate_once(
        self,
        model_name: str,
        prompt: str,
        options: Dict[str, object],
    ) -> tuple[str, str]:
        payload = {
            "model": model_name,
            "prompt": prompt,
            "options": {**self._default_options, **options},
            "stream": False,
        }
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(
            self._endpoint,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req) as resp:
                response = json.loads(resp.read().decode("utf-8"))
        except error.HTTPError as exc:  # pragma: no cover - network/ollama errors
            raise RuntimeError(f"Ollama returned HTTP {exc.code}: {exc.reason}") from exc
        except error.URLError as exc:  # pragma: no cover - network/ollama errors
            raise RuntimeError("Unable to reach Ollama server") from exc

        if response.get("error"):
            raise RuntimeError(response["error"])

        return response.get("response", ""), response.get("model", model_name)
