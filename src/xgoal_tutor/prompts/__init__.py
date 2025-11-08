"""Prompt template utilities backed by Markdown + Jinja2 files."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from typing import Any, Dict, Iterable, Mapping, Tuple

import yaml
from jinja2 import Environment, Template

_TEMPLATE_CACHE: Dict[str, "PromptTemplate"] = {}
_JINJA_ENV = Environment(autoescape=False, trim_blocks=False, lstrip_blocks=False, keep_trailing_newline=True)


@dataclass(frozen=True)
class PromptTemplate:
    """Parsed prompt template with YAML metadata and a Jinja2 body."""

    name: str
    metadata: Mapping[str, Any]
    template: Template

    @property
    def required_fields(self) -> Tuple[str, ...]:
        requires = self.metadata.get("requires", ())
        if isinstance(requires, str):  # pragma: no cover - defensive guard
            return (requires,)
        if isinstance(requires, Iterable):
            return tuple(str(field) for field in requires)
        return ()

    def render(self, context: Mapping[str, Any]) -> str:
        missing = [field for field in self.required_fields if field not in context]
        if missing:
            raise KeyError(f"Missing required fields {missing} for template '{self.name}'")

        defaults = {
            key: value
            for key, value in self.metadata.items()
            if key not in {"requires", "notes", "id", "version", "description"}
        }
        merged = {**defaults, **context}
        return self.template.render(**merged)


def load_template(name: str) -> PromptTemplate:
    """Load a prompt template by filename, caching the parsed metadata and body."""

    if name not in _TEMPLATE_CACHE:
        package_file = resources.files(__name__).joinpath(name)
        with package_file.open("r", encoding="utf-8") as file:
            raw_text = file.read()
        metadata, body = _parse_template(raw_text)
        template = _JINJA_ENV.from_string(body)
        _TEMPLATE_CACHE[name] = PromptTemplate(name=name, metadata=metadata, template=template)
    return _TEMPLATE_CACHE[name]


def _parse_template(text: str) -> Tuple[Dict[str, Any], str]:
    lines = text.splitlines()
    if not lines:
        return {}, ""

    if lines[0].strip() != "---":
        return {}, text

    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            front_matter = "\n".join(lines[1:index])
            body = "\n".join(lines[index + 1 :])
            metadata = yaml.safe_load(front_matter) or {}
            if not isinstance(metadata, dict):  # pragma: no cover - defensive guard
                raise ValueError("Template front matter must be a mapping")
            return metadata, body

    raise ValueError("Unterminated YAML front matter in template")
