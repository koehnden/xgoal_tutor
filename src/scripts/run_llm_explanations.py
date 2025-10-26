"""Generate natural language explanations for xGoal model outputs using Ollama."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from xgoal_tutor.llm import (
    ExplanationPipeline,
    EventExplanationInput,
    OllamaConfig,
    OllamaLLM,
    normalize_feature_contributions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_file",
        type=Path,
        help="Path to a JSON file containing match metadata and feature contributions.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output path. If omitted the JSON payload is printed to stdout.",
    )
    parser.add_argument(
        "--output-format",
        choices=["json", "markdown"],
        default="json",
        help="Choose between JSON (default) or Markdown output.",
    )
    parser.add_argument(
        "--primary-model",
        default="qwen2.5:7b-instruct-q4_0",
        help="Primary Ollama model to use (default: qwen2.5:7b-instruct-q4_0).",
    )
    parser.add_argument(
        "--fallback-model",
        action="append",
        default=["mistral:7b-instruct-q4_0"],
        help="Fallback model(s) to try if the primary model fails. Can be passed multiple times.",
    )
    parser.add_argument(
        "--host",
        help="Custom Ollama host (default uses http://localhost:11434).",
    )
    parser.add_argument(
        "--top-features",
        type=int,
        default=6,
        help="How many of the strongest features to surface in prompts (default: 6).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Sampling temperature sent to the language model (default: 0.3).",
    )
    return parser.parse_args()


def load_payload(path: Path) -> Tuple[Dict[str, Any], List[EventExplanationInput]]:
    raw = json.loads(path.read_text())
    match_metadata = raw.get("match", {})
    events_raw: Iterable[Dict[str, Any]] = raw.get("events", [])

    events: List[EventExplanationInput] = []
    for idx, item in enumerate(events_raw):
        contributions_raw = _resolve_contributions(item)

        minute, second = _resolve_time_fields(item)

        events.append(
            EventExplanationInput(
                event_id=str(item.get("event_id", idx)),
                minute=minute,
                second=second,
                team=item["team"],
                player=item["player"],
                xg=float(item["xg"]),
                contributions=normalize_feature_contributions(contributions_raw),
                context=item.get("context"),
            )
        )

    if not events:
        raise ValueError("No events found in input file")

    return match_metadata, events


def _resolve_time_fields(item: Dict[str, Any]) -> Tuple[int, int]:
    if "minute" in item and "second" in item:
        return int(item["minute"]), int(item["second"])

    timestamp = item.get("timestamp")
    if isinstance(timestamp, str) and ":" in timestamp:
        parts = timestamp.split(":")
        if len(parts) == 3:  # HH:MM:SS
            _, minute_str, second_str = parts
        else:
            minute_str, second_str = parts[-2:]
        return int(minute_str), int(float(second_str))

    raise ValueError("Events must provide either minute/second integers or a timestamp string")


def _resolve_contributions(item: Dict[str, Any]) -> Any:
    for key in (
        "contributions",
        "feature_importance",
        "coefficients",
        "model_coefficients",
        "shap_values",
    ):
        if key in item and item[key] is not None:
            return item[key]
    raise ValueError(
        "Each event must include one of 'contributions', 'feature_importance', 'coefficients',"
        " 'model_coefficients', or 'shap_values'",
    )


def to_json_payload(result, match_metadata: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "match": match_metadata,
        "models_used": list(dict.fromkeys(result.models_used)),
        "match_summary": result.match_summary,
        "player_summaries": result.player_summaries,
        "team_summaries": result.team_summaries,
        "events": [
            {
                "event_id": explanation.event.event_id,
                "minute": explanation.event.minute,
                "second": explanation.event.second,
                "team": explanation.event.team,
                "player": explanation.event.player,
                "xg": explanation.event.xg,
                "context": explanation.event.context,
                "model_used": explanation.model_used,
                "explanation": explanation.explanation,
            }
            for explanation in result.event_explanations
        ],
    }


def to_markdown(result, match_metadata: Dict[str, Any]) -> str:
    teams = match_metadata.get("teams", {})
    home = teams.get("home", "Home Team")
    away = teams.get("away", "Away Team")
    header = f"# xGoal Match Explanations\n\n**Fixture:** {home} vs {away}\n\n"

    event_lines = []
    for explanation in result.event_explanations:
        event = explanation.event
        time_stamp = f"{event.minute:02d}:{event.second:02d}"
        event_lines.append(
            f"- **{time_stamp} – {event.player} ({event.team})** | xG {event.xg:.3f} | {explanation.explanation}"
        )

    markdown = (
        header
        + "## Match Summary\n\n"
        + result.match_summary
        + "\n\n## Player Summaries\n\n"
        + result.player_summaries
        + "\n\n## Team Summaries\n\n"
        + result.team_summaries
        + "\n\n## Key Moments\n\n"
        + "\n".join(event_lines)
        + "\n"
    )
    return markdown


def main() -> None:
    args = parse_args()
    match_metadata, events = load_payload(args.input_file)

    config = OllamaConfig(
        primary_model=args.primary_model,
        fallback_models=tuple(args.fallback_model or ()),
        host=args.host,
        default_options={"temperature": args.temperature},
    )
    llm = OllamaLLM(config)
    pipeline = ExplanationPipeline(llm, top_features=args.top_features, temperature=args.temperature)
    result = pipeline.run(match_metadata, events)

    if args.output_format == "json":
        payload = to_json_payload(result, match_metadata)
        output_text = json.dumps(payload, indent=2)
    else:
        output_text = to_markdown(result, match_metadata)

    if args.output:
        args.output.write_text(output_text)
    else:
        print(output_text)


if __name__ == "__main__":  # pragma: no cover - manual execution entrypoint
    main()
