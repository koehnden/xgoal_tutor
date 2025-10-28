"""Prompt builders used by the explanation pipeline."""

from __future__ import annotations

from typing import Dict, List, Sequence

from xgoal_tutor.llm.models import EventExplanationInput, EventExplanationResult


def build_event_prompt(
    match_metadata: Dict[str, object],
    event: EventExplanationInput,
    *,
    top_features: int,
) -> str:
    feature_block = _format_features(event.contributions, top_features)
    context_block = f"Additional context: {event.context}\n" if event.context else ""
    teams = match_metadata.get("teams", {})
    home = teams.get("home", "Home Team")
    away = teams.get("away", "Away Team")
    competition = match_metadata.get("competition", "")
    season = match_metadata.get("season", "")

    header_lines: List[str] = [
        "You are an analyst translating xGoal model outputs into plain football language.",
        "Explain how the listed features influenced the shot's expected goals.",
        "Focus on tactical insight and avoid repeating raw numbers verbatim.",
        "Keep the explanation under 120 words and speak to a coach or tactics nerd.",
    ]
    header = "\n".join(header_lines)

    prompt = (
        f"{header}\n\n"
        f"Match: {home} vs {away} ({competition} {season}).\n"
        f"Event time: {event.minute:02d}:{event.second:02d}.\n"
        f"Player: {event.player} ({event.team}).\n"
        f"Model xG: {event.xg:.3f}.\n"
        f"{context_block}"
        "Feature contributions (positive increases xG, negative lowers it):\n"
        f"{feature_block}\n"
        "Provide a concise explanation referencing the most influential factors."
    )
    return prompt


def build_match_summary_prompt(
    match_metadata: Dict[str, object],
    events: Sequence[EventExplanationResult],
) -> str:
    lines: List[str] = []
    for result in events:
        event = result.event
        lines.append(
            f"- {event.minute:02d}:{event.second:02d} {event.team} - {event.player}: {result.explanation}"
        )

    teams = match_metadata.get("teams", {})
    home = teams.get("home", "Home Team")
    away = teams.get("away", "Away Team")
    competition = match_metadata.get("competition", "Competition")
    season = match_metadata.get("season", "")

    prompt = (
        "You are preparing a post-match tactical summary for coaching staff.\n"
        "Using the event explanations below, highlight the overall attacking patterns, key"
        " tactical themes, and how chance quality evolved across the match."
        " Provide 3-4 short paragraphs.\n\n"
        f"Match: {home} vs {away} ({competition} {season}).\n"
        "Important attacking moments:\n"
        + "\n".join(lines)
    )
    return prompt


def build_player_summary_prompt(
    match_metadata: Dict[str, object],
    events: Sequence[EventExplanationResult],
) -> str:
    per_player: Dict[str, List[str]] = {}
    for result in events:
        per_player.setdefault(result.event.player, []).append(
            f"{result.event.minute:02d}:{result.event.second:02d} - xG {result.event.xg:.3f}: {result.explanation}"
        )

    lines = []
    for player, notes in sorted(per_player.items()):
        joined = "\n".join(f"    {note}" for note in notes)
        lines.append(f"Player: {player}\n{joined}")

    prompt = (
        "You are writing player-facing feedback for an attacking coach. Summarise each player's"
        " involvement using the provided notes. Be specific about movement, decision making,"
        " and shot quality. Use bullet lists per player and keep the tone constructive.\n\n"
        + "\n\n".join(lines)
    )
    return prompt


def build_team_summary_prompt(
    match_metadata: Dict[str, object],
    events: Sequence[EventExplanationResult],
) -> str:
    per_team: Dict[str, List[str]] = {}
    for result in events:
        per_team.setdefault(result.event.team, []).append(
            f"{result.event.minute:02d}:{result.event.second:02d} {result.event.player}: {result.explanation}"
        )

    teams = match_metadata.get("teams", {})
    home = teams.get("home", "Home Team")
    away = teams.get("away", "Away Team")

    lines = []
    for team, notes in per_team.items():
        joined = "\n".join(f"    {note}" for note in notes)
        lines.append(f"Team: {team}\n{joined}")

    prompt = (
        "Provide a tactical summary for each team covering chance creation patterns, build-up"
        " tendencies, and finishing quality. Use the notes below and relate them to broader"
        " team-level trends. Aim for 2-3 bullet points per team.\n\n"
        f"Fixture: {home} vs {away}.\n\n"
        + "\n\n".join(lines)
    )
    return prompt


def _format_features(contributions: Dict[str, float], top_features: int) -> str:
    sorted_items = sorted(
        contributions.items(),
        key=lambda item: abs(item[1]),
        reverse=True,
    )
    lines = []
    for feature, value in sorted_items[:top_features]:
        if value > 0:
            direction = "higher xG"
        elif value < 0:
            direction = "lower xG"
        else:
            direction = "neutral"
        lines.append(f"- {feature}: {value:+.3f} ({direction})")
    return "\n".join(lines)
