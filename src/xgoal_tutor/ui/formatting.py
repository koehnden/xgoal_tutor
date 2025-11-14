"""Formatting helpers for the Streamlit UI."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional


def build_match_option_label(match: Mapping[str, Any]) -> str:
    """Return a human-friendly label for a match option."""
    label = match.get("label")
    if isinstance(label, str) and label.strip():
        return label
    home = match.get("home_team", {}).get("name", "Home")
    away = match.get("away_team", {}).get("name", "Away")
    kickoff = match.get("kickoff_utc")
    if isinstance(kickoff, str):
        return f"{home} vs {away} ({kickoff})"
    return f"{home} vs {away}"


def format_scoreline(result: Mapping[str, Any]) -> str:
    """Format the final scoreline for display."""
    home = result.get("home_team", {}).get("name", "Home")
    away = result.get("away_team", {}).get("name", "Away")
    score_home = result.get("score_home", "?")
    score_away = result.get("score_away", "?")
    return f"{home} {score_home} – {score_away} {away}"


def format_goal_events(shots: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Extract goal events from shots and format them for tabular display."""
    events: List[Dict[str, Any]] = []
    for shot in shots:
        result_value = shot.get("result") or shot.get("outcome")
        is_goal = str(result_value).lower() == "goal" if result_value else shot.get("is_goal")
        if not is_goal:
            continue
        shooter = shot.get("shooter") or shot.get("player", {})
        player = shooter.get("player_name") or shooter.get("name") or "Unknown"
        team = shooter.get("team_name") or shooter.get("team", {}).get("name") or "Team"
        minute = shot.get("minute")
        second = shot.get("second")
        display_time: str
        if isinstance(minute, int) and isinstance(second, int):
            display_time = f"{minute:02d}:{second:02d}"
        elif isinstance(minute, int):
            display_time = f"{minute}'"
        else:
            display_time = shot.get("clock") or "?"
        score_after = shot.get("score_after") or shot.get("scoreline_after")
        if isinstance(score_after, Mapping):
            score_home = score_after.get("home")
            score_away = score_after.get("away")
        else:
            score_home = shot.get("score_home")
            score_away = shot.get("score_away")
        scoreline = None
        if score_home is not None and score_away is not None:
            scoreline = f"{score_home}:{score_away}"
        events.append(
            {
                "Time": display_time,
                "Team": team,
                "Scorer": player,
                "Score": scoreline or "-",
            }
        )
    return events


def extract_player_summary(
    summary: Mapping[str, Any], player_id: str
) -> Optional[Mapping[str, Any]]:
    """Extract the summary entry for a player from the match summary payload."""
    for key in ("player_summaries", "players", "playerInsights", "player_insights"):
        players = summary.get(key)
        if isinstance(players, list):
            for item in players:
                player = item.get("player") if isinstance(item, Mapping) else None
                if isinstance(player, Mapping) and player.get("id") == player_id:
                    return item
    return None


def extract_player_shot_ids(player_summary: Mapping[str, Any]) -> List[str]:
    """Return the shot identifiers referenced by the summary entry."""
    shot_ids: List[str] = []
    for key in ("evidence_shot_ids", "shot_ids", "shots"):
        value = player_summary.get(key)
        if isinstance(value, list):
            for entry in value:
                if isinstance(entry, str):
                    shot_ids.append(entry)
                elif isinstance(entry, Mapping) and "shot_id" in entry:
                    shot_ids.append(str(entry["shot_id"]))
    return shot_ids


def extract_insight_texts(items: Optional[Iterable[Mapping[str, Any]]]) -> List[str]:
    """Return a list of plain-text insight strings."""
    if items is None:
        return []
    texts: List[str] = []
    for item in items:
        if not isinstance(item, Mapping):
            continue
        text = item.get("explanation") or item.get("text")
        if isinstance(text, str) and text.strip():
            texts.append(text.strip())
    return texts


def describe_player_highlight(highlight: Optional[Mapping[str, Any]]) -> Optional[str]:
    """Return a formatted description of a highlighted player."""
    if not isinstance(highlight, Mapping):
        return None
    player_data = highlight.get("player") or {}
    player_name = player_data.get("name") or player_data.get("player_name")
    rationale = highlight.get("explanation") or highlight.get("rationale")
    if player_name and rationale:
        return f"**{player_name}** — {rationale}"
    if player_name:
        return f"**{player_name}**"
    return None


def build_shot_option_label(shot: Mapping[str, Any]) -> str:
    """Create a concise label for a shot selector option."""
    minute = shot.get("minute")
    second = shot.get("second")
    clock = shot.get("clock")
    if isinstance(minute, int) and isinstance(second, int):
        time_part = f"{minute:02d}:{second:02d}"
    elif isinstance(clock, str) and clock:
        time_part = clock
    elif isinstance(minute, int):
        time_part = f"{minute}'"
    else:
        time_part = "?"
    shooter = shot.get("shooter") or {}
    player_name = shooter.get("player_name") or shooter.get("name") or "Unknown"
    team_name = shooter.get("team_name") or shooter.get("team", {}).get("name")
    result_value = shot.get("result") or shot.get("outcome")
    is_goal = str(result_value).lower() == "goal" if isinstance(result_value, str) else shot.get("is_goal")
    outcome = "Goal" if is_goal else (result_value or "Shot")
    xg_value = shot.get("xg")
    if not isinstance(xg_value, (int, float)):
        xg_value = shot.get("model_prediction", {}).get("xg")
    xg_part = f"xG {xg_value:.2f}" if isinstance(xg_value, (int, float)) else ""
    score_after = shot.get("score_after") or shot.get("scoreline_after")
    score_home = score_after.get("home") if isinstance(score_after, Mapping) else None
    score_away = score_after.get("away") if isinstance(score_after, Mapping) else None
    if is_goal and score_home is not None and score_away is not None:
        score_part = f" {score_home}:{score_away}"
    else:
        score_part = ""
    team_part = f" – {team_name}" if team_name else ""
    parts = [time_part, player_name, outcome]
    label = ", ".join(parts)
    if xg_part:
        label = f"{label} ({xg_part})"
    return f"{label}{score_part}{team_part}"
