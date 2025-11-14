"""Formatting helpers for labels in the Streamlit UI."""

from __future__ import annotations

from ui.models import Goal, Match


def format_match_label(match: Match) -> str:
    """Return the human-friendly label for a match."""

    return f"{match.date} — {match.home_team} vs {match.away_team} ({match.competition})"


def format_goal_label(goal: Goal) -> str:
    """Return the human-friendly label for a goal."""

    assist_part = f"assist: {goal.assist}" if goal.assist else "no assist"
    return f"{goal.minute}' — {goal.scorer} ({assist_part})"


__all__ = ["format_goal_label", "format_match_label"]
