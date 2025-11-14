"""Streamlit client for the xGoal Tutor project."""

from __future__ import annotations

import math
from typing import Callable, List, Optional, TypeVar

import requests
import streamlit as st

from ui.api import fetch_explanation, fetch_goals, fetch_matches
from ui.config import get_base_url
from ui.formatting import format_goal_label, format_match_label
from ui.models import Goal, Match
from ui.pitch import plot_pitch_with_players

T = TypeVar("T")


def _handle_request_errors(operation: Callable[[], T]) -> Optional[T]:
    """Run an operation and convert request errors into Streamlit messages."""

    try:
        return operation()
    except requests.HTTPError as error:
        status_code = error.response.status_code if error.response is not None else "Unknown"
        reason = error.response.reason if error.response is not None else "No response"
        st.error(f"Request failed with status {status_code}: {reason}")
    except requests.RequestException as error:
        st.error(f"Network error: {error}")
    except ValueError as error:
        st.error(str(error))
    return None


def _retry_button() -> None:
    """Provide a retry button that clears cached API responses."""

    if st.button("Retry"):
        st.cache_data.clear()
        st.experimental_rerun()


def main() -> None:
    """Entry point for the Streamlit app."""

    st.set_page_config(page_title="xGoal Tutor", layout="wide")
    st.title("xGoal Tutor — Shot Explanation")

    base_url = get_base_url()
    with st.expander("Settings", expanded=False):
        st.markdown(f"**API base URL:** {base_url}")

    matches = _handle_request_errors(lambda: fetch_matches(base_url))
    if matches is None:
        _retry_button()
        return

    if not matches:
        st.info("No matches available right now. Please try again later.")
        _retry_button()
        return

    match_column, output_column = st.columns([1, 2])

    with match_column:
        if "selected_goal" not in st.session_state:
            st.session_state.selected_goal = None

        def _reset_goal() -> None:
            st.session_state.selected_goal = None

        selected_match: Optional[Match] = st.selectbox(
            "Match",
            options=matches,
            format_func=format_match_label,
            key="selected_match",
            on_change=_reset_goal,
        )

        goals: Optional[List[Goal]] = None
        if selected_match:
            goals = _handle_request_errors(lambda: fetch_goals(base_url, selected_match.match_id))
            if goals is None:
                _retry_button()
                return
            if not goals:
                st.info("No goals recorded for this match yet.")
                st.session_state.selected_goal = None
            else:
                goal_ids = {goal.goal_id for goal in goals}
                current_goal: Optional[Goal] = st.session_state.get("selected_goal")
                if current_goal and current_goal.goal_id not in goal_ids:
                    st.session_state.selected_goal = None
        else:
            st.session_state.selected_goal = None

        goal_disabled = selected_match is None or not goals
        selected_goal: Optional[Goal] = st.selectbox(
            "Goal",
            options=goals or [],
            format_func=format_goal_label,
            key="selected_goal",
            disabled=goal_disabled,
        )

    with output_column:
        if selected_match and selected_goal:
            if st.button("Refresh explanation"):
                st.experimental_rerun()

            explanation = _handle_request_errors(
                lambda: fetch_explanation(base_url, selected_match.match_id, selected_goal.goal_id)
            )
            if explanation is None:
                _retry_button()
                return

            xgoal_percentage = explanation.xgoal_probability * 100
            delta = (explanation.xgoal_probability - 0.5) * 100
            st.metric("xGoal", f"{xgoal_percentage:.1f}%", delta=f"{delta:+.1f}%" if not math.isnan(delta) else None)
            st.write(explanation.explanation)

            fig = plot_pitch_with_players(explanation.pitch, explanation.players)
            st.pyplot(fig)
            st.caption("Pitch (meters, origin bottom-left)")
        else:
            st.info("Select a match and a goal to view the xGoal explanation.")


if __name__ == "__main__":
    main()
