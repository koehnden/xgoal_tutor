"""Streamlit app for the xGoal Tutor project."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

import pandas as pd
import requests
import streamlit as st

from xgoal_tutor.ui.api_client import (
    generate_match_summary,
    get_api_base_url,
    get_match_lineups,
    get_match_shots,
    get_shot_detail,
    list_matches,
)
from xgoal_tutor.ui.formatting import (
    build_match_option_label,
    build_shot_option_label,
    describe_player_highlight,
    extract_insight_texts,
    extract_player_shot_ids,
    extract_player_summary,
    format_goal_events,
    format_scoreline,
)
from xgoal_tutor.ui.plots import create_shot_positions_figure


def _render_error(message: str, error: Exception) -> None:
    st.error(f"{message}: {error}")


def _fetch_matches(base_url: str) -> List[Mapping[str, Any]]:
    try:
        return list_matches(base_url)
    except requests.RequestException as exc:
        _render_error("Unable to load matches", exc)
        return []


def _fetch_match_summary(base_url: str, match_id: str) -> Optional[Mapping[str, Any]]:
    try:
        with st.spinner("Generating match summary..."):
            return generate_match_summary(base_url, match_id)
    except requests.RequestException as exc:
        _render_error("Could not generate match summary", exc)
        return None


def _fetch_match_lineups(base_url: str, match_id: str) -> Optional[Mapping[str, Any]]:
    try:
        return get_match_lineups(base_url, match_id)
    except requests.RequestException as exc:
        _render_error("Failed to retrieve lineups", exc)
        return None


def _fetch_match_shots(base_url: str, match_id: str) -> List[Mapping[str, Any]]:
    try:
        return get_match_shots(base_url, match_id, include="model_prediction,explanation")
    except requests.RequestException as exc:
        _render_error("Unable to load shots for the match", exc)
        return []


def _fetch_shot_detail(base_url: str, shot_id: str) -> Optional[Mapping[str, Any]]:
    try:
        return get_shot_detail(base_url, shot_id)
    except requests.RequestException as exc:
        _render_error("Unable to load shot detail", exc)
        return None


def _render_team_insights(title: str, insights: Optional[Mapping[str, Any]]) -> None:
    st.subheader(title)
    if not isinstance(insights, Mapping):
        st.info("No insights available yet.")
        return
    scoreline = insights.get("team", {}).get("name")
    if scoreline:
        st.markdown(f"**Team:** {scoreline}")
    positives = extract_insight_texts(insights.get("positives"))
    improvements = extract_insight_texts(insights.get("improvements"))
    if positives:
        st.markdown("**What went well**")
        for item in positives:
            st.markdown(f"- {item}")
    if improvements:
        st.markdown("**Opportunities to improve**")
        for item in improvements:
            st.markdown(f"- {item}")
    highlight_best = describe_player_highlight(insights.get("best_player"))
    highlight_improve = describe_player_highlight(insights.get("improve_player"))
    if highlight_best or highlight_improve:
        st.markdown("**Highlighted players**")
        if highlight_best:
            st.markdown(f"- Best performer: {highlight_best}")
        if highlight_improve:
            st.markdown(f"- Biggest opportunity: {highlight_improve}")


def _render_lineup_table(lineup: Optional[Mapping[str, Any]], title: str) -> None:
    st.markdown(f"### {title}")
    if not isinstance(lineup, Mapping):
        st.info("Lineup unavailable.")
        return
    team_name = lineup.get("team", {}).get("name", title)
    st.markdown(f"**{team_name}**")
    starters = _lineup_entries_to_dataframe(lineup.get("starters"))
    bench = _lineup_entries_to_dataframe(lineup.get("bench"))
    if not starters.empty:
        st.markdown("Starters")
        st.dataframe(starters, hide_index=True, use_container_width=True)
    else:
        st.info("No starters data provided.")
    if not bench.empty:
        st.markdown("Bench")
        st.dataframe(bench, hide_index=True, use_container_width=True)


def _lineup_entries_to_dataframe(entries: Any) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if isinstance(entries, list):
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            player = entry.get("player", {})
            rows.append(
                {
                    "Player": player.get("name"),
                    "#": player.get("jersey_number") or entry.get("jersey_number"),
                    "Position": entry.get("position_name") or player.get("position"),
                    "Starter": entry.get("is_starter"),
                }
            )
    return pd.DataFrame(rows)


def _collect_players_from_lineups(lineups: Optional[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    players: Dict[str, Dict[str, Any]] = {}
    if not isinstance(lineups, Mapping):
        return []
    for side in ("home", "away"):
        team_section = lineups.get(side)
        if not isinstance(team_section, Mapping):
            continue
        team_name = team_section.get("team", {}).get("name", side.title())
        for group in ("starters", "bench"):
            for entry in team_section.get(group, []) or []:
                if not isinstance(entry, Mapping):
                    continue
                player = entry.get("player")
                if not isinstance(player, Mapping):
                    continue
                player_id = player.get("id")
                if not player_id:
                    continue
                players[player_id] = {
                    "player": player,
                    "team_name": team_name,
                }
    return list(players.values())


def _render_player_drilldown(
    *,
    lineups: Optional[Mapping[str, Any]],
    summary: Optional[Mapping[str, Any]],
    shots_by_id: Mapping[str, Mapping[str, Any]],
) -> None:
    st.subheader("Player drilldown")
    player_records = _collect_players_from_lineups(lineups)
    if not player_records:
        st.info("Player information unavailable for this match.")
        return
    default_index = 0 if player_records else None
    selected = st.selectbox(
        "Select a player",
        options=player_records,
        format_func=lambda item: f"{item['player'].get('name')} ({item['team_name']})",
        index=default_index,
        key="player_drilldown",
    )
    if not selected:
        return
    player = selected["player"]
    player_id = player.get("id")
    st.markdown(f"#### {player.get('name')} – {selected['team_name']}")
    player_summary = extract_player_summary(summary or {}, player_id) if player_id else None
    if player_summary:
        positives = extract_insight_texts(player_summary.get("positives"))
        improvements = extract_insight_texts(player_summary.get("improvements"))
        if positives:
            st.markdown("**What went well**")
            for item in positives:
                st.markdown(f"- {item}")
        if improvements:
            st.markdown("**Focus areas**")
            for item in improvements:
                st.markdown(f"- {item}")
    else:
        st.info("No player-specific summary available from the match insights.")

    shot_ids = extract_player_shot_ids(player_summary or {}) if player_summary else []
    if not shot_ids:
        # Fallback to shots taken by the player
        shot_ids = [shot_id for shot_id, shot in shots_by_id.items() if shot.get("player", {}).get("id") == player_id]
    if not shot_ids:
        st.info("No shot involvement recorded for this player in the available data.")
        return
    involvement_rows: List[Dict[str, Any]] = []
    for shot_id in shot_ids:
        shot = shots_by_id.get(shot_id)
        if not shot:
            continue
        explanation = None
        latest_exp = shot.get("latest_explanation")
        if isinstance(latest_exp, Mapping):
            explanation = latest_exp.get("text")
        if not explanation:
            model_pred = shot.get("model_prediction")
            if isinstance(model_pred, Mapping):
                explanation = model_pred.get("explanation")
        if explanation and len(explanation) > 140:
            explanation = f"{explanation[:140]}…"
        involvement_rows.append(
            {
                "Time": shot.get("clock") or shot.get("minute"),
                "Outcome": "Goal" if shot.get("is_goal") else shot.get("outcome") or "Shot",
                "xG": shot.get("model_prediction", {}).get("xg") or shot.get("xg_statsbomb"),
                "Score": f"{shot.get('score_home')}:{shot.get('score_away')}" if shot.get("score_home") is not None and shot.get("score_away") is not None else "",
                "Explanation": explanation,
            }
        )
    if involvement_rows:
        st.markdown("**Shot involvement**")
        st.dataframe(pd.DataFrame(involvement_rows), hide_index=True, use_container_width=True)
    else:
        st.info("Could not match any shot involvement details for this player.")


def _render_shot_detail_section(
    *, base_url: str, shots_by_id: Mapping[str, Mapping[str, Any]]
) -> None:
    st.subheader("Shot analysis")
    if not shots_by_id:
        st.info("Shot data unavailable for this match.")
        return
    shot_ids = list(shots_by_id.keys())
    selected_shot_id = st.selectbox(
        "Choose a shot",
        options=shot_ids,
        format_func=lambda shot_id: build_shot_option_label(shots_by_id[shot_id]),
        key="shot_selector",
    )
    if not selected_shot_id:
        return
    shot_detail = _fetch_shot_detail(base_url, selected_shot_id)
    if not shot_detail:
        return
    meta = shot_detail.get("meta", {})
    st.markdown(
        f"**{meta.get('team', {}).get('name', 'Team')} – {meta.get('player', {}).get('name', 'Unknown')}**"
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Minute", meta.get("clock") or meta.get("minute"))
    xg_model = None
    model_prediction = shot_detail.get("model_prediction")
    if isinstance(model_prediction, Mapping):
        xg_model = model_prediction.get("xg")
    with col2:
        if isinstance(xg_model, (int, float)):
            st.metric("Model xG", f"{xg_model:.2f}")
        elif isinstance(meta.get("xg_statsbomb"), (int, float)):
            st.metric("StatsBomb xG", f"{meta['xg_statsbomb']:.2f}")
        else:
            st.metric("xG", "n/a")
    with col3:
        if meta.get("is_goal"):
            scoreline = f"{meta.get('score_home')}:{meta.get('score_away')}"
            st.metric("Outcome", f"Goal ({scoreline})")
        else:
            st.metric("Outcome", meta.get("outcome") or "No goal")
    explanation = None
    latest_explanation = shot_detail.get("latest_explanation")
    if isinstance(latest_explanation, Mapping):
        explanation = latest_explanation.get("text")
    if not explanation and isinstance(model_prediction, Mapping):
        explanation = model_prediction.get("explanation")
    if explanation:
        st.markdown("**Explanation**")
        st.write(explanation)
    positions = shot_detail.get("positions")
    if isinstance(positions, Mapping) and positions.get("items"):
        actor_id = meta.get("player", {}).get("id")
        fig = create_shot_positions_figure(positions, actor_id=actor_id)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Freeze-frame positions are not available for this shot.")


def main() -> None:
    st.set_page_config(page_title="xGoal Tutor", layout="wide")
    st.title("xGoal Tutor Match Insights")

    base_url = get_api_base_url()
    with st.expander("Settings", expanded=False):
        st.text(f"API base URL: {base_url}")

    matches = _fetch_matches(base_url)
    if not matches:
        st.stop()

    selected_match = st.selectbox(
        "Select a match",
        options=matches,
        format_func=build_match_option_label,
        key="match_selector",
    )
    if not selected_match:
        st.stop()
    match_id = selected_match.get("id")
    if not match_id:
        st.warning("Selected match is missing an identifier.")
        st.stop()

    summary = _fetch_match_summary(base_url, match_id)
    if summary is None:
        st.stop()
    result = summary.get("result", {})
    st.subheader("Match overview")
    st.markdown(format_scoreline(result))

    shots = _fetch_match_shots(base_url, match_id)
    shot_map: Dict[str, Mapping[str, Any]] = {}
    for shot in shots:
        shot_id = shot.get("id")
        if shot_id:
            shot_map[str(shot_id)] = shot

    goal_events = format_goal_events(shots)
    if goal_events:
        st.markdown("**Score timeline**")
        st.dataframe(pd.DataFrame(goal_events), hide_index=True, use_container_width=True)

    lineups = _fetch_match_lineups(base_url, match_id)
    if lineups:
        home_col, away_col = st.columns(2)
        with home_col:
            _render_lineup_table(lineups.get("home"), "Home lineup")
        with away_col:
            _render_lineup_table(lineups.get("away"), "Away lineup")

    home_insights = summary.get("home_insights")
    away_insights = summary.get("away_insights")
    insight_col1, insight_col2 = st.columns(2)
    with insight_col1:
        _render_team_insights("Home team insights", home_insights)
    with insight_col2:
        _render_team_insights("Away team insights", away_insights)

    _render_player_drilldown(
        lineups=lineups,
        summary=summary,
        shots_by_id=shot_map,
    )

    _render_shot_detail_section(base_url=base_url, shots_by_id=shot_map)


if __name__ == "__main__":
    main()
