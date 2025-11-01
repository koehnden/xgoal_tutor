"""Streamlit app for the xGoal Tutor project."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import pandas as pd
import requests
import streamlit as st

from xgoal_tutor.ui.api_client import (
    generate_match_summary,
    generate_player_summary,
    get_api_base_url,
    get_match_lineups,
    get_match_shots,
    list_matches,
    predict_shots,
)
from xgoal_tutor.ui.config import get_default_logistic_model, get_llm_model_override
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


def _format_clock(minute: Any, second: Any) -> Optional[str]:
    if isinstance(minute, int) and isinstance(second, int):
        return f"{minute:02d}:{second:02d}"
    if isinstance(minute, int):
        return f"{minute}'"
    return None


def _normalise_shot_records(shots: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    normalised: List[Dict[str, Any]] = []
    for index, raw in enumerate(shots):
        if not isinstance(raw, Mapping):
            continue
        features = raw.get("features")
        if not isinstance(features, Mapping):
            continue
        shot_id_raw = features.get("shot_id") or f"shot-{index + 1}"
        shot_id = str(shot_id_raw)
        shooter = raw.get("shooter") or {}
        score_after = raw.get("scoreline_after")
        score_before = raw.get("scoreline_before")
        normalised_shot: Dict[str, Any] = {
            "shot_id": shot_id,
            "id": shot_id,
            "match_id": features.get("match_id"),
            "minute": raw.get("minute"),
            "second": raw.get("second"),
            "period": raw.get("period"),
            "result": raw.get("result"),
            "score_before": score_before,
            "score_after": score_after,
            "shooter": shooter,
            "features": dict(features),
            "clock": _format_clock(raw.get("minute"), raw.get("second")),
            "player": {
                "id": shooter.get("player_id"),
                "name": shooter.get("player_name"),
            },
            "team": {
                "id": shooter.get("team_id"),
                "name": shooter.get("team_name"),
            },
            "player_id": shooter.get("player_id"),
            "team_id": shooter.get("team_id"),
        }
        normalised_shot["is_goal"] = str(raw.get("result")).lower() == "goal"
        if isinstance(score_after, Mapping):
            normalised_shot["score_home"] = score_after.get("home")
            normalised_shot["score_away"] = score_after.get("away")
        elif isinstance(score_before, Mapping):
            normalised_shot["score_home"] = score_before.get("home")
            normalised_shot["score_away"] = score_before.get("away")
        else:
            normalised_shot["score_home"] = None
            normalised_shot["score_away"] = None
        positions = raw.get("positions") or raw.get("freeze_frame")
        if isinstance(positions, Mapping):
            normalised_shot["positions"] = positions
        coordinate_system = raw.get("coordinate_system")
        if coordinate_system:
            normalised_shot["coordinate_system"] = coordinate_system
        normalised.append(normalised_shot)
    return normalised


def _build_shot_map(shots: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    shot_map: Dict[str, Dict[str, Any]] = {}
    for shot in shots:
        shot_id = shot.get("shot_id") or shot.get("id")
        if not shot_id:
            continue
        shot_map[str(shot_id)] = shot  # type: ignore[assignment]
    return shot_map


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
        raw = get_match_shots(base_url, match_id)
    except requests.RequestException as exc:
        _render_error("Unable to load shots for the match", exc)
        return []
    return _normalise_shot_records(raw)


def _attach_predictions_to_shots(
    *,
    base_url: str,
    shots: Sequence[Dict[str, Any]],
    logistic_model: Optional[Mapping[str, Any]],
    llm_model: Optional[str],
) -> Optional[Mapping[str, Any]]:
    if not logistic_model or not shots:
        return None
    payload_shots = [shot.get("features") for shot in shots if isinstance(shot.get("features"), Mapping)]
    if not payload_shots:
        return None
    try:
        response = predict_shots(
            base_url,
            shots=payload_shots,
            model=logistic_model,
            llm_model=llm_model,
        )
    except requests.RequestException as exc:
        _render_error("Unable to compute shot predictions", exc)
        return None
    predictions = response.get("shots", []) if isinstance(response, Mapping) else []
    for shot, prediction in zip(shots, predictions):
        if isinstance(prediction, Mapping):
            shot["prediction"] = prediction
            shot["xg"] = prediction.get("xg")
            shot["reason_codes"] = prediction.get("reason_codes")
    return response if isinstance(response, Mapping) else None


def _get_player_summary(
    *,
    base_url: str,
    match_id: str,
    player_id: str,
    match_summary: Optional[Mapping[str, Any]],
) -> Optional[Mapping[str, Any]]:
    cached = extract_player_summary(match_summary or {}, player_id) if match_summary else None
    if cached:
        return cached

    cache_key = f"player_summary::{match_id}::{player_id}"
    if cache_key in st.session_state:
        return st.session_state.get(cache_key)

    try:
        with st.spinner("Generating player summary..."):
            summary = generate_player_summary(base_url, match_id, player_id)
    except (requests.RequestException, RuntimeError, TimeoutError) as exc:
        _render_error("Failed to generate player summary", exc)
        st.session_state[cache_key] = None
        return None

    st.session_state[cache_key] = summary
    return summary


def _render_team_insights(title: str, insights: Optional[Mapping[str, Any]]) -> None:
    st.subheader(title)
    if not isinstance(insights, Mapping):
        st.info("No insights available yet.")
        return
    team_name = insights.get("team", {}).get("name")
    if team_name:
        st.markdown(f"**Team:** {team_name}")
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
    base_url: str,
    match_id: str,
    lineups: Optional[Mapping[str, Any]],
    match_summary: Optional[Mapping[str, Any]],
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
    player_summary = (
        _get_player_summary(
            base_url=base_url,
            match_id=match_id,
            player_id=player_id,
            match_summary=match_summary,
        )
        if player_id
        else None
    )
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
        shot_ids = [
            shot_id
            for shot_id, shot in shots_by_id.items()
            if shot.get("player_id") == player_id
        ]
    if not shot_ids:
        st.info("No shot involvement recorded for this player in the available data.")
        return
    involvement_rows: List[Dict[str, Any]] = []
    for shot_id in shot_ids:
        shot = shots_by_id.get(shot_id)
        if not shot:
            continue
        involvement_rows.append(
            {
                "Time": shot.get("clock") or shot.get("minute"),
                "Outcome": shot.get("result") or ("Goal" if shot.get("is_goal") else "Shot"),
                "xG": shot.get("xg"),
                "Score": (
                    f"{shot.get('score_home')}:{shot.get('score_away')}"
                    if shot.get("score_home") is not None and shot.get("score_away") is not None
                    else ""
                ),
            }
        )
    if involvement_rows:
        st.markdown("**Shot involvement**")
        st.dataframe(pd.DataFrame(involvement_rows), hide_index=True, use_container_width=True)
    else:
        st.info("Could not match any shot involvement details for this player.")


def _render_shot_detail_section(
    *,
    base_url: str,
    shots_by_id: Mapping[str, Mapping[str, Any]],
    logistic_model: Optional[Mapping[str, Any]],
    llm_model: Optional[str],
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
    shot = shots_by_id.get(selected_shot_id)
    if not shot:
        st.warning("Selected shot details are unavailable.")
        return
    meta_team = shot.get("team", {}).get("name") or shot.get("shooter", {}).get("team_name")
    meta_player = shot.get("player", {}).get("name") or shot.get("shooter", {}).get("player_name")
    st.markdown(f"**{meta_team or 'Team'} – {meta_player or 'Unknown'}**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Minute", shot.get("clock") or shot.get("minute"))
    with col2:
        xg_value = shot.get("xg")
        if isinstance(xg_value, (int, float)):
            st.metric("Model xG", f"{xg_value:.2f}")
        else:
            st.metric("Model xG", "n/a")
    with col3:
        if shot.get("is_goal"):
            scoreline = None
            if shot.get("score_home") is not None and shot.get("score_away") is not None:
                scoreline = f"{shot.get('score_home')}:{shot.get('score_away')}"
            st.metric("Outcome", f"Goal ({scoreline or 'n/a'})")
        else:
            st.metric("Outcome", shot.get("result") or "No goal")

    detail_key = f"shot_detail::{selected_shot_id}"
    detail_payload = st.session_state.get(detail_key)

    if detail_payload is None and logistic_model:
        features = shot.get("features") if isinstance(shot.get("features"), Mapping) else None
        if features:
            try:
                with st.spinner("Scoring and explaining shot..."):
                    response = predict_shots(
                        base_url,
                        shots=[features],
                        model=logistic_model,
                        llm_model=llm_model,
                    )
            except requests.RequestException as exc:
                _render_error("Unable to score shot", exc)
                response = None
            if isinstance(response, Mapping):
                prediction_items = response.get("shots") or []
                prediction = prediction_items[0] if prediction_items else {}
                detail_payload = {
                    "prediction": prediction,
                    "explanation": response.get("explanation"),
                    "llm_model": response.get("llm_model"),
                }
                st.session_state[detail_key] = detail_payload
                if isinstance(prediction, Mapping):
                    shot["xg"] = prediction.get("xg", shot.get("xg"))
                    shot["reason_codes"] = prediction.get("reason_codes")
                    shot["detail_prediction"] = prediction
            else:
                detail_payload = None

    if detail_payload:
        explanation = detail_payload.get("explanation")
        if explanation:
            st.markdown("**Explanation**")
            st.write(explanation)
        prediction = detail_payload.get("prediction")
        reason_codes = prediction.get("reason_codes") if isinstance(prediction, Mapping) else None
        if isinstance(reason_codes, list) and reason_codes:
            reason_rows = []
            for reason in reason_codes:
                if not isinstance(reason, Mapping):
                    continue
                reason_rows.append(
                    {
                        "Feature": reason.get("feature"),
                        "Value": reason.get("value"),
                        "Coefficient": reason.get("coefficient"),
                        "Contribution": reason.get("contribution"),
                    }
                )
            if reason_rows:
                st.markdown("**Top contributing features**")
                st.dataframe(pd.DataFrame(reason_rows), hide_index=True, use_container_width=True)
    elif not logistic_model:
        st.info("Provide a logistic regression model to view detailed shot predictions.")

    positions = shot.get("positions")
    if isinstance(positions, Mapping) and positions.get("items"):
        actor_id = shot.get("player", {}).get("id") or shot.get("player_id")
        fig = create_shot_positions_figure(positions, actor_id=actor_id)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Freeze-frame positions are not available for this shot.")


def main() -> None:
    st.set_page_config(page_title="xGoal Tutor", layout="wide")
    st.title("xGoal Tutor Match Insights")

    base_url = get_api_base_url()
    try:
        logistic_model = get_default_logistic_model()
        logistic_model_error: Optional[Exception] = None
    except Exception as exc:  # pragma: no cover - defensive configuration path
        logistic_model = None
        logistic_model_error = exc
    llm_model = get_llm_model_override()

    with st.expander("Settings", expanded=False):
        st.text(f"API base URL: {base_url}")
        if logistic_model_error:
            st.error(f"Failed to load logistic model: {logistic_model_error}")
        elif logistic_model:
            coefficient_count = len(logistic_model.get("coefficients", {}))
            st.caption(
                f"Logistic regression model configured ({coefficient_count} coefficients)."
            )
        else:
            st.warning(
                "No logistic regression model configured. Shot predictions will be unavailable until "
                "you provide one via st.secrets or environment variables."
            )
        if llm_model:
            st.caption(f"LLM override: {llm_model}")

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
    _attach_predictions_to_shots(
        base_url=base_url,
        shots=shots,
        logistic_model=logistic_model,
        llm_model=None,
    )
    shot_map = _build_shot_map(shots)

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
        base_url=base_url,
        match_id=match_id,
        lineups=lineups,
        match_summary=summary,
        shots_by_id=shot_map,
    )

    _render_shot_detail_section(
        base_url=base_url,
        shots_by_id=shot_map,
        logistic_model=logistic_model,
        llm_model=llm_model,
    )


if __name__ == "__main__":
    main()
