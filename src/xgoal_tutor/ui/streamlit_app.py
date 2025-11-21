"""Streamlit UI for the xGoal tutor demo."""

from __future__ import annotations

import json
import os
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
import requests
import streamlit as st

from xgoal_tutor.ui.api_client import (
    enqueue_match_summary,
    enqueue_player_summary,
    get_api_base_url,
    get_match_lineups,
    get_match_shots,
    get_match_summary_status,
    get_player_summary_status,
    list_matches,
    predict_shots,
)
from xgoal_tutor.ui.formatting import (
    build_match_option_label,
    build_shot_option_label,
    describe_player_highlight,
    extract_insight_texts,
    extract_player_shot_ids,
    format_goal_events,
    format_scoreline,
)
from xgoal_tutor.ui.plots import create_shot_positions_figure

POLL_INTERVAL_SECONDS = 1.0
MAX_POLL_ATTEMPTS = 60
MODEL_ENV_VAR = "XGOAL_TUTOR_LOGREG_MODEL_JSON"
DEFAULT_MODEL: Mapping[str, Any] = {"intercept": -1.4, "coefficients": {}}


def _render_error(message: str, error: Exception) -> None:
    st.error(f"{message}: {error}")


def _initial_model_json() -> str:
    env_value = os.getenv(MODEL_ENV_VAR)
    if env_value:
        try:
            json.loads(env_value)
            return env_value
        except json.JSONDecodeError:
            st.warning(
                "Environment variable %s does not contain valid JSON. Using default model." % MODEL_ENV_VAR
            )
    return json.dumps(DEFAULT_MODEL, indent=2)


def _model_configuration_widget() -> Optional[Mapping[str, Any]]:
    default_json = _initial_model_json()
    model_text = st.text_area(
        "Logistic regression model JSON",
        value=default_json,
        height=220,
        key="logreg_model_json",
    )
    try:
        parsed = json.loads(model_text)
    except json.JSONDecodeError:
        st.error("Logistic regression model configuration is not valid JSON.")
        return None
    if not isinstance(parsed, Mapping):
        st.error("Model configuration must be a JSON object.")
        return None
    if "intercept" not in parsed or "coefficients" not in parsed:
        st.warning("Model JSON should define 'intercept' and 'coefficients'.")
    return parsed


def _poll_until_complete(
    fetch_status: Callable[[], Mapping[str, Any]], *, description: str
) -> Optional[Mapping[str, Any]]:
    """Poll an async endpoint until it returns a terminal status."""

    for _ in range(MAX_POLL_ATTEMPTS):
        status_payload = fetch_status()
        state = status_payload.get("status")
        if state == "done":
            return status_payload
        if state == "failed":
            message = status_payload.get("error_message") or f"{description} failed"
            st.error(message)
            return None
        time.sleep(POLL_INTERVAL_SECONDS)
    st.error(f"{description} did not finish in time. Please try again.")
    return None


def _fetch_matches(base_url: str) -> List[Mapping[str, Any]]:
    try:
        return list_matches(base_url)
    except requests.RequestException as exc:
        _render_error("Unable to load matches", exc)
        return []


def _fetch_match_summary(base_url: str, match_id: str) -> Optional[Mapping[str, Any]]:
    try:
        submission = enqueue_match_summary(base_url, match_id)
    except requests.RequestException as exc:
        _render_error("Could not enqueue match summary", exc)
        return None
    generation_id = submission.get("generation_id")
    if not generation_id:
        st.error("The summary submission did not include a generation identifier.")
        return None

    with st.spinner("Generating match summary..."):
        try:
            status_payload = _poll_until_complete(
                lambda: get_match_summary_status(
                    base_url, match_id, generation_id=str(generation_id)
                ),
                description="Match summary",
            )
        except requests.RequestException as exc:
            _render_error("Polling match summary status failed", exc)
            return None
    if not status_payload:
        return None
    result = status_payload.get("result")
    if not isinstance(result, Mapping):
        st.error("Match summary did not return the expected payload.")
        return None
    return result


def _fetch_player_summary(
    base_url: str, match_id: str, player_id: str
) -> Optional[Mapping[str, Any]]:
    try:
        submission = enqueue_player_summary(base_url, match_id, player_id)
    except requests.RequestException as exc:
        _render_error("Could not enqueue player summary", exc)
        return None
    generation_id = submission.get("generation_id")
    if not generation_id:
        st.error("Player summary submission did not return a generation id.")
        return None
    with st.spinner("Generating player summary..."):
        try:
            status_payload = _poll_until_complete(
                lambda: get_player_summary_status(
                    base_url,
                    match_id,
                    player_id,
                    generation_id=str(generation_id),
                ),
                description="Player summary",
            )
        except requests.RequestException as exc:
            _render_error("Polling player summary status failed", exc)
            return None
    if not status_payload:
        return None
    result = status_payload.get("result")
    if not isinstance(result, Mapping):
        st.error("Player summary result is missing.")
        return None
    return result


def _fetch_match_lineups(base_url: str, match_id: str) -> Optional[Mapping[str, Any]]:
    try:
        return get_match_lineups(base_url, match_id)
    except requests.RequestException as exc:
        _render_error("Failed to retrieve lineups", exc)
        return None


def _normalise_shot_records(
    match_id: str, items: Sequence[Mapping[str, Any]]
) -> List[Dict[str, Any]]:
    shots: List[Dict[str, Any]] = []
    for index, shot in enumerate(items):
        if not isinstance(shot, Mapping):
            continue
        features = shot.get("features")
        if isinstance(features, Mapping):
            feature_payload = dict(features)
        else:
            feature_payload = {}
        shot_id = (
            feature_payload.get("shot_id")
            or shot.get("shot_id")
            or f"{match_id}-shot-{index+1}"
        )
        feature_payload.setdefault("match_id", match_id)
        feature_payload["shot_id"] = str(shot_id)
        normalised = dict(shot)
        normalised["shot_id"] = str(shot_id)
        normalised["features"] = feature_payload
        shots.append(normalised)
    return shots


def _fetch_match_shots(
    base_url: str, match_id: str
) -> Tuple[List[Dict[str, Any]], Mapping[str, Any]]:
    try:
        payload = get_match_shots(base_url, match_id)
    except requests.RequestException as exc:
        _render_error("Unable to load shots for the match", exc)
        return [], {}
    items = payload.get("items") if isinstance(payload, Mapping) else None
    if not isinstance(items, Sequence):
        return [], payload if isinstance(payload, Mapping) else {}
    shots = _normalise_shot_records(match_id, items)
    metadata = {k: v for k, v in payload.items() if k != "items"} if isinstance(payload, Mapping) else {}
    return shots, metadata


def _predict_shots_for_match(
    base_url: str, shots: Sequence[Mapping[str, Any]], model: Mapping[str, Any]
) -> Mapping[str, Mapping[str, Any]]:
    features: List[Mapping[str, Any]] = []
    shot_ids: List[str] = []
    for shot in shots:
        feature_payload = shot.get("features")
        if not isinstance(feature_payload, Mapping):
            continue
        features.append(dict(feature_payload))
        shot_ids.append(str(feature_payload.get("shot_id")))
    if not features:
        return {}
    try:
        response = predict_shots(base_url, {"shots": features, "model": model})
    except requests.RequestException as exc:
        _render_error("Unable to compute shot predictions", exc)
        return {}
    predictions: Dict[str, Mapping[str, Any]] = {}
    for shot_id, prediction in zip(shot_ids, response.get("shots", [])):
        if isinstance(prediction, Mapping) and shot_id:
            predictions[shot_id] = prediction
    return predictions


def _fetch_shot_prediction(
    base_url: str, feature_payload: Mapping[str, Any], model: Mapping[str, Any]
) -> Optional[Mapping[str, Any]]:
    try:
        response = predict_shots(
            base_url,
            {"shots": [dict(feature_payload)], "model": model},
        )
    except requests.RequestException as exc:
        _render_error("Unable to generate prediction for the selected shot", exc)
        return None
    shots = response.get("shots")
    if not isinstance(shots, list) or not shots:
        st.error("Prediction response did not include any shots.")
        return None
    prediction = dict(shots[0]) if isinstance(shots[0], Mapping) else {}
    prediction["explanation"] = response.get("explanation")
    prediction["llm_model"] = response.get("llm_model")
    return prediction


def _lineup_entries_to_dataframe(entries: Any) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if isinstance(entries, list):
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            player = entry.get("player", {}) if isinstance(entry.get("player"), Mapping) else {}
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
        section = lineups.get(side)
        if not isinstance(section, Mapping):
            continue
        team_name = section.get("team", {}).get("name", side.title())
        for group in ("starters", "bench"):
            for entry in section.get(group, []) or []:
                if not isinstance(entry, Mapping):
                    continue
                player = entry.get("player")
                if not isinstance(player, Mapping):
                    continue
                player_id = player.get("id")
                if not player_id:
                    continue
                players[str(player_id)] = {
                    "player": player,
                    "team_name": team_name,
                }
    return list(players.values())


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


def _render_player_drilldown(
    *,
    base_url: str,
    match_id: str,
    lineups: Optional[Mapping[str, Any]],
    shots_by_id: Mapping[str, Mapping[str, Any]],
    predictions_by_id: Mapping[str, Mapping[str, Any]],
    model: Mapping[str, Any],
) -> None:
    st.subheader("Player drilldown")
    player_records = _collect_players_from_lineups(lineups)
    if not player_records:
        st.info("Player information unavailable for this match.")
        return
    selected = st.selectbox(
        "Select a player",
        options=player_records,
        format_func=lambda item: f"{item['player'].get('name')} ({item['team_name']})",
        key="player_drilldown",
    )
    if not selected:
        return
    player = selected["player"]
    player_id = player.get("id")
    st.markdown(f"#### {player.get('name')} – {selected['team_name']}")
    player_summary = (
        _fetch_player_summary(base_url, match_id, player_id)
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
        st.info("No player-specific summary available yet.")

    shot_ids = extract_player_shot_ids(player_summary or {}) if player_summary else []
    if not shot_ids and player_id:
        shot_ids = [
            shot_id
            for shot_id, shot in shots_by_id.items()
            if shot.get("shooter", {}).get("player_id") == player_id
        ]
    if not shot_ids:
        st.info("No shot involvement recorded for this player in the available data.")
        return
    involvement_rows: List[Dict[str, Any]] = []
    for shot_id in shot_ids:
        shot = shots_by_id.get(shot_id)
        if not shot:
            continue
        prediction = predictions_by_id.get(shot_id, {})
        minute = shot.get("minute")
        second = shot.get("second")
        clock = shot.get("clock")
        if isinstance(minute, int) and isinstance(second, int):
            time_display = f"{minute:02d}:{second:02d}"
        elif isinstance(clock, str):
            time_display = clock
        elif isinstance(minute, int):
            time_display = f"{minute}'"
        else:
            time_display = "-"
        score_after = shot.get("scoreline_after") or {}
        score_text = (
            f"{score_after.get('home')}:{score_after.get('away')}"
            if score_after.get("home") is not None and score_after.get("away") is not None
            else ""
        )
        involvement_rows.append(
            {
                "Time": time_display,
                "Outcome": shot.get("result") or shot.get("outcome") or "Shot",
                "xG": prediction.get("xg"),
                "Score": score_text,
            }
        )
    if involvement_rows:
        st.markdown("**Shot involvement**")
        st.dataframe(pd.DataFrame(involvement_rows), hide_index=True, use_container_width=True)
    else:
        st.info("Could not match any shot involvement details for this player.")


def _build_positions_payload(shot: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    features = shot.get("features") if isinstance(shot.get("features"), Mapping) else {}
    freeze_frame = features.get("freeze_frame")
    coordinate_system = features.get("coordinate_system") or shot.get("coordinate_system")
    if isinstance(freeze_frame, Mapping):
        payload = dict(freeze_frame)
        if coordinate_system and "coordinate_system" not in payload:
            payload["coordinate_system"] = coordinate_system
        return payload
    if isinstance(freeze_frame, Sequence):
        payload = {"items": list(freeze_frame)}
        if isinstance(coordinate_system, Mapping):
            payload["coordinate_system"] = coordinate_system
        return payload
    return None


def _render_shot_analysis(
    *,
    base_url: str,
    shots_by_id: Mapping[str, Mapping[str, Any]],
    predictions_by_id: Mapping[str, Mapping[str, Any]],
    model: Mapping[str, Any],
) -> None:
    st.subheader("Shot analysis")
    if not shots_by_id:
        st.info("Shot data unavailable for this match.")
        return
    shot_ids = list(shots_by_id.keys())
    selected_shot_id = st.selectbox(
        "Choose a shot",
        options=shot_ids,
        format_func=lambda shot_id: build_shot_option_label(
            {**shots_by_id[shot_id], "predicted_xg": predictions_by_id.get(shot_id, {}).get("xg")}
        ),
        key="shot_selector",
    )
    if not selected_shot_id:
        return
    shot = shots_by_id[selected_shot_id]
    feature_payload = shot.get("features") if isinstance(shot.get("features"), Mapping) else None
    if not isinstance(feature_payload, Mapping):
        st.error("Shot features are unavailable for prediction.")
        return
    prediction = _fetch_shot_prediction(base_url, feature_payload, model)
    if not prediction:
        return
    shooter = shot.get("shooter", {})
    team_name = shooter.get("team_name") or shooter.get("team", {}).get("name", "Team")
    player_name = shooter.get("player_name") or shooter.get("name", "Unknown")
    st.markdown(f"**{team_name} – {player_name}**")
    minute = shot.get("minute")
    second = shot.get("second")
    clock = shot.get("clock")
    if isinstance(minute, int) and isinstance(second, int):
        minute_value = f"{minute:02d}:{second:02d}"
    elif isinstance(clock, str):
        minute_value = clock
    elif isinstance(minute, int):
        minute_value = f"{minute}'"
    else:
        minute_value = "-"
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Minute", minute_value)
    with col2:
        xg_value = prediction.get("xg")
        if isinstance(xg_value, (int, float)):
            st.metric("Model xG", f"{xg_value:.2f}")
        else:
            st.metric("Model xG", "n/a")
    with col3:
        score_after = shot.get("scoreline_after") or {}
        if shot.get("result") == "Goal" and score_after:
            scoreline = f"{score_after.get('home')}:{score_after.get('away')}"
            st.metric("Outcome", f"Goal ({scoreline})")
        else:
            st.metric("Outcome", shot.get("result") or shot.get("outcome") or "No goal")
    explanation = prediction.get("explanation")
    if explanation:
        st.markdown("**Explanation**")
        st.write(explanation)
    reason_codes = prediction.get("reason_codes")
    if isinstance(reason_codes, list) and reason_codes:
        reason_rows = [
            {
                "Feature": rc.get("feature"),
                "Value": rc.get("value"),
                "Coefficient": rc.get("coefficient"),
                "Contribution": rc.get("contribution"),
            }
            for rc in reason_codes
            if isinstance(rc, Mapping)
        ]
        if reason_rows:
            st.markdown("**Reason codes**")
            st.dataframe(pd.DataFrame(reason_rows), hide_index=True, use_container_width=True)
    positions = _build_positions_payload(shot)
    if positions and positions.get("items"):
        actor_id = shooter.get("player_id")
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
        model = _model_configuration_widget()
    if model is None:
        st.stop()

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

    shots, _ = _fetch_match_shots(base_url, match_id)
    predictions_by_id = _predict_shots_for_match(base_url, shots, model)
    shots_by_id: Dict[str, Mapping[str, Any]] = {}
    for shot in shots:
        shot_id = shot.get("shot_id")
        if not shot_id:
            continue
        shot["predicted_xg"] = predictions_by_id.get(shot_id, {}).get("xg")
        shots_by_id[str(shot_id)] = shot

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
        shots_by_id=shots_by_id,
        predictions_by_id=predictions_by_id,
        model=model,
    )

    _render_shot_analysis(
        base_url=base_url,
        shots_by_id=shots_by_id,
        predictions_by_id=predictions_by_id,
        model=model,
    )


if __name__ == "__main__":
    main()
