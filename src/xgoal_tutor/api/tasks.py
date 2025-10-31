"""Celery tasks orchestrating match and player summary generation."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from uuid import uuid4

from xgoal_tutor.api._jobs import (
    cache_match_predictions,
    get_cached_match_predictions,
    insert_job,
    store_result,
    update_status,
)
from xgoal_tutor.api._queries import (
    MatchInfo,
    PlayerInfo,
    ShotRow,
    fetch_match,
    fetch_match_players,
    fetch_shots,
)
from xgoal_tutor.api.celery_app import app as celery_app
from xgoal_tutor.api.default_model import load_default_model
from xgoal_tutor.api.models import ShotContext, ShotFeatures, ShotPrediction, ShotPredictionResponse
from xgoal_tutor.api.services import (
    create_llm_client,
    generate_llm_explanation,
    generate_shot_predictions,
)

logger = logging.getLogger(__name__)

_LLM_CLIENT = create_llm_client()


@celery_app.task(name="xgoal.match_summary", bind=True)
def match_summary_task(self, generation_id: str, match_id: str) -> None:
    """Generate a tactical summary for the full match and enqueue player jobs."""

    update_status(generation_id, "running")
    try:
        match = fetch_match(match_id)
        if match is None:
            raise ValueError(f"Match {match_id} not found")

        all_shots, prediction_contexts, prediction_response = _load_shots_and_predictions(match_id)
        players = fetch_match_players(match_id)

        _spawn_player_jobs(match_id, prediction_contexts, players)
        summary = _build_match_summary(match, all_shots, prediction_contexts, players, prediction_response.llm_model)
        store_result(generation_id, summary)
    except Exception as exc:  # pragma: no cover - defensive logging for worker crashes
        logger.exception("match summary task failed for match %s", match_id)
        update_status(generation_id, "failed", error_message=str(exc))
        raise


@celery_app.task(name="xgoal.player_summary", bind=True)
def player_summary_task(self, generation_id: str, match_id: str, player_id: str) -> None:
    """Generate a player-focused summary for the specified match."""

    update_status(generation_id, "running")
    try:
        match = fetch_match(match_id)
        if match is None:
            raise ValueError(f"Match {match_id} not found")

        players = fetch_match_players(match_id)
        all_shots, prediction_contexts, prediction_response = _load_shots_and_predictions(match_id)
        player_summary = _build_player_summary(match, players, prediction_contexts, player_id, prediction_response.llm_model)
        if player_summary is None:
            raise ValueError(f"Player {player_id} not found for match {match_id}")
        store_result(generation_id, player_summary)
    except Exception as exc:  # pragma: no cover - defensive logging for worker crashes
        logger.exception("player summary task failed for match %s player %s", match_id, player_id)
        update_status(generation_id, "failed", error_message=str(exc))
        raise


def queue_match_summary(generation_id: str, match_id: str) -> None:
    """Helper for API layer to dispatch the match summary task."""

    match_summary_task.delay(generation_id, match_id)


def queue_player_summary(generation_id: str, match_id: str, player_id: str) -> None:
    """Helper for API layer to dispatch the player summary task."""

    player_summary_task.delay(generation_id, match_id, player_id)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_shots_and_predictions(match_id: str) -> Tuple[List[ShotRow], List[ShotContext], ShotPredictionResponse]:
    """Return raw shots, contexts with predictions, and the prediction response."""

    all_shots = fetch_shots(match_id)
    selectable_shots = [row for row in all_shots if row.start_x is not None and row.start_y is not None]

    cached = get_cached_match_predictions(match_id)
    response: Optional[ShotPredictionResponse] = None
    if cached is not None:
        try:
            candidate = ShotPredictionResponse.model_validate(cached)
            if len(candidate.shots) == len(selectable_shots):
                response = candidate
        except Exception:  # pragma: no cover - defensive parse
            logger.warning("cached predictions for match %s could not be parsed", match_id)
            response = None

    if response is None:
        response = _compute_predictions(match_id, selectable_shots)
        cache_match_predictions(match_id, response.model_dump())

    contexts = [ShotContext(row=row, prediction=prediction) for row, prediction in zip(selectable_shots, response.shots)]
    return all_shots, contexts, response


def _compute_predictions(match_id: str, shot_rows: Sequence[ShotRow]) -> ShotPredictionResponse:
    """Compute predictions (and explanation) for the provided shots."""

    if not shot_rows:
        return ShotPredictionResponse(shots=[], explanation="", llm_model="offline")

    features = [_build_shot_features(row) for row in shot_rows]
    model = load_default_model()
    predictions, contributions = generate_shot_predictions(features, model)

    try:
        explanation, llm_model = generate_llm_explanation(
            _LLM_CLIENT,
            features,
            predictions,
            contributions,
        )
    except Exception as exc:  # pragma: no cover - best-effort fallback if LLM unavailable
        logger.warning("LLM explanation unavailable for match %s: %s", match_id, exc)
        explanation = "Automated explanation unavailable."
        llm_model = "unavailable"

    return ShotPredictionResponse(shots=predictions, explanation=explanation.strip(), llm_model=llm_model)


def _build_match_summary(
    match: MatchInfo,
    all_shots: Sequence[ShotRow],
    contexts: Sequence[ShotContext],
    players: Dict[str, PlayerInfo],
    llm_model: str,
) -> Dict[str, object]:
    """Assemble the match summary payload."""

    home_team = _team_payload(match.home_team_id, match.home_team_name)
    away_team = _team_payload(match.away_team_id, match.away_team_name)
    score_home, score_away = _compute_final_score(match, all_shots)
    period_breakdown = _period_breakdown(all_shots)

    home_contexts = [ctx for ctx in contexts if ctx.row.team_id == match.home_team_id]
    away_contexts = [ctx for ctx in contexts if ctx.row.team_id == match.away_team_id]

    summary = {
        "result": {
            "home_team": home_team,
            "away_team": away_team,
            "score_home": score_home,
            "score_away": score_away,
            "final": True,
            "period_breakdown": period_breakdown,
        },
        "home_insights": _build_team_insights(match.home_team_id, match.home_team_name, home_contexts, players),
        "away_insights": _build_team_insights(match.away_team_id, match.away_team_name, away_contexts, players),
        "generated_at": _timestamp_now(),
        "llm_model": llm_model,
    }
    return summary


def _build_team_insights(
    team_id: Optional[str],
    team_name: str,
    contexts: Sequence[ShotContext],
    players: Dict[str, PlayerInfo],
) -> Dict[str, object]:
    """Produce team-level positives, improvements, and player highlights."""

    team_payload = _team_payload(team_id, team_name)
    positives: List[Dict[str, object]] = []
    improvements: List[Dict[str, object]] = []
    best_player: Optional[Dict[str, object]] = None
    improve_player: Optional[Dict[str, object]] = None

    if contexts:
        top_shot = max(contexts, key=lambda ctx: ctx.prediction.xg)
        positives.append(
            {
                "explanation": _format_positive(top_shot),
                "evidence_shot_ids": _evidence_ids(top_shot),
            }
        )

        missed = [ctx for ctx in contexts if ctx.row.is_goal != 1]
        if missed:
            biggest_miss = max(missed, key=lambda ctx: ctx.prediction.xg)
            improvements.append(
                {
                    "explanation": _format_improvement(biggest_miss),
                    "evidence_shot_ids": _evidence_ids(biggest_miss),
                }
            )

        stats = _aggregate_player_stats(contexts)
        if stats:
            best_id, best_stats = max(stats.items(), key=lambda item: item[1]["xg"])
            best_player = {
                "player": _player_payload(best_id, contexts, players, team_id),
                "explanation": _player_explanation(best_stats, positive=True),
                "evidence_shot_ids": best_stats["shot_ids"],
            }
            improvers = [(pid, info) for pid, info in stats.items() if info["goals"] == 0 and info["shots"] > 0]
            if improvers:
                improv_id, improv_stats = max(improvers, key=lambda item: item[1]["xg"])
                improve_player = {
                    "player": _player_payload(improv_id, contexts, players, team_id),
                    "explanation": _player_explanation(improv_stats, positive=False),
                    "evidence_shot_ids": improv_stats["shot_ids"],
                }

    return {
        "team": team_payload,
        "positives": positives,
        "improvements": improvements,
        "best_player": best_player,
        "improve_player": improve_player,
    }


def _build_player_summary(
    match: MatchInfo,
    players: Dict[str, PlayerInfo],
    contexts: Sequence[ShotContext],
    player_id: str,
    llm_model: str,
) -> Optional[Dict[str, object]]:
    """Create the player summary payload using available predictions."""

    relevant = [ctx for ctx in contexts if ctx.row.player_id == player_id]
    player_info = players.get(player_id)
    team_id = player_info.team_id if player_info and player_info.team_id else (relevant[0].row.team_id if relevant else None)
    team_name = _resolve_team_name(match, team_id)
    team_payload = _team_payload(team_id, team_name)

    if player_info is None and not relevant:
        return None

    positives: List[Dict[str, object]] = []
    improvements: List[Dict[str, object]] = []

    if relevant:
        goals = [ctx for ctx in relevant if ctx.row.is_goal == 1]
        best_goal = None
        if goals:
            best_goal = max(goals, key=lambda ctx: ctx.prediction.xg)
            positives.append(
                {
                    "explanation": _format_goal(best_goal),
                    "evidence_shot_ids": _evidence_ids(best_goal),
                }
            )
        top_chance = max(relevant, key=lambda ctx: ctx.prediction.xg)
        if best_goal is None or top_chance.row.shot_id not in _evidence_ids(best_goal):
            positives.append(
                {
                    "explanation": _format_positive(top_chance),
                    "evidence_shot_ids": _evidence_ids(top_chance),
                }
            )

        misses = [ctx for ctx in relevant if ctx.row.is_goal != 1]
        if misses:
            biggest_miss = max(misses, key=lambda ctx: ctx.prediction.xg)
            improvements.append(
                {
                    "explanation": _format_improvement(biggest_miss),
                    "evidence_shot_ids": _evidence_ids(biggest_miss),
                }
            )
    else:
        improvements.append(
            {
                "explanation": "No shooting events recorded for this player in the match.",
                "evidence_shot_ids": [],
            }
        )

    summary = {
        "player": _player_payload(player_id, relevant, players, team_id),
        "team": team_payload,
        "positives": positives,
        "improvements": improvements,
        "generated_at": _timestamp_now(),
        "llm_model": llm_model,
    }
    return summary


def _spawn_player_jobs(match_id: str, contexts: Sequence[ShotContext], players: Dict[str, PlayerInfo]) -> None:
    """Enqueue child player summary jobs for the match."""

    known_player_ids = {player_id for player_id in players.keys()}
    known_player_ids.update({ctx.row.player_id for ctx in contexts if ctx.row.player_id})

    for player_id in sorted(pid for pid in known_player_ids if pid):
        child_id = uuid4().hex
        insert_job(child_id, kind="player", match_id=match_id, player_id=player_id)
        queue_player_summary(child_id, match_id, player_id)


def _build_shot_features(row: ShotRow) -> ShotFeatures:
    """Translate a raw shot row into the ShotFeatures model."""

    return ShotFeatures(
        shot_id=row.shot_id,
        match_id=row.match_id,
        start_x=float(row.start_x),
        start_y=float(row.start_y),
        is_set_piece=_to_bool(row.is_set_piece),
        is_corner=_to_bool(row.is_corner),
        is_free_kick=_to_bool(row.is_free_kick),
        first_time=_to_bool(row.first_time),
        under_pressure=_to_bool(row.under_pressure),
        body_part=row.body_part,
        ff_keeper_x=float(row.ff_keeper_x) if row.ff_keeper_x is not None else None,
        ff_keeper_y=float(row.ff_keeper_y) if row.ff_keeper_y is not None else None,
        ff_opponents=float(row.ff_opponents) if row.ff_opponents is not None else None,
        freeze_frame_available=int(row.freeze_frame_available) if row.freeze_frame_available is not None else None,
        ff_keeper_count=int(row.ff_keeper_count) if row.ff_keeper_count is not None else None,
        one_on_one=_to_bool(row.one_on_one),
        open_goal=_to_bool(row.open_goal),
        follows_dribble=_to_bool(row.follows_dribble),
        deflected=_to_bool(row.deflected),
        aerial_won=_to_bool(row.aerial_won),
    )


def _compute_final_score(match: MatchInfo, shots: Sequence[ShotRow]) -> Tuple[int, int]:
    """Derive the final scoreline for the match."""

    last_with_score = None
    for shot in shots:
        if shot.score_home is not None and shot.score_away is not None:
            last_with_score = shot
    if last_with_score is not None:
        return int(last_with_score.score_home), int(last_with_score.score_away)

    home_id = match.home_team_id
    away_id = match.away_team_id
    home_goals = sum(1 for shot in shots if shot.is_goal == 1 and shot.team_id == home_id)
    away_goals = sum(1 for shot in shots if shot.is_goal == 1 and shot.team_id == away_id)
    return int(home_goals), int(away_goals)


def _period_breakdown(shots: Sequence[ShotRow]) -> List[Dict[str, int]]:
    """Build a simple period-by-period score snapshot."""

    breakdown: Dict[int, Tuple[int, int]] = {}
    for shot in shots:
        if shot.period is None or shot.score_home is None or shot.score_away is None:
            continue
        breakdown[int(shot.period)] = (int(shot.score_home), int(shot.score_away))
    return [
        {"period": period, "score_home": scores[0], "score_away": scores[1]}
        for period, scores in sorted(breakdown.items())
    ]


def _aggregate_player_stats(contexts: Sequence[ShotContext]) -> Dict[str, Dict[str, object]]:
    stats: Dict[str, Dict[str, object]] = {}
    for ctx in contexts:
        player_id = ctx.row.player_id
        if not player_id:
            continue
        entry = stats.setdefault(player_id, {"xg": 0.0, "shots": 0, "goals": 0, "shot_ids": []})
        entry["xg"] = float(entry["xg"]) + float(ctx.prediction.xg)
        entry["shots"] = int(entry["shots"]) + 1
        if ctx.row.is_goal == 1:
            entry["goals"] = int(entry["goals"]) + 1
        if ctx.row.shot_id:
            entry["shot_ids"].append(ctx.row.shot_id)
    return stats


def _team_payload(team_id: Optional[str], team_name: str) -> Dict[str, Optional[str]]:
    return {
        "id": team_id,
        "name": team_name,
        "short_name": team_name,
    }


def _player_payload(
    player_id: str,
    contexts: Sequence[ShotContext],
    players: Dict[str, PlayerInfo],
    fallback_team_id: Optional[str],
) -> Dict[str, Optional[object]]:
    info = players.get(player_id)
    name = info.name if info else _infer_player_name(player_id, contexts)
    team_id = info.team_id if info and info.team_id else fallback_team_id
    return {
        "id": player_id,
        "name": name,
        "jersey_number": info.jersey_number if info else None,
        "team_id": team_id,
        "position": info.position if info else None,
    }


def _infer_player_name(player_id: str, contexts: Sequence[ShotContext]) -> str:
    for ctx in contexts:
        if ctx.row.player_id == player_id and ctx.row.player_name:
            return ctx.row.player_name
    return f"Player {player_id}"


def _player_explanation(stats: Dict[str, object], *, positive: bool) -> str:
    shots = stats["shots"]
    xg = stats["xg"]
    if positive:
        return f"Influential with {xg:.2f} expected goals across {shots} shots."
    return f"Created {xg:.2f} xG from {shots} shots without finding the net."


def _format_positive(context: ShotContext) -> str:
    name = context.row.player_name or f"Player {context.row.player_id or 'unknown'}"
    return f"{name} generated a {context.prediction.xg:.2f} xG chance."


def _format_improvement(context: ShotContext) -> str:
    name = context.row.player_name or f"Player {context.row.player_id or 'unknown'}"
    return f"{name} can improve finishing on a {context.prediction.xg:.2f} xG opportunity."


def _format_goal(context: ShotContext) -> str:
    name = context.row.player_name or f"Player {context.row.player_id or 'unknown'}"
    return f"{name} converted a {context.prediction.xg:.2f} xG chance."


def _evidence_ids(context: ShotContext) -> List[str]:
    return [context.row.shot_id] if context.row.shot_id else []


def _to_bool(value: Optional[int]) -> Optional[bool]:
    if value is None:
        return None
    return bool(value)


def _resolve_team_name(match: MatchInfo, team_id: Optional[str]) -> str:
    if team_id == match.home_team_id:
        return match.home_team_name
    if team_id == match.away_team_id:
        return match.away_team_name
    return "Unknown Team"


def _timestamp_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
