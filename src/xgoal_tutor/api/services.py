"""Core prediction utilities used by the xGoal inference API."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
import os
import sqlite3
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from xgoal_tutor.api.models import (
    DEFAULT_FALLBACK_MODELS,
    DEFAULT_PRIMARY_MODEL,
    LogisticRegressionModel,
    ReasonCode,
    ShotFeatures,
    ShotPrediction,
)
from xgoal_tutor.api.database import get_db
from xgoal_tutor.llm.client import OllamaConfig, OllamaLLM
from xgoal_tutor.llm.xgoal_prompt_builder import build_xgoal_prompt
from xgoal_tutor.modeling.constants import GOAL_HALF_WIDTH_SB, GOAL_Y_CENTER_SB, PITCH_LENGTH_SB
from xgoal_tutor.modeling.feature_engineering import build_feature_matrix


def create_llm_client() -> OllamaLLM:
    """Initialise an Ollama client with sensible defaults."""

    if os.environ.get("XGOAL_TUTOR_STUB_LLM", "").lower() in {"1", "true", "yes", "on"}:
        class _StubLLM:
            def generate(self, prompt: str, model: Optional[str] = None, **_: object) -> tuple[str, str]:
                return (" stub explanation ", model or DEFAULT_PRIMARY_MODEL)

        return _StubLLM()  # type: ignore[return-value]

    config = OllamaConfig(
        primary_model=DEFAULT_PRIMARY_MODEL,
        fallback_models=DEFAULT_FALLBACK_MODELS,
    )
    return OllamaLLM(config)


def build_feature_dataframe(shots: Iterable[ShotFeatures]) -> pd.DataFrame:
    """Convert incoming shot payloads into the engineered feature matrix."""

    raw_records = [shot.model_dump() for shot in shots]
    frame = pd.DataFrame(raw_records)
    frame = frame.where(pd.notnull(frame), np.nan)
    return build_feature_matrix(frame)


def generate_shot_predictions(
    shots: Sequence[ShotFeatures], model: LogisticRegressionModel
) -> Tuple[List[ShotPrediction], pd.DataFrame]:
    """Produce model predictions and contribution breakdowns for the provided shots."""

    feature_frame = build_feature_dataframe(shots)
    probabilities, contributions = calculate_probabilities(feature_frame, model)

    predictions: List[ShotPrediction] = []
    for index, shot in enumerate(shots):
        row_contrib = contributions.iloc[index]
        reason_codes = format_reason_codes(feature_frame.iloc[index], row_contrib, model)
        predictions.append(
            ShotPrediction(
                shot_id=shot.shot_id,
                match_id=shot.match_id,
                xg=float(probabilities.iloc[index]),
                reason_codes=reason_codes,
            )
        )

    return predictions, contributions


def calculate_probabilities(
    features: pd.DataFrame, model: LogisticRegressionModel
) -> Tuple[pd.Series, pd.DataFrame]:
    """Apply the logistic regression model to the provided feature matrix."""

    coefficients = model.coefficients
    ordered_columns = list(coefficients.keys())
    aligned = features.reindex(columns=ordered_columns, fill_value=0.0)
    coef_vector = pd.Series(coefficients)
    linear_score = aligned.mul(coef_vector, axis=1).sum(axis=1) + model.intercept
    probabilities = 1.0 / (1.0 + np.exp(-linear_score))
    contributions = aligned.mul(coef_vector, axis=1)
    return probabilities, contributions


def format_reason_codes(
    row_values: pd.Series,
    contributions: pd.Series,
    model: LogisticRegressionModel,
    max_reasons: int = 3,
) -> List[ReasonCode]:
    """Summarise the most influential features for a prediction."""

    reason_codes: List[ReasonCode] = []
    sorted_features = contributions.abs().sort_values(ascending=False)
    for feature in sorted_features.index:
        contribution = contributions[feature]
        if math.isclose(contribution, 0.0, abs_tol=1e-9):
            continue
        value_raw = row_values.get(feature, 0.0)
        value = float(0.0 if pd.isna(value_raw) else value_raw)
        coefficient = float(model.coefficients.get(feature, 0.0))
        reason_codes.append(
            ReasonCode(
                feature=feature,
                value=value,
                coefficient=coefficient,
                contribution=float(contribution),
            )
        )
        if len(reason_codes) >= max_reasons:
            break
    return reason_codes


@dataclass
class FreezeFramePlayer:
    player_id: Optional[str]
    player_name: Optional[str]
    teammate: bool
    keeper: bool
    x: Optional[float]
    y: Optional[float]


@dataclass
class TeammateContext:
    teammate_scoring_potential: List[Dict[str, Optional[float | str]]] = field(default_factory=list)
    team_mate_in_better_position_count: int = 0
    max_teammate_xgoal_diff: Optional[float] = None
    teammate_name_with_max_xgoal: Optional[str] = None

    def as_dict(self) -> Dict[str, Optional[float | int | str]]:
        return {
            "teammate_scoring_potential": self.teammate_scoring_potential,
            "team_mate_in_better_position_count": self.team_mate_in_better_position_count,
            "max_teammate_xgoal_diff": self.max_teammate_xgoal_diff,
            "teammate_name_with_max_xgoal": self.teammate_name_with_max_xgoal,
        }


def _compute_teammate_context(
    shots: Sequence[ShotFeatures], predictions: Sequence[ShotPrediction], model: LogisticRegressionModel
) -> List[TeammateContext]:
    if len(shots) != len(predictions):
        raise ValueError("Shots and predictions length mismatch")

    contexts: List[TeammateContext] = []
    with get_db() as connection:
        connection.row_factory = sqlite3.Row
        for shot, prediction in zip(shots, predictions):
            contexts.append(_compute_context_for_shot(connection, shot, prediction.xg, model))
    return contexts


def _apply_teammate_context(
    shots: Sequence[ShotFeatures], predictions: Sequence[ShotPrediction], model: LogisticRegressionModel
) -> List[ShotPrediction]:
    contexts = _compute_teammate_context(shots, predictions, model)
    enriched: List[ShotPrediction] = []
    for prediction, context in zip(predictions, contexts):
        data = prediction.model_dump()
        data.update(context.as_dict())
        enriched.append(ShotPrediction(**data))
    return enriched


def _compute_context_for_shot(
    connection: sqlite3.Connection, shot: ShotFeatures, shooter_xg: float, model: LogisticRegressionModel
) -> TeammateContext:
    if not shot.shot_id:
        return TeammateContext(teammate_scoring_potential=[])

    players = _load_freeze_frame_players(connection, shot.shot_id)
    if not players:
        return TeammateContext(teammate_scoring_potential=[])

    shooter_id = _lookup_shooter_id(connection, shot.shot_id)
    keeper = next((player for player in players if player.keeper), None)
    opponents = [player for player in players if not player.teammate and not player.keeper]
    teammates = [player for player in players if player.teammate and not player.keeper]
    teammates = _filter_teammates(teammates, shooter_id, shot.start_x, shot.start_y)

    if not teammates:
        return TeammateContext(teammate_scoring_potential=[])

    offside_flags = [is_offside(teammate, shot.start_x, opponents, keeper) for teammate in teammates]
    onside_teammates = [teammate for teammate, offside in zip(teammates, offside_flags) if not offside]
    teammate_shots = _build_teammate_shots(shot, onside_teammates, opponents, keeper)

    onside_probabilities: List[float] = []
    if teammate_shots:
        feature_frame = build_feature_dataframe(teammate_shots)
        probabilities, _ = calculate_probabilities(feature_frame, model)
        onside_probabilities = [float(prob) if prob is not None else 0.0 for prob in probabilities.tolist()]

    probability_values: List[float] = []
    onside_iter = iter(onside_probabilities)
    for offside in offside_flags:
        probability_values.append(0.0 if offside else float(next(onside_iter, 0.0)))

    probability_series = pd.Series(probability_values, dtype=float)

    scoring_potential: List[Dict[str, Optional[float | str]]] = []
    for prob, teammate in sorted(zip(probability_values, teammates), key=lambda entry: entry[0], reverse=True):
        scoring_potential.append({"player_name": teammate.player_name, "xg": float(prob)})

    better_count = int((probability_series > shooter_xg).sum())
    if probability_series.empty:
        return TeammateContext(teammate_scoring_potential=scoring_potential)

    best_prob = float(probability_series.max())
    if math.isnan(best_prob):
        return TeammateContext(
            teammate_scoring_potential=scoring_potential,
            team_mate_in_better_position_count=better_count,
        )

    best_index = int(probability_series.idxmax())
    best_name = teammates[best_index].player_name
    diff = shooter_xg - best_prob

    return TeammateContext(
        teammate_scoring_potential=scoring_potential,
        team_mate_in_better_position_count=better_count,
        max_teammate_xgoal_diff=diff,
        teammate_name_with_max_xgoal=best_name,
    )


def _load_freeze_frame_players(connection: sqlite3.Connection, shot_id: str) -> List[FreezeFramePlayer]:
    rows = connection.execute(
        "SELECT player_id, player_name, teammate, keeper, x, y FROM freeze_frames WHERE shot_id = ?",
        (shot_id,),
    ).fetchall()

    players: List[FreezeFramePlayer] = []
    for row in rows:
        if hasattr(row, "keys"):
            player_id = row["player_id"]
            player_name = row["player_name"]
            teammate = bool(row["teammate"])
            keeper = bool(row["keeper"])
            x = row["x"]
            y = row["y"]
        else:
            player_id, player_name, teammate, keeper, x, y = row
            teammate = bool(teammate)
            keeper = bool(keeper)
        players.append(
            FreezeFramePlayer(
                player_id=str(player_id) if player_id is not None else None,
                player_name=player_name if player_name is None or isinstance(player_name, str) else str(player_name),
                teammate=teammate,
                keeper=keeper,
                x=float(x) if x is not None else None,
                y=float(y) if y is not None else None,
            )
        )
    return players


def _lookup_shooter_id(connection: sqlite3.Connection, shot_id: str) -> Optional[str]:
    try:
        row = connection.execute("SELECT player_id FROM shots WHERE shot_id = ?", (shot_id,)).fetchone()
    except sqlite3.Error:
        return None
    if not row:
        return None
    if hasattr(row, "keys"):
        value = row["player_id"]
    else:
        value = row[0]
    return str(value) if value is not None else None


def _filter_teammates(
    teammates: Sequence[FreezeFramePlayer], shooter_id: Optional[str], shooter_x: float, shooter_y: float
) -> List[FreezeFramePlayer]:
    filtered: List[FreezeFramePlayer] = []
    for player in teammates:
        if shooter_id and player.player_id == shooter_id:
            continue
        if player.x is not None and player.y is not None:
            if math.hypot(player.x - shooter_x, player.y - shooter_y) <= 0.5:
                continue
        filtered.append(player)
    return filtered


def is_offside(
    teammate: FreezeFramePlayer,
    shooter_x: Optional[float],
    opponents: Sequence[FreezeFramePlayer],
    keeper: Optional[FreezeFramePlayer],
) -> bool:
    if teammate.x is None or shooter_x is None:
        return False
    if teammate.x <= shooter_x:
        return False

    opponent_positions: List[float] = [player.x for player in opponents if player.x is not None]
    if keeper and keeper.x is not None:
        opponent_positions.append(keeper.x)

    if len(opponent_positions) < 2:
        return False

    opponent_positions.sort(reverse=True)
    second_last_line = opponent_positions[1]
    return teammate.x > second_last_line


def _build_teammate_shots(
    template: ShotFeatures,
    teammates: Sequence[FreezeFramePlayer],
    opponents: Sequence[FreezeFramePlayer],
    keeper: Optional[FreezeFramePlayer],
) -> List[ShotFeatures]:
    shots: List[ShotFeatures] = []
    for index, teammate in enumerate(teammates):
        if teammate.x is None or teammate.y is None:
            continue
        opponents_in_cone = _count_opponents_in_cone(teammate, opponents)
        shots.append(
            ShotFeatures(
                shot_id=f"{template.shot_id}-tm-{index}",
                match_id=template.match_id,
                start_x=teammate.x,
                start_y=teammate.y,
                is_set_piece=bool(template.is_set_piece),
                is_corner=bool(template.is_corner),
                is_free_kick=bool(template.is_free_kick),
                first_time=False,
                under_pressure=False,
                body_part=None,
                ff_keeper_x=keeper.x if keeper else None,
                ff_keeper_y=keeper.y if keeper else None,
                ff_opponents=opponents_in_cone,
                freeze_frame_available=1,
                ff_keeper_count=1 if keeper else 0,
                one_on_one=False,
                open_goal=False,
                follows_dribble=False,
                deflected=False,
                aerial_won=False,
            )
        )
    return shots


def _count_opponents_in_cone(teammate: FreezeFramePlayer, opponents: Sequence[FreezeFramePlayer]) -> int:
    count = 0
    for opponent in opponents:
        if opponent.x is None or opponent.y is None or teammate.x is None or teammate.y is None:
            continue
        if _is_in_goal_cone(teammate.x, teammate.y, opponent.x, opponent.y):
            count += 1
    return count


def _is_in_goal_cone(origin_x: float, origin_y: float, target_x: float, target_y: float) -> bool:
    if target_x <= origin_x:
        return False
    left_angle = math.atan2((GOAL_Y_CENTER_SB - GOAL_HALF_WIDTH_SB) - origin_y, PITCH_LENGTH_SB - origin_x)
    right_angle = math.atan2((GOAL_Y_CENTER_SB + GOAL_HALF_WIDTH_SB) - origin_y, PITCH_LENGTH_SB - origin_x)
    target_angle = math.atan2(target_y - origin_y, target_x - origin_x)
    low = min(left_angle, right_angle)
    high = max(left_angle, right_angle)
    return low <= target_angle <= high


def generate_llm_explanation(
    client: OllamaLLM,
    shots: Sequence[ShotFeatures],
    predictions: Sequence[ShotPrediction],
    contributions: pd.DataFrame,
    *,
    llm_model: Optional[str] = None,
    prompt_override: Optional[str] = None,
    prompt_template_name: Optional[str] = None,

) -> Tuple[List[str], str]:
    """Create explanations for the predictions via the configured LLM."""

    prompts: List[str]
    if prompt_override is not None:
        prompts = [prompt_override]
    else:
        prompts = _build_xgoal_prompts(
            shots, predictions, contributions, template_name=prompt_template_name or "xgoal_offense_prompt.md"
        )

    explanations: List[str] = []
    model_used: Optional[str] = None
    for prompt in prompts:
        text, used_model = client.generate(prompt, model=llm_model)
        explanations.append(text.strip())
        if not model_used:
            model_used = used_model

    if not model_used:
        model_used = llm_model or DEFAULT_PRIMARY_MODEL

    if len(explanations) == 1 and len(predictions) > 1:
        explanations = explanations * len(predictions)
    elif len(explanations) != len(predictions):
        raise ValueError("Explanation count does not match prediction count")

    return explanations, model_used


def group_predictions_by_match(
    predictions: Iterable[ShotPrediction],
) -> Dict[str, List[ShotPrediction]]:
    """Partition predictions by match identifier for caching."""

    grouped: Dict[str, List[ShotPrediction]] = defaultdict(list)
    for prediction in predictions:
        if prediction.match_id:
            grouped[prediction.match_id].append(prediction)
    return grouped


def _build_xgoal_prompts(
    shots: Sequence[ShotFeatures],
    predictions: Sequence[ShotPrediction],
    contributions: pd.DataFrame,
    *,
    template_name: str = "xgoal_offense_prompt.md",
) -> List[str]:
    """Render the xGoal Markdown prompts for the provided shots."""

    if len(shots) != len(predictions):
        raise ValueError("Shots and predictions length mismatch")

    if len(contributions) < len(shots):
        raise ValueError("Missing contribution rows for the provided shots")

    feature_frame = build_feature_dataframe(shots)

    prompts: List[str] = []
    with get_db() as connection:
        for index, (shot, prediction) in enumerate(zip(shots, predictions)):
            shot_id = shot.shot_id or prediction.shot_id
            if not shot_id:
                raise ValueError("Each shot must include a shot_id to build prompts")

            contribution_row = contributions.iloc[index]
            raw_feature_row = feature_frame.iloc[index] if index < len(feature_frame) else pd.Series(dtype=float)
            feature_block = _format_feature_block(contribution_row, raw_feature_row, limit=10)
            context_block = _format_teammate_context_line(prediction)
            team_mates_scoring_potential_block = _format_teammate_scoring_potential_block(prediction)

            prompts.append(
                build_xgoal_prompt(
                    connection,
                    str(shot_id),
                    feature_block=feature_block,
                    context_block=context_block,
                    team_mates_scoring_potential_block=team_mates_scoring_potential_block,
                    template_name=template_name,
                )
            )

    return prompts
        

def _format_feature_block(
    contribution_row: pd.Series,
    feature_row: pd.Series,
    limit: int = 5,
) -> List[str]:
    """Convert a contribution row into formatted prompt bullet points."""

    sortable = (
        contribution_row.dropna()
        if hasattr(contribution_row, "dropna")
        else pd.Series(contribution_row)
    )

    if not isinstance(sortable, pd.Series):
        sortable = pd.Series(sortable)

    sortable = sortable[[col for col in sortable.index if not str(col).startswith("__")]]
    ordered = sortable.reindex(sortable.abs().sort_values(ascending=False).index)

    raw_values = (
        feature_row
        if isinstance(feature_row, pd.Series)
        else pd.Series(feature_row)
    )

    lines: List[str] = []
    for feature, value in ordered.iloc[:limit].items():
        if math.isclose(float(value), 0.0, abs_tol=1e-9):
            continue
        arrow = "↑" if value > 0 else "↓"
        raw_value = raw_values.get(feature, float("nan"))
        formatted_raw = _format_raw_value_for_prompt(raw_value)
        lines.append(f"{arrow} {feature} ({value:+.3f}) (raw value:{formatted_raw})")

    return lines


def _format_teammate_context_line(prediction: ShotPrediction) -> str:
    count = prediction.team_mate_in_better_position_count
    diff = prediction.max_teammate_xgoal_diff
    best_name = prediction.teammate_name_with_max_xgoal

    parts: List[str] = []
    if count is not None:
        parts.append(f"Teammates with higher xG: {count}")

    best_xg = None
    if diff is not None and not math.isnan(diff):
        best_xg = prediction.xg - diff

    if best_name:
        if best_xg is not None:
            parts.append(f"Best option: {best_name} (xG {best_xg:.3f})")
        else:
            parts.append(f"Best option: {best_name}")
    elif diff is not None:
        parts.append(f"Best teammate xG difference: {diff:.3f}")

    return "\n".join(parts)


def _format_teammate_scoring_potential_block(prediction: ShotPrediction) -> str:
    lines: List[str] = []
    count = prediction.team_mate_in_better_position_count
    diff = prediction.max_teammate_xgoal_diff
    best_name = prediction.teammate_name_with_max_xgoal

    lines.append(f"- team_mate_in_better_position_count: {count if count is not None else 0}")

    if diff is None or math.isnan(diff):
        lines.append("- max_teammate_xgoal_diff: n/a")
    else:
        lines.append(f"- max_teammate_xgoal_diff: {diff:+.3f}")

    if best_name:
        lines.append(f"- teammate_name_with_max_xgoal: {best_name}")
    else:
        lines.append("- teammate_name_with_max_xgoal: unknown")

    for entry in prediction.teammate_scoring_potential or []:
        name = entry.get("player_name") if isinstance(entry, dict) else None
        if not name and hasattr(entry, "player_name"):
            name = getattr(entry, "player_name")
        xg_value = entry.get("xg") if isinstance(entry, dict) else getattr(entry, "xg", None)
        label = name or "unknown teammate"
        if xg_value is None or (isinstance(xg_value, float) and (math.isnan(xg_value) or not math.isfinite(xg_value))):
            lines.append(f"- {label}: xG n/a")
        else:
            lines.append(f"- {label}: xG {float(xg_value):.3f}")

    return "\n".join(lines) if lines else "none"


def _format_raw_value_for_prompt(value: object) -> str:
    """Return a human-friendly string for raw feature values in prompts."""

    if value is None:
        return "n/a"

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)

    if pd.isna(numeric) or not math.isfinite(numeric):
        return "n/a"

    if math.isclose(numeric, round(numeric), abs_tol=1e-6):
        return str(int(round(numeric)))

    text = f"{numeric:.3f}".rstrip("0").rstrip(".")
    return text or "0"
