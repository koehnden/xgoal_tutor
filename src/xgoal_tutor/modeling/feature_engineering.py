"""Feature construction helpers for xGoal models."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from xgoal_tutor.modeling.constants import (
    GOAL_HALF_WIDTH_SB,
    GOAL_Y_CENTER_SB,
    PITCH_LENGTH_SB,
)


def _distance_sb(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    dx = PITCH_LENGTH_SB - x
    dy = GOAL_Y_CENTER_SB - y
    return np.hypot(dx, dy)


def _opening_angle_deg_sb(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    left = np.array([PITCH_LENGTH_SB, GOAL_Y_CENTER_SB - GOAL_HALF_WIDTH_SB])
    right = np.array([PITCH_LENGTH_SB, GOAL_Y_CENTER_SB + GOAL_HALF_WIDTH_SB])
    p = np.column_stack([x, y])
    v1 = left - p
    v2 = right - p
    dot = (v1 * v2).sum(axis=1)
    n1 = np.linalg.norm(v1, axis=1)
    n2 = np.linalg.norm(v2, axis=1)
    cosang = np.clip(dot / (n1 * n2), -1, 1)
    return np.degrees(np.arccos(cosang))


def _to_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(int)
    return pd.to_numeric(series, errors="coerce")


def _dist_point_to_segment(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    dx = bx - ax
    dy = by - ay
    if dx == 0 and dy == 0:
        return np.hypot(px - ax, py - ay)
    t = max(0, min(1, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    closest_x = ax + t * dx
    closest_y = ay + t * dy
    return np.hypot(px - closest_x, py - closest_y)


def _compute_cutback_for_shot(
    shot_x: float,
    shot_y: float,
    freeze_frame_rows: list[tuple[float, float, int, int]],
) -> tuple[int, float, float]:
    PASS_CLEAR_THRESHOLD = 1.0
    GOLD_Y_MIN = GOAL_Y_CENTER_SB - 8.0
    GOLD_Y_MAX = GOAL_Y_CENTER_SB + 8.0
    
    if not freeze_frame_rows:
        return 0, 0.0, 0.0
    
    teammates = [(x, y) for x, y, teammate, keeper in freeze_frame_rows if teammate == 1 and keeper == 0]
    opponents = [(x, y) for x, y, teammate, keeper in freeze_frame_rows if teammate == 0 and keeper == 0]
    
    candidates = [
        (tx, ty) for tx, ty in teammates 
        if tx > shot_x and GOLD_Y_MIN <= ty <= GOLD_Y_MAX
    ]
    
    if not candidates:
        return 0, 0.0, 0.0
    
    best_candidate = None
    best_distance = -1.0
    
    for tx, ty in candidates:
        if not opponents:
            min_opponent_dist = float('inf')
        else:
            min_opponent_dist = min(
                _dist_point_to_segment(ox, oy, shot_x, shot_y, tx, ty)
                for ox, oy in opponents
            )
        
        if min_opponent_dist > PASS_CLEAR_THRESHOLD:
            candidate_to_goal_dist = np.hypot(PITCH_LENGTH_SB - tx, GOAL_Y_CENTER_SB - ty)
            if best_candidate is None or candidate_to_goal_dist < best_distance:
                best_candidate = (tx, ty)
                best_distance = candidate_to_goal_dist
    
    if best_candidate:
        return 1, best_candidate[0], best_candidate[1]
    return 0, 0.0, 0.0


def _compute_cutback_features(shot_ids: pd.Series, shot_x: pd.Series, shot_y: pd.Series, db_path: Path | str) -> pd.DataFrame:
    result = pd.DataFrame(index=shot_ids.index)
    result["has_cutback"] = 0
    result["cutback_target_x"] = 0.0
    result["cutback_target_y"] = 0.0
    
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        for idx in shot_ids.index:
            shot_id = shot_ids.loc[idx]
            sx = shot_x.loc[idx]
            sy = shot_y.loc[idx]
            
            cursor.execute(
                "SELECT x, y, teammate, keeper FROM freeze_frames WHERE shot_id = ?",
                (shot_id,)
            )
            freeze_frame_rows = cursor.fetchall()
            
            has_cutback, target_x, target_y = _compute_cutback_for_shot(sx, sy, freeze_frame_rows)
            result.loc[idx, "has_cutback"] = has_cutback
            result.loc[idx, "cutback_target_x"] = target_x
            result.loc[idx, "cutback_target_y"] = target_y
    
    return result


def build_feature_matrix(data: pd.DataFrame, db_path: Path | str | None = None) -> pd.DataFrame:
    """Construct the feature matrix used by the baseline notebook."""

    X = pd.DataFrame(index=data.index)

    if {"start_x", "start_y"}.issubset(data.columns):
        start_x = data["start_x"].to_numpy()
        start_y = data["start_y"].to_numpy()
        X["dist_sb"] = _distance_sb(start_x, start_y)
        X["angle_deg_sb"] = _opening_angle_deg_sb(start_x, start_y)

    for col in ["is_set_piece", "is_corner", "is_free_kick", "first_time", "under_pressure"]:
        if col in data.columns:
            X[col] = _to_numeric(data[col].astype("boolean"))
        else:
            X[col] = 0

    if "body_part" in data.columns:
        X["is_header"] = (data["body_part"] == "Head").astype(int)
    else:
        X["is_header"] = 0

    if {"ff_keeper_x", "ff_keeper_y"}.issubset(data.columns):
        X["gk_depth_sb"] = np.maximum(0.0, PITCH_LENGTH_SB - data["ff_keeper_x"].to_numpy())
        X["gk_offset_sb"] = data["ff_keeper_y"].to_numpy() - GOAL_Y_CENTER_SB
    else:
        X["gk_depth_sb"] = 0.0
        X["gk_offset_sb"] = 0.0

    if "ff_opponents" in data.columns:
        X["ff_opponents"] = pd.to_numeric(data["ff_opponents"], errors="coerce")
    else:
        X["ff_opponents"] = 0.0

    if {"freeze_frame_available", "ff_keeper_count"}.issubset(data.columns):
        has_gk = (data["freeze_frame_available"] == 1) & (data["ff_keeper_count"].fillna(0) > 0)
        X.loc[has_gk, "gk_depth_sb"] = np.maximum(
            0.0, PITCH_LENGTH_SB - data.loc[has_gk, "ff_keeper_x"]
        )
        X.loc[has_gk, "gk_offset_sb"] = data.loc[has_gk, "ff_keeper_y"] - GOAL_Y_CENTER_SB
        X.loc[~has_gk, ["gk_depth_sb", "gk_offset_sb"]] = 0.0

    for col in [
        "first_time",
        "one_on_one",
        "open_goal",
        "follows_dribble",
        "deflected",
        "aerial_won",
        "under_pressure",
    ]:
        if col in data.columns:
            X[f"{col}_miss"] = data[col].isna().astype(int)
        else:
            X[f"{col}_miss"] = 0

    if db_path and {"shot_id", "start_x", "start_y"}.issubset(data.columns):
        cutback_df = _compute_cutback_features(
            data["shot_id"],
            data["start_x"],
            data["start_y"],
            db_path
        )
        X["has_cutback"] = cutback_df["has_cutback"]
    else:
        X["has_cutback"] = 0

    X = X.apply(pd.to_numeric, errors="coerce")
    return X
