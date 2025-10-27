"""Feature construction helpers for xGoal models."""

from __future__ import annotations

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


def build_feature_matrix(data: pd.DataFrame) -> pd.DataFrame:
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

    X = X.apply(pd.to_numeric, errors="coerce")
    return X
