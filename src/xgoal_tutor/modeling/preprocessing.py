"""Data preparation utilities for xGoal modeling."""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import pandas as pd


SHOT_CLIP_RANGES: Dict[str, Tuple[float, float]] = {
    "statsbomb_xg": (0.0, 1.0),
    "start_x": (0.0, 120.0),
    "start_y": (0.0, 80.0),
    "end_x": (0.0, 120.0),
    "end_y": (0.0, 80.0),
}


def clip_shot_angles(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with shot angle columns clipped to valid ranges."""

    clipped = df.copy()

    for column, (lower, upper) in SHOT_CLIP_RANGES.items():
        if column in clipped.columns:
            clipped[column] = clipped[column].clip(lower=lower, upper=upper)

    return clipped


def build_shot_filter_mask(
    df: pd.DataFrame,
    *,
    drop_penalties: bool,
    drop_own_goals: bool,
) -> pd.Series:
    """Return a boolean mask for rows to retain in shot preprocessing."""

    mask = pd.Series(True, index=df.index)

    if drop_penalties and "is_penalty" in df.columns:
        mask &= ~df["is_penalty"].astype(bool)

    if drop_own_goals and "is_own_goal" in df.columns:
        mask &= ~df["is_own_goal"].astype(bool)

    return mask


def prepare_shot_dataframe(
    df: pd.DataFrame,
    *,
    drop_penalties: bool = True,
    drop_own_goals: bool = True,
    outcome_column: str = "outcome",
    goal_value: str = "Goal",
    drop_columns: Iterable[str] = ("end_x", "end_y", "end_z"),
) -> Tuple[pd.DataFrame, pd.Series]:
    """Filter the raw shots dataframe and build the binary target column."""

    df0 = clip_shot_angles(df)

    mask = build_shot_filter_mask(
        df0, drop_penalties=drop_penalties, drop_own_goals=drop_own_goals
    )

    filtered = df0.loc[mask].copy()
    y = (filtered[outcome_column] == goal_value).astype(int)

    if drop_columns:
        filtered = filtered.drop(columns=list(drop_columns), errors="ignore")

    return filtered, y
